#pragma once

#include <nvcomp/gzip.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief What a gzip file's header and trailer say, read without inflating it.
struct gzip_file_probe {
    uint64_t compressed = 0;
    uint32_t isize = 0;   // trailer ISIZE: the last member's length modulo 2^32
    uint32_t crc = 0;     // trailer CRC32 of the last member
    bool device = false;  // a single-member candidate the device may inflate
};

/// @brief Reads the header and trailer of @p path.
///
/// A candidate is a gzip file without FEXTRA (so never BGZF) whose name does not mark it FASTQ,
/// and whose trailer names a non-empty member. Additional member signatures are checked
/// after reading the compressed bytes, before accepting the device output.
[[nodiscard]] inline gzip_file_probe probe_gzip_file(std::filesystem::path const& path) {
    gzip_file_probe probe;
    int const fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) return probe;
    struct stat status{};
    unsigned char header[4] = {};
    unsigned char trailer[8] = {};
    if (::fstat(fd, &status) == 0 && status.st_size >= 18 &&
        ::pread(fd, header, sizeof(header), 0) == sizeof(header) &&
        ::pread(fd, trailer, sizeof(trailer), status.st_size - 8) == sizeof(trailer)) {
        probe.compressed = static_cast<uint64_t>(status.st_size);
        std::memcpy(&probe.crc, trailer, 4);
        std::memcpy(&probe.isize, trailer + 4, 4);
        auto name = path.filename().string();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        auto const fastq =
            name.find(".fq") != std::string::npos || name.find(".fastq") != std::string::npos;
        probe.device = header[0] == 0x1f && header[1] == 0x8b && header[2] == 8 &&
                       (header[3] & 0xe4U) == 0 && probe.isize != 0 && !fastq;
    }
    ::close(fd);
    return probe;
}

/// @brief Runs @p work(index) for every index below @p count on @p workers threads.
template <typename Work>
void parallel_for(size_t count, size_t workers, Work&& work) {
    std::atomic<size_t> next{0};
    auto run = [&] {
        for (size_t index = next++; index < count; index = next++) work(index);
    };
    std::vector<std::jthread> threads;
    auto const helpers = std::min(workers, count) > 0 ? std::min(workers, count) - 1 : 0;
    threads.reserve(helpers);
    for (size_t thread = 0; thread < helpers; ++thread) threads.emplace_back(run);
    run();
}

namespace device_gzip {

inline constexpr uint32_t crc_polynomial = 0xEDB88320U;
inline constexpr uint32_t block_size = 256;

__device__ __forceinline__ bool is_whitespace(char c) {
    return c == '\n' || c == '\r' || c == ' ' || c == '\t';
}

/// @brief The bytes a FASTA record keeps: everything but sequence whitespace.
struct kept_byte {
    __device__ bool operator()(char c) const {
        return !is_whitespace(c);
    }
};

/// @brief a * b modulo the CRC-32 polynomial, bit-reflected (zlib's multmodp).
__device__ inline uint32_t crc_multiply(uint32_t a, uint32_t b) {
    uint32_t m = 1U << 31U, p = 0;
    while (true) {
        if ((a & m) != 0) {
            p ^= b;
            if ((a & (m - 1)) == 0) break;
        }
        m >>= 1U;
        b = (b & 1U) != 0 ? (b >> 1U) ^ crc_polynomial : b >> 1U;
    }
    return p;
}

/// @brief crc32(A || B) from crc32(A), crc32(B) and |B| (zlib's crc32_combine).
__device__ inline uint32_t
crc_combine(uint32_t first, uint32_t second, uint64_t second_bytes, uint32_t const* x2n) {
    uint32_t shift = 1U << 31U;
    for (uint32_t k = 3; second_bytes != 0; second_bytes >>= 1U, ++k) {
        if ((second_bytes & 1U) != 0) shift = crc_multiply(x2n[k & 31U], shift);
    }
    return crc_multiply(shift, first) ^ second;
}

/// @brief Writes each slot's guard line ending and clears its first header.
static __global__ void slot_init_kernel(
    char* buffer,
    uint64_t const* slots,
    uint32_t files,
    unsigned long long* first_header,
    unsigned long long* header_count
) {
    auto const file = blockIdx.x * blockDim.x + threadIdx.x;
    if (file == 0) *header_count = 0;
    if (file >= files) return;
    buffer[slots[file]] = '\n';
    first_header[file] = slots[file + 1];
}

/// @brief CRC32 of each slot's inflated bytes, one block per file.
///
/// Each thread checksums a contiguous chunk with slice-by-4 tables, and the chunks fold together
/// in order with crc_combine.
static __global__ void slot_crc_kernel(char const* buffer, uint64_t const* slots, uint32_t* crcs) {
    __shared__ uint32_t table[4][256];
    __shared__ uint32_t x2n[32];
    __shared__ uint32_t crc[block_size];
    __shared__ uint64_t bytes[block_size];
    auto const t = threadIdx.x;
    uint32_t entry = t;
    _Pragma("unroll")
    for (int bit = 0; bit < 8; ++bit) {
        entry = (entry & 1U) != 0 ? (entry >> 1U) ^ crc_polynomial : entry >> 1U;
    }
    table[0][t] = entry;
    __syncthreads();
    for (int slice = 1; slice < 4; ++slice) {
        auto const previous = table[slice - 1][t];
        table[slice][t] = (previous >> 8U) ^ table[0][previous & 0xFFU];
        __syncthreads();
    }
    if (t == 0) {
        uint32_t power = 1U << 30U;
        x2n[0] = power;
        for (int n = 1; n < 32; ++n) x2n[n] = power = crc_multiply(power, power);
    }
    auto const* const data = reinterpret_cast<unsigned char const*>(buffer + slots[blockIdx.x] + 1);
    uint64_t const size = slots[blockIdx.x + 1] - slots[blockIdx.x] - 1;
    uint64_t const per = ((size + block_size - 1) / block_size + 15) & ~uint64_t{15};
    uint64_t const begin = cuda::std::min<uint64_t>(size, t * per);
    uint64_t const end = cuda::std::min<uint64_t>(size, begin + per);
    uint32_t c = 0xFFFFFFFFU;
    auto byte = [&](unsigned char value) {
        c = table[0][(c ^ value) & 0xFFU] ^ (c >> 8U);
    };
    uint64_t at = begin;
    for (; at < end && (reinterpret_cast<uintptr_t>(data + at) & 15U) != 0; ++at) byte(data[at]);
    for (; at + 16 <= end; at += 16) {
        auto const words = *reinterpret_cast<uint4 const*>(data + at);
        uint32_t const lanes[4] = {words.x, words.y, words.z, words.w};
        _Pragma("unroll")
        for (uint32_t word : lanes) {
            c ^= word;
            c = table[3][c & 0xFFU] ^ table[2][(c >> 8U) & 0xFFU] ^ table[1][(c >> 16U) & 0xFFU] ^
                table[0][c >> 24U];
        }
    }
    for (; at < end; ++at) byte(data[at]);
    crc[t] = ~c;
    bytes[t] = end - begin;
    __syncthreads();
    for (uint32_t stride = 1; stride < block_size; stride *= 2) {
        if (t % (2 * stride) == 0) {
            crc[t] = crc_combine(crc[t], crc[t + stride], bytes[t + stride], x2n);
            bytes[t] += bytes[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) crcs[blockIdx.x] = crc[0];
}

/// @brief Collects every '>' that starts a line: one past a line ending, which every slot's
/// guard byte provides for its first line.
static __global__ void header_kernel(
    char const* buffer,
    uint64_t size,
    uint64_t* headers,
    uint32_t capacity,
    unsigned long long* count
) {
    constexpr uint64_t ones = 0x0101010101010101ULL;
    auto const has_marker = [](uint64_t word) {
        auto const x = word ^ (ones * '>');
        return ((x - ones) & ~x & 0x8080808080808080ULL) != 0;
    };
    uint64_t const words = (size + 15) / 16;
    for (uint64_t w = blockIdx.x * uint64_t{blockDim.x} + threadIdx.x; w < words;
         w += uint64_t{gridDim.x} * blockDim.x) {
        auto const value = *reinterpret_cast<ulonglong2 const*>(buffer + w * 16);
        if (!has_marker(value.x) && !has_marker(value.y)) continue;
        for (uint32_t j = 0; j < 16; ++j) {
            auto const position = w * 16 + j;
            if (position == 0 || position >= size || buffer[position] != '>') continue;
            auto const previous = buffer[position - 1];
            if (previous != '\n' && previous != '\r') continue;
            auto const slot = atomicAdd(count, 1ULL);
            if (slot < capacity) headers[slot] = position;
        }
    }
}

/// @brief Blanks each header's text to line endings, keeping its '>' between records, and
/// records each file's first header. One warp per header.
static __global__ void blank_header_kernel(
    char* buffer,
    uint64_t const* headers,
    unsigned long long const* count,
    uint32_t capacity,
    uint64_t const* slots,
    uint32_t files,
    unsigned long long* first_header
) {
    auto const lane = threadIdx.x % 32U;
    auto const total = cuda::std::min<unsigned long long>(*count, capacity);
    auto const warps = uint64_t{gridDim.x} * blockDim.x / 32U;
    for (uint64_t h = (blockIdx.x * uint64_t{blockDim.x} + threadIdx.x) / 32U; h < total;
         h += warps) {
        auto const position = headers[h];
        uint32_t low = 0, high = files;  // last slot starting at or before the header
        while (high - low > 1) {
            auto const middle = (low + high) / 2;
            if (slots[middle] <= position) {
                low = middle;
            } else {
                high = middle;
            }
        }
        if (lane == 0) atomicMin(first_header + low, static_cast<unsigned long long>(position));
        auto const end = slots[low + 1];
        for (uint64_t at = position + 1; at < end; at += 32) {
            auto const index = at + lane;
            auto const c = index < end ? buffer[index] : '\n';
            auto const stops = __ballot_sync(0xFFFFFFFFU, c == '\n' || c == '\r');
            auto const stop = stops != 0 ? static_cast<uint32_t>(__ffs(stops) - 1) : 32U;
            if (lane < stop) buffer[index] = '\n';
            if (stops != 0) break;
        }
    }
}

/// @brief Per file: the first byte that is not a line ending, the bytes before the first header
/// blanked, and the count of bytes compaction keeps. One block per file.
static __global__ void slot_finish_kernel(
    char* buffer,
    uint64_t const* slots,
    unsigned long long const* first_header,
    char* first_char,
    unsigned long long* kept
) {
    using reduce = cub::BlockReduce<unsigned long long, block_size>;
    __shared__ typename reduce::TempStorage storage;
    auto const begin = slots[blockIdx.x] + 1;
    auto const end = slots[blockIdx.x + 1];
    if (threadIdx.x == 0) {
        auto at = begin;
        while (at < end && (buffer[at] == '\n' || buffer[at] == '\r')) ++at;
        first_char[blockIdx.x] = at < end ? buffer[at] : '\0';
    }
    __syncthreads();
    auto const head = cuda::std::min<uint64_t>(first_header[blockIdx.x], end);
    for (auto at = begin + threadIdx.x; at < head; at += block_size) buffer[at] = '\n';
    __syncthreads();
    unsigned long long count = 0;
    for (auto at = begin + threadIdx.x; at < end; at += block_size) {
        count += is_whitespace(buffer[at]) ? 0 : 1;
    }
    auto const total = reduce(storage).Sum(count);
    if (threadIdx.x == 0) kept[blockIdx.x] = total;
}

}  // namespace device_gzip

/// @brief One genome the device inflated: its compacted bases, resident on the device.
struct inflated_genome {
    size_t id;
    char const* bases;
    size_t size;
};

/// @brief Inflates single-member gzip FASTA files on the device and compacts their sequence.
///
/// Each lane takes a batch of files: the host reads the compressed bytes, nvCOMP inflates
/// them into device slots, and a few kernels blank header text and leading bytes before an
/// in-place select drops whitespace. A genome comes out as one run of bases with its records
/// joined by the '>' that opened each one, a byte no k-mer window accepts, so it sketches exactly
/// as its separate records would.
///
/// A file is handed back to the host loader when anything is off: an inflate failure, an output
/// whose size or CRC32 disagrees with the trailer (a multi-member stream inflates only its first
/// member here), FASTQ content, or no header. The host loader then gives it its usual result or
/// error.
class device_gzip_inflater {
   public:
    static constexpr size_t lane_count = 6;

    /// @param slot_capacity Inflated bytes one lane holds, one guard byte per file included.
    /// @param compressed_capacity Compressed bytes one lane reads per batch.
    device_gzip_inflater(
        cuda::stream_ref stream,
        size_t slot_capacity,
        size_t compressed_capacity,
        size_t workers
    )
        : workers_(std::max<size_t>(1, workers)),
          compressed_capacity_(compressed_capacity),
          slot_capacity_(slot_capacity),
          header_capacity_(
              static_cast<uint32_t>(std::clamp<size_t>(
                  slot_capacity / 4096,
                  1 << 16,
                  std::numeric_limits<uint32_t>::max()
              ))
          ) {
        auto const options = nvcompBatchedGzipDecompressDefaultOpts;
        nvcomp_temp_bytes_ = 0;
        if (nvcompBatchedGzipDecompressGetTempSizeAsync(
                max_files,
                std::min<size_t>(slot_capacity_, std::numeric_limits<uint32_t>::max()),
                options,
                &nvcomp_temp_bytes_,
                slot_capacity_
            ) != nvcompSuccess) {
            throw std::bad_alloc();
        }
        select_temp_bytes_ = 0;
        cub::DeviceSelect::If(
            nullptr,
            select_temp_bytes_,
            static_cast<char*>(nullptr),
            static_cast<int64_t*>(nullptr),
            static_cast<int64_t>(slot_capacity_),
            device_gzip::kept_byte{},
            stream.get()
        );
        auto const device = stream.device();
        for (auto& lane : lanes_) {
            lane.stream.emplace(device);
            auto const s = cuda::stream_ref{*lane.stream};
            lane.uploaded.emplace(s);
            lane.classified.emplace(s);
            lane.released.emplace(stream);
            lane.pinned.emplace(
                s, cuda::pinned_default_memory_pool(), compressed_capacity_, cuda::no_init
            );
            lane.host.emplace(
                s, cuda::pinned_default_memory_pool(), host_table_bytes(), cuda::no_init
            );
            lane.compressed.emplace(
                cuda::make_device_buffer<char>(s, device, compressed_capacity_, cuda::no_init)
            );
            // Sixteen bytes of tail padding let the header scan read whole words.
            lane.slots.emplace(
                cuda::make_device_buffer<char>(s, device, slot_capacity_ + 16, cuda::no_init)
            );
            lane.tables.emplace(
                cuda::make_device_buffer<char>(s, device, host_table_bytes(), cuda::no_init)
            );
            lane.headers.emplace(
                cuda::make_device_buffer<uint64_t>(s, device, header_capacity_, cuda::no_init)
            );
            lane.temp.emplace(
                cuda::make_device_buffer<char>(
                    s, device, std::max(nvcomp_temp_bytes_, select_temp_bytes_) + 1, cuda::no_init
                )
            );
        }
    }

    [[nodiscard]] size_t slot_capacity() const noexcept {
        return slot_capacity_;
    }

    [[nodiscard]] size_t compressed_capacity() const noexcept {
        return compressed_capacity_;
    }

    /// @brief Device bytes a lane needs besides its slots and compressed bytes.
    [[nodiscard]] static size_t lane_overhead(size_t slot_capacity) noexcept {
        return slot_capacity / 4096 * sizeof(uint64_t) + (size_t{64} << 20);
    }

    /// @brief Starts inflating files from the front of @p ids on @p lane.
    ///
    /// Every id must be a device candidate that fits one lane on its own.
    /// Files with extra member signatures reach @ref finish without device inflation.
    /// @return How many ids the batch took.
    [[nodiscard]] Result<size_t> submit(
        size_t lane_index,
        std::span<size_t const> ids,
        std::span<std::filesystem::path const> paths,
        std::span<gzip_file_probe const> probes
    ) {
        auto& lane = lanes_[lane_index];
        if (ids.empty()) return Err(Error::invalid_argument("an inflate batch needs a file"));
        size_t files = 0, compressed = 0, slots = 0;
        while (files < ids.size() && files < max_files) {
            auto const& probe = probes[ids[files]];
            if (compressed + probe.compressed > compressed_capacity_ ||
                slots + probe.isize + 1 > slot_capacity_) {
                break;
            }
            compressed += probe.compressed;
            slots += probe.isize + 1;
            ++files;
        }
        // The pinned bytes are rewritten below, so the previous batch's upload must be done.
        CUDDL_CUDA_TRY(lane.uploaded->sync());
        auto table = host_table(lane);
        uint64_t compressed_at = 0, slot_at = 0;
        for (size_t file = 0; file < files; ++file) {
            auto const& probe = probes[ids[file]];
            table.compressed_offsets[file] = compressed_at;
            table.compressed_bytes[file] = probe.compressed;
            compressed_at += probe.compressed;
        }
        std::atomic<bool> failed{false};
        parallel_for(files, workers_, [&](size_t file) {
            auto const* const path = paths[ids[file]].c_str();
            auto* const out = lane.pinned->data() + table.compressed_offsets[file];
            auto const bytes = table.compressed_bytes[file];
            int const fd = ::open(path, O_RDONLY);
            size_t done = 0;
            if (fd != -1) {
                while (done < bytes) {
                    auto const got =
                        ::pread(fd, out + done, bytes - done, static_cast<off_t>(done));
                    if (got <= 0) break;
                    done += static_cast<size_t>(got);
                }
                ::close(fd);
            }
            if (done != bytes) failed = true;
            // A matching trailer cannot distinguish repeated first and last members.
            // A signature inside compressed data is ambiguous too, so use the host parser.
            lane.possible_members[file] =
                done > 10 && ::memmem(out + 10, done - 10, "\x1f\x8b\x08", 3) != nullptr;
        });
        if (failed) return Err(Error::invalid_argument("cannot read gzip reference inputs"));
        lane.files.clear();
        lane.fallback.clear();
        for (size_t file = 0; file < files; ++file) {
            if (lane.possible_members[file]) {
                lane.fallback.push_back(ids[file]);
                continue;
            }
            auto const target = lane.files.size();
            auto const& probe = probes[ids[file]];
            table.compressed_ptrs[target] =
                lane.compressed->data() + table.compressed_offsets[file];
            table.compressed_bytes[target] = probe.compressed;
            table.slots[target] = slot_at;
            table.output_ptrs[target] = lane.slots->data() + slot_at + 1;
            table.output_bytes[target] = probe.isize;
            slot_at += probe.isize + 1;
            lane.files.push_back(ids[file]);
        }
        table.slots[lane.files.size()] = slot_at;
        lane.slot_bytes = slot_at;
        if (lane.files.empty()) return files;

        auto const s = cuda::stream_ref{*lane.stream};
        auto* const stream = s.get();
        auto const device_table = device_tables(lane);
        // The slots are rewritten below, so the sketch kernel reading them must be done.
        CUDDL_CUDA_TRY(s.wait(*lane.released));
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                s,
                cuda::std::span{lane.pinned->data(), compressed_at},
                device_span<char>{lane.compressed->data(), compressed_at}
            )
        );
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                s,
                cuda::std::span{lane.host->data(), inputs_bytes()},
                device_span<char>{lane.tables->data(), inputs_bytes()}
            )
        );
        CUDDL_CUDA_TRY(lane.uploaded->record(s));
        auto const count = static_cast<uint32_t>(lane.files.size());
        device_gzip::slot_init_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
            lane.slots->data(),
            device_table.slots,
            count,
            device_table.first_header,
            device_table.header_count
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const inflated = nvcompBatchedGzipDecompressAsync(
            reinterpret_cast<void const* const*>(device_table.compressed_ptrs),
            device_table.compressed_bytes,
            device_table.output_bytes,
            device_table.inflated_bytes,
            count,
            lane.temp->data(),
            nvcomp_temp_bytes_,
            reinterpret_cast<void* const*>(device_table.output_ptrs),
            nvcompBatchedGzipDecompressDefaultOpts,
            device_table.statuses,
            stream
        );
        if (inflated != nvcompSuccess) {
            return Err(Error::resource("nvCOMP gzip launch failed: " + std::to_string(inflated)));
        }
        device_gzip::slot_crc_kernel<<<count, device_gzip::block_size, 0, stream>>>(
            lane.slots->data(), device_table.slots, device_table.crcs
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const grid = static_cast<uint32_t>(
            s.device().attribute(cuda::device_attributes::multiprocessor_count) * 8
        );
        device_gzip::header_kernel<<<grid, 256, 0, stream>>>(
            lane.slots->data(),
            slot_at,
            lane.headers->data(),
            header_capacity_,
            device_table.header_count
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        device_gzip::blank_header_kernel<<<grid, 256, 0, stream>>>(
            lane.slots->data(),
            lane.headers->data(),
            device_table.header_count,
            header_capacity_,
            device_table.slots,
            count,
            device_table.first_header
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        device_gzip::slot_finish_kernel<<<count, device_gzip::block_size, 0, stream>>>(
            lane.slots->data(),
            device_table.slots,
            device_table.first_header,
            device_table.first_char,
            device_table.kept
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto temp_bytes = select_temp_bytes_;
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                lane.temp->data(),
                temp_bytes,
                lane.slots->data(),
                device_table.selected,
                static_cast<int64_t>(slot_at),
                device_gzip::kept_byte{},
                stream
            )
        );
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                s,
                cuda::std::span{lane.tables->data() + inputs_bytes(), outputs_bytes()},
                device_span<char>{lane.host->data() + inputs_bytes(), outputs_bytes()}
            )
        );
        CUDDL_CUDA_TRY(lane.classified->record(s));
        return files;
    }

    /// @brief Waits for @p lane's batch and splits it into genomes the device holds and ids
    /// the host loader must take.
    ///
    /// The genomes stay valid until @ref release, which must follow once the kernels that read
    /// them are enqueued.
    [[nodiscard]] Result<void> finish(
        size_t lane_index,
        std::span<gzip_file_probe const> probes,
        std::vector<inflated_genome>& genomes,
        std::vector<size_t>& fallback
    ) {
        auto& lane = lanes_[lane_index];
        fallback.insert(fallback.end(), lane.fallback.begin(), lane.fallback.end());
        if (lane.files.empty()) return Ok();
        CUDDL_CUDA_TRY(lane.classified->sync());
        auto const table = host_table(lane);
        auto const overflow = *table.header_count > header_capacity_;
        uint64_t offset = 0;
        for (size_t file = 0; file < lane.files.size(); ++file) {
            auto const id = lane.files[file];
            auto const& probe = probes[id];
            auto const size = table.kept[file];
            auto const intact = table.statuses[file] == nvcompSuccess &&
                                table.inflated_bytes[file] == probe.isize &&
                                table.crcs[file] == probe.crc;
            auto const fasta =
                table.first_char[file] != '@' && table.first_header[file] < table.slots[file + 1];
            if (!overflow && intact && fasta) {
                genomes.push_back({id, lane.slots->data() + offset, size});
            } else {
                fallback.push_back(id);
            }
            offset += size;
        }
        if (static_cast<int64_t>(offset) != *table.selected) {
            return Err(Error::resource("device FASTA compaction lost track of its bytes"));
        }
        return Ok();
    }

    /// @brief Lets @p lane reuse its slots once @p consumer reaches this point.
    [[nodiscard]] Result<void> release(size_t lane_index, cuda::stream_ref consumer) {
        CUDDL_CUDA_TRY(lanes_[lane_index].released->record(consumer));
        return Ok();
    }

   private:
    static constexpr size_t max_files = 4096;

    /// Per-batch tables. Inputs go host to device, outputs come back; one copy each way.
    struct table_view {
        uint64_t* compressed_offsets;  // host only
        char** compressed_ptrs;
        uint64_t* compressed_bytes;
        uint64_t* output_bytes;
        char** output_ptrs;
        uint64_t* slots;  // max_files + 1
        // Outputs.
        uint64_t* inflated_bytes;
        nvcompStatus_t* statuses;
        uint32_t* crcs;
        unsigned long long* first_header;
        unsigned long long* kept;
        unsigned long long* header_count;
        int64_t* selected;
        char* first_char;
    };

    static constexpr size_t inputs_bytes() noexcept {
        return 6 * max_files * 8 + 8;
    }

    static constexpr size_t outputs_bytes() noexcept {
        return max_files * (8 + sizeof(nvcompStatus_t) + 4 + 8 + 8 + 1) + 16;
    }

    static constexpr size_t host_table_bytes() noexcept {
        return inputs_bytes() + outputs_bytes();
    }

    static table_view view(char* base) {
        table_view view{};
        auto* at = base;
        auto take = [&]<typename T>(T*& field, size_t count) {
            field = reinterpret_cast<T*>(at);
            at += count * sizeof(T);
        };
        take(view.compressed_offsets, max_files);
        take(view.compressed_ptrs, max_files);
        take(view.compressed_bytes, max_files);
        take(view.output_bytes, max_files);
        take(view.output_ptrs, max_files);
        take(view.slots, max_files + 1);
        take(view.inflated_bytes, max_files);
        take(view.first_header, max_files);
        take(view.kept, max_files);
        take(view.header_count, 1);
        take(view.selected, 1);
        take(view.statuses, max_files);
        take(view.crcs, max_files);
        take(view.first_char, max_files);
        return view;
    }

    struct lane_state {
        std::optional<cuda::stream> stream;
        std::optional<cuda::event> uploaded, classified, released;
        std::optional<cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>>
            pinned, host;
        std::optional<cuda::device_buffer<char>> compressed, slots, tables, temp;
        std::optional<cuda::device_buffer<uint64_t>> headers;
        std::vector<size_t> files, fallback;
        std::array<bool, max_files> possible_members;
        uint64_t slot_bytes = 0;
    };

    static table_view host_table(lane_state& lane) {
        return view(lane.host->data());
    }

    static table_view device_tables(lane_state& lane) {
        return view(lane.tables->data());
    }

    size_t workers_;
    size_t compressed_capacity_ = 0, slot_capacity_ = 0;
    size_t nvcomp_temp_bytes_ = 0, select_temp_bytes_ = 0;
    uint32_t header_capacity_ = 0;
    std::array<lane_state, lane_count> lanes_;
};

}  // namespace cuddl::detail
