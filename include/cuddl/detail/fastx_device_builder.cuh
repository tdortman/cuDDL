#pragma once

#include <cub/device/device_select.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/std/bit>
#include <cuda/stream>
#include <cusbf/Alphabet.cuh>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>
#include <cuda/devices>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

struct sequence_byte {
    __device__ bool operator()(char c) const {
        return c != '\n' && c != '\r' && c != ' ' && c != '\t';
    }
};

// A tile encodes each ASCII base once; each thread rolls eight adjacent windows.
// cuSBF's symbol-tile helpers require compile-time Bloom-filter Config; this path
// accepts runtime k and feeds cuDDL's existing register update/merge operations.
template <size_t BucketCount, typename Layout>
__device__ __forceinline__ void add_sequence_windows(
    char const* sequence,
    uint32_t windows,
    size_t first_tile,
    size_t tile_stride,
    uint32_t k,
    uint32_t* registers,
    uint32_t& saturation
) {
    if (windows == 0) return;
    constexpr uint32_t tile_size = 256 * 8;
    constexpr bool shared_sketch = BucketCount <= 8192;
    __shared__ uint8_t bases[tile_size + 30];
    __shared__ uint32_t local[shared_sketch ? BucketCount : 1];
    if constexpr (shared_sketch) {
        for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
            local[i] = 0;
        }
    }
    auto* target = shared_sketch ? local : registers;
    auto const mask = (uint64_t{1} << (2 * k)) - 1;
    for (size_t tile = first_tile; tile < windows; tile += tile_stride) {
        auto const count = min(tile_size, windows - static_cast<uint32_t>(tile));
        for (uint32_t i = threadIdx.x; i < count + k - 1; i += blockDim.x) {
            bases[i] = cusbf::DnaAlphabet::encode(sequence + tile + i);
        }
        __syncthreads();
        auto const start = threadIdx.x * 8;
        if (start < count) {
            uint64_t forward = 0;
            uint32_t valid = 0;
            auto const end = min(start + 8, count) + k - 1;
            for (uint32_t i = start; i < end; ++i) {
                auto const symbol = bases[i];
                forward = ((forward << 2) | (symbol & 3U)) & mask;
                valid = symbol == cusbf::DnaAlphabet::invalidSymbol ? 0 : valid + 1;
                if (valid >= k) {
                    auto const bits = cuda::std::bit_reverse(forward);
                    auto const pairs = ((bits & 0xAAAAAAAAAAAAAAAAULL) >> 1) |
                                       ((bits & 0x5555555555555555ULL) << 1);
                    auto const reverse = (pairs ^ 0xAAAAAAAAAAAAAAAAULL) >> (64 - 2 * k);
                    auto const hash = hash_kmer(forward > reverse ? forward : reverse);
                    update(&target[bucket_of<BucketCount>(hash)], score<Layout>(hash), saturation);
                }
            }
        }
        __syncthreads();
    }
    if constexpr (shared_sketch) {
        for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
            merge_register(&registers[i], local[i], saturation);
        }
    }
}

template <size_t BucketCount, typename Layout>
__global__ void add_sequence_tile_kernel(
    char const* sequence,
    uint32_t const* window_count,
    char* carry,
    uint32_t k,
    uint32_t* registers,
    uint32_t& saturation
) {
    auto const windows = *window_count;
    if (blockIdx.x == 0 && threadIdx.x < k - 1) {
        carry[threadIdx.x] = sequence[windows + threadIdx.x];
    }
    add_sequence_windows<BucketCount, Layout>(
        sequence,
        windows,
        size_t{blockIdx.x} * 2048,
        size_t{gridDim.x} * 2048,
        k,
        registers,
        saturation
    );
}

struct sequence_batch_chunk {
    size_t offset;
    size_t block_end;
    uint32_t genome;
    uint32_t windows;
};

// Logical blocks cover every chunk in one launch, including independent short records.
template <size_t BucketCount, typename Layout>
__global__ void add_sequence_batch_kernel(
    char const* sequence,
    sequence_batch_chunk const* chunks,
    size_t chunk_count,
    size_t block_count,
    uint32_t k,
    uint32_t* registers
) {
    __shared__ sequence_batch_chunk chunk;
    __shared__ size_t first_block;
    for (size_t block = blockIdx.x; block < block_count; block += gridDim.x) {
        if (threadIdx.x == 0) {
            size_t low = 0, high = chunk_count;
            while (low < high) {
                auto const middle = low + (high - low) / 2;
                if (chunks[middle].block_end <= block) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            chunk = chunks[low];
            first_block = low ? chunks[low - 1].block_end : 0;
        }
        __syncthreads();
        auto* target = registers + size_t{chunk.genome} * (BucketCount + 1);
        add_sequence_windows<BucketCount, Layout>(
            sequence + chunk.offset,
            chunk.windows,
            (block - first_block) * 2048,
            (chunk.block_end - first_block) * 2048,
            k,
            target,
            target[BucketCount]
        );
        __syncthreads();
    }
}

// Persistent stripe-copy workers for host staging. Single-thread copies from
// file-backed mappings stall the GPU feed (measured ~4.7 GB/s alone versus ~13 GB/s
// with eight stripes); the caller copies stripe zero and waits for the rest.
class staging_copy_pool {
   public:
    explicit staging_copy_pool(unsigned workers) {
        workers_.reserve(workers);
        for (unsigned stripe = 1; stripe <= workers; ++stripe) {
            workers_.emplace_back([this, stripe] { run(stripe); });
        }
    }
    ~staging_copy_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    staging_copy_pool(staging_copy_pool const&) = delete;
    staging_copy_pool& operator=(staging_copy_pool const&) = delete;

    void copy(char* dst, char const* src, size_t n) {
        if (workers_.empty() || n < kParallelThreshold) {
            std::memcpy(dst, src, n);
            return;
        }
        auto const stripes = workers_.size() + 1;
        auto const span = (n + stripes - 1) / stripes;
        {
            std::lock_guard lock(mutex_);
            dst_ = dst;
            src_ = src;
            n_ = n;
            span_ = span;
            pending_ = static_cast<unsigned>(workers_.size());
            ++generation_;
        }
        wake_.notify_all();
        std::memcpy(dst, src, std::min(span, n));
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
    }

   private:
    void run(unsigned stripe) {
        unsigned seen = 0;
        while (true) {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || seen != generation_; });
            if (stop_) return;
            seen = generation_;
            char* dst = dst_;
            char const* src = src_;
            size_t n = n_, span = span_;
            lock.unlock();
            auto const begin = std::min<size_t>(stripe * span, n);
            auto const end = std::min<size_t>(begin + span, n);
            if (end > begin) std::memcpy(dst + begin, src + begin, end - begin);
            lock.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }
    static constexpr size_t kParallelThreshold = size_t{1} << 20;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    char* dst_ = nullptr;
    char const* src_ = nullptr;
    size_t n_ = 0, span_ = 0;
    unsigned pending_ = 0, generation_ = 0;
    bool stop_ = false;
};

[[nodiscard]] inline unsigned staging_copy_worker_count() noexcept {
    auto const hardware = std::max(1U, std::thread::hardware_concurrency());
    return std::min(5U, hardware) - 1U;
}

class fastx_device_builder {
   public:
    static constexpr size_t capacity = size_t{1} << 22;

    explicit fastx_device_builder(cuda::stream_ref stream)
        : transfer_(stream.device()),
          uploaded_(stream.device()),
          raw_(
              cuda::make_device_buffer<char>(stream, stream.device(), 2 * capacity, cuda::no_init)
          ),
          clean_(
              cuda::make_device_buffer<char>(stream, stream.device(), capacity + 31, cuda::no_init)
          ),
          carry_(cuda::make_device_buffer<char>(stream, stream.device(), 31, cuda::no_init)),
          scratch_(
              cuda::make_device_buffer<unsigned char>(stream, stream.device(), 0, cuda::no_init)
          ),
          counts_(cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, cuda::no_init)),
          upload_(stream, cuda::pinned_default_memory_pool(), 2 * capacity, cuda::no_init),
          consumed_{cuda::event{stream}, cuda::event{stream}},
          staging_(staging_copy_worker_count()) {}

    Result<void> prepare(cuda::stream_ref stream) {
        CUDDL_CUDA_TRY(transfer_.wait(consumed_[0]));
        size_t compact_bytes = 0;
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                nullptr,
                compact_bytes,
                raw_.data(),
                clean_.data(),
                counts_.data(),
                static_cast<int32_t>(capacity),
                sequence_byte{},
                stream.get()
            )
        );
        if (compact_bytes > (size_t{16} << 20) - capacity * 3 - 66) {
            return Err(Error::resource("FASTX GPU workspace exceeds 16 MiB staging budget"));
        }
        scratch_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<unsigned char>(
                stream, stream.device(), compact_bytes, cuda::no_init
            )
        );
        return Ok();
    }

    void reset() noexcept {
        reset_carry_ = true;
    }

    // Alternate pinned/device slots so the next upload overlaps sketch construction.
    // Wait only before slot reuse. An invalid prefix excludes windows before a record.
    template <size_t BucketCount, typename Layout = default_register_layout>
    Result<void> add(
        std::string_view bytes,
        uint32_t k,
        uint32_t* registers,
        uint32_t& saturation,
        cuda::stream_ref stream
    ) {
        if (bytes.size() > capacity || k < 1 || k > 31) {
            return Err(Error::invalid_argument("invalid FASTX packing chunk"));
        }
        if (bytes.empty()) return Ok();
        CUDDL_CUDA_TRY(consumed_[slot_].sync());
        auto* input = upload_.data() + slot_ * capacity;
        staging_.copy(input, bytes.data(), bytes.size());
        auto* device_input = raw_.data() + slot_ * capacity;
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                transfer_,
                cuda::std::span{input, bytes.size()},
                device_span<char>{device_input, bytes.size()}
            )
        );
        CUDDL_CUDA_TRY(uploaded_.record(transfer_));
        CUDDL_CUDA_TRY(stream.wait(uploaded_));
        if (reset_carry_) {
            CUDDL_CUDA_TRY(cuda::fill_bytes(stream, carry_, 'N'));
            reset_carry_ = false;
        }
        if (k > 1) {
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream,
                    device_span<char const>{carry_.data(), size_t{k - 1}},
                    device_span<char>{clean_.data(), size_t{k - 1}}
                )
            );
        }
        auto scratch_bytes = scratch_.size();
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                scratch_.data(),
                scratch_bytes,
                device_input,
                clean_.data() + k - 1,
                counts_.data(),
                static_cast<int32_t>(bytes.size()),
                sequence_byte{},
                stream.get()
            )
        );
        // Both DMA and CUB have finished with this slot before it can be reused.
        CUDDL_CUDA_TRY(consumed_[slot_].record(stream));
        slot_ ^= 1;
        auto const sm = stream.device().attribute(cuda::device_attributes::multiprocessor_count);
        auto const blocks = std::min<uint32_t>(sm * 2, (bytes.size() + 2047) / 2048);
        add_sequence_tile_kernel<BucketCount, Layout><<<blocks, 256, 0, stream.get()>>>(
            clean_.data(), counts_.data(), carry_.data(), k, registers, saturation
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        return Ok();
    }

   private:
    cuda::stream transfer_;
    cuda::event uploaded_;
    cuda::device_buffer<char> raw_, clean_, carry_;
    cuda::device_buffer<unsigned char> scratch_;
    cuda::device_buffer<uint32_t> counts_;
    cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible> upload_;
    cuda::event consumed_[2];
    staging_copy_pool staging_;
    unsigned slot_ = 0;
    bool reset_carry_ = true;
};

}  // namespace cuddl::detail
