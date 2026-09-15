#pragma once

#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>
#include <zlib.h>

#include <cuddl/detail/sequence_encode.cuh>
#include <cuddl/fastx.hpp>
#include <cuddl/reference_database.cuh>
#include <cuddl/sketch.cuh>

namespace cuddl::detail {

// The file format uses little-endian integers, never native struct representations.
template <typename T>
inline T database_file_little_endian(T value) {
    if constexpr (std::endian::native == std::endian::little) {
        return value;
    } else {
        T result = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            result = static_cast<T>((result << 8U) | (value & 0xffU));
            value >>= 8U;
        }
        return result;
    }
}

struct database_file_reader {
    std::ifstream input;
    uint64_t remaining{};
    uLong checksum = crc32(0, nullptr, 0);

    Result<void> bytes(void* destination, size_t size) {
        if (size > remaining ||
            size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            return Err(Error::invalid_argument("truncated reference database file"));
        }
        if (size != 0) {
            input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
            if (!input) return Err(Error::resource("cannot read reference database file"));
            checksum = crc32_z(checksum, static_cast<Bytef const*>(destination), size);
        }
        remaining -= size;
        return Ok();
    }

    template <typename T>
    Result<void> value(T& item) {
        CUDDL_TRY(bytes(&item, sizeof(item)));
        item = database_file_little_endian(item);
        return Ok();
    }

    Result<void> words(std::vector<uint32_t>& items) {
        CUDDL_TRY(bytes(items.data(), items.size() * sizeof(uint32_t)));
        if constexpr (std::endian::native != std::endian::little) {
            for (auto& item : items) {
                item = database_file_little_endian(item);
            }
        }
        return Ok();
    }
};

struct database_file_writer {
    std::string temporary;
    FILE* output{};
    uLong checksum = crc32(0, nullptr, 0);

    ~database_file_writer() {
        if (output) std::fclose(output);
        if (!temporary.empty()) {
            ::unlink(temporary.c_str());
        }
    }

    Result<void> bytes(void const* source, size_t size) {
        if (size != 0) {
            if (std::fwrite(source, 1, size, output) != size) {
                return Err(Error::resource("cannot write reference database file"));
            }
            checksum = crc32_z(checksum, static_cast<Bytef const*>(source), size);
        }
        return Ok();
    }

    template <typename T>
    Result<void> value(T item) {
        item = database_file_little_endian(item);
        return bytes(&item, sizeof(item));
    }

    Result<void> words(std::vector<uint32_t> const& items) {
        if constexpr (std::endian::native == std::endian::little) {
            return bytes(items.data(), items.size() * sizeof(uint32_t));
        } else {
            for (auto item : items) {
                CUDDL_TRY(value(item));
            }
            return Ok();
        }
    }
};

template <typename IO>
Result<void> database_file_metadata(IO& io, reference_database_metadata& metadata) {
    auto& c = metadata.compatibility;
    CUDDL_TRY(io.value(c.kmer_length));
    CUDDL_TRY(io.value(c.bucket_count));
    CUDDL_TRY(io.value(c.indexed_bucket_count));
    CUDDL_TRY(io.value(c.score_encoder_identity));
    CUDDL_TRY(io.value(c.exponent_bits));
    CUDDL_TRY(io.value(c.mantissa_bits));
    CUDDL_TRY(io.value(c.hash_identity));
    CUDDL_TRY(io.value(c.hash_seed));
    CUDDL_TRY(io.value(c.canonicalisation_policy));
    CUDDL_TRY(io.value(c.blacklist_identity));
    CUDDL_TRY(io.value(c.blacklist_version));
    CUDDL_TRY(io.value(c.key_mask));
    CUDDL_TRY(io.value(metadata.reference_count));
    return Ok();
}

// Page-locked sequence storage handed to loader threads. A lease returns to the pool when the
// parsed file holding it dies, which is after the caller enqueued every copy that reads it.
// Releasing records that point on the stream; the next loader to take the slot waits for it
// there, so a buffer is never rewritten under a DMA and the build never has to drain.
class pinned_sequence_pool {
   public:
    /// @p limit bounds one buffer; larger genomes stay on the loader's own growing buffer.
    pinned_sequence_pool(cuda::stream_ref stream, size_t buffers, size_t limit)
        : stream_(stream), limit_(limit) {
        slots_.reserve(buffers);
        in_use_.reserve(buffers);
        consumed_.reserve(buffers);
    }

    /// @brief Number of buffers the pool may grow to.
    void set_capacity(size_t buffers) noexcept {
        capacity_ = std::max<size_t>(1, buffers);
    }

    /// @brief Buffers allocated so far. Read once the loaders have stopped.
    [[nodiscard]] size_t buffers() const noexcept {
        return slots_.size();
    }

    /// @brief Returns page-locked bytes, or a null target when the pool cannot serve.
    [[nodiscard]] decompression_target acquire(size_t bytes) {
        if (bytes == 0 || bytes > limit_) return {};
        size_t index = 0;
        {
            std::lock_guard lock(mutex_);
            index = claim(bytes);
            if (index == slots_.size()) return {};
        }
        // Wait on the loader thread rather than the build loop: the slot is already reserved,
        // so this blocks only the genome that needs it. A failing wait throws out of the loader,
        // which reports it through the pool's error slot like any other load failure.
        consumed_[index].sync();
        return lease(index);
    }

   private:
    /// @brief Reserves a free slot holding at least @p bytes, or returns slots_.size().
    [[nodiscard]] size_t claim(size_t bytes) {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (in_use_[i]) continue;
            if (slots_[i]->buffer.size() < bytes) {
                // Grow a free buffer instead of falling back. Genome sizes vary within a
                // corpus, and a buffer sized for the first small genome would otherwise
                // never serve the larger ones. Rounding up to a power of two bounds how many
                // times any buffer is reallocated, and page-locked allocation is not cheap.
                auto grown = size_t{1} << 16;
                while (grown < bytes) grown *= 2;
                slots_[i] = std::make_unique<slot>(
                    cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>(
                        stream_, cuda::pinned_default_memory_pool(), grown, cuda::no_init
                    )
                );
            }
            in_use_[i] = true;
            return i;
        }
        if (slots_.size() >= capacity_) return slots_.size();
        slots_.push_back(
            std::make_unique<slot>(
                cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>(
                    stream_, cuda::pinned_default_memory_pool(), bytes, cuda::no_init
                )
            )
        );
        in_use_.push_back(true);
        consumed_.emplace_back(stream_);
        return slots_.size() - 1;
    }

   private:
    struct slot {
        cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible> buffer;
    };

    [[nodiscard]] decompression_target lease(size_t index) {
        auto* pool = this;
        auto* owner = slots_[index].get();
        return {
            owner->buffer.data(),
            owner->buffer.size(),
            std::shared_ptr<void>(owner, [pool, index](void*) {
                std::lock_guard lock(pool->mutex_);
                // Record before publishing the slot: a loader that takes it must see the
                // event of every copy the previous lease fed.
                pool->consumed_[index].record(pool->stream_);
                pool->in_use_[index] = false;
            })
        };
    }

    cuda::stream_ref stream_;
    size_t limit_;
    size_t capacity_{1};
    std::vector<std::unique_ptr<slot>> slots_;
    std::vector<bool> in_use_;
    std::vector<cuda::event> consumed_;
    std::mutex mutex_;
};

/// @brief `decompression_source` trampoline for `pinned_sequence_pool`.
[[nodiscard]] inline decompression_target acquire_pinned_target(void* context, size_t bytes) {
    return static_cast<pinned_sequence_pool*>(context)->acquire(bytes);
}

// Fixed worker pool loading FASTX files ahead of the GPU loop. A pthread spawn and join
// per genome costs tens of microseconds and dominates at high file counts. Workers pull
// ids in order while the main thread takes results in order, so at most depth loads are
// in flight and output order never depends on completion order. Loader exceptions are
// rethrown by take, matching std::async propagation into the build error handlers.
class fastx_load_pool {
   public:
    fastx_load_pool(
        std::span<std::filesystem::path const> paths,
        size_t depth,
        pinned_sequence_pool* pinned
    )
        : paths_(paths),
          depth_(std::max(size_t{1}, depth)),
          pinned_(pinned),
          results_(paths.size()),
          errors_(paths.size()) {
        workers_.reserve(depth_);
        for (size_t i = 0; i < depth_; ++i) workers_.emplace_back([this] { work(); });
    }
    ~fastx_load_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        assign_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    fastx_load_pool(fastx_load_pool const&) = delete;
    fastx_load_pool& operator=(fastx_load_pool const&) = delete;

    [[nodiscard]] Result<std::unique_ptr<fastx_sequence_file>> take(size_t id) {
        std::unique_lock lock(mutex_);
        filled_.wait(lock, [&] { return results_[id].has_value() || errors_[id] != nullptr; });
        ++taken_;
        assign_.notify_all();
        lock.unlock();
        if (errors_[id] != nullptr) std::rethrow_exception(errors_[id]);
        return std::move(*results_[id]);
    }

   private:
    void work() {
        while (true) {
            size_t id;
            {
                std::unique_lock lock(mutex_);
                assign_.wait(lock, [&] {
                    return stop_ || next_ >= paths_.size() || next_ - taken_ < depth_;
                });
                if (stop_ || next_ >= paths_.size()) return;
                if (next_ - taken_ >= depth_) continue;
                id = next_++;
            }
            try {
                auto loaded = load_fastx_sequence_file(
                    paths_[id].string(),
                    pinned_ != nullptr
                        ? decompression_source{acquire_pinned_target, pinned_}
                        : decompression_source{}
                );
                std::lock_guard lock(mutex_);
                results_[id] = std::move(loaded);
            } catch (...) {
                std::lock_guard lock(mutex_);
                errors_[id] = std::current_exception();
            }
            filled_.notify_all();
        }
    }
    std::span<std::filesystem::path const> paths_;
    size_t depth_;
    pinned_sequence_pool* pinned_;
    std::vector<std::optional<Result<std::unique_ptr<fastx_sequence_file>>>> results_;
    std::vector<std::exception_ptr> errors_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable assign_, filled_;
    size_t next_ = 0, taken_ = 0;
    bool stop_ = false;
};

}  // namespace cuddl::detail

namespace cuddl {

/// @brief Whether a build transfers decompressed bytes from page-locked memory by default.
///
/// Page-locked transfers win when the host writes to device-visible memory at full speed:
/// measured 2.1x over staging on an x86 host with a discrete GPU. On a coherent CPU/GPU system
/// the same mapping costs host writes more than the staging copy it removes, which is 2.9x
/// slower end to end on Grace Hopper. The architecture picks the default; `pinned` overrides it.
#if defined(__aarch64__)
inline constexpr bool default_pinned_transfer = false;
#else
inline constexpr bool default_pinned_transfer = true;
#endif

/// @brief How a build moved sequence bytes to the device.
///
/// A run that reports every byte as `direct` is reading the decompressed genome in place; a
/// large `staged` share means page-locked buffers were unavailable and bytes were copied into
/// staging on the consumer thread instead.
struct reference_build_statistics {
    size_t direct_bytes = 0;
    size_t staged_bytes = 0;
    size_t direct_chunks = 0;
    size_t staged_chunks = 0;
    size_t pinned_buffers = 0;
    size_t staging_bytes = 0;
    size_t batches = 0;
    size_t pieces = 0;
};

/**
 * @brief Host-owned reference sketches and labels, ready for binary storage or GPU upload.
 *
 * One input file is one genome, including all of its FASTA/FASTQ records. Input order defines
 * stable reference IDs; labels preserve the supplied paths. Files store packed winner/count
 * rows, saturation flags and compatibility metadata. GPU indexes are rebuilt on upload.
 * Build and upload are synchronous. The supplied stream must outlive the uploaded database.
 */
class reference_database_file {
   public:
    [[nodiscard]] reference_database_metadata metadata() const noexcept {
        return metadata_;
    }
    [[nodiscard]] std::span<std::string const> names() const noexcept {
        return names_;
    }
    /// @brief Packed winner/count registers in reference-ID order, one bucket row per genome.
    [[nodiscard]] std::span<uint32_t const> rows() const noexcept {
        return rows_;
    }
    /// @brief Saturation flags in reference-ID order.
    [[nodiscard]] std::span<uint32_t const> saturation() const noexcept {
        return saturation_;
    }

    /**
     * @brief Builds one sketch per plain or gzip/BGZF FASTA/FASTQ file on the GPU.
     *
     * K-mers never cross record boundaries or ambiguous bases. FASTQ qualities are validated
     * and ignored. Loader threads compact and stage records into a device arena sized from free
     * device memory, and one batch kernel encodes every record in it, so the device is fed in
     * large batches rather than slice by slice. Register rows stay on the device for a group of
     * genomes and return to the host once, on the single drain at the end of the build.
     * @p staging_bytes caps the arena explicitly; zero sizes it from free device memory.
     * Zero workers selects up to eight. Use one worker to load only one genome at a time.
     * Empty genomes retain their IDs and have zero rows.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    /// @p pinned transfers decompressed bytes straight from page-locked memory. Disabling it
    /// restores the heap buffer plus a staging copy, which some hosts do better; the
    /// `direct_bytes` and `staged_bytes` counters say which path a build actually took.
    [[nodiscard]] static Result<reference_database_file> build(
        std::span<std::filesystem::path const> paths,
        cuda::stream_ref stream,
        unsigned parser_workers = 0,
        reference_build_statistics* statistics = nullptr,
        bool pinned = default_pinned_transfer,
        size_t staging_bytes = 0
    ) try {
        // NVCC 13.3 crashes on CUDDL_TRY directly inside a try block. Keep its GNU statement
        // expressions in a separate lambda scope, outside the exception-catching function.
        return [&]() -> Result<reference_database_file> {
            using database_type = reference_database<K, BucketCount, Layout>;
            if (paths.size() > std::numeric_limits<uint32_t>::max() / BucketCount ||
                paths.size() >
                    std::numeric_limits<size_t>::max() / (BucketCount * sizeof(uint32_t))) {
                return Err(Error::resource("reference collection exceeds database index capacity"));
            }
            reference_database_file result;
            result.metadata_ = {
                score_compatibility::current<K, BucketCount, Layout>(),
                static_cast<uint32_t>(paths.size())
            };
            result.rows_.resize(paths.size() * BucketCount);
            result.saturation_.resize(paths.size());
            result.names_.reserve(paths.size());
            // No inputs means no staging budget to resolve and nothing to encode.
            if (paths.empty()) return result;
            // Larger genomes keep the loader's own buffer rather than pinning that much RAM
            // per loader for the whole run.
            constexpr size_t pinned_limit = size_t{32} << 20;
            auto const hardware_threads = std::max(1U, std::thread::hardware_concurrency());
            auto const workers = std::max(
                size_t{1},
                std::min(
                    paths.size(),
                    size_t{std::min(parser_workers ? parser_workers : 8U, hardware_threads)}
                )
            );
            // Page-locked sequence buffers, one per loader, so the transfer engine reads the
            // decompressed genome in place instead of restaging it on the consumer thread.
            detail::pinned_sequence_pool pinned_buffers(stream, workers, pinned_limit);
            // One buffer per in-flight file, plus headroom: a worker asks for its next file
            // while every loaded file still holds a lease, so an exact match would refuse.
            pinned_buffers.set_capacity(workers + 2);
            auto* pinned_pool = pinned ? &pinned_buffers : nullptr;
            std::optional<detail::fastx_load_pool> loader;
            if (workers > 1) loader.emplace(paths, workers, pinned_pool);
            // Device budget: one arena of staged bases plus a store of register rows. The arena
            // is sized from what is actually free, so a batch is as large as the device allows
            // and one batch kernel covers every record staged into it, instead of a launch per
            // slice with the host waiting on the device between them.
            size_t free_bytes = 0, device_bytes = 0;
            CUDDL_CUDA_TRY(cudaMemGetInfo(&free_bytes, &device_bytes));
            auto const& device_pool = cuda::device_default_memory_pool(stream.device());
            auto const pool_reserved =
                device_pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
            auto const pool_used =
                device_pool.attribute(cuda::memory_pool_attributes::used_mem_current);
            // Cached pool storage is reused by the next allocation, so it counts as available.
            size_t const available = free_bytes + pool_reserved - std::min(pool_reserved, pool_used);
            size_t const usable = available - available / 10;
            // (BucketCount + 1) words hold one genome's registers plus its saturation flag.
            size_t const row_words = BucketCount + 1;
            size_t const row_bytes = row_words * sizeof(uint32_t);
            // Rows are small next to the input, so the whole collection stays resident whenever
            // that costs at most a quarter of what is affordable; otherwise rows stream back one
            // group at a time as each group completes.
            size_t const group = std::min(paths.size(), std::max<size_t>(1, usable / (4 * row_bytes)));
            uint64_t input_bytes = 0;
            for (auto const& path : paths) {
                std::error_code error;
                auto const size = std::filesystem::file_size(path, error);
                if (!error) input_bytes += size;
            }
            // At most six staged bytes per input byte, so a small collection does not reserve an
            // arena it could never fill.
            size_t const arena_ceiling =
                static_cast<size_t>(std::min<uint64_t>(input_bytes * 6, uint64_t{6} << 40));
            size_t const row_store_bytes = group * row_bytes;
            size_t const after_rows = usable > row_store_bytes ? usable - row_store_bytes : 0;
            size_t staging = staging_bytes;
            if (staging == 0) {
                staging = std::min(after_rows, arena_ceiling);
            }
            // One piece of at least k bases plus its k - 1 byte overlap has to fit.
            if (staging < K + 1 || staging > after_rows) {
                return Err(Error::resource(
                    "insufficient free device memory for reference staging (free " +
                    std::to_string(free_bytes >> 20) + " MiB)"
                ));
            }
            // One descriptor per staged piece; the cap keeps a corpus of very short records from
            // demanding a descriptor per record.
            size_t const max_pieces =
                std::min<size_t>(size_t{1} << 20, std::max<size_t>(64, staging / K));
            // Pieces are a quarter of the arena at most, so a batch always holds several, and
            // short enough that a piece's window count stays representable.
            size_t const max_piece = std::min(
                {std::max<size_t>(K, staging / 4),
                 static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - K + 1}
            );
            auto rows = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint32_t>(
                    stream, stream.device(), group * row_words, cuda::no_init
                )
            );
            auto arena = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<char>(stream, stream.device(), staging, cuda::no_init)
            );
            auto descriptors = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<detail::sequence_batch_chunk>(
                    stream, stream.device(), max_pieces, cuda::no_init
                )
            );
            std::vector<detail::sequence_batch_chunk> staged_chunks;
            staged_chunks.reserve(std::min<size_t>(max_pieces, size_t{1} << 16));
            auto const sm = static_cast<size_t>(
                stream.device().attribute(cuda::device_attributes::multiprocessor_count)
            );
            auto const max_grid = static_cast<size_t>(
                stream.device().attribute(cuda::device_attributes::max_grid_dim_x)
            );
            size_t arena_used = 0, block_end = 0, batches = 0, pieces = 0;

            // Stages one span of bases and records its windows. A span that continues a record
            // was widened by k - 1 bases on the host, so it holds one window per base it adds.
            auto stage = [&](char const* source, size_t size, size_t genome, uint32_t windows
                         ) -> Result<void> {
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{source, size},
                        device_span<char>{arena.data() + arena_used, size}
                    )
                );
                auto const blocks =
                    windows == 0
                        ? size_t{0}
                        : std::min(sm * 2, (static_cast<size_t>(windows) + 2047) / 2048);
                block_end += blocks;
                staged_chunks.push_back(
                    {arena_used, block_end, static_cast<uint32_t>(genome), windows}
                );
                arena_used += size;
                ++pieces;
                return Ok();
            };
            // One launch for everything staged so far. Copies queued after it land in the arena
            // only once the device has finished reading it, so no host wait is needed here.
            auto flush = [&]() -> Result<void> {
                if (staged_chunks.empty()) return Ok();
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{staged_chunks.data(), staged_chunks.size()},
                        device_span<detail::sequence_batch_chunk>{
                            descriptors.data(), staged_chunks.size()
                        }
                    )
                );
                auto const grid = std::min(block_end, max_grid);
                if (grid != 0) {
                    detail::add_sequence_batch_kernel<BucketCount, Layout>
                        <<<static_cast<uint32_t>(grid), 256, 0, stream.get()>>>(
                            arena.data(),
                            descriptors.data(),
                            staged_chunks.size(),
                            block_end,
                            K,
                            rows.data()
                        );
                    CUDDL_CUDA_TRY(cudaGetLastError());
                }
                ++batches;
                staged_chunks.clear();
                arena_used = 0;
                block_end = 0;
                return Ok();
            };
            size_t base = 0;
            while (base < paths.size()) {
                auto const count = std::min(group, paths.size() - base);
                CUDDL_CUDA_TRY(
                    cuda::fill_bytes(
                        stream, cuda::std::span{rows.data(), count * row_words}, uint32_t{0}
                    )
                );
                for (size_t id = base; id < base + count; ++id) {
                    auto sequence = CUDDL_TRY(
                        workers == 1
                            ? detail::load_fastx_sequence_file(
                                  paths[id].string(),
                                  pinned_pool != nullptr
                                      ? detail::decompression_source{
                                            detail::acquire_pinned_target, pinned_pool
                                        }
                                      : detail::decompression_source{}
                              )
                            : loader->take(id)
                    );
                    auto const* const pinned_base = sequence->decompressed_target;
                    auto const pinned_size = sequence->decompressed_size;
                    auto const genome = id - base;
                    for (auto const& extent : sequence->extents) {
                        auto const* const record = extent.begin;
                        auto const size = static_cast<size_t>(extent.end - extent.begin);
                        size_t offset = 0;
                        while (offset < size) {
                            // A continuation re-stages the previous piece's last k - 1 bases, so
                            // kmers spanning a piece boundary survive the split.
                            auto const overlap =
                                offset == 0 ? size_t{0} : std::min(offset, size_t{K - 1});
                            auto const bases = std::min(size - offset, max_piece);
                            auto const span = overlap + bases;
                            if (span < K) break;  // fewer bases than one window
                            if (staged_chunks.size() >= max_pieces || arena_used + span > staging) {
                                CUDDL_TRY(flush());
                            }
                            auto const* const source = record + offset - overlap;
                            auto const direct = pinned_base != nullptr && source >= pinned_base &&
                                                source + span <= pinned_base + pinned_size;
                            if (statistics != nullptr) {
                                if (direct) {
                                    statistics->direct_bytes += span;
                                    ++statistics->direct_chunks;
                                } else {
                                    statistics->staged_bytes += span;
                                    ++statistics->staged_chunks;
                                }
                            }
                            CUDDL_TRY(
                                stage(source, span, genome, static_cast<uint32_t>(span - K + 1))
                            );
                            offset += bases;
                        }
                    }
                    result.names_.push_back(paths[id].string());
                    sequence.reset();
                }
                // The group's rows are read back after its last batch, so a batch never spans a
                // group boundary and the copies below always follow the writes they read.
                CUDDL_TRY(flush());
                for (size_t i = 0; i < count; ++i) {
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream,
                            device_span<uint32_t const>{rows.data() + i * row_words, BucketCount},
                            cuda::std::span{
                                result.rows_.data() + (base + i) * BucketCount, BucketCount
                            }
                        )
                    );
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream,
                            device_span<uint32_t const>{
                                rows.data() + i * row_words + BucketCount, 1
                            },
                            cuda::std::span{result.saturation_.data() + base + i, size_t{1}}
                        )
                    );
                }
                base += count;
            }
            // One drain for the whole collection: every row copy above queues on this stream in
            // reference order, so nothing needs to synchronise per genome.
            CUDDL_CUDA_TRY(stream.sync());
            // Loaders still hold leases; join them before reading pool state.
            loader.reset();
            if (statistics != nullptr) {
                statistics->pinned_buffers = pinned_buffers.buffers();
                statistics->staging_bytes = staging;
                statistics->batches = batches;
                statistics->pieces = pieces;
            }
            // Instantiate the same constraints as the destination GPU database.
            static_assert(sizeof(database_type) > 0);
            return result;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Uploads packed rows and builds a sparse index by default; waits for completion.
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] Result<reference_database<K, BucketCount, Layout>>
    upload(cuda::stream_ref stream, index_storage storage = index_storage::sparse) const {
        CUDDL_TRY((detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(
            metadata_.compatibility
        )));
        auto rows =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, stream.device(), rows_));
        auto saturation = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), saturation_)
        );
        auto database = CUDDL_TRY((reference_database<K, BucketCount, Layout>::build_indexed_async(
            {rows.data(), rows.size()},
            {saturation.data(), saturation.size()},
            metadata_.compatibility,
            stream,
            storage
        )));
        CUDDL_CUDA_TRY(stream.sync());
        return std::move(database);
    }

    /**
     * @brief Writes a versioned, little-endian, CRC32-checked binary file.
     *
     * Writes to an exclusive temporary file in the destination directory, closes it, then
     * atomically replaces the destination. Failure leaves an existing destination intact.
     * This guarantees atomic publication, not durability across power loss.
     */
    [[nodiscard]] Result<void> save(std::filesystem::path const& path) const try {
        return [&]() -> Result<void> {
            detail::database_file_writer writer;
            writer.temporary = path.string() + ".tmp.XXXXXX";
            auto fd = ::mkstemp(writer.temporary.data());
            if (fd < 0) {
                writer.temporary.clear();
                return Err(Error::resource("cannot create database file beside: " + path.string()));
            }
            writer.output = ::fdopen(fd, "wb");
            if (!writer.output) {
                ::close(fd);
                return Err(Error::resource("cannot open temporary database file"));
            }
            CUDDL_TRY(writer.bytes("CUDDLDB\0", 8));
            CUDDL_TRY(writer.value(uint32_t{1}));
            auto metadata = metadata_;
            CUDDL_TRY(detail::database_file_metadata(writer, metadata));
            for (auto const& name : names_) {
                if (name.size() > std::numeric_limits<uint32_t>::max()) {
                    return Err(Error::resource("reference label exceeds binary format capacity"));
                }
                CUDDL_TRY(writer.value(static_cast<uint32_t>(name.size())));
                CUDDL_TRY(writer.bytes(name.data(), name.size()));
            }
            CUDDL_TRY(writer.words(rows_));
            CUDDL_TRY(writer.words(saturation_));
            CUDDL_TRY(writer.value(static_cast<uint32_t>(writer.checksum)));
            auto closed = std::fclose(std::exchange(writer.output, nullptr));
            if (closed != 0) return Err(Error::resource("cannot close database file"));
            std::filesystem::rename(writer.temporary, path);
            writer.temporary.clear();
            return Ok();
        }();
    } catch (std::filesystem::filesystem_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Reads and validates a binary database without requiring a GPU.
    [[nodiscard]] static Result<reference_database_file> load(
        std::filesystem::path const& path
    ) try {
        return [&]() -> Result<reference_database_file> {
            detail::database_file_reader reader{
                std::ifstream(path, std::ios::binary | std::ios::ate)
            };
            auto size = reader.input.tellg();
            if (size < 0) {
                return Err(Error::resource("cannot open database file: " + path.string()));
            }
            reader.remaining = static_cast<uint64_t>(size);
            reader.input.seekg(0);
            char magic[8];
            CUDDL_TRY(reader.bytes(magic, sizeof(magic)));
            if (std::string_view(magic, sizeof(magic)) != std::string_view("CUDDLDB\0", 8)) {
                return Err(Error::invalid_argument("not a cuDDL reference database file"));
            }
            uint32_t version{};
            CUDDL_TRY(reader.value(version));
            if (version != 1) {
                return Err(Error::invalid_argument("unsupported database file version"));
            }
            reference_database_file result;
            CUDDL_TRY(detail::database_file_metadata(reader, result.metadata_));
            auto const& c = result.metadata_.compatibility;
            if (c.kmer_length < 1 || c.kmer_length > 31 || c.bucket_count < 2048 ||
                c.bucket_count > 131072 || !std::has_single_bit(c.bucket_count) ||
                c.indexed_bucket_count != c.bucket_count || c.key_mask != 0xffffU ||
                c.exponent_bits == 0 || c.mantissa_bits == 0 ||
                c.exponent_bits + c.mantissa_bits != 16 || c.score_encoder_identity != 1 ||
                c.hash_identity != 1 || c.hash_seed != detail::seed ||
                c.canonicalisation_policy != 1 || c.blacklist_identity != 0 ||
                c.blacklist_version != 0) {
                return Err(Error::invalid_argument("unsupported database construction metadata"));
            }
            uint64_t const count = result.metadata_.reference_count;
            uint64_t const words = count * c.bucket_count;
            uint64_t const payload_bytes = (words + count) * sizeof(uint32_t);
            if (words > std::numeric_limits<uint32_t>::max() ||
                words > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
                payload_bytes + count * sizeof(uint32_t) + sizeof(uint32_t) > reader.remaining) {
                return Err(Error::invalid_argument("invalid reference database extents"));
            }
            result.names_.reserve(count);
            for (uint64_t id = 0; id < count; ++id) {
                uint32_t length{};
                CUDDL_TRY(reader.value(length));
                auto const required =
                    payload_bytes + (count - id - 1) * sizeof(uint32_t) + sizeof(uint32_t);
                if (required > reader.remaining || length > reader.remaining - required) {
                    return Err(Error::invalid_argument("invalid reference label extent"));
                }
                std::string name(length, '\0');
                CUDDL_TRY(reader.bytes(name.data(), name.size()));
                result.names_.push_back(std::move(name));
            }
            if (reader.remaining != payload_bytes + sizeof(uint32_t)) {
                return Err(Error::invalid_argument("unexpected reference database payload size"));
            }
            result.rows_.resize(words);
            result.saturation_.resize(count);
            CUDDL_TRY(reader.words(result.rows_));
            CUDDL_TRY(reader.words(result.saturation_));
            auto const expected_checksum = static_cast<uint32_t>(reader.checksum);
            uint32_t checksum{};
            CUDDL_TRY(reader.value(checksum));
            if (checksum != expected_checksum) {
                return Err(Error::invalid_argument("database checksum mismatch"));
            }
            for (auto state : result.saturation_) {
                if (state > 1) {
                    return Err(Error::invalid_argument("invalid database saturation flag"));
                }
            }
            for (auto row : result.rows_) {
                if ((detail::winner(row) == 0) != (detail::count(row) == 0)) {
                    return Err(Error::invalid_argument("invalid packed database register"));
                }
            }
            return result;
        }();
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

   private:
    reference_database_file() = default;
    reference_database_metadata metadata_{};
    std::vector<std::string> names_;
    std::vector<uint32_t> rows_;
    std::vector<uint32_t> saturation_;
};

}  // namespace cuddl
