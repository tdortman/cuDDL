#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cuddl/detail/fastx_sequence_file.hpp>
#include <cuddl/detail/sequence_encode.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief How a build gets its sequence bytes to the device.
enum class transfer_mode {
    /// Page-locked buffers where the host writes them well, in-place staging where the device
    /// reads pageable memory itself, and a staging copy everywhere else.
    automatic,
    /// Decompress into page-locked buffers, which the transfer engine reads directly. Measured
    /// 2.1x faster than staging on an x86 host with a discrete GPU, and slower on a coherent
    /// system, where the same mapping costs host writes more than the copy it removes.
    pinned,
    /// Decompress into a heap buffer and copy it to the device. On a coherent system that copy
    /// is a single-threaded bounce through the driver, which is what leaves 72 loaders parked
    /// while one core copies and the device waits for data.
    staged,
    /// Decompress into a heap buffer and let the kernels read it there. Only a device that reads
    /// pageable host memory can do this, and it pays neither the copy nor the page-locking.
    in_place,
};

/// @brief Whether @p device reads pageable host memory itself.
///
/// This is what makes a CPU/GPU system coherent, Grace Hopper and Grace Blackwell among them,
/// and what lets a kernel read the loader's buffer instead of a copy of it.
[[nodiscard]] inline bool device_reads_pageable_memory(cuda::device_ref device) noexcept {
    return device.attribute(cuda::device_attributes::pageable_memory_access) != 0 &&
           device.attribute(cuda::device_attributes::pageable_memory_access_uses_host_page_tables
           ) != 0;
}

/// @brief Whether a build in @p mode stages the caller's bytes in place.
[[nodiscard]] inline bool stages_in_place(transfer_mode mode, cuda::device_ref device) noexcept {
    if (mode == transfer_mode::in_place) return true;
    if (mode != transfer_mode::automatic) return false;
    return device_reads_pageable_memory(device);
}

/// @brief Whether a build in @p mode decompresses into page-locked buffers.
[[nodiscard]] inline bool pages_locked(transfer_mode mode, cuda::device_ref device) noexcept {
    if (mode == transfer_mode::pinned) return true;
    if (mode != transfer_mode::automatic) return false;
    return !device_reads_pageable_memory(device);
}

/// @brief Sizing a build resolves before its first genome.
struct staging_plan {
    size_t group = 0;       // genomes whose register rows stay resident at once
    size_t staging = 0;     // arena bytes
    size_t max_piece = 0;   // bytes one piece may occupy
    size_t max_pieces = 0;  // descriptors one batch may hold
};

/// @brief Sizes the arena and the row store from free device memory and the corpus.
///
/// @p staged_ceiling bounds the bytes the corpus can stage, which keeps a small collection from
/// reserving an arena it could never fill. An unset @p staging_bytes sizes the arena from what
/// is free; a value is honoured, or rejected when it does not fit.
template <uint32_t K, size_t BucketCount>
[[nodiscard]] inline Result<staging_plan> plan_staging(
    size_t references,
    uint64_t staged_ceiling,
    std::optional<size_t> staging_bytes,
    cuda::stream_ref stream
) {
    size_t free_bytes = 0, device_bytes = 0;
    CUDDL_CUDA_TRY(cudaMemGetInfo(&free_bytes, &device_bytes));
    auto const& device_pool = cuda::device_default_memory_pool(stream.device());
    auto const pool_reserved =
        device_pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
    auto const pool_used = device_pool.attribute(cuda::memory_pool_attributes::used_mem_current);
    // Cached pool storage is reused by the next allocation, so it counts as available.
    size_t const available = free_bytes + pool_reserved - std::min(pool_reserved, pool_used);
    size_t const usable = available - available / 10;
    // (BucketCount + 1) words hold one genome's registers plus its saturation flag.
    size_t const row_bytes = (BucketCount + 1) * sizeof(uint32_t);
    // Rows are small next to the input, so the whole collection stays resident whenever that
    // costs at most a quarter of what is affordable; otherwise rows stream back one group at a
    // time as each group completes.
    size_t const group = std::min(references, std::max<size_t>(1, usable / (4 * row_bytes)));
    // Four fifths of what is affordable goes to the arena, which leaves room for the row store,
    // the descriptors and whatever the caller keeps on the device. On a coherent CPU/GPU system
    // the free memory reported here is host memory, so a share of host memory bounds it as well.
    auto const pages = ::sysconf(_SC_PHYS_PAGES);
    auto const page = ::sysconf(_SC_PAGE_SIZE);
    size_t const host_share = pages > 0 && page > 0
                                  ? static_cast<size_t>(pages) * static_cast<size_t>(page) / 8
                                  : std::numeric_limits<size_t>::max();
    size_t const arena_ceiling = static_cast<size_t>(std::min(
        {staged_ceiling,
         static_cast<uint64_t>(usable - usable / 5),
         static_cast<uint64_t>(host_share)}
    ));
    size_t const row_store_bytes = group * row_bytes;
    size_t const after_rows = usable > row_store_bytes ? usable - row_store_bytes : 0;
    if (staging_bytes.has_value() && *staging_bytes < K + 1) {
        return Err(Error::invalid_argument("an explicit staging arena must hold one window"));
    }
    size_t const staging =
        staging_bytes.has_value() ? *staging_bytes : std::min(after_rows, arena_ceiling);
    // One piece of at least k bases plus its k - 1 byte overlap has to fit.
    if (staging < K + 1 || staging > after_rows) {
        return Err(
            Error::resource(
                "insufficient free device memory for reference staging (free " +
                std::to_string(free_bytes >> 20) + " MiB)"
            )
        );
    }
    // One descriptor per staged piece; the cap keeps a corpus of very short records from
    // demanding a descriptor per record.
    size_t const max_pieces = std::min<size_t>(size_t{1} << 20, std::max<size_t>(64, staging / K));
    // Pieces are a quarter of the arena at most, so a batch always holds several, and short
    // enough that a piece's window count stays representable.
    size_t const max_piece = std::min(
        {std::max<size_t>(K, staging / 4),
         static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - K + 1}
    );
    return staging_plan{group, staging, max_piece, max_pieces};
}

/// @brief Encodes genome records into register rows through a device arena.
///
/// One copy of the staging machinery, fed by every entry point that builds a reference database
/// from bases. A producer supplies, per genome, the spans of its records: one span per record,
/// bases only, no line breaks, and no k-mer crossing two spans.
///
/// The arena, its descriptors, the record runs, the `k - 1` overlap a record too large for the
/// arena needs, the batch launches and the row readback all live here. A producer keeps only
/// what is its own: where the bytes come from, and which of them must outlive the copies the
/// device makes from them.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class database_stager {
   public:
    /// Host memory the device still has to read must stay alive: the transfer engine's read of a
    /// pageable buffer is not ordered with the host writes that follow it. Sixteen slots take the
    /// tail off that wait; at four the tail reached hundreds of milliseconds while dozens of
    /// loaders fed one consumer.
    static constexpr size_t hold_slots = 16;

    static constexpr size_t row_words() noexcept {
        return BucketCount + 1;
    }

    /// @throws cuda::cuda_error or std::bad_alloc when the device buffers cannot be allocated.
    /// @param direct stages the caller's bytes in place, for a device that reads pageable host
    /// memory: no copy, and the arena stays unused.
    database_stager(
        cuda::stream_ref stream,
        size_t references,
        staging_plan bounds,
        reference_build_statistics* statistics = nullptr,
        bool direct = false
    )
        : stream_(stream),
          bounds_(bounds),
          statistics_(statistics),
          direct_(direct),
          rows_(
              cuda::make_device_buffer<uint32_t>(
                  stream,
                  stream.device(),
                  bounds.group * row_words(),
                  cuda::no_init
              )
          ),
          arena_(
              cuda::make_device_buffer<char>(stream, stream.device(), bounds.staging, cuda::no_init)
          ),
          descriptors_(
              cuda::make_device_buffer<sequence_batch_chunk>(
                  stream,
                  stream.device(),
                  bounds.max_pieces,
                  cuda::no_init
              )
          ),
          host_rows_(references * row_words()),
          held_(hold_slots),
          sm_(static_cast<size_t>(
              stream.device().attribute(cuda::device_attributes::multiprocessor_count)
          )),
          max_grid_(
              static_cast<size_t>(
                  stream.device().attribute(cuda::device_attributes::max_grid_dim_x)
              )
          ) {
        held_consumed_.reserve(hold_slots);
        for (size_t slot = 0; slot < hold_slots; ++slot) {
            held_consumed_.emplace_back(stream);
        }
        staged_chunks_.reserve(std::min<size_t>(bounds_.max_pieces, size_t{1} << 16));
        copied_chunks_.reserve(staged_chunks_.capacity());
    }

    /// @brief Stages one genome. @p holder keeps its bytes alive until the device is done.
    ///
    /// A genome with no records, or with records shorter than one window, keeps its row and
    /// records no descriptor. @p pinned_base names the page-locked buffer the bytes came from,
    /// when the producer has one, so the statistics can say which transfer path was taken.
    [[nodiscard]] Result<void> add_genome(
        size_t genome,
        std::span<fastx_sequence_extent const> records,
        char const* pinned_base = nullptr,
        size_t pinned_size = 0,
        std::unique_ptr<fastx_sequence_file> holder = {}
    ) {
        auto const record_size = [&](size_t index) {
            return static_cast<size_t>(records[index].end - records[index].begin);
        };
        size_t record = 0;
        while (record < records.size()) {
            // Staged in place, the arena is not the limit; the descriptors and the piece size are.
            auto const room = direct_ ? std::numeric_limits<size_t>::max()
                                      : bounds_.staging - arena_used_;
            size_t run = record;
            while (run < records.size()) {
                auto const run_bytes =
                    static_cast<size_t>(records[run].end - records[record].begin);
                if (run_bytes > room || record_size(run) > bounds_.max_piece ||
                    staged_chunks_.size() + (run - record) >= bounds_.max_pieces) {
                    break;
                }
                ++run;
            }
            if (run == record) {
                if (arena_used_ != 0) {
                    CUDDL_TRY(flush());
                    continue;
                }
                // One record larger than the whole arena: split it, and give each piece the
                // k - 1 bases of overlap its kmers need.
                auto const size = record_size(record);
                size_t offset = 0;
                while (offset < size) {
                    auto const overlap = offset == 0 ? size_t{0} : std::min(offset, size_t{K - 1});
                    auto const bases = std::min(size - offset, bounds_.max_piece);
                    auto const span = overlap + bases;
                    if (span < K) break;
                    if (arena_used_ + span > bounds_.staging ||
                        staged_chunks_.size() >= bounds_.max_pieces) {
                        CUDDL_TRY(flush());
                    }
                    auto const* const source = records[record].begin + offset - overlap;
                    CUDDL_TRY(stage(
                        source,
                        span,
                        genome,
                        static_cast<uint32_t>(span - K + 1),
                        pinned_base,
                        pinned_size
                    ));
                    offset += bases;
                }
                ++record;
                continue;
            }
            // Copy the run in one transfer, then describe each record inside it. Staged in
            // place, the records keep their own addresses and there is nothing to copy.
            auto const* const source = records[record].begin;
            auto const run_bytes = static_cast<size_t>(records[run - 1].end - source);
            account(source, run_bytes, pinned_base, pinned_size);
            if (!direct_) {
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream_,
                        cuda::std::span{source, run_bytes},
                        device_span<char>{arena_.data() + arena_used_, run_bytes}
                    )
                );
            }
            for (size_t index = record; index < run; ++index) {
                auto const size = record_size(index);
                if (size < K) continue;  // no window to record
                auto const* const bases =
                    direct_ ? records[index].begin
                            : arena_.data() + arena_used_ +
                                  static_cast<size_t>(records[index].begin - source);
                describe(bases, static_cast<uint32_t>(size - K + 1), genome);
            }
            if (!direct_) {
                arena_used_ += run_bytes;
                ++transfers_;
            }
            record = run;
        }
        if (holder) hold(std::move(holder));
        return Ok();
    }

    /// @brief Clears the register rows of the next group of genomes.
    [[nodiscard]] Result<void> begin_group(size_t genomes) {
        CUDDL_CUDA_TRY(
            cuda::fill_bytes(
                stream_, cuda::std::span{rows_.data(), genomes * row_words()}, uint32_t{0}
            )
        );
        return Ok();
    }

    /// @brief Launches everything staged, then copies the group's rows towards the host.
    [[nodiscard]] Result<void> end_group(size_t base, size_t genomes) {
        CUDDL_TRY(flush());
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream_,
                device_span<uint32_t const>{rows_.data(), genomes * row_words()},
                cuda::std::span{host_rows_.data() + base * row_words(), genomes * row_words()}
            )
        );
        return Ok();
    }

    /// @brief Waits for the device, then hands over the store it filled.
    ///
    /// The store holds `row_words()` words per genome: `BucketCount` packed registers followed by
    /// that genome's saturation word, the layout a single sketch allocation has.
    [[nodiscard]] Result<std::vector<uint32_t>> release_store() {
        CUDDL_CUDA_TRY(stream_.sync());
        if (statistics_ != nullptr) {
            statistics_->staging_bytes = bounds_.staging;
            statistics_->batches = batches_;
            statistics_->transfers = transfers_;
        }
        return std::move(host_rows_);
    }

   private:
    /// @brief Records one piece of sequence for the kernel, naming its bytes.
    void describe(char const* bases, uint32_t windows, size_t genome) {
        auto const blocks = std::min(sm_ * 2, (static_cast<size_t>(windows) + 2047) / 2048);
        block_end_ += blocks;
        staged_chunks_.push_back({bases, block_end_, static_cast<uint32_t>(genome), windows});
    }

    void account(char const* source, size_t size, char const* pinned_base, size_t pinned_size) {
        if (statistics_ == nullptr) return;
        auto const direct = pinned_base != nullptr && source >= pinned_base &&
                            source + size <= pinned_base + pinned_size;
        if (direct) {
            statistics_->direct_bytes += size;
            ++statistics_->direct_chunks;
        } else {
            statistics_->staged_bytes += size;
            ++statistics_->staged_chunks;
        }
    }

    /// @brief Stages one span of bases and records its windows.
    ///
    /// A span that continues a record was widened by k - 1 bases, so it holds one window per
    /// base it adds.
    [[nodiscard]] Result<void> stage(
        char const* source,
        size_t size,
        size_t genome,
        uint32_t windows,
        char const* pinned_base,
        size_t pinned_size
    ) {
        account(source, size, pinned_base, pinned_size);
        if (direct_) {
            // The device reads the caller's bytes where they are, so nothing is copied and the
            // arena stays empty; the hold keeps those bytes alive until the batch has run.
            describe(source, windows, genome);
            return Ok();
        }
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream_,
                cuda::std::span{source, size},
                device_span<char>{arena_.data() + arena_used_, size}
            )
        );
        describe(arena_.data() + arena_used_, windows, genome);
        arena_used_ += size;
        ++transfers_;
        return Ok();
    }

    /// @brief One launch for everything staged so far.
    ///
    /// Copies queued after it land in the arena only once the device has finished reading it, so
    /// no host wait is needed beyond the descriptor copy's own.
    [[nodiscard]] Result<void> flush() {
        if (staged_chunks_.empty()) {
            // Records shorter than k stage bytes but record no window, so a flush can arrive
            // with nothing to launch. The offsets still have to reset, or a caller that flushed
            // to make room would ask again for the same room.
            arena_used_ = 0;
            block_end_ = 0;
            return Ok();
        }
        copied_consumed_.sync();
        copied_chunks_.swap(staged_chunks_);
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream_,
                cuda::std::span{copied_chunks_.data(), copied_chunks_.size()},
                device_span<sequence_batch_chunk>{descriptors_.data(), copied_chunks_.size()}
            )
        );
        copied_consumed_.record(stream_);
        auto const grid = std::min(block_end_, max_grid_);
        if (grid != 0) {
            add_sequence_batch_kernel<BucketCount, Layout>
                <<<static_cast<uint32_t>(grid), 256, 0, stream_.get()>>>(
                    descriptors_.data(),
                    copied_chunks_.size(),
                    block_end_,
                    K,
                    rows_.data()
                );
            CUDDL_CUDA_TRY(cudaGetLastError());
        }
        ++batches_;
        staged_chunks_.clear();
        arena_used_ = 0;
        block_end_ = 0;
        return Ok();
    }

    /// @brief Keeps one genome's bytes until the device has taken every copy made from them.
    void hold(std::unique_ptr<fastx_sequence_file> holder) {
        auto const slot = held_count_++ % held_.size();
        if (held_[slot]) {
            held_consumed_[slot].sync();
            held_[slot].reset();
        }
        held_consumed_[slot].record(stream_);
        held_[slot] = std::move(holder);
    }

    cuda::stream_ref stream_;
    staging_plan bounds_;
    reference_build_statistics* statistics_;
    bool direct_ = false;
    cuda::device_buffer<uint32_t> rows_;
    cuda::device_buffer<char> arena_;
    cuda::device_buffer<sequence_batch_chunk> descriptors_;
    std::vector<uint32_t> host_rows_;
    std::vector<std::optional<std::unique_ptr<fastx_sequence_file>>> held_;
    std::vector<cuda::event> held_consumed_;
    cuda::event copied_consumed_{stream_};
    std::vector<sequence_batch_chunk> staged_chunks_, copied_chunks_;
    size_t const sm_, max_grid_;
    size_t arena_used_ = 0, block_end_ = 0, batches_ = 0, transfers_ = 0, held_count_ = 0;
};

/// @brief One genome of bases the caller already holds.
///
/// Records are the shape a parsed FASTX record has: bases only, no header, no line breaks, no
/// FASTQ qualities. k-mers never cross two records.
struct sequence_record {
    std::string_view bases;
};

/// @brief One genome as its records, in order. No records keeps the genome's ID with a zero row.
struct sequence_genome {
    std::span<sequence_record const> records;
    std::string_view name;  // copied into the labels; may be empty
};

/// @brief Unpacks a store into packed rows and one saturation word per genome.
///
/// The store pads each genome with its saturation word, so a build unpacked as it read the store
/// would copy once per genome; this reads the padding it already holds.
template <size_t BucketCount>
void unpack_store(
    std::span<uint32_t const> store,
    std::span<uint32_t> rows,
    std::span<uint32_t> saturation
) noexcept {
    for (size_t genome = 0; genome < saturation.size(); ++genome) {
        std::memcpy(
            rows.data() + genome * BucketCount,
            store.data() + genome * (BucketCount + 1),
            BucketCount * sizeof(uint32_t)
        );
        saturation[genome] = store[genome * (BucketCount + 1) + BucketCount];
    }
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

/// @brief Loader state a build fed by paths keeps for its whole run.
///
/// One page-locked sequence buffer per loader lets the transfer engine read the decompressed
/// genome in place instead of restaging it on the consumer thread. The pool hands out load
/// results a few files ahead of the consumer, which takes them in order: a window of one
/// worker-worth leaves the consumer waiting on a straggler while every other loader sits idle.
class path_loaders {
   public:
    /// @param page_locked gives the loaders page-locked buffers to decompress into.
    path_loaders(
        cuda::stream_ref stream,
        std::span<std::filesystem::path const> paths,
        unsigned parser_workers,
        bool page_locked
    )
        : buffers_(stream, worker_count(paths.size(), parser_workers), size_t{32} << 20),
          page_locked_(page_locked) {
        // One buffer per in-flight file, plus headroom: a worker asks for its next file while
        // every loaded file still holds a lease, so an exact match would refuse.
        buffers_.set_capacity(workers() + 2);
        if (page_locked_) source_ = {acquire_pinned_target, &buffers_};
        if (workers() > 1) pool_.emplace(paths, workers(), source_, workers() * 4);
    }

    /// @brief Loaders this build runs.
    [[nodiscard]] size_t workers() const noexcept {
        return buffers_.buffers();
    }

    /// @brief Page-locked buffers the loaders held, for the statistics.
    [[nodiscard]] size_t page_locked_buffers() const noexcept {
        return buffers_.buffers();
    }

    /// @brief Loads genome @p id, through the pool when the build runs more than one loader.
    [[nodiscard]] Result<std::unique_ptr<fastx_sequence_file>>
    take(std::span<std::filesystem::path const> paths, size_t id) {
        if (pool_.has_value()) return pool_->take(id);
        return load_fastx_sequence_file(paths[id].string(), source_);
    }

    /// @brief Joins the loaders. They hold leases, so pool state is only readable after this.
    void reset() noexcept {
        pool_.reset();
    }

   private:
    /// @brief Loaders to run: at most one per input, at most the requested workers, at most the
    /// hardware, and never none.
    static size_t worker_count(size_t paths, unsigned parser_workers) noexcept {
        auto const hardware = std::max(1U, std::thread::hardware_concurrency());
        auto const loaders = static_cast<size_t>(std::min(parser_workers, hardware));
        return std::max<size_t>(1, std::min(paths, loaders));
    }

    pinned_sequence_pool buffers_;
    bool page_locked_ = false;
    decompression_source source_{};
    std::optional<fastx_load_pool> pool_;
};

/// @brief Runs the staging loop over a collection, filling one group at a time.
///
/// @p fill receives the stager, a group's first genome and its size, and stages that whole group.
/// Genomes are numbered within their group, so a fill stages genome @c genome - @c base.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout, typename Fill>
[[nodiscard]] Result<std::vector<uint32_t>> stage_groups(
    size_t genomes,
    uint64_t staged_ceiling,
    std::optional<size_t> staging_bytes,
    cuda::stream_ref stream,
    reference_build_statistics* statistics,
    Fill&& fill,
    bool in_place = false
) {
    auto const plan =
        CUDDL_TRY((plan_staging<K, BucketCount>(genomes, staged_ceiling, staging_bytes, stream)));
    database_stager<K, BucketCount, Layout> stager(stream, genomes, plan, statistics, in_place);
    size_t base = 0;
    while (base < genomes) {
        auto const count = std::min(plan.group, genomes - base);
        CUDDL_TRY(stager.begin_group(count));
        CUDDL_TRY(fill(stager, base, count));
        CUDDL_TRY(stager.end_group(base, count));
        base += count;
    }
    return stager.release_store();
}

/// @brief Stages a path collection and returns the store.
///
/// @p parser_workers is a value, not a sentinel: the build clamps it to what the inputs and the
/// hardware allow, so asking for the default is asking for the default.
///
/// The arena ceiling is the decompressed size of every input: a bound from compressed sizes alone
/// overestimates a corpus several times over, and an arena past what the corpus can hold is
/// memory the device never needs.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] Result<std::vector<uint32_t>> stage_paths(
    std::span<std::filesystem::path const> paths,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    unsigned parser_workers,
    transfer_mode transfer,
    reference_build_statistics* statistics
) {
    if (parser_workers == 0) {
        return Err(Error::invalid_argument("a path build needs at least one loader"));
    }
    if (paths.empty()) return std::vector<uint32_t>{};
    auto const in_place = stages_in_place(transfer, stream.device());
    auto const page_locked = pages_locked(transfer, stream.device());
    path_loaders loaders(stream, paths, parser_workers, page_locked);
    uint64_t staged_ceiling = 0;
    for (auto const& path : paths) {
        auto const decompressed = gzip_decompressed_size(path.string());
        if (decompressed != 0) {
            staged_ceiling += decompressed;
            continue;
        }
        std::error_code error;
        auto const size = std::filesystem::file_size(path, error);
        if (!error) staged_ceiling += size;
    }
    auto store = CUDDL_TRY((stage_groups<K, BucketCount, Layout>(
        paths.size(),
        staged_ceiling,
        staging_bytes,
        stream,
        statistics,
        [&](database_stager<K, BucketCount, Layout>& stager, size_t base, size_t count)
            -> Result<void> {
            for (size_t id = base; id < base + count; ++id) {
                auto sequence = CUDDL_TRY(loaders.take(paths, id));
                // Read the loaded file's parts before the move below: argument order is
                // unspecified, so a moved-from Result must not be dereferenced.
                auto const* const pinned_base = sequence->decompressed_target;
                auto const pinned_size = sequence->decompressed_size;
                auto const& extents = sequence->extents;
                CUDDL_TRY(stager.add_genome(
                    id - base, extents, pinned_base, pinned_size, std::move(sequence)
                ));
            }
            return Ok();
        },
        in_place
    )));
    if (statistics != nullptr) {
        statistics->pinned_buffers = loaders.page_locked_buffers();
        statistics->workers = static_cast<unsigned>(loaders.workers());
        statistics->in_place = in_place;
    }
    loaders.reset();
    return store;
}

/// @brief Stages bases the caller already holds and returns the store.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] Result<std::vector<uint32_t>> stage_sequences(
    std::span<sequence_genome const> genomes,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    reference_build_statistics* statistics
) {
    if (genomes.empty()) return std::vector<uint32_t>{};
    // The caller knows exactly what the corpus holds, so the ceiling is the sum of its bases
    // rather than an estimate read out of file headers.
    uint64_t staged_ceiling = 0;
    for (auto const& genome : genomes) {
        for (auto const& record : genome.records) staged_ceiling += record.bases.size();
    }
    std::vector<fastx_sequence_extent> records;
    return stage_groups<K, BucketCount, Layout>(
        genomes.size(),
        staged_ceiling,
        staging_bytes,
        stream,
        statistics,
        [&](database_stager<K, BucketCount, Layout>& stager, size_t base, size_t count)
            -> Result<void> {
            for (size_t id = base; id < base + count; ++id) {
                auto const& genome = genomes[id];
                records.clear();
                records.reserve(genome.records.size());
                for (auto const& record : genome.records) {
                    if (record.bases.empty()) continue;
                    records.push_back(
                        {record.bases.data(), record.bases.data() + record.bases.size()}
                    );
                }
                CUDDL_TRY(stager.add_genome(id - base, records));
            }
            return Ok();
        }
    );
}

}  // namespace cuddl::detail
