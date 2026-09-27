#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#ifndef CUDDL_HAS_NVCOMP
    #define CUDDL_HAS_NVCOMP 0
#endif

#if CUDDL_HAS_NVCOMP
    #include <cuddl/detail/device_gzip.cuh>
#endif
#include <cuddl/detail/fastx_sequence_file.hpp>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/detail/sequence_encode.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Where inputs are decompressed; sketch construction always runs on the GPU.
enum class decompression_backend {
    automatic,
    cpu,
    /// nvCOMP for eligible gzip inputs; unsupported formats use host loaders.
    gpu,
    /// CPU inflation, followed by GPU FASTA normalisation through coherent host memory.
    coherent,
};

/// @brief How a build gets CPU-parsed sequence bytes to the device.
enum class transfer_mode {
    /// In-place host reads on coherent devices, page-locked transfers elsewhere.
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
           device.attribute(
               cuda::device_attributes::pageable_memory_access_uses_host_page_tables
           ) != 0;
}

/// @brief Whether a build in @p mode stages the caller's bytes in place.
///
/// Automatic transfer reads CPU-decoded buffers in place on coherent devices.
[[nodiscard]] inline bool stages_in_place(transfer_mode mode, cuda::device_ref device) noexcept {
    return mode == transfer_mode::in_place ||
           (mode == transfer_mode::automatic && device_reads_pageable_memory(device));
}

/// @brief Whether a build in @p mode decompresses into page-locked buffers.
///
/// `automatic` keys on the device rather than the architecture: a device that reads pageable host
/// memory is coherent, and there the host writes to page-locked memory are the slow part.
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

/// Default arena ceiling. A batch is sketched only once its arena fills, so an arena that holds
/// the whole corpus leaves the device idle until the last genome loads. Bounding it lets each
/// batch's kernel overlap the loading of the next: on a GH200 building 2048 genomes this cut the
/// build from 239 to 194 ms, and it cost nothing on a 24-thread RTX 5070 Ti host.
inline constexpr size_t default_arena_ceiling = size_t{128} << 20;

/// Staging arenas the copy stream rotates through, so copies run ahead of the kernels that
/// read earlier arenas while other work shares the device.
inline constexpr size_t stager_arenas = 4;

/// @brief Device bytes an allocation on @p stream can still get.
///
/// Storage the default pool caches from earlier allocations is reused by the next one, so it
/// counts as available alongside what the device reports free.
[[nodiscard]] inline Result<size_t> available_device_bytes(cuda::stream_ref stream) {
    size_t free_bytes = 0, device_bytes = 0;
    CUDDL_CUDA_TRY(cudaMemGetInfo(&free_bytes, &device_bytes));
    auto const& device_pool = cuda::device_default_memory_pool(stream.device());
    auto const pool_reserved =
        device_pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
    auto const pool_used = device_pool.attribute(cuda::memory_pool_attributes::used_mem_current);
    return free_bytes + pool_reserved - std::min(pool_reserved, pool_used);
}

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
    size_t const available = CUDDL_TRY(available_device_bytes(stream));
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
         static_cast<uint64_t>(host_share),
         static_cast<uint64_t>(default_arena_ceiling)}
    ));
    size_t const row_store_bytes = group * row_bytes;
    size_t const after_rows = usable > row_store_bytes ? usable - row_store_bytes : 0;
    if (staging_bytes.has_value() && *staging_bytes < K + 1) {
        return Err(Error::invalid_argument("an explicit staging arena must hold one window"));
    }
    // Every arena gets the same share: one fills while kernels read the others.
    size_t const staging = staging_bytes.has_value()
                               ? *staging_bytes
                               : std::min(after_rows / stager_arenas, arena_ceiling);
    // One piece of at least k bases plus its k - 1 byte overlap has to fit.
    if (staging < K + 1 || staging > after_rows / stager_arenas) {
        return Err(
            Error::resource(
                "insufficient free device memory for reference staging (free " +
                std::to_string(available >> 20) + " MiB)"
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

/// Host memory the device still has to read must stay alive: the transfer engine's read of a
/// pageable buffer is not ordered with the host writes that follow it. Sixteen slots take the
/// tail off that wait; at four the tail reached hundreds of milliseconds while dozens of
/// loaders fed one consumer.
inline constexpr size_t stager_hold_slots = 16;

/// @brief Encodes genome records into register rows through a device arena.
///
/// One copy of the staging machinery, fed by every entry point that builds a reference database
/// from bases. A producer supplies, per genome, the spans of its records: one span per record,
/// bases only, no line breaks, and no k-mer crossing two spans.
///
/// The arena, its descriptors, the record runs, the `k - 1` overlap a record too large for the
/// arena needs and the batch launches all live here. Rows stay on the device: each group's rows
/// go to the caller's sink, which copies them wherever the build wants them. A producer keeps
/// only what is its own: where the bytes come from, and which of them must outlive the copies
/// the device makes from them.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class database_stager {
    struct resident_event_span {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;

        resident_event_span() = default;
        resident_event_span(resident_event_span const&) = delete;
        resident_event_span& operator=(resident_event_span const&) = delete;
        resident_event_span(resident_event_span&& other) noexcept
            : start(std::exchange(other.start, nullptr)),
              stop(std::exchange(other.stop, nullptr)) {}
        resident_event_span& operator=(resident_event_span&& other) noexcept {
            if (this == &other) return *this;
            if (start != nullptr) CUDDL_CUDA_ABORT(cudaEventDestroy(start));
            if (stop != nullptr) CUDDL_CUDA_ABORT(cudaEventDestroy(stop));
            start = std::exchange(other.start, nullptr);
            stop = std::exchange(other.stop, nullptr);
            return *this;
        }
        ~resident_event_span() {
            if (start != nullptr) CUDDL_CUDA_ABORT(cudaEventDestroy(start));
            if (stop != nullptr) CUDDL_CUDA_ABORT(cudaEventDestroy(stop));
        }
    };

   public:
    static constexpr size_t hold_slots = stager_hold_slots;

    static constexpr size_t row_words() noexcept {
        return BucketCount + 1;
    }

    /// @throws cuda::cuda_error or std::bad_alloc when the device buffers cannot be allocated.
    /// @param stream Stream owning the arena, descriptors, and row store.
    /// @param bounds Arena sizing the device row store, arena, and descriptors.
    /// @param statistics Optional build statistics sink, null to skip.
    /// @param direct stages the caller's bytes in place, for a device that reads pageable host
    /// memory: no copy, and the arena stays unused.
    database_stager(
        cuda::stream_ref stream,
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
          copy_stream_(stream.device()),
          kernel_stream_(stream.device(), greatest_stream_priority()),
          // The copy stream fills one arena while kernels read the others.
          arena_(
              cuda::make_device_buffer<char>(
                  stream,
                  stream.device(),
                  direct ? bounds.staging : stager_arenas * arena_stride(bounds.staging),
                  cuda::no_init
              )
          ),
          descriptors_(
              cuda::make_device_buffer<sequence_batch_chunk>(
                  stream,
                  stream.device(),
                  bounds.max_pieces,
                  cuda::no_init
              )
          ),
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
        // The arenas are allocated in order on the main stream.
        copy_stream_.wait(stream_);
    }
    /// Frees the arenas only after every copy into them.
    ~database_stager() {
        CUDDL_CUDA_ABORT(cudaEventRecord(copies_done_.get(), copy_stream_.get()));
        CUDDL_CUDA_ABORT(cudaStreamWaitEvent(stream_.get(), copies_done_.get()));
    }
    database_stager(database_stager const&) = delete;
    database_stager& operator=(database_stager const&) = delete;

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
            auto const room =
                direct_ ? std::numeric_limits<size_t>::max() : bounds_.staging - arena_used_;
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
                        copy_stream_,
                        cuda::std::span{source, run_bytes},
                        device_span<char>{arena() + arena_used_, run_bytes}
                    )
                );
            }
            for (size_t index = record; index < run; ++index) {
                auto const size = record_size(index);
                if (size < K) continue;  // no window to record
                auto const* const bases =
                    direct_ ? records[index].begin
                            : arena() + arena_used_ +
                                  static_cast<size_t>(records[index].begin - source);
                describe(bases, static_cast<uint32_t>(size - K + 1), genome);
            }
            if (direct_) {
                // Staged in place the arena never fills, so the bytes held for the device are
                // what bounds a batch. Half the arena, because a batch is still held while the
                // next one fills: the two together then cost what the copies would have.
                held_bytes_ += run_bytes;
                if (held_bytes_ >= bounds_.staging / 2) {
                    CUDDL_TRY(flush());
                }
            } else {
                arena_used_ += run_bytes;
                ++transfers_;
            }
            record = run;
        }
        if (holder) hold(std::move(holder));
        return Ok();
    }

    /// @brief Stages one genome that @ref pack_fastx_sequence_file packed into two-bit runs.
    ///
    /// Runs keep their words together. One too large for a batch splits into pieces of whole
    /// words whose window counts are multiples of sixteen, so every piece starts on a word.
    [[nodiscard]] Result<void> add_packed_genome(
        size_t genome,
        std::unique_ptr<fastx_sequence_file> holder,
        char const* pinned_base = nullptr,
        size_t pinned_size = 0
    ) {
        constexpr size_t word_bases = 16;
        auto const& file = *holder;
        auto const& runs = file.packed;
        auto const words = [&](size_t index) {
            return (runs[index].bases + word_bases - 1) / word_bases;
        };
        size_t run = 0;
        while (run < runs.size()) {
            auto const room = direct_ ? std::numeric_limits<size_t>::max() : arena_room_words();
            size_t last = run;
            while (last < runs.size()) {
                auto const span_words = runs[last].word + words(last) - runs[run].word;
                if (span_words > room || words(last) * sizeof(uint32_t) > bounds_.max_piece ||
                    staged_chunks_.size() + (last - run) >= bounds_.max_pieces) {
                    break;
                }
                ++last;
            }
            if (last == run) {
                if (arena_used_ != 0 || !staged_chunks_.empty()) {
                    CUDDL_TRY(flush());
                    continue;
                }
                auto const halo = (size_t{K} - 1 + word_bases - 1) / word_bases;
                auto const capacity =
                    std::min(bounds_.max_piece, bounds_.staging) / sizeof(uint32_t);
                if (capacity <= halo) {
                    return Err(Error::resource("staging arena too small for a packed run"));
                }
                auto const step = (capacity - halo) * word_bases;
                auto const total = runs[run].bases - K + 1;
                for (size_t start = 0; start < total; start += step) {
                    auto const windows = std::min(step, total - start);
                    auto const first = runs[run].word + start / word_bases;
                    auto const count = (start + windows + K - 1 + word_bases - 1) / word_bases -
                                       start / word_bases;
                    if ((!direct_ && count > arena_room_words()) ||
                        staged_chunks_.size() >= bounds_.max_pieces) {
                        CUDDL_TRY(flush());
                    }
                    auto const* const bases =
                        CUDDL_TRY(place_words(file.words + first, count, pinned_base, pinned_size));
                    describe(bases, static_cast<uint32_t>(windows), genome, true);
                }
                ++run;
                continue;
            }
            auto const first = runs[run].word;
            auto const count = runs[last - 1].word + words(last - 1) - first;
            auto const* const base =
                CUDDL_TRY(place_words(file.words + first, count, pinned_base, pinned_size));
            for (size_t index = run; index < last; ++index) {
                describe(
                    base + (runs[index].word - first) * sizeof(uint32_t),
                    static_cast<uint32_t>(runs[index].bases - K + 1),
                    genome,
                    true
                );
            }
            if (direct_ && held_bytes_ >= bounds_.staging / 2) CUDDL_TRY(flush());
            run = last;
        }
        hold(std::move(holder));
        return Ok();
    }

    /// @brief Stages one genome's bases that already sit on the device, without a copy.
    ///
    /// The bytes must stay valid until the batch that reads them runs: through the next
    /// @ref flush, whose kernel the caller then orders its reuse of them after.
    [[nodiscard]] Result<void> add_resident(size_t genome, char const* bases, size_t size) {
        if (size < K) return Ok();
        if (staged_chunks_.size() >= bounds_.max_pieces) CUDDL_TRY(flush());
        describe(bases, static_cast<uint32_t>(size - K + 1), genome);
        return Ok();
    }

    /// @brief Clears the register rows of the next group of genomes.
    [[nodiscard]] Result<void> begin_group(size_t genomes) {
        auto const resident = CUDDL_TRY(begin_resident(stream_));
        CUDDL_CUDA_TRY(
            cuda::fill_bytes(
                stream_, cuda::std::span{rows_.data(), genomes * row_words()}, uint32_t{0}
            )
        );
        CUDDL_TRY(end_resident(resident, stream_));
        return Ok();
    }

    /// @brief Launches everything staged and returns the group's rows on the device.
    ///
    /// Each genome's row holds `row_words()` words: `BucketCount` packed registers followed by
    /// its saturation word, the layout a single sketch allocation has. The rows are complete once
    /// the stream reaches this point, and the next @ref begin_group reuses them, so a consumer
    /// enqueues its reads on the stager's stream before then.
    [[nodiscard]] Result<device_span<uint32_t const>> end_group(size_t genomes) {
        CUDDL_TRY(flush());
        return device_span<uint32_t const>{rows_.data(), genomes * row_words()};
    }

    /// @brief Waits for the device and publishes the statistics.
    [[nodiscard]] Result<void> finish() {
        CUDDL_CUDA_TRY(stream_.sync());
        CUDDL_CUDA_TRY(copy_stream_.sync());
        if (statistics_ != nullptr) {
            statistics_->staging_bytes = bounds_.staging;
            statistics_->batches = batches_;
            statistics_->transfers = transfers_;
            for (auto const& span : resident_spans_) {
                float elapsed_ms = 0;
                CUDDL_CUDA_TRY(cudaEventElapsedTime(&elapsed_ms, span.start, span.stop));
                statistics_->resident_compute_ms += elapsed_ms;
            }
        }
        return Ok();
    }

   private:
    /// Offset between arenas, aligned for the two-bit words a packed genome stages.
    [[nodiscard]] static constexpr size_t arena_stride(size_t staging) noexcept {
        return (staging + 15) & ~size_t{15};
    }

    [[nodiscard]] static int greatest_stream_priority() {
        int least = 0, greatest = 0;
        auto const status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return greatest;
    }

    [[nodiscard]] Result<size_t> begin_resident(cuda::stream_ref stream) {
        if (statistics_ == nullptr || !statistics_->measure_resident) {
            return std::numeric_limits<size_t>::max();
        }
        resident_event_span span;
        CUDDL_CUDA_TRY(cudaEventCreate(&span.start));
        CUDDL_CUDA_TRY(cudaEventCreate(&span.stop));
        CUDDL_CUDA_TRY(cudaEventRecord(span.start, stream.get()));
        resident_spans_.push_back(std::move(span));
        return resident_spans_.size() - 1;
    }

    [[nodiscard]] Result<void> end_resident(size_t index, cuda::stream_ref stream) {
        if (index == std::numeric_limits<size_t>::max()) return Ok();
        CUDDL_CUDA_TRY(cudaEventRecord(resident_spans_[index].stop, stream.get()));
        return Ok();
    }

    /// @brief Records one piece of sequence for the kernel, naming its bytes.
    void describe(char const* bases, uint32_t windows, size_t genome, bool packed = false) {
        auto const blocks = std::min(sm_ * 2, (static_cast<size_t>(windows) + 2047) / 2048);
        block_end_ += blocks;
        staged_chunks_.push_back(
            {bases, block_end_, static_cast<uint32_t>(genome), windows, packed ? 1U : 0U}
        );
    }

    /// @brief Two-bit words the arena still takes, from the next word-aligned byte.
    [[nodiscard]] size_t arena_room_words() const noexcept {
        auto const aligned = (arena_used_ + 3) & ~size_t{3};
        return aligned < bounds_.staging ? (bounds_.staging - aligned) / sizeof(uint32_t) : 0;
    }

    /// @brief Makes @p count two-bit words readable by the kernel and returns where they are.
    [[nodiscard]] Result<char const*>
    place_words(uint32_t const* words, size_t count, char const* pinned_base, size_t pinned_size) {
        auto const* const source = reinterpret_cast<char const*>(words);
        auto const bytes = count * sizeof(uint32_t);
        account(source, bytes, pinned_base, pinned_size);
        if (direct_) {
            held_bytes_ += bytes;
            return source;
        }
        arena_used_ = (arena_used_ + 3) & ~size_t{3};
        auto* const target = arena() + arena_used_;
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                copy_stream_, cuda::std::span{source, bytes}, device_span<char>{target, bytes}
            )
        );
        arena_used_ += bytes;
        ++transfers_;
        return static_cast<char const*>(target);
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
                copy_stream_,
                cuda::std::span{source, size},
                device_span<char>{arena() + arena_used_, size}
            )
        );
        describe(arena() + arena_used_, windows, genome);
        arena_used_ += size;
        ++transfers_;
        return Ok();
    }

   public:
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
        // The sketch kernel runs ahead of lower-priority work sharing the device, such as
        // nvCOMP inflation: the copies that feed the next batch wait on it.
        queued_.record(stream_);
        kernel_stream_.wait(queued_);
        if (!direct_) {
            copies_done_.record(copy_stream_);
            kernel_stream_.wait(copies_done_);
        }
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                kernel_stream_,
                cuda::std::span{copied_chunks_.data(), copied_chunks_.size()},
                device_span<sequence_batch_chunk>{descriptors_.data(), copied_chunks_.size()}
            )
        );
        copied_consumed_.record(kernel_stream_);
        auto const grid = std::min(block_end_, max_grid_);
        if (grid != 0) {
            auto const resident = CUDDL_TRY(begin_resident(kernel_stream_));
            add_sequence_batch_kernel<BucketCount, Layout>
                <<<static_cast<uint32_t>(grid), 256, 0, kernel_stream_.get()>>>(
                    descriptors_.data(), copied_chunks_.size(), block_end_, K, rows_.data()
                );
            CUDDL_CUDA_TRY(cudaGetLastError());
            CUDDL_TRY(end_resident(resident, kernel_stream_));
        }
        kernel_done_.record(kernel_stream_);
        stream_.wait(kernel_done_);
        if (direct_ && !batch_files_.empty()) {
            // The launch above is what reads those buffers, so they wait on this batch's event.
            released_.push_back({cuda::event{stream_}, std::move(batch_files_)});
            batch_files_.clear();
            released_.back().first.record(stream_);
            while (released_.size() > held_batches) {
                released_.front().first.sync();
                released_.pop_front();
            }
        }
        if (!direct_) {
            // The next batch fills the other arena once the kernel that last read it is done.
            arena_done_[arena_index_].record(stream_);
            arena_index_ = (arena_index_ + 1) % stager_arenas;
            copy_stream_.wait(arena_done_[arena_index_]);
        }
        ++batches_;
        staged_chunks_.clear();
        arena_used_ = 0;
        held_bytes_ = 0;
        block_end_ = 0;
        return Ok();
    }

    /// @brief Waits until the next genome staged by copy can be held without blocking.
    ///
    /// The thread that stages copied genomes may call this outside any lock it shares with
    /// other staging threads, so waiting on the transfers does not hold them up.
    void wait_hold_slot() {
        if (direct_) return;
        auto const slot = held_count_ % held_.size();
        if (held_[slot]) held_consumed_[slot].sync();
    }

   private:
    [[nodiscard]] char* arena() noexcept {
        return arena_.data() + arena_index_ * arena_stride(bounds_.staging);
    }

    /// @brief Keeps one genome's bytes until the device has finished reading them.
    ///
    /// A copy is read by the transfer engine before the enqueue returns, so a ring of slots that
    /// a later genome releases is enough. Staged in place the *kernel* reads the bytes, and it
    /// runs at the next flush: releasing on the next genome would free a buffer the device is
    /// still reading.
    void hold(std::unique_ptr<fastx_sequence_file> holder) {
        if (direct_) {
            batch_files_.push_back(std::move(holder));
            return;
        }
        auto const slot = held_count_++ % held_.size();
        if (held_[slot]) {
            held_consumed_[slot].sync();
            held_[slot].reset();
        }
        held_consumed_[slot].record(copy_stream_);
        held_[slot] = std::move(holder);
    }

    cuda::stream_ref stream_;
    staging_plan bounds_;
    reference_build_statistics* statistics_;
    bool direct_ = false;
    cuda::device_buffer<uint32_t> rows_;
    cuda::stream copy_stream_;
    cuda::stream kernel_stream_;
    cuda::device_buffer<char> arena_;
    cuda::event copies_done_{stream_};
    cuda::event queued_{stream_};
    cuda::event kernel_done_{stream_};
    std::array<cuda::event, stager_arenas> arena_done_ =
        [&]<size_t... I>(std::index_sequence<I...>) {
            return std::array<cuda::event, stager_arenas>{((void)I, cuda::event{stream_})...};
        }(std::make_index_sequence<stager_arenas>{});
    size_t arena_index_ = 0;
    cuda::device_buffer<sequence_batch_chunk> descriptors_;
    /// Batches whose kernels may still be reading a held buffer.
    static constexpr size_t held_batches = 1;
    std::deque<std::pair<cuda::event, std::vector<std::unique_ptr<fastx_sequence_file>>>> released_;
    std::vector<std::unique_ptr<fastx_sequence_file>> batch_files_;
    std::vector<std::optional<std::unique_ptr<fastx_sequence_file>>> held_;
    std::vector<cuda::event> held_consumed_;
    cuda::event copied_consumed_{stream_};
    std::vector<sequence_batch_chunk> staged_chunks_, copied_chunks_;
    std::vector<resident_event_span> resident_spans_;
    size_t const sm_, max_grid_;
    size_t arena_used_ = 0, block_end_ = 0, batches_ = 0, transfers_ = 0, held_count_ = 0;
    size_t held_bytes_ = 0;
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

/// @brief Copies the winner scores of device store rows into host score rows.
///
/// Scores are extracted on the device, so only two bytes per bucket cross the bus. @p scores
/// must stay alive until @p stream reaches this point.
template <size_t BucketCount>
[[nodiscard]] Result<void> download_store(
    device_span<uint32_t const> store,
    std::span<uint16_t> scores,
    cuda::stream_ref stream
) {
    auto const genomes = scores.size() / BucketCount;
    if (genomes == 0) return Ok();
    auto device_scores =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), scores.size(), cuda::no_init);
    batch_scores_kernel<BucketCount>
        <<<static_cast<uint32_t>(genomes), block_size, 0, stream.get()>>>(
            store.data(), static_cast<uint32_t>(genomes), device_scores.data()
        );
    CUDDL_CUDA_TRY(cudaGetLastError());
    CUDDL_CUDA_TRY(cuda::copy_bytes(stream, device_scores, scores));
    return Ok();
}

// Reusable sequence storage handed to loader threads. A lease returns to the pool when the
// parsed file holding it dies, which is after the caller enqueued every copy that reads it.
// Releasing records that point on the stream; the next loader to take the slot waits for it
// there, so a buffer is never rewritten under a DMA and the build never has to drain.
class sequence_buffer_pool {
    struct slot;

   public:
    /// @p limit bounds one buffer; larger genomes stay on the loader's own growing buffer.
    sequence_buffer_pool(cuda::stream_ref stream, size_t buffers, size_t limit, bool page_locked)
        : stream_(stream), limit_(limit), page_locked_(page_locked) {
        slots_.reserve(buffers);
    }

    /// @brief Number of buffers the pool may grow to.
    void set_capacity(size_t buffers) noexcept {
        capacity_ = std::max<size_t>(1, buffers);
    }

    /// @brief Buffers allocated so far. Read once the loaders have stopped.
    [[nodiscard]] size_t buffers() const noexcept {
        return slots_.size();
    }

    /// @brief Returns reusable bytes, or a null target when the pool cannot serve.
    [[nodiscard]] decompression_target acquire(size_t bytes) {
        if (bytes == 0 || bytes > limit_) return {};
        slot* owner = nullptr;
        {
            std::lock_guard lock(mutex_);
            owner = claim(bytes);
            if (owner == nullptr) return {};
        }
        // Wait on the loader thread rather than the build loop: the slot is already reserved,
        // so this blocks only the genome that needs it. A failing wait throws out of the loader,
        // which reports it through the pool's error slot like any other load failure.
        owner->consumed.sync();
        if (owner->size < bytes) {
            auto grown = size_t{1} << 16;
            while (grown < bytes) {
                grown *= 2;
            }
            owner->resize(stream_, grown, page_locked_);
        }
        return lease(owner);
    }

   private:
    /// @brief Reserves the oldest released slot, or returns null when the pool is full.
    [[nodiscard]] slot* claim(size_t bytes) {
        if (free_head_ != nullptr) {
            auto* owner = free_head_;
            free_head_ = owner->next_free;
            return owner;
        }
        if (slots_.size() >= capacity_) return nullptr;
        slots_.push_back(std::make_unique<slot>(stream_, bytes, page_locked_));
        return slots_.back().get();
    }

   private:
    struct slot {
        std::optional<cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>>
            pinned;
        std::unique_ptr<char[]> heap;
        size_t size = 0;
        cuda::event consumed;
        slot* next_free = nullptr;

        slot(cuda::stream_ref stream, size_t bytes, bool page_locked) : consumed(stream) {
            resize(stream, bytes, page_locked);
        }

        void resize(cuda::stream_ref stream, size_t bytes, bool page_locked) {
            if (page_locked) {
                pinned.emplace(stream, cuda::pinned_default_memory_pool(), bytes, cuda::no_init);
            } else {
                heap = std::make_unique_for_overwrite<char[]>(bytes);
            }
            size = bytes;
        }

        char* data() {
            return heap ? heap.get() : pinned->data();
        }
    };

    [[nodiscard]] decompression_target lease(slot* owner) {
        auto* pool = this;
        return {owner->data(), owner->size, std::shared_ptr<void>(owner, [pool, owner](void*) {
                    std::lock_guard lock(pool->mutex_);
                    // Record before publishing the slot: a loader that takes it must see the
                    // event of every copy the previous lease fed.
                    owner->consumed.record(pool->stream_);
                    owner->next_free = nullptr;
                    if (pool->free_head_ == nullptr) {
                        pool->free_head_ = owner;
                    } else {
                        pool->free_tail_->next_free = owner;
                    }
                    pool->free_tail_ = owner;
                })};
    }

    cuda::stream_ref stream_;
    size_t limit_;
    bool page_locked_;
    size_t capacity_{1};
    std::vector<std::unique_ptr<slot>> slots_;
    slot* free_head_ = nullptr;
    slot* free_tail_ = nullptr;
    std::mutex mutex_;
};

/// @brief `decompression_source` trampoline for `sequence_buffer_pool`.
[[nodiscard]] inline decompression_target acquire_sequence_target(void* context, size_t bytes) {
    return static_cast<sequence_buffer_pool*>(context)->acquire(bytes);
}

/// @brief Loader state a build fed by paths keeps for its whole run.
///
/// Reused decompression buffers avoid repeated allocations, and can be page-locked for DMA.
/// The pool hands out load results a few files ahead of the consumer, which takes them in order.
/// A window of one worker-worth leaves the consumer waiting on a straggler while every other
/// loader sits idle.
class path_loaders {
   public:
    /// @param stream Stream owning the loader buffers.
    /// @param paths Genome files in reference-ID order.
    /// @param parser_workers Loader ceiling; the build clamps it to inputs and hardware.
    /// @param page_locked gives the loaders page-locked buffers to decompress into.
    path_loaders(
        cuda::stream_ref stream,
        std::span<std::filesystem::path const> paths,
        unsigned parser_workers,
        bool page_locked
    )
        : workers_(worker_count(paths.size(), parser_workers)),
          buffers_(stream, workers_, size_t{32} << 20, page_locked),
          page_locked_(page_locked) {
        // One buffer per in-flight file plus the stager's hold ring, plus headroom: a worker
        // asks for its next file while every loaded file still holds a lease. A pool smaller
        // than the window makes the remaining files allocate their own buffers.
        auto const window = workers_ * 4;
        buffers_.set_capacity(window + stager_hold_slots + 2);
        source_ = {acquire_sequence_target, &buffers_};
        if (workers_ > 1) pool_.emplace(paths, workers_, source_, window);
    }

    /// @brief Loaders this build runs.
    ///
    /// The pool allocates buffers lazily, independently of the number of loader threads.
    [[nodiscard]] size_t workers() const noexcept {
        return workers_;
    }

    /// @brief Page-locked buffers the loaders held, for the statistics.
    [[nodiscard]] size_t page_locked_buffers() const noexcept {
        return page_locked_ ? buffers_.buffers() : 0;
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

    /// @brief Loaders to run: at most one per input, at most the requested workers, at most the
    /// hardware, and never none.
    static size_t worker_count(size_t paths, unsigned parser_workers) noexcept {
        auto const hardware = std::max(1U, std::thread::hardware_concurrency());
        auto const loaders = static_cast<size_t>(std::min(parser_workers, hardware));
        return std::max<size_t>(1, std::min(paths, loaders));
    }

   private:
    size_t workers_ = 1;
    sequence_buffer_pool buffers_;
    bool page_locked_ = false;
    decompression_source source_{};
    std::optional<fastx_load_pool> pool_;
};

/// @brief Runs the staging loop over a collection, filling one group at a time.
///
/// @p fill receives the stager, a group's first genome and its size, and stages that whole group.
/// Genomes are numbered within their group, so a fill stages genome @c genome - @c base.
/// @p sink receives each finished group's device rows with its first genome and size, and must
/// enqueue every read of them on @p stream: the next group reuses the rows.
template <
    uint32_t K,
    size_t BucketCount,
    typename Layout = default_register_layout,
    typename Fill,
    typename Sink>
[[nodiscard]] Result<void> stage_groups(
    size_t genomes,
    uint64_t staged_ceiling,
    std::optional<size_t> staging_bytes,
    cuda::stream_ref stream,
    reference_build_statistics* statistics,
    Fill&& fill,
    Sink&& sink,
    bool in_place = false
) {
    auto const plan =
        CUDDL_TRY((plan_staging<K, BucketCount>(genomes, staged_ceiling, staging_bytes, stream)));
    database_stager<K, BucketCount, Layout> stager(stream, plan, statistics, in_place);
    size_t base = 0;
    while (base < genomes) {
        auto const count = std::min(plan.group, genomes - base);
        CUDDL_TRY(stager.begin_group(count));
        CUDDL_TRY(fill(stager, base, count));
        auto const rows = CUDDL_TRY(stager.end_group(count));
        CUDDL_TRY(sink(rows, base, count));
        base += count;
    }
    return stager.finish();
}

#if CUDDL_HAS_NVCOMP
/// @brief Loads @p ids through host loaders and stages them, genome numbers relative to @p base.
/// Starts at most one loader per input file.
template <uint32_t K, size_t BucketCount, typename Layout>
[[nodiscard]] Result<void> stage_host_loaded(
    database_stager<K, BucketCount, Layout>& stager,
    std::span<std::filesystem::path const> paths,
    std::span<size_t const> ids,
    size_t base,
    size_t workers,
    sequence_buffer_pool& buffers,
    bool page_locked
) {
    if (ids.empty()) return Ok();
    workers = std::min(workers, ids.size());
    std::vector<std::filesystem::path> subset;
    subset.reserve(ids.size());
    for (auto const id : ids) subset.push_back(paths[id]);
    fastx_load_pool pool(subset, workers, {acquire_sequence_target, &buffers}, workers * 4);
    for (size_t index = 0; index < ids.size(); ++index) {
        auto sequence = CUDDL_TRY(pool.take(index));
        auto const& extents = sequence->extents;
        auto const* pinned_base = page_locked ? sequence->decompressed_target : nullptr;
        auto const pinned_size = page_locked ? sequence->decompressed_size : 0;
        CUDDL_TRY(stager.add_genome(
            ids[index] - base, extents, pinned_base, pinned_size, std::move(sequence)
        ));
    }
    return Ok();
}

/// @brief Inflates on CPU workers, then normalises FASTA and sketches it in device memory.
template <uint32_t K, size_t BucketCount, typename Layout, typename Sink>
[[nodiscard]] Result<void> stage_paths_normalising(
    std::span<std::filesystem::path const> paths,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    unsigned parser_workers,
    transfer_mode transfer,
    reference_build_statistics* statistics,
    Sink&& sink
) {
    auto const workers = path_loaders::worker_count(paths.size(), parser_workers);
    auto const in_place = stages_in_place(transfer, stream.device());
    auto const page_locked = pages_locked(transfer, stream.device());
    auto const window = workers * 8;
    auto const batch_limit = std::min(workers * 4, device_fasta_pipeline::max_files);
    sequence_buffer_pool buffers(stream, workers, size_t{32} << 20, page_locked);
    buffers.set_capacity(
        window + batch_limit * device_fasta_pipeline::lane_count + stager_hold_slots + 2
    );
    std::optional<device_fasta_pipeline> normaliser;
    struct loaded_fasta {
        size_t id;
        std::unique_ptr<fastx_sequence_file> sequence;
    };
    std::array<std::vector<loaded_fasta>, device_fasta_pipeline::lane_count> batches;
    for (auto& batch : batches) {
        batch.reserve(batch_limit);
    }
    auto fill = [&](database_stager<K, BucketCount, Layout>& stager,
                    size_t base,
                    size_t count) -> Result<void> {
        if (!normaliser) {
            auto const available = CUDDL_TRY(available_device_bytes(stream));
            auto const budget = (available - available / 5) / device_fasta_pipeline::lane_count;
            auto const wanted = staging_bytes.value_or(default_arena_ceiling);
            auto const overhead = device_fasta_pipeline::lane_overhead(wanted);
            auto const capacity = std::min(wanted, budget > overhead ? budget - overhead : 0);
            if (capacity < 2) return Err(Error::resource("no room for FASTA normalisation"));
            CUDDL_TRY(cuda_try([&] { normaliser.emplace(stream, capacity, 0, workers); }));
        }
        fastx_load_pool pool(
            paths.subspan(base, count),
            workers,
            {acquire_sequence_target, &buffers},
            window,
            true,
            true
        );
        std::vector<decoded_fasta> sources;
        sources.reserve(batch_limit);
        std::vector<inflated_genome> genomes;
        std::vector<size_t> fallback;
        auto stage_host = [&](size_t id,
                              std::unique_ptr<fastx_sequence_file> sequence) -> Result<void> {
            if (!sequence->input.empty()) {
                CUDDL_TRY(parse_fastx_sequence_file(*sequence, paths[base + id].string()));
            }
            auto const& extents = sequence->extents;
            auto const* pinned_base = page_locked ? sequence->decompressed_target : nullptr;
            auto const pinned_size = page_locked ? sequence->decompressed_size : 0;
            return stager.add_genome(id, extents, pinned_base, pinned_size, std::move(sequence));
        };
        auto finish_lane = [&](size_t lane) -> Result<void> {
            auto& batch = batches[lane];
            if (batch.empty()) return Ok();
            genomes.clear();
            fallback.clear();
            CUDDL_TRY(normaliser->finish(lane, {}, genomes, fallback));
            for (auto const& genome : genomes) {
                CUDDL_TRY(stager.add_resident(genome.id, genome.bases, genome.size));
            }
            size_t rejected = 0;
            for (auto& loaded : batch) {
                if (rejected < fallback.size() && loaded.id == fallback[rejected]) {
                    CUDDL_TRY(stage_host(loaded.id, std::move(loaded.sequence)));
                    ++rejected;
                }
            }
            CUDDL_TRY(stager.flush());
            CUDDL_TRY(normaliser->release(lane, stream));
            batch.clear();
            return Ok();
        };
        size_t next = 0, lane = 0;
        std::optional<loaded_fasta> pending;
        while (next < count || pending) {
            CUDDL_TRY(finish_lane(lane));
            sources.clear();
            size_t bytes = 0;
            auto& batch = batches[lane];
            while (next < count || pending) {
                if (!pending) {
                    auto loaded = CUDDL_TRY(pool.take_ready());
                    pending.emplace(loaded_fasta{loaded.first, std::move(loaded.second)});
                    ++next;
                }
                auto const input = pending->sequence->input;
                if (input.empty() || input.size() >= normaliser->slot_capacity()) {
                    CUDDL_TRY(stage_host(pending->id, std::move(pending->sequence)));
                    pending.reset();
                    continue;
                }
                if (batch.size() == batch_limit ||
                    input.size() >= normaliser->slot_capacity() - bytes)
                    break;
                bytes += input.size() + 1;
                sources.push_back({pending->id, input});
                batch.push_back(std::move(*pending));
                pending.reset();
            }
            if (!batch.empty()) CUDDL_TRY(normaliser->submit_decoded(lane, sources));
            lane = (lane + 1) % batches.size();
        }
        for (size_t offset = 0; offset < batches.size(); ++offset) {
            CUDDL_TRY(finish_lane((lane + offset) % batches.size()));
        }
        return Ok();
    };
    // Capture errors before draining GPU work so pending input owners remain alive.
    std::exception_ptr failure;
    auto staged = [&]() -> Result<void> {
        try {
            return stage_groups<K, BucketCount, Layout>(
                paths.size(),
                default_arena_ceiling,
                staging_bytes,
                stream,
                statistics,
                fill,
                sink,
                in_place
            );
        } catch (...) {
            failure = std::current_exception();
            return Ok();
        }
    }();
    if (normaliser) CUDDL_TRY(normaliser->wait_lanes(stream));
    if (failure) std::rethrow_exception(failure);
    if (!staged) return staged;
    if (statistics != nullptr) {
        statistics->workers = static_cast<unsigned>(workers);
        statistics->in_place = in_place;
        statistics->pinned_buffers = page_locked ? buffers.buffers() : 0;
    }
    return Ok();
}

/// @brief @ref stage_paths with gzip FASTA inflated and compacted on the device.
///
/// The host only reads compressed bytes for the files the device takes. Everything else, and any
/// file the device hands back, loads through host loaders exactly as the host path would load it,
/// so results and errors match it.
template <uint32_t K, size_t BucketCount, typename Layout, typename Sink>
[[nodiscard]] Result<void> stage_paths_inflating(
    std::span<std::filesystem::path const> paths,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    unsigned parser_workers,
    transfer_mode transfer,
    reference_build_statistics* statistics,
    Sink&& sink
) {
    auto const workers = path_loaders::worker_count(paths.size(), parser_workers);
    bool const in_place = stages_in_place(transfer, stream.device());
    bool const page_locked = pages_locked(transfer, stream.device());
    std::vector<gzip_file_probe> probes(paths.size());
    parallel_for(paths.size(), workers, [&](size_t id) {
        probes[id] = probe_gzip_file(paths[id]);
    });
    std::optional<device_fasta_pipeline> inflater;
    sequence_buffer_pool host_buffers(stream, workers, size_t{32} << 20, page_locked);
    host_buffers.set_capacity(workers * 4 + stager_hold_slots + 2);
    auto const fits = [&](gzip_file_probe const& probe) {
        return probe.device && probe.isize + size_t{1} <= inflater->slot_capacity() &&
               probe.compressed <= inflater->compressed_capacity();
    };
    // Sized once the stager holds its rows and arena, from what is left and what the inputs need.
    auto const make_inflater = [&]() -> Result<void> {
        size_t slots = 0, compressed = 0;
        for (auto const& probe : probes) {
            if (!probe.device) continue;
            slots += probe.isize + size_t{1};
            compressed += probe.compressed;
        }
        auto const available = CUDDL_TRY(available_device_bytes(stream));
        auto const budget = (available - available / 5) / device_fasta_pipeline::lane_count;
        auto const overhead = device_fasta_pipeline::lane_overhead(budget);
        auto const usable = budget > overhead ? budget - overhead : 0;
        auto const compressed_capacity =
            slots + compressed == 0
                ? size_t{0}
                : std::min(
                      compressed,
                      static_cast<size_t>(
                          static_cast<long double>(usable) * compressed / (slots + compressed)
                      )
                  );
        auto const room = std::min(
            usable / 2,
            budget > compressed_capacity + overhead ? budget - compressed_capacity - overhead : 0
        );
        return cuda_try([&] {
            inflater.emplace(stream, std::min(room, slots), compressed_capacity, workers);
        });
    };
    CUDDL_TRY((stage_groups<K, BucketCount, Layout>(
        paths.size(),
        default_arena_ceiling,
        staging_bytes,
        stream,
        statistics,
        [&](database_stager<K, BucketCount, Layout>& stager, size_t base, size_t count)
            -> Result<void> {
            if (!inflater) CUDDL_TRY(make_inflater());
            // Host loaders take `order` from the front: files the device cannot hold, then the
            // device candidates from the back. nvCOMP takes candidates from the front and caps
            // the loaders before each batch, so whichever side is faster on this host takes more.
            // Candidates run smallest first: one thread block inflates a whole file, so large
            // files suit the CPU, and a batch of similar sizes has no straggler holding it open.
            std::vector<size_t> device_ids, order, handed_back;
            for (size_t id = base; id < base + count; ++id) {
                (fits(probes[id]) ? device_ids : order).push_back(id);
            }
            std::ranges::stable_sort(device_ids, {}, [&](size_t id) { return probes[id].compressed; });
            order.insert(order.end(), device_ids.rbegin(), device_ids.rend());
            std::vector<std::filesystem::path> host_paths;
            host_paths.reserve(order.size());
            for (auto id : order) host_paths.push_back(paths[id]);
            fastx_load_pool host_pool(
                host_paths,
                std::min(workers, host_paths.size()),
                decompression_source{acquire_sequence_target, &host_buffers},
                workers * 4,
                false,
                true,
                K
            );
            // The host path stages on its own thread: it waits on copies over the bus, which
            // must not hold up the nvCOMP lanes this thread drives. The stager is shared.
            std::mutex staging;
            auto host_staged = std::async(std::launch::async, [&]() -> Result<void> {
                CUDDL_CUDA_TRY(cudaSetDevice(stream.device().get()));
                while (true) {
                    auto loaded = CUDDL_TRY(host_pool.take_ready_until_limit());
                    if (!loaded) return Ok();
                    auto& [index, sequence] = *loaded;
                    auto const* pinned_base = page_locked ? sequence->decompressed_target : nullptr;
                    auto const pinned_size = page_locked ? sequence->decompressed_size : 0;
                    auto const genome = order[index] - base;
                    stager.wait_hold_slot();
                    std::lock_guard lock(staging);
                    if (sequence->words != nullptr) {
                        CUDDL_TRY(stager.add_packed_genome(
                            genome, std::move(sequence), pinned_base, pinned_size
                        ));
                        continue;
                    }
                    auto const& extents = sequence->extents;
                    CUDDL_TRY(stager.add_genome(
                        genome, extents, pinned_base, pinned_size, std::move(sequence)
                    ));
                }
            });
            // On an early return, stop the loaders at what they claimed so the host thread ends.
            struct stop_loaders {
                fastx_load_pool& pool;
                ~stop_loaders() {
                    pool.cap(0);
                }
            } stop{host_pool};
            std::vector<inflated_genome> genomes;
            std::array<bool, device_fasta_pipeline::lane_count> busy{};
            size_t next = 0, lane = 0;
            bool device_open = !device_ids.empty();
            // Cycle through lanes, submitting a batch before consuming the oldest one.
            while (device_open || std::ranges::any_of(busy, [](bool value) { return value; })) {
                if (device_open && !busy[lane]) {
                    auto const remaining = std::span{device_ids}.subspan(next);
                    auto const wanted = inflater->batch_files(remaining, probes);
                    auto const limit = host_pool.cap(order.size() - next - wanted);
                    auto const files = order.size() - next - limit;
                    device_open = files == wanted && next + files < device_ids.size();
                    if (files != 0) {
                        next += CUDDL_TRY(
                            inflater->submit(lane, remaining.first(files), paths, probes)
                        );
                        busy[lane] = true;
                    }
                }
                auto const other = (lane + 1) % busy.size();
                if (busy[other]) {
                    genomes.clear();
                    CUDDL_TRY(inflater->finish(other, probes, genomes, handed_back));
                    std::lock_guard lock(staging);
                    for (auto const& genome : genomes) {
                        CUDDL_TRY(stager.add_resident(genome.id - base, genome.bases, genome.size));
                    }
                    CUDDL_TRY(stager.flush());
                    CUDDL_TRY(inflater->release(other, stream));
                    busy[other] = false;
                }
                lane = other;
            }
            CUDDL_TRY(host_staged.get());
            // ponytail: one fallback pool per group; pipeline it if fallback-heavy inputs dominate.
            return stage_host_loaded<K, BucketCount, Layout>(
                stager, paths, handed_back, base, workers, host_buffers, page_locked
            );
        },
        sink,
        in_place
    )));
    if (statistics != nullptr) {
        statistics->workers = static_cast<unsigned>(workers);
        statistics->in_place = in_place;
        statistics->pinned_buffers = page_locked ? host_buffers.buffers() : 0;
    }
    return Ok();
}

#endif

/// @brief Stages a path collection, handing each group's device rows to @p sink.
///
/// @p parser_workers is a value, not a sentinel: the build clamps it to what the inputs and the
/// hardware allow, so asking for the default is asking for the default.
///
/// The arena ceiling is the decompressed size of every input: a bound from compressed sizes alone
/// overestimates a corpus several times over, and an arena past what the corpus can hold is
/// memory the device never needs.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout, typename Sink>
[[nodiscard]] Result<void> stage_paths(
    std::span<std::filesystem::path const> paths,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    unsigned parser_workers,
    transfer_mode transfer,
    decompression_backend decompression,
    reference_build_statistics* statistics,
    Sink&& sink
) {
    if (parser_workers == 0) {
        return Err(Error::invalid_argument("a path build needs at least one loader"));
    }
#if !CUDDL_HAS_NVCOMP
    if (decompression == decompression_backend::gpu ||
        decompression == decompression_backend::coherent) {
        return Err(
            Error::invalid_argument("requested decompression backend requires nvCOMP support")
        );
    }
    decompression = decompression_backend::cpu;
#endif
    if (transfer == transfer_mode::in_place && !device_reads_pageable_memory(stream.device())) {
        return Err(
            Error::invalid_argument(
                "in-place staging needs a device that reads pageable host memory"
            )
        );
    }
    bool const coherent = device_reads_pageable_memory(stream.device());
    if (decompression == decompression_backend::coherent && !coherent) {
        return Err(
            Error::invalid_argument("coherent decompression needs coherent pageable host memory")
        );
    }
    if (paths.empty()) return Ok();
    if (decompression == decompression_backend::automatic) {
        decompression = !coherent ? decompression_backend::gpu
                        : path_loaders::worker_count(paths.size(), parser_workers) > 1
                            ? decompression_backend::coherent
                            : decompression_backend::cpu;
    }
#if CUDDL_HAS_NVCOMP
    if (decompression == decompression_backend::coherent) {
        return stage_paths_normalising<K, BucketCount, Layout>(
            paths,
            stream,
            staging_bytes,
            parser_workers,
            transfer,
            statistics,
            std::forward<Sink>(sink)
        );
    }
    if (decompression != decompression_backend::cpu) {
        return stage_paths_inflating<K, BucketCount, Layout>(
            paths,
            stream,
            staging_bytes,
            parser_workers,
            transfer,
            statistics,
            std::forward<Sink>(sink)
        );
    }
#endif
    auto const in_place = stages_in_place(transfer, stream.device());
    auto const page_locked = pages_locked(transfer, stream.device());
    path_loaders loaders(stream, paths, parser_workers, page_locked);
    // The corpus size only bounds a default arena, and only below its ceiling, so probing stops
    // there: a corpus of any size then costs at most a few dozen file opens here.
    uint64_t staged_ceiling = 0;
    for (auto const& path : paths) {
        if (staging_bytes.has_value() || staged_ceiling >= default_arena_ceiling) break;
        auto const decompressed = gzip_decompressed_size(path.string());
        if (decompressed != 0) {
            staged_ceiling += decompressed;
            continue;
        }
        std::error_code error;
        auto const size = std::filesystem::file_size(path, error);
        if (!error) staged_ceiling += size;
    }
    CUDDL_TRY((stage_groups<K, BucketCount, Layout>(
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
                auto const* const pinned_base =
                    page_locked ? sequence->decompressed_target : nullptr;
                auto const pinned_size = page_locked ? sequence->decompressed_size : 0;
                auto const& extents = sequence->extents;
                CUDDL_TRY(stager.add_genome(
                    id - base, extents, pinned_base, pinned_size, std::move(sequence)
                ));
            }
            return Ok();
        },
        sink,
        in_place
    )));
    if (statistics != nullptr) {
        statistics->pinned_buffers = loaders.page_locked_buffers();
        statistics->workers = static_cast<unsigned>(loaders.workers());
        statistics->in_place = in_place;
    }
    loaders.reset();
    return Ok();
}

/// @brief Stages bases the caller already holds, handing each group's device rows to @p sink.
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout, typename Sink>
[[nodiscard]] Result<void> stage_sequences(
    std::span<sequence_genome const> genomes,
    cuda::stream_ref stream,
    std::optional<size_t> staging_bytes,
    reference_build_statistics* statistics,
    Sink&& sink
) {
    if (genomes.empty()) return Ok();
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
        [&](database_stager<K, BucketCount, Layout>& stager,
            size_t base,
            size_t count) -> Result<void> {
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
        },
        sink
    );
}

}  // namespace cuddl::detail
