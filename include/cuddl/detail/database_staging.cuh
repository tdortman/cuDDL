#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

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
    /// @brief Sizing the producer resolves before the first genome.
    struct limits {
        size_t group = 0;       // genomes whose register rows stay resident at once
        size_t staging = 0;     // arena bytes
        size_t max_piece = 0;   // bytes one piece may occupy
        size_t max_pieces = 0;  // descriptors one batch may hold
    };

    /// Host memory the device still has to read must stay alive: the transfer engine's read of a
    /// pageable buffer is not ordered with the host writes that follow it. Sixteen slots take the
    /// tail off that wait; at four the tail reached hundreds of milliseconds while dozens of
    /// loaders fed one consumer.
    static constexpr size_t hold_slots = 16;

    static constexpr size_t row_words() noexcept {
        return BucketCount + 1;
    }

    /// @throws cuda::cuda_error or std::bad_alloc when the device buffers cannot be allocated.
    database_stager(
        cuda::stream_ref stream,
        size_t references,
        limits bounds,
        reference_build_statistics* statistics = nullptr
    )
        : stream_(stream),
          bounds_(bounds),
          statistics_(statistics),
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
            auto const room = bounds_.staging - arena_used_;
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
            // Copy the run in one transfer, then describe each record inside it.
            auto const* const source = records[record].begin;
            auto const run_bytes = static_cast<size_t>(records[run - 1].end - source);
            account(source, run_bytes, pinned_base, pinned_size);
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream_,
                    cuda::std::span{source, run_bytes},
                    device_span<char>{arena_.data() + arena_used_, run_bytes}
                )
            );
            for (size_t index = record; index < run; ++index) {
                auto const size = record_size(index);
                if (size < K) continue;  // no window to record
                describe(
                    arena_used_ + static_cast<size_t>(records[index].begin - source),
                    static_cast<uint32_t>(size - K + 1),
                    genome
                );
            }
            arena_used_ += run_bytes;
            ++transfers_;
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

    /// @brief Waits for the device, then unpacks the rows into the caller's storage.
    ///
    /// The store pads each genome with its saturation word, so the rows are unpacked from the
    /// single read `end_group` issued rather than copied once per genome.
    [[nodiscard]] Result<void> finish(std::span<uint32_t> rows, std::span<uint32_t> saturation) {
        CUDDL_CUDA_TRY(stream_.sync());
        for (size_t genome = 0; genome < saturation.size(); ++genome) {
            std::memcpy(
                rows.data() + genome * BucketCount,
                host_rows_.data() + genome * row_words(),
                BucketCount * sizeof(uint32_t)
            );
            saturation[genome] = host_rows_[genome * row_words() + BucketCount];
        }
        if (statistics_ != nullptr) {
            statistics_->staging_bytes = bounds_.staging;
            statistics_->batches = batches_;
            statistics_->transfers = transfers_;
        }
        return Ok();
    }

   private:
    /// @brief Records one chunk of the arena for the kernel.
    void describe(size_t offset, uint32_t windows, size_t genome) {
        auto const blocks = std::min(sm_ * 2, (static_cast<size_t>(windows) + 2047) / 2048);
        block_end_ += blocks;
        staged_chunks_.push_back({offset, block_end_, static_cast<uint32_t>(genome), windows});
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
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream_,
                cuda::std::span{source, size},
                device_span<char>{arena_.data() + arena_used_, size}
            )
        );
        describe(arena_used_, windows, genome);
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
                    arena_.data(),
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
    limits bounds_;
    reference_build_statistics* statistics_;
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

}  // namespace cuddl::detail
