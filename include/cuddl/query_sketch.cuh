#pragma once

#include <cuda/buffer>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <filesystem>
#include <span>
#include <utility>
#include <vector>

#include <cuddl/batch.cuh>
#include <cuddl/detail/database_staging.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/reference_database.cuh>
#include <cuddl/reference_database_file.cuh>

namespace cuddl {

/**
 * @brief Query sketches held on the device, in the shape a search consumes them.
 *
 * A query is the winner score of every register, one `uint16_t` row per query genome, which is
 * what `reference_database::search_async` and `search_batch_async` take. Sketching queries
 * through @ref reference_database_file would retain host score rows and labels that queries
 * do not need. This keeps the sketches on the device in the shape a
 * search wants, and nothing else.
 *
 * The batch is move-only and owns its device allocation. Query IDs follow the order of the paths,
 * or of the genomes handed to @ref sketch_from_sequences. The allocating stream must outlive the
 * batch, and `@ref scores` stays valid until the device work on that stream completes.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class query_sketch_batch {
   public:
    using score_type = uint16_t;

    query_sketch_batch(query_sketch_batch const&) = delete;
    query_sketch_batch& operator=(query_sketch_batch const&) = delete;

    // Explicit host bodies prevent NVCC from inferring device-side buffer operations.
    __host__ ~query_sketch_batch() {}  // NOLINT(modernize-use-equals-default)

    query_sketch_batch(query_sketch_batch&& other) noexcept
        : scores_(std::move(other.scores_)),
          query_count_(std::exchange(other.query_count_, 0U)),
          compatibility_(other.compatibility_) {}

    /// @brief Move-assigns the batch, leaving the source empty.
    query_sketch_batch& operator=(query_sketch_batch&& other) noexcept {
        if (this != &other) {
            scores_ = std::move(other.scores_);
            query_count_ = std::exchange(other.query_count_, 0U);
            compatibility_ = other.compatibility_;
        }
        return *this;
    }

    /**
     * @brief Sketches one query per plain or gzip/BGZF FASTA/FASTQ file on the GPU.
     *
     * K-mers never cross record boundaries or ambiguous bases, exactly as a reference build
     * treats them, and FASTQ qualities are validated and ignored. A file with no usable record
     * keeps its query ID and scores zero everywhere.
     */
    [[nodiscard]] static Result<query_sketch_batch> sketch(
        std::span<std::filesystem::path const> paths,
        cuda::stream_ref stream,
        path_build_options options = {}
    ) try {
        return [&]() -> Result<query_sketch_batch> {
            query_sketch_batch batch(stream, static_cast<uint32_t>(paths.size()));
            CUDDL_TRY(batch.set_blacklist(options.blacklist, stream));
            CUDDL_TRY((detail::stage_paths<K, BucketCount, Layout>(
                paths,
                stream,
                options.staging_bytes,
                options.parser_workers,
                options.transfer,
                options.decompression,
                options.statistics,
                batch.reduce_rows(stream),
                options.blacklist ? options.blacklist->get().view() : cuda::std::nullopt
            )));
            return batch;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Sketches one query per genome of bases the caller already holds.
    ///
    /// Each genome supplies its records as spans of consecutive bases with the line breaks
    /// already removed, the shape a parsed FASTX record has, and the caller keeps every byte
    /// alive until this returns.
    [[nodiscard]] static Result<query_sketch_batch> sketch_from_sequences(
        std::span<sequence_genome const> genomes,
        cuda::stream_ref stream,
        sequence_build_options options = {}
    ) try {
        return [&]() -> Result<query_sketch_batch> {
            query_sketch_batch batch(stream, static_cast<uint32_t>(genomes.size()));
            CUDDL_TRY(batch.set_blacklist(options.blacklist, stream));
            CUDDL_TRY((detail::stage_sequences<K, BucketCount, Layout>(
                genomes,
                stream,
                options.staging_bytes,
                options.statistics,
                batch.reduce_rows(stream),
                options.blacklist ? options.blacklist->get().view() : cuda::std::nullopt
            )));
            return batch;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Query rows in query-ID order, `BucketCount` scores each.
    [[nodiscard]] device_span<score_type const> scores() const noexcept {
        return {scores_.data(), static_cast<size_t>(query_count_) * BucketCount};
    }

    /// @brief Query count, in the order the paths or genomes were supplied.
    [[nodiscard]] uint32_t query_count() const noexcept {
        return query_count_;
    }

    /// @brief Compatibility a search must be given alongside @ref scores.
    [[nodiscard]] score_compatibility compatibility() const noexcept {
        return compatibility_;
    }

   private:
    query_sketch_batch(cuda::stream_ref stream, uint32_t queries)
        : scores_(
              cuda::make_device_buffer<score_type>(
                  stream,
                  stream.device(),
                  static_cast<size_t>(queries) * BucketCount,
                  cuda::no_init
              )
          ),
          query_count_(queries) {}

    /// @brief Staging sink that reduces each group's registers to query scores on the device.
    /// The staging's final sync covers the reduction.
    [[nodiscard]] auto reduce_rows(cuda::stream_ref stream) {
        return [this, stream](
                   device_span<uint32_t const> group, size_t base, size_t count
               ) -> Result<void> {
            return extract_scores_batch_async<BucketCount>(
                group,
                device_span<score_type>{scores_.data() + base * BucketCount, count * BucketCount},
                stream
            );
        };
    }

    cuda::device_buffer<score_type> scores_;
    uint32_t query_count_ = 0;
    score_compatibility compatibility_ = score_compatibility::current<K, BucketCount, Layout>();

    Result<void> set_blacklist(
        std::optional<std::reference_wrapper<device_blacklist const>> filter,
        cuda::stream_ref stream
    ) {
        if (!filter) return Ok();
        CUDDL_TRY(filter->get().validate(K, BucketCount, stream));
        compatibility_.blacklist_identity = filter->get().source().identity();
        compatibility_.blacklist_version = filter->get().source().version();
        return Ok();
    }
};

}  // namespace cuddl
