#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl {

/// @brief Construction parameters shared by compatible compact and packed rows.
struct score_compatibility {
    uint32_t kmer_length{};
    uint32_t bucket_count{};
    uint32_t indexed_bucket_count{};
    uint32_t score_encoder_identity{};
    uint16_t exponent_bits{};
    uint16_t mantissa_bits{};
    uint32_t hash_identity{};
    uint64_t hash_seed{};
    uint32_t canonicalisation_policy{};
    uint64_t blacklist_identity{};
    uint32_t blacklist_version{};
    uint16_t key_mask{};

    /// @brief Metadata for score rows produced by the current cuDDL construction path.
    template <uint32_t K, size_t BucketCount>
    [[nodiscard]] static constexpr score_compatibility current() noexcept {
        static_assert(BucketCount <= std::numeric_limits<uint32_t>::max());
        return {
            .kmer_length = K,
            .bucket_count = static_cast<uint32_t>(BucketCount),
            .indexed_bucket_count = static_cast<uint32_t>(BucketCount),
            .score_encoder_identity = 1U,
            .exponent_bits = static_cast<uint16_t>(16U - detail::mantissa_bits),
            .mantissa_bits = static_cast<uint16_t>(detail::mantissa_bits),
            .hash_identity = 1U,
            .hash_seed = detail::seed,
            .canonicalisation_policy = 1U,
            .blacklist_identity = 0U,
            .blacklist_version = 0U,
            .key_mask = std::numeric_limits<uint16_t>::max(),
        };
    }

    friend bool operator==(score_compatibility const&, score_compatibility const&) = default;
};

/// @brief Recorded metadata for an immutable reference database.
struct reference_database_metadata {
    score_compatibility compatibility{};
    uint32_t reference_count{};

    friend bool operator==(reference_database_metadata const&, reference_database_metadata const&) =
        default;
};

/// @brief Stable reference ID and exact query-relative pairwise summary.
struct reference_search_result {
    uint32_t reference_id{};
    pairwise_summary summary{};

    friend bool operator==(reference_search_result const&, reference_search_result const&) =
        default;
};

/// @brief Stable query/reference IDs and their exact query-relative pairwise summary.
struct batch_search_result {
    uint32_t query_id{};
    uint32_t reference_id{};
    pairwise_summary summary{};

    friend bool operator==(batch_search_result const&, batch_search_result const&) = default;
};

/// @brief Caller-owned storage requirements for one bounded query tile.
struct batch_search_requirements {
    uint32_t maximum_pair_count{};
    size_t counter_bytes{};
    size_t candidate_bytes{};
    size_t temporary_bytes{};
    size_t workspace_bytes{};
    size_t result_bytes{};
    size_t match_count_bytes{};

    friend bool operator==(batch_search_requirements const&, batch_search_requirements const&) =
        default;
};

/// @brief Per-search candidate threshold for the indexed DDLIndex.
struct indexed_search_options {
    uint32_t minimum_matches = 5;
};

namespace detail {

template <uint32_t K, size_t BucketCount>
[[nodiscard]] inline Result<void> validate_score_compatibility(
    score_compatibility const& compatibility
) {
    if (compatibility.kmer_length != K) {
        return Err(Error::invalid_argument("k-mer length does not match the database type"));
    }
    if (compatibility.bucket_count != BucketCount) {
        return Err(Error::invalid_argument("bucket count does not match the database type"));
    }
    if (compatibility.score_encoder_identity == 0U || compatibility.exponent_bits == 0U ||
        compatibility.mantissa_bits == 0U ||
        static_cast<uint32_t>(compatibility.exponent_bits) + compatibility.mantissa_bits != 16U) {
        return Err(Error::invalid_argument("score encoding must identify a 16-bit format"));
    }
    if (compatibility.hash_identity == 0U) {
        return Err(Error::invalid_argument("hash identity must be specified"));
    }
    if (compatibility.canonicalisation_policy == 0U) {
        return Err(Error::invalid_argument("canonicalisation policy must be specified"));
    }
    return Ok();
}

template <uint32_t K, size_t BucketCount>
[[nodiscard]] inline Result<void> validate_non_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount>(compatibility);
        !validation) {
        return validation;
    }
    if (compatibility.indexed_bucket_count != BucketCount) {
        return Err(Error::invalid_argument("non-indexed builds require every bucket"));
    }
    if (compatibility.key_mask != std::numeric_limits<uint16_t>::max()) {
        return Err(Error::invalid_argument("non-indexed builds require an unmasked key"));
    }
    return Ok();
}

template <uint32_t K, size_t BucketCount>
[[nodiscard]] inline Result<void> validate_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount>(compatibility);
        !validation) {
        return validation;
    }
    auto const full_bucket_count = static_cast<uint32_t>(BucketCount);
    if (compatibility.indexed_bucket_count != full_bucket_count &&
        compatibility.indexed_bucket_count != full_bucket_count / 2U) {
        return Err(Error::invalid_argument("indexed builds require a full or half bucket count"));
    }
    if (compatibility.key_mask != std::numeric_limits<uint16_t>::max() &&
        compatibility.key_mask != 0x7fffU) {
        return Err(Error::invalid_argument("indexed builds require a 16-bit or 15-bit key mask"));
    }
    return Ok();
}

[[nodiscard]] inline constexpr uint64_t indexed_cell_count(
    score_compatibility const& compatibility
) noexcept {
    return static_cast<uint64_t>(compatibility.indexed_bucket_count) *
           (static_cast<uint64_t>(compatibility.key_mask) + 1U);
}
[[nodiscard]] inline constexpr uint64_t
indexed_posting_count(uint32_t reference_count, score_compatibility const& compatibility) noexcept {
    return static_cast<uint64_t>(reference_count) * compatibility.indexed_bucket_count;
}

[[nodiscard]] inline uintptr_t align_up(uintptr_t address, size_t alignment) noexcept {
    return (address + alignment - 1U) & ~(static_cast<uintptr_t>(alignment) - 1U);
}

}  // namespace detail

/**
 * @brief Non-owning, trivially copyable view of one selected reference-row backing.
 *
 * Inputs, database rows, workspace, and results must remain valid until the supplied stream
 * completes.
 */
template <uint32_t K, size_t BucketCount>
class reference_database_ref {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using score_type = uint16_t;
    using register_type = uint32_t;
    using result_type = reference_search_result;
    using batch_result_type = batch_search_result;

    __host__ __device__ constexpr reference_database_ref(
        device_span<score_type const> rows,
        reference_database_metadata metadata,
        device_span<uint32_t const> index_offsets = {},
        device_span<uint32_t const> index_postings = {},
        bool indexed = false
    ) noexcept
        : rows_(rows),
          metadata_(metadata),
          index_offsets_(index_offsets),
          index_postings_(index_postings),
          indexed_(indexed) {}

    __host__ __device__ constexpr reference_database_ref(
        device_span<register_type const> packed_rows,
        device_span<uint32_t const> saturation_states,
        reference_database_metadata metadata,
        device_span<uint32_t const> index_offsets = {},
        device_span<uint32_t const> index_postings = {},
        bool indexed = false
    ) noexcept
        : packed_rows_(packed_rows),
          saturation_states_(saturation_states),
          metadata_(metadata),
          index_offsets_(index_offsets),
          index_postings_(index_postings),
          packed_(true),
          indexed_(indexed) {}

    [[nodiscard]] __host__ __device__ constexpr device_span<score_type const>
    data() const noexcept {
        return rows_;
    }

    /// @brief Packed winner/count rows, or an empty span for compact databases.
    [[nodiscard]] __host__ __device__ constexpr device_span<register_type const>
    packed_data() const noexcept {
        return packed_rows_;
    }

    /// @brief Per-reference saturation states retained with packed rows.
    [[nodiscard]] __host__ __device__ constexpr device_span<uint32_t const>
    saturation_states() const noexcept {
        return saturation_states_;
    }

    [[nodiscard]] __host__ __device__ constexpr bool preserves_multiplicity() const noexcept {
        return packed_;
    }

    [[nodiscard]] constexpr reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] constexpr uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    [[nodiscard]] constexpr bool has_index() const noexcept {
        return indexed_;
    }

    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return static_cast<size_t>(reference_count) * BucketCount * sizeof(score_type);
    }

    [[nodiscard]] static constexpr size_t persistent_packed_row_bytes(
        uint32_t reference_count
    ) noexcept {
        return static_cast<size_t>(reference_count) *
               (BucketCount * sizeof(register_type) + sizeof(uint32_t));
    }

    [[nodiscard]] constexpr size_t persistent_row_bytes() const noexcept {
        return packed_ ? packed_rows_.size_bytes() + saturation_states_.size_bytes()
                       : rows_.size_bytes();
    }

    [[nodiscard]] constexpr size_t persistent_index_bytes() const noexcept {
        return index_offsets_.size_bytes() + index_postings_.size_bytes();
    }

    [[nodiscard]] static constexpr size_t single_query_workspace_bytes(uint32_t) noexcept {
        return 0;
    }

    [[nodiscard]] constexpr size_t single_query_workspace_bytes() const noexcept {
        return 0;
    }

    [[nodiscard]] static constexpr uint32_t single_query_result_count(
        uint32_t reference_count
    ) noexcept {
        return reference_count;
    }

    [[nodiscard]] constexpr uint32_t single_query_result_count() const noexcept {
        return metadata_.reference_count;
    }

    /// @brief Caller-owned bytes required by one positive-threshold indexed query.
    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes() const {
        CUDDL_TRY(validate_index_storage());
        if (metadata_.reference_count == 0U) {
            return size_t{0};
        }

        size_t selection_bytes = 0;
        auto const ids = thrust::make_counting_iterator(uint32_t{0});
        auto const selection = cuda_try(
            cub::DeviceSelect::If(
                nullptr,
                selection_bytes,
                ids,
                static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(metadata_.reference_count),
                detail::minimum_match_predicate{nullptr, 1U}
            )
        );
        if (!selection) {
            return Err(selection.error());
        }

        constexpr size_t alignment_slack = 3U + 255U;
        auto const arrays_bytes =
            static_cast<size_t>(metadata_.reference_count) * 2U * sizeof(uint32_t);
        if (selection_bytes > std::numeric_limits<size_t>::max() - arrays_bytes - alignment_slack) {
            return Err(Error::resource("indexed single-query workspace size overflows"));
        }
        return arrays_bytes + alignment_slack + selection_bytes;
    }

    /// @brief Storage required to exhaustively search @p query_count compact rows.
    [[nodiscard]] Result<cuddl::batch_search_requirements> batch_search_requirements(
        uint32_t query_count
    ) const {
        auto const pair_count = CUDDL_TRY(dense_batch_pair_count(query_count));
        return make_batch_requirements(pair_count, pair_count, false);
    }

    /// @brief Storage required to search @p query_count compact rows through the index.
    [[nodiscard]] Result<cuddl::batch_search_requirements> indexed_batch_search_requirements(
        uint32_t query_count
    ) const {
        CUDDL_TRY(validate_index_storage());
        auto const pair_count = CUDDL_TRY(dense_batch_pair_count(query_count));
        return make_batch_requirements(pair_count, pair_count, true);
    }

    /// @brief Storage required for one exhaustive all-to-all query tile.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    all_to_all_search_requirements(uint32_t first_query_id, uint32_t query_count) const {
        auto const pair_count = CUDDL_TRY(all_to_all_pair_count(first_query_id, query_count));
        return make_batch_requirements(0U, pair_count, false);
    }

    /// @brief Storage required for one indexed all-to-all query tile.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_all_to_all_search_requirements(uint32_t first_query_id, uint32_t query_count) const {
        CUDDL_TRY(validate_index_storage());
        auto const dense_pair_count = CUDDL_TRY(dense_batch_pair_count(query_count));
        auto const result_pair_count =
            CUDDL_TRY(all_to_all_pair_count(first_query_id, query_count));
        return make_batch_requirements(dense_pair_count, result_pair_count, true);
    }

    /// @brief Compares one compact query row with every reference row on @p stream.
    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        if (query.size() != BucketCount || query.data() == nullptr) {
            return Err(Error::invalid_argument("query must contain one complete score row"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount>(query_compatibility);
            !validation) {
            return validation;
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        if (workspace.size_bytes() < single_query_workspace_bytes()) {
            return Err(Error::resource("single-query workspace is too small"));
        }
        if (results.size() < metadata_.reference_count) {
            return Err(Error::resource("single-query result capacity is too small"));
        }
        if (metadata_.reference_count == 0U) {
            return Ok();
        }
        if (results.data() == nullptr) {
            return Err(Error::invalid_argument("result storage must be device accessible"));
        }

        if constexpr (BucketCount == 2048U) {
            // ponytail: fixed crossover tuned on the RTX 5070 Ti; retune for other GPU families.
            constexpr uint32_t small_reference_limit = 2560U;
            constexpr uint32_t small_block_size = 256U;
            constexpr uint32_t small_warps_per_reference = 8U;
            constexpr uint32_t large_block_size = 128U;
            constexpr uint32_t large_warps_per_reference = 2U;
            if (metadata_.reference_count <= small_reference_limit) {
                constexpr uint32_t references_per_block =
                    small_block_size / (32U * small_warps_per_reference);
                auto const blocks =
                    (metadata_.reference_count + references_per_block - 1U) / references_per_block;
                if (packed_) {
                    detail::exhaustive_search_kernel<
                        BucketCount,
                        small_block_size,
                        small_warps_per_reference><<<blocks, small_block_size, 0, stream.get()>>>(
                        packed_rows_.data(), metadata_.reference_count, query.data(), results.data()
                    );
                } else {
                    detail::exhaustive_search_kernel<
                        BucketCount,
                        small_block_size,
                        small_warps_per_reference><<<blocks, small_block_size, 0, stream.get()>>>(
                        rows_.data(), metadata_.reference_count, query.data(), results.data()
                    );
                }
            } else {
                constexpr uint32_t references_per_block =
                    large_block_size / (32U * large_warps_per_reference);
                auto const blocks =
                    (metadata_.reference_count + references_per_block - 1U) / references_per_block;
                if (packed_) {
                    detail::exhaustive_search_kernel<
                        BucketCount,
                        large_block_size,
                        large_warps_per_reference><<<blocks, large_block_size, 0, stream.get()>>>(
                        packed_rows_.data(), metadata_.reference_count, query.data(), results.data()
                    );
                } else {
                    detail::exhaustive_search_kernel<
                        BucketCount,
                        large_block_size,
                        large_warps_per_reference><<<blocks, large_block_size, 0, stream.get()>>>(
                        rows_.data(), metadata_.reference_count, query.data(), results.data()
                    );
                }
            }
        } else {
            constexpr uint32_t fallback_block_size = 128U;
            constexpr uint32_t fallback_warps_per_reference = 2U;
            constexpr uint32_t references_per_block =
                fallback_block_size / (32U * fallback_warps_per_reference);
            auto const blocks =
                (metadata_.reference_count + references_per_block - 1U) / references_per_block;
            if (packed_) {
                detail::exhaustive_search_kernel<
                    BucketCount,
                    fallback_block_size,
                    fallback_warps_per_reference><<<blocks, fallback_block_size, 0, stream.get()>>>(
                    packed_rows_.data(), metadata_.reference_count, query.data(), results.data()
                );
            } else {
                detail::exhaustive_search_kernel<
                    BucketCount,
                    fallback_block_size,
                    fallback_warps_per_reference><<<blocks, fallback_block_size, 0, stream.get()>>>(
                    rows_.data(), metadata_.reference_count, query.data(), results.data()
                );
            }
        }
        return cuda_try(cudaGetLastError());
    }

    /// @brief Finds and exactly refines references meeting @p options on @p stream.
    [[nodiscard]] Result<void> search_indexed_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        device_span<uint32_t> result_count,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        CUDDL_TRY(validate_index_storage());
        if (query.size() != BucketCount || query.data() == nullptr) {
            return Err(Error::invalid_argument("query must contain one complete score row"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount>(query_compatibility);
            !validation) {
            return validation;
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        if (options.minimum_matches > metadata_.compatibility.indexed_bucket_count) {
            return Err(Error::invalid_argument("minimum matches exceeds indexed bucket count"));
        }
        if (results.size() < metadata_.reference_count) {
            return Err(Error::resource("indexed result capacity is too small"));
        }
        if (result_count.empty()) {
            return Err(Error::resource("indexed result count capacity is too small"));
        }
        if (result_count.data() == nullptr) {
            return Err(Error::invalid_argument("result count storage must be device accessible"));
        }
        if (metadata_.reference_count != 0U && results.data() == nullptr) {
            return Err(Error::invalid_argument("result storage must be device accessible"));
        }

        if (options.minimum_matches == 0U) {
            CUDDL_TRY(search_async(query, query_compatibility, {}, results, stream));
            detail::write_result_count_kernel<<<1, 1, 0, stream.get()>>>(
                metadata_.reference_count, result_count.data()
            );
            return cuda_try(cudaGetLastError());
        }
        if (metadata_.reference_count == 0U) {
            detail::write_result_count_kernel<<<1, 1, 0, stream.get()>>>(0U, result_count.data());
            return cuda_try(cudaGetLastError());
        }

        auto const required_workspace = CUDDL_TRY(indexed_single_query_workspace_bytes());
        if (workspace.size_bytes() < required_workspace) {
            return Err(Error::resource("indexed single-query workspace is too small"));
        }
        if (workspace.data() == nullptr) {
            return Err(Error::invalid_argument("indexed workspace must be device accessible"));
        }

        auto const workspace_begin = reinterpret_cast<uintptr_t>(workspace.data());
        auto const workspace_end = workspace_begin + workspace.size_bytes();
        if (workspace_end < workspace_begin) {
            return Err(Error::resource("indexed workspace address range overflows"));
        }
        auto address = detail::align_up(workspace_begin, alignof(uint32_t));
        auto* match_counts = reinterpret_cast<uint32_t*>(address);
        address += static_cast<size_t>(metadata_.reference_count) * sizeof(uint32_t);
        address = detail::align_up(address, alignof(uint32_t));
        auto* candidate_ids = reinterpret_cast<uint32_t*>(address);
        address += static_cast<size_t>(metadata_.reference_count) * sizeof(uint32_t);
        address = detail::align_up(address, 256U);
        if (address > workspace_end) {
            return Err(Error::resource("indexed workspace layout exceeds its capacity"));
        }
        auto* selection_workspace = reinterpret_cast<void*>(address);
        auto selection_bytes = static_cast<size_t>(workspace_end - address);

        CUDDL_CUDA_TRY(cudaMemsetAsync(
            match_counts,
            0,
            static_cast<size_t>(metadata_.reference_count) * sizeof(uint32_t),
            stream.get()
        ));
        auto const indexed_bucket_count = metadata_.compatibility.indexed_bucket_count;
        auto const bucket_blocks = static_cast<uint32_t>(
            (indexed_bucket_count + detail::block_size - 1U) / detail::block_size
        );
        detail::count_index_matches_kernel<BucketCount>
            <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                query.data(),
                index_offsets_.data(),
                index_postings_.data(),
                indexed_bucket_count,
                metadata_.compatibility.key_mask,
                match_counts
            );
        CUDDL_CUDA_TRY(cudaGetLastError());

        auto const ids = thrust::make_counting_iterator(uint32_t{0});
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                selection_workspace,
                selection_bytes,
                ids,
                candidate_ids,
                result_count.data(),
                static_cast<int64_t>(metadata_.reference_count),
                detail::minimum_match_predicate{match_counts, options.minimum_matches},
                stream.get()
            )
        );

        constexpr uint32_t warp_width = 32;
        constexpr uint32_t references_per_block = detail::block_size / warp_width;
        auto const refinement_blocks =
            (metadata_.reference_count + references_per_block - 1U) / references_per_block;
        if (packed_) {
            detail::refine_index_candidates_kernel<BucketCount>
                <<<refinement_blocks, detail::block_size, 0, stream.get()>>>(
                    packed_rows_.data(),
                    query.data(),
                    candidate_ids,
                    result_count.data(),
                    results.data()
                );
        } else {
            detail::refine_index_candidates_kernel<BucketCount>
                <<<refinement_blocks, detail::block_size, 0, stream.get()>>>(
                    rows_.data(), query.data(), candidate_ids, result_count.data(), results.data()
                );
        }
        return cuda_try(cudaGetLastError());
    }

    /// @brief Exhaustively searches a bounded compact query tile.
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        static_cast<void>(workspace);
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(batch_search_requirements(query_count));
        CUDDL_TRY(
            validate_batch_outputs(requirements, results, result_count, result_match_counts, true)
        );
        if (requirements.maximum_pair_count == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        if (packed_) {
            return launch_batch_exhaustive<false>(
                queries,
                0U,
                query_count,
                query_id_offset,
                packed_rows_,
                requirements.maximum_pair_count,
                results,
                result_count,
                result_match_counts,
                stream
            );
        }
        return launch_batch_exhaustive<false>(
            queries,
            0U,
            query_count,
            query_id_offset,
            rows_,
            requirements.maximum_pair_count,
            results,
            result_count,
            result_match_counts,
            stream
        );
    }

    /**
     * @brief Indexed search for a bounded compact query tile.
     *
     * After stream completion, a count larger than @p results reports insufficient pair capacity.
     * In that case neither results nor optional diagnostics are modified.
     */
    [[nodiscard]] Result<void> search_batch_indexed_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(indexed_batch_search_requirements(query_count));
        CUDDL_TRY(validate_indexed_batch_inputs(
            requirements, workspace, results, result_count, result_match_counts, options
        ));
        if (requirements.counter_bytes == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        if (packed_) {
            return launch_batch_indexed<false>(
                queries,
                0U,
                query_count,
                query_id_offset,
                packed_rows_,
                requirements,
                workspace,
                results,
                result_count,
                result_match_counts,
                options,
                stream
            );
        }
        return launch_batch_indexed<false>(
            queries,
            0U,
            query_count,
            query_id_offset,
            rows_,
            requirements,
            workspace,
            results,
            result_count,
            result_match_counts,
            options,
            stream
        );
    }

    /// @brief Exhaustively searches one database-row tile against later database rows.
    [[nodiscard]] Result<void> search_all_to_all_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        static_cast<void>(workspace);
        CUDDL_TRY(validate_stored_rows());
        auto const requirements =
            CUDDL_TRY(all_to_all_search_requirements(first_query_id, query_count));
        CUDDL_TRY(
            validate_batch_outputs(requirements, results, result_count, result_match_counts, true)
        );
        if (requirements.maximum_pair_count == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        if (packed_) {
            return launch_batch_exhaustive<true>(
                packed_rows_,
                first_query_id,
                query_count,
                first_query_id,
                packed_rows_,
                requirements.maximum_pair_count,
                results,
                result_count,
                result_match_counts,
                stream
            );
        }
        return launch_batch_exhaustive<true>(
            rows_,
            first_query_id,
            query_count,
            first_query_id,
            rows_,
            requirements.maximum_pair_count,
            results,
            result_count,
            result_match_counts,
            stream
        );
    }

    /**
     * @brief Indexed search for one database-row tile against later database rows.
     *
     * After stream completion, a count larger than @p results reports insufficient pair capacity.
     * In that case neither results nor optional diagnostics are modified.
     */
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        CUDDL_TRY(validate_stored_rows());
        auto const requirements =
            CUDDL_TRY(indexed_all_to_all_search_requirements(first_query_id, query_count));
        CUDDL_TRY(validate_indexed_batch_inputs(
            requirements, workspace, results, result_count, result_match_counts, options
        ));
        if (requirements.counter_bytes == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        if (packed_) {
            return launch_batch_indexed<true>(
                packed_rows_,
                first_query_id,
                query_count,
                first_query_id,
                packed_rows_,
                requirements,
                workspace,
                results,
                result_count,
                result_match_counts,
                options,
                stream
            );
        }
        return launch_batch_indexed<true>(
            rows_,
            first_query_id,
            query_count,
            first_query_id,
            rows_,
            requirements,
            workspace,
            results,
            result_count,
            result_match_counts,
            options,
            stream
        );
    }

   private:
    [[nodiscard]] Result<void> validate_stored_rows() const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        return Ok();
    }

    [[nodiscard]] Result<void> validate_index_storage() const {
        CUDDL_TRY((
            detail::validate_indexed_score_compatibility<K, BucketCount>(metadata_.compatibility)
        ));
        auto const cell_count = detail::indexed_cell_count(metadata_.compatibility);
        auto const expected_postings = static_cast<size_t>(
            detail::indexed_posting_count(metadata_.reference_count, metadata_.compatibility)
        );
        if (!indexed_ || index_offsets_.size() != static_cast<size_t>(cell_count + 1U) ||
            index_offsets_.data() == nullptr || index_postings_.size() != expected_postings ||
            (expected_postings != 0U && index_postings_.data() == nullptr)) {
            return Err(Error::invalid_argument("database has no valid retrieval index"));
        }
        return Ok();
    }

    [[nodiscard]] Result<uint32_t> dense_batch_pair_count(uint32_t query_count) const {
        auto const pair_count = static_cast<uint64_t>(query_count) * metadata_.reference_count;
        if (pair_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("batch pair count exceeds 32-bit workspace IDs"));
        }
        return static_cast<uint32_t>(pair_count);
    }

    [[nodiscard]] Result<uint32_t>
    all_to_all_pair_count(uint32_t first_query_id, uint32_t query_count) const {
        if (first_query_id > metadata_.reference_count ||
            query_count > metadata_.reference_count - first_query_id) {
            return Err(Error::invalid_argument("all-to-all query tile is outside the database"));
        }
        if (query_count == 0U) {
            return 0U;
        }

        uint64_t left = query_count;
        uint64_t right = 2U * static_cast<uint64_t>(metadata_.reference_count) -
                         2U * first_query_id - query_count - 1U;
        if ((left & 1U) == 0U) {
            left /= 2U;
        } else {
            right /= 2U;
        }
        auto const pair_count = left * right;
        if (pair_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("all-to-all tile exceeds 32-bit result counts"));
        }
        return static_cast<uint32_t>(pair_count);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> make_batch_requirements(
        uint32_t dense_pair_count,
        uint32_t maximum_pair_count,
        bool indexed
    ) const {
        cuddl::batch_search_requirements requirements{
            .maximum_pair_count = maximum_pair_count,
            .result_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(batch_result_type),
            .match_count_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(uint32_t),
        };
        if (!indexed || dense_pair_count == 0U) {
            return requirements;
        }

        size_t selection_bytes = 0;
        auto const ids = thrust::make_counting_iterator(uint32_t{0});
        auto const selection = cuda_try(
            cub::DeviceSelect::If(
                nullptr,
                selection_bytes,
                ids,
                static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(dense_pair_count),
                detail::batch_minimum_match_predicate{
                    nullptr, 0U, metadata_.reference_count, 0U, false
                }
            )
        );
        if (!selection) {
            return Err(selection.error());
        }

        requirements.counter_bytes = static_cast<size_t>(dense_pair_count) * sizeof(uint32_t);
        requirements.candidate_bytes = static_cast<size_t>(dense_pair_count) * sizeof(uint32_t);
        requirements.temporary_bytes = selection_bytes;
        constexpr size_t alignment_slack = alignof(uint32_t) - 1U + 255U;
        auto const arrays_bytes = requirements.counter_bytes + requirements.candidate_bytes;
        if (selection_bytes > std::numeric_limits<size_t>::max() - arrays_bytes - alignment_slack) {
            return Err(Error::resource("indexed batch workspace size overflows"));
        }
        requirements.workspace_bytes = arrays_bytes + alignment_slack + selection_bytes;
        return requirements;
    }

    [[nodiscard]] Result<uint32_t> validate_batch_queries(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset
    ) const {
        CUDDL_TRY(validate_stored_rows());
        if (queries.size() % BucketCount != 0U) {
            return Err(Error::invalid_argument("query tile must contain complete score rows"));
        }
        if (!queries.empty() && queries.data() == nullptr) {
            return Err(Error::invalid_argument("query tile must be device accessible"));
        }
        auto const query_count = queries.size() / BucketCount;
        if (query_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("query tile exceeds stable 32-bit IDs"));
        }
        auto const count = static_cast<uint32_t>(query_count);
        if (count != 0U && query_id_offset > std::numeric_limits<uint32_t>::max() - (count - 1U)) {
            return Err(Error::resource("query IDs exceed the stable 32-bit range"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount>(query_compatibility);
            !validation) {
            return Err(validation.error());
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        return count;
    }

    [[nodiscard]] Result<void> validate_batch_outputs(
        cuddl::batch_search_requirements const& requirements,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        bool require_full_capacity
    ) const {
        if (result_count.empty()) {
            return Err(Error::resource("batch result count capacity is too small"));
        }
        if (result_count.data() == nullptr) {
            return Err(Error::invalid_argument("batch result count must be device accessible"));
        }
        if (require_full_capacity && results.size() < requirements.maximum_pair_count) {
            return Err(Error::resource("batch result capacity is too small"));
        }
        auto const written_capacity = results.size() < requirements.maximum_pair_count
                                          ? results.size()
                                          : requirements.maximum_pair_count;
        if (written_capacity != 0U && results.data() == nullptr) {
            return Err(Error::invalid_argument("batch results must be device accessible"));
        }
        if (!result_match_counts.empty()) {
            if (result_match_counts.size() < written_capacity) {
                return Err(Error::resource("batch match-count capacity is too small"));
            }
            if (result_match_counts.data() == nullptr) {
                return Err(Error::invalid_argument("batch match counts must be device accessible"));
            }
        }
        return Ok();
    }

    [[nodiscard]] Result<void> validate_indexed_batch_inputs(
        cuddl::batch_search_requirements const& requirements,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options
    ) const {
        CUDDL_TRY(validate_index_storage());
        if (options.minimum_matches > metadata_.compatibility.indexed_bucket_count) {
            return Err(Error::invalid_argument("minimum matches exceeds indexed bucket count"));
        }
        if (workspace.size_bytes() < requirements.workspace_bytes) {
            return Err(Error::resource("indexed batch workspace is too small"));
        }
        if (requirements.workspace_bytes != 0U && workspace.data() == nullptr) {
            return Err(
                Error::invalid_argument("indexed batch workspace must be device accessible")
            );
        }
        return validate_batch_outputs(
            requirements, results, result_count, result_match_counts, false
        );
    }

    [[nodiscard]] static Result<void> write_batch_result_count(
        uint32_t count,
        device_span<uint32_t> result_count,
        cuda::stream_ref stream
    ) {
        detail::write_result_count_kernel<<<1, 1, 0, stream.get()>>>(count, result_count.data());
        return cuda_try(cudaGetLastError());
    }

    template <bool AllToAll, typename QueryRow, typename ReferenceRow>
    [[nodiscard]] Result<void> launch_batch_exhaustive(
        device_span<QueryRow const> queries,
        size_t query_row_offset,
        uint32_t query_count,
        uint32_t query_id_offset,
        device_span<ReferenceRow const> rows,
        uint32_t pair_count,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        constexpr uint32_t warp_width = 32;
        constexpr uint32_t references_per_block = detail::block_size / warp_width;
        auto const reference_blocks = static_cast<uint32_t>(
            (static_cast<uint64_t>(metadata_.reference_count) + references_per_block - 1U) /
            references_per_block
        );
        // The exhaustive (non-all-to-all) path is reference-major and ignores the query grid
        // dimension; the all-to-all path strides queries across gridDim.y.
        auto const query_blocks = AllToAll ? (query_count < 65535U ? query_count : 65535U) : 1U;
        dim3 const blocks{reference_blocks, query_blocks, 1U};
        detail::batch_exhaustive_search_kernel<BucketCount, AllToAll>
            <<<blocks, detail::block_size, 0, stream.get()>>>(
                queries.data(),
                query_row_offset,
                query_count,
                query_id_offset,
                rows.data(),
                metadata_.reference_count,
                results.data(),
                result_match_counts.empty() ? nullptr : result_match_counts.data()
            );
        CUDDL_CUDA_TRY(cudaGetLastError());
        return write_batch_result_count(pair_count, result_count, stream);
    }

    template <bool AllToAll, typename QueryRow, typename ReferenceRow>
    [[nodiscard]] Result<void> launch_batch_indexed(
        device_span<QueryRow const> queries,
        size_t query_row_offset,
        uint32_t query_count,
        uint32_t query_id_offset,
        device_span<ReferenceRow const> rows,
        cuddl::batch_search_requirements const& requirements,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        auto const workspace_begin = reinterpret_cast<uintptr_t>(workspace.data());
        auto const workspace_end = workspace_begin + workspace.size_bytes();
        if (workspace_end < workspace_begin) {
            return Err(Error::resource("indexed batch workspace address range overflows"));
        }
        auto address = detail::align_up(workspace_begin, alignof(uint32_t));
        auto* match_counts = reinterpret_cast<uint32_t*>(address);
        address += requirements.counter_bytes;
        address = detail::align_up(address, alignof(uint32_t));
        auto* candidate_ids = reinterpret_cast<uint32_t*>(address);
        address += requirements.candidate_bytes;
        address = detail::align_up(address, 256U);
        if (address > workspace_end) {
            return Err(Error::resource("indexed batch workspace layout exceeds its capacity"));
        }
        auto* selection_workspace = reinterpret_cast<void*>(address);
        auto selection_bytes = static_cast<size_t>(workspace_end - address);
        auto const dense_pair_count =
            static_cast<uint32_t>(requirements.counter_bytes / sizeof(uint32_t));

        CUDDL_CUDA_TRY(cudaMemsetAsync(match_counts, 0, requirements.counter_bytes, stream.get()));
        auto const query_buckets =
            static_cast<size_t>(query_count) * metadata_.compatibility.indexed_bucket_count;
        auto const required_bucket_blocks =
            (query_buckets + detail::block_size - 1U) / detail::block_size;
        auto const bucket_blocks = static_cast<uint32_t>(
            required_bucket_blocks < 65535U ? required_bucket_blocks : 65535U
        );
        detail::count_batch_index_matches_kernel<BucketCount>
            <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                queries.data(),
                query_row_offset,
                query_count,
                index_offsets_.data(),
                index_postings_.data(),
                metadata_.reference_count,
                metadata_.compatibility.indexed_bucket_count,
                metadata_.compatibility.key_mask,
                match_counts
            );
        CUDDL_CUDA_TRY(cudaGetLastError());

        auto const ids = thrust::make_counting_iterator(uint32_t{0});
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                selection_workspace,
                selection_bytes,
                ids,
                candidate_ids,
                result_count.data(),
                static_cast<int64_t>(dense_pair_count),
                detail::batch_minimum_match_predicate{
                    match_counts,
                    options.minimum_matches,
                    metadata_.reference_count,
                    query_id_offset,
                    AllToAll
                },
                stream.get()
            )
        );

        auto const written_capacity = results.size() < requirements.maximum_pair_count
                                          ? results.size()
                                          : requirements.maximum_pair_count;
        if (written_capacity == 0U) {
            return Ok();
        }
        constexpr uint32_t warp_width = 32;
        constexpr uint32_t candidates_per_block = detail::block_size / warp_width;
        auto const capacity = static_cast<uint32_t>(written_capacity);
        auto const refinement_blocks =
            (capacity + candidates_per_block - 1U) / candidates_per_block;
        detail::refine_batch_index_candidates_kernel<BucketCount>
            <<<refinement_blocks, detail::block_size, 0, stream.get()>>>(
                queries.data(),
                query_row_offset,
                query_id_offset,
                rows.data(),
                metadata_.reference_count,
                match_counts,
                candidate_ids,
                result_count.data(),
                capacity,
                results.data(),
                result_match_counts.empty() ? nullptr : result_match_counts.data()
            );
        return cuda_try(cudaGetLastError());
    }

   private:
    [[nodiscard]] __host__ __device__ constexpr bool rows_match_metadata(
        size_t expected_scores
    ) const noexcept {
        if (packed_) {
            return packed_rows_.size() == expected_scores &&
                   saturation_states_.size() == metadata_.reference_count &&
                   (expected_scores == 0U || packed_rows_.data() != nullptr) &&
                   (metadata_.reference_count == 0U || saturation_states_.data() != nullptr);
        }
        return rows_.size() == expected_scores &&
               (expected_scores == 0U || rows_.data() != nullptr);
    }

    device_span<score_type const> rows_;
    device_span<register_type const> packed_rows_;
    device_span<uint32_t const> saturation_states_;
    reference_database_metadata metadata_;
    device_span<uint32_t const> index_offsets_;
    device_span<uint32_t const> index_postings_;
    bool packed_{};
    bool indexed_{};
};

/**
 * @brief Move-only owner of one immutable contiguous reference database.
 *
 * Building enqueues row copies on the supplied stream. Inputs and the returned database must
 * remain alive until that stream completes.
 */
template <uint32_t K, size_t BucketCount>
class reference_database {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using ref_type = reference_database_ref<K, BucketCount>;
    using score_type = typename ref_type::score_type;
    using register_type = typename ref_type::register_type;
    using result_type = typename ref_type::result_type;
    using batch_result_type = typename ref_type::batch_result_type;

    reference_database(reference_database const&) = delete;
    reference_database& operator=(reference_database const&) = delete;

    reference_database(reference_database&& other) noexcept
        : rows_(other.rows_),
          saturation_states_(other.saturation_states_),
          index_offsets_(other.index_offsets_),
          index_postings_(other.index_postings_),
          index_posting_capacity_(other.index_posting_capacity_),
          metadata_(other.metadata_),
          packed_(other.packed_),
          indexed_(other.indexed_) {
        other.rows_ = nullptr;
        other.saturation_states_ = nullptr;
        other.index_offsets_ = nullptr;
        other.index_postings_ = nullptr;
        other.index_posting_capacity_ = 0;
        other.metadata_ = {};
        other.packed_ = false;
        other.indexed_ = false;
    }

    reference_database& operator=(reference_database&& other) noexcept {
        if (this != &other) {
            destroy();
            rows_ = other.rows_;
            saturation_states_ = other.saturation_states_;
            index_offsets_ = other.index_offsets_;
            index_postings_ = other.index_postings_;
            index_posting_capacity_ = other.index_posting_capacity_;
            metadata_ = other.metadata_;
            packed_ = other.packed_;
            indexed_ = other.indexed_;
            other.rows_ = nullptr;
            other.saturation_states_ = nullptr;
            other.index_offsets_ = nullptr;
            other.index_postings_ = nullptr;
            other.index_posting_capacity_ = 0;
            other.metadata_ = {};
            other.packed_ = false;
            other.indexed_ = false;
        }
        return *this;
    }

    ~reference_database() {
        destroy();
    }

    /// @brief Builds a database by copying flat row-major scores on @p stream.
    [[nodiscard]] static Result<reference_database> build_async(
        device_span<score_type const> rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_rows_async(rows, {}, compatibility, stream);
    }

    /// @brief Builds a multiplicity-preserving database from packed rows and saturation states.
    [[nodiscard]] static Result<reference_database> build_async(
        device_span<register_type const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_rows_async(rows, saturation_states, compatibility, stream);
    }

    /// @brief Thrust overload for flat row-major compact device storage.
    [[nodiscard]] static Result<reference_database> build_async(
        thrust::device_vector<score_type> const& rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()}, compatibility, stream
        );
    }

    /// @brief Thrust overload for packed rows and per-reference saturation states.
    [[nodiscard]] static Result<reference_database> build_async(
        thrust::device_vector<register_type> const& rows,
        thrust::device_vector<uint32_t> const& saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()},
            {thrust::raw_pointer_cast(saturation_states.data()), saturation_states.size()},
            compatibility,
            stream
        );
    }

    /// @brief Builds compact rows and their indexed dense-offset index.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        device_span<score_type const> rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_indexed_rows_async(rows, {}, compatibility, stream);
    }

    /// @brief Builds packed rows and their indexed winner-score dense-offset index.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        device_span<register_type const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_indexed_rows_async(rows, saturation_states, compatibility, stream);
    }

    /// @brief Thrust overload for indexed flat row-major compact device storage.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        thrust::device_vector<score_type> const& rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_indexed_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()}, compatibility, stream
        );
    }

    /// @brief Thrust overload for indexed packed rows and saturation states.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        thrust::device_vector<register_type> const& rows,
        thrust::device_vector<uint32_t> const& saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_indexed_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()},
            {thrust::raw_pointer_cast(saturation_states.data()), saturation_states.size()},
            compatibility,
            stream
        );
    }

    [[nodiscard]] ref_type ref() const noexcept {
        auto const offset_count =
            indexed_ ? static_cast<size_t>(detail::indexed_cell_count(metadata_.compatibility) + 1U)
                     : 0U;
        auto const row_count = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        auto const offsets = device_span<uint32_t const>{index_offsets_, offset_count};
        auto const postings = device_span<uint32_t const>{index_postings_, index_posting_capacity_};
        if (packed_) {
            return ref_type(
                {static_cast<register_type const*>(rows_), row_count},
                {saturation_states_, metadata_.reference_count},
                metadata_,
                offsets,
                postings,
                indexed_
            );
        }
        return ref_type(
            {static_cast<score_type const*>(rows_), row_count},
            metadata_,
            offsets,
            postings,
            indexed_
        );
    }

    [[nodiscard]] reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    [[nodiscard]] bool has_index() const noexcept {
        return indexed_;
    }

    [[nodiscard]] bool preserves_multiplicity() const noexcept {
        return packed_;
    }

    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return ref_type::persistent_row_bytes(reference_count);
    }

    [[nodiscard]] static constexpr size_t persistent_packed_row_bytes(
        uint32_t reference_count
    ) noexcept {
        return ref_type::persistent_packed_row_bytes(reference_count);
    }

    [[nodiscard]] size_t persistent_row_bytes() const noexcept {
        return ref().persistent_row_bytes();
    }

    [[nodiscard]] size_t persistent_index_bytes() const noexcept {
        return ref().persistent_index_bytes();
    }

    [[nodiscard]] static constexpr size_t single_query_workspace_bytes(
        uint32_t reference_count
    ) noexcept {
        return ref_type::single_query_workspace_bytes(reference_count);
    }

    [[nodiscard]] size_t single_query_workspace_bytes() const noexcept {
        return ref().single_query_workspace_bytes();
    }

    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes() const {
        return ref().indexed_single_query_workspace_bytes();
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> batch_search_requirements(
        uint32_t query_count
    ) const {
        return ref().batch_search_requirements(query_count);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> indexed_batch_search_requirements(
        uint32_t query_count
    ) const {
        return ref().indexed_batch_search_requirements(query_count);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    all_to_all_search_requirements(uint32_t first_query_id, uint32_t query_count) const {
        return ref().all_to_all_search_requirements(first_query_id, query_count);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_all_to_all_search_requirements(uint32_t first_query_id, uint32_t query_count) const {
        return ref().indexed_all_to_all_search_requirements(first_query_id, query_count);
    }

    [[nodiscard]] static constexpr uint32_t single_query_result_count(
        uint32_t reference_count
    ) noexcept {
        return ref_type::single_query_result_count(reference_count);
    }

    [[nodiscard]] uint32_t single_query_result_count() const noexcept {
        return ref().single_query_result_count();
    }

    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_async(query, query_compatibility, workspace, results, stream);
    }

    /// @brief Thrust overload for caller-owned query, workspace, and result storage.
    [[nodiscard]] Result<void> search_async(
        thrust::device_vector<score_type> const& query,
        score_compatibility const& query_compatibility,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<result_type>& results,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_async(
            {thrust::raw_pointer_cast(query.data()), query.size()},
            query_compatibility,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            stream
        );
    }

    [[nodiscard]] Result<void> search_indexed_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        device_span<uint32_t> result_count,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_indexed_async(
            query, query_compatibility, workspace, results, result_count, options, stream
        );
    }

    /// @brief Thrust overload for caller-owned indexed-query storage.
    [[nodiscard]] Result<void> search_indexed_async(
        thrust::device_vector<score_type> const& query,
        score_compatibility const& query_compatibility,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_indexed_async(
            {thrust::raw_pointer_cast(query.data()), query.size()},
            query_compatibility,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            options,
            stream
        );
    }

    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_batch_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            result_count,
            result_match_counts,
            stream
        );
    }

    /// @brief Thrust overload for caller-owned exhaustive batch-search storage.
    [[nodiscard]] Result<void> search_batch_async(
        thrust::device_vector<score_type> const& queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        thrust::device_vector<uint32_t>& result_match_counts,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_batch_async(
            {thrust::raw_pointer_cast(queries.data()), queries.size()},
            query_compatibility,
            query_id_offset,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {thrust::raw_pointer_cast(result_match_counts.data()), result_match_counts.size()},
            stream
        );
    }

    /// @brief Thrust overload without optional match-count diagnostics.
    [[nodiscard]] Result<void> search_batch_async(
        thrust::device_vector<score_type> const& queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_batch_async(
            {thrust::raw_pointer_cast(queries.data()), queries.size()},
            query_compatibility,
            query_id_offset,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {},
            stream
        );
    }

    [[nodiscard]] Result<void> search_batch_indexed_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_batch_indexed_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            result_count,
            result_match_counts,
            options,
            stream
        );
    }

    /// @brief Thrust overload for caller-owned indexed batch-search storage.
    [[nodiscard]] Result<void> search_batch_indexed_async(
        thrust::device_vector<score_type> const& queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        thrust::device_vector<uint32_t>& result_match_counts,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_batch_indexed_async(
            {thrust::raw_pointer_cast(queries.data()), queries.size()},
            query_compatibility,
            query_id_offset,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {thrust::raw_pointer_cast(result_match_counts.data()), result_match_counts.size()},
            options,
            stream
        );
    }

    /// @brief Thrust overload without optional match-count diagnostics.
    [[nodiscard]] Result<void> search_batch_indexed_async(
        thrust::device_vector<score_type> const& queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_batch_indexed_async(
            {thrust::raw_pointer_cast(queries.data()), queries.size()},
            query_compatibility,
            query_id_offset,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {},
            options,
            stream
        );
    }

    [[nodiscard]] Result<void> search_all_to_all_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_all_to_all_async(
            first_query_id,
            query_count,
            workspace,
            results,
            result_count,
            result_match_counts,
            stream
        );
    }

    /// @brief Thrust overload for caller-owned exhaustive all-to-all storage.
    [[nodiscard]] Result<void> search_all_to_all_async(
        uint32_t first_query_id,
        uint32_t query_count,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        thrust::device_vector<uint32_t>& result_match_counts,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_all_to_all_async(
            first_query_id,
            query_count,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {thrust::raw_pointer_cast(result_match_counts.data()), result_match_counts.size()},
            stream
        );
    }

    /// @brief Thrust overload without optional match-count diagnostics.
    [[nodiscard]] Result<void> search_all_to_all_async(
        uint32_t first_query_id,
        uint32_t query_count,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_all_to_all_async(
            first_query_id,
            query_count,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {},
            stream
        );
    }

    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts = {},
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return ref().search_all_to_all_indexed_async(
            first_query_id,
            query_count,
            workspace,
            results,
            result_count,
            result_match_counts,
            options,
            stream
        );
    }

    /// @brief Thrust overload for caller-owned indexed all-to-all storage.
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        uint32_t first_query_id,
        uint32_t query_count,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        thrust::device_vector<uint32_t>& result_match_counts,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_all_to_all_indexed_async(
            first_query_id,
            query_count,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {thrust::raw_pointer_cast(result_match_counts.data()), result_match_counts.size()},
            options,
            stream
        );
    }

    /// @brief Thrust overload without optional match-count diagnostics.
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        uint32_t first_query_id,
        uint32_t query_count,
        thrust::device_vector<uint8_t>& workspace,
        thrust::device_vector<batch_result_type>& results,
        thrust::device_vector<uint32_t>& result_count,
        indexed_search_options options = {},
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return search_all_to_all_indexed_async(
            first_query_id,
            query_count,
            {thrust::raw_pointer_cast(workspace.data()), workspace.size()},
            {thrust::raw_pointer_cast(results.data()), results.size()},
            {thrust::raw_pointer_cast(result_count.data()), result_count.size()},
            {},
            options,
            stream
        );
    }

   private:
    reference_database() = default;

    template <typename Row>
    [[nodiscard]] static Result<uint32_t> validate_rows(
        device_span<Row const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        bool indexed
    ) {
        static_assert(std::is_same_v<Row, score_type> || std::is_same_v<Row, register_type>);
        if (indexed) {
            if (auto const validation =
                    detail::validate_indexed_score_compatibility<K, BucketCount>(compatibility);
                !validation) {
                return Err(validation.error());
            }
        } else {
            if (auto const validation =
                    detail::validate_non_indexed_score_compatibility<K, BucketCount>(compatibility);
                !validation) {
                return Err(validation.error());
            }
        }
        if (rows.size() % BucketCount != 0U) {
            return Err(Error::invalid_argument("row extent must contain complete rows"));
        }
        if (!rows.empty() && rows.data() == nullptr) {
            return Err(Error::invalid_argument("rows must be device accessible"));
        }
        auto const reference_count = rows.size() / BucketCount;
        if (reference_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("reference count exceeds stable 32-bit IDs"));
        }
        if constexpr (std::is_same_v<Row, register_type>) {
            if (saturation_states.size() != reference_count) {
                return Err(
                    Error::invalid_argument(
                        "saturation extent must match the packed reference rows"
                    )
                );
            }
            if (!saturation_states.empty() && saturation_states.data() == nullptr) {
                return Err(Error::invalid_argument("saturation states must be device accessible"));
            }
        }
        return static_cast<uint32_t>(reference_count);
    }

    template <typename Row>
    [[nodiscard]] static Result<reference_database> build_storage_async(
        device_span<Row const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        uint32_t reference_count,
        cuda::stream_ref stream
    ) {
        reference_database database;
        database.metadata_ = {
            .compatibility = compatibility,
            .reference_count = reference_count,
        };
        database.packed_ = std::is_same_v<Row, register_type>;
        if (rows.empty()) {
            return Result<reference_database>::ok(std::move(database));
        }
        if (auto const allocation = cuda_try(cudaMalloc(&database.rows_, rows.size_bytes()));
            !allocation) {
            return Err(allocation.error());
        }
        if (auto const copy = cuda_try(cudaMemcpyAsync(
                database.rows_,
                rows.data(),
                rows.size_bytes(),
                cudaMemcpyDeviceToDevice,
                stream.get()
            ));
            !copy) {
            return Err(copy.error());
        }
        if constexpr (std::is_same_v<Row, register_type>) {
            if (auto const allocation = cuda_try(
                    cudaMalloc(&database.saturation_states_, saturation_states.size_bytes())
                );
                !allocation) {
                return Err(allocation.error());
            }
            if (auto const copy = cuda_try(cudaMemcpyAsync(
                    database.saturation_states_,
                    saturation_states.data(),
                    saturation_states.size_bytes(),
                    cudaMemcpyDeviceToDevice,
                    stream.get()
                ));
                !copy) {
                return Err(copy.error());
            }
        }
        return Result<reference_database>::ok(std::move(database));
    }

    template <typename Row>
    [[nodiscard]] static Result<reference_database> build_rows_async(
        device_span<Row const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream
    ) {
        auto const validated = validate_rows(rows, saturation_states, compatibility, false);
        if (!validated) {
            return Err(validated.error());
        }
        return build_storage_async(rows, saturation_states, compatibility, *validated, stream);
    }

    template <typename Row>
    [[nodiscard]] static Result<reference_database> build_indexed_rows_async(
        device_span<Row const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream
    ) {
        auto const validated = validate_rows(rows, saturation_states, compatibility, true);
        if (!validated) {
            return Err(validated.error());
        }
        auto const posting_capacity = detail::indexed_posting_count(*validated, compatibility);
        if (posting_capacity > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("index postings exceed 32-bit offsets"));
        }
        if (posting_capacity > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return Err(Error::resource("index posting allocation overflows"));
        }
        auto const cell_count = detail::indexed_cell_count(compatibility);
        auto const offset_count = cell_count + 1U;
        if (offset_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return Err(Error::resource("dense index offset allocation overflows"));
        }

        auto built =
            build_storage_async(rows, saturation_states, compatibility, *validated, stream);
        if (!built) {
            return Err(built.error());
        }
        auto database = std::move(*built);
        auto const offset_bytes = static_cast<size_t>(offset_count) * sizeof(uint32_t);
        if (auto const allocation = cuda_try(cudaMalloc(&database.index_offsets_, offset_bytes));
            !allocation) {
            return Err(allocation.error());
        }
        if (rows.empty()) {
            CUDDL_CUDA_TRY(cudaMemsetAsync(database.index_offsets_, 0, offset_bytes, stream.get()));
            database.indexed_ = true;
            return Result<reference_database>::ok(std::move(database));
        }

        auto const posting_bytes = static_cast<size_t>(posting_capacity) * sizeof(uint32_t);
        if (auto const allocation = cuda_try(cudaMalloc(&database.index_postings_, posting_bytes));
            !allocation) {
            return Err(allocation.error());
        }
        database.index_posting_capacity_ = static_cast<size_t>(posting_capacity);

        size_t scan_bytes = 0;
        if (auto const query = cuda_try(
                cub::DeviceScan::ExclusiveSum(
                    nullptr,
                    scan_bytes,
                    database.index_offsets_,
                    static_cast<int64_t>(cell_count + 1U),
                    stream.get()
                )
            );
            !query) {
            return Err(query.error());
        }
        auto const scratch_bytes =
            detail::align_up(static_cast<uintptr_t>(scan_bytes), alignof(uint8_t));
        uint8_t* scratch = nullptr;
        if (auto const allocation =
                cuda_try(cudaMallocAsync(&scratch, scratch_bytes, stream.get()));
            !allocation) {
            return Err(allocation.error());
        }
        Row* transposed = nullptr;
        auto fail = [&](Error const& error) -> Result<reference_database> {
            if (transposed != nullptr) {
                auto const release = cuda_try(cudaFreeAsync(transposed, stream.get()));
                if (!release) {
                    return Err(release.error());
                }
            }
            auto const release = cuda_try(cudaFreeAsync(scratch, stream.get()));
            if (!release) {
                return Err(release.error());
            }
            return Err(error);
        };
        // Construction scratch is scan storage plus one bucket-major transpose of the indexed
        // rows; the scatter's per-key ranks live in shared memory, so no per-cell cursor array
        // is allocated at all.

        auto const indexed_row_count = static_cast<size_t>(posting_capacity);
        auto const reference_count = static_cast<uint32_t>(rows.size() / BucketCount);

        // Bucket-major transpose of the indexed buckets: the per-bucket count and scatter
        // passes read contiguous references from it instead of striding across rows, keeping
        // each bucket's dense key range L2-resident for its atomics.
        auto const transpose_bytes = indexed_row_count * sizeof(Row);
        if (auto const allocation =
                cuda_try(cudaMallocAsync(&transposed, transpose_bytes, stream.get()));
            !allocation) {
            return fail(allocation.error());
        }
        constexpr uint32_t transpose_tile = 32U;
        dim3 const transpose_grid(
            (compatibility.indexed_bucket_count + transpose_tile - 1U) / transpose_tile,
            (reference_count + transpose_tile - 1U) / transpose_tile
        );
        detail::transpose_indexed_scores_kernel<<<
            transpose_grid,
            dim3(transpose_tile, transpose_tile),
            0,
            stream.get()>>>(
            static_cast<Row const*>(database.rows_),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(BucketCount),
            transposed
        );
        if (auto const launch = cuda_try(cudaGetLastError()); !launch) {
            return fail(launch.error());
        }

        // A small wave-blocked grid sweeps the buckets in rounds so the in-flight buckets'
        // dense key slices and cursor ranges stay L2-resident; one bucket per resident wave
        // would leave the SMs idle, and one block per bucket thrashes the cache (the profiled
        // failure mode of the first bucket-major version).
        constexpr uint32_t build_wave_blocks = 64;
        constexpr uint32_t build_bucket_block_size = 1024;
        auto const bucket_blocks =
            std::min<uint32_t>(compatibility.indexed_bucket_count, build_wave_blocks);
        // Both bucket kernels keep a quarter-key table in dynamic shared memory (64 KiB for
        // 16-bit keys, 32 KiB for 15-bit keys, over the default 48 KiB cap). The required size
        // depends on this build's key width, so the limit is (re)configured before every launch.
        auto const key_count = static_cast<uint32_t>(compatibility.key_mask) + 1U;
        auto const bucket_smem_bytes = static_cast<size_t>(key_count / 4U) * sizeof(uint32_t);
        if (auto const attribute = cuda_try(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(detail::count_index_cells_bucket_kernel<Row>),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(bucket_smem_bytes)
            ));
            !attribute) {
            return fail(attribute.error());
        }
        if (auto const attribute = cuda_try(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(detail::scatter_index_postings_bucket_kernel<Row>),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(bucket_smem_bytes)
            ));
            !attribute) {
            return fail(attribute.error());
        }

        // The count flush writes every cell of every bucket slice exactly once (plain stores
        // from the owning block), so the offsets array needs no separate zeroing before the
        // exclusive scan.
        detail::count_index_cells_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed,
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            database.index_offsets_
        );
        if (auto const launch = cuda_try(cudaGetLastError()); !launch) {
            return fail(launch.error());
        }
        if (auto const scan = cuda_try(
                cub::DeviceScan::ExclusiveSum(
                    scratch,
                    scan_bytes,
                    database.index_offsets_,
                    static_cast<int64_t>(cell_count + 1U),
                    stream.get()
                )
            );
            !scan) {
            return fail(scan.error());
        }

        detail::scatter_index_postings_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed,
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            database.index_offsets_,
            database.index_postings_
        );
        if (auto const launch = cuda_try(cudaGetLastError()); !launch) {
            return fail(launch.error());
        }
        if (auto const release = cuda_try(cudaFreeAsync(transposed, stream.get())); !release) {
            return Err(release.error());
        }
        transposed = nullptr;
        if (auto const release = cuda_try(cudaFreeAsync(scratch, stream.get())); !release) {
            return Err(release.error());
        }

        database.indexed_ = true;
        return Result<reference_database>::ok(std::move(database));
    }

    void destroy() noexcept {
        if (index_postings_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(index_postings_));
            index_postings_ = nullptr;
        }
        if (index_offsets_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(index_offsets_));
            index_offsets_ = nullptr;
        }
        if (saturation_states_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(saturation_states_));
            saturation_states_ = nullptr;
        }
        if (rows_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(rows_));
            rows_ = nullptr;
        }
    }

    void* rows_ = nullptr;
    uint32_t* saturation_states_ = nullptr;
    uint32_t* index_offsets_ = nullptr;
    uint32_t* index_postings_ = nullptr;
    size_t index_posting_capacity_{};
    reference_database_metadata metadata_{};
    bool packed_{};
    bool indexed_{};
};

}  // namespace cuddl
