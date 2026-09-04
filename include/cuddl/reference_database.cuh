#pragma once

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>

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

/// @brief Dense offsets favor query latency; sparse sorted keys reduce index memory and build time.
enum class index_storage { dense, sparse };

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
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static constexpr score_compatibility current() noexcept {
        static_assert(BucketCount <= std::numeric_limits<uint32_t>::max());
        return {
            .kmer_length = K,
            .bucket_count = static_cast<uint32_t>(BucketCount),
            .indexed_bucket_count = static_cast<uint32_t>(BucketCount),
            .score_encoder_identity = 1U,
            .exponent_bits = static_cast<uint16_t>(Layout::exponent_bits),
            .mantissa_bits = static_cast<uint16_t>(Layout::mantissa_bits),
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

struct sparse_index_key {
    uint16_t mask;
    template <typename Row>
    __host__ __device__ uint16_t operator()(Row row) const noexcept {
        auto const score = reference_score(row);
        return score == 0U || mask == 0xffffU ? score : static_cast<uint16_t>((score & mask) + 1U);
    }
};
struct sparse_reference_id {
    uint32_t reference_count;
    __host__ __device__ uint32_t operator()(uint32_t i) const noexcept {
        return i % reference_count;
    }
};
struct sparse_segment_offset {
    uint32_t reference_count;
    __host__ __device__ int64_t operator()(uint32_t bucket) const noexcept {
        return static_cast<int64_t>(bucket) * reference_count;
    }
};

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_score_compatibility(
    score_compatibility const& compatibility
) {
    if (compatibility.kmer_length != K) {
        return Err(Error::invalid_argument("k-mer length does not match the database type"));
    }
    if (compatibility.bucket_count != BucketCount) {
        return Err(Error::invalid_argument("bucket count does not match the database type"));
    }
    if (compatibility.score_encoder_identity == 0U ||
        compatibility.exponent_bits != Layout::exponent_bits ||
        compatibility.mantissa_bits != Layout::mantissa_bits) {
        return Err(Error::invalid_argument("score encoding does not match the register layout"));
    }
    if (compatibility.hash_identity == 0U) {
        return Err(Error::invalid_argument("hash identity must be specified"));
    }
    if (compatibility.canonicalisation_policy == 0U) {
        return Err(Error::invalid_argument("canonicalisation policy must be specified"));
    }
    return Ok();
}

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_non_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount, Layout>(compatibility);
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

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount, Layout>(compatibility);
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

constexpr size_t indexed_batch_pair_storage_bytes = 200U * 1024U * 1024U;
constexpr uint32_t batch_query_tile_count = 128U;

[[nodiscard]] constexpr uint32_t
batch_query_tile_size(uint32_t reference_count, uint32_t query_count) noexcept {
    if (reference_count == 0U || query_count == 0U) {
        return 0U;
    }
    auto const count_capacity = std::numeric_limits<uint32_t>::max() / reference_count;
    return std::min(query_count, std::min(batch_query_tile_count, std::max(1U, count_capacity)));
}

[[nodiscard]] constexpr uint32_t
indexed_batch_query_tile_size(uint32_t reference_count, uint32_t query_count) noexcept {
    constexpr size_t bytes_per_pair = 2U * sizeof(uint32_t);
    auto const pair_capacity = indexed_batch_pair_storage_bytes / bytes_per_pair;
    auto const query_capacity = pair_capacity / reference_count;
    auto const bounded_capacity = query_capacity == 0U ? size_t{1} : query_capacity;
    return static_cast<uint32_t>(
        static_cast<size_t>(query_count) < bounded_capacity ? query_count : bounded_capacity
    );
}

}  // namespace detail

namespace detail {

/**
 * @brief Non-owning, trivially copyable view of one selected reference-row backing.
 *
 * Inputs, database rows, workspace, and results must remain valid until the supplied stream
 * completes.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class reference_database_view {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using score_type = uint16_t;
    using register_type = uint32_t;
    using layout_type = Layout;
    using result_type = reference_search_result;
    using batch_result_type = batch_search_result;

    __host__ __device__ constexpr reference_database_view(
        device_span<score_type const> rows,
        reference_database_metadata metadata,
        device_span<uint32_t const> index_offsets = {},
        device_span<uint32_t const> index_postings = {},
        bool indexed = false,
        device_span<uint16_t const> index_keys = {}
    ) noexcept
        : rows_(rows),
          metadata_(metadata),
          index_offsets_(index_offsets),
          index_postings_(index_postings),
          index_keys_(index_keys),
          indexed_(indexed) {}

    __host__ __device__ constexpr reference_database_view(
        device_span<register_type const> packed_rows,
        device_span<uint32_t const> saturation_states,
        reference_database_metadata metadata,
        device_span<uint32_t const> index_offsets = {},
        device_span<uint32_t const> index_postings = {},
        bool indexed = false,
        device_span<uint16_t const> index_keys = {}
    ) noexcept
        : packed_rows_(packed_rows),
          saturation_states_(saturation_states),
          metadata_(metadata),
          index_offsets_(index_offsets),
          index_postings_(index_postings),
          index_keys_(index_keys),
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
        return index_offsets_.size_bytes() + index_postings_.size_bytes() +
               index_keys_.size_bytes();
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
    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes(
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_index_storage());
        if (metadata_.reference_count == 0U) {
            return size_t{0};
        }

        size_t selection_bytes = 0;
        auto const ids = cuda::make_counting_iterator(uint32_t{0});
        auto const selection = cuda_try(
            cub::DeviceSelect::If(
                nullptr,
                selection_bytes,
                ids,
                static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(metadata_.reference_count),
                detail::minimum_match_predicate{nullptr, 1U},
                stream.get()
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

    /// @brief Storage reused while exhaustively searching @p query_count compact rows.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        auto const pair_count = CUDDL_TRY(dense_batch_pair_count(
            detail::batch_query_tile_size(metadata_.reference_count, query_count)
        ));
        return make_batch_requirements(pair_count, pair_count, false, stream);
    }

    /// @brief Storage reused while searching @p query_count compact rows through the index.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        CUDDL_TRY(validate_index_storage());
        auto const pair_count = CUDDL_TRY(dense_batch_pair_count(
            detail::batch_query_tile_size(metadata_.reference_count, query_count)
        ));
        return make_batch_requirements(pair_count, pair_count, true, stream);
    }

    /// @brief Storage reused while exhaustively searching every unique database-row pair.
    [[nodiscard]] Result<cuddl::batch_search_requirements> all_to_all_search_requirements(
        cuda::stream_ref stream
    ) const {
        return all_to_all_tile_search_requirements(
            0U,
            detail::batch_query_tile_size(metadata_.reference_count, metadata_.reference_count),
            stream
        );
    }

    [[nodiscard]] static constexpr uint32_t all_to_all_result_capacity(
        uint32_t reference_count
    ) noexcept {
        if (reference_count < 2U) {
            return 0U;
        }
        auto const query_count = detail::batch_query_tile_size(reference_count, reference_count);
        auto const pair_count = static_cast<uint64_t>(query_count) *
                                (2ULL * reference_count - query_count - 1ULL) / 2ULL;
        return static_cast<uint32_t>(pair_count);
    }

    /// @brief Storage reused while searching every unique database-row pair through the index.
    [[nodiscard]] Result<cuddl::batch_search_requirements> indexed_all_to_all_search_requirements(
        cuda::stream_ref stream
    ) const {
        return indexed_all_to_all_tile_search_requirements(
            0U,
            detail::batch_query_tile_size(metadata_.reference_count, metadata_.reference_count),
            stream
        );
    }

    /// @brief Compares one compact query row with every reference row on @p stream.
    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream
    ) const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        if (query.size() != BucketCount || query.data() == nullptr) {
            return Err(Error::invalid_argument("query must contain one complete score row"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
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
        indexed_search_options options,
        cuda::stream_ref stream
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
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
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
            return write_batch_result_count(metadata_.reference_count, result_count, stream);
        }
        if (metadata_.reference_count == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }

        auto const required_workspace = CUDDL_TRY(indexed_single_query_workspace_bytes(stream));
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

        CUDDL_CUDA_TRY(
            cuda::fill_bytes(
                stream,
                cuda::std::span{match_counts, static_cast<size_t>(metadata_.reference_count)},
                0
            )
        );
        auto const indexed_bucket_count = metadata_.compatibility.indexed_bucket_count;
        constexpr uint32_t warp_width = 32;
        constexpr uint32_t warps_per_block = detail::block_size / warp_width;
        auto const bucket_blocks =
            static_cast<uint32_t>((indexed_bucket_count + warps_per_block - 1U) / warps_per_block);
        detail::count_index_matches_kernel<BucketCount>
            <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                query.data(),
                index_offsets_.data(),
                index_postings_.data(),
                indexed_bucket_count,
                metadata_.compatibility.key_mask,
                match_counts,
                index_keys_.data(),
                metadata_.reference_count
            );
        CUDDL_CUDA_TRY(cudaGetLastError());

        auto const ids = cuda::make_counting_iterator(uint32_t{0});
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

    /**
     * @brief Exhaustively searches compact queries using bounded reusable storage.
     *
     * @p on_tile receives each tile's exact result capacity. Before returning, it must consume
     * the tile synchronously or enqueue dependent work on @p stream because the supplied storage
     * is reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        static_cast<void>(workspace);
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(batch_search_requirements(query_count, stream));
        CUDDL_TRY(
            validate_batch_outputs(requirements, results, result_count, result_match_counts, true)
        );
        auto const tile_size =
            detail::batch_query_tile_size(metadata_.reference_count, query_count);
        if (tile_size == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        for (uint32_t first_query_id = 0U; first_query_id < query_count;
             first_query_id += tile_size) {
            auto const tile_query_count = std::min(tile_size, query_count - first_query_id);
            auto const tile_requirements = CUDDL_TRY(make_batch_requirements(
                CUDDL_TRY(dense_batch_pair_count(tile_query_count)),
                CUDDL_TRY(dense_batch_pair_count(tile_query_count)),
                false,
                stream
            ));
            if (packed_) {
                CUDDL_TRY(
                    launch_batch_exhaustive<false>(
                        queries,
                        first_query_id,
                        tile_query_count,
                        query_id_offset + first_query_id,
                        packed_rows_,
                        tile_requirements.maximum_pair_count,
                        results,
                        result_count,
                        result_match_counts,
                        stream
                    )
                );
            } else {
                CUDDL_TRY(
                    launch_batch_exhaustive<false>(
                        queries,
                        first_query_id,
                        tile_query_count,
                        query_id_offset + first_query_id,
                        rows_,
                        tile_requirements.maximum_pair_count,
                        results,
                        result_count,
                        result_match_counts,
                        stream
                    )
                );
            }
            on_tile(tile_requirements.maximum_pair_count);
        }
        return Ok();
    }

    /**
     * @brief Indexed search for compact queries using bounded reusable storage.
     *
     * @p on_tile receives each tile's maximum result count. Before returning, it must consume the
     * tile synchronously or enqueue dependent work on @p stream because the supplied storage is
     * reused by the next tile. The actual result count is written to @p result_count.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_indexed_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(indexed_batch_search_requirements(query_count, stream));
        CUDDL_TRY(validate_indexed_batch_inputs(
            requirements, workspace, results, result_count, result_match_counts, options
        ));
        auto const tile_size =
            detail::batch_query_tile_size(metadata_.reference_count, query_count);
        if (tile_size == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        for (uint32_t first_query_id = 0U; first_query_id < query_count;
             first_query_id += tile_size) {
            auto const tile_query_count = std::min(tile_size, query_count - first_query_id);
            auto const tile_requirements = CUDDL_TRY(make_batch_requirements(
                CUDDL_TRY(dense_batch_pair_count(tile_query_count)),
                CUDDL_TRY(dense_batch_pair_count(tile_query_count)),
                true,
                stream
            ));
            if (packed_) {
                CUDDL_TRY(
                    launch_batch_indexed<false>(
                        queries,
                        first_query_id,
                        tile_query_count,
                        query_id_offset + first_query_id,
                        packed_rows_,
                        tile_requirements,
                        workspace,
                        results,
                        result_count,
                        result_match_counts,
                        options,
                        stream
                    )
                );
            } else {
                CUDDL_TRY(
                    launch_batch_indexed<false>(
                        queries,
                        first_query_id,
                        tile_query_count,
                        query_id_offset + first_query_id,
                        rows_,
                        tile_requirements,
                        workspace,
                        results,
                        result_count,
                        result_match_counts,
                        options,
                        stream
                    )
                );
            }
            on_tile(tile_requirements.maximum_pair_count);
        }
        return Ok();
    }

    /**
     * @brief Exhaustively searches every unique database-row pair using bounded storage.
     *
     * @p on_tile is called after each internal tile is enqueued with its exact result count.
     * Before returning, it must consume the tile synchronously or enqueue dependent work on
     * @p stream because the supplied storage is reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        auto const tile_size =
            detail::batch_query_tile_size(metadata_.reference_count, metadata_.reference_count);
        if (tile_size == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        for (uint32_t first_query_id = 0U; first_query_id < metadata_.reference_count;
             first_query_id += tile_size) {
            auto const query_count =
                std::min(tile_size, metadata_.reference_count - first_query_id);
            auto const requirements =
                CUDDL_TRY(all_to_all_tile_search_requirements(first_query_id, query_count, stream));
            CUDDL_TRY(search_all_to_all_tile_async(
                first_query_id,
                query_count,
                workspace,
                results,
                result_count,
                result_match_counts,
                stream
            ));
            on_tile(requirements.maximum_pair_count);
        }
        return Ok();
    }

    /**
     * @brief Indexed search of every unique database-row pair using bounded storage.
     *
     * @p on_tile is called after each internal tile is enqueued with its maximum result count.
     * Before returning, it must consume the tile synchronously or enqueue dependent work on
     * @p stream because the supplied storage is reused by the next tile. The actual result count
     * is written to @p result_count.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        auto const tile_size =
            detail::batch_query_tile_size(metadata_.reference_count, metadata_.reference_count);
        if (tile_size == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }
        for (uint32_t first_query_id = 0U; first_query_id < metadata_.reference_count;
             first_query_id += tile_size) {
            auto const query_count =
                std::min(tile_size, metadata_.reference_count - first_query_id);
            auto const requirements = CUDDL_TRY(
                indexed_all_to_all_tile_search_requirements(first_query_id, query_count, stream)
            );
            CUDDL_TRY(search_all_to_all_indexed_tile_async(
                first_query_id,
                query_count,
                workspace,
                results,
                result_count,
                result_match_counts,
                options,
                stream
            ));
            on_tile(requirements.maximum_pair_count);
        }
        return Ok();
    }

   private:
    [[nodiscard]] Result<cuddl::batch_search_requirements> all_to_all_tile_search_requirements(
        uint32_t first_query_id,
        uint32_t query_count,
        cuda::stream_ref stream
    ) const {
        auto const pair_count = CUDDL_TRY(all_to_all_pair_count(first_query_id, query_count));
        return make_batch_requirements(0U, pair_count, false, stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_all_to_all_tile_search_requirements(
        uint32_t first_query_id,
        uint32_t query_count,
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_index_storage());
        auto const dense_pair_count = CUDDL_TRY(dense_batch_pair_count(query_count));
        auto const result_pair_count =
            CUDDL_TRY(all_to_all_pair_count(first_query_id, query_count));
        return make_batch_requirements(dense_pair_count, result_pair_count, true, stream);
    }

    [[nodiscard]] Result<void> search_all_to_all_tile_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        static_cast<void>(workspace);
        CUDDL_TRY(validate_stored_rows());
        auto const requirements =
            CUDDL_TRY(all_to_all_tile_search_requirements(first_query_id, query_count, stream));
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

    [[nodiscard]] Result<void> search_all_to_all_indexed_tile_async(
        uint32_t first_query_id,
        uint32_t query_count,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_stored_rows());
        auto const requirements = CUDDL_TRY(
            indexed_all_to_all_tile_search_requirements(first_query_id, query_count, stream)
        );
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

    [[nodiscard]] Result<void> validate_stored_rows() const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        return Ok();
    }

    [[nodiscard]] Result<void> validate_index_storage() const {
        CUDDL_TRY((detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(
            metadata_.compatibility
        )));
        auto const cell_count = detail::indexed_cell_count(metadata_.compatibility);
        auto const expected_postings = static_cast<size_t>(
            detail::indexed_posting_count(metadata_.reference_count, metadata_.compatibility)
        );
        auto const sparse = index_offsets_.empty();
        auto const valid_keys =
            sparse ? index_keys_.size() == expected_postings &&
                         (expected_postings == 0U || index_keys_.data() != nullptr)
                   : index_keys_.empty() &&
                         index_offsets_.size() == static_cast<size_t>(cell_count + 1U) &&
                         index_offsets_.data() != nullptr;
        if (!indexed_ || !valid_keys || index_postings_.size() != expected_postings ||
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
        bool indexed,
        cuda::stream_ref stream
    ) const {
        cuddl::batch_search_requirements requirements{
            .maximum_pair_count = maximum_pair_count,
            .result_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(batch_result_type),
            .match_count_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(uint32_t),
        };
        if (!indexed || dense_pair_count == 0U) {
            return requirements;
        }

        auto const query_count = dense_pair_count / metadata_.reference_count;
        auto const tile_query_count =
            detail::indexed_batch_query_tile_size(metadata_.reference_count, query_count);
        auto const tile_pair_count = tile_query_count * metadata_.reference_count;
        size_t selection_bytes = 0;
        auto const ids = cuda::make_counting_iterator(uint32_t{0});
        auto const selection = cuda_try(
            cub::DeviceSelect::If(
                nullptr,
                selection_bytes,
                ids,
                static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(tile_pair_count),
                detail::batch_minimum_match_predicate{
                    nullptr, 0U, metadata_.reference_count, 0U, false
                },
                stream.get()
            )
        );
        if (!selection) {
            return Err(selection.error());
        }

        constexpr size_t scalar_count = 2U;
        requirements.counter_bytes =
            (static_cast<size_t>(tile_pair_count) + scalar_count) * sizeof(uint32_t);
        requirements.candidate_bytes = static_cast<size_t>(tile_pair_count) * sizeof(uint32_t);
        requirements.temporary_bytes = selection_bytes;
        constexpr size_t alignment_slack = alignof(uint32_t) - 1U + 255U;
        if (requirements.counter_bytes > std::numeric_limits<size_t>::max() - alignment_slack ||
            requirements.candidate_bytes >
                std::numeric_limits<size_t>::max() - alignment_slack - requirements.counter_bytes) {
            return Err(Error::resource("indexed batch workspace size overflows"));
        }
        auto const arrays_bytes = requirements.counter_bytes + requirements.candidate_bytes;
        if (requirements.temporary_bytes >
            std::numeric_limits<size_t>::max() - arrays_bytes - alignment_slack) {
            return Err(Error::resource("indexed batch workspace size overflows"));
        }
        requirements.workspace_bytes =
            arrays_bytes + alignment_slack + requirements.temporary_bytes;
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
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
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
        return cuda_try(
            cub::DeviceTransform::Transform(
                cuda::make_constant_iterator(count),
                result_count.data(),
                1,
                cuda::std::identity{},
                stream
            )
        );
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
                result_match_counts.empty() ? nullptr : result_match_counts.data(),
                result_count.data(),
                pair_count
            );
        return cuda_try(cudaGetLastError());
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
        auto* tile_candidate_count = reinterpret_cast<uint32_t*>(address);
        auto* write_offset = tile_candidate_count + 1U;
        auto* match_counts = write_offset + 1U;
        address += requirements.counter_bytes;
        address = detail::align_up(address, alignof(uint32_t));
        auto* candidate_ids = reinterpret_cast<uint32_t*>(address);
        address += requirements.candidate_bytes;
        address = detail::align_up(address, 256U);
        if (address > workspace_end) {
            return Err(Error::resource("indexed batch workspace layout exceeds its capacity"));
        }
        auto* temporary_workspace = reinterpret_cast<void*>(address);
        auto temporary_bytes = static_cast<size_t>(workspace_end - address);
        constexpr uint32_t warp_width = 32;
        constexpr uint32_t warps_per_block = detail::block_size / warp_width;
        auto const written_capacity = results.size() < requirements.maximum_pair_count
                                          ? results.size()
                                          : requirements.maximum_pair_count;
        auto const capacity = static_cast<uint32_t>(written_capacity);
        auto const ids = cuda::make_counting_iterator(uint32_t{0});
        auto const launch_tiles = [&](bool refine,
                                      uint32_t* result_offset,
                                      uint32_t const* required_count) -> Result<void> {
            auto const query_tile_size =
                detail::indexed_batch_query_tile_size(metadata_.reference_count, query_count);
            for (uint32_t first_query = 0U; first_query < query_count;
                 first_query += query_tile_size) {
                auto const remaining = query_count - first_query;
                auto const tile_query_count =
                    remaining < query_tile_size ? remaining : query_tile_size;
                auto const tile_pair_count = tile_query_count * metadata_.reference_count;
                CUDDL_CUDA_TRY(
                    cuda::fill_bytes(
                        stream,
                        cuda::std::span{match_counts, static_cast<size_t>(tile_pair_count)},
                        0
                    )
                );
                auto const query_buckets = static_cast<size_t>(tile_query_count) *
                                           metadata_.compatibility.indexed_bucket_count;
                auto const required_bucket_blocks =
                    (query_buckets + warps_per_block - 1U) / warps_per_block;
                auto const bucket_blocks = static_cast<uint32_t>(
                    required_bucket_blocks < 65535U ? required_bucket_blocks : 65535U
                );
                detail::count_batch_index_matches_kernel<BucketCount>
                    <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                        queries.data(),
                        query_row_offset + first_query,
                        tile_query_count,
                        index_offsets_.data(),
                        index_postings_.data(),
                        metadata_.reference_count,
                        metadata_.compatibility.indexed_bucket_count,
                        metadata_.compatibility.key_mask,
                        match_counts,
                        index_keys_.data()
                    );
                CUDDL_CUDA_TRY(cudaGetLastError());
                CUDDL_CUDA_TRY(
                    cub::DeviceSelect::If(
                        temporary_workspace,
                        temporary_bytes,
                        ids,
                        candidate_ids,
                        tile_candidate_count,
                        static_cast<int64_t>(tile_pair_count),
                        detail::batch_minimum_match_predicate{
                            match_counts,
                            options.minimum_matches,
                            metadata_.reference_count,
                            query_id_offset + first_query,
                            AllToAll
                        },
                        stream.get()
                    )
                );
                if (refine) {
                    constexpr uint32_t refinement_blocks = 128U;
                    detail::refine_batch_index_candidates_kernel<BucketCount>
                        <<<refinement_blocks, detail::block_size, 0, stream.get()>>>(
                            queries.data(),
                            query_row_offset + first_query,
                            query_id_offset + first_query,
                            rows.data(),
                            metadata_.reference_count,
                            match_counts,
                            candidate_ids,
                            tile_candidate_count,
                            result_offset,
                            required_count,
                            capacity,
                            results.data(),
                            result_match_counts.empty() ? nullptr : result_match_counts.data()
                        );
                    CUDDL_CUDA_TRY(cudaGetLastError());
                }
                detail::advance_indexed_result_count_kernel<<<1, 1, 0, stream.get()>>>(
                    tile_candidate_count, result_offset, required_count, capacity
                );
                CUDDL_CUDA_TRY(cudaGetLastError());
            }
            return Ok();
        };

        CUDDL_CUDA_TRY(cuda::fill_bytes(stream, result_count.first(1), 0));
        if (written_capacity == requirements.maximum_pair_count) {
            return launch_tiles(true, result_count.data(), nullptr);
        }
        CUDDL_TRY(launch_tiles(false, result_count.data(), nullptr));
        CUDDL_CUDA_TRY(cuda::fill_bytes(stream, cuda::std::span{write_offset, size_t{1}}, 0));
        if (query_count ==
            detail::indexed_batch_query_tile_size(metadata_.reference_count, query_count)) {
            // The first pass retained all candidates and match counts. The device-side
            // capacity guard preserves the no-partial-output contract on overflow.
            detail::refine_batch_index_candidates_kernel<BucketCount>
                <<<128U, detail::block_size, 0, stream.get()>>>(
                    queries.data(),
                    query_row_offset,
                    query_id_offset,
                    rows.data(),
                    metadata_.reference_count,
                    match_counts,
                    candidate_ids,
                    tile_candidate_count,
                    write_offset,
                    result_count.data(),
                    capacity,
                    results.data(),
                    result_match_counts.empty() ? nullptr : result_match_counts.data()
                );
            return cuda_try(cudaGetLastError());
        }
        // ponytail: batches exceeding the 200 MiB workspace replay selection; retaining
        // every tile would require additional caller-owned storage.
        return launch_tiles(true, write_offset, result_count.data());
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
    device_span<uint16_t const> index_keys_;
    bool packed_{};
    bool indexed_{};
};

}  // namespace detail

/**
 * @brief Move-only owner of one immutable contiguous reference database.
 *
 * Building enqueues row copies on the supplied stream. Inputs and the returned database must
 * remain alive until that stream completes. The allocation stream must outlive the database.
 * Complete work on other streams before destroying or move-assigning the database.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class reference_database {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using layout_type = Layout;
    using score_type = uint16_t;
    using register_type = uint32_t;
    using result_type = reference_search_result;
    using batch_result_type = batch_search_result;

    reference_database(reference_database const&) = delete;
    reference_database& operator=(reference_database const&) = delete;

    reference_database(reference_database&& other) noexcept
        : rows_(std::move(other.rows_)),
          saturation_states_(std::move(other.saturation_states_)),
          index_offsets_(std::move(other.index_offsets_)),
          index_postings_(std::move(other.index_postings_)),
          index_keys_(std::move(other.index_keys_)),
          index_posting_capacity_(std::exchange(other.index_posting_capacity_, 0)),
          metadata_(std::exchange(other.metadata_, {})),
          packed_(std::exchange(other.packed_, false)),
          indexed_(std::exchange(other.indexed_, false)) {}

    reference_database& operator=(reference_database&& other) noexcept {
        if (this != &other) {
            rows_ = std::move(other.rows_);
            saturation_states_ = std::move(other.saturation_states_);
            index_offsets_ = std::move(other.index_offsets_);
            index_postings_ = std::move(other.index_postings_);
            index_keys_ = std::move(other.index_keys_);
            index_posting_capacity_ = std::exchange(other.index_posting_capacity_, 0);
            metadata_ = std::exchange(other.metadata_, {});
            packed_ = std::exchange(other.packed_, false);
            indexed_ = std::exchange(other.indexed_, false);
        }
        return *this;
    }

    /// @brief Builds a database by copying flat row-major scores on @p stream.
    [[nodiscard]] static Result<reference_database> build_async(
        device_span<score_type const> rows,
        score_compatibility compatibility,
        cuda::stream_ref stream
    ) {
        return build_rows_async(rows, {}, compatibility, stream);
    }

    /// @brief Builds a multiplicity-preserving database from packed rows and saturation states.
    [[nodiscard]] static Result<reference_database> build_async(
        device_span<register_type const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream
    ) {
        return build_rows_async(rows, saturation_states, compatibility, stream);
    }

    /// @brief Builds compact rows and their retrieval index.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        device_span<score_type const> rows,
        score_compatibility compatibility,
        cuda::stream_ref stream,
        index_storage storage = index_storage::dense
    ) {
        return build_indexed_rows_async(rows, {}, compatibility, stream, storage);
    }

    /// @brief Builds packed rows and their winner-score retrieval index.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        device_span<register_type const> rows,
        device_span<uint32_t const> saturation_states,
        score_compatibility compatibility,
        cuda::stream_ref stream,
        index_storage storage = index_storage::dense
    ) {
        return build_indexed_rows_async(rows, saturation_states, compatibility, stream, storage);
    }

    [[nodiscard]] device_span<score_type const> data() const noexcept {
        return view().data();
    }

    /// @brief Packed winner/count rows, or an empty span for compact databases.
    [[nodiscard]] device_span<register_type const> packed_data() const noexcept {
        return view().packed_data();
    }

    /// @brief Per-reference saturation states retained with packed rows.
    [[nodiscard]] device_span<uint32_t const> saturation_states() const noexcept {
        return view().saturation_states();
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
        return view_type::persistent_row_bytes(reference_count);
    }

    [[nodiscard]] static constexpr size_t persistent_packed_row_bytes(
        uint32_t reference_count
    ) noexcept {
        return view_type::persistent_packed_row_bytes(reference_count);
    }

    [[nodiscard]] size_t persistent_row_bytes() const noexcept {
        return view().persistent_row_bytes();
    }

    [[nodiscard]] size_t persistent_index_bytes() const noexcept {
        return view().persistent_index_bytes();
    }

    [[nodiscard]] static constexpr size_t single_query_workspace_bytes(
        uint32_t reference_count
    ) noexcept {
        return view_type::single_query_workspace_bytes(reference_count);
    }

    [[nodiscard]] size_t single_query_workspace_bytes() const noexcept {
        return view().single_query_workspace_bytes();
    }

    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes(
        cuda::stream_ref stream
    ) const {
        return view().indexed_single_query_workspace_bytes(stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        return view().batch_search_requirements(query_count, stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        return view().indexed_batch_search_requirements(query_count, stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> all_to_all_search_requirements(
        cuda::stream_ref stream
    ) const {
        return view().all_to_all_search_requirements(stream);
    }

    [[nodiscard]] static constexpr uint32_t all_to_all_result_capacity(
        uint32_t reference_count
    ) noexcept {
        return view_type::all_to_all_result_capacity(reference_count);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> indexed_all_to_all_search_requirements(
        cuda::stream_ref stream
    ) const {
        return view().indexed_all_to_all_search_requirements(stream);
    }

    [[nodiscard]] static constexpr uint32_t single_query_result_count(
        uint32_t reference_count
    ) noexcept {
        return view_type::single_query_result_count(reference_count);
    }

    [[nodiscard]] uint32_t single_query_result_count() const noexcept {
        return view().single_query_result_count();
    }

    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream
    ) const {
        return view().search_async(query, query_compatibility, workspace, results, stream);
    }

    [[nodiscard]] Result<void> search_indexed_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        device_span<uint32_t> result_count,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        return view().search_indexed_async(
            query, query_compatibility, workspace, results, result_count, options, stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        return view().search_batch_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            result_count,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_indexed_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        return view().search_batch_indexed_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            result_count,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            options,
            stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        return view().search_all_to_all_async(
            workspace,
            results,
            result_count,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_count,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        indexed_search_options options,
        cuda::stream_ref stream
    ) const {
        return view().search_all_to_all_indexed_async(
            workspace,
            results,
            result_count,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            options,
            stream
        );
    }

   private:
    using view_type = detail::reference_database_view<K, BucketCount, Layout>;

    [[nodiscard]] view_type view() const noexcept {
        auto const offset_count = index_offsets_.size();
        auto const row_count = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        auto const offsets = device_span<uint32_t const>{index_offsets_.data(), offset_count};
        auto const postings =
            device_span<uint32_t const>{index_postings_.data(), index_posting_capacity_};
        if (packed_) {
            return view_type(
                {reinterpret_cast<register_type const*>(rows_.data()), row_count},
                {saturation_states_.data(), metadata_.reference_count},
                metadata_,
                offsets,
                postings,
                indexed_,
                index_keys_
            );
        }
        return view_type(
            {reinterpret_cast<score_type const*>(rows_.data()), row_count},
            metadata_,
            offsets,
            postings,
            indexed_,
            index_keys_
        );
    }

    explicit reference_database(cuda::stream_ref stream)
        : rows_(stream, cuda::device_default_memory_pool(stream.device())),
          saturation_states_(stream, cuda::device_default_memory_pool(stream.device())),
          index_offsets_(stream, cuda::device_default_memory_pool(stream.device())),
          index_postings_(stream, cuda::device_default_memory_pool(stream.device())),
          index_keys_(stream, cuda::device_default_memory_pool(stream.device())) {}

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
                    detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(
                        compatibility
                    );
                !validation) {
                return Err(validation.error());
            }
        } else {
            if (auto const validation =
                    detail::validate_non_indexed_score_compatibility<K, BucketCount, Layout>(
                        compatibility
                    );
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
        auto database = CUDDL_CUDA_TRY(reference_database(stream));
        database.metadata_ = {.compatibility = compatibility, .reference_count = reference_count};
        database.packed_ = std::is_same_v<Row, register_type>;
        if (!rows.empty()) {
            database.rows_ = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint8_t>(
                    stream, stream.device(), rows.size_bytes(), cuda::no_init
                )
            );
            CUDDL_CUDA_TRY(cuda::copy_bytes(stream, rows, database.rows_));
            if constexpr (std::is_same_v<Row, register_type>) {
                database.saturation_states_ = CUDDL_CUDA_TRY(
                    cuda::make_device_buffer<uint32_t>(stream, stream.device(), saturation_states)
                );
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
        cuda::stream_ref stream,
        index_storage storage
    ) {
        if (storage != index_storage::dense && storage != index_storage::sparse) {
            return Err(Error::invalid_argument("unsupported index storage"));
        }
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
        if (storage == index_storage::sparse) {
            CUDDL_TRY(database.template build_sparse_index<Row>(compatibility, *validated, stream));
            database.indexed_ = true;
            return Result<reference_database>::ok(std::move(database));
        }
        database.index_offsets_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), static_cast<size_t>(offset_count), cuda::no_init
            )
        );
        if (rows.empty()) {
            CUDDL_CUDA_TRY(cuda::fill_bytes(stream, database.index_offsets_, 0));
            database.indexed_ = true;
            return Result<reference_database>::ok(std::move(database));
        }
        database.index_postings_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), static_cast<size_t>(posting_capacity), cuda::no_init
            )
        );
        database.index_posting_capacity_ = static_cast<size_t>(posting_capacity);

        auto const indexed_row_count = static_cast<size_t>(posting_capacity);
        auto const reference_count = static_cast<uint32_t>(rows.size() / BucketCount);

        // Bucket-major transpose of the indexed buckets: the per-bucket count and scatter
        // passes read contiguous references from it instead of striding across rows, keeping
        // each bucket's dense key range L2-resident for its atomics.
        auto transposed = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<Row>(stream, stream.device(), indexed_row_count, cuda::no_init)
        );
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
            reinterpret_cast<Row const*>(database.rows_.data()),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(BucketCount),
            transposed.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());

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
        CUDDL_CUDA_TRY(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(detail::count_index_cells_bucket_kernel<Row>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bucket_smem_bytes)
        ));
        CUDDL_CUDA_TRY(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(detail::scatter_index_postings_bucket_kernel<Row>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bucket_smem_bytes)
        ));

        // The count flush writes every cell of every bucket slice exactly once (plain stores
        // from the owning block), so the offsets array needs no separate zeroing before the
        // exclusive scan.
        detail::count_index_cells_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed.data(),
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            database.index_offsets_.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        // CUB allocates and releases scan scratch on this stream through its pooled resource.
        CUDDL_CUDA_TRY(
            cub::DeviceScan::ExclusiveSum(
                database.index_offsets_.data(), static_cast<int64_t>(cell_count + 1U), stream
            )
        );

        detail::scatter_index_postings_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed.data(),
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            database.index_offsets_.data(),
            database.index_postings_.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        database.indexed_ = true;
        return Result<reference_database>::ok(std::move(database));
    }

    template <typename Row>
    [[nodiscard]] Result<void> build_sparse_index(
        score_compatibility const& compatibility,
        uint32_t reference_count,
        cuda::stream_ref stream
    ) {
        auto const size =
            static_cast<size_t>(detail::indexed_posting_count(reference_count, compatibility));
        index_posting_capacity_ = size;
        if (size == 0U) {
            return Ok();
        }
        auto const device = stream.device();
        index_keys_ =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint16_t>(stream, device, size, cuda::no_init));
        index_postings_ =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, size, cuda::no_init));
        auto keys =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint16_t>(stream, device, size, cuda::no_init));
        auto ids =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, size, cuda::no_init));
        auto transposed =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<Row>(stream, device, size, cuda::no_init));
        detail::transpose_indexed_scores_kernel<<<
            dim3((compatibility.indexed_bucket_count + 31U) / 32U, (reference_count + 31U) / 32U),
            dim3(32U, 32U),
            0,
            stream.get()>>>(
            reinterpret_cast<Row const*>(rows_.data()),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(BucketCount),
            transposed.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const mask = compatibility.key_mask;
        // Zero denotes an empty row. Folded 15-bit keys are shifted by one so that
        // a nonempty score masked to zero remains distinguishable from an empty row.
        CUDDL_CUDA_TRY(
            cub::DeviceTransform::Transform(
                transposed.data(), keys.data(), size, detail::sparse_index_key{mask}, stream
            )
        );
        CUDDL_CUDA_TRY(
            cub::DeviceTransform::Transform(
                cuda::make_counting_iterator(uint32_t{0}),
                ids.data(),
                size,
                detail::sparse_reference_id{reference_count},
                stream
            )
        );
        auto const segment_offsets = cuda::make_transform_iterator(
            cuda::make_counting_iterator(uint32_t{0}),
            detail::sparse_segment_offset{reference_count}
        );
        CUDDL_CUDA_TRY(
            cub::DeviceSegmentedSort::SortPairs(
                keys.data(),
                index_keys_.data(),
                ids.data(),
                index_postings_.data(),
                static_cast<int64_t>(size),
                static_cast<int64_t>(compatibility.indexed_bucket_count),
                segment_offsets,
                segment_offsets + 1,
                stream
            )
        );
        return Ok();
    }

    cuda::device_buffer<uint8_t> rows_;
    cuda::device_buffer<uint32_t> saturation_states_;
    cuda::device_buffer<uint32_t> index_offsets_;
    cuda::device_buffer<uint32_t> index_postings_;
    cuda::device_buffer<uint16_t> index_keys_;
    size_t index_posting_capacity_{};
    reference_database_metadata metadata_{};
    bool packed_{};
    bool indexed_{};
};

}  // namespace cuddl
