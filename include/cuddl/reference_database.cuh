#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cstddef>
#include <limits>
#include <utility>

#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl {

/// @brief Construction parameters that must match between compact score rows.
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

/// @brief Recorded metadata for an immutable compact reference database.
struct reference_database_metadata {
    score_compatibility compatibility{};
    uint32_t reference_count{};

    friend bool
    operator==(reference_database_metadata const&, reference_database_metadata const&) = default;
};

/// @brief Stable reference ID and exact query-relative pairwise summary.
struct reference_search_result {
    uint32_t reference_id{};
    pairwise_summary summary{};

    friend bool operator==(reference_search_result const&, reference_search_result const&) = default;
};

namespace detail {

template <uint32_t K, size_t BucketCount>
[[nodiscard]] inline Result<void>
validate_score_compatibility(score_compatibility const& compatibility) {
    if (compatibility.kmer_length != K) {
        return Err(Error::invalid_argument("k-mer length does not match the database type"));
    }
    if (compatibility.bucket_count != BucketCount) {
        return Err(Error::invalid_argument("bucket count does not match the database type"));
    }
    if (compatibility.indexed_bucket_count != BucketCount) {
        return Err(Error::invalid_argument("compact exhaustive search requires every bucket"));
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
    if (compatibility.key_mask != std::numeric_limits<uint16_t>::max()) {
        return Err(Error::invalid_argument("compact exhaustive search requires full 16-bit scores"));
    }
    return Ok();
}

}  // namespace detail

/**
 * @brief Non-owning, trivially copyable view of contiguous compact reference rows.
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
    using result_type = reference_search_result;

    __host__ __device__ constexpr reference_database_ref(
        device_span<score_type const> rows, reference_database_metadata metadata
    ) noexcept
        : rows_(rows), metadata_(metadata) {}

    [[nodiscard]] __host__ __device__ constexpr device_span<score_type const> data() const noexcept {
        return rows_;
    }

    [[nodiscard]] constexpr reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] constexpr uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return static_cast<size_t>(reference_count) * BucketCount * sizeof(score_type);
    }

    [[nodiscard]] constexpr size_t persistent_row_bytes() const noexcept {
        return rows_.size_bytes();
    }

    [[nodiscard]] static constexpr size_t
    single_query_workspace_bytes(uint32_t) noexcept {
        return 0;
    }

    [[nodiscard]] constexpr size_t single_query_workspace_bytes() const noexcept {
        return 0;
    }

    [[nodiscard]] static constexpr uint32_t
    single_query_result_count(uint32_t reference_count) noexcept {
        return reference_count;
    }

    [[nodiscard]] constexpr uint32_t single_query_result_count() const noexcept {
        return metadata_.reference_count;
    }

    /// @brief Compares one compact query row with every reference row on @p stream.
    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto const expected_scores =
            static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (rows_.size() != expected_scores ||
            (expected_scores != 0U && rows_.data() == nullptr)) {
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

        constexpr uint32_t warp_width = 32;
        constexpr uint32_t references_per_block = detail::block_size / warp_width;
        auto const blocks =
            (metadata_.reference_count + references_per_block - 1U) / references_per_block;
        detail::exhaustive_search_kernel<BucketCount><<<
            blocks,
            detail::block_size,
            0,
            stream.get()>>>(rows_, metadata_.reference_count, query, results.data());
        return cuda_try(cudaGetLastError());
    }

   private:
    device_span<score_type const> rows_;
    reference_database_metadata metadata_;
};

/**
 * @brief Move-only owner of one immutable contiguous compact reference database.
 *
 * Building enqueues the row copy on the supplied stream. The input and returned database must
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
    using result_type = typename ref_type::result_type;

    reference_database(reference_database const&) = delete;
    reference_database& operator=(reference_database const&) = delete;

    reference_database(reference_database&& other) noexcept
        : rows_(other.rows_), metadata_(other.metadata_) {
        other.rows_ = nullptr;
        other.metadata_ = {};
    }

    reference_database& operator=(reference_database&& other) noexcept {
        if (this != &other) {
            destroy();
            rows_ = other.rows_;
            metadata_ = other.metadata_;
            other.rows_ = nullptr;
            other.metadata_ = {};
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
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount>(compatibility);
            !validation) {
            return Err(validation.error());
        }
        if (rows.size() % BucketCount != 0U) {
            return Err(Error::invalid_argument("score extent must contain complete rows"));
        }
        if (!rows.empty() && rows.data() == nullptr) {
            return Err(Error::invalid_argument("score rows must be device accessible"));
        }
        auto const reference_count = rows.size() / BucketCount;
        if (reference_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("reference count exceeds stable 32-bit IDs"));
        }

        reference_database database;
        database.metadata_ = {
            .compatibility = compatibility,
            .reference_count = static_cast<uint32_t>(reference_count),
        };
        if (rows.empty()) {
            return Result<reference_database>::ok(std::move(database));
        }
        if (auto const allocation =
                cuda_try(cudaMalloc(&database.rows_, rows.size_bytes()));
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
        return Result<reference_database>::ok(std::move(database));
    }

    /// @brief Thrust overload for flat row-major device storage.
    [[nodiscard]] static Result<reference_database> build_async(
        thrust::device_vector<score_type> const& rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()}, compatibility, stream
        );
    }

    [[nodiscard]] ref_type ref() const noexcept {
        return ref_type(
            {rows_, static_cast<size_t>(metadata_.reference_count) * BucketCount}, metadata_
        );
    }

    [[nodiscard]] reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    [[nodiscard]] uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return ref_type::persistent_row_bytes(reference_count);
    }

    [[nodiscard]] size_t persistent_row_bytes() const noexcept {
        return ref().persistent_row_bytes();
    }

    [[nodiscard]] static constexpr size_t
    single_query_workspace_bytes(uint32_t reference_count) noexcept {
        return ref_type::single_query_workspace_bytes(reference_count);
    }

    [[nodiscard]] size_t single_query_workspace_bytes() const noexcept {
        return ref().single_query_workspace_bytes();
    }

    [[nodiscard]] static constexpr uint32_t
    single_query_result_count(uint32_t reference_count) noexcept {
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

   private:
    reference_database() = default;

    void destroy() noexcept {
        if (rows_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(rows_));
            rows_ = nullptr;
        }
    }

    score_type* rows_ = nullptr;
    reference_database_metadata metadata_{};
};

}  // namespace cuddl
