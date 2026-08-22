#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <thrust/iterator/counting_iterator.h>

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

/// @brief Per-search candidate threshold for the full 16-bit DDLIndex.
struct indexed_search_options {
    uint32_t minimum_matches = 5;
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

template <size_t BucketCount>
inline constexpr uint64_t indexed_cell_count =
    static_cast<uint64_t>(BucketCount) * (uint64_t{1} << 16U);

[[nodiscard]] inline uintptr_t align_up(uintptr_t address, size_t alignment) noexcept {
    return (address + alignment - 1U) & ~(static_cast<uintptr_t>(alignment) - 1U);
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

    [[nodiscard]] __host__ __device__ constexpr device_span<score_type const>
    data() const noexcept {
        return rows_;
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

    [[nodiscard]] constexpr size_t persistent_row_bytes() const noexcept {
        return rows_.size_bytes();
    }

    [[nodiscard]] constexpr size_t persistent_index_bytes() const noexcept {
        return index_offsets_.size_bytes() + index_postings_.size_bytes();
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

    /// @brief Caller-owned bytes required by one positive-threshold indexed query.
    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes() const {
        if (!indexed_) {
            return Err(Error::invalid_argument("database has no full 16-bit index"));
        }
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
        constexpr auto cell_count = detail::indexed_cell_count<BucketCount>;
        if (rows_.size() != expected_scores || (expected_scores != 0U && rows_.data() == nullptr)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        if (!indexed_ || index_offsets_.size() != static_cast<size_t>(cell_count + 1U) ||
            index_offsets_.data() == nullptr || index_postings_.size() != expected_scores ||
            (expected_scores != 0U && index_postings_.data() == nullptr)) {
            return Err(Error::invalid_argument("database has no valid full 16-bit index"));
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
        if (options.minimum_matches > BucketCount) {
            return Err(Error::invalid_argument("minimum matches exceeds the bucket count"));
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
        auto const bucket_blocks =
            static_cast<uint32_t>((BucketCount + detail::block_size - 1U) / detail::block_size);
        detail::count_index_matches_kernel<BucketCount>
            <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                query, index_offsets_, index_postings_, match_counts
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
        detail::refine_index_candidates_kernel<BucketCount>
            <<<refinement_blocks, detail::block_size, 0, stream.get()>>>(
                rows_, query, candidate_ids, result_count.data(), results.data()
            );
        return cuda_try(cudaGetLastError());
    }

   private:
    device_span<score_type const> rows_;
    reference_database_metadata metadata_;
    device_span<uint32_t const> index_offsets_;
    device_span<uint32_t const> index_postings_;
    bool indexed_{};
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
        : rows_(other.rows_),
          index_offsets_(other.index_offsets_),
          index_postings_(other.index_postings_),
          index_posting_capacity_(other.index_posting_capacity_),
          metadata_(other.metadata_),
          indexed_(other.indexed_) {
        other.rows_ = nullptr;
        other.index_offsets_ = nullptr;
        other.index_postings_ = nullptr;
        other.index_posting_capacity_ = 0;
        other.metadata_ = {};
        other.indexed_ = false;
    }

    reference_database& operator=(reference_database&& other) noexcept {
        if (this != &other) {
            destroy();
            rows_ = other.rows_;
            index_offsets_ = other.index_offsets_;
            index_postings_ = other.index_postings_;
            index_posting_capacity_ = other.index_posting_capacity_;
            metadata_ = other.metadata_;
            indexed_ = other.indexed_;
            other.rows_ = nullptr;
            other.index_offsets_ = nullptr;
            other.index_postings_ = nullptr;
            other.index_posting_capacity_ = 0;
            other.metadata_ = {};
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

    /// @brief Builds the immutable row store and its full 16-bit dense-offset index.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
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
        if (rows.size() > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("index postings exceed 32-bit offsets"));
        }

        constexpr auto cell_count = detail::indexed_cell_count<BucketCount>;
        if (cell_count + 1U > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return Err(Error::resource("dense index offset allocation overflows"));
        }

        auto built = build_async(rows, compatibility, stream);
        if (!built) {
            return Err(built.error());
        }
        auto database = std::move(*built);
        auto const offset_bytes = static_cast<size_t>(cell_count + 1U) * sizeof(uint32_t);
        if (auto const allocation = cuda_try(cudaMalloc(&database.index_offsets_, offset_bytes));
            !allocation) {
            return Err(allocation.error());
        }
        if (rows.empty()) {
            CUDDL_CUDA_TRY(cudaMemsetAsync(database.index_offsets_, 0, offset_bytes, stream.get()));
            database.indexed_ = true;
            return Result<reference_database>::ok(std::move(database));
        }

        if (auto const allocation =
                cuda_try(cudaMalloc(&database.index_postings_, rows.size() * sizeof(uint32_t)));
            !allocation) {
            return Err(allocation.error());
        }
        database.index_posting_capacity_ = rows.size();

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
        auto const cursor_offset = static_cast<size_t>(
            detail::align_up(static_cast<uintptr_t>(scan_bytes), alignof(uint32_t))
        );
        auto const cursor_bytes = static_cast<size_t>(cell_count) * sizeof(uint32_t);
        if (cursor_offset > std::numeric_limits<size_t>::max() - cursor_bytes) {
            return Err(Error::resource("index construction workspace size overflows"));
        }
        auto const scratch_bytes = cursor_offset + cursor_bytes;
        uint8_t* scratch = nullptr;
        if (auto const allocation =
                cuda_try(cudaMallocAsync(&scratch, scratch_bytes, stream.get()));
            !allocation) {
            return Err(allocation.error());
        }
        auto* cursors = reinterpret_cast<uint32_t*>(scratch + cursor_offset);
        auto fail = [&](Error const& error) -> Result<reference_database> {
            auto const release = cuda_try(cudaFreeAsync(scratch, stream.get()));
            if (!release) {
                return Err(release.error());
            }
            return Err(error);
        };
        // CUB histogram scratch scales with the full-score domain; direct counting keeps scratch
        // to the scan temporary storage and one cursor per cell.

        if (auto const clear_counts =
                cuda_try(cudaMemsetAsync(database.index_offsets_, 0, offset_bytes, stream.get()));
            !clear_counts) {
            return fail(clear_counts.error());
        }
        auto const blocks =
            static_cast<uint32_t>((rows.size() + detail::block_size - 1U) / detail::block_size);
        detail::count_index_cells_kernel<BucketCount>
            <<<blocks, detail::block_size, 0, stream.get()>>>(
                database.ref().data(), database.index_offsets_
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
        if (auto const copy = cuda_try(cudaMemcpyAsync(
                cursors,
                database.index_offsets_,
                cursor_bytes,
                cudaMemcpyDeviceToDevice,
                stream.get()
            ));
            !copy) {
            return fail(copy.error());
        }

        detail::scatter_index_postings_kernel<BucketCount>
            <<<blocks, detail::block_size, 0, stream.get()>>>(
                database.ref().data(), cursors, database.index_postings_
            );
        if (auto const launch = cuda_try(cudaGetLastError()); !launch) {
            return fail(launch.error());
        }
        if (auto const release = cuda_try(cudaFreeAsync(scratch, stream.get())); !release) {
            return Err(release.error());
        }

        database.indexed_ = true;
        return Result<reference_database>::ok(std::move(database));
    }

    /// @brief Thrust overload for indexed flat row-major device storage.
    [[nodiscard]] static Result<reference_database> build_indexed_async(
        thrust::device_vector<score_type> const& rows,
        score_compatibility compatibility,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        return build_indexed_async(
            {thrust::raw_pointer_cast(rows.data()), rows.size()}, compatibility, stream
        );
    }

    [[nodiscard]] ref_type ref() const noexcept {
        constexpr auto offset_count =
            static_cast<size_t>(detail::indexed_cell_count<BucketCount> + 1U);
        return ref_type(
            {rows_, static_cast<size_t>(metadata_.reference_count) * BucketCount},
            metadata_,
            {index_offsets_, indexed_ ? offset_count : 0U},
            {index_postings_, index_posting_capacity_},
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

    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return ref_type::persistent_row_bytes(reference_count);
    }

    [[nodiscard]] size_t persistent_row_bytes() const noexcept {
        return ref().persistent_row_bytes();
    }

    [[nodiscard]] size_t persistent_index_bytes() const noexcept {
        return ref().persistent_index_bytes();
    }

    [[nodiscard]] static constexpr size_t
    single_query_workspace_bytes(uint32_t reference_count) noexcept {
        return ref_type::single_query_workspace_bytes(reference_count);
    }

    [[nodiscard]] size_t single_query_workspace_bytes() const noexcept {
        return ref().single_query_workspace_bytes();
    }

    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes() const {
        return ref().indexed_single_query_workspace_bytes();
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

   private:
    reference_database() = default;

    void destroy() noexcept {
        if (index_postings_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(index_postings_));
            index_postings_ = nullptr;
        }
        if (index_offsets_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(index_offsets_));
            index_offsets_ = nullptr;
        }
        if (rows_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(rows_));
            rows_ = nullptr;
        }
    }

    score_type* rows_ = nullptr;
    uint32_t* index_offsets_ = nullptr;
    uint32_t* index_postings_ = nullptr;
    size_t index_posting_capacity_{};
    reference_database_metadata metadata_{};
    bool indexed_{};
};

}  // namespace cuddl
