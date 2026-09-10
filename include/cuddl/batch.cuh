#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>

/**
 * @brief Batch operations over contiguous device rows.
 *
 * Every function takes flat row-major spans and runs the whole batch in one kernel launch, so a
 * collection of N rows costs one launch instead of N. Two row layouts appear below:
 *
 * - compact rows hold `BucketCount` packed `uint32_t` registers, which is what
 *   `compare_batch_async` consumes;
 * - stored sketches hold `BucketCount` packed registers followed by the sketch's saturation
 *   word, the layout of a single sketch allocation and of the rows the streamed tile builder
 *   writes. `detail::stored_sketch_words` gives that row width.
 *
 * Inputs and outputs must remain valid until @p stream completes.
 */

namespace cuddl {

/**
 * @brief Compares corresponding rows from two contiguous packed-register batches.
 *
 * The input spans are row-major and must contain the same whole number of BucketCount rows.
 * @p outputs must hold at least one summary per row pair. Inputs and outputs must remain valid
 * until @p stream completes. The whole batch runs in one kernel launch.
 */
template <size_t BucketCount>
[[nodiscard]] Result<void> compare_batch_async(
    device_span<uint32_t const> left_rows,
    device_span<uint32_t const> right_rows,
    device_span<pairwise_summary> outputs,
    cuda::stream_ref stream
) {
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1U)) == 0U);

    if (left_rows.size() != right_rows.size()) {
        return Err(Error::invalid_argument("batch register spans must have equal sizes"));
    }
    if (left_rows.size() % BucketCount != 0U) {
        return Err(Error::invalid_argument("batch register spans must contain whole rows"));
    }
    auto const pair_count = left_rows.size() / BucketCount;
    if (outputs.size() < pair_count) {
        return Err(Error::invalid_argument("batch output span is too small"));
    }
    if (pair_count != 0U && (left_rows.data() == nullptr || right_rows.data() == nullptr ||
                             outputs.data() == nullptr)) {
        return Err(Error::invalid_argument("nonempty batch buffers must not be null"));
    }
    if (pair_count == 0U) {
        return Ok();
    }

    constexpr size_t warps_per_block = detail::block_size / 32U;
    constexpr size_t maximum_blocks = 65535U;
    auto const blocks =
        std::min((pair_count + warps_per_block - 1U) / warps_per_block, maximum_blocks);
    detail::batch_summary_kernel<BucketCount>
        <<<static_cast<uint32_t>(blocks), detail::block_size, 0, stream.get()>>>(
            left_rows.data(), right_rows.data(), pair_count, outputs.data()
        );
    return cuda_try(cudaGetLastError());
}

/**
 * @brief Computes the cardinality of every stored sketch.
 *
 * @p empty_out receives the empty-register count per row and @p estimates_out the cardinality
 * estimate per row.
 */
template <size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> cardinality_batch_async(
    device_span<uint32_t const> sketches,
    device_span<uint64_t> empty_out,
    device_span<double> estimates_out,
    cuda::stream_ref stream
) {
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1U)) == 0U);

    if (sketches.size() % detail::stored_sketch_words<BucketCount> != 0U) {
        return Err(Error::invalid_argument("sketch store must contain whole stored sketches"));
    }
    auto const row_count = sketches.size() / detail::stored_sketch_words<BucketCount>;
    if (empty_out.size() < row_count || estimates_out.size() < row_count) {
        return Err(Error::invalid_argument("sketch store outputs are too small"));
    }
    if (row_count == 0U) {
        return Ok();
    }
    if (sketches.data() == nullptr || empty_out.data() == nullptr ||
        estimates_out.data() == nullptr) {
        return Err(Error::invalid_argument("nonempty sketch store buffers must not be null"));
    }
    if (row_count > detail::maximum_batch_rows) {
        return Err(Error::invalid_argument("sketch store has too many rows for one launch"));
    }

    detail::batch_cardinality_kernel<BucketCount, Layout>
        <<<static_cast<uint32_t>(row_count), detail::block_size, 0, stream.get()>>>(
            sketches.data(),
            static_cast<uint32_t>(row_count),
            empty_out.data(),
            estimates_out.data()
        );
    return cuda_try(cudaGetLastError());
}

/**
 * @brief Extracts the winner counts and saturation flag of every stored sketch.
 *
 * @p counts_out receives `BucketCount` `uint16_t` counts per row and @p saturation_out one
 * `uint32_t` per row.
 */
template <size_t BucketCount>
[[nodiscard]] inline Result<void> winner_counts_batch_async(
    device_span<uint32_t const> sketches,
    device_span<uint16_t> counts_out,
    device_span<uint32_t> saturation_out,
    cuda::stream_ref stream
) {
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1U)) == 0U);

    if (sketches.size() % detail::stored_sketch_words<BucketCount> != 0U) {
        return Err(Error::invalid_argument("sketch store must contain whole stored sketches"));
    }
    auto const row_count = sketches.size() / detail::stored_sketch_words<BucketCount>;
    if (counts_out.size() < row_count * BucketCount || saturation_out.size() < row_count) {
        return Err(Error::invalid_argument("sketch store outputs are too small"));
    }
    if (row_count == 0U) {
        return Ok();
    }
    if (sketches.data() == nullptr || counts_out.data() == nullptr ||
        saturation_out.data() == nullptr) {
        return Err(Error::invalid_argument("nonempty sketch store buffers must not be null"));
    }
    if (row_count > detail::maximum_batch_rows) {
        return Err(Error::invalid_argument("sketch store has too many rows for one launch"));
    }

    detail::batch_winner_counts_kernel<BucketCount>
        <<<static_cast<uint32_t>(row_count), detail::block_size, 0, stream.get()>>>(
            sketches.data(),
            static_cast<uint32_t>(row_count),
            counts_out.data(),
            saturation_out.data()
        );
    return cuda_try(cudaGetLastError());
}

/**
 * @brief Copies the winning score of every register into compact row-major scores.
 *
 * @p scores_out holds one `uint16_t`-per-register row per stored sketch.
 */
template <size_t BucketCount>
[[nodiscard]] inline Result<void> extract_scores_batch_async(
    device_span<uint32_t const> sketches,
    device_span<uint16_t> scores_out,
    cuda::stream_ref stream
) {
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1U)) == 0U);

    if (sketches.size() % detail::stored_sketch_words<BucketCount> != 0U) {
        return Err(Error::invalid_argument("sketch store must contain whole stored sketches"));
    }
    auto const row_count = sketches.size() / detail::stored_sketch_words<BucketCount>;
    if (scores_out.size() < row_count * BucketCount) {
        return Err(Error::invalid_argument("sketch store outputs are too small"));
    }
    if (row_count == 0U) {
        return Ok();
    }
    if (sketches.data() == nullptr || scores_out.data() == nullptr) {
        return Err(Error::invalid_argument("nonempty sketch store buffers must not be null"));
    }
    if (row_count > detail::maximum_batch_rows) {
        return Err(Error::invalid_argument("sketch store has too many rows for one launch"));
    }

    detail::batch_scores_kernel<BucketCount>
        <<<static_cast<uint32_t>(row_count), detail::block_size, 0, stream.get()>>>(
            sketches.data(), static_cast<uint32_t>(row_count), scores_out.data()
        );
    return cuda_try(cudaGetLastError());
}

/**
 * @brief Copies every stored sketch's registers into compact row-major rows.
 *
 * @p packed_out holds one `uint32_t`-per-register row per stored sketch; the saturation words
 * stay behind in the store.
 */
template <size_t BucketCount>
[[nodiscard]] inline Result<void> extract_packed_rows_batch_async(
    device_span<uint32_t const> sketches,
    device_span<uint32_t> packed_out,
    cuda::stream_ref stream
) {
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1U)) == 0U);

    if (sketches.size() % detail::stored_sketch_words<BucketCount> != 0U) {
        return Err(Error::invalid_argument("sketch store must contain whole stored sketches"));
    }
    auto const row_count = sketches.size() / detail::stored_sketch_words<BucketCount>;
    if (packed_out.size() < row_count * BucketCount) {
        return Err(Error::invalid_argument("sketch store outputs are too small"));
    }
    if (row_count == 0U) {
        return Ok();
    }
    if (sketches.data() == nullptr || packed_out.data() == nullptr) {
        return Err(Error::invalid_argument("nonempty sketch store buffers must not be null"));
    }
    if (row_count > detail::maximum_batch_rows) {
        return Err(Error::invalid_argument("sketch store has too many rows for one launch"));
    }

    detail::batch_packed_rows_kernel<BucketCount>
        <<<static_cast<uint32_t>(row_count), detail::block_size, 0, stream.get()>>>(
            sketches.data(), static_cast<uint32_t>(row_count), packed_out.data()
        );
    return cuda_try(cudaGetLastError());
}

}  // namespace cuddl
