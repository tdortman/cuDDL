#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>

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

}  // namespace cuddl
