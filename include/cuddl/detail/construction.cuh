#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Grid width for direct global construction, capped at two inputs per thread.
__host__ inline uint32_t
construction_blocks(size_t count, cudaStream_t stream = nullptr) noexcept {
    int device = 0;
    (void)stream;
    cudaGetDevice(&device);
    int multiprocessors = 0;
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) !=
        cudaSuccess) {
        multiprocessors = 0;
    }
    auto const blocks = static_cast<size_t>(multiprocessors) * 16U;
    auto const required = count / (block_size * 2U) + (count % (block_size * 2U) != 0U);
    return static_cast<uint32_t>(cuda::std::max<size_t>(1U, cuda::std::min(blocks, required)));
}

/// @brief Launches direct global packed-CAS construction of a sketch.
template <size_t BucketCount>
__host__ inline Result<void> launch_construction(
    uint64_t const* first,
    size_t const count,
    uint32_t* const registers,
    uint32_t* const saturation,
    cudaStream_t stream
) {
    if (count == 0U) {
        return Ok();
    }
    add_kernel<BucketCount><<<construction_blocks(count, stream), block_size, 0, stream>>>(
        first, count, registers, saturation
    );
    return cuda_try(cudaGetLastError());
}

}  // namespace cuddl::detail
