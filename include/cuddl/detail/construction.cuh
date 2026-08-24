#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Grid width for direct global construction, capped at two inputs per thread.
__host__ inline uint32_t construction_blocks(size_t count, cudaStream_t stream = nullptr) noexcept {
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

/// @brief Launches construction of a sketch.
///
/// Sketches that fit in default static shared memory are built through a CTA-local shared winner
/// array with an exact count fix-up and merge (`add_shared_kernel`); larger sketches fall back to
/// the direct global packed-CAS kernel.
template <size_t BucketCount>
__host__ inline Result<void> launch_construction(
    device_span<uint64_t const> input,
    device_span<uint32_t> registers,
    uint32_t& saturation,
    cudaStream_t stream
) {
    if (input.empty()) {
        return {};
    }
    if constexpr (BucketCount <= shared_construction_max_buckets) {
        auto const vector_input = (reinterpret_cast<uintptr_t>(input.data()) & 31U) == 0U;
        // Two CTAs per SM keep the whole grid resident in one wave; the kernel's
        // runtime grid-stride loop keeps the work balanced for every input size.
        auto const capacity = static_cast<size_t>(shared_construction_block_size) * 4U;
        auto const needed = input.size() / capacity + (input.size() % capacity != 0U);
        uint32_t blocks = 1U;
        int device = 0;
        if (cudaGetDevice(&device) == cudaSuccess) {
            int multiprocessors = 0;
            if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) ==
                    cudaSuccess &&
                multiprocessors > 0) {
                blocks = static_cast<uint32_t>(multiprocessors) * 2U;
            }
        }
        blocks = static_cast<uint32_t>(
            cuda::std::min<size_t>(blocks, cuda::std::max<size_t>(1U, needed))
        );
        add_shared_kernel<BucketCount><<<blocks, shared_construction_block_size, 0, stream>>>(
            input.data(), input.size(), registers.data(), saturation, vector_input
        );
        return cuda_try(cudaGetLastError());
    }
    add_kernel<BucketCount><<<construction_blocks(input.size(), stream), block_size, 0, stream>>>(
        input.data(), input.size(), registers.data(), saturation
    );
    return cuda_try(cudaGetLastError());
}

}  // namespace cuddl::detail
