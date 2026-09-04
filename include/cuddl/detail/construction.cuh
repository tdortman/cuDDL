#pragma once

#include <cuda_runtime.h>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Launches construction of a sketch.
///
/// Sketches that fit in default static shared memory are built through a CTA-local shared winner
/// array with an exact count fix-up and merge (`add_shared_kernel`); larger sketches fall back to
/// the direct global packed-CAS kernel.
template <size_t BucketCount, typename Layout = default_register_layout>
__host__ inline Result<void> launch_construction(
    device_span<uint64_t const> input,
    device_span<uint32_t> registers,
    uint32_t& saturation,
    cuda::stream_ref stream
) {
    if (input.empty()) {
        return {};
    }
    return cuda_try([&] {
        auto const multiprocessors =
            stream.device().attribute(cuda::device_attributes::multiprocessor_count);
        if constexpr (BucketCount <= shared_construction_max_buckets) {
            auto const vector_input = (reinterpret_cast<uintptr_t>(input.data()) & 31U) == 0U;
            // Two CTAs per SM; the grid-stride loop covers the remaining inputs.
            auto const capacity = static_cast<size_t>(shared_construction_block_size) * 4U;
            auto const needed = input.size() / capacity + (input.size() % capacity != 0U);
            auto const blocks =
                static_cast<uint32_t>(cuda::std::min<size_t>(multiprocessors * 2U, needed));
            add_shared_kernel<BucketCount, Layout>
                <<<blocks, shared_construction_block_size, 0, stream.get()>>>(
                    input.data(), input.size(), registers.data(), saturation, vector_input
                );

        } else {
            auto const capacity = static_cast<size_t>(block_size) * 2U;
            auto const needed = input.size() / capacity + (input.size() % capacity != 0U);
            auto const blocks =
                static_cast<uint32_t>(cuda::std::min<size_t>(multiprocessors * 16U, needed));
            add_kernel<BucketCount, Layout><<<blocks, block_size, 0, stream.get()>>>(
                input.data(), input.size(), registers.data(), saturation
            );
        }
        return cudaGetLastError();
    });
}

}  // namespace cuddl::detail
