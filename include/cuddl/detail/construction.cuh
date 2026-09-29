#pragma once

#include <cuda_runtime.h>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Launches construction of a sketch through a CTA-local shared register array merged
/// into @p registers (`add_shared_kernel`).
template <size_t BucketCount, typename Layout = default_register_layout>
__host__ inline Result<void> launch_construction(
    device_span<uint64_t const> input,
    device_span<uint32_t> registers,
    cuda::stream_ref stream,
    cuda::std::optional<blacklist_view> blacklist = cuda::std::nullopt
) {
    if (input.empty()) {
        return {};
    }
    return cuda_try([&] {
        auto const multiprocessors =
            stream.device().attribute(cuda::device_attributes::multiprocessor_count);
        auto const vector_input = (reinterpret_cast<uintptr_t>(input.data()) & 31U) == 0U;
        // Two CTAs per SM; the grid-stride loop covers the remaining inputs.
        auto const capacity = static_cast<size_t>(shared_construction_block_size) * 4U;
        auto const needed = input.size() / capacity + (input.size() % capacity != 0U);
        auto const blocks =
            static_cast<uint32_t>(cuda::std::min<size_t>(multiprocessors * 2U, needed));
        auto launch = [&]<bool Filter>() {
            add_shared_kernel<BucketCount, Layout, 1U, Filter>
                <<<blocks, shared_construction_block_size, 0, stream.get()>>>(
                    input.data(),
                    input.size(),
                    registers.data(),
                    vector_input,
                    blacklist.value_or(blacklist_view{})
                );
        };
        if (blacklist) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
        return cudaGetLastError();
    });
}

}  // namespace cuddl::detail
