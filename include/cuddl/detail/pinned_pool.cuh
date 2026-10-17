#pragma once

#include <cuda_runtime.h>
#include <cuda/devices>
#include <cuda/memory_pool>

#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include <cuddl/cuda_error.hpp>

namespace cuddl::detail {

/// @brief Where a @ref device_pinned_pool takes its pages.
enum class pinned_placement : uint8_t {
    /// The device's host NUMA node, where the system supports NUMA pools: for buffers the device
    /// writes, such as downloaded results.
    device_node,
    /// Any node: for buffers host threads fill. Sketch loaders run on every node, and on an A40
    /// node with 16 NUMA nodes and two GPUs per node, device-node loader buffers sketched
    /// 100,000 genomes in 18.6 s against 15.1 s.
    host,
};

/// @brief A page-locked memory pool accessible from @p device only.
///
/// CCCL's default pinned pool is accessible from every visible device, so every allocation is
/// mapped into all of them. With one MPI rank per GPU and all 8 GPUs of an A40 node visible to
/// every rank, a 4-8 MB allocation took 27 ms instead of 1.3 ms, and sketch loaders spent most
/// of their time waiting on the driver. Downloads into it with 8 ranks took 1.3 s, against
/// 0.39 s from a device-node pool.
///
/// One pool per device and placement for the life of the process, like CUDA's default pools:
/// never destroying it keeps it valid for buffers that live until process exit.
[[nodiscard]] inline cuda::pinned_memory_pool_ref
device_pinned_pool(cuda::device_ref device, pinned_placement placement) {
    static std::mutex mutex;
    static std::vector<std::array<cudaMemPool_t, 2>> pools(cuda::devices.size());
    std::lock_guard lock(mutex);
    auto& pool = pools.at(static_cast<size_t>(device.get()))[static_cast<size_t>(placement)];
    if (pool == nullptr) {
        int node = -1;
        CUDDL_CUDA_CALL(cudaDeviceGetAttribute(&node, cudaDevAttrHostNumaId, device.get()));
        bool const numa =
            placement == pinned_placement::device_node && node >= 0 &&
            device.attribute(cuda::device_attributes::host_numa_memory_pools_supported) != 0;
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.location = numa ? cudaMemLocation{cudaMemLocationTypeHostNuma, node}
                                   : cudaMemLocation{cudaMemLocationTypeHost, 0};
        cudaMemPool_t created = nullptr;
        CUDDL_CUDA_CALL(cudaMemPoolCreate(&created, &properties));
        cudaMemAccessDesc const access{
            {cudaMemLocationTypeDevice, device.get()}, cudaMemAccessFlagsProtReadWrite
        };
        CUDDL_CUDA_CALL(cudaMemPoolSetAccess(created, &access, 1));
        // Pinning is the expensive part of an allocation: with the default threshold of zero,
        // every synchronisation returns freed pages to the OS and the next allocation pins them
        // again, which cost sketch loaders more time than inflating their files.
        auto keep = std::numeric_limits<uint64_t>::max();
        CUDDL_CUDA_CALL(cudaMemPoolSetAttribute(created, cudaMemPoolAttrReleaseThreshold, &keep));
        pool = created;
    }
    return cuda::pinned_memory_pool_ref(pool);
}

/// @brief Returns the free pages of a @ref device_pinned_pool to the OS when it leaves scope.
///
/// The pool keeps freed pages so work that allocates and frees pinned buffers repeatedly pins
/// them once; a guard around that work keeps the process from holding them afterwards. A trim
/// that fails only leaves the pages pinned, so its status is not reported.
class pinned_pool_trim {
   public:
    pinned_pool_trim(cuda::device_ref device, pinned_placement placement)
        : pool_(device_pinned_pool(device, placement)) {}
    pinned_pool_trim(pinned_pool_trim const&) = delete;
    pinned_pool_trim& operator=(pinned_pool_trim const&) = delete;
    ~pinned_pool_trim() {
        (void)cudaMemPoolTrimTo(pool_.get(), 0);
    }

   private:
    cuda::pinned_memory_pool_ref pool_;
};

}  // namespace cuddl::detail
