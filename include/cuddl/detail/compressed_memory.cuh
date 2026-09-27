#pragma once

#include <cuda.h>
#include <cuda/buffer>
#include <cuda/memory_resource>

#include <algorithm>
#include <limits>
#include <memory>

#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief CUDA VMM allocations with transparent hardware compression.
/// Deallocation waits for the allocation stream because VMM unmapping is synchronous.
class compressed_memory_resource {
    template <typename Function>
    static Function entry(char const* name) {
        void* address = nullptr;
        cudaDriverEntryPointQueryResult result{};
        auto const status =
            cudaGetDriverEntryPointByVersion(name, &address, 12000U, cudaEnableDefault, &result);
        if (status != cudaSuccess) {
            throw cuda::cuda_error(status, name);
        }
        if (result != cudaDriverEntryPointSuccess || address == nullptr) {
            throw cuda::cuda_error(cudaErrorNotSupported, name);
        }
        return reinterpret_cast<Function>(address);
    }

    struct driver {
        decltype(&cuDeviceGetAttribute) attribute =
            entry<decltype(attribute)>("cuDeviceGetAttribute");
        decltype(&cuMemGetAllocationGranularity) granularity =
            entry<decltype(granularity)>("cuMemGetAllocationGranularity");
        decltype(&cuMemCreate) create = entry<decltype(create)>("cuMemCreate");
        decltype(&cuMemRelease) release = entry<decltype(release)>("cuMemRelease");
        decltype(&cuMemAddressReserve) reserve = entry<decltype(reserve)>("cuMemAddressReserve");
        decltype(&cuMemAddressFree) free = entry<decltype(free)>("cuMemAddressFree");
        decltype(&cuMemMap) map = entry<decltype(map)>("cuMemMap");
        decltype(&cuMemUnmap) unmap = entry<decltype(unmap)>("cuMemUnmap");
        decltype(&cuMemSetAccess) access = entry<decltype(access)>("cuMemSetAccess");
    };

    static driver const& api() {
        static driver const functions;
        return functions;
    }

    static void check(CUresult status, char const* operation) {
        if (status != CUDA_SUCCESS) {
            throw cuda::cuda_error(static_cast<cudaError_t>(status), operation);
        }
    }

    [[nodiscard]] CUmemAllocationProp properties() const noexcept {
        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device_;
        properties.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        return properties;
    }

    [[nodiscard]] size_t allocation_size(size_t bytes) const {
        if (bytes > std::numeric_limits<size_t>::max() - (granularity_ - 1U)) {
            throw std::bad_alloc{};
        }
        return (bytes + granularity_ - 1U) / granularity_ * granularity_;
    }

    int device_;
    size_t granularity_ = 0U;

   public:
    using default_queries = cuda::mr::properties_list<cuda::mr::device_accessible>;

    explicit compressed_memory_resource(cuda::device_ref device) : device_(device.get()) {
        auto const props = properties();
        check(
            api().granularity(&granularity_, &props, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity"
        );
    }

    [[nodiscard]] static bool supported(cuda::device_ref device) {
        // Kernel stores can initialize compressed memory from Hopper onward.
        if (device.attribute(cuda::device_attributes::compute_capability_major) < 9) return false;
        int supported = 0;
        check(
            api().attribute(
                &supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device.get()
            ),
            "cuDeviceGetAttribute"
        );
        return supported != 0;
    }

    [[nodiscard]] void* allocate_sync(size_t bytes, size_t alignment) {
        auto const size = allocation_size(bytes);
        auto const props = properties();
        CUmemGenericAllocationHandle handle{};
        check(api().create(&handle, size, &props, 0U), "cuMemCreate");
        auto release_handle = [](CUmemGenericAllocationHandle* handle) noexcept {
            cuda_abort_on_error(static_cast<cudaError_t>(api().release(*handle)));
        };
        std::unique_ptr<CUmemGenericAllocationHandle, decltype(release_handle)> physical{
            &handle, release_handle
        };
        CUdeviceptr address{};
        check(
            api().reserve(&address, size, std::max(alignment, granularity_), 0U, 0U),
            "cuMemAddressReserve"
        );
        auto free_address = [size](void* pointer) noexcept {
            cuda_abort_on_error(
                static_cast<cudaError_t>(api().free(reinterpret_cast<CUdeviceptr>(pointer), size))
            );
        };
        std::unique_ptr<void, decltype(free_address)> reserved{
            reinterpret_cast<void*>(address), free_address
        };
        check(api().map(address, size, 0U, handle, 0U), "cuMemMap");
        auto unmap_address = [size](void* pointer) noexcept {
            cuda_abort_on_error(
                static_cast<cudaError_t>(api().unmap(reinterpret_cast<CUdeviceptr>(pointer), size))
            );
        };
        std::unique_ptr<void, decltype(unmap_address)> mapped{
            reinterpret_cast<void*>(address), unmap_address
        };
        CUmemAccessDesc access{};
        access.location = props.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(api().access(address, size, &access, 1U), "cuMemSetAccess");
        mapped.release();
        return reserved.release();
    }

    void deallocate_sync(void* pointer, size_t bytes, size_t) noexcept {
        auto const size = allocation_size(bytes);
        auto const address = reinterpret_cast<CUdeviceptr>(pointer);
        cuda_abort_on_error(static_cast<cudaError_t>(api().unmap(address, size)));
        cuda_abort_on_error(static_cast<cudaError_t>(api().free(address, size)));
    }

    [[nodiscard]] void* allocate(cuda::stream_ref, size_t bytes, size_t alignment) {
        return allocate_sync(bytes, alignment);
    }

    void
    deallocate(cuda::stream_ref stream, void* pointer, size_t bytes, size_t alignment) noexcept {
        cuda_abort_on_error(cudaStreamSynchronize(stream.get()));
        deallocate_sync(pointer, bytes, alignment);
    }

    friend constexpr void
    get_property(compressed_memory_resource const&, cuda::mr::device_accessible) noexcept {}
    friend bool operator==(compressed_memory_resource const&, compressed_memory_resource const&) =
        default;
};

/// @brief Allocates compressed device storage when the device supports it, ordinary storage
/// otherwise.
template <typename T>
[[nodiscard]] cuda::device_buffer<T> make_compressed_buffer(cuda::stream_ref stream, size_t size) {
    if (compressed_memory_resource::supported(stream.device())) {
        return cuda::make_buffer<T>(
            stream, compressed_memory_resource{stream.device()}, size, cuda::no_init
        );
    }
    return cuda::make_device_buffer<T>(stream, stream.device(), size, cuda::no_init);
}

}  // namespace cuddl::detail
