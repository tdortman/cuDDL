// Pinned host-to-device and device-to-host copy bandwidth, one JSON line per host memory kind,
// direction, and size. "cudaMallocHost" is driver-allocated pinned memory; "registered" is
// 4096-byte-aligned aligned_alloc memory with transparent huge pages off, registered with
// cudaHostRegister, which is how cuddl::detail::file_writer allocates its write buffers.
//
//   gpu-copy-bandwidth [copies]     default 30 timed copies per row, after 3 warm-up copies

#include <cuda_runtime_api.h>

#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void measure(char const* kind, char const* direction, size_t bytes, int copies, void* pinned, void* device, cudaStream_t stream) {
    bool const to_host = std::string(direction) == "d2h";
    auto const copy = [&] {
        check(to_host ? cudaMemcpyAsync(pinned, device, bytes, cudaMemcpyDeviceToHost, stream)
                      : cudaMemcpyAsync(device, pinned, bytes, cudaMemcpyHostToDevice, stream));
    };
    for (int i = 0; i < 3; ++i) {
        copy();
    }
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin{}, end{};
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    std::vector<double> gbps;
    for (int i = 0; i < copies; ++i) {
        check(cudaEventRecord(begin, stream));
        copy();
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float ms = 0.0F;
        check(cudaEventElapsedTime(&ms, begin, end));
        gbps.push_back(static_cast<double>(bytes) / 1e9 / (static_cast<double>(ms) * 1e-3));
    }
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
    std::sort(gbps.begin(), gbps.end());
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    std::printf(
        "{\"gpu\":\"%s\",\"host_memory\":\"%s\",\"direction\":\"%s\",\"bytes\":%zu,\"copies\":%d,\"median_gbps\":%.2f,"
        "\"min_gbps\":%.2f,\"max_gbps\":%.2f}\n",
        props.name, kind, direction, bytes, copies, gbps[gbps.size() / 2], gbps.front(), gbps.back()
    );
}

}  // namespace

int main(int argc, char** argv) {
    int const copies = argc > 1 ? std::atoi(argv[1]) : 30;
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    for (size_t const bytes : {size_t{1} << 27, size_t{1} << 30}) {
        void* device = nullptr;
        check(cudaMalloc(&device, bytes));
        check(cudaMemset(device, 1, bytes));
        {
            void* pinned = nullptr;
            check(cudaMallocHost(&pinned, bytes));
            std::memset(pinned, 1, bytes);
            for (char const* direction : {"d2h", "h2d"}) {
                measure("cudaMallocHost", direction, bytes, copies, pinned, device, stream);
            }
            check(cudaFreeHost(pinned));
        }
        {
            void* registered = std::aligned_alloc(4096, bytes);
            if (registered == nullptr) {
                throw std::bad_alloc();
            }
            static_cast<void>(madvise(registered, bytes, MADV_NOHUGEPAGE));
            std::memset(registered, 1, bytes);
            check(cudaHostRegister(registered, bytes, cudaHostRegisterPortable));
            for (char const* direction : {"d2h", "h2d"}) {
                measure("registered", direction, bytes, copies, registered, device, stream);
            }
            check(cudaHostUnregister(registered));
            std::free(registered);
        }
        check(cudaFree(device));
    }
    check(cudaStreamDestroy(stream));
    return 0;
}
