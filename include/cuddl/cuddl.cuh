#pragma once

#include <cuddl/cuda_error.hpp>
#include <cuddl/cuddl_ref.cuh>
#include <cuddl/hashutil.cuh>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuddl {
namespace detail {

constexpr uint32_t block_size = 256;
constexpr uint64_t seed = 42;

__device__ inline uint16_t score(uint64_t const hash) noexcept {
    auto const exponent = static_cast<uint16_t>(min(__clzll(hash | 1ULL), 62));
    auto const mantissa = static_cast<uint16_t>((hash >> 43) & 0x03ffULL);
    return static_cast<uint16_t>(1U + (exponent << 10U) + (mantissa ^ 0x03ffU));
}

__device__ inline void atomic_max_u16(uint16_t* const address, uint16_t const value) {
    auto const raw_address = reinterpret_cast<uintptr_t>(address);
    auto* const word = reinterpret_cast<unsigned int*>(raw_address & ~uintptr_t{3});
    auto const shift = static_cast<unsigned int>((raw_address & 2U) * 8U);
    auto observed = *word;
    while (value > ((observed >> shift) & 0xffffU)) {
        auto const replacement =
            (observed & ~(0xffffU << shift)) | (static_cast<unsigned int>(value) << shift);
        auto const previous = atomicCAS(word, observed, replacement);
        if (previous == observed) {
            break;
        }
        observed = previous;
    }
}

template <size_t BucketCount>
__global__ void
add_kernel(uint64_t const* const first, size_t const count, uint16_t* const registers) {
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        auto const hash = cuddl::detail::splitmix64(first[index] ^ seed);
        atomic_max_u16(registers + (hash & (BucketCount - 1)), score(hash));
    }
}

template <size_t BucketCount>
__global__ void compare_kernel(
    uint16_t const* const left,
    uint16_t const* const right,
    pairwise_counts* const output
) {
    using block_reduce = cub::BlockReduce<pairwise_counts, block_size>;
    __shared__ typename block_reduce::TempStorage storage;

    pairwise_counts local{};
    for (auto bucket = threadIdx.x; bucket < BucketCount; bucket += blockDim.x) {
        auto const lhs = left[bucket];
        auto const rhs = right[bucket];
        if ((lhs | rhs) != 0U) {
            local.lower += lhs < rhs;
            local.equal += lhs == rhs;
            local.higher += lhs > rhs;
        }
    }
    auto const total =
        block_reduce(storage).Reduce(local, [] __device__(auto a, auto b) { return a + b; });
    if (threadIdx.x == 0) {
        *output = total;
    }
}

struct cuda_deleter {
    void operator()(uint16_t* pointer) const noexcept {
        if (pointer != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(pointer));
        }
    }
};

}  // namespace detail

template <uint32_t K, size_t BucketCount = 2048>
class sketch {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount != 0 && (BucketCount & (BucketCount - 1)) == 0);

   public:
    using ref_type = sketch_ref<K, BucketCount>;
    using register_type = typename ref_type::register_type;

    explicit sketch(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        register_type* storage = nullptr;
        CUDDL_CUDA_CALL(cudaMalloc(&storage, BucketCount * sizeof(register_type)));
        registers_.reset(storage);
        clear_async(stream);
        stream.sync();
    }

    sketch(sketch const&) = delete;
    sketch& operator=(sketch const&) = delete;
    sketch(sketch&&) = default;
    sketch& operator=(sketch&&) = default;

    [[nodiscard]] ref_type ref() const noexcept {
        return ref_type{registers_.get()};
    }

    void clear_async(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        CUDDL_CUDA_CALL(
            cudaMemsetAsync(registers_.get(), 0, BucketCount * sizeof(register_type), stream.get())
        );
    }

    void clear(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        clear_async(stream);
        stream.sync();
    }

    void add_async(
        uint64_t const* first,
        uint64_t const* last,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        auto const count = static_cast<size_t>(last - first);
        if (count == 0) {
            return;
        }
        int device = 0;
        CUDDL_CUDA_CALL(cudaGetDevice(&device));
        int multiprocessors = 0;
        CUDDL_CUDA_CALL(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device)
        );
        auto const blocks = static_cast<unsigned int>(multiprocessors * 4);
        detail::add_kernel<BucketCount>
            <<<blocks, detail::block_size, 0, stream.get()>>>(first, count, registers_.get());
        CUDDL_CUDA_CALL(cudaGetLastError());
    }

    void add(
        uint64_t const* first,
        uint64_t const* last,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) {
        add_async(first, last, stream);
        stream.sync();
    }

    void compare_async(
        ref_type other,
        pairwise_counts* output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        detail::compare_kernel<BucketCount>
            <<<1, detail::block_size, 0, stream.get()>>>(registers_.get(), other.data(), output);
        CUDDL_CUDA_CALL(cudaGetLastError());
    }

    [[nodiscard]] pairwise_counts compare(
        ref_type other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        register_type* storage = nullptr;
        CUDDL_CUDA_CALL(cudaMalloc(&storage, sizeof(pairwise_counts)));
        std::unique_ptr<register_type, detail::cuda_deleter> device_output{storage};
        compare_async(other, reinterpret_cast<pairwise_counts*>(device_output.get()), stream);
        pairwise_counts result{};
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            &result, device_output.get(), sizeof(result), cudaMemcpyDeviceToHost, stream.get()
        ));
        stream.sync();
        return result;
    }

   private:
    std::unique_ptr<register_type, detail::cuda_deleter> registers_;
};

}  // namespace cuddl
