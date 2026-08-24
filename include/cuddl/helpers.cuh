#pragma once

#include <cuda_runtime.h>

#include <cuda/std/bit>
#include <cuda/std/concepts>

#include <cstddef>
#include <cstdint>

#include <cuddl/cuda_error.hpp>

namespace cuddl::detail {

/**
 * @brief Loads 256 bits from global memory using the non-coherent cache path.
 *
 * @tparam T Element type (uint32_t or uint64_t)
 * @param ptr Source pointer (must be 32-byte aligned)
 * @param out Output array (4 elements for uint64_t, 8 for uint32_t)
 */
template <typename T>
__device__ __forceinline__ void load_256_global_nc(const T* ptr, T* out) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "T must be uint32_t or uint64_t");

#if __CUDA_ARCH__ >= 1000
    if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];"
                     : "=l"(out[0]), "=l"(out[1]), "=l"(out[2]), "=l"(out[3])
                     : "l"(ptr));
    } else {
        asm volatile("ld.global.nc.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(out[0]),
                       "=r"(out[1]),
                       "=r"(out[2]),
                       "=r"(out[3]),
                       "=r"(out[4]),
                       "=r"(out[5]),
                       "=r"(out[6]),
                       "=r"(out[7])
                     : "l"(ptr));
    }
#else
    if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.nc.v2.u64 {%0, %1}, [%2];" : "=l"(out[0]), "=l"(out[1]) : "l"(ptr));
        asm volatile("ld.global.nc.v2.u64 {%0, %1}, [%2];"
                     : "=l"(out[2]), "=l"(out[3])
                     : "l"(ptr + 2));
    } else {
        asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                     : "l"(ptr));
        asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                     : "l"(ptr + 4));
    }
#endif
}

}  // namespace cuddl::detail
