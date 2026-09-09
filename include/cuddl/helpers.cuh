#pragma once

#include <cuda_runtime.h>

#include <cuda/std/bit>
#include <cuda/std/concepts>

#include <cstddef>
#include <cstdint>

#include <cuddl/cuda_error.hpp>

namespace cuddl::detail {

/**
 * @brief Byte alignment @ref load_256_global_nc requires on the compiled architecture.
 *
 * sm_100 and later issue one 256-bit load per call, which needs 32-byte alignment. Earlier
 * architectures issue two 128-bit loads, which need only 16 bytes, so a 16-byte-aligned row
 * still takes the wide path there.
 */
__host__ __device__ constexpr uint32_t load_256_alignment() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return 32U;
#else
    return 16U;
#endif
}

/// @brief True when both row bases meet @ref load_256_alignment on the compiled architecture.
template <typename QueryScore, typename ReferenceScore>
__host__ __device__ inline bool
wide_rows_aligned(QueryScore const* query, ReferenceScore const* reference) noexcept {
    constexpr uintptr_t mask = load_256_alignment() - 1U;
    return (reinterpret_cast<uintptr_t>(query) & mask) == 0U &&
           (reinterpret_cast<uintptr_t>(reference) & mask) == 0U;
}

/**
 * @brief Loads 256 bits from global memory using the non-coherent cache path.
 *
 * @tparam T Element type (uint32_t or uint64_t)
 * @param ptr Source pointer (must be aligned to @ref load_256_alignment)
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
