#pragma once

#include <cuda/std/cstdint>

namespace cuddl::detail {

/// @brief Maps one DNA byte to its 2-bit symbol (A=0, C=1, T=2, G=3) or `0xff` when invalid.
///
/// Case-insensitive; any ambiguous byte (including 'N') is invalid.
/// The `__host__ __device__` annotation only exists under the CUDA compiler so plain
/// host translation units (e.g. via `cuddl/fastx.hpp`) see a normal inline function.
#ifdef __CUDACC__
__host__ __device__
#endif
inline constexpr uint8_t encode_base(char base) noexcept {
    auto const byte = static_cast<uint8_t>(base);
    auto const upper = static_cast<uint8_t>(byte & 0xDFu);
    auto const x = (byte >> 1u) & 3u;
    auto const valid =
        static_cast<uint8_t>((upper == 'A') | (upper == 'C') | (upper == 'G') | (upper == 'T'));
    auto const mask = static_cast<uint8_t>(0u - valid);
    return static_cast<uint8_t>((x & mask) | (0xFFu & ~mask));
}

}  // namespace cuddl::detail
