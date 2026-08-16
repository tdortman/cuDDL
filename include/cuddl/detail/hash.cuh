#pragma once

#include <cuda/std/cstdint>


namespace cuddl::detail {

/// @brief SplitMix64 finalizer, applied to a packed k-mer XOR the domain seed.
__host__ __device__ constexpr uint64_t splitmix64(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

/// @brief Domain separation constant folded into every packed k-mer before hashing.
constexpr uint64_t seed = 42;

/// @brief Hashes a canonical packed k-mer into the value space used by DDL registers.
__host__ __device__ constexpr uint64_t hash_kmer(uint64_t packed_kmer) noexcept {
    return splitmix64(packed_kmer ^ seed);
}

/// @brief Selects the register bucket for @p hash given a power-of-two @p BucketCount.
template <size_t BucketCount>
__host__ __device__ constexpr size_t bucket_of(uint64_t hash) noexcept {
    static_assert(BucketCount != 0 && (BucketCount & (BucketCount - 1)) == 0);
    return hash & (BucketCount - 1);
}

} // namespace cuddl::detail

