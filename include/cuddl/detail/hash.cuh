#pragma once

#include <cuda/std/bit>
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

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

/// @brief Rotates @p x left by @p r bits.
constexpr __host__ __device__ __forceinline__ uint64_t rotl64(uint64_t x, int8_t r) {
    return cuda::std::rotl(x, static_cast<int>(r));
}

/// @brief Loads a chunk of type @p T from @p data at byte offset @c index*sizeof(T).
template <typename T>
__host__ __device__ __forceinline__ T load_chunk(const uint8_t* data, uint64_t index) {
    T chunk;
    memcpy(&chunk, data + index * sizeof(T), sizeof(T));
    return chunk;
}

/// @brief Applies the xxHash-64 final mixing (avalanche) step.
constexpr __host__ __device__ __forceinline__ uint64_t finalize(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

/// @brief Computes xxHash-64 over a byte range without allocating or copying it.
__host__ __device__ inline uint64_t xxhash64(const uint8_t* bytes, uint64_t size, uint64_t seed) {
    uint64_t offset = 0;
    uint64_t h64;

    // Process 32-byte chunks
    if (size >= 32) {
        uint64_t limit = size - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME64_1;

        do {
            const uint64_t pipeline_offset = offset / 8;
            v1 += load_chunk<uint64_t>(bytes, pipeline_offset + 0) * PRIME64_2;
            v1 = rotl64(v1, 31);
            v1 *= PRIME64_1;
            v2 += load_chunk<uint64_t>(bytes, pipeline_offset + 1) * PRIME64_2;
            v2 = rotl64(v2, 31);
            v2 *= PRIME64_1;
            v3 += load_chunk<uint64_t>(bytes, pipeline_offset + 2) * PRIME64_2;
            v3 = rotl64(v3, 31);
            v3 *= PRIME64_1;
            v4 += load_chunk<uint64_t>(bytes, pipeline_offset + 3) * PRIME64_2;
            v4 = rotl64(v4, 31);
            v4 *= PRIME64_1;
            offset += 32;
        } while (offset <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);

        v1 *= PRIME64_2;
        v1 = rotl64(v1, 31);
        v1 *= PRIME64_1;
        h64 ^= v1;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v2 *= PRIME64_2;
        v2 = rotl64(v2, 31);
        v2 *= PRIME64_1;
        h64 ^= v2;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v3 *= PRIME64_2;
        v3 = rotl64(v3, 31);
        v3 *= PRIME64_1;
        h64 ^= v3;
        h64 = h64 * PRIME64_1 + PRIME64_4;

        v4 *= PRIME64_2;
        v4 = rotl64(v4, 31);
        v4 *= PRIME64_1;
        h64 ^= v4;
        h64 = h64 * PRIME64_1 + PRIME64_4;
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += size;

    // Process remaining 8-byte chunks
    if ((size % 32) >= 8) {
        for (; offset <= size - 8; offset += 8) {
            uint64_t k1 = load_chunk<uint64_t>(bytes, offset / 8) * PRIME64_2;
            k1 = rotl64(k1, 31) * PRIME64_1;
            h64 ^= k1;
            h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        }
    }

    // Process remaining 4-byte chunks
    if ((size % 8) >= 4) {
        for (; offset <= size - 4; offset += 4) {
            h64 ^= (load_chunk<uint32_t>(bytes, offset / 4) & 0xffffffffULL) * PRIME64_1;
            h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        }
    }

    // Process remaining bytes
    if (size % 4) {
        while (offset < size) {
            h64 ^= (bytes[offset] & 0xff) * PRIME64_5;
            h64 = rotl64(h64, 11) * PRIME64_1;
            ++offset;
        }
    }

    return finalize(h64);
}

/// @brief Computes xxHash-64 over the raw bytes of a value.
template <typename T>
__host__ __device__ inline uint64_t xxhash64(const T& key, uint64_t seed = 0) {
    return xxhash64(reinterpret_cast<const uint8_t*>(&key), sizeof(T), seed);
}

}  // namespace cuddl::detail
