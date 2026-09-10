#pragma once

#include <cuda_runtime.h>
#include <cuda/devices>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cstdint>
#include <limits>

#include <cuddl/detail/dna.hpp>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

// Each shared word holds eight two-bit bases and eight ambiguity bits. Threads
// reuse neighboring words to construct eight overlapping windows on both strands.
template <size_t BucketCount, typename Layout>
__device__ __forceinline__ void add_sequence_windows(
    char const* sequence,
    uint32_t windows,
    size_t first_tile,
    size_t tile_stride,
    uint32_t k,
    uint32_t* registers,
    uint32_t& saturation
) {
    constexpr uint32_t tile_size = 256 * 8;
    constexpr bool shared_sketch = BucketCount <= 8192;
    __shared__ uint32_t cells[tile_size / 8 + 4];
    __shared__ uint32_t local[shared_sketch ? BucketCount : 1];
    if constexpr (shared_sketch) {
        for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
            local[i] = 0;
        }
    }
    auto* target = shared_sketch ? local : registers;
    auto const mask = (uint64_t{1} << (2 * k)) - 1;
    auto const valid_mask = (uint32_t{1} << k) - 1;
    for (size_t tile = first_tile; tile < windows; tile += tile_stride) {
        auto const count = cuda::std::min(tile_size, windows - static_cast<uint32_t>(tile));
        // Four extra cells cover the k-1 halo, including the last partial cell.
        auto const padded = (count + 7) / 8 + 4;
        for (uint32_t cell = threadIdx.x; cell < padded; cell += blockDim.x) {
            auto const pos = cell * 8;
            uint32_t ascii[2] = {};
            if ((reinterpret_cast<uintptr_t>(sequence + tile) & 7U) == 0 &&
                pos + 7 < count + k - 1) {
                auto const value = *reinterpret_cast<uint2 const*>(sequence + tile + pos);
                ascii[0] = value.x;
                ascii[1] = value.y;
            } else {
                _Pragma("unroll")
                for (uint32_t j = 0; j < 8; ++j) {
                    auto const byte = pos + j < count + k - 1
                                          ? static_cast<uint8_t>(sequence[tile + pos + j])
                                          : uint8_t{0xFF};
                    ascii[j / 4] |= static_cast<uint32_t>(byte) << (8 * (j % 4));
                }
            }
            uint32_t packed = 0, bad = 0;
            _Pragma("unroll")
            for (uint32_t j = 0; j < 8; ++j) {
                auto const symbol = encode_base(static_cast<char>(ascii[j / 4] >> (8 * (j % 4))));
                packed = (packed << 2) | (symbol & 3U);
                bad |= static_cast<uint32_t>(symbol == 0xFFU) << j;
            }
            cells[cell] = packed | (bad << 16);
        }
        __syncthreads();
        auto const start = threadIdx.x * 8;
        if (start < count) {
            auto const cell = start / 8;
            auto const a = cells[cell], b = cells[cell + 1];
            auto const c = cells[cell + 2], d = cells[cell + 3];
            auto high =
                (uint64_t{a} << 48) | (uint64_t{b & 0xFFFFU} << 32) | (c << 16) | (d & 0xFFFFU);
            uint64_t low = 0;
            auto bad = uint64_t{a >> 16} | (uint64_t{b >> 16} << 8) | (uint64_t{c >> 16} << 16) |
                       (uint64_t{d >> 16} << 24);
            if (k > 25) {
                auto const e = cells[cell + 4];
                low = uint64_t{e & 0xFFFFU} << 48;
                bad |= uint64_t{e >> 16} << 32;
            }
            // Reverse the span once, then slide both strands across its eight windows.
            auto reverse_word = [](uint64_t word) {
                auto const bits = cuda::std::bit_reverse(word);
                return (((bits & 0xAAAAAAAAAAAAAAAAULL) >> 1) |
                        ((bits & 0x5555555555555555ULL) << 1)) ^
                       0xAAAAAAAAAAAAAAAAULL;
            };
            auto reverse_high = reverse_word(high);
            auto reverse_low = reverse_word(low);
            auto const end = cuda::std::min(8U, count - start);
            for (uint32_t i = 0; i < end; ++i) {
                if ((bad & valid_mask) == 0) {
                    auto const forward = high >> (64 - 2 * k);
                    auto const reverse = reverse_high & mask;
                    auto const hash = hash_kmer(forward > reverse ? forward : reverse);
                    update(&target[bucket_of<BucketCount>(hash)], score<Layout>(hash), saturation);
                }
                // Eight overlapping windows fit in one 32-base word when k <= 25.
                if (k <= 25) {
                    high <<= 2;
                    reverse_high >>= 2;
                } else {
                    high = (high << 2) | (low >> 62);
                    low <<= 2;
                    reverse_high = (reverse_high >> 2) | (reverse_low << 62);
                    reverse_low >>= 2;
                }
                bad >>= 1;
            }
        }
        __syncthreads();
    }
    if constexpr (shared_sketch) {
        for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
            merge_register(&registers[i], local[i], saturation);
        }
    }
}

template <size_t BucketCount, typename Layout>
__global__ void add_sequence_tile_kernel(
    char const* sequence,
    uint32_t const* window_count,
    char* carry,
    uint32_t k,
    uint32_t* registers,
    uint32_t& saturation
) {
    auto const windows = *window_count;
    if (blockIdx.x == 0 && threadIdx.x < k - 1) {
        carry[threadIdx.x] = sequence[windows + threadIdx.x];
    }
    add_sequence_windows<BucketCount, Layout>(
        sequence,
        windows,
        size_t{blockIdx.x} * 2048,
        size_t{gridDim.x} * 2048,
        k,
        registers,
        saturation
    );
}

struct sequence_batch_chunk {
    size_t offset;
    size_t block_end;
    uint32_t genome;
    uint32_t windows;
};

// Logical blocks cover every chunk in one launch, including independent short records.
template <size_t BucketCount, typename Layout>
__global__ void add_sequence_batch_kernel(
    char const* sequence,
    sequence_batch_chunk const* chunks,
    size_t chunk_count,
    size_t block_count,
    uint32_t k,
    uint32_t* registers
) {
    __shared__ sequence_batch_chunk chunk;
    __shared__ size_t first_block;
    for (size_t block = blockIdx.x; block < block_count; block += gridDim.x) {
        if (threadIdx.x == 0) {
            size_t low = 0, high = chunk_count;
            while (low < high) {
                auto const middle = low + (high - low) / 2;
                if (chunks[middle].block_end <= block) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            chunk = chunks[low];
            first_block = low ? chunks[low - 1].block_end : 0;
        }
        __syncthreads();
        auto* target = registers + size_t{chunk.genome} * (BucketCount + 1);
        add_sequence_windows<BucketCount, Layout>(
            sequence + chunk.offset,
            chunk.windows,
            (block - first_block) * 2048,
            (chunk.block_end - first_block) * 2048,
            k,
            target,
            target[BucketCount]
        );
        __syncthreads();
    }
}

// Single-sequence launch for the public raw-ASCII API. Unlike the file-builder tile path,
// there is no cross-chunk carry: the caller supplies any K-1 overlap explicitly.
template <size_t BucketCount, typename Layout>
__global__ void add_sequence_single_kernel(
    char const* sequence,
    uint32_t windows,
    uint32_t k,
    uint32_t* registers,
    uint32_t& saturation
) {
    add_sequence_windows<BucketCount, Layout>(
        sequence,
        windows,
        size_t{blockIdx.x} * 2048,
        size_t{gridDim.x} * 2048,
        k,
        registers,
        saturation
    );
}

// Shared launch for one device-resident ASCII chunk. Windows derive as
// size >= k ? size - k + 1 : 0; short input is a no-op. Window counts above
// UINT32_MAX are rejected instead of narrowing, so size must not exceed
// UINT32_MAX + k - 1. No hidden carry: the caller supplies any K-1 overlap
// explicitly.
template <size_t BucketCount, typename Layout = default_register_layout>
__host__ inline Result<void> launch_sequence_add(
    device_span<char const> sequence,
    uint32_t k,
    device_span<uint32_t> registers,
    uint32_t& saturation,
    cuda::stream_ref stream
) {
    if (k < 1 || k > 31) {
        return Err(Error::invalid_argument("invalid k-mer length for sequence add"));
    }
    size_t const size = sequence.size();
    if (size < k) {
        return Ok();
    }
    size_t const windows_size = size - k + 1;
    if (windows_size > std::numeric_limits<uint32_t>::max()) {
        return Err(Error::invalid_argument("sequence chunk exceeds 32-bit window capacity"));
    }
    auto const windows = static_cast<uint32_t>(windows_size);
    return cuda_try([&] {
        auto const multiprocessors =
            stream.device().attribute(cuda::device_attributes::multiprocessor_count);
        // Fill one resident wave, accounting for the GPU and this sketch's shared memory.
        int blocks_per_sm = 0;
        auto const occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, add_sequence_single_kernel<BucketCount, Layout>, 256, 0
        );
        if (occupancy != cudaSuccess) return occupancy;
        size_t const need = (static_cast<size_t>(windows) + 2047) / 2048;
        size_t const capacity = static_cast<size_t>(multiprocessors) * blocks_per_sm;
        size_t const blocks_size = capacity == 0 ? size_t{1} : (capacity < need ? capacity : need);
        auto const blocks = static_cast<uint32_t>(blocks_size);
        add_sequence_single_kernel<BucketCount, Layout><<<blocks, 256, 0, stream.get()>>>(
            sequence.data(), windows, k, registers.data(), saturation
        );
        return cudaGetLastError();
    });
}

}  // namespace cuddl::detail
