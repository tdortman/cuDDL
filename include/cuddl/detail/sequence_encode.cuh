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

// A tile encodes each ASCII base once; each thread rolls eight adjacent windows
// with runtime k and feeds the register update/merge operations.
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
    __shared__ uint8_t bases[tile_size + 30];
    __shared__ uint32_t local[shared_sketch ? BucketCount : 1];
    if constexpr (shared_sketch) {
        for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
            local[i] = 0;
        }
    }
    auto* target = shared_sketch ? local : registers;
    auto const mask = (uint64_t{1} << (2 * k)) - 1;
    for (size_t tile = first_tile; tile < windows; tile += tile_stride) {
        auto const count = cuda::std::min(tile_size, windows - static_cast<uint32_t>(tile));
        for (uint32_t i = threadIdx.x; i < count + k - 1; i += blockDim.x) {
            bases[i] = encode_base(sequence[tile + i]);
        }
        __syncthreads();
        auto const start = threadIdx.x * 8;
        if (start < count) {
            uint64_t forward = 0;
            uint32_t valid = 0;
            auto const end = cuda::std::min(start + 8, count) + k - 1;
            for (uint32_t i = start; i < end; ++i) {
                auto const symbol = bases[i];
                forward = ((forward << 2) | (symbol & 3U)) & mask;
                valid = symbol == 0xFFu ? 0 : valid + 1;
                if (valid >= k) {
                    auto const bits = cuda::std::bit_reverse(forward);
                    auto const pairs = ((bits & 0xAAAAAAAAAAAAAAAAULL) >> 1) |
                                       ((bits & 0x5555555555555555ULL) << 1);
                    auto const reverse = (pairs ^ 0xAAAAAAAAAAAAAAAAULL) >> (64 - 2 * k);
                    auto const hash = hash_kmer(forward > reverse ? forward : reverse);
                    update(&target[bucket_of<BucketCount>(hash)], score<Layout>(hash), saturation);
                }
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
        size_t const need = (static_cast<size_t>(windows) + 2047) / 2048;
        size_t const capacity = static_cast<size_t>(multiprocessors) * 2U;
        size_t const blocks_size = capacity == 0 ? size_t{1} : (capacity < need ? capacity : need);
        auto const blocks = static_cast<uint32_t>(blocks_size);
        add_sequence_single_kernel<BucketCount, Layout><<<blocks, 256, 0, stream.get()>>>(
            sequence.data(), windows, k, registers.data(), saturation
        );
        return cudaGetLastError();
    });
}

}  // namespace cuddl::detail
