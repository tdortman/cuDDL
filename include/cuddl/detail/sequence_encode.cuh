#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cstdint>
#include <limits>

#include <cub/block/block_reduce.cuh>
#include <cuddl/blacklist.cuh>
#include <cuddl/detail/dna.hpp>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Relaxed CTA-scope load of a shared word that other threads update atomically.
///
/// `cuda::atomic_ref` emits a generic-address load here; the shared state space avoids the
/// address conversion and the 64-bit address registers.
__device__ __forceinline__ uint32_t load_shared_relaxed(uint32_t const* address) {
    uint32_t value;
    asm volatile("ld.relaxed.cta.shared.u32 %0, [%1];"
                 : "=r"(value)
                 : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(address)))
                 : "memory");
    return value;
}

/// @brief Encodes raw ASCII windows into registers using shared overlapping tiles.
///
/// Each shared word holds eight two-bit bases and eight ambiguity bits. Threads reuse
/// neighboring words to construct eight overlapping windows on both strands.
template <
    size_t BucketCount,
    typename Layout,
    bool HasBlacklist = false,
    bool UseFloor = HasBlacklist>
__device__ __forceinline__ void add_sequence_windows(
    char const* sequence,
    uint32_t windows,
    size_t first_tile,
    size_t tile_stride,
    uint32_t k,
    uint32_t* registers,
    bool packed_input = false,
    blacklist_view blacklist = {}
) {
    constexpr uint32_t tile_size = 256 * 8;
    __shared__ uint32_t cells[tile_size / 8 + 4];
    __shared__ uint32_t local[BucketCount];
    for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
        local[i] = 0;
    }
    uint32_t floor = 0;
    uint32_t hash_ceiling = UINT32_MAX;
    size_t tiles = 0;
    auto const mask = (uint64_t{1} << (2 * k)) - 1;
    auto const valid_mask = (uint32_t{1} << k) - 1;
    for (size_t tile = first_tile; tile < windows; tile += tile_stride) {
        auto const count = cuda::std::min(tile_size, windows - static_cast<uint32_t>(tile));
        // Four extra cells cover the k-1 halo, including the last partial cell.
        auto const padded = (count + 7) / 8 + 4;
        if (packed_input) {
            // Two-bit words hold sixteen bases, base j in bits 2j, and no ambiguous base. Tiles
            // start on a multiple of sixteen bases, so a tile's cells are whole halves of words.
            auto const* const words = reinterpret_cast<uint32_t const*>(sequence);
            auto const total = (static_cast<size_t>(windows) + k - 1 + 7) / 8;
            for (uint32_t cell = threadIdx.x; cell < padded; cell += blockDim.x) {
                auto const index = tile / 8 + cell;
                if (index >= total) {
                    cells[cell] = missing_cell;
                    continue;
                }
                // A cell holds its first base in its highest bits: reverse the codes' order.
                auto const reversed = __brev((words[index / 2] >> (16 * (index % 2))) << 16);
                cells[cell] = ((reversed & 0x5555U) << 1) | ((reversed >> 1) & 0x5555U);
            }
        }
        for (uint32_t cell = threadIdx.x; !packed_input && cell < padded; cell += blockDim.x) {
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
            for (uint32_t j = 0; j < 2; ++j) {
                // Only the T code differs in ASCII bits 0 and 4.
                auto const thymine = (ascii[j] & ~(ascii[j] << 1) & 0x04040404U) >> 2;
                auto const normalised = (ascii[j] ^ (thymine * 0x11U)) & 0xD9D9D9D9U;
                auto const bad_bytes = __vsetne4(normalised, 0x41414141U);
                auto const codes = (ascii[j] >> 1) & 0x03030303U;
                // Gather four codes in reverse byte order and four invalid flags in byte order.
                packed = __dp4a(codes, 0x01041040U, packed << 8);
                bad = __dp4a(bad_bytes, 0x08040201U << (4 * j), bad);
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
            // An invalid base in the common overlap excludes all eight windows.
            auto const end = (bad & valid_mask & ~uint64_t{0x7F}) == 0
                                 ? cuda::std::min(8U, count - start)
                                 : 0U;
            _Pragma("unroll")
            for (uint32_t i = 0; i < end; ++i) {
                if ((bad & valid_mask) == 0) {
                    auto const forward = high >> (64 - 2 * k);
                    auto const reverse = reverse_high & mask;
                    auto const word = forward > reverse ? forward : reverse;
                    auto const hash = hash_kmer(word);
                    if (static_cast<uint32_t>(hash >> 32) <= hash_ceiling) {
                        auto const incoming = score<Layout>(hash);
                        auto const bucket = bucket_of<BucketCount>(hash);
                        if (!UseFloor || incoming > floor) {
                            // Look up only scores that can still raise the CTA's register.
                            if constexpr (HasBlacklist) {
                                auto const current = load_shared_relaxed(&local[bucket]);
                                if (incoming > current && !blacklist.contains(word, bucket)) {
                                    atomicMax(&local[bucket], incoming);
                                }
                            } else {
                                atomicMax(&local[bucket], incoming);
                            }
                        }
                    }
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
        if constexpr (UseFloor) {
            // Warm up with 16 windows per bucket; a single tile leaves the minimum at zero.
            if (++tiles == (16 * BucketCount + tile_size - 1) / tile_size &&
                tile + tile_stride < windows) {
                using reduce = cub::BlockReduce<uint32_t, 256>;
                __shared__ typename reduce::TempStorage scratch;
                __shared__ uint32_t minimum;
                uint32_t value = 0xffffU;
                for (uint32_t b = threadIdx.x; b < BucketCount; b += blockDim.x) {
                    value = cuda::std::min(value, local[b]);
                }
                auto const result = reduce(scratch).Reduce(value, cuda::minimum<>{});
                if (threadIdx.x == 0) minimum = result;
                __syncthreads();
                floor = minimum;
                hash_ceiling = floor == 0 ? UINT32_MAX : uint32_t(restore<Layout>(floor) >> 32);
            }
        }
        if constexpr (!UseFloor) {
            // Share accepted winners across CTAs before processing the remaining tiles.
            if ((++tiles == 4 || tiles % 16 == 0) && tile + tile_stride < windows) {
                using reduce = cub::BlockReduce<uint32_t, 256>;
                __shared__ typename reduce::TempStorage exchange_scratch;
                __shared__ uint32_t exchange_ceiling;
                uint32_t minimum = 0xffffU;
                // Each thread owns its buckets here, so the merged value feeds the floor directly.
                for (uint32_t b = threadIdx.x; b < BucketCount; b += blockDim.x) {
                    auto const value = local[b];
                    auto global = cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{registers[b]}
                                      .load(cuda::memory_order_relaxed);
                    if (value > global) global = cuda::std::max(value, atomicMax(&registers[b], value));
                    local[b] = global;
                    minimum = cuda::std::min(minimum, global);
                }
                auto const result = reduce(exchange_scratch).Reduce(minimum, cuda::minimum<>{});
                if (threadIdx.x == 0) {
                    // An inclusive high-word cutoff admits ties for exact per-bucket pruning.
                    exchange_ceiling = result == 0 ? UINT32_MAX : uint32_t(restore<Layout>(result) >> 32);
                }
                __syncthreads();
                hash_ceiling = exchange_ceiling;
            }
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < BucketCount; i += blockDim.x) {
        if (local[i] != 0U) atomicMax(&registers[i], local[i]);
    }
}

/// @brief Tile kernel for one staged file-builder chunk, carrying K-1 bases across tiles.
template <
    size_t BucketCount,
    typename Layout,
    bool HasBlacklist = false,
    bool UseFloor = HasBlacklist>
__global__ void add_sequence_tile_kernel(
    char const* sequence,
    uint32_t const* window_count,
    char* carry,
    uint32_t k,
    uint32_t* registers,
    blacklist_view blacklist = {}
) {
    auto const windows = *window_count;
    if (blockIdx.x == 0 && threadIdx.x < k - 1) {
        carry[threadIdx.x] = sequence[windows + threadIdx.x];
    }
    add_sequence_windows<BucketCount, Layout, HasBlacklist, UseFloor>(
        sequence,
        windows,
        size_t{blockIdx.x} * 2048,
        size_t{gridDim.x} * 2048,
        k,
        registers,
        false,
        blacklist
    );
}

/// @brief One piece of sequence of one genome, and where its bytes are.
///
/// @p bases names the piece directly: a device arena the producer copied it into, or the
/// producer's own pageable buffer on a host whose device reads pageable memory itself.
struct sequence_batch_chunk {
    char const* bases;
    size_t block_end;
    uint32_t genome;
    uint32_t windows;
    uint32_t packed;  // nonzero: @p bases holds two-bit words, not ASCII
};

/// @brief Batch kernel covering every staged chunk in one launch, including short records.
template <
    size_t BucketCount,
    typename Layout,
    bool HasBlacklist = false,
    bool UseFloor = HasBlacklist>
__global__ void add_sequence_batch_kernel(
    sequence_batch_chunk const* chunks,
    size_t chunk_count,
    size_t block_count,
    uint32_t k,
    uint32_t* registers,
    blacklist_view blacklist = {}
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
        auto* target = registers + size_t{chunk.genome} * BucketCount;
        add_sequence_windows<BucketCount, Layout, HasBlacklist, UseFloor>(
            chunk.bases,
            chunk.windows,
            (block - first_block) * 2048,
            (chunk.block_end - first_block) * 2048,
            k,
            target,
            chunk.packed != 0,
            blacklist
        );
        __syncthreads();
    }
}

/// @brief Single-sequence kernel for the public raw-ASCII API.
///
/// Unlike the file-builder tile path, there is no cross-chunk carry: the caller supplies any
/// K-1 overlap explicitly.
template <
    size_t BucketCount,
    typename Layout,
    bool HasBlacklist = false,
    bool UseFloor = HasBlacklist>
__global__ __launch_bounds__(256, 6) void add_sequence_single_kernel(
    char const* sequence,
    uint32_t windows,
    uint32_t k,
    uint32_t* registers,
    blacklist_view blacklist = {}
) {
    add_sequence_windows<BucketCount, Layout, HasBlacklist, UseFloor>(
        sequence,
        windows,
        size_t{blockIdx.x} * 2048,
        size_t{gridDim.x} * 2048,
        k,
        registers,
        false,
        blacklist
    );
}

/// @brief Shared launch for one device-resident ASCII chunk.
///
/// Windows derive as size >= k ? size - k + 1 : 0; short input is a no-op. Window counts above
/// UINT32_MAX are rejected instead of narrowing, so size must not exceed UINT32_MAX + k - 1.
/// No hidden carry: the caller supplies any K-1 overlap explicitly.
template <size_t BucketCount, typename Layout = default_register_layout, bool UseFloor = true>
__host__ inline Result<void> launch_sequence_add(
    device_span<char const> sequence,
    uint32_t k,
    device_span<uint32_t> registers,
    cuda::stream_ref stream,
    cuda::std::optional<blacklist_view> blacklist = cuda::std::nullopt
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
    auto launch = [&]<bool Filter>() -> Result<void> {
        return cuda_try([&] {
            auto const multiprocessors =
                stream.device().attribute(cuda::device_attributes::multiprocessor_count);
            // Fill one resident wave, accounting for the GPU and this sketch's shared memory.
            if constexpr (Filter) {
                // Leave L1 room for the blacklist presence map; the default favors shared memory.
                auto const carveout = cudaFuncSetAttribute(
                    add_sequence_single_kernel<BucketCount, Layout, Filter, false>,
                    cudaFuncAttributePreferredSharedMemoryCarveout,
                    50
                );
                if (carveout != cudaSuccess) return carveout;
            }
            int blocks_per_sm = 0;
            auto const occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm,
                add_sequence_single_kernel<BucketCount, Layout, Filter, false>,
                256,
                0
            );
            if (occupancy != cudaSuccess) return occupancy;
            size_t const need = (static_cast<size_t>(windows) + 2047) / 2048;
            size_t const capacity = static_cast<size_t>(multiprocessors) * blocks_per_sm;
            size_t const blocks_size =
                capacity == 0 ? size_t{1} : (capacity < need ? capacity : need);
            auto const blocks = static_cast<uint32_t>(blocks_size);
            add_sequence_single_kernel<BucketCount, Layout, Filter, false>
                <<<blocks, 256, 0, stream.get()>>>(
                    sequence.data(),
                    windows,
                    k,
                    registers.data(),
                    blacklist.value_or(blacklist_view{})
                );
            return cudaGetLastError();
        });
    };
    return blacklist ? launch.template operator()<true>() : launch.template operator()<false>();
}

}  // namespace cuddl::detail
