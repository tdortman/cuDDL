#pragma once

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>

#include <optional>

#include <cuddl/detail/kernels.cuh>
#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Most buckets in which a child reference may differ from its base reference.
constexpr uint32_t delta_change_limit = 160U;
/// @brief Exact bucket signatures that propose base candidates, and buckets per signature.
constexpr uint32_t delta_signatures = 8U;
constexpr uint32_t delta_signature_buckets = 8U;
/// @brief Highest-ID rows taken from each matching signature.
constexpr uint32_t delta_signature_candidates = 4U;
constexpr uint32_t delta_candidates = delta_signatures * delta_signature_candidates;
/// @brief Warps per @ref apply_reference_deltas_kernel block and children per block cell.
constexpr uint32_t delta_correction_warps = 24U;
constexpr uint32_t delta_child_tile = 2048U;

/// @brief Queries one correction cell stages bucket-major: 64 KiB of scores for every bucket
/// count. Lane l serves query l % group and every (32 / group)-th change from l / group.
template <size_t BucketCount>
constexpr uint32_t delta_query_group = static_cast<uint32_t>(32768U / BucketCount);

template <size_t BucketCount>
constexpr size_t delta_correction_shared_bytes =
    BucketCount * delta_query_group<BucketCount> * sizeof(uint16_t) +
    size_t{delta_correction_warps} * delta_change_limit * sizeof(uint2);

/// @brief Reference rows stored as a base reference plus the buckets that differ from it.
///
/// Bases are exactly compared with queries; each child's counts then follow from its base's by
/// the change in classification of its changed buckets. Every child's base has a higher ID, so
/// an upper-triangle pair (query, child) implies the pair (query, base).
struct reference_delta_view {
    /// One bit per reference in the candidate-bitmap layout, set for bases.
    uint32_t const* base_bits = nullptr;
    /// Per child in ascending ID order: ID, base ID, first record, and
    /// `count | zero_count << 10 | increase_count << 20`.
    uint4 const* children = nullptr;
    /// Per changed bucket: x is the bucket's byte offset in the bucket-major staged scores; y is
    /// `low | width << 16` with low = min(base score, child score) and width their difference.
    /// A child's records hold low-zero changes first (low replaced by 1 for a decrease), then
    /// increases, then decreases.
    uint2 const* records = nullptr;
    uint32_t child_count = 0U;
};

struct reference_deltas {
    cuda::device_buffer<uint32_t> base_bits;
    cuda::device_buffer<uint4> children;
    cuda::device_buffer<uint2> records;

    [[nodiscard]] reference_delta_view view() const noexcept {
        return {
            base_bits.data(),
            children.data(),
            records.data(),
            static_cast<uint32_t>(children.size()),
        };
    }
};

/// @brief Hashes signature @p signature of every row; rows are emitted in descending ID order so
/// a stable sort keeps the highest IDs first within a signature.
template <size_t BucketCount>
__global__ __launch_bounds__(block_size) void delta_signature_kernel(
    uint16_t const* rows,
    uint32_t count,
    uint32_t signature,
    uint64_t* keys,
    uint32_t* ids
) {
    constexpr uint32_t spacing =
        static_cast<uint32_t>(BucketCount / (delta_signatures * delta_signature_buckets));
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        auto const row = count - 1U - i;
        uint64_t key = signature + 1U;
        _Pragma("unroll")
        for (uint32_t j = 0; j < delta_signature_buckets; ++j) {
            auto const bucket = (signature * delta_signature_buckets + j) * spacing;
            key = (key ^ rows[static_cast<size_t>(row) * BucketCount + bucket]) *
                  0x9e3779b97f4a7c15ULL;
            key ^= key >> 29U;
        }
        keys[i] = key;
        ids[i] = row;
    }
}

/// @brief Marks the first position of every run of equal sorted keys with its index.
__global__ __launch_bounds__(
    block_size
) void delta_run_heads_kernel(uint64_t const* keys, uint32_t count, uint32_t* starts) {
    for (auto p = blockIdx.x * blockDim.x + threadIdx.x; p < count; p += gridDim.x * blockDim.x) {
        starts[p] = p == 0U || keys[p] != keys[p - 1U] ? p : 0U;
    }
}

/// @brief Records, for every row, the highest-ID rows sharing its signature that exceed its ID.
__global__ __launch_bounds__(block_size) void delta_candidates_kernel(
    uint32_t const* ids,
    uint32_t const* starts,
    uint32_t count,
    uint32_t signature,
    uint32_t* candidates
) {
    for (auto p = blockIdx.x * blockDim.x + threadIdx.x; p < count; p += gridDim.x * blockDim.x) {
        auto* const out = candidates + static_cast<size_t>(ids[p]) * delta_candidates +
                          signature * delta_signature_candidates;
        _Pragma("unroll")
        for (uint32_t j = 0; j < delta_signature_candidates; ++j) {
            auto const q = starts[p] + j;
            out[j] = q < p ? ids[q] : ~0U;
        }
    }
}

/// @brief Select the highest viable target, or retarget children to closer marked bases.
template <size_t BucketCount, bool Closest>
__global__ __launch_bounds__(block_size) void delta_targets_kernel(
    uint16_t const* rows,
    uint32_t count,
    uint32_t const* candidates,
    uint8_t const* is_base,
    uint32_t* targets,
    uint32_t* changes
) {
    constexpr uint32_t vectors = static_cast<uint32_t>(BucketCount / 8U);
    auto const lane = threadIdx.x % 32U;
    for (auto row = (blockIdx.x * blockDim.x + threadIdx.x) / 32U; row < count;
         row += gridDim.x * blockDim.x / 32U) {
        if constexpr (Closest) {
            if (is_base[row] != 0U || changes[row] == 0U) {
                continue;
            }
        }
        auto target = Closest ? targets[row] : row;
        uint32_t changed = Closest ? changes[row] : 0U;
        auto candidate = candidates[static_cast<size_t>(row) * delta_candidates + lane];
        if constexpr (Closest) {
            if (candidate != ~0U && (candidate == target || is_base[candidate] == 0U)) {
                candidate = ~0U;
            }
        }
        auto const* own =
            reinterpret_cast<uint4 const*>(rows + static_cast<size_t>(row) * BucketCount);
        for (;;) {
            auto const best =
                __reduce_max_sync(0xffffffffU, candidate == ~0U ? 0U : candidate + 1U);
            if (best == 0U) {
                break;
            }
            auto const* other =
                reinterpret_cast<uint4 const*>(rows + static_cast<size_t>(best - 1U) * BucketCount);
            uint32_t differ = 0U;
            for (auto v = lane; v < vectors; v += 32U) {
                auto const a = own[v];
                auto const b = other[v];
                uint32_t const words[4] = {a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w};
                _Pragma("unroll")
                for (auto const x : words) {
                    differ += ((x & 0xffffU) != 0U) + ((x >> 16U) != 0U);
                }
            }
            differ = __reduce_add_sync(0xffffffffU, differ);
            if (Closest ? differ < changed : differ <= delta_change_limit) {
                target = best - 1U;
                changed = differ;
                if (!Closest || differ == 0U) {
                    break;
                }
            }
            if (candidate == best - 1U) {
                candidate = ~0U;
            }
        }
        if (lane == 0U) {
            targets[row] = target;
            changes[row] = changed;
        }
    }
}

__global__ __launch_bounds__(
    block_size
) void delta_mark_bases_kernel(uint32_t const* targets, uint32_t count, uint8_t* is_base) {
    for (auto row = blockIdx.x * blockDim.x + threadIdx.x; row < count;
         row += gridDim.x * blockDim.x) {
        auto const target = targets[row];
        is_base[target] = 1U;
    }
}

/// @brief Every row that is some row's target, or its own, is a base; other rows become
/// children of their targets. Writes the base bitmap, child flags, and per-row change counts.
__global__ __launch_bounds__(block_size) void delta_classify_kernel(
    uint8_t const* is_base,
    uint32_t count,
    uint32_t* changes,
    uint8_t* is_child,
    uint32_t* base_bits
) {
    auto const words = candidate_bit_words(count);
    for (auto row = blockIdx.x * blockDim.x + threadIdx.x; row < words * 32U;
         row += gridDim.x * blockDim.x) {
        auto const base = row < count && is_base[row] != 0U;
        if (row < count) {
            is_child[row] = !base;
            if (base) {
                changes[row] = 0U;
            }
        }
        auto const bits = __ballot_sync(0xffffffffU, base);
        if (row % 32U == 0U) {
            base_bits[row / 32U] = bits;
        }
    }
}

/// @brief One warp per child: writes its change records, low-zero changes first, then
/// increases, then decreases, and its child entry.
template <size_t BucketCount>
__global__ __launch_bounds__(block_size) void delta_records_kernel(
    uint16_t const* rows,
    uint32_t const* child_ids,
    uint32_t const* child_count,
    uint32_t const* targets,
    uint32_t const* offsets,
    uint4* children,
    uint2* records
) {
    constexpr uint32_t groups = static_cast<uint32_t>(BucketCount / 32U);
    constexpr uint32_t row_bytes = delta_query_group<BucketCount> * sizeof(uint16_t);
    auto const lane = threadIdx.x % 32U;
    auto const below = (1U << lane) - 1U;
    for (auto index = (blockIdx.x * blockDim.x + threadIdx.x) / 32U; index < *child_count;
         index += gridDim.x * blockDim.x / 32U) {
        auto const child = child_ids[index];
        auto const base = targets[child];
        auto const* after_row = rows + static_cast<size_t>(child) * BucketCount;
        auto const* before_row = rows + static_cast<size_t>(base) * BucketCount;
        uint32_t zeros = 0U;
        uint32_t increases = 0U;
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t const before = before_row[g * 32U + lane];
            uint32_t const after = after_row[g * 32U + lane];
            auto const changed = before != after;
            auto const zero = changed && (before == 0U || after == 0U);
            zeros += static_cast<uint32_t>(__popc(__ballot_sync(0xffffffffU, zero)));
            increases += static_cast<uint32_t>(
                __popc(__ballot_sync(0xffffffffU, changed && !zero && after > before))
            );
        }
        auto const begin = offsets[child];
        auto const count = offsets[child + 1U] - begin;
        uint32_t next[3] = {begin, begin + zeros, begin + zeros + increases};
        for (uint32_t g = 0; g < groups; ++g) {
            auto const bucket = g * 32U + lane;
            uint32_t const before = before_row[bucket];
            uint32_t const after = after_row[bucket];
            auto const changed = before != after;
            auto const low = min(before, after);
            auto const kind = low == 0U ? 0U : after > before ? 1U : 2U;
            _Pragma("unroll")
            for (uint32_t k = 0; k < 3U; ++k) {
                auto const members = __ballot_sync(0xffffffffU, changed && kind == k);
                if (changed && kind == k) {
                    auto const width = max(before, after) - low;
                    records[next[k] + static_cast<uint32_t>(__popc(members & below))] = {
                        bucket * row_bytes,
                        (k == 0U ? static_cast<uint32_t>(after < before) : low) | width << 16U,
                    };
                }
                next[k] += static_cast<uint32_t>(__popc(members));
            }
        }
        if (lane == 0U) {
            children[index] = {child, base, begin, count | zeros << 10U | increases << 20U};
        }
    }
}

/// @brief Builds the delta model of @p count row-major score rows; empty when no row has a
/// base within @ref delta_change_limit changed buckets. Blocks @p stream to size the model.
template <size_t BucketCount>
[[nodiscard]] Result<std::optional<reference_deltas>>
build_reference_deltas(uint16_t const* rows, uint32_t count, cuda::stream_ref stream) {
    if (count < 2U) {
        return std::optional<reference_deltas>{};
    }
    auto const device = stream.device();
    auto const grid = warp_grid_blocks(count);
    auto const warp_grid = warp_grid_blocks(static_cast<size_t>(count) * 32U);
    auto candidates = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<uint32_t>(
            stream, device, static_cast<size_t>(count) * delta_candidates, cuda::no_init
        )
    );
    {
        auto keys = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint64_t>(stream, device, count, cuda::no_init)
        );
        auto sorted_keys = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint64_t>(stream, device, count, cuda::no_init)
        );
        auto ids = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, device, count, cuda::no_init)
        );
        auto sorted_ids = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, device, count, cuda::no_init)
        );
        auto starts = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, device, count, cuda::no_init)
        );
        for (uint32_t signature = 0; signature < delta_signatures; ++signature) {
            delta_signature_kernel<BucketCount><<<grid, block_size, 0, stream.get()>>>(
                rows, count, signature, keys.data(), ids.data()
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
            CUDDL_CUDA_TRY(
                cub::DeviceRadixSort::SortPairs(
                    keys.data(),
                    sorted_keys.data(),
                    ids.data(),
                    sorted_ids.data(),
                    count,
                    0,
                    64,
                    stream
                )
            );
            delta_run_heads_kernel<<<grid, block_size, 0, stream.get()>>>(
                sorted_keys.data(), count, starts.data()
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
            CUDDL_CUDA_TRY(
                cub::DeviceScan::InclusiveScan(
                    starts.data(), cuda::maximum<uint32_t>{}, static_cast<int64_t>(count), stream
                )
            );
            delta_candidates_kernel<<<grid, block_size, 0, stream.get()>>>(
                sorted_ids.data(), starts.data(), count, signature, candidates.data()
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
        }
    }
    auto targets =
        CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, count, cuda::no_init));
    // One extra zero entry makes the exclusive scan's last element the record total.
    auto offsets =
        CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, count + 1U, 0U));
    delta_targets_kernel<BucketCount, false><<<warp_grid, block_size, 0, stream.get()>>>(
        rows, count, candidates.data(), nullptr, targets.data(), offsets.data()
    );
    CUDDL_CUDA_TRY(cudaGetLastError());
    auto is_base = CUDDL_CUDA_TRY(cuda::make_device_buffer<uint8_t>(stream, device, count, 0U));
    auto is_child =
        CUDDL_CUDA_TRY(cuda::make_device_buffer<uint8_t>(stream, device, count, cuda::no_init));
    auto base_bits = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<uint32_t>(
            stream, device, candidate_bit_words(count), cuda::no_init
        )
    );
    delta_mark_bases_kernel<<<grid, block_size, 0, stream.get()>>>(
        targets.data(), count, is_base.data()
    );
    CUDDL_CUDA_TRY(cudaGetLastError());
    delta_targets_kernel<BucketCount, true><<<warp_grid, block_size, 0, stream.get()>>>(
        rows, count, candidates.data(), is_base.data(), targets.data(), offsets.data()
    );
    CUDDL_CUDA_TRY(cudaGetLastError());
    delta_classify_kernel<<<
        warp_grid_blocks(candidate_bit_words(count)),
        block_size,
        0,
        stream.get()>>>(is_base.data(), count, offsets.data(), is_child.data(), base_bits.data());
    CUDDL_CUDA_TRY(cudaGetLastError());
    CUDDL_CUDA_TRY(
        cub::DeviceScan::ExclusiveSum(offsets.data(), static_cast<int64_t>(count + 1U), stream)
    );
    // The candidate buffer is no longer needed; its first words take the child IDs and count.
    auto* const child_ids = candidates.data();
    auto* const child_count = candidates.data() + count;
    CUDDL_CUDA_TRY(
        cub::DeviceSelect::Flagged(
            cuda::make_counting_iterator(uint32_t{0}),
            is_child.data(),
            child_ids,
            child_count,
            static_cast<int64_t>(count),
            stream
        )
    );
    uint32_t totals[2] = {};
    CUDDL_CUDA_TRY(cudaMemcpyAsync(
        &totals[0], child_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.get()
    ));
    CUDDL_CUDA_TRY(cudaMemcpyAsync(
        &totals[1], offsets.data() + count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.get()
    ));
    if (auto const synced = cuda_try([&] { stream.sync(); }); !synced) {
        return Err(synced.error());
    }
    if (totals[0] == 0U) {
        return std::optional<reference_deltas>{};
    }
    auto children =
        CUDDL_CUDA_TRY(cuda::make_device_buffer<uint4>(stream, device, totals[0], cuda::no_init));
    auto records = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<uint2>(
            stream, device, cuda::std::max(totals[1], 1U), cuda::no_init
        )
    );
    auto deltas = reference_deltas{std::move(base_bits), std::move(children), std::move(records)};
    delta_records_kernel<BucketCount>
        <<<warp_grid_blocks(static_cast<size_t>(totals[0]) * 32U), block_size, 0, stream.get()>>>(
            rows,
            child_ids,
            child_count,
            targets.data(),
            offsets.data(),
            deltas.children.data(),
            deltas.records.data()
        );
    CUDDL_CUDA_TRY(cudaGetLastError());
    return std::optional<reference_deltas>{std::move(deltas)};
}

/// @brief Completes a refined tile from its bases' results: each (query, child) pair's counts
/// are its (query, base) counts adjusted for the child's changed buckets.
///
/// A block stages a group of queries bucket-major from their planes, then its warps take
/// children of a child tile; each warp copies a child's records to shared memory while the
/// next child's records load. Every pair (query, base) of the tile must already be refined.
template <size_t BucketCount, typename SearchResult, bool UpperTriangle>
__global__ __launch_bounds__(delta_correction_warps * 32U, 1) void apply_reference_deltas_kernel(
    uint32_t const* query_planes,
    uint32_t query_id_offset,
    uint32_t query_count,
    uint32_t reference_count,
    reference_delta_view deltas,
    SearchResult* results
) {
    constexpr uint32_t group = delta_query_group<BucketCount>;
    constexpr uint32_t slices = 32U / group;
    constexpr uint32_t record_registers = delta_change_limit / 32U;
    constexpr uint32_t bucket_groups = static_cast<uint32_t>(BucketCount / 32U);
    extern __shared__ uint16_t staged[];
    auto const lane = threadIdx.x % 32U;
    auto const warp = threadIdx.x / 32U;
    auto const member = lane % group;
    auto const slice = lane / group;
    auto const query_groups = (query_count + group - 1U) / group;
    auto const child_tiles = (deltas.child_count + delta_child_tile - 1U) / delta_child_tile;
    auto const* const row = reinterpret_cast<char const*>(staged) + member * sizeof(uint16_t);
    auto* const buffer =
        reinterpret_cast<uint2*>(staged + BucketCount * group) + warp * delta_change_limit;
    auto const load_records = [&](uint4 child, uint2(&values)[record_registers]) {
        _Pragma("unroll")
        for (uint32_t k = 0; k < record_registers; ++k) {
            auto const index = lane + k * 32U;
            values[k] = index < (child.w & 1023U) ? deltas.records[child.z + index] : uint2{};
        }
    };
    for (auto cell = blockIdx.x; cell < query_groups * child_tiles; cell += gridDim.x) {
        // Adjacent child tiles reuse a query group's base-result region.
        auto const first_member = (cell / child_tiles) * group;
        auto const members = cuda::std::min(group, query_count - first_member);
        auto const begin_child = (cell % child_tiles) * delta_child_tile;
        auto const end_child = cuda::std::min(deltas.child_count, begin_child + delta_child_tile);
        if constexpr (UpperTriangle) {
            // Children ascend, so a tile whose last child precedes the group has no pairs.
            if (deltas.children[end_child - 1U].x <= query_id_offset + first_member) {
                continue;
            }
        }
        __syncthreads();
        for (auto item = warp; item < group * bucket_groups; item += delta_correction_warps) {
            auto const m = item % group;
            auto const bucket_group = item / group;
            uint32_t score = 0U;
            if (m < members) {
                uint32_t words[score_planes];
                load_plane_group(
                    reinterpret_cast<uint4 const*>(
                        query_planes + static_cast<size_t>(first_member + m) * (BucketCount / 2U)
                    ),
                    bucket_group,
                    words
                );
                _Pragma("unroll")
                for (uint32_t p = 0; p < score_planes; ++p) {
                    score |= ((words[p] >> lane) & 1U) << p;
                }
            }
            staged[(bucket_group * 32U + lane) * group + m] = static_cast<uint16_t>(score);
        }
        __syncthreads();
        auto const query = first_member + member;
        auto child_index = begin_child + warp;
        if (child_index >= end_child) {
            continue;
        }
        auto info = deltas.children[child_index];
        uint2 pending[record_registers];
        load_records(info, pending);
        auto next_info = child_index + delta_correction_warps < end_child
                             ? deltas.children[child_index + delta_correction_warps]
                             : uint4{};
        for (; child_index < end_child; child_index += delta_correction_warps) {
            auto const count = info.w & 1023U;
            __syncwarp();
            _Pragma("unroll")
            for (uint32_t k = 0; k < record_registers; ++k) {
                if (lane + k * 32U < count) buffer[lane + k * 32U] = pending[k];
            }
            __syncwarp();
            auto const next_index = child_index + delta_correction_warps;
            if (next_index < end_child) load_records(next_info, pending);
            auto const following_info = next_index + delta_correction_warps < end_child
                                            ? deltas.children[next_index + delta_correction_warps]
                                            : uint4{};
            auto const child = info.x;
            auto included = member < members;
            if constexpr (UpperTriangle) {
                included = included && query_id_offset + query < child;
            }
            if (__any_sync(0xffffffffU, included)) {
                SearchResult base{};
                if (included && slice == 0U) {
                    base = results[batch_result_slot<UpperTriangle>(
                        query, info.y, query_id_offset, reference_count
                    )];
                }
                auto const zeros = (info.w >> 10U) & 1023U;
                auto const increases = info.w >> 20U;
                int lower = 0;
                int higher = 0;
                int empty_equal = 0;
                // A low-zero change moves the query's empty buckets between lower and both-empty
                // rather than equal, which the derived equal count corrects.
                for (auto i = slice; i < zeros; i += slices) {
                    auto const record = buffer[i];
                    uint32_t const value = *reinterpret_cast<uint16_t const*>(row + record.x);
                    auto const width = record.y >> 16U;
                    int const sign = (record.y & 1U) != 0U ? -1 : 1;
                    lower += value < width ? sign : 0;
                    higher -= value - 1U < width ? sign : 0;
                    empty_equal += value == 0U ? sign : 0;
                }
                // Counts query scores in [low, low + width) and in (low, low + width].
                auto const accumulate = [&](uint32_t begin, uint32_t end, int& inside, int& above) {
                    _Pragma("unroll 4")
                    for (auto i = begin + slice; i < end; i += slices) {
                        auto const record = buffer[i];
                        uint32_t const value = *reinterpret_cast<uint16_t const*>(row + record.x);
                        auto const offset = value - (record.y & 0xffffU);
                        auto const width = record.y >> 16U;
                        inside += offset < width;
                        above += offset - 1U < width;
                    }
                };
                int increase_inside = 0;
                int increase_above = 0;
                int decrease_inside = 0;
                int decrease_above = 0;
                accumulate(zeros, zeros + increases, increase_inside, increase_above);
                accumulate(zeros + increases, count, decrease_inside, decrease_above);
                lower += increase_inside - decrease_inside;
                higher += decrease_above - increase_above;
                _Pragma("unroll")
                for (uint32_t offset = group; offset < 32U; offset *= 2U) {
                    lower += __shfl_xor_sync(0xffffffffU, lower, offset);
                    higher += __shfl_xor_sync(0xffffffffU, higher, offset);
                    empty_equal += __shfl_xor_sync(0xffffffffU, empty_equal, offset);
                }
                if (included && slice == 0U) {
                    auto const counts = base.unpack(static_cast<uint32_t>(BucketCount));
                    results[batch_result_slot<UpperTriangle>(
                        query, child, query_id_offset, reference_count
                    )] =
                        SearchResult::pack(
                            static_cast<uint32_t>(static_cast<int>(counts.lower) + lower),
                            static_cast<uint32_t>(
                                static_cast<int>(counts.equal) - lower - higher + empty_equal
                            ),
                            static_cast<uint32_t>(static_cast<int>(counts.higher) + higher)
                        );
                }
            }
            info = next_info;
            next_info = following_info;
        }
    }
}

}  // namespace cuddl::detail
