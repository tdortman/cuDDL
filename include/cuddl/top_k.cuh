#pragma once

#include <cub/device/device_segmented_sort.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <algorithm>
#include <span>
#include <vector>

#include <cuddl/error.hpp>
#include <cuddl/reference_database.cuh>

namespace cuddl {

namespace detail {

/**
 * @brief Ranking key of a neighbour: its weighted k-mer identity above its complemented ID.
 *
 * Keys are unique, a larger key ranks higher, and equal identities rank the smaller ID higher,
 * so any selection of the k largest keys is the same set. Zero means no candidate; a real key
 * is never zero because IDs stay below `UINT32_MAX`.
 */
[[nodiscard]] __host__ __device__ constexpr uint64_t
top_k_key(pairwise_counts const& counts, uint32_t id) noexcept {
    auto const identity = cuda::std::bit_cast<uint32_t>(weighted_kmer_identity(counts));
    return (uint64_t{identity} << 32U) | static_cast<uint32_t>(~id);
}

/// Each genome's k best neighbours so far: keys and packed counts in parallel rows of k, each row
/// a min-heap of keys whose root is the row's worst key, zero while the row has room.
struct top_k_slots {
    uint64_t* keys;
    uint64_t* counts;
    uint32_t k;
    uint32_t first_genome_id;
};

/// Start of genome row @p i of a collector's heaps, rows of k slots.
struct top_k_row_offset {
    uint32_t k;

    __host__ __device__ int64_t operator()(int64_t i) const noexcept {
        return i * k;
    }
};

/// Warps per merging block, each holding one heap; fewer when the heaps exceed shared memory.
constexpr uint32_t top_k_warps = block_size / 32U;

/// Words of shared memory one heap takes: k keys, then their k packed counts.
[[nodiscard]] constexpr size_t top_k_heap_words(uint32_t k) noexcept {
    return size_t{k} * 2U;
}

/**
 * @brief Replaces the root of a min-heap of @p k keys at @p heap, whose values follow the keys,
 * with @p key and @p value, and sifts it down. The caller owns the heap and has checked that
 * @p key beats the root.
 */
__device__ inline void
top_k_replace_root(uint64_t* heap, uint32_t k, uint64_t key, uint64_t value) {
    auto* const values = heap + k;
    uint32_t at = 0U;
    for (auto child = 1U; child < k; child = 2U * at + 1U) {
        auto child_key = heap[child];
        if (child + 1U < k && heap[child + 1U] < child_key) child_key = heap[++child];
        if (child_key >= key) break;
        heap[at] = child_key;
        values[at] = values[child];
        at = child;
    }
    heap[at] = key;
    values[at] = value;
}

/// The pass bits of word @p word of tile row @p query: the stored bits, or for exhaustive tiles
/// every reference of the word. Pairs on or below the diagonal of all-to-all tiles are masked.
template <bool Thresholded, bool UpperTriangle>
__device__ inline uint32_t
top_k_word_bits(batch_result_tile const& tile, uint32_t query, uint32_t word, uint32_t words) {
    uint32_t bits = 0U;
    if constexpr (Thresholded) {
        bits = __ldg(tile.pass_bits + static_cast<size_t>(query) * words + word);
    } else {
        auto const left = tile.reference_count - word * 32U;
        bits = left >= 32U ? ~0U : (1U << left) - 1U;
    }
    if constexpr (UpperTriangle) {
        auto const query_id = tile.first_query_id + query;
        if (word * 32U <= query_id) {
            auto const skipped = query_id - word * 32U + 1U;
            bits &= skipped >= 32U ? 0U : ~0U << skipped;
        }
    }
    return bits;
}

/// The stored result of tile row @p query and @p reference, which must pass.
template <bool UpperTriangle>
__device__ inline packed_pairwise_counts
top_k_stored(batch_result_tile const& tile, uint32_t query, uint32_t reference) {
    auto const slot = batch_result_slot<UpperTriangle>(
        query, reference, tile.first_query_id, tile.reference_count
    );
    return packed_pairwise_counts{__ldg(&tile.results[slot].bits)};
}

/// Position of set bit @p n, counting from zero, of @p mask, which has more than @p n set bits.
__device__ inline uint32_t top_k_nth_bit(uint32_t mask, uint32_t n) {
    uint32_t position = 0U;
    asm("fns.b32 %0, %1, 0, %2;" : "=r"(position) : "r"(mask), "r"(static_cast<int>(n + 1U)));
    return position;
}

/**
 * @brief One warp merges the candidates of @p words bitmap words into the heap of @p k entries
 * at @p heap, in shared memory, which only this warp updates, so it needs no lock.
 *
 * @p bits_of gives word w's candidates and @p entry_of candidate `w * 32 + bit`'s key and value.
 * The warp takes 32 words at a time and hands their set bits out 32 per step, one per lane in
 * order, so every step offers 32 candidates however the words are filled. Lanes test their
 * candidates against the root; the warp then inserts the survivors one at a time.
 */
template <typename Bits, typename Entry>
__device__ inline void
top_k_warp_merge(uint64_t* heap, uint32_t k, uint32_t words, Bits bits_of, Entry entry_of) {
    auto const lane = static_cast<uint32_t>(threadIdx.x) % 32U;
    for (uint32_t base = 0U; base < words; base += 32U) {
        auto const bits = base + lane < words ? bits_of(base + lane) : 0U;
        // Inclusive prefix of the lanes' set bits, so candidate j is in the first lane whose
        // prefix exceeds j.
        auto const count = static_cast<uint32_t>(cuda::std::popcount(bits));
        auto prefix = count;
        _Pragma("unroll")
        for (uint32_t offset = 1U; offset < 32U; offset *= 2U) {
            auto const below = __shfl_up_sync(~0U, prefix, offset);
            if (lane >= offset) prefix += below;
        }
        auto const total = __shfl_sync(~0U, prefix, 31);
        for (uint32_t first = 0U; first < total; first += 32U) {
            auto const candidate = first + lane;
            // Every lane takes part in the search's shuffles, with or without a candidate.
            uint32_t source = 0U;
            _Pragma("unroll")
            for (uint32_t step = 16U; step != 0U; step /= 2U) {
                if (__shfl_sync(~0U, prefix, source + step - 1U) <= candidate) source += step;
            }
            auto const source_bits = __shfl_sync(~0U, bits, source);
            auto const before = __shfl_sync(~0U, prefix - count, source);
            uint64_t key = 0U, value = 0U;
            if (candidate < total) {
                entry_of(
                    (base + source) * 32U + top_k_nth_bit(source_bits, candidate - before),
                    key,
                    value
                );
            }
            for (auto pending = __ballot_sync(~0U, key > heap[0]); pending != 0U;
                 pending &= pending - 1U) {
                if (lane == static_cast<uint32_t>(__ffs(pending) - 1) && key > heap[0]) {
                    top_k_replace_root(heap, k, key, value);
                }
                __syncwarp();
            }
        }
    }
}

/// Copies the @p k entries of global heap @p row into, or for @p ToShared false out of, the
/// shared heap @p heap with the warp's lanes.
template <bool ToShared>
__device__ inline void top_k_warp_copy(top_k_slots const& slots, size_t row, uint64_t* heap) {
    auto const k = slots.k;
    for (auto i = static_cast<uint32_t>(threadIdx.x) % 32U; i < k; i += 32U) {
        if constexpr (ToShared) {
            heap[i] = slots.keys[row * k + i];
            heap[k + i] = slots.counts[row * k + i];
        } else {
            slots.keys[row * k + i] = heap[i];
            slots.counts[row * k + i] = heap[k + i];
        }
    }
    __syncwarp();
}

/// One warp per tile row merges the row's passing pairs into the row genome's heap.
template <bool Thresholded, bool UpperTriangle>
__global__ __launch_bounds__(
    block_size
) void top_k_rows_kernel(batch_result_tile tile, top_k_slots slots) {
    extern __shared__ uint64_t heaps[];
    auto const warp = static_cast<uint32_t>(threadIdx.x) / 32U;
    auto const query = static_cast<uint32_t>(blockIdx.x) * (blockDim.x / 32U) + warp;
    if (query >= tile.query_count) return;
    auto* const heap = heaps + warp * top_k_heap_words(slots.k);
    auto const query_id = tile.first_query_id + query;
    auto const row = static_cast<size_t>(query_id - slots.first_genome_id);
    top_k_warp_copy<true>(slots, row, heap);
    auto const words = tile.words_per_query();
    // All-to-all rows start after the diagonal's word.
    auto const skipped = UpperTriangle ? (query_id + 1U) / 32U : 0U;
    top_k_warp_merge(
        heap,
        slots.k,
        words - cuda::std::min(skipped, words),
        [&](uint32_t word) {
            return top_k_word_bits<Thresholded, UpperTriangle>(tile, query, skipped + word, words);
        },
        [&](uint32_t candidate, uint64_t& key, uint64_t& value) {
            auto const reference = skipped * 32U + candidate;
            auto const stored = top_k_stored<UpperTriangle>(tile, query, reference);
            key = top_k_key(stored.unpack(tile.bucket_count), reference);
            value = stored.bits;
        }
    );
    top_k_warp_copy<false>(slots, row, heap);
}

/**
 * @brief Transposes an all-to-all tile's pass bits from @p first_word on into per-genome
 * columns: word `w * reference_count + g` of @p columns holds which of tile rows `32 w` to
 * `32 w + 31` pass with genome g. One warp per 32-by-32 bit block, which its lanes load a row
 * each and turn with 32 ballots; neighbouring warps take neighbouring words of the same rows, so
 * they share the rows' sectors, and every warp writes 32 adjacent words.
 */
template <bool Thresholded>
__global__ __launch_bounds__(
    block_size
) void top_k_transpose_kernel(batch_result_tile tile, uint32_t* columns, uint32_t first_word) {
    auto const lane = static_cast<uint32_t>(threadIdx.x) % 32U;
    auto const block = static_cast<uint64_t>(blockIdx.x) * (blockDim.x / 32U) + threadIdx.x / 32U;
    auto const words = tile.words_per_query();
    auto const word_count = words - first_word;
    auto const row_words = (tile.query_count + 31U) / 32U;
    if (block >= static_cast<uint64_t>(row_words) * word_count) return;
    auto const word = first_word + static_cast<uint32_t>(block % word_count);
    auto const row_word = static_cast<uint32_t>(block / word_count);
    auto const query = row_word * 32U + lane;
    auto const bits = query < tile.query_count
                          ? top_k_word_bits<Thresholded, true>(tile, query, word, words)
                          : 0U;
    uint32_t column = 0U;
    _Pragma("unroll")
    for (uint32_t bit = 0U; bit < 32U; ++bit) {
        auto const passing = __ballot_sync(~0U, ((bits >> bit) & 1U) != 0U);
        if (lane == bit) column = passing;
    }
    auto const genome = word * 32U + lane;
    if (genome < tile.reference_count) {
        columns[static_cast<size_t>(row_word) * tile.reference_count + genome] = column;
    }
}

/**
 * @brief All-to-all tiles: one warp per genome from @p first_genome merges the mirrored pairs of
 * the tile rows before it, read from @p columns as @ref top_k_transpose_kernel wrote them, into
 * its heap.
 */
__global__ __launch_bounds__(block_size) void top_k_columns_kernel(
    batch_result_tile tile,
    top_k_slots slots,
    uint32_t const* columns,
    uint32_t first_genome
) {
    extern __shared__ uint64_t heaps[];
    auto const warp = static_cast<uint32_t>(threadIdx.x) / 32U;
    auto const genome =
        first_genome + static_cast<uint32_t>(blockIdx.x) * (blockDim.x / 32U) + warp;
    if (genome >= tile.reference_count) return;
    auto* const heap = heaps + warp * top_k_heap_words(slots.k);
    top_k_warp_copy<true>(slots, genome, heap);
    // Rows before the genome, whose pairs with it the tile stores in their rows.
    auto const rows = cuda::std::min(tile.query_count, genome - tile.first_query_id);
    top_k_warp_merge(
        heap,
        slots.k,
        (rows + 31U) / 32U,
        [&](uint32_t row_word) {
            return __ldg(columns + static_cast<size_t>(row_word) * tile.reference_count + genome);
        },
        [&](uint32_t query, uint64_t& key, uint64_t& value) {
            auto const counts =
                top_k_stored<true>(tile, query, genome).unpack(tile.bucket_count).mirrored();
            key = top_k_key(counts, tile.first_query_id + query);
            value = packed_pairwise_counts::pack(counts.lower, counts.equal, counts.higher).bits;
        }
    );
    top_k_warp_copy<false>(slots, genome, heap);
}

}  // namespace detail

/**
 * @brief Keeps each genome's @p k most similar genomes from search tiles on the device.
 *
 * Pass @ref add as the tile consumer of `search_batch_async` or `search_all_to_all_async`, for
 * either the exhaustive or the indexed search, in place of downloading every passing pair. A
 * genome's neighbours rank by weighted k-mer identity, the smaller ID first on a tie; a genome
 * with fewer than @p k passing pairs keeps all of them.
 *
 * Batch searches keep each query's best references, and all-to-all searches each database
 * row's best other rows, in both directions of the stored upper triangle. Each genome keeps a
 * min-heap of its k best. Per tile, one warp per query row merges the row's passing pairs into
 * its genome's heap; for all-to-all tiles the pass bits are then transposed, and one warp per
 * later genome merges its column of mirrored pairs. Each merging warp holds its genome's heap
 * in shared memory and alone updates it, so no update takes a lock, and the work follows the
 * passing pairs.
 *
 * A collector sees only the tiles given to it. Replicas sharing an all-to-all tile queue each
 * keep their own, and @ref select_top_k merges their downloads.
 */
class top_k_search {
   public:
    /**
     * @param genome_count Queries of a batch search, or rows of an all-to-all search.
     * @param reference_count Database rows, the length of every result row.
     * @param bucket_count Buckets per sketch, from which downloads recover both-empty counts.
     * @param first_genome_id ID of the first query of a batch search; zero for all-to-all.
     *
     * Blocks use as many warps as their heaps fit the shared memory @p stream's device gives a
     * block. Fails when one heap does not fit.
     */
    [[nodiscard]] static Result<top_k_search> make(
        uint32_t genome_count,
        uint32_t reference_count,
        uint32_t bucket_count,
        uint32_t k,
        cuda::stream_ref stream,
        uint32_t first_genome_id = 0U
    ) {
        if (k == 0U) return Err(Error::invalid_argument("top-k search needs k of at least 1"));
        detail::device_scope const scope{stream};
        int shared_limit = 0;
        CUDDL_CUDA_TRY(cudaDeviceGetAttribute(
            &shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, stream.device().get()
        ));
        auto const heap_bytes = detail::top_k_heap_words(k) * sizeof(uint64_t);
        auto const warps = static_cast<uint32_t>(
            std::min<size_t>(detail::top_k_warps, static_cast<size_t>(shared_limit) / heap_bytes)
        );
        if (warps == 0U) {
            return Err(Error::invalid_argument("top-k heap exceeds this GPU's shared memory"));
        }
        auto const bytes = warps * heap_bytes;
        CUDDL_TRY(allow_shared(detail::top_k_rows_kernel<false, false>, bytes));
        CUDDL_TRY(allow_shared(detail::top_k_rows_kernel<true, false>, bytes));
        CUDDL_TRY(allow_shared(detail::top_k_rows_kernel<false, true>, bytes));
        CUDDL_TRY(allow_shared(detail::top_k_rows_kernel<true, true>, bytes));
        CUDDL_TRY(allow_shared(detail::top_k_columns_kernel, bytes));
        auto const slots = static_cast<size_t>(genome_count) * k;
        return CUDDL_CUDA_TRY(top_k_search(
            genome_count,
            reference_count,
            bucket_count,
            k,
            first_genome_id,
            warps,
            cuda::make_device_buffer<uint64_t>(stream, stream.device(), slots, uint64_t{0U}),
            cuda::make_device_buffer<uint64_t>(stream, stream.device(), slots, cuda::no_init),
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 0U, cuda::no_init)
        ));
    }

    /// @brief Merges @p tile's passing pairs into the running neighbours, on @p stream. The tile
    /// is one a search passes to its consumer, with results in their device slots.
    [[nodiscard]] Result<void> add(batch_result_tile const& tile, cuda::stream_ref stream) {
        detail::device_scope const scope{stream};
        if (tile.reference_count != reference_count_) {
            return Err(Error::invalid_argument("tile belongs to a database of another size"));
        }
        if (tile.query_count > genome_count_ || tile.first_query_id < first_genome_id_ ||
            tile.first_query_id - first_genome_id_ > genome_count_ - tile.query_count) {
            return Err(Error::invalid_argument("tile rows lie outside the collected genomes"));
        }
        if (tile.word_offsets != nullptr) {
            return Err(Error::invalid_argument("top-k search takes device tiles, not host copies"));
        }
        if (tile.upper_triangle && (first_genome_id_ != 0U || genome_count_ != reference_count_)) {
            return Err(
                Error::invalid_argument("an all-to-all tile needs a collector of every row")
            );
        }
        if (tile.query_count == 0U || reference_count_ == 0U) return Ok();
        detail::top_k_slots const slots{keys_.data(), counts_.data(), k_, first_genome_id_};
        auto const thresholded = tile.pass_bits != nullptr;
        auto const heap_bytes = detail::top_k_heap_words(k_) * sizeof(uint64_t);
        auto const rows = tile.upper_triangle
                              ? (thresholded ? detail::top_k_rows_kernel<true, true>
                                             : detail::top_k_rows_kernel<false, true>)
                              : (thresholded ? detail::top_k_rows_kernel<true, false>
                                             : detail::top_k_rows_kernel<false, false>);
        auto const shared = warps_ * heap_bytes;
        rows<<<(tile.query_count + warps_ - 1U) / warps_, warps_ * 32U, shared, stream.get()>>>(
            tile, slots
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        if (!tile.upper_triangle) return Ok();
        // Genomes after the tile's first row take its rows before them as candidates.
        auto const first_genome = tile.first_query_id + 1U;
        if (first_genome >= reference_count_) return Ok();
        auto const first_word = first_genome / 32U;
        auto const row_words = (tile.query_count + 31U) / 32U;
        auto const needed = static_cast<size_t>(row_words) * reference_count_;
        if (columns_.size() < needed) {
            columns_ = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint32_t>(stream, stream.device(), needed, cuda::no_init)
            );
        }
        // One warp per 32-by-32 block of the pass bits from the first genome's word on.
        auto const bit_blocks =
            static_cast<uint64_t>(row_words) * (tile.words_per_query() - first_word);
        auto const warps_per_block = detail::block_size / 32U;
        (thresholded ? detail::top_k_transpose_kernel<true>
                     : detail::top_k_transpose_kernel<false>)<<<
            static_cast<uint32_t>((bit_blocks + warps_per_block - 1U) / warps_per_block),
            detail::block_size,
            0,
            stream.get()>>>(tile, columns_.data(), first_word);
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const genomes = reference_count_ - first_genome;
        detail::top_k_columns_kernel<<<
            (genomes + warps_ - 1U) / warps_,
            warps_ * 32U,
            shared,
            stream.get()>>>(tile, slots, columns_.data(), first_genome);
        CUDDL_CUDA_TRY(cudaGetLastError());
        return Ok();
    }

    /// @brief Each genome's neighbours, best first, genome by genome; waits for @p stream.
    [[nodiscard]] Result<std::vector<batch_search_result>> download(cuda::stream_ref stream) const {
        detail::device_scope const scope{stream};
        auto const slots = keys_.size();
        std::vector<uint64_t> keys(slots), counts(slots);
        if (slots != 0U) {
            // Rows are heaps on the device; sorting them there leaves the host a linear pass.
            auto sorted_keys = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint64_t>(stream, stream.device(), slots, cuda::no_init)
            );
            auto sorted_counts = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint64_t>(stream, stream.device(), slots, cuda::no_init)
            );
            auto const offsets = cuda::make_transform_iterator(
                cuda::counting_iterator<int64_t>{0}, detail::top_k_row_offset{k_}
            );
            CUDDL_CUDA_TRY(
                cub::DeviceSegmentedSort::SortPairsDescending(
                    keys_.data(),
                    sorted_keys.data(),
                    counts_.data(),
                    sorted_counts.data(),
                    static_cast<int64_t>(slots),
                    static_cast<int64_t>(genome_count_),
                    offsets,
                    offsets + 1,
                    stream
                )
            );
            CUDDL_CUDA_TRY(cuda::copy_bytes(stream, sorted_keys, cuda::std::span{keys}));
            CUDDL_CUDA_TRY(cuda::copy_bytes(stream, sorted_counts, cuda::std::span{counts}));
        }
        CUDDL_CUDA_TRY(stream.sync());
        std::vector<batch_search_result> rows;
        rows.reserve(slots);
        for (size_t slot = 0U; slot < slots; ++slot) {
            if (keys[slot] == 0U) continue;
            rows.push_back({
                .query_id = first_genome_id_ + static_cast<uint32_t>(slot / k_),
                .reference_id = ~static_cast<uint32_t>(keys[slot]),
                .counts = packed_pairwise_counts{counts[slot]}.unpack(bucket_count_),
            });
        }
        return rows;
    }

    [[nodiscard]] uint32_t k() const noexcept {
        return k_;
    }

   private:
    top_k_search(
        uint32_t genome_count,
        uint32_t reference_count,
        uint32_t bucket_count,
        uint32_t k,
        uint32_t first_genome_id,
        uint32_t warps,
        cuda::device_buffer<uint64_t> keys,
        cuda::device_buffer<uint64_t> counts,
        cuda::device_buffer<uint32_t> columns
    )
        : genome_count_(genome_count),
          reference_count_(reference_count),
          bucket_count_(bucket_count),
          k_(k),
          first_genome_id_(first_genome_id),
          warps_(warps),
          keys_(std::move(keys)),
          counts_(std::move(counts)),
          columns_(std::move(columns)) {}

    /// Lets @p kernel take @p bytes of dynamic shared memory on the current device; launches
    /// above the default 48 KiB need it.
    template <typename Kernel>
    [[nodiscard]] static Result<void> allow_shared(Kernel kernel, size_t bytes) {
        return cuda_try(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)
        ));
    }

    uint32_t genome_count_;
    uint32_t reference_count_;
    uint32_t bucket_count_;
    uint32_t k_;
    uint32_t first_genome_id_;
    /// Warps per merging block, each holding one heap.
    uint32_t warps_;
    cuda::device_buffer<uint64_t> keys_, counts_;
    /// The last all-to-all tile's transposed pass bits; grows with the tallest tile.
    cuda::device_buffer<uint32_t> columns_;
};

/**
 * @brief The @p k best rows of each query among @p rows, as @ref top_k_search ranks them, best
 * first and query by query. Merges the downloads of collectors that saw disjoint tiles.
 */
[[nodiscard]] inline std::vector<batch_search_result>
select_top_k(std::vector<batch_search_result> rows, uint32_t k) {
    std::ranges::sort(rows, [](batch_search_result const& a, batch_search_result const& b) {
        if (a.query_id != b.query_id) return a.query_id < b.query_id;
        return detail::top_k_key(a.counts, a.reference_id) >
               detail::top_k_key(b.counts, b.reference_id);
    });
    std::vector<batch_search_result> kept;
    for (size_t first = 0U; first < rows.size();) {
        auto last = first;
        while (last < rows.size() && rows[last].query_id == rows[first].query_id) ++last;
        kept.insert(
            kept.end(),
            rows.begin() + static_cast<std::ptrdiff_t>(first),
            rows.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(last, first + k))
        );
        first = last;
    }
    return kept;
}

}  // namespace cuddl
