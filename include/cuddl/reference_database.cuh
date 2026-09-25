#pragma once

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/optional>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl {

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class reference_index;

/// @brief Dense offsets favor query latency; sparse sorted keys reduce index memory and build time.
enum class index_storage { dense, sparse };

/// @brief Construction parameters of compatible score rows.
struct score_compatibility {
    uint32_t kmer_length{};
    uint32_t bucket_count{};
    uint32_t indexed_bucket_count{};
    uint32_t score_encoder_identity{};
    uint16_t exponent_bits{};
    uint16_t mantissa_bits{};
    uint32_t hash_identity{};
    uint64_t hash_seed{};
    uint32_t canonicalisation_policy{};
    uint64_t blacklist_identity{};
    uint32_t blacklist_version{};
    uint16_t key_mask{};

    /// @brief Metadata for score rows produced by the current cuDDL construction path.
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static constexpr score_compatibility current() noexcept {
        static_assert(BucketCount <= std::numeric_limits<uint32_t>::max());
        return {
            .kmer_length = K,
            .bucket_count = static_cast<uint32_t>(BucketCount),
            .indexed_bucket_count = static_cast<uint32_t>(BucketCount),
            .score_encoder_identity = 1U,
            .exponent_bits = static_cast<uint16_t>(Layout::exponent_bits),
            .mantissa_bits = static_cast<uint16_t>(Layout::mantissa_bits),
            .hash_identity = 1U,
            .hash_seed = detail::seed,
            .canonicalisation_policy = 1U,
            .blacklist_identity = 0U,
            .blacklist_version = 0U,
            // Genomic winner scores virtually never set the top bit, so folding it halves the
            // dense index at no practical cost in selectivity.
            .key_mask = 0x7fffU,
        };
    }

    friend bool operator==(score_compatibility const&, score_compatibility const&) = default;
};

/// @brief Recorded metadata for an immutable reference database.
struct reference_database_metadata {
    score_compatibility compatibility{};
    uint32_t reference_count{};

    friend bool operator==(reference_database_metadata const&, reference_database_metadata const&) =
        default;
};

/// @brief Stable reference ID and exact query-relative bucket comparison counts.
struct reference_search_result {
    uint32_t reference_id{};
    pairwise_counts counts{};

    friend bool operator==(reference_search_result const&, reference_search_result const&) =
        default;
};

/// @brief Stable query/reference IDs and their exact query-relative bucket comparison counts.
struct batch_search_result {
    uint32_t query_id{};
    uint32_t reference_id{};
    pairwise_counts counts{};

    friend bool operator==(batch_search_result const&, batch_search_result const&) = default;
};

/**
 * @brief Non-owning view of one batch-search tile's results.
 *
 * Results are stored as @ref packed_pairwise_counts; a result's position identifies its pair, so
 * the accessors return full @ref batch_search_result values. On the device, results sit at fixed
 * slots: one row of `reference_count` slots per query, or for all-to-all tiles the packed strict
 * upper triangle of pairs with `query_id < reference_id`. Exhaustive tiles fill every slot.
 * Threshold (indexed) tiles fill only the passing pairs' slots and mark them in @ref pass_bits,
 * one bit per reference in rows of `(reference_count + 31) / 32` words per query. Host copies
 * (@ref download) hold only the passing results, packed in order, and locate them through
 * @ref word_offsets.
 */
struct batch_result_tile {
    packed_pairwise_counts const* results{};
    /// Index match counts parallel to @ref results, when requested; otherwise null.
    uint32_t const* match_counts{};
    /// Passing pairs, when the tile applied a threshold; null when every slot is filled.
    uint32_t const* pass_bits{};
    /// For packed results: the number of passing results before each pass-bit word.
    uint32_t const* word_offsets{};
    uint32_t first_query_id{};
    uint32_t query_count{};
    uint32_t reference_count{};
    /// Buckets per sketch, from which both-empty counts are recovered.
    uint32_t bucket_count{};
    bool upper_triangle{};

    [[nodiscard]] __host__ __device__ constexpr uint32_t words_per_query() const noexcept {
        return (reference_count + 31U) / 32U;
    }

    /// @brief Slots of the tile's layout, filled or not.
    [[nodiscard]] __host__ __device__ uint64_t slot_count() const noexcept {
        if (query_count == 0U) return 0U;
        auto const last = query_count - 1U;
        if (upper_triangle) {
            return first_query_id + last + 1U >= reference_count
                       ? slot(last, reference_count)
                       : slot(last, reference_count - 1U) + 1U;
        }
        return static_cast<uint64_t>(query_count) * reference_count;
    }

    /// @brief Whether the pair has a result in this tile.
    [[nodiscard]] __host__ __device__ bool
    passed(uint32_t query_id, uint32_t reference_id) const noexcept {
        if (query_id < first_query_id || query_id - first_query_id >= query_count ||
            reference_id >= reference_count || (upper_triangle && reference_id <= query_id)) {
            return false;
        }
        if (pass_bits == nullptr) return true;
        return ((pass_bits[word(query_id, reference_id)] >> (reference_id % 32U)) & 1U) != 0U;
    }

    /// @brief The pair's result, or nothing when it did not pass.
    [[nodiscard]] __host__ __device__ cuda::std::optional<batch_search_result>
    find(uint32_t query_id, uint32_t reference_id) const noexcept {
        if (!passed(query_id, reference_id)) return cuda::std::nullopt;
        return result(query_id, reference_id, results[position(query_id, reference_id)]);
    }

    /// @brief The pair's index match count; the pair must pass and counts must be present.
    [[nodiscard]] __host__ __device__ uint32_t
    match_count(uint32_t query_id, uint32_t reference_id) const noexcept {
        return match_counts[position(query_id, reference_id)];
    }

    /// @brief Calls @p f with every passing result, query-major and in reference order.
    template <typename F>
    __host__ __device__ void for_each_passing(F&& f) const {
        uint64_t next = 0U;
        for (uint32_t query = 0; query < query_count; ++query) {
            auto const query_id = first_query_id + query;
            if (pass_bits == nullptr) {
                // Every slot passes, and slots are in this order.
                for (auto reference = upper_triangle ? query_id + 1U : 0U;
                     reference < reference_count;
                     ++reference) {
                    f(result(query_id, reference, results[next++]));
                }
                continue;
            }
            auto const words = words_per_query();
            for (uint32_t w = 0; w < words; ++w) {
                for (auto bits = pass_bits[static_cast<size_t>(query) * words + w]; bits != 0U;
                     bits &= bits - 1U) {
                    auto const reference =
                        w * 32U + static_cast<uint32_t>(cuda::std::countr_zero(bits));
                    auto const at = word_offsets != nullptr ? next++ : slot(query, reference);
                    f(result(query_id, reference, results[at]));
                }
            }
        }
    }

    /// @brief Number of passing results.
    [[nodiscard]] __host__ __device__ uint64_t count() const noexcept {
        if (pass_bits == nullptr) return slot_count();
        auto const words = static_cast<size_t>(query_count) * words_per_query();
        if (words == 0U) return 0U;
        if (word_offsets != nullptr) {
            return static_cast<uint64_t>(word_offsets[words - 1U]) +
                   static_cast<uint64_t>(cuda::std::popcount(pass_bits[words - 1U]));
        }
        uint64_t total = 0U;
        for (size_t w = 0; w < words; ++w) {
            total += static_cast<uint64_t>(cuda::std::popcount(pass_bits[w]));
        }
        return total;
    }

   private:
    [[nodiscard]] __host__ __device__ batch_search_result
    result(uint32_t query_id, uint32_t reference_id, packed_pairwise_counts counts) const noexcept {
        return {
            .query_id = query_id,
            .reference_id = reference_id,
            .counts = counts.unpack(bucket_count),
        };
    }

    [[nodiscard]] __host__ __device__ size_t
    word(uint32_t query_id, uint32_t reference_id) const noexcept {
        return static_cast<size_t>(query_id - first_query_id) * words_per_query() +
               reference_id / 32U;
    }

    [[nodiscard]] __host__ __device__ uint64_t
    position(uint32_t query_id, uint32_t reference_id) const noexcept {
        if (word_offsets != nullptr) {
            auto const w = word(query_id, reference_id);
            auto const below = (1U << (reference_id % 32U)) - 1U;
            return static_cast<uint64_t>(word_offsets[w]) +
                   static_cast<uint64_t>(cuda::std::popcount(pass_bits[w] & below));
        }
        return slot(query_id - first_query_id, reference_id);
    }

    [[nodiscard]] __host__ __device__ uint64_t
    slot(uint32_t query, uint32_t reference_id) const noexcept {
        return upper_triangle ? detail::batch_result_slot<true>(
                                    query, reference_id, first_query_id, reference_count
                                )
                              : detail::batch_result_slot<false>(
                                    query, reference_id, first_query_id, reference_count
                                );
    }
};

/**
 * @brief Host copy of a @ref batch_result_tile holding only the passing results, packed in
 * query-major, reference order, in pinned memory; @ref tile views it.
 */
class host_batch_results {
   public:
    host_batch_results(host_batch_results const&) = delete;
    host_batch_results& operator=(host_batch_results const&) = delete;

    __host__ host_batch_results(host_batch_results&& other) noexcept
        : results_(std::move(other.results_)),
          match_counts_(std::move(other.match_counts_)),
          pass_bits_(std::move(other.pass_bits_)),
          word_offsets_(std::move(other.word_offsets_)),
          count_(std::exchange(other.count_, 0U)),
          layout_(other.layout_) {}

    __host__ ~host_batch_results() {}

    /// @brief Every passing result, query-major and in reference order, as full values.
    ///
    /// This builds a 24-byte value per result; @ref tile and `for_each_passing` read the 8-byte
    /// packed results without the copy.
    [[nodiscard]] std::vector<batch_search_result> passing() const {
        std::vector<batch_search_result> kept;
        kept.reserve(static_cast<size_t>(count_));
        tile().for_each_passing([&](batch_search_result const& result) { kept.push_back(result); });
        return kept;
    }

    /// @brief The passing results' index match counts, in @ref passing order; empty unless
    /// match counts were requested.
    // A span into a temporary copy would dangle, so rvalues are rejected.
    [[nodiscard]] std::span<uint32_t const> passing_match_counts() const& noexcept {
        return {
            match_counts_.data(), match_counts_.size() == 0U ? 0U : static_cast<size_t>(count_)
        };
    }
    void passing_match_counts() const&& = delete;

    /// @brief View of the packed host results; valid while this object lives.
    [[nodiscard]] batch_result_tile tile() const& noexcept {
        auto view = layout_;
        view.results = results_.data();
        view.match_counts = match_counts_.size() == 0U ? nullptr : match_counts_.data();
        view.pass_bits = layout_.pass_bits == nullptr ? nullptr : pass_bits_.data();
        view.word_offsets = layout_.pass_bits == nullptr ? nullptr : word_offsets_.data();
        return view;
    }
    void tile() const&& = delete;

   private:
    template <typename T>
    using pinned_buffer = cuda::buffer<T, cuda::mr::host_accessible, cuda::mr::device_accessible>;

    explicit host_batch_results(cuda::stream_ref stream)
        : results_(stream, cuda::pinned_default_memory_pool(), 0, cuda::no_init),
          match_counts_(stream, cuda::pinned_default_memory_pool(), 0, cuda::no_init),
          pass_bits_(stream, cuda::pinned_default_memory_pool(), 0, cuda::no_init),
          word_offsets_(stream, cuda::pinned_default_memory_pool(), 0, cuda::no_init) {}

    friend Result<host_batch_results> download(batch_result_tile const&, cuda::stream_ref);

    pinned_buffer<packed_pairwise_counts> results_;
    pinned_buffer<uint32_t> match_counts_;
    pinned_buffer<uint32_t> pass_bits_;
    pinned_buffer<uint32_t> word_offsets_;
    uint64_t count_{};
    batch_result_tile layout_{};
};

/**
 * @brief Copies a device tile's passing results to pinned host memory, synchronizing @p stream.
 *
 * Exhaustive tiles are copied as they are, since every slot passes in order. Threshold tiles are
 * packed on the device: a scan of the pass-bit words gives each word's first packed position, and
 * a gather writes the passing records straight into the host buffer.
 */
[[nodiscard]] inline Result<host_batch_results>
download(batch_result_tile const& tile, cuda::stream_ref stream) {
    host_batch_results host(stream);
    host.layout_ = tile;
    host.layout_.word_offsets = nullptr;
    auto const memory = cuda::pinned_default_memory_pool();
    using pinned_results = host_batch_results::pinned_buffer<packed_pairwise_counts>;
    using pinned_words = host_batch_results::pinned_buffer<uint32_t>;
    if (tile.pass_bits == nullptr) {
        auto const slots = static_cast<size_t>(tile.slot_count());
        host.count_ = slots;
        host.results_ = pinned_results(stream, memory, slots, cuda::no_init);
        if (slots != 0U) {
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream, cuda::std::span{tile.results, slots}, cuda::std::span{host.results_}
                )
            );
        }
        if (tile.match_counts != nullptr) {
            host.match_counts_ = pinned_words(stream, memory, slots, cuda::no_init);
            if (slots != 0U) {
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{tile.match_counts, slots},
                        cuda::std::span{host.match_counts_}
                    )
                );
            }
        }
        CUDDL_CUDA_TRY(stream.sync());
        return host;
    }

    auto const words = static_cast<size_t>(tile.query_count) * tile.words_per_query();
    host.pass_bits_ = pinned_words(stream, memory, words, cuda::no_init);
    host.word_offsets_ = pinned_words(stream, memory, words, cuda::no_init);
    if (words == 0U) {
        CUDDL_CUDA_TRY(stream.sync());
        return host;
    }
    auto const popcounts = cuda::make_transform_iterator(tile.pass_bits, detail::word_popcount{});
    size_t scan_bytes = 0;
    CUDDL_CUDA_TRY(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_bytes,
            popcounts,
            host.word_offsets_.data(),
            static_cast<int64_t>(words),
            stream.get()
        )
    );
    auto scan_storage = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), scan_bytes, cuda::no_init)
    );
    CUDDL_CUDA_TRY(
        cub::DeviceScan::ExclusiveSum(
            scan_storage.data(),
            scan_bytes,
            popcounts,
            host.word_offsets_.data(),
            static_cast<int64_t>(words),
            stream.get()
        )
    );
    CUDDL_CUDA_TRY(
        cuda::copy_bytes(
            stream, cuda::std::span{tile.pass_bits, words}, cuda::std::span{host.pass_bits_}
        )
    );
    CUDDL_CUDA_TRY(stream.sync());
    host.count_ = static_cast<uint64_t>(host.word_offsets_.data()[words - 1U]) +
                  static_cast<uint64_t>(cuda::std::popcount(host.pass_bits_.data()[words - 1U]));
    auto const count = static_cast<size_t>(host.count_);
    host.results_ = pinned_results(stream, memory, count, cuda::no_init);
    if (tile.match_counts != nullptr) {
        host.match_counts_ = pinned_words(stream, memory, count, cuda::no_init);
    }
    if (count == 0U) {
        CUDDL_CUDA_TRY(stream.sync());
        return host;
    }
    // Passing records are gathered into bounded device staging, so each chunk crosses the bus as
    // one bulk copy rather than as scattered small writes.
    constexpr size_t staging_records = size_t{1} << 23U;
    auto const staged = std::min(count, staging_records);
    auto staging = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<packed_pairwise_counts>(
            stream, stream.device(), staged, cuda::no_init
        )
    );
    auto staging_matches = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<uint32_t>(
            stream, stream.device(), tile.match_counts == nullptr ? 0U : staged, cuda::no_init
        )
    );
    auto const kernel = tile.upper_triangle
                            ? detail::gather_passing_kernel<true, packed_pairwise_counts>
                            : detail::gather_passing_kernel<false, packed_pairwise_counts>;
    auto const* offsets = host.word_offsets_.data();
    for (size_t first_word = 0U; first_word < words;) {
        auto const base = offsets[first_word];
        // The last word whose results all fit the staging buffer ends the chunk.
        auto const last_word = static_cast<size_t>(
            std::upper_bound(offsets + first_word + 1U, offsets + words, base + staged - 32U) -
            offsets
        );
        auto const end = last_word == words ? count : static_cast<size_t>(offsets[last_word]);
        kernel<<<
            detail::warp_grid_blocks(last_word - first_word),
            detail::block_size,
            0,
            stream.get()>>>(
            tile.results,
            tile.match_counts,
            tile.pass_bits,
            host.word_offsets_.data(),
            tile.first_query_id,
            tile.reference_count,
            first_word,
            last_word,
            base,
            staging.data(),
            tile.match_counts == nullptr ? nullptr : staging_matches.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const records = end - base;
        if (records != 0U) {
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream,
                    cuda::std::span{staging.data(), records},
                    cuda::std::span{host.results_.data() + base, records}
                )
            );
            if (tile.match_counts != nullptr) {
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{staging_matches.data(), records},
                        cuda::std::span{host.match_counts_.data() + base, records}
                    )
                );
            }
        }
        first_word = last_word;
    }
    CUDDL_CUDA_TRY(stream.sync());
    return host;
}

namespace detail {

/// @brief Records per chunk that @ref cuddl::for_each_passing moves to the host at once.
inline constexpr size_t visit_chunk_records = size_t{1} << 20U;

/// @brief One chunk of a tile's passing results: exhaustive tiles split at query rows, threshold
/// tiles at pass-bit words. Packed positions `[first_record, end_record)` hold its results.
struct visit_chunk {
    size_t first{};
    size_t last{};
    size_t first_record{};
    size_t end_record{};
};

}  // namespace detail

/**
 * @brief Calls @p f on the host with every passing result of a device tile, query-major and in
 * reference order; synchronizes @p stream.
 *
 * Results cross the bus in chunks through two pinned buffers, so the copy of chunk k + 1 runs
 * while @p f visits chunk k. Threshold tiles are packed on the device first, so only passing
 * results are copied. This streams where @ref download materializes the whole tile.
 */
template <typename F>
[[nodiscard]] Result<void>
for_each_passing(batch_result_tile const& tile, cuda::stream_ref stream, F&& f) {
    using pinned_counts = cuda::
        buffer<packed_pairwise_counts, cuda::mr::host_accessible, cuda::mr::device_accessible>;
    auto const memory = cuda::pinned_default_memory_pool();
    constexpr auto chunk_records = detail::visit_chunk_records;
    auto const threshold = tile.pass_bits != nullptr;
    auto const words = static_cast<size_t>(tile.query_count) * tile.words_per_query();

    // Chunk plan, and for threshold tiles the host copies of the pass bits and word offsets.
    std::vector<detail::visit_chunk> chunks;
    std::vector<uint32_t> pass_bits;
    std::vector<uint32_t> offsets;
    cuda::device_buffer<uint32_t> device_offsets(
        stream, cuda::device_default_memory_pool(stream.device())
    );
    if (!threshold) {
        size_t record = 0U;
        detail::visit_chunk chunk{};
        for (uint32_t query = 0; query < tile.query_count; ++query) {
            auto const query_id = tile.first_query_id + query;
            auto const row = static_cast<size_t>(
                tile.upper_triangle
                    ? (tile.reference_count > query_id + 1U ? tile.reference_count - query_id - 1U
                                                            : 0U)
                    : tile.reference_count
            );
            if (record + row - chunk.first_record > chunk_records && query != chunk.first) {
                chunk.last = query;
                chunk.end_record = record;
                chunks.push_back(chunk);
                chunk = {.first = query, .first_record = record};
            }
            record += row;
        }
        chunk.last = tile.query_count;
        chunk.end_record = record;
        if (record != 0U) chunks.push_back(chunk);
    } else if (words != 0U) {
        device_offsets = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), words, cuda::no_init)
        );
        auto const popcounts =
            cuda::make_transform_iterator(tile.pass_bits, detail::word_popcount{});
        size_t scan_bytes = 0;
        CUDDL_CUDA_TRY(
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                scan_bytes,
                popcounts,
                device_offsets.data(),
                static_cast<int64_t>(words),
                stream.get()
            )
        );
        auto scan_storage = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), scan_bytes, cuda::no_init)
        );
        CUDDL_CUDA_TRY(
            cub::DeviceScan::ExclusiveSum(
                scan_storage.data(),
                scan_bytes,
                popcounts,
                device_offsets.data(),
                static_cast<int64_t>(words),
                stream.get()
            )
        );
        pass_bits.resize(words);
        offsets.resize(words);
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream, cuda::std::span{tile.pass_bits, words}, cuda::std::span{pass_bits}
            )
        );
        CUDDL_CUDA_TRY(cuda::copy_bytes(stream, device_offsets, cuda::std::span{offsets}));
        CUDDL_CUDA_TRY(stream.sync());
        auto const total = static_cast<size_t>(offsets.back()) +
                           static_cast<size_t>(cuda::std::popcount(pass_bits.back()));
        for (size_t first = 0U; first < words;) {
            auto const base = static_cast<size_t>(offsets[first]);
            // Whole words only: a word holds at most 32 results.
            auto const last = static_cast<size_t>(
                std::upper_bound(
                    offsets.begin() + static_cast<std::ptrdiff_t>(first) + 1,
                    offsets.end(),
                    static_cast<uint32_t>(base + chunk_records - 32U)
                ) -
                offsets.begin()
            );
            auto const end = last == words ? total : static_cast<size_t>(offsets[last]);
            if (end != base) chunks.push_back({first, last, base, end});
            first = last;
        }
    }
    if (chunks.empty()) {
        CUDDL_CUDA_TRY(stream.sync());
        return Ok();
    }

    auto staging = CUDDL_CUDA_TRY(
        cuda::make_device_buffer<packed_pairwise_counts>(
            stream, stream.device(), threshold ? chunk_records : 0U, cuda::no_init
        )
    );
    std::array<pinned_counts, 2> buffers{
        pinned_counts(stream, memory, chunk_records, cuda::no_init),
        pinned_counts(stream, memory, chunk_records, cuda::no_init),
    };
    std::array<cuda::event, 2> ready{cuda::event(stream.device()), cuda::event(stream.device())};
    auto const enqueue = [&](size_t k) -> Result<void> {
        auto const& chunk = chunks[k];
        auto const records = chunk.end_record - chunk.first_record;
        packed_pairwise_counts const* source = tile.results + chunk.first_record;
        if (threshold) {
            auto const kernel = tile.upper_triangle
                                    ? detail::gather_passing_kernel<true, packed_pairwise_counts>
                                    : detail::gather_passing_kernel<false, packed_pairwise_counts>;
            kernel<<<
                detail::warp_grid_blocks(chunk.last - chunk.first),
                detail::block_size,
                0,
                stream.get()>>>(
                tile.results,
                nullptr,
                tile.pass_bits,
                device_offsets.data(),
                tile.first_query_id,
                tile.reference_count,
                chunk.first,
                chunk.last,
                static_cast<uint32_t>(chunk.first_record),
                staging.data(),
                nullptr
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
            source = staging.data();
        }
        CUDDL_CUDA_TRY(
            cuda::copy_bytes(
                stream,
                cuda::std::span{source, records},
                cuda::std::span{buffers[k % 2U].data(), records}
            )
        );
        ready[k % 2U].record(stream);
        return Ok();
    };

    CUDDL_TRY(enqueue(0U));
    for (size_t k = 0; k < chunks.size(); ++k) {
        // Buffer (k + 1) % 2 last held chunk k - 1, which the host has finished visiting.
        if (k + 1U < chunks.size()) CUDDL_TRY(enqueue(k + 1U));
        CUDDL_CUDA_TRY(ready[k % 2U].sync());
        auto const* packed = buffers[k % 2U].data();
        auto const& chunk = chunks[k];
        size_t next = 0U;
        auto const emit = [&](uint32_t query_id, uint32_t reference_id) {
            f(batch_search_result{
                .query_id = query_id,
                .reference_id = reference_id,
                .counts = packed[next++].unpack(tile.bucket_count),
            });
        };
        if (!threshold) {
            for (auto query = static_cast<uint32_t>(chunk.first); query < chunk.last; ++query) {
                auto const query_id = tile.first_query_id + query;
                for (auto reference = tile.upper_triangle ? query_id + 1U : 0U;
                     reference < tile.reference_count;
                     ++reference) {
                    emit(query_id, reference);
                }
            }
            continue;
        }
        auto const words_per_query = tile.words_per_query();
        for (auto w = chunk.first; w < chunk.last; ++w) {
            auto const query_id = tile.first_query_id + static_cast<uint32_t>(w / words_per_query);
            auto const base = static_cast<uint32_t>(w % words_per_query) * 32U;
            for (auto bits = pass_bits[w]; bits != 0U; bits &= bits - 1U) {
                emit(query_id, base + static_cast<uint32_t>(cuda::std::countr_zero(bits)));
            }
        }
    }
    return Ok();
}

/// @brief Caller-owned storage requirements for one bounded query tile.
struct batch_search_requirements {
    uint32_t maximum_pair_count{};
    size_t counter_bytes{};
    size_t candidate_bytes{};
    size_t temporary_bytes{};
    size_t workspace_bytes{};
    size_t result_bytes{};
    size_t match_count_bytes{};

    friend bool operator==(batch_search_requirements const&, batch_search_requirements const&) =
        default;
};

/// @brief Matching-bucket threshold, independent of whether an index accelerates the search.
struct search_options {
    uint32_t minimum_matches = 5;
};

namespace detail {

struct sparse_index_key {
    uint16_t mask;
    template <typename Row>
    __host__ __device__ uint16_t operator()(Row row) const noexcept {
        auto const score = reference_score(row);
        return score == 0U || mask == 0xffffU ? score : static_cast<uint16_t>((score & mask) + 1U);
    }
};
struct sparse_reference_id {
    uint32_t reference_count;
    __host__ __device__ uint32_t operator()(uint32_t i) const noexcept {
        return i % reference_count;
    }
};
struct sparse_segment_offset {
    uint32_t reference_count;
    __host__ __device__ int64_t operator()(uint32_t bucket) const noexcept {
        return static_cast<int64_t>(bucket) * reference_count;
    }
};

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_score_compatibility(
    score_compatibility const& compatibility
) {
    if (compatibility.kmer_length != K) {
        return Err(Error::invalid_argument("k-mer length does not match the database type"));
    }
    if (compatibility.bucket_count != BucketCount) {
        return Err(Error::invalid_argument("bucket count does not match the database type"));
    }
    if (compatibility.score_encoder_identity == 0U ||
        compatibility.exponent_bits != Layout::exponent_bits ||
        compatibility.mantissa_bits != Layout::mantissa_bits) {
        return Err(Error::invalid_argument("score encoding does not match the register layout"));
    }
    if (compatibility.hash_identity == 0U) {
        return Err(Error::invalid_argument("hash identity must be specified"));
    }
    if (compatibility.canonicalisation_policy == 0U) {
        return Err(Error::invalid_argument("canonicalisation policy must be specified"));
    }
    return Ok();
}

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_non_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount, Layout>(compatibility);
        !validation) {
        return validation;
    }
    if (compatibility.indexed_bucket_count != BucketCount) {
        return Err(Error::invalid_argument("non-indexed builds require every bucket"));
    }
    if (compatibility.key_mask != 0xffffU && compatibility.key_mask != 0x7fffU) {
        return Err(Error::invalid_argument("non-indexed builds require a 16-bit or 15-bit key"));
    }
    return Ok();
}

template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
[[nodiscard]] inline Result<void> validate_indexed_score_compatibility(
    score_compatibility const& compatibility
) {
    if (auto const validation = validate_score_compatibility<K, BucketCount, Layout>(compatibility);
        !validation) {
        return validation;
    }
    auto const full_bucket_count = static_cast<uint32_t>(BucketCount);
    if (compatibility.indexed_bucket_count != full_bucket_count &&
        compatibility.indexed_bucket_count != full_bucket_count / 2U) {
        return Err(Error::invalid_argument("indexed builds require a full or half bucket count"));
    }
    if (compatibility.key_mask != 0xffffU && compatibility.key_mask != 0x7fffU) {
        return Err(Error::invalid_argument("indexed builds require a 16-bit or 15-bit key mask"));
    }
    return Ok();
}

[[nodiscard]] inline constexpr uint64_t indexed_cell_count(
    score_compatibility const& compatibility
) noexcept {
    return static_cast<uint64_t>(compatibility.indexed_bucket_count) *
           (static_cast<uint64_t>(compatibility.key_mask) + 1U);
}
[[nodiscard]] inline constexpr uint64_t
indexed_posting_count(uint32_t reference_count, score_compatibility const& compatibility) noexcept {
    return static_cast<uint64_t>(reference_count) * compatibility.indexed_bucket_count;
}

[[nodiscard]] inline uintptr_t align_up(uintptr_t address, size_t alignment) noexcept {
    return (address + alignment - 1U) & ~(static_cast<uintptr_t>(alignment) - 1U);
}

/// @brief Index pair fraction (reference_index::pair_fraction) above which batch search compares
/// bit-planes for every pair instead of counting through the index.
// ponytail: one crossover measured on an RTX 5070 Ti with 2048-bucket RefSeq/synthetic data;
// make it per-device if other GPUs disagree.
constexpr double index_pair_fraction_limit = 0.03;

/// @brief Minimum query rows per batch tile.
constexpr uint32_t batch_query_tile_count = 128U;

[[nodiscard]] constexpr uint32_t batch_query_tile_size(
    uint32_t reference_count,
    uint32_t query_count,
    uint32_t tile_limit = batch_query_tile_count
) noexcept {
    if (reference_count == 0U || query_count == 0U) {
        return 0U;
    }
    auto const count_capacity = std::numeric_limits<uint32_t>::max() / reference_count;
    return std::min(query_count, std::min(tile_limit, std::max(1U, count_capacity)));
}

/// @brief Query rows one all-to-all tile compares: about 2^24 pairs, and at least
/// @ref batch_query_tile_count. Enough query groups to fill the GPU for each launch, while result
/// storage holds two tiles so one is searched as the other is consumed.
[[nodiscard]] constexpr uint32_t all_to_all_tile_queries(uint32_t reference_count) noexcept {
    constexpr uint32_t pair_budget = 1U << 24U;
    auto const limit = reference_count == 0U
                           ? batch_query_tile_count
                           : std::max(batch_query_tile_count, pair_budget / reference_count);
    return batch_query_tile_size(reference_count, reference_count, limit);
}

/// @brief Query rows one external batch tile may hold on @p device.
///
/// A tile's results, match counts and counters cost about 20 bytes per pair, and a quarter of
/// the device's memory goes to them, so a 16 GB device holds about 200 million pairs per tile.
/// Larger tiles give the counting kernels more queries to overlap. The limit depends only on the
/// device, so storage sized from the requirements always matches the search.
[[nodiscard]] inline uint32_t
external_batch_query_limit(uint32_t reference_count, cuda::device_ref device) {
    constexpr size_t bytes_per_pair = sizeof(packed_pairwise_counts) + 3U * sizeof(uint32_t);
    auto const budget = device.attribute(cuda::device_attributes::total_global_memory) / 4U;
    auto const pairs = budget / bytes_per_pair;
    auto const queries = reference_count == 0U ? pairs : pairs / reference_count;
    return static_cast<uint32_t>(
        std::clamp<size_t>(queries, batch_query_tile_count, std::numeric_limits<uint32_t>::max())
    );
}

}  // namespace detail

namespace detail {

/**
 * @brief Non-owning, trivially copyable view of one selected reference-row backing.
 *
 * Inputs, database rows, workspace, and results must remain valid until the supplied stream
 * completes.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class reference_database_view {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using score_type = uint16_t;
    using layout_type = Layout;
    using result_type = reference_search_result;
    using batch_result_type = packed_pairwise_counts;

    __host__ __device__ constexpr reference_database_view(
        device_span<uint32_t const> planes,
        reference_database_metadata metadata,
        device_span<uint32_t const> index_offsets = {},
        device_span<uint32_t const> index_postings = {},
        bool indexed = false,
        device_span<uint16_t const> index_keys = {},
        device_span<uint32_t const> key_directory = {},
        double index_pair_fraction = 0.0
    ) noexcept
        : planes_(planes),
          metadata_(metadata),
          index_offsets_(index_offsets),
          index_postings_(index_postings),
          index_keys_(index_keys),
          key_directory_(key_directory),
          index_pair_fraction_(index_pair_fraction),
          indexed_(indexed) {}

    /// @brief Reference bit-plane rows (detail::score_plane_index layout), in reference-ID order.
    [[nodiscard]] __host__ __device__ constexpr device_span<uint32_t const>
    planes() const noexcept {
        return planes_;
    }

    /// @brief Compatibility metadata and reference count.
    [[nodiscard]] constexpr reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    /// @brief Reference count.
    [[nodiscard]] constexpr uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    /// @brief True when an index backs this view.
    [[nodiscard]] constexpr bool has_index() const noexcept {
        return indexed_;
    }

    /// @brief Bytes one compact row store needs for @p reference_count references.
    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return static_cast<size_t>(reference_count) * BucketCount * sizeof(score_type);
    }

    /// @brief Bytes this view's own row store needs.
    [[nodiscard]] constexpr size_t persistent_row_bytes() const noexcept {
        return planes_.size_bytes();
    }

    /// @brief Bytes this view's index needs for offsets, postings, and keys.
    [[nodiscard]] constexpr size_t persistent_index_bytes() const noexcept {
        return index_offsets_.size_bytes() + index_postings_.size_bytes() +
               index_keys_.size_bytes();
    }

    /// @brief Caller-owned bytes for one exhaustive query: the query's bit-planes.
    [[nodiscard]] static constexpr size_t single_query_workspace_bytes(
        uint32_t reference_count
    ) noexcept {
        return reference_count == 0U ? 0U : query_plane_bytes(1U) + plane_alignment - 1U;
    }

    /// @brief Caller-owned bytes for one exhaustive query on this view.
    [[nodiscard]] constexpr size_t single_query_workspace_bytes() const noexcept {
        return single_query_workspace_bytes(metadata_.reference_count);
    }

    [[nodiscard]] static constexpr uint32_t single_query_result_count(
        uint32_t reference_count
    ) noexcept {
        return reference_count;
    }

    /// @brief Results one exhaustive query writes: one per reference.
    [[nodiscard]] constexpr uint32_t single_query_result_count() const noexcept {
        return metadata_.reference_count;
    }

    /// @brief Caller-owned bytes required by one positive-threshold indexed query.
    [[nodiscard]] Result<size_t> indexed_single_query_workspace_bytes(
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_index_storage());
        if (metadata_.reference_count == 0U) {
            return size_t{0};
        }

        size_t selection_bytes = 0;
        auto const ids = cuda::make_counting_iterator(uint32_t{0});
        auto const selection = cuda_try(
            cub::DeviceSelect::If(
                nullptr,
                selection_bytes,
                ids,
                static_cast<uint32_t*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(metadata_.reference_count),
                detail::minimum_match_predicate{nullptr, 1U},
                stream.get()
            )
        );
        if (!selection) {
            return Err(selection.error());
        }
        // The counting pass stages two words per indexed bucket in the selection storage.
        selection_bytes = std::max(
            selection_bytes,
            static_cast<size_t>(metadata_.compatibility.indexed_bucket_count) * 2U *
                sizeof(uint32_t)
        );

        constexpr size_t alignment_slack = plane_alignment - 1U + 3U + 255U;
        auto const arrays_bytes =
            query_plane_bytes(1U) +
            static_cast<size_t>(metadata_.reference_count) * 2U * sizeof(uint32_t);
        if (selection_bytes > std::numeric_limits<size_t>::max() - arrays_bytes - alignment_slack) {
            return Err(Error::resource("indexed single-query workspace size overflows"));
        }
        return arrays_bytes + alignment_slack + selection_bytes;
    }

    /// @brief Storage reused while exhaustively searching @p query_count compact rows.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        auto const pair_count =
            CUDDL_TRY(dense_batch_pair_count(external_batch_tile(query_count, stream)));
        return make_batch_requirements(
            pair_count, pair_count, false, pair_count / reference_divisor()
        );
    }

    /// @brief Storage reused while searching @p query_count compact rows through the index.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_batch_search_requirements(uint32_t query_count, cuda::stream_ref stream) const {
        CUDDL_TRY(validate_index_storage());
        auto const pair_count =
            CUDDL_TRY(dense_batch_pair_count(external_batch_tile(query_count, stream)));
        return make_batch_requirements(
            pair_count, pair_count, true, pair_count / reference_divisor()
        );
    }

    /// @brief Storage reused while exhaustively searching every unique database-row pair.
    [[nodiscard]] Result<cuddl::batch_search_requirements> all_to_all_search_requirements() const {
        auto const slot = CUDDL_TRY(all_to_all_slot_pairs());
        if (slot > std::numeric_limits<uint32_t>::max() / 2U) {
            return Err(Error::resource("all-to-all tile exceeds 32-bit result counts"));
        }
        return make_batch_requirements(
            0U, pipeline_slots(all_to_all_tile_count()) * slot, false, 0U
        );
    }

    [[nodiscard]] static constexpr uint32_t all_to_all_result_capacity(
        uint32_t reference_count
    ) noexcept {
        if (reference_count < 2U) {
            return 0U;
        }
        auto const query_count = detail::all_to_all_tile_queries(reference_count);
        auto const pair_count = static_cast<uint64_t>(query_count) *
                                (2ULL * reference_count - query_count - 1ULL) / 2ULL;
        auto const tiles = (reference_count + query_count - 1U) / query_count;
        return static_cast<uint32_t>(pair_count * pipeline_slots(tiles));
    }

    /// @brief Storage for every unique database-row pair through the index, or with a threshold
    /// but no index; it also covers the exhaustive all-to-all search.
    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_all_to_all_search_requirements() const {
        auto requirements = CUDDL_TRY(indexed_all_to_all_tile_search_requirements(
            0U, detail::all_to_all_tile_queries(metadata_.reference_count)
        ));
        auto const exhaustive = CUDDL_TRY(all_to_all_search_requirements());
        requirements.maximum_pair_count =
            std::max(requirements.maximum_pair_count, exhaustive.maximum_pair_count);
        requirements.result_bytes = std::max(requirements.result_bytes, exhaustive.result_bytes);
        requirements.match_count_bytes =
            std::max(requirements.match_count_bytes, exhaustive.match_count_bytes);
        return requirements;
    }

    /// @brief Compares one compact query row with every reference row on @p stream.
    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream
    ) const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        if (query.size() != BucketCount || query.data() == nullptr) {
            return Err(Error::invalid_argument("query must contain one complete score row"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
            !validation) {
            return validation;
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        if (workspace.size_bytes() < single_query_workspace_bytes()) {
            return Err(Error::resource("single-query workspace is too small"));
        }
        if (results.size() < metadata_.reference_count) {
            return Err(Error::resource("single-query result capacity is too small"));
        }
        if (metadata_.reference_count == 0U) {
            return Ok();
        }
        if (results.data() == nullptr) {
            return Err(Error::invalid_argument("result storage must be device accessible"));
        }

        if (workspace.data() == nullptr) {
            return Err(Error::invalid_argument("single-query workspace must be device accessible"));
        }
        auto* query_planes = CUDDL_TRY(stage_query_planes(query, 0U, 1U, workspace, stream));
        detail::single_query_planes_kernel<BucketCount>
            <<<detail::warp_grid_blocks(metadata_.reference_count),
               detail::block_size,
               0,
               stream.get()>>>(
                query_planes,
                planes_.data(),
                metadata_.reference_count,
                nullptr,
                nullptr,
                results.data()
            );
        return cuda_try(cudaGetLastError());
    }

    /// @brief Finds and exactly refines references meeting @p options on @p stream.
    [[nodiscard]] Result<void> search_indexed_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        device_span<uint32_t> result_count,
        search_options options,
        cuda::stream_ref stream
    ) const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        CUDDL_TRY(validate_index_storage());
        if (query.size() != BucketCount || query.data() == nullptr) {
            return Err(Error::invalid_argument("query must contain one complete score row"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
            !validation) {
            return validation;
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        if (options.minimum_matches > metadata_.compatibility.indexed_bucket_count) {
            return Err(Error::invalid_argument("minimum matches exceeds indexed bucket count"));
        }
        if (results.size() < metadata_.reference_count) {
            return Err(Error::resource("indexed result capacity is too small"));
        }
        if (result_count.empty()) {
            return Err(Error::resource("indexed result count capacity is too small"));
        }
        if (result_count.data() == nullptr) {
            return Err(Error::invalid_argument("result count storage must be device accessible"));
        }
        if (metadata_.reference_count != 0U && results.data() == nullptr) {
            return Err(Error::invalid_argument("result storage must be device accessible"));
        }

        if (options.minimum_matches == 0U) {
            CUDDL_TRY(search_async(query, query_compatibility, workspace, results, stream));
            return write_batch_result_count(metadata_.reference_count, result_count, stream);
        }
        if (metadata_.reference_count == 0U) {
            return write_batch_result_count(0U, result_count, stream);
        }

        auto const required_workspace = CUDDL_TRY(indexed_single_query_workspace_bytes(stream));
        if (workspace.size_bytes() < required_workspace) {
            return Err(Error::resource("indexed single-query workspace is too small"));
        }
        if (workspace.data() == nullptr) {
            return Err(Error::invalid_argument("indexed workspace must be device accessible"));
        }

        auto const workspace_begin = reinterpret_cast<uintptr_t>(workspace.data());
        auto const workspace_end = workspace_begin + workspace.size_bytes();
        if (workspace_end < workspace_begin) {
            return Err(Error::resource("indexed workspace address range overflows"));
        }
        auto* query_planes = CUDDL_TRY(stage_query_planes(query, 0U, 1U, workspace, stream));
        auto address = detail::align_up(
            reinterpret_cast<uintptr_t>(query_planes) + query_plane_bytes(1U), alignof(uint32_t)
        );
        auto* match_counts = reinterpret_cast<uint32_t*>(address);
        address += static_cast<size_t>(metadata_.reference_count) * sizeof(uint32_t);
        address = detail::align_up(address, alignof(uint32_t));
        auto* candidate_ids = reinterpret_cast<uint32_t*>(address);
        address += static_cast<size_t>(metadata_.reference_count) * sizeof(uint32_t);
        address = detail::align_up(address, 256U);
        if (address > workspace_end) {
            return Err(Error::resource("indexed workspace layout exceeds its capacity"));
        }
        auto* selection_workspace = reinterpret_cast<void*>(address);
        auto selection_bytes = static_cast<size_t>(workspace_end - address);

        if (!indexed_) {
            detail::count_plane_matches_kernel<BucketCount>
                <<<detail::warp_grid_blocks(metadata_.reference_count),
                   detail::block_size,
                   0,
                   stream.get()>>>(
                    query_planes,
                    1U,
                    planes_.data(),
                    metadata_.reference_count,
                    metadata_.compatibility.indexed_bucket_count,
                    metadata_.compatibility.key_mask,
                    match_counts
                );
            CUDDL_CUDA_TRY(cudaGetLastError());
        } else {
            CUDDL_CUDA_TRY(
                cuda::fill_bytes(
                    stream,
                    cuda::std::span{match_counts, static_cast<size_t>(metadata_.reference_count)},
                    0
                )
            );
            // The posting ranges live in the selection workspace, which is free until the
            // selection below runs on the same stream.
            auto const indexed_bucket_count = metadata_.compatibility.indexed_bucket_count;
            auto* range_begin = static_cast<uint32_t*>(selection_workspace);
            auto* range_end_scan = range_begin + indexed_bucket_count;
            detail::index_posting_ranges_kernel<<<
                1,
                detail::posting_range_block_size,
                0,
                stream.get()>>>(
                query.data(),
                index_offsets_.data(),
                index_keys_.empty() ? nullptr : index_keys_.data(),
                metadata_.reference_count,
                indexed_bucket_count,
                metadata_.compatibility.key_mask,
                range_begin,
                range_end_scan
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
            auto const counting_blocks =
                static_cast<uint32_t>(
                    stream.device().attribute(cuda::device_attributes::multiprocessor_count)
                ) *
                static_cast<uint32_t>(stream.device().attribute(
                    cuda::device_attributes::max_threads_per_multiprocessor
                )) /
                detail::block_size;
            detail::count_balanced_index_matches_kernel<<<
                counting_blocks,
                detail::block_size,
                0,
                stream.get()>>>(
                range_begin,
                range_end_scan,
                indexed_bucket_count,
                index_postings_.data(),
                match_counts
            );
            CUDDL_CUDA_TRY(cudaGetLastError());
        }

        auto const ids = cuda::make_counting_iterator(uint32_t{0});
        CUDDL_CUDA_TRY(
            cub::DeviceSelect::If(
                selection_workspace,
                selection_bytes,
                ids,
                candidate_ids,
                result_count.data(),
                static_cast<int64_t>(metadata_.reference_count),
                detail::minimum_match_predicate{match_counts, options.minimum_matches},
                stream.get()
            )
        );

        detail::single_query_planes_kernel<BucketCount>
            <<<detail::warp_grid_blocks(metadata_.reference_count),
               detail::block_size,
               0,
               stream.get()>>>(
                query_planes,
                planes_.data(),
                metadata_.reference_count,
                candidate_ids,
                result_count.data(),
                results.data()
            );
        return cuda_try(cudaGetLastError());
    }

    /**
     * @brief Exhaustively searches compact queries using bounded reusable storage.
     *
     * @p on_tile receives each tile's @ref batch_result_tile. Before returning, it must consume
     * the tile synchronously or enqueue dependent work on @p stream because the supplied storage
     * is reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(batch_search_requirements(query_count, stream));
        CUDDL_TRY(validate_batch_outputs(requirements, results, result_match_counts));
        CUDDL_TRY(validate_workspace(requirements, workspace));
        // Result storage holds one full-size tile; two half-size tiles alternate in it.
        auto const storage_queries = external_batch_tile(query_count, stream);
        auto const tile_size = storage_queries < 2U ? storage_queries : storage_queries / 2U;
        auto const tiles = tile_size == 0U ? 0U : (query_count + tile_size - 1U) / tile_size;
        auto const slot = static_cast<size_t>(tile_size) * metadata_.reference_count;
        return pipeline_tiles(
            tiles,
            slot,
            results,
            result_match_counts,
            [&](uint32_t tile,
                device_span<batch_result_type> slot_results,
                device_span<uint32_t> slot_matches,
                cuda::stream_ref work) {
                auto const first_query_id = tile * tile_size;
                return launch_batch_exhaustive<false>(
                    queries,
                    first_query_id,
                    std::min(tile_size, query_count - first_query_id),
                    query_id_offset + first_query_id,
                    workspace,
                    slot_results,
                    slot_matches,
                    work
                );
            },
            on_tile,
            stream
        );
    }

    /**
     * @brief Indexed search for compact queries using bounded reusable storage.
     *
     * @p on_tile receives each tile's @ref batch_result_tile, holding exactly the pairs with at
     * least `options.minimum_matches` index matches. Before returning, it must consume the tile
     * synchronously or enqueue dependent work on @p stream because the supplied storage is
     * reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_indexed_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        search_options options,
        cuda::stream_ref stream
    ) const {
        auto const query_count =
            CUDDL_TRY(validate_batch_queries(queries, query_compatibility, query_id_offset));
        auto const requirements = CUDDL_TRY(indexed_batch_search_requirements(query_count, stream));
        CUDDL_TRY(validate_indexed_batch_inputs(
            requirements, workspace, results, result_match_counts, options
        ));
        auto const tile_size = external_batch_tile(query_count, stream);
        for (uint32_t first_query_id = 0U; tile_size != 0U && first_query_id < query_count;
             first_query_id += tile_size) {
            auto const tile_query_count = std::min(tile_size, query_count - first_query_id);
            on_tile(CUDDL_TRY(
                launch_batch_indexed<false>(
                    queries.data() + static_cast<size_t>(first_query_id) * BucketCount,
                    first_query_id,
                    tile_query_count,
                    query_id_offset + first_query_id,
                    workspace,
                    results,
                    result_match_counts,
                    options,
                    stream
                )
            ));
        }
        return Ok();
    }

    /**
     * @brief Exhaustively searches every unique database-row pair using bounded storage.
     *
     * @p on_tile receives each tile's @ref batch_result_tile. Before returning, it must consume
     * the tile synchronously or enqueue dependent work on @p stream because the supplied storage
     * is reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_stored_rows());
        auto const requirements = CUDDL_TRY(all_to_all_search_requirements());
        CUDDL_TRY(validate_batch_outputs(requirements, results, result_match_counts));
        auto const tile_size = detail::all_to_all_tile_queries(metadata_.reference_count);
        return pipeline_tiles(
            all_to_all_tile_count(),
            CUDDL_TRY(all_to_all_slot_pairs()),
            results,
            result_match_counts,
            [&](uint32_t tile,
                device_span<batch_result_type> slot_results,
                device_span<uint32_t> slot_matches,
                cuda::stream_ref work) {
                auto const first_query_id = tile * tile_size;
                return launch_batch_exhaustive<true>(
                    {},
                    first_query_id,
                    std::min(tile_size, metadata_.reference_count - first_query_id),
                    first_query_id,
                    workspace,
                    slot_results,
                    slot_matches,
                    work
                );
            },
            on_tile,
            stream
        );
    }

    /**
     * @brief Indexed search of every unique database-row pair using bounded storage.
     *
     * @p on_tile receives each tile's @ref batch_result_tile, holding exactly the pairs with at
     * least `options.minimum_matches` index matches. Before returning, it must consume the tile
     * synchronously or enqueue dependent work on @p stream because the supplied storage is
     * reused by the next tile.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_indexed_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        search_options options,
        cuda::stream_ref stream
    ) const {
        CUDDL_TRY(validate_stored_rows());
        auto const requirements = CUDDL_TRY(indexed_all_to_all_search_requirements());
        CUDDL_TRY(validate_indexed_batch_inputs(
            requirements, workspace, results, result_match_counts, options
        ));
        auto const tile_size = detail::all_to_all_tile_queries(metadata_.reference_count);
        for (uint32_t first_query_id = 0U;
             tile_size != 0U && first_query_id < metadata_.reference_count;
             first_query_id += tile_size) {
            auto const query_count =
                std::min(tile_size, metadata_.reference_count - first_query_id);
            on_tile(CUDDL_TRY(
                launch_batch_indexed<true>(
                    nullptr,
                    first_query_id,
                    query_count,
                    first_query_id,
                    workspace,
                    results,
                    result_match_counts,
                    options,
                    stream
                )
            ));
        }
        return Ok();
    }

   private:
    /// @brief Result-storage slots a search of @p tiles tiles alternates between.
    [[nodiscard]] static constexpr uint32_t pipeline_slots(uint32_t tiles) noexcept {
        return tiles < 2U ? 1U : 2U;
    }

    [[nodiscard]] constexpr uint32_t all_to_all_tile_count() const noexcept {
        auto const tile_size = detail::all_to_all_tile_queries(metadata_.reference_count);
        return tile_size == 0U ? 0U : (metadata_.reference_count + tile_size - 1U) / tile_size;
    }

    /// @brief Results in the largest (first) all-to-all tile.
    [[nodiscard]] Result<uint32_t> all_to_all_slot_pairs() const {
        return all_to_all_pair_count(
            0U, detail::all_to_all_tile_queries(metadata_.reference_count)
        );
    }

    /**
     * @brief Searches @p tiles tiles on an internal stream, one tile ahead of @p on_tile.
     *
     * Tile n uses result slot `n % 2` of @p slot results each. Tile n + 1 is enqueued, after
     * everything the caller has queued on @p stream so far (tile n - 1's consumers, which share
     * its slot), before @p on_tile receives tile n, so the search of one tile overlaps the
     * consumption of the previous one. @p stream waits for each tile's search before its
     * callback and for the last search before returning.
     */
    template <typename Launch, typename OnTile>
    [[nodiscard]] Result<void> pipeline_tiles(
        uint32_t tiles,
        size_t slot,
        device_span<batch_result_type> results,
        device_span<uint32_t> match_counts,
        Launch&& launch,
        OnTile&& on_tile,
        cuda::stream_ref stream
    ) const {
        auto const slot_of = [&](uint32_t tile) {
            auto const offset = static_cast<size_t>(tile % 2U) * slot;
            return std::pair{
                device_span<batch_result_type>{results.data() + offset, slot},
                match_counts.empty() ? match_counts
                                     : device_span<uint32_t>{match_counts.data() + offset, slot}
            };
        };
        if (tiles == 0U) return Ok();
        if (tiles == 1U) {
            auto const [slot_results, slot_matches] = slot_of(0U);
            on_tile(CUDDL_TRY(launch(0U, slot_results, slot_matches, stream)));
            return Ok();
        }
        auto work = CUDDL_CUDA_TRY(cuda::stream(stream.device()));
        std::array<cuda::event, 2> searched{
            CUDDL_CUDA_TRY(cuda::event(stream.device())),
            CUDDL_CUDA_TRY(cuda::event(stream.device())),
        };
        std::array<batch_result_tile, 2> pending{};
        auto const enqueue = [&](uint32_t tile) -> Result<void> {
            // The slot's previous tile is consumed once the caller's stream reaches this point.
            CUDDL_CUDA_TRY(work.wait(stream));
            auto const [slot_results, slot_matches] = slot_of(tile);
            pending[tile % 2U] = CUDDL_TRY(launch(tile, slot_results, slot_matches, work));
            CUDDL_CUDA_TRY(searched[tile % 2U].record(work));
            return Ok();
        };
        CUDDL_TRY(enqueue(0U));
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            if (tile + 1U < tiles) CUDDL_TRY(enqueue(tile + 1U));
            CUDDL_CUDA_TRY(stream.wait(searched[tile % 2U]));
            on_tile(pending[tile % 2U]);
        }
        return Ok();
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements>
    indexed_all_to_all_tile_search_requirements(
        uint32_t first_query_id,
        uint32_t query_count
    ) const {
        CUDDL_TRY(validate_index_storage());
        auto const dense_pair_count = CUDDL_TRY(dense_batch_pair_count(query_count));
        auto const result_pair_count =
            CUDDL_TRY(all_to_all_pair_count(first_query_id, query_count));
        return make_batch_requirements(dense_pair_count, result_pair_count, true, query_count);
    }

    [[nodiscard]] Result<void> validate_stored_rows() const {
        auto const expected_scores = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (!rows_match_metadata(expected_scores)) {
            return Err(Error::invalid_argument("database extent does not match its metadata"));
        }
        return Ok();
    }

    [[nodiscard]] Result<void> validate_index_storage() const {
        CUDDL_TRY((detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(
            metadata_.compatibility
        )));
        if (!indexed_) return Ok();
        auto const cell_count = detail::indexed_cell_count(metadata_.compatibility);
        auto const expected_postings = static_cast<size_t>(
            detail::indexed_posting_count(metadata_.reference_count, metadata_.compatibility)
        );
        auto const sparse = index_offsets_.empty();
        auto const valid_keys =
            sparse ? index_keys_.size() == expected_postings &&
                         (expected_postings == 0U || index_keys_.data() != nullptr)
                   : index_keys_.empty() &&
                         index_offsets_.size() == static_cast<size_t>(cell_count + 1U) &&
                         index_offsets_.data() != nullptr;
        if (!indexed_ || !valid_keys || index_postings_.size() != expected_postings ||
            (expected_postings != 0U && index_postings_.data() == nullptr)) {
            return Err(Error::invalid_argument("database has no valid retrieval index"));
        }
        return Ok();
    }

    [[nodiscard]] uint32_t
    external_batch_tile(uint32_t query_count, cuda::stream_ref stream) const {
        return detail::batch_query_tile_size(
            metadata_.reference_count,
            query_count,
            detail::external_batch_query_limit(metadata_.reference_count, stream.device())
        );
    }

    [[nodiscard]] Result<uint32_t> dense_batch_pair_count(uint32_t query_count) const {
        auto const pair_count = static_cast<uint64_t>(query_count) * metadata_.reference_count;
        if (pair_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("batch pair count exceeds 32-bit workspace IDs"));
        }
        return static_cast<uint32_t>(pair_count);
    }

    [[nodiscard]] Result<uint32_t>
    all_to_all_pair_count(uint32_t first_query_id, uint32_t query_count) const {
        if (first_query_id > metadata_.reference_count ||
            query_count > metadata_.reference_count - first_query_id) {
            return Err(Error::invalid_argument("all-to-all query tile is outside the database"));
        }
        if (query_count == 0U) {
            return 0U;
        }

        uint64_t left = query_count;
        uint64_t right = 2U * static_cast<uint64_t>(metadata_.reference_count) -
                         2U * first_query_id - query_count - 1U;
        if ((left & 1U) == 0U) {
            left /= 2U;
        } else {
            right /= 2U;
        }
        auto const pair_count = left * right;
        if (pair_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("all-to-all tile exceeds 32-bit result counts"));
        }
        return static_cast<uint32_t>(pair_count);
    }

    /// @brief Byte alignment of staged query bit-planes, which kernels read as uint4.
    static constexpr size_t plane_alignment = 16U;

    /// @brief Bytes of @p query_count rows staged as bit-planes or decoded scores.
    [[nodiscard]] static constexpr size_t query_plane_bytes(uint32_t query_count) noexcept {
        return static_cast<size_t>(query_count) * BucketCount * sizeof(uint16_t);
    }

    /// @brief Reference count, or one for an empty database, to recover tile query counts.
    [[nodiscard]] constexpr uint32_t reference_divisor() const noexcept {
        return metadata_.reference_count == 0U ? 1U : metadata_.reference_count;
    }

    /// @brief Whether threshold search compares bit-planes for every pair: always without an
    /// index, and when the index's lookups would visit many cells, counting slower than that.
    [[nodiscard]] constexpr bool compares_planes() const noexcept {
        return !indexed_ || index_pair_fraction_ > detail::index_pair_fraction_limit;
    }

    /// @brief Requirements for one tile.
    ///
    /// @p staged_queries rows get a staging region: external queries are converted to bit-planes
    /// there, and all-to-all index-counted tiles decode their database queries to scores there.
    /// Threshold tiles also get the pass bitmap, and index-counted tiles per-pair counters and
    /// sparse posting ranges.
    [[nodiscard]] Result<cuddl::batch_search_requirements> make_batch_requirements(
        uint32_t dense_pair_count,
        uint32_t maximum_pair_count,
        bool indexed,
        uint32_t staged_queries
    ) const {
        cuddl::batch_search_requirements requirements{
            .maximum_pair_count = maximum_pair_count,
            .result_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(batch_result_type),
            .match_count_bytes = static_cast<size_t>(maximum_pair_count) * sizeof(uint32_t),
        };
        auto const staging_bytes = query_plane_bytes(staged_queries);
        if (!indexed || dense_pair_count == 0U) {
            if (staging_bytes != 0U) {
                requirements.workspace_bytes = staging_bytes + plane_alignment - 1U;
            }
            return requirements;
        }

        auto const tile_query_count = dense_pair_count / reference_divisor();
        auto const word_count = static_cast<size_t>(tile_query_count) *
                                detail::candidate_bit_words(metadata_.reference_count);
        requirements.candidate_bytes = word_count * sizeof(uint32_t);
        if (!compares_planes()) {
            requirements.counter_bytes = static_cast<size_t>(dense_pair_count) * sizeof(uint32_t);
            if (!index_keys_.empty()) {
                // Sparse counting first stages the tile's posting ranges.
                requirements.temporary_bytes = static_cast<size_t>(tile_query_count) *
                                               metadata_.compatibility.indexed_bucket_count *
                                               sizeof(uint2);
            }
        }
        constexpr size_t alignment_slack = alignof(uint32_t) - 1U + plane_alignment - 1U + 255U;
        requirements.workspace_bytes = requirements.counter_bytes + requirements.candidate_bytes +
                                       staging_bytes + requirements.temporary_bytes +
                                       alignment_slack;
        return requirements;
    }

    [[nodiscard]] static Result<void> validate_workspace(
        cuddl::batch_search_requirements const& requirements,
        device_span<uint8_t> workspace
    ) {
        if (workspace.size_bytes() < requirements.workspace_bytes) {
            return Err(Error::resource("batch workspace is too small"));
        }
        if (requirements.workspace_bytes != 0U && workspace.data() == nullptr) {
            return Err(Error::invalid_argument("batch workspace must be device accessible"));
        }
        return Ok();
    }

    /// @brief Converts @p count score rows starting at @p first_row into bit-planes at the
    /// start of @p workspace; returns where they begin.
    [[nodiscard]] Result<uint32_t*> stage_query_planes(
        device_span<score_type const> queries,
        size_t first_row,
        uint32_t count,
        device_span<uint8_t> workspace,
        cuda::stream_ref stream
    ) const {
        auto const address =
            detail::align_up(reinterpret_cast<uintptr_t>(workspace.data()), plane_alignment);
        if (address + query_plane_bytes(count) >
            reinterpret_cast<uintptr_t>(workspace.data()) + workspace.size_bytes()) {
            return Err(Error::resource("query staging exceeds the workspace"));
        }
        auto* planes = reinterpret_cast<uint32_t*>(address);
        detail::build_score_planes_kernel<BucketCount>
            <<<detail::warp_grid_blocks(static_cast<size_t>(count) * (BucketCount / 32U)),
               detail::block_size,
               0,
               stream.get()>>>(queries.data() + first_row * BucketCount, count, planes);
        CUDDL_CUDA_TRY(cudaGetLastError());
        return planes;
    }

    [[nodiscard]] Result<uint32_t> validate_batch_queries(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset
    ) const {
        CUDDL_TRY(validate_stored_rows());
        if (queries.size() % BucketCount != 0U) {
            return Err(Error::invalid_argument("query tile must contain complete score rows"));
        }
        if (!queries.empty() && queries.data() == nullptr) {
            return Err(Error::invalid_argument("query tile must be device accessible"));
        }
        auto const query_count = queries.size() / BucketCount;
        if (query_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("query tile exceeds stable 32-bit IDs"));
        }
        auto const count = static_cast<uint32_t>(query_count);
        if (count != 0U && query_id_offset > std::numeric_limits<uint32_t>::max() - (count - 1U)) {
            return Err(Error::resource("query IDs exceed the stable 32-bit range"));
        }
        if (auto const validation =
                detail::validate_score_compatibility<K, BucketCount, Layout>(query_compatibility);
            !validation) {
            return Err(validation.error());
        }
        if (query_compatibility != metadata_.compatibility) {
            return Err(Error::invalid_argument("query construction metadata is incompatible"));
        }
        return count;
    }

    [[nodiscard]] Result<void> validate_batch_outputs(
        cuddl::batch_search_requirements const& requirements,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts
    ) const {
        if (results.size() < requirements.maximum_pair_count) {
            return Err(Error::resource("batch result capacity is too small"));
        }
        if (requirements.maximum_pair_count != 0U && results.data() == nullptr) {
            return Err(Error::invalid_argument("batch results must be device accessible"));
        }
        if (!result_match_counts.empty()) {
            if (result_match_counts.size() < requirements.maximum_pair_count) {
                return Err(Error::resource("batch match-count capacity is too small"));
            }
            if (result_match_counts.data() == nullptr) {
                return Err(Error::invalid_argument("batch match counts must be device accessible"));
            }
        }
        return Ok();
    }

    [[nodiscard]] Result<void> validate_indexed_batch_inputs(
        cuddl::batch_search_requirements const& requirements,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts,
        search_options options
    ) const {
        CUDDL_TRY(validate_index_storage());
        if (options.minimum_matches > metadata_.compatibility.indexed_bucket_count) {
            return Err(Error::invalid_argument("minimum matches exceeds indexed bucket count"));
        }
        if (workspace.size_bytes() < requirements.workspace_bytes) {
            return Err(Error::resource("indexed batch workspace is too small"));
        }
        if (requirements.workspace_bytes != 0U && workspace.data() == nullptr) {
            return Err(
                Error::invalid_argument("indexed batch workspace must be device accessible")
            );
        }
        return validate_batch_outputs(requirements, results, result_match_counts);
    }

    [[nodiscard]] static Result<void> write_batch_result_count(
        uint32_t count,
        device_span<uint32_t> result_count,
        cuda::stream_ref stream
    ) {
        return cuda_try(
            cub::DeviceTransform::Transform(
                cuda::make_constant_iterator(count),
                result_count.data(),
                1,
                cuda::std::identity{},
                stream
            )
        );
    }

    /// @brief Compares a query tile with every reference, filling every slot. External queries
    /// are converted to bit-planes in @p workspace; all-to-all tiles read the database's own
    /// planes.
    template <bool AllToAll>
    [[nodiscard]] Result<batch_result_tile> launch_batch_exhaustive(
        device_span<score_type const> queries,
        uint32_t first_query,
        uint32_t query_count,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        uint32_t const* query_planes = nullptr;
        if constexpr (AllToAll) {
            query_planes = planes_.data() + static_cast<size_t>(first_query) * (BucketCount / 2U);
        } else {
            query_planes =
                CUDDL_TRY(stage_query_planes(queries, first_query, query_count, workspace, stream));
        }
        // Over every bucket with the full key, index match counts are the equal buckets.
        CUDDL_TRY((launch_refine<detail::refine_candidates::all, AllToAll>(
            query_planes,
            query_id_offset,
            query_count,
            static_cast<uint32_t>(BucketCount),
            uint16_t{0xffffU},
            nullptr,
            results,
            result_match_counts,
            stream
        )));
        return make_tile(
            results, result_match_counts, nullptr, query_id_offset, query_count, AllToAll
        );
    }

    [[nodiscard]] batch_result_tile make_tile(
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts,
        uint32_t const* pass_bits,
        uint32_t first_query_id,
        uint32_t query_count,
        bool upper_triangle
    ) const noexcept {
        return {
            .results = results.data(),
            .match_counts = result_match_counts.empty() ? nullptr : result_match_counts.data(),
            .pass_bits = pass_bits,
            .first_query_id = first_query_id,
            .query_count = query_count,
            .reference_count = metadata_.reference_count,
            .bucket_count = static_cast<uint32_t>(BucketCount),
            .upper_triangle = upper_triangle,
        };
    }

    /// @brief Exactly compares query bit-planes with the selected references' planes, a group of
    /// queries per reference row read; see detail::refine_batch_bitmap_kernel.
    template <detail::refine_candidates Candidates, bool UpperTriangle>
    [[nodiscard]] Result<void> launch_refine(
        uint32_t const* query_planes,
        uint32_t query_id_offset,
        uint32_t query_count,
        uint32_t indexed_bucket_count,
        uint16_t key_mask,
        uint32_t const* candidate_bits,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream,
        uint32_t minimum_matches = 0U,
        uint32_t* pass_bits = nullptr
    ) const {
        auto const counting = !result_match_counts.empty() || pass_bits != nullptr;
        auto const kernel = counting ? detail::refine_batch_bitmap_kernel<
                                           BucketCount,
                                           true,
                                           batch_result_type,
                                           Candidates,
                                           UpperTriangle>
                                     : detail::refine_batch_bitmap_kernel<
                                           BucketCount,
                                           false,
                                           batch_result_type,
                                           Candidates,
                                           UpperTriangle>;
        constexpr auto shared_bytes = detail::bitmap_refine_dynamic_bytes<BucketCount>;
        CUDDL_CUDA_TRY(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)
        ));
        int resident_blocks = 0;
        CUDDL_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, kernel, detail::bitmap_refine_block_size, shared_bytes
        ));
        auto const multiprocessors = static_cast<uint32_t>(
            stream.device().attribute(cuda::device_attributes::multiprocessor_count)
        );
        kernel<<<
            multiprocessors* static_cast<uint32_t>(resident_blocks > 0 ? resident_blocks : 1),
            detail::bitmap_refine_block_size,
            shared_bytes,
            stream.get()>>>(
            query_planes,
            query_id_offset,
            query_count,
            planes_.data(),
            metadata_.reference_count,
            indexed_bucket_count,
            key_mask,
            candidate_bits,
            results.data(),
            result_match_counts.empty() ? nullptr : result_match_counts.data(),
            minimum_matches,
            pass_bits
        );
        return cuda_try(cudaGetLastError());
    }

    /// @brief Finds and refines one query tile's pairs with at least `options.minimum_matches`
    /// index matches.
    ///
    /// @p external_queries points at the tile's first external query row, or is null for an
    /// all-to-all tile, whose queries are database rows `first_query ...`. The pass bitmap comes
    /// either from the refinement itself (@ref compares_planes) or from index counting, whose
    /// exact match counts select the pairs the refinement then compares.
    template <bool AllToAll>
    [[nodiscard]] Result<batch_result_tile> launch_batch_indexed(
        score_type const* external_queries,
        uint32_t first_query,
        uint32_t query_count,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        device_span<uint32_t> result_match_counts,
        search_options options,
        cuda::stream_ref stream
    ) const {
        auto const requirements = CUDDL_TRY(make_batch_requirements(
            CUDDL_TRY(dense_batch_pair_count(query_count)), 0U, true, query_count
        ));
        auto const workspace_begin = reinterpret_cast<uintptr_t>(workspace.data());
        auto const workspace_end = workspace_begin + workspace.size_bytes();
        if (workspace_end < workspace_begin) {
            return Err(Error::resource("indexed batch workspace address range overflows"));
        }
        auto const word_count = static_cast<size_t>(query_count) *
                                detail::candidate_bit_words(metadata_.reference_count);
        auto address = detail::align_up(workspace_begin, alignof(uint32_t));
        auto* match_counts = reinterpret_cast<uint32_t*>(address);
        address += requirements.counter_bytes;
        auto* candidate_bits = reinterpret_cast<uint32_t*>(address);
        address += requirements.candidate_bytes;
        address = detail::align_up(address, plane_alignment);
        auto* staging = reinterpret_cast<void*>(address);
        address += query_plane_bytes(query_count);
        address = detail::align_up(address, 256U);
        if (address + requirements.temporary_bytes > workspace_end) {
            return Err(Error::resource("indexed batch workspace layout exceeds its capacity"));
        }
        auto* temporary_workspace = reinterpret_cast<void*>(address);
        constexpr uint32_t warp_width = 32;
        constexpr uint32_t warps_per_block = detail::block_size / warp_width;
        auto const multiprocessors = static_cast<uint32_t>(
            stream.device().attribute(cuda::device_attributes::multiprocessor_count)
        );
        auto const tile_pair_count = query_count * metadata_.reference_count;
        auto const reference_count = metadata_.reference_count;
        auto const indexed_bucket_count = metadata_.compatibility.indexed_bucket_count;
        auto const key_mask = metadata_.compatibility.key_mask;
        auto const result_tile = make_tile(
            results, result_match_counts, candidate_bits, query_id_offset, query_count, AllToAll
        );
        if (word_count == 0U) {
            return result_tile;
        }

        // External queries arrive as scores and are converted to bit-planes; all-to-all queries
        // are database planes, decoded to scores only for the index lookups.
        uint16_t const* query_scores = external_queries;
        uint32_t const* query_planes = nullptr;
        if constexpr (AllToAll) {
            query_planes = planes_.data() + static_cast<size_t>(first_query) * (BucketCount / 2U);
            if (!compares_planes()) {
                auto* decoded = static_cast<uint16_t*>(staging);
                detail::decode_score_planes_kernel<BucketCount>
                    <<<detail::warp_grid_blocks(
                           static_cast<size_t>(query_count) * (BucketCount / warp_width)
                       ),
                       detail::block_size,
                       0,
                       stream.get()>>>(query_planes, query_count, decoded);
                CUDDL_CUDA_TRY(cudaGetLastError());
                query_scores = decoded;
            }
        } else {
            auto* converted = static_cast<uint32_t*>(staging);
            detail::build_score_planes_kernel<BucketCount>
                <<<detail::warp_grid_blocks(
                       static_cast<size_t>(query_count) * (BucketCount / warp_width)
                   ),
                   detail::block_size,
                   0,
                   stream.get()>>>(external_queries, query_count, converted);
            CUDDL_CUDA_TRY(cudaGetLastError());
            query_planes = converted;
        }

        if (compares_planes()) {
            // Every passing pair's summary lands at its slot as the pass bitmap is written.
            CUDDL_TRY((launch_refine<detail::refine_candidates::all, AllToAll>(
                query_planes,
                query_id_offset,
                query_count,
                indexed_bucket_count,
                key_mask,
                nullptr,
                results,
                result_match_counts,
                stream,
                options.minimum_matches,
                candidate_bits
            )));
            return result_tile;
        }

        auto const bits_from_counts = [&]() -> Result<void> {
            detail::candidate_bits_from_counts_kernel<<<
                detail::warp_grid_blocks((word_count + warp_width - 1U) / warp_width),
                detail::block_size,
                0,
                stream.get()>>>(
                match_counts,
                query_count,
                reference_count,
                options.minimum_matches,
                query_id_offset,
                AllToAll,
                candidate_bits
            );
            return cuda_try(cudaGetLastError());
        };
        if (detail::uses_tiled_index_counts(reference_count, query_count, indexed_bucket_count)) {
            auto const tile =
                detail::index_tile_references(reference_count, query_count, multiprocessors);
            auto const tiles = (reference_count + tile - 1U) / tile;
            // Sparse ranges are resolved bucket-major first: a block's lookups all search one
            // bucket's keys, so their shared upper search levels stay in L1.
            uint2* ranges = nullptr;
            if (!index_keys_.empty()) {
                ranges = static_cast<uint2*>(temporary_workspace);
                detail::sparse_batch_posting_ranges_kernel<BucketCount>
                    <<<indexed_bucket_count, detail::block_size, 0, stream.get()>>>(
                        query_scores,
                        0U,
                        query_count,
                        index_keys_.data(),
                        key_directory_.data(),
                        reference_count,
                        indexed_bucket_count,
                        key_mask,
                        ranges
                    );
                CUDDL_CUDA_TRY(cudaGetLastError());
            }
            CUDDL_CUDA_TRY(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(
                    detail::count_batch_index_tile_kernel<BucketCount, uint16_t>
                ),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(detail::index_tile_max_references / 2U * sizeof(uint32_t))
            ));
            detail::count_batch_index_tile_kernel<BucketCount>
                <<<query_count * tiles,
                   detail::index_tile_block_size,
                   tile / 2U * sizeof(uint32_t),
                   stream.get()>>>(
                    query_scores,
                    0U,
                    index_offsets_.data(),
                    index_postings_.data(),
                    reference_count,
                    indexed_bucket_count,
                    key_mask,
                    tile,
                    match_counts,
                    index_keys_.empty() ? nullptr : index_keys_.data(),
                    ranges,
                    options.minimum_matches,
                    query_id_offset,
                    AllToAll,
                    candidate_bits
                );
            CUDDL_CUDA_TRY(cudaGetLastError());
        } else {
            CUDDL_CUDA_TRY(
                cuda::fill_bytes(
                    stream, cuda::std::span{match_counts, static_cast<size_t>(tile_pair_count)}, 0
                )
            );
            auto const query_buckets = static_cast<size_t>(query_count) * indexed_bucket_count;
            constexpr auto cells_per_block = warps_per_block * detail::index_match_cells_per_warp;
            auto const bucket_blocks = static_cast<uint32_t>(
                std::min<size_t>((query_buckets + cells_per_block - 1U) / cells_per_block, 65535U)
            );
            detail::count_batch_index_matches_kernel<BucketCount>
                <<<bucket_blocks, detail::block_size, 0, stream.get()>>>(
                    query_scores,
                    0U,
                    query_count,
                    index_offsets_.data(),
                    index_postings_.data(),
                    reference_count,
                    indexed_bucket_count,
                    key_mask,
                    match_counts,
                    index_keys_.data()
                );
            CUDDL_CUDA_TRY(cudaGetLastError());
            CUDDL_TRY(bits_from_counts());
        }
        CUDDL_TRY((launch_refine<detail::refine_candidates::bitmap, AllToAll>(
            query_planes,
            query_id_offset,
            query_count,
            indexed_bucket_count,
            key_mask,
            candidate_bits,
            results,
            result_match_counts,
            stream
        )));
        return result_tile;
    }

   private:
    [[nodiscard]] __host__ __device__ constexpr bool rows_match_metadata(
        size_t expected_scores
    ) const noexcept {
        return planes_.size() * 2U == expected_scores &&
               (expected_scores == 0U || planes_.data() != nullptr);
    }

    device_span<uint32_t const> planes_;
    reference_database_metadata metadata_;
    device_span<uint32_t const> index_offsets_;
    device_span<uint32_t const> index_postings_;
    device_span<uint16_t const> index_keys_;
    device_span<uint32_t const> key_directory_;
    double index_pair_fraction_{};
    bool indexed_{};
};

}  // namespace detail

/**
 * @brief Move-only owner of one immutable contiguous reference database.
 *
 * Building enqueues row copies on the supplied stream. Inputs and the returned database must
 * remain alive until that stream completes. The allocation stream must outlive the database.
 * Complete work on other streams before destroying or move-assigning the database.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class reference_database {
    friend class reference_index_file;
    template <uint32_t, size_t, typename>
    friend class reference_index;
    friend class reference_database_file;
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using layout_type = Layout;
    using score_type = uint16_t;
    using result_type = reference_search_result;
    using batch_result_type = packed_pairwise_counts;

    reference_database(reference_database const&) = delete;
    reference_database& operator=(reference_database const&) = delete;

    // An explicit body keeps NVCC from inferring device-side buffer destruction.
    __host__ ~reference_database() {}  // NOLINT(modernize-use-equals-default)

    reference_database(reference_database&& other) noexcept
        : planes_(std::move(other.planes_)),
          metadata_(std::exchange(other.metadata_, {})),
          names_(std::move(other.names_)),
          identity_(std::move(other.identity_)) {}

    reference_database& operator=(reference_database&& other) noexcept {
        if (this != &other) {
            planes_ = std::move(other.planes_);
            metadata_ = std::exchange(other.metadata_, {});
            names_ = std::move(other.names_);
            identity_ = std::move(other.identity_);
        }
        return *this;
    }

    /// @brief Builds a database from flat row-major scores on @p stream.
    ///
    /// The database keeps each row only as bit-planes (detail::score_plane_index), the layout
    /// every search compares 32 buckets at a time; @p rows is not retained.
    [[nodiscard]] static Result<reference_database> build_async(
        device_span<score_type const> rows,
        score_compatibility compatibility,
        cuda::stream_ref stream
    ) {
        auto const reference_count = CUDDL_TRY(validate_rows(rows, compatibility));
        auto database = CUDDL_CUDA_TRY(reference_database(stream));
        database.metadata_ = {.compatibility = compatibility, .reference_count = reference_count};
        if (!rows.empty()) {
            database.planes_ = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint32_t>(
                    stream, stream.device(), rows.size() / 2U, cuda::no_init
                )
            );
            detail::build_score_planes_kernel<BucketCount>
                <<<detail::
                       warp_grid_blocks(static_cast<size_t>(reference_count) * (BucketCount / 32U)),
                   detail::block_size,
                   0,
                   stream.get()>>>(rows.data(), reference_count, database.planes_.data());
            CUDDL_CUDA_TRY(cudaGetLastError());
        }
        return Result<reference_database>::ok(std::move(database));
    }

    /// @brief Writes the reference scores, row-major in reference-ID order, to @p scores.
    [[nodiscard]] Result<void>
    copy_scores_async(device_span<score_type> scores, cuda::stream_ref stream) const {
        auto const count = static_cast<size_t>(metadata_.reference_count) * BucketCount;
        if (scores.size() < count) {
            return Err(Error::resource("score destination is too small"));
        }
        if (count == 0U) {
            return Ok();
        }
        if (scores.data() == nullptr) {
            return Err(Error::invalid_argument("score destination must be device accessible"));
        }
        detail::decode_score_planes_kernel<BucketCount>
            <<<detail::warp_grid_blocks(
                   static_cast<size_t>(metadata_.reference_count) * (BucketCount / 32U)
               ),
               detail::block_size,
               0,
               stream.get()>>>(planes_.data(), metadata_.reference_count, scores.data());
        return cuda_try(cudaGetLastError());
    }

    /// @brief Reference labels in reference-ID order.
    [[nodiscard]] std::span<std::string const> names() const noexcept {
        return names_;
    }

    /// @brief Compatibility metadata and reference count.
    [[nodiscard]] reference_database_metadata metadata() const noexcept {
        return metadata_;
    }

    /// @brief Reference count.
    [[nodiscard]] uint32_t reference_count() const noexcept {
        return metadata_.reference_count;
    }

    /// @brief Bytes one compact row store needs for @p reference_count references.
    [[nodiscard]] static constexpr size_t persistent_row_bytes(uint32_t reference_count) noexcept {
        return view_type::persistent_row_bytes(reference_count);
    }

    /// @brief Bytes this database's own row store needs.
    [[nodiscard]] size_t persistent_row_bytes() const noexcept {
        return view().persistent_row_bytes();
    }

    [[nodiscard]] static constexpr size_t single_query_workspace_bytes(
        uint32_t reference_count
    ) noexcept {
        return view_type::single_query_workspace_bytes(reference_count);
    }

    /// @brief Caller-owned bytes for one exhaustive query on this database; always zero.
    [[nodiscard]] size_t single_query_workspace_bytes() const noexcept {
        return view().single_query_workspace_bytes();
    }

    [[nodiscard]] static constexpr uint32_t all_to_all_result_capacity(
        uint32_t reference_count
    ) noexcept {
        return view_type::all_to_all_result_capacity(reference_count);
    }

    [[nodiscard]] static constexpr uint32_t single_query_result_count(
        uint32_t reference_count
    ) noexcept {
        return view_type::single_query_result_count(reference_count);
    }

    /// @brief Results one exhaustive query writes: one per reference.
    [[nodiscard]] uint32_t single_query_result_count() const noexcept {
        return view().single_query_result_count();
    }

    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        cuda::stream_ref stream
    ) const {
        return view().search_async(query, query_compatibility, workspace, results, stream);
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        return view().search_batch_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        cuda::stream_ref stream
    ) const {
        return view().search_all_to_all_async(
            workspace, results, std::forward<OnTile>(on_tile), result_match_counts, stream
        );
    }

    [[nodiscard]] Result<size_t> search_workspace_bytes(
        cuda::stream_ref stream,
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.indexed_single_query_workspace_bytes(stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> batch_search_requirements(
        uint32_t query_count,
        cuda::stream_ref stream,
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.indexed_batch_search_requirements(query_count, stream);
    }

    [[nodiscard]] Result<cuddl::batch_search_requirements> all_to_all_search_requirements(
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.indexed_all_to_all_search_requirements();
    }

    [[nodiscard]] Result<void> search_async(
        device_span<score_type const> query,
        score_compatibility const& query_compatibility,
        device_span<uint8_t> workspace,
        device_span<result_type> results,
        device_span<uint32_t> result_count,
        search_options options,
        cuda::stream_ref stream,
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.search_indexed_async(
            query, query_compatibility, workspace, results, result_count, options, stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_batch_async(
        device_span<score_type const> queries,
        score_compatibility const& query_compatibility,
        uint32_t query_id_offset,
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        search_options options,
        cuda::stream_ref stream,
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.search_batch_indexed_async(
            queries,
            query_compatibility,
            query_id_offset,
            workspace,
            results,
            std::forward<OnTile>(on_tile),
            result_match_counts,
            options,
            stream
        );
    }

    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all_async(
        device_span<uint8_t> workspace,
        device_span<batch_result_type> results,
        OnTile&& on_tile,
        device_span<uint32_t> result_match_counts,
        search_options options,
        cuda::stream_ref stream,
        reference_index<K, BucketCount, Layout> const* index = nullptr
    ) const {
        auto search_view = CUDDL_TRY(view(index));
        return search_view.search_all_to_all_indexed_async(
            workspace, results, std::forward<OnTile>(on_tile), result_match_counts, options, stream
        );
    }

   private:
    using view_type = detail::reference_database_view<K, BucketCount, Layout>;

    [[nodiscard]] view_type view() const noexcept {
        return view_type({planes_.data(), planes_.size()}, metadata_);
    }

    [[nodiscard]] Result<view_type> view(
        reference_index<K, BucketCount, Layout> const* index
    ) const {
        if (index == nullptr) return view();
        if (!identity_ || index->identity_ != identity_ || !index->indexed_) {
            return Err(Error::invalid_argument("index does not belong to this database"));
        }
        auto const offset_count = index->index_offsets_.size();
        auto const offsets =
            device_span<uint32_t const>{index->index_offsets_.data(), offset_count};
        auto const postings = device_span<uint32_t const>{
            index->index_postings_.data(), index->index_posting_capacity_
        };
        return view_type(
            {planes_.data(), planes_.size()},
            metadata_,
            offsets,
            postings,
            index->indexed_,
            index->index_keys_,
            index->key_directory_,
            index->pair_fraction_
        );
    }

    explicit reference_database(cuda::stream_ref stream)
        : planes_(stream, cuda::device_default_memory_pool(stream.device())) {}

    [[nodiscard]] static Result<uint32_t>
    validate_rows(device_span<score_type const> rows, score_compatibility compatibility) {
        CUDDL_TRY(
            (detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(compatibility))
        );
        if (rows.size() % BucketCount != 0U) {
            return Err(Error::invalid_argument("row extent must contain complete rows"));
        }
        if (!rows.empty() && rows.data() == nullptr) {
            return Err(Error::invalid_argument("rows must be device accessible"));
        }
        auto const reference_count = rows.size() / BucketCount;
        if (reference_count > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("reference count exceeds stable 32-bit IDs"));
        }
        return static_cast<uint32_t>(reference_count);
    }

    cuda::device_buffer<uint32_t> planes_;
    reference_database_metadata metadata_{};
    std::vector<std::string> names_;
    // A shared identity prevents accepting an index for a different or recycled row allocation.
    std::shared_ptr<char const> identity_ = std::make_shared<char>(0);
};

}  // namespace cuddl

#include <cuddl/reference_index.cuh>
