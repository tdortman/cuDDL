#pragma once

#include <cuda/std/cstdint>

#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuddl/detail/dna.hpp>
#include <cuddl/detail/fastx_sequence_file.hpp>

#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Result of parsing one FASTA/FASTQ file into packed canonical k-mers.
struct fasta_parse_result {
    /// Packed canonical k-mers from all records, treated as one combined genome.
    std::vector<uint64_t> kmers;
    /// Total valid sequence bases read.
    uint64_t bases = 0;
    /// Number of k-mers emitted (one per rolling window of `k` valid bases).
    uint64_t valid_kmers = 0;
    /// Number of partial windows broken by an invalid base before reaching length `k`.
    uint64_t invalid_windows = 0;
};

/// @brief Reverse-complements a packed k-mer (2 bits per base) with bit-parallel reversal.
///
/// Reverses the order of the 2-bit base digits with mask-swap rounds, shifts the reversed `2k`
/// bits down from the high end, then complements every base (`^0b10`). Byte-identical to the
/// scalar loop formulation for every `k <= 32`; longer k-mers fall back to the scalar loop.
inline constexpr uint64_t reverse_complement(uint64_t packed, uint32_t k) noexcept {
    if (k > 32) {
        uint64_t result = 0;
        for (uint32_t base = 0; base < k; ++base) {
            result = (result << 2U) | static_cast<uint64_t>(static_cast<uint8_t>(packed & 3U) ^ 2U);
            packed >>= 2U;
        }
        return result;
    }
    uint64_t x = packed;
    x = ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2U) | ((x & 0x3333333333333333ULL) << 2U);
    x = ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4U) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4U);
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8U) | ((x & 0x00FF00FF00FF00FFULL) << 8U);
    x = ((x & 0xFFFF0000FFFF0000ULL) >> 16U) | ((x & 0x0000FFFF0000FFFFULL) << 16U);
    x = (x >> 32U) | (x << 32U);
    x >>= (64U - 2U * k);
    auto const mask = (k == 32) ? ~uint64_t{0} : ((1ULL << (2U * k)) - 1ULL);
    return (x ^ 0xAAAAAAAAAAAAAAAAULL) & mask;
}

/// @brief One parallel scan span of the combined FASTA sequence stream.
///
/// `segments` covers the span's bytes in order; it may cross record boundaries, with header bytes
/// excluded. `prefix` holds the raw bytes immediately before the span's first position (headers
/// excluded) that a rolling-window consumer must replay to rebuild its window state: up to
/// `seed_bases` valid bases, stopping early at an invalid base or the start of the buffer.
struct fastx_sequence_span {
    std::string prefix;
    std::vector<std::string_view> segments;
    uint64_t total_bytes = 0;
};

namespace {

inline bool span_step_back(
    std::vector<fastx_sequence_extent> const& extents,
    size_t& extent,
    char const*& position
) {
    if (position > extents[extent].begin) {
        --position;
        return true;
    }
    if (extent > 0) {
        --extent;
        position = extents[extent].end - 1;
        return true;
    }
    return false;
}

inline bool span_step_forward(
    std::vector<fastx_sequence_extent> const& extents,
    size_t& extent,
    char const*& position
) {
    if (position + 1 < extents[extent].end) {
        ++position;
        return true;
    }
    if (extent + 1 < extents.size()) {
        ++extent;
        position = extents[extent].begin;
        return true;
    }
    return false;
}

}  // namespace

/**
 * @brief Splits the combined FASTA sequence stream into @p count contiguous spans.
 *
 * Each span carries the raw byte ranges of its slice of the sequence stream plus a materialised
 * prefix of the preceding @p seed_bases valid bases (whitespace included, headers excluded, cut
 * short at an invalid base or the buffer start). A rolling-window consumer that replays the
 * prefix, then consumes the segments, reproduces the single-threaded byte stream exactly, so the
 * per-span results can be concatenated in order. Whitespace between bases is preserved in both
 * the prefix and the segments; consumers skip it with @ref fastx_is_sequence_whitespace.
 */
[[nodiscard]] inline std::vector<fastx_sequence_span> fastx_split_sequence_spans(
    std::vector<fastx_sequence_extent> const& extents,
    uint32_t seed_bases,
    uint32_t count
) {
    std::vector<fastx_sequence_span> spans;
    if (extents.empty() || count == 0U) {
        return spans;
    }
    size_t total = 0;
    for (auto const& extent : extents) {
        total += static_cast<size_t>(extent.end - extent.begin);
    }
    if (total == 0) {
        return spans;
    }
    if (static_cast<size_t>(count) > total) {
        count = static_cast<uint32_t>(total);
    }
    auto const per = total / count + (total % count != 0U);
    spans.reserve(count);

    for (uint32_t t = 0; t < count; ++t) {
        auto const span_begin = static_cast<size_t>(t) * per;
        if (span_begin >= total) {
            break;
        }
        auto const span_end = span_begin + per < total ? span_begin + per : total;

        fastx_sequence_span span;
        span.total_bytes = span_end - span_begin;

        // Locate the extent and offset of the span start.
        size_t extent = 0;
        size_t offset = span_begin;
        while (extent < extents.size() &&
               offset >= static_cast<size_t>(extents[extent].end - extents[extent].begin)) {
            offset -= static_cast<size_t>(extents[extent].end - extents[extent].begin);
            ++extent;
        }

        // Seed: walk back up to seed_bases valid bases (or until an invalid base / EOF).
        size_t seed_extent = extent;
        char const* seed_start = extents[extent].begin + offset;
        {
            size_t e = extent;
            char const* q = seed_start;
            uint32_t collected = 0;
            while (collected < seed_bases && span_step_back(extents, e, q)) {
                auto const ch = *q;
                if (fastx_is_sequence_whitespace(ch)) {
                    continue;
                }
                if (encode_base(ch) == 0xffu) {
                    seed_extent = e;
                    seed_start = q;
                    span_step_forward(extents, seed_extent, seed_start);
                    break;
                }
                ++collected;
                seed_extent = e;
                seed_start = q;
            }
        }

        // Materialise the prefix from the seed start to the span start.
        {
            size_t e = seed_extent;
            char const* q = seed_start;
            char const* target = extents[extent].begin + offset;
            while (e < extent || q < target) {
                if (q >= extents[e].end) {
                    ++e;
                    q = extents[e].begin;
                    continue;
                }
                span.prefix.push_back(*q);
                ++q;
            }
        }

        // Collect the span's segments.
        {
            size_t remaining = span_end - span_begin;
            char const* q = extents[extent].begin + offset;
            size_t e = extent;
            while (remaining > 0 && e < extents.size()) {
                auto const limit = static_cast<size_t>(extents[e].end - q);
                auto const step = limit < remaining ? limit : remaining;
                span.segments.emplace_back(q, step);
                remaining -= step;
                if (remaining > 0) {
                    ++e;
                    if (e < extents.size()) {
                        q = extents[e].begin;
                    }
                }
            }
        }

        spans.push_back(std::move(span));
    }
    return spans;
}

/// @brief One worker's rolling-window state and output.
struct span_result {
    std::vector<uint64_t> kmers;
    uint64_t bases = 0;
    uint64_t valid_kmers = 0;
    uint64_t invalid_windows = 0;
};

/// @brief Feeds one byte of sequence through the DDL rolling-window extraction.
inline void consume_byte(
    char ch,
    uint32_t k,
    uint64_t mask,
    span_result& result,
    uint64_t& window,
    uint32_t& window_len
) {
    if (fastx_is_sequence_whitespace(ch)) {
        return;
    }
    auto const encoded = encode_base(ch);
    if (encoded == 0xffu) {
        if (window_len < k && window_len > 0) {
            ++result.invalid_windows;
        }
        window_len = 0;
        window = 0;
        return;
    }
    ++result.bases;
    window = ((window << 2U) | encoded) & mask;
    if (window_len < k) {
        ++window_len;
    }
    if (window_len == k) {
        auto const rev = reverse_complement(window, k);
        result.kmers.push_back(window > rev ? window : rev);
        ++result.valid_kmers;
    }
}

/**
 * @brief Parses a FASTA/FASTQ file's sequence bases into packed canonical k-mers of length @p k.
 *
 * Record headers and FASTQ qualities are excluded, and every record's sequence is fed through the
 * DDL rolling-window extraction as one combined genome. An invalid or ambiguous base breaks the
 * rolling window, and no k-mer spanning it is emitted. Each k-mer is canonicalised to the larger
 * packed orientation. Files above 1 MiB are parsed in parallel: the sequence stream is split
 * into seeded spans (@ref fastx_split_sequence_spans), each worker replays its
 * window prefix and then consumes its segments, and the per-span results concatenate to the
 * byte-identical single-threaded output.
 *
 * @param threads Worker count for parallel files (0 selects
 *        `std::thread::hardware_concurrency()`); files at or below 1 MiB parse serially
 *        regardless.
 *
 * @return A `fasta_parse_result`, or an error if the file cannot be opened, mapped, or contains
 *         bytes but no FASTA records, or malformed FASTQ records.
 */
inline Result<fasta_parse_result>
parse_fasta(std::string const& path, uint32_t k, unsigned threads = 0) {
    auto source = CUDDL_TRY(load_fastx_sequence_file(path));
    auto& extents = source->extents;
    if (extents.empty()) return fasta_parse_result{};

    // The span splitter otherwise joins extents and seeds windows across record headers.
    // A separate invalid-base extent stops prefix replay. Workers recognize its address and
    // reset without counting the synthetic separator as an invalid input window.
    static constexpr char record_boundary = 'N';
    if (extents.size() > 1) {
        std::vector<fastx_sequence_extent> separated;
        separated.reserve(extents.size() * 2 - 1);
        for (auto const& extent : extents) {
            if (!separated.empty()) separated.push_back({&record_boundary, &record_boundary + 1});
            separated.push_back(extent);
        }
        extents = std::move(separated);
    }

    uint64_t total = 0;
    for (auto const& extent : extents) {
        total += static_cast<uint64_t>(extent.end - extent.begin);
    }

    unsigned worker_count = 1;
    if (total >= (uint64_t{1} << 20)) {
        auto const available = threads > 0 ? threads : std::thread::hardware_concurrency();
        worker_count = available > 0 ? available : 1U;
    }

    auto const spans = fastx_split_sequence_spans(extents, k > 0 ? k - 1U : 0U, worker_count);

    auto const mask = (k >= 32) ? ~uint64_t{0} : ((1ULL << (2U * k)) - 1ULL);
    std::vector<span_result> partials(spans.size());
    auto const per_thread = total / spans.size() + (total % spans.size() != 0U);
    for (auto& partial : partials) {
        partial.kmers.reserve(per_thread / 2U + 64U);
    }

    std::vector<std::thread> workers;
    workers.reserve(spans.size());
    for (size_t t = 0; t < spans.size(); ++t) {
        workers.emplace_back([&, t] {
            uint64_t window = 0;
            uint32_t window_len = 0;
            span_result sink{};  // prefix replay: no counts, no emissions
            for (char ch : spans[t].prefix) {
                consume_byte(ch, k, mask, sink, window, window_len);
            }
            for (auto const& segment : spans[t].segments) {
                if (segment.data() == &record_boundary) {
                    window = 0;
                    window_len = 0;
                    continue;
                }
                for (char ch : segment) {
                    consume_byte(ch, k, mask, partials[t], window, window_len);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    fasta_parse_result result;
    uint64_t total_valid = 0;
    for (auto const& partial : partials) {
        result.bases += partial.bases;
        total_valid += partial.valid_kmers;
        result.invalid_windows += partial.invalid_windows;
    }
    result.valid_kmers = total_valid;
    // One contiguous allocation for the final stream: reserve the exact count, then move each
    // partial in and release it immediately so the peak never exceeds the final size plus one
    // partial (the machine's working set is the result stream itself).
    result.kmers.reserve(total_valid);
    for (auto& partial : partials) {
        result.kmers.insert(
            result.kmers.end(),
            std::make_move_iterator(partial.kmers.begin()),
            std::make_move_iterator(partial.kmers.end())
        );
        partial.kmers = std::vector<uint64_t>{};
    }
    return result;
}

}  // namespace cuddl::detail
