#pragma once

#include <cuda/std/cstdint>

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cusbf/detail/fastx_file_buffer.hpp>
#include <cusbf/detail/fastx_sequence_scan.hpp>

#include <cuddl/error.hpp>

namespace cuddl::detail {

/// @brief Result of parsing one FASTA file into packed canonical k-mers.
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

/// @brief Maps one DNA byte to its 2-bit symbol (A=0, C=1, T=2, G=3) or `0xff` when invalid.
///
/// Mirrors cuSBF's `DnaAlphabet::encode` bit trick; case-insensitive.
inline constexpr uint8_t encode_base(char base) noexcept {
    auto const byte = static_cast<uint8_t>(base);
    auto const upper = static_cast<uint8_t>(byte & 0xDFu);
    auto const x = (byte >> 1u) & 3u;
    auto const valid =
        static_cast<uint8_t>((upper == 'A') | (upper == 'C') | (upper == 'G') | (upper == 'T'));
    auto const mask = static_cast<uint8_t>(0u - valid);
    return static_cast<uint8_t>((x & mask) | (0xFFu & ~mask));
}

/// @brief Reverse-complements a packed k-mer (2 bits per base) with bit-parallel reversal.
///
/// Reverses the order of the 2-bit base digits with mask-swap rounds, shifts the reversed `2k`
/// bits down from the high end, then complements every base (`^0b10`). Byte-identical to the
/// scalar loop formulation for every `k <= 32`; longer k-mers fall back to the scalar loop.
inline constexpr uint64_t reverse_complement(uint64_t packed, uint32_t k) noexcept {
    if (k > 32) {
        uint64_t result = 0;
        for (uint32_t base = 0; base < k; ++base) {
            result =
                (result << 2U) | static_cast<uint64_t>(static_cast<uint8_t>(packed & 3U) ^ 2U);
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

namespace {

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
    if (cusbf::detail::fastx_is_sequence_whitespace(ch)) {
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

}  // namespace

/**
 * @brief Parses a FASTA file's sequence bases into packed canonical k-mers of length @p k.
 *
 * Record header lines are excluded, and every record's sequence is fed through the DDL
 * rolling-window extraction as one combined genome. An invalid or ambiguous base breaks the
 * rolling window, and no k-mer spanning it is emitted. Each k-mer is canonicalised to the larger
 * packed orientation. Files above 1 MiB are parsed in parallel: cuSBF splits the sequence stream
 * into seeded spans (`cusbf::detail::fastx_split_sequence_spans`), each worker replays its
 * window prefix and then consumes its segments, and the per-span results concatenate to the
 * byte-identical single-threaded output.
 *
 * @param threads Worker count for parallel files (0 selects
 *        `std::thread::hardware_concurrency()`); files at or below 1 MiB parse serially
 *        regardless.
 *
 * @return A `fasta_parse_result`, or an error if the file cannot be opened, mapped, or contains
 *         bytes but no FASTA records.
 */
inline Result<fasta_parse_result> parse_fasta(std::string const& path, uint32_t k, unsigned threads = 0) {
    auto file = cusbf::detail::FastxFileBuffer::load(path);
    if (!file) {
        return Err(Error::invalid_argument("cannot open FASTA file: " + path));
    }
    auto const data = (*file)->data();

    auto const extents = cusbf::detail::fastx_fasta_extents(data);
    if (extents.empty()) {
        if (data.size() > 0) {
            return Err(Error::invalid_argument("FASTA parse error near: " + path));
        }
        return fasta_parse_result{};
    }

    uint64_t total = 0;
    for (auto const& extent : extents) {
        total += static_cast<uint64_t>(extent.end - extent.begin);
    }

    unsigned worker_count = 1;
    if (total >= (uint64_t{1} << 20)) {
        auto const available =
            threads > 0 ? threads : std::thread::hardware_concurrency();
        worker_count = available > 0 ? available : 1U;
    }

    auto const spans =
        cusbf::detail::fastx_split_sequence_spans(extents, k > 0 ? k - 1U : 0U, worker_count);

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
