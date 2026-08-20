#pragma once

#include <cuda/std/cstdint>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cusbf/detail/fastx_buffer_reader.hpp>
#include <cusbf/detail/fastx_file_buffer.hpp>

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

/// @brief Reverse-complements a packed k-mer of length @p k (2 bits per base).
inline constexpr uint64_t reverse_complement(uint64_t packed, uint32_t k) noexcept {
    uint64_t result = 0;
    for (uint32_t base = 0; base < k; ++base) {
        result = (result << 2U) | static_cast<uint64_t>(static_cast<uint8_t>(packed & 3U) ^ 2U);
        packed >>= 2U;
    }
    return result;
}

namespace {

/// @brief Feeds one record's sequence through the DDL rolling-window extraction.
void consume_sequence(
    std::string_view sequence,
    uint32_t k,
    fasta_parse_result& result,
    uint64_t& window,
    uint32_t& window_len
) {
    constexpr uint8_t invalid = 0xffu;
    const uint64_t mask = (k >= 32) ? ~uint64_t{0} : ((1ULL << (2U * k)) - 1ULL);
    for (char ch : sequence) {
        if (ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t') {
            continue;
        }
        auto const encoded = encode_base(ch);
        if (encoded == invalid) {
            if (window_len < k && window_len > 0) {
                ++result.invalid_windows;
            }
            window_len = 0;
            window = 0;
            continue;
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
}

}  // namespace

/**
 * @brief Parses a FASTA file's sequence bases into packed canonical k-mers of length @p k.
 *
 * Uses cuSBF's `FastxFileBuffer` (Linux mmap) to load the file and its `FastxBufferReader` to
 * recover every record, then feeds each record's sequence through the DDL rolling-window
 * extraction as one combined genome. An invalid or ambiguous base breaks the rolling window, and
 * no k-mer spanning it is emitted. Each k-mer is canonicalised to the larger packed orientation.
 *
 * @return A `fasta_parse_result`, or an error if the file cannot be opened or mapped.
 */
inline Result<fasta_parse_result> parse_fasta(std::string const& path, uint32_t k) {
    auto file = cusbf::detail::FastxFileBuffer::load(path);
    if (!file) {
        return Err(Error::invalid_argument("cannot open FASTA file: " + path));
    }
    auto const data = (*file)->data();

    cusbf::detail::FastxBufferReader reader(data);
    cusbf::detail::FastxRecord record;

    fasta_parse_result result;
    result.kmers.reserve(data.size() / 2);
    uint64_t window = 0;
    uint32_t window_len = 0;

    while (true) {
        auto next = reader.nextRecord(record);
        if (!next) {
            return Err(Error::invalid_argument("FASTA parse error near: " + path));
        }
        if (!*next) {
            break;  // end of stream
        }
        consume_sequence(record.sequence, k, result, window, window_len);
    }
    return result;
}

}  // namespace cuddl::detail
