#pragma once

#include <cuda/std/cstdint>

#include <string>
#include <utility>
#include <vector>

#include <cuddl/detail/fasta_parser.hpp>
#include <cuddl/error.hpp>

namespace cuddl {

/// @brief Result of parsing one FASTA/FASTQ file into packed canonical k-mers.
using fasta_parse_result = detail::fasta_parse_result;

/**
 * @brief Parses a FASTA/FASTQ file's sequence bases into packed canonical k-mers of length @p k.
 *
 * Gzip/BGZF is detected by magic bytes and decompressed before parsing.
 * Format is detected from the first record header; FASTQ qualities are validated and ignored.
 * This is the FASTX entry point. It reads every record as one combined genome and emits a
 * packed canonical k-mer per contiguous run of `k` valid bases; an invalid or ambiguous base
 * breaks the rolling window, and no k-mer spanning it is emitted. Record boundaries also reset
 * the window, without contributing to `invalid_windows`. See @ref detail::parse_fasta.
 *
 * @param threads Worker count for parallel files (0 selects
 *        `std::thread::hardware_concurrency()`); files at or below 1 MiB parse serially.
 *
 * @return Parsed k-mers and counts, or an error if the file cannot be opened.
 */
inline Result<fasta_parse_result> parse_fasta_file(
    std::string const& path, uint32_t k, unsigned threads = 0
) {
    return detail::parse_fasta(path, k, threads);
}

}  // namespace cuddl
