#pragma once

#include <cuda/std/cstdint>

#include <string>
#include <utility>
#include <vector>

#include <cuddl/detail/fasta_parser.hpp>
#include <cuddl/error.hpp>

namespace cuddl {

/// @brief Result of parsing one FASTA file into packed canonical k-mers.
using fasta_parse_result = detail::fasta_parse_result;

/**
 * @brief Parses a FASTA file's sequence bases into packed canonical k-mers of length @p k.
 *
 * This is the FASTX entry point. It reads every record as one combined genome and emits a
 * packed canonical k-mer per contiguous run of `k` valid bases; an invalid or ambiguous base
 * breaks the rolling window, and no k-mer spanning it is emitted. See @ref detail::parse_fasta.
 *
 * @return Parsed k-mers and counts, or an error if the file cannot be opened.
 */
inline Result<fasta_parse_result> parse_fasta_file(std::string const& path, uint32_t k) {
    return detail::parse_fasta(path, k);
}

}  // namespace cuddl
