#pragma once

#include <cuddl/reference_database_file.cuh>

#include <stdexcept>
#include <string>

/// Sketch configuration fixed at build time by the cli_* Meson options.
namespace cli {

constexpr uint32_t kmer_length = CUDDL_CLI_KMER_LENGTH;
constexpr size_t buckets = CUDDL_CLI_BUCKETS;
constexpr uint32_t exponent_bits = CUDDL_CLI_EXPONENT_BITS;
using layout = cuddl::register_layout<exponent_bits, 16U - exponent_bits>;

/// @brief Rejects a database whose sketch configuration differs from this build's.
inline void require_compatible(cuddl::score_compatibility const& compatibility) {
    if (compatibility.kmer_length == kmer_length && compatibility.bucket_count == buckets &&
        compatibility.exponent_bits == exponent_bits) {
        return;
    }
    throw std::invalid_argument(
        "database uses k=" + std::to_string(compatibility.kmer_length) + ", " +
        std::to_string(compatibility.bucket_count) + " buckets and " +
        std::to_string(compatibility.exponent_bits) +
        " exponent bits; reconfigure with -Dcli_kmer_length=" +
        std::to_string(compatibility.kmer_length) +
        " -Dcli_buckets=" + std::to_string(compatibility.bucket_count) +
        " -Dcli_exponent_bits=" + std::to_string(compatibility.exponent_bits) + " and rebuild"
    );
}

}  // namespace cli
