#pragma once

#include <cuddl/a48.hpp>
#include <cuddl/reference_database.cuh>

namespace cuddl {

/// @brief Builds a cuDDL score-compatibility object describing a decoded BBTools A48 database.
///
/// Decoded rows come from the BBTools encoder/hash pipeline, not the cuDDL one. The encoder and
/// hash identity fields therefore carry nonzero sentinels (2) distinct from cuDDL's own
/// construction constants, while the mantissa/exponent split and bucket/key masks are taken from
/// the asset's header so decoded rows pass cuDDL's normal validation paths.
[[nodiscard]] inline score_compatibility
decoded_compatibility(uint32_t k, uint32_t buckets, uint32_t exponent_bits, uint64_t seed) {
    score_compatibility compat;
    compat.kmer_length = k;
    compat.bucket_count = buckets;
    compat.indexed_bucket_count = buckets;
    compat.score_encoder_identity = 2U;
    compat.exponent_bits = static_cast<uint16_t>(exponent_bits);
    compat.mantissa_bits = static_cast<uint16_t>(16U - exponent_bits);
    compat.hash_identity = 2U;
    compat.hash_seed = seed;
    compat.canonicalisation_policy = 1U;
    compat.blacklist_identity = 0U;
    compat.blacklist_version = 0U;
    compat.key_mask = 0xffffU;
    return compat;
}

}  // namespace cuddl
