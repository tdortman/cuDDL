#pragma once

#include <array>
#include <string_view>

#if CUDDL_ABLATIONS
    #include <algorithm>
    #include <cstdio>
    #include <cstdlib>
#endif

namespace cuddl::detail {

/// @brief Optimisations that the `CUDDL_ABLATE` environment variable can turn off in a build
/// configured with `-Dablations=true` (`CUDDL_ABLATIONS=1`).
///
/// They exist to measure what each optimisation contributes (`scripts/ablation.py`); results
/// are the same either way. Other builds compile every switch out.
inline constexpr std::array<std::string_view, 22> ablations{
    "co-scheduling",       // nvCOMP claims a batch only if it finishes before the host loaders
    "read-depth",          // readers keep parallel_read_depth reads in flight, not one per loader
    "probe-depth",         // 4 * parallel_read_depth gzip probes in flight, not parallel_read_depth
    "pinned-pool",         // device-only pinned pools that keep their pages, not CCCL-like ones
    "numa-pool",           // device-written pinned buffers on the device's host NUMA node
    "pinned-reservation",  // packed-word buffers pinned before the loaders start
    "input-lanes",         // GPU lanes sized from the input, not a fixed 1 GiB
    "simd-parsing",        // AVX2/AVX-512/NEON line scans, compaction, packing and gzip scans
    "base-packing",        // the CPU path packs bases into two-bit words before copying them
    "kmer-encoding",       // the GPU builds eight windows per thread from shared SWAR-packed tiles
    "deltas",              // near-duplicate rows corrected from a base row, not compared in full
    "bit-slicing",         // exhaustive search on bit-planes, not packed row-major 16-bit scores
    "lop3-kernel",         // the tiled LOP3 plane kernel, not one bucket per lane step
    "tile-queue",          // ranks claim all-to-all tiles from a shared counter, not a fixed split
    "progress-thread",     // rank 0 drives MPI progress for the shared tile counter
    "match-filter",        // a low-plane bound selects candidates before exact refinement
    "posting-bitmaps",     // long posting lists of the inverted index are kept as bitmaps
    "index-fallback",      // an index whose lists cover too many pairs is bypassed
    "inverted-index",      // threshold searches use an inverted index (benchmarks only)
    "direct-io",           // result files are written with O_DIRECT, not through the page cache
    "io-uring",            // direct result writes keep several blocks in flight with io_uring
    "write-probe",         // writers that need not be durable use the page cache while it is faster
};

#if CUDDL_ABLATIONS
/// @brief Whether this build compiles the ablation switches in.
inline constexpr bool ablations_enabled = true;

/// @brief Whether `CUDDL_ABLATE`, a comma-separated list of @ref ablations, turns off @p name.
///
/// The list is read once. A name not in @ref ablations aborts the process, so a misspelt
/// ablation cannot pass for a measurement of the optimised code.
inline bool ablated(std::string_view name) noexcept {
    static auto const list = [] {
        char const* const value = std::getenv("CUDDL_ABLATE");
        std::string_view const text = value == nullptr ? std::string_view{} : value;
        for (size_t start = 0; start < text.size();) {
            auto end = text.find(',', start);
            if (end == std::string_view::npos) end = text.size();
            auto const item = text.substr(start, end - start);
            if (!item.empty() && std::ranges::find(ablations, item) == ablations.end()) {
                std::fprintf(
                    stderr, "CUDDL_ABLATE: unknown ablation '%.*s'\n", int(item.size()), item.data()
                );
                std::abort();
            }
            start = end + 1;
        }
        return text;
    }();
    for (size_t start = 0; start < list.size();) {
        auto end = list.find(',', start);
        if (end == std::string_view::npos) end = list.size();
        if (list.substr(start, end - start) == name) return true;
        start = end + 1;
    }
    return false;
}
#else
inline constexpr bool ablations_enabled = false;

/// @brief Always false: this build compiles the ablation switches out.
constexpr bool ablated(std::string_view) noexcept {
    return false;
}
#endif

}  // namespace cuddl::detail
