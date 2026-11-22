// All-CPU ingest for cuddl_cpu_bitsliced_benchmark.cpp: lists genomes, then reads, gunzips,
// FASTA-parses and 2-bit packs each one with the library's host loader and sketches the packed
// runs into the DynamicDemiLog registers (k = 25, 2048 buckets, layout (5, 11)) that the GPU
// builder produces. The headers are only parsed: nothing here calls into the CUDA runtime.
// GenomeSketch::add_run splits each packed run into kLanes stretches rolled in lockstep by
// AVX-512, AVX2 or NEON vectors, selected from the target ISA (-march=native, -mcpu=native);
// add_windows is the scalar reference, also used for prefixes, tails and other ISAs.
// benchmarks/cuddl_cpu_sketch_check.cpp compares the two on synthetic runs.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) && defined(__AVX2__)
    #include <immintrin.h>
#endif

#include <cuddl/detail/fastx_sequence_file.hpp>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/register.cuh>

namespace cpu_ingest {

constexpr uint32_t kKmer = 25U;
constexpr uint32_t kBuckets = 2048U;
using Layout = cuddl::register_layout<5, 11>;

/// Wall-clock results of listing and ingesting one genome set.
struct IngestStats {
    double list_s = 0.0;
    double ingest_s = 0.0;
    uint64_t bases = 0;        // bases of every packed run of at least k bases
    uint64_t kmers = 0;        // windows hashed
    uint64_t input_bytes = 0;  // size of the input files on disk
    unsigned workers = 0;
};

/// The genome paths `cuddl::list_genomes` returns: a directory searched recursively and sorted
/// by path, or the lines of a text file in file order. These are the reference IDs of
/// `cuddl-build-reference-db`.
inline bool list_genomes(std::filesystem::path const& input, std::vector<std::filesystem::path>& paths) {
    try {
        if (std::filesystem::is_directory(input)) {
            for (auto const& entry : std::filesystem::recursive_directory_iterator(
                     input, std::filesystem::directory_options::skip_permission_denied
                 )) {
                if (!entry.is_symlink() && !entry.is_regular_file()) {
                    continue;
                }
                auto name = entry.path().filename().string();
                std::ranges::transform(name, name.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                auto extension = std::filesystem::path(name).extension().string();
                if (extension == ".gz" || extension == ".bgz" || extension == ".bgzf") {
                    extension = std::filesystem::path(name).stem().extension().string();
                }
                if (extension == ".fa" || extension == ".fna" || extension == ".fasta" ||
                    extension == ".ffn" || extension == ".frn" || extension == ".fq" ||
                    extension == ".fastq") {
                    paths.push_back(entry.path());
                }
            }
            std::ranges::sort(paths);
            return true;
        }
    } catch (std::filesystem::filesystem_error const& error) {
        std::fprintf(stderr, "cannot list %s: %s\n", input.c_str(), error.what());
        return false;
    }
    std::ifstream list(input);
    if (!list) {
        std::fprintf(stderr, "cannot read genome list %s\n", input.c_str());
        return false;
    }
    for (std::string line; std::getline(list, line);) {
        if (!line.empty()) {
            paths.emplace_back(line);
        }
    }
    return true;
}

namespace splitmix {
constexpr uint64_t kFirst = 0xbf58476d1ce4e5b9ULL;
constexpr uint64_t kSecond = 0x94d049bb133111ebULL;
}  // namespace splitmix

/// Lanes of the vectorized sketcher, selected from the target ISA; zero leaves the scalar path the
/// only one. Where the ISA has 64-bit vector multiplies (AVX-512 DQ and VL) or emulates them
/// cheaply (AVX2) the lanes also hash. NEON has none: its lanes only roll the k-mers and pick the
/// canonical ones, which scalar code hashes, and that beat both a vector hash and SVE2.
#if defined(__x86_64__) && defined(__AVX2__)
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
        #define CUDDL_CPU_SKETCH_AVX512 1
        #ifndef CUDDL_CPU_SKETCH_LANES
            #define CUDDL_CPU_SKETCH_LANES 8
        #endif
    #else
        #define CUDDL_CPU_SKETCH_AVX512 0
        #define CUDDL_CPU_SKETCH_LANES 4
    #endif
    #ifndef CUDDL_CPU_SKETCH_VECTOR_HASH
        #define CUDDL_CPU_SKETCH_VECTOR_HASH 1
    #endif
#elif defined(__aarch64__) && defined(__ARM_NEON)
    #define CUDDL_CPU_SKETCH_AVX512 0
    #ifndef CUDDL_CPU_SKETCH_LANES
        #define CUDDL_CPU_SKETCH_LANES 4
    #endif
    #define CUDDL_CPU_SKETCH_VECTOR_HASH 0
#else
    #define CUDDL_CPU_SKETCH_AVX512 0
    #define CUDDL_CPU_SKETCH_LANES 0
    #define CUDDL_CPU_SKETCH_VECTOR_HASH 0
#endif
constexpr uint32_t kLanes = CUDDL_CPU_SKETCH_LANES;

#if CUDDL_CPU_SKETCH_LANES
/// One 64-bit lane per concurrently rolled stretch of a run.
using Vec = uint64_t __attribute__((vector_size(8 * kLanes)));
using SignedVec = int64_t __attribute__((vector_size(8 * kLanes)));

inline Vec splat(uint64_t value) noexcept {
    return Vec{} + value;
}

inline void store(Vec v, uint64_t* out) noexcept {
    std::memcpy(out, &v, sizeof(v));
}

/// Lane l holds word `origin[l * stride + t]`.
inline Vec load_words(uint32_t const* origin, uint32_t stride, uint32_t t) noexcept {
    return [&]<uint32_t... L>(std::integer_sequence<uint32_t, L...>) noexcept {
        return Vec{origin[L * stride + t]...};
    }(std::make_integer_sequence<uint32_t, kLanes>{});
}

/// Lane-wise maximum of values below 2^63.
inline Vec larger(Vec a, Vec b) noexcept {
#if CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 8
    return (Vec)_mm512_max_epi64((__m512i)a, (__m512i)b);
#elif CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 4
    return (Vec)_mm256_max_epi64((__m256i)a, (__m256i)b);
#else
    return (Vec)((SignedVec)a > (SignedVec)b ? (SignedVec)a : (SignedVec)b);
#endif
}

#if CUDDL_CPU_SKETCH_VECTOR_HASH
/// @p x times the constant @p C modulo 2^64.
template <uint64_t C>
inline Vec mul(Vec x) noexcept {
#if CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 8
    return (Vec)_mm512_mullo_epi64((__m512i)x, _mm512_set1_epi64(static_cast<long long>(C)));
#elif CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 4
    return (Vec)_mm256_mullo_epi64((__m256i)x, _mm256_set1_epi64x(static_cast<long long>(C)));
#else
    auto const xv = (__m256i)x;
    auto const low = _mm256_mul_epu32(xv, _mm256_set1_epi64x(static_cast<long long>(C & 0xFFFFFFFFU)));
    auto const cross = _mm256_add_epi64(
        _mm256_mul_epu32(_mm256_srli_epi64(xv, 32), _mm256_set1_epi64x(static_cast<long long>(C & 0xFFFFFFFFU))),
        _mm256_mul_epu32(xv, _mm256_set1_epi64x(static_cast<long long>(C >> 32)))
    );
    return (Vec)_mm256_add_epi64(low, _mm256_slli_epi64(cross, 32));
#endif
}

/// What @ref mark_hits compares hashes with: AVX-512 compares whole hashes with the largest one a
/// ceiling lets through, AVX2 has only signed compares and takes the upper halves.
inline Vec hit_limit(uint32_t ceiling) noexcept {
#if CUDDL_CPU_SKETCH_AVX512
    return splat(uint64_t{ceiling} << 32 | 0xFFFFFFFFU);
#else
    return splat(uint64_t{ceiling} + 1U);
#endif
}

/// @p marks with @p bit set in every lane whose @p hash is within the ceiling @p limit stands for.
inline Vec mark_hits(Vec marks, Vec hash, Vec limit, uint64_t bit) noexcept {
#if CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 8
    return (Vec)_mm512_mask_or_epi64(
        (__m512i)marks,
        _mm512_cmple_epu64_mask((__m512i)hash, (__m512i)limit),
        (__m512i)marks,
        _mm512_set1_epi64(static_cast<long long>(bit))
    );
#elif CUDDL_CPU_SKETCH_AVX512 && CUDDL_CPU_SKETCH_LANES == 4
    return (Vec)_mm256_mask_or_epi64(
        (__m256i)marks,
        _mm256_cmple_epu64_mask((__m256i)hash, (__m256i)limit),
        (__m256i)marks,
        _mm256_set1_epi64x(static_cast<long long>(bit))
    );
#else
    return marks | ((Vec)((SignedVec)(hash >> 32) < (SignedVec)limit) & bit);
#endif
}

/// `hash_kmer(canonical)` in every lane. The seed is below 2^30, so it commutes with the first
/// xor-shift.
inline Vec hash_full(Vec canonical) noexcept {
    static_assert(cuddl::detail::seed < (uint64_t{1} << 30));
    auto x = mul<splitmix::kFirst>(canonical ^ cuddl::detail::seed ^ (canonical >> 30));
    x ^= x >> 27;
    x = mul<splitmix::kSecond>(x);
    return x ^ (x >> 31);
}
#endif

/// Appends base I of @p word, with @p flipped its codes ^ 2, to every lane's k-mers. forward holds
/// the first base in its highest bits, reverse the complement of the first base in its lowest.
template <uint32_t I>
inline void roll_lanes(Vec& forward, Vec& reverse, Vec word, Vec flipped) noexcept {
    constexpr uint32_t top = 2 * (kKmer - 1U);
    forward = ((forward << 2) | ((word >> (2 * I)) & 3U)) & ((uint64_t{1} << (2 * kKmer)) - 1U);
    reverse = (reverse >> 2) | ((flipped << (top - 2 * I)) & (uint64_t{3} << top));
}

/// Rolls bases [I, End) of @p word into the k-mers without hashing them.
template <uint32_t I, uint32_t End>
inline void prime_lanes(Vec& forward, Vec& reverse, Vec word, Vec flipped) noexcept {
    if constexpr (I < End) {
        roll_lanes<I>(forward, reverse, word, flipped);
        prime_lanes<I + 1, End>(forward, reverse, word, flipped);
    }
}

#if CUDDL_CPU_SKETCH_VECTOR_HASH
/// Rolls the sixteen bases of @p word into the k-mers, keeps each step's hashes in @p hashes and
/// sets bit I in @p marks for every lane whose step I hash is within @p limit.
template <uint32_t I>
inline void hash_lanes(
    Vec& forward,
    Vec& reverse,
    Vec word,
    Vec flipped,
    Vec& marks,
    Vec limit,
    uint64_t (*hashes)[kLanes]
) noexcept {
    if constexpr (I < 16U) {
        roll_lanes<I>(forward, reverse, word, flipped);
        Vec const hash = hash_full(larger(forward, reverse));
        store(hash, hashes[I]);
        marks = mark_hits(marks, hash, limit, uint64_t{1} << I);
        hash_lanes<I + 1>(forward, reverse, word, flipped, marks, limit, hashes);
    }
}
#else
/// Rolls the sixteen bases of @p word into the k-mers and keeps each step's canonical k-mers in
/// @p canonicals. @p offer takes the previous step's, so that scalar hashing and vector rolling
/// interleave in program order.
template <uint32_t I, typename Offer>
inline void canonical_lanes(
    Vec& forward, Vec& reverse, Vec word, Vec flipped, uint64_t (*canonicals)[kLanes], Offer&& offer
) noexcept {
    if constexpr (I < 16U) {
        roll_lanes<I>(forward, reverse, word, flipped);
        // The seed is xored here rather than in the scalar hash, one vector op for kLanes k-mers.
        store(larger(forward, reverse) ^ cuddl::detail::seed, canonicals[I]);
        if constexpr (I > 0U) {
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                offer(canonicals[I - 1U][lane]);
            }
        }
        canonical_lanes<I + 1>(forward, reverse, word, flipped, canonicals, offer);
    }
}
#endif
#endif

/// Running state of one genome's sketch: its registers and the pruning ceiling.
///
/// A hash whose upper 32 bits exceed those of the restored minimum register is larger than
/// every score that register holds, so it can raise no register. The minimum is rescanned each
/// time an eighth more windows have been hashed (at most every 2^18 and at least every 512), which
/// is what the GPU's global floor does per tile. Pruning never drops a hash that could raise a
/// register, so the registers do not depend on when the ceiling is rescanned or which hashes are
/// tested against it. The vector path therefore hashes the windows of a run in any order.
struct GenomeSketch {
    uint16_t* registers;
    uint32_t ceiling = UINT32_MAX;
    uint64_t hashed = 0;
    uint64_t next_refresh = 4096;

    /// Windows the scalar prefix of a run hashes while the registers fill, before the lanes start.
    static constexpr uint64_t kWarmup = uint64_t{1} << 15;

    void refresh() noexcept {
        uint16_t minimum = UINT16_MAX;
        for (uint32_t b = 0; b < kBuckets; ++b) {
            minimum = std::min(minimum, registers[b]);
        }
        ceiling =
            minimum == 0U ? UINT32_MAX : static_cast<uint32_t>(cuddl::detail::restore<Layout>(minimum) >> 32);
        next_refresh = hashed + std::clamp<uint64_t>(hashed / 8U, 512U, uint64_t{1} << 18);
    }

    /// Adds every canonical k-mer of one run of @p bases >= k bases that the library packed into
    /// two-bit words, base j in bits 2(j % 16) of word j / 16 with codes A 0, C 1, T 2, G 3.
    /// The run holds no ambiguous base, so every window is a k-mer.
    void add_run(uint32_t const* words, uint64_t bases) noexcept {
        uint64_t const windows = bases - (kKmer - 1U);
#if CUDDL_CPU_SKETCH_LANES
        // The scalar prefix fills the registers so that the ceiling prunes the lanes' hashes, and
        // ends where the lane chunks begin eight bases into a word.
        uint64_t const prefix = (ceiling == UINT32_MAX ? kWarmup : 0U) + 8U;
        uint64_t const chunk = windows > prefix ? (windows - prefix) / (16U * kLanes) * 16U : 0U;
        if (chunk != 0U) {
            add_windows(words, 0, prefix);
            add_lanes(words, prefix, chunk);
            add_windows(words, prefix + kLanes * chunk, windows);
            return;
        }
#endif
        add_windows(words, 0, windows);
    }

    /// The scalar reference: windows [@p first, @p end) of a run, one at a time.
    void add_windows(uint32_t const* words, uint64_t first, uint64_t end) noexcept {
        uint64_t forward = 0, reverse = 0;
        for (uint64_t base = first; base + 1U < first + kKmer; ++base) {
            roll(forward, reverse, (words[base >> 4] >> (2 * (base & 15U))) & 3U);
        }
        for (uint64_t window = first; window < end;) {
            // The window ends at base window + k - 1.
            uint64_t const last = window + kKmer - 1U;
            uint32_t const word = words[last >> 4] >> (2 * (last & 15U));
            auto const count = std::min<uint64_t>(16U - (last & 15U), end - window);
            consume(forward, reverse, word, count);
            window += count;
            hashed += count;
            if (hashed >= next_refresh) {
                refresh();
            }
        }
    }

#if CUDDL_CPU_SKETCH_LANES
    /// Windows [@p first, @p first + kLanes * @p chunk) in kLanes lockstep stretches of @p chunk
    /// windows each, with @p first % 16 == 8 and @p chunk % 16 == 0. A lane rolls the k - 1 bases
    /// before its first window, which end in the word boundary 16 bases before its first hashed
    /// base, and then one word of sixteen bases per step group.
    void add_lanes(uint32_t const* words, uint64_t first, uint64_t chunk) noexcept {
        uint32_t const stride = static_cast<uint32_t>(chunk / 16U);
        uint32_t const* const origin = words + (first - 8U) / 16U;
        constexpr uint64_t flip = 0xAAAAAAAAU;
        Vec forward = splat(0), reverse = splat(0);
        {
            Vec const word = load_words(origin, stride, 0) >> 16;
            prime_lanes<0, 8>(forward, reverse, word, word ^ flip);
        }
        {
            Vec const word = load_words(origin, stride, 1);
            prime_lanes<0, 16>(forward, reverse, word, word ^ flip);
        }
        alignas(64) uint64_t lane_values[16][kLanes];
#if CUDDL_CPU_SKETCH_VECTOR_HASH
        alignas(64) uint64_t lane_hits[kLanes];
        Vec limit = hit_limit(ceiling);
#endif
        for (uint32_t t = 2; t < stride + 2U; ++t) {
            Vec const word = load_words(origin, stride, t);
            hashed += 16U * kLanes;
#if CUDDL_CPU_SKETCH_VECTOR_HASH
            // Bit i of a lane's mark is set when its step i hash can reach the registers.
            Vec marks = splat(0);
            hash_lanes<0>(forward, reverse, word, word ^ flip, marks, limit, lane_values);
            store(marks, lane_hits);
            uint32_t hit_lanes = 0;
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                hit_lanes |= uint32_t{lane_hits[lane] != 0U} << lane;
            }
            for (; hit_lanes != 0U; hit_lanes &= hit_lanes - 1U) {
                auto const lane = static_cast<uint32_t>(std::countr_zero(hit_lanes));
                for (uint64_t bits = lane_hits[lane]; bits != 0U; bits &= bits - 1U) {
                    offer(lane_values[std::countr_zero(bits)][lane]);
                }
            }
            if (hashed >= next_refresh) {
                refresh();
                limit = hit_limit(ceiling);
            }
#else
            canonical_lanes<0>(forward, reverse, word, word ^ flip, lane_values, [&](uint64_t seeded) {
                offer_hash(cuddl::detail::splitmix64(seeded));
            });
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                offer_hash(cuddl::detail::splitmix64(lane_values[15][lane]));
            }
            if (hashed >= next_refresh) {
                refresh();
            }
#endif
        }
    }
#endif

    /// Appends one base of @p code to the forward and reverse-complement k-mers.
    static void roll(uint64_t& forward, uint64_t& reverse, uint32_t code) noexcept {
        constexpr uint64_t mask = (uint64_t{1} << (2 * kKmer)) - 1U;
        constexpr uint32_t top = 2 * (kKmer - 1U);
        // The forward k-mer holds its first base in the highest bits, the reverse
        // complement (codes ^ 2) its first base in the lowest.
        forward = ((forward << 2) | code) & mask;
        reverse = (reverse >> 2) | (uint64_t{code ^ 2U} << top);
    }

    /// Rolls @p count bases of @p word into the k-mers and offers each canonical k-mer.
    void consume(uint64_t& forward, uint64_t& reverse, uint32_t word, uint64_t count) noexcept {
        for (uint64_t i = 0; i < count; ++i, word >>= 2) {
            roll(forward, reverse, word & 3U);
            offer_canonical(forward > reverse ? forward : reverse);
        }
    }

    void offer_canonical(uint64_t canonical) noexcept {
        offer_hash(cuddl::detail::hash_kmer(canonical));
    }

    void offer_hash(uint64_t hash) noexcept {
        if (hash <= (uint64_t{ceiling} << 32 | 0xFFFFFFFFU)) {
            offer(hash);
        }
    }

    void offer(uint64_t hash) noexcept {
        auto& stored = registers[cuddl::detail::bucket_of<kBuckets>(hash)];
        stored = std::max(stored, cuddl::detail::score<Layout>(hash));
    }
};

/// Per-thread scratch for the loader: the inflated genome is packed in place and only its
/// words reach the buffer this returns.
inline cuddl::detail::decompression_target thread_buffer(void*, size_t bytes) {
    thread_local std::unique_ptr<char[]> buffer;
    thread_local size_t capacity = 0;
    if (capacity < bytes) {
        capacity = std::bit_ceil(bytes);
        buffer = std::make_unique_for_overwrite<char[]>(capacity);
    }
    return {buffer.get(), capacity, {}};
}

/// Loads and sketches every genome of @p paths with @p workers threads, one genome per worker
/// at a time in list order. @p rows receives one row of kBuckets uint16 scores per genome.
inline bool ingest_genomes(
    std::vector<std::filesystem::path> const& paths,
    unsigned workers,
    std::vector<uint16_t>& rows,
    IngestStats& stats
) {
    auto const start = std::chrono::steady_clock::now();
    rows.assign(paths.size() * kBuckets, 0U);
    stats.workers = static_cast<unsigned>(std::max<size_t>(1U, std::min<size_t>(workers, paths.size())));
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> bases{0}, kmers{0}, input_bytes{0};
    std::atomic<bool> failed{false};
    std::mutex failure_mutex;
    std::string failure;
    auto const fail = [&](std::string const& message) {
        std::lock_guard lock(failure_mutex);
        if (!failed.exchange(true)) {
            failure = message;
        }
    };
    auto const work = [&] {
        cuddl::detail::decompression_source const source{.acquire = thread_buffer};
        for (size_t id = next++; id < paths.size() && !failed; id = next++) {
            try {
                auto const path = paths[id].string();
                std::error_code size_error;
                input_bytes += std::filesystem::file_size(paths[id], size_error);
                auto loaded = cuddl::detail::load_fastx_sequence_file(path, source, false, true, kKmer);
                if (!loaded) {
                    fail(loaded.error().message());
                    return;
                }
                auto& file = **loaded;
                cuddl::detail::pack_fastx_sequence_file(file, kKmer);
                GenomeSketch sketch{rows.data() + id * kBuckets};
                for (auto const& run : file.packed) {
                    sketch.add_run(file.words + run.word, run.bases);
                    bases += run.bases;
                    kmers += run.bases - kKmer + 1U;
                }
            } catch (std::exception const& error) {
                fail(std::string(error.what()) + ": " + paths[id].string());
                return;
            }
        }
    };
    {
        std::vector<std::jthread> threads;
        for (unsigned t = 0; t < stats.workers; ++t) {
            threads.emplace_back(work);
        }
    }
    stats.ingest_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    stats.bases = bases;
    stats.kmers = kmers;
    stats.input_bytes = input_bytes;
    if (failed) {
        std::fprintf(stderr, "ingest failed: %s\n", failure.c_str());
        return false;
    }
    return true;
}

}  // namespace cpu_ingest
