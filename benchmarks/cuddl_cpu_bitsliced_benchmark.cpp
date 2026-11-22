// Standalone CPU bit-sliced four-way DDL all-to-all comparator (AVX-512 vpternlogd, ARM SVE2 BSL,
// AVX2). Input is a GPU-built .cuddl (--db) or, with --genomes, genomes ingested on the CPU alone
// (read, gunzip, parse, 2-bit pack, sketch). Build from the repository root with
//   g++ -O3 -march=native -fopenmp -std=c++20 -pthread -DCUDDL_ABLATIONS=0 -Iinclude
//       -Isubprojects/CLI11-2.7.2/include -Isubprojects/libdeflate
//       -Isubprojects/cccl/libcudacxx/include -I"$(dirname "$(dirname "$(which nvcc)")")/include"
//       benchmarks/cuddl_cpu_bitsliced_benchmark.cpp build/subprojects/libdeflate/liblibdeflate.a
//       -o cuddl-cpu-bitsliced
// The CUDA and CCCL include paths only supply headers the library includes; the binary links no
// CUDA library and never touches a GPU.
// --deltas ports the GPU's near-duplicate references: rows within 160 changed buckets of a
// higher-ID base compare only against that base, and the child's counts come from the base's counts
// and its changed buckets. The grouping time (deltas_build_ms) counts toward total_ms and
// end_to_end_ms.
// --index (format edges) finds the pairs that can reach --min-equal through an inverted index over
// (bucket, score) cells instead of the 12-plane pre-filter and refines them with the exact
// comparison; the index build time (index_build_ms) counts toward total_ms and end_to_end_ms.
// Sparse and edges output with --o-direct runs as a stripe pipeline (cuddl_cpu_pipeline.hpp): the
// encode of one stripe overlaps the comparison of the next, so the output costs little more than
// the slower of the comparison and the drive. --profile-stripes prints where the threads' time went.

#include <CLI/CLI.hpp>

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SVE)
    #include <arm_sve.h>
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#include "cuddl_cpu_deltas.hpp"
#include "cuddl_cpu_index.hpp"
#include "cuddl_cpu_ingest.hpp"
#include "cuddl_cpu_pipeline.hpp"

namespace {

constexpr uint32_t kBuckets = 2048U;
constexpr uint32_t kPlanes = 16U;
constexpr uint32_t kWords64 = kBuckets / 64U;  // 32 uint64_t per plane
constexpr uint32_t kFilterPlanes = 12U;        // the GPU match filter's low planes
constexpr uint32_t kPeakChains = 12U;          // independent chains in --logic-peak

struct alignas(64) BitSlicedSketch {
    uint64_t planes[kPlanes][kWords64];
    uint64_t nonzero[kWords64];
    bool all_full;
};

struct Counts {
    uint32_t lower;
    uint32_t equal;
    uint32_t higher;
    uint32_t empty;
};

inline float wkid_from_counts(uint32_t lower, uint32_t equal, uint32_t higher) noexcept {
    uint32_t const denom = std::max(equal + std::min(lower, higher), 6U);
    return static_cast<float>(equal) / static_cast<float>(denom);
}

Counts scalar_compare(uint16_t const* q, uint16_t const* r) noexcept {
    Counts c{0, 0, 0, 0};
    for (uint32_t b = 0; b < kBuckets; ++b) {
        if (q[b] == 0U && r[b] == 0U) {
            ++c.empty;
        } else if (q[b] < r[b]) {
            ++c.lower;
        } else if (q[b] == r[b]) {
            ++c.equal;
        } else {
            ++c.higher;
        }
    }
    return c;
}

void transpose_row(uint16_t const* row, BitSlicedSketch& out) noexcept {
    std::memset(out.planes, 0, sizeof(out.planes));
    std::memset(out.nonzero, 0, sizeof(out.nonzero));
    bool full = true;
    for (uint32_t b = 0; b < kBuckets; ++b) {
        uint16_t const s = row[b];
        uint32_t const w = b >> 6U;
        uint64_t const bit = uint64_t{1} << (b & 63U);
        if (s != 0U) {
            out.nonzero[w] |= bit;
            for (uint32_t p = 0; p < kPlanes; ++p) {
                if ((s >> p) & 1U) {
                    out.planes[p][w] |= bit;
                }
            }
        } else {
            full = false;
        }
    }
    out.all_full = full;
}

// Upper bound on E from the low planes, as in the GPU match filter: a bucket that differs in any
// low plane cannot be equal.
inline uint32_t low_plane_bound(BitSlicedSketch const& q, BitSlicedSketch const& r) noexcept {
    uint64_t ne[kWords64] = {};
    for (uint32_t p = 0; p < kFilterPlanes; ++p) {
        for (uint32_t w = 0; w < kWords64; ++w) ne[w] |= q.planes[p][w] ^ r.planes[p][w];
    }
    uint32_t diff = 0;
    for (uint32_t w = 0; w < kWords64; ++w) diff += static_cast<uint32_t>(std::popcount(ne[w]));
    return kBuckets - diff;
}

#if defined(__AVX512F__) && defined(__AVX512VPOPCNTDQ__)
constexpr char const* kIsaName = "avx512-vpternlogd-vpopcnt";

inline uint32_t hsum_epi64(__m512i v) noexcept {
    alignas(64) uint64_t tmp[8];
    _mm512_store_si512(tmp, v);
    return static_cast<uint32_t>(
        tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
    );
}

// Compare 1 query against 4 references simultaneously to expose 4x ILP across the 16-deep lt chain.
inline void compare_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out[4]
) noexcept {
    bool const all_full = q.all_full || (r0.all_full && r1.all_full && r2.all_full && r3.all_full);
    __m512i sum_lt0 = _mm512_setzero_si512(), sum_ne0 = _mm512_setzero_si512();
    __m512i sum_lt1 = _mm512_setzero_si512(), sum_ne1 = _mm512_setzero_si512();
    __m512i sum_lt2 = _mm512_setzero_si512(), sum_ne2 = _mm512_setzero_si512();
    __m512i sum_lt3 = _mm512_setzero_si512(), sum_ne3 = _mm512_setzero_si512();
    __m512i sum_occ0 = _mm512_setzero_si512(), sum_occ1 = _mm512_setzero_si512();
    __m512i sum_occ2 = _mm512_setzero_si512(), sum_occ3 = _mm512_setzero_si512();

    for (uint32_t g = 0; g < 4U; ++g) {
        size_t const off = g * 8U;
        // Plane 0 initializes lt = ~q & r and ne = q ^ r directly.
        __m512i qp = _mm512_load_si512(&q.planes[0][off]);
        __m512i rp0 = _mm512_load_si512(&r0.planes[0][off]);
        __m512i rp1 = _mm512_load_si512(&r1.planes[0][off]);
        __m512i rp2 = _mm512_load_si512(&r2.planes[0][off]);
        __m512i rp3 = _mm512_load_si512(&r3.planes[0][off]);

        __m512i lt0 = _mm512_andnot_si512(qp, rp0);
        __m512i ne0 = _mm512_xor_si512(qp, rp0);
        __m512i lt1 = _mm512_andnot_si512(qp, rp1);
        __m512i ne1 = _mm512_xor_si512(qp, rp1);
        __m512i lt2 = _mm512_andnot_si512(qp, rp2);
        __m512i ne2 = _mm512_xor_si512(qp, rp2);
        __m512i lt3 = _mm512_andnot_si512(qp, rp3);
        __m512i ne3 = _mm512_xor_si512(qp, rp3);

        // Intel vpternlogd(A, B, C, imm8):
        //   0x8E = (~A & B) | (~(A ^ B) & C)
        //   0xBE = C | (A ^ B)
        for (uint32_t p = 1; p < kPlanes; ++p) {
            qp = _mm512_load_si512(&q.planes[p][off]);
            rp0 = _mm512_load_si512(&r0.planes[p][off]);
            rp1 = _mm512_load_si512(&r1.planes[p][off]);
            rp2 = _mm512_load_si512(&r2.planes[p][off]);
            rp3 = _mm512_load_si512(&r3.planes[p][off]);

            lt0 = _mm512_ternarylogic_epi32(qp, rp0, lt0, 0x8E);
            ne0 = _mm512_ternarylogic_epi32(qp, rp0, ne0, 0xBE);
            lt1 = _mm512_ternarylogic_epi32(qp, rp1, lt1, 0x8E);
            ne1 = _mm512_ternarylogic_epi32(qp, rp1, ne1, 0xBE);
            lt2 = _mm512_ternarylogic_epi32(qp, rp2, lt2, 0x8E);
            ne2 = _mm512_ternarylogic_epi32(qp, rp2, ne2, 0xBE);
            lt3 = _mm512_ternarylogic_epi32(qp, rp3, lt3, 0x8E);
            ne3 = _mm512_ternarylogic_epi32(qp, rp3, ne3, 0xBE);
        }
        sum_lt0 = _mm512_add_epi64(sum_lt0, _mm512_popcnt_epi64(lt0));
        sum_ne0 = _mm512_add_epi64(sum_ne0, _mm512_popcnt_epi64(ne0));
        sum_lt1 = _mm512_add_epi64(sum_lt1, _mm512_popcnt_epi64(lt1));
        sum_ne1 = _mm512_add_epi64(sum_ne1, _mm512_popcnt_epi64(ne1));
        sum_lt2 = _mm512_add_epi64(sum_lt2, _mm512_popcnt_epi64(lt2));
        sum_ne2 = _mm512_add_epi64(sum_ne2, _mm512_popcnt_epi64(ne2));
        sum_lt3 = _mm512_add_epi64(sum_lt3, _mm512_popcnt_epi64(lt3));
        sum_ne3 = _mm512_add_epi64(sum_ne3, _mm512_popcnt_epi64(ne3));
        if (!all_full) {
            __m512i qnz = _mm512_load_si512(&q.nonzero[off]);
            sum_occ0 = _mm512_add_epi64(
                sum_occ0,
                _mm512_popcnt_epi64(_mm512_or_si512(qnz, _mm512_load_si512(&r0.nonzero[off])))
            );
            sum_occ1 = _mm512_add_epi64(
                sum_occ1,
                _mm512_popcnt_epi64(_mm512_or_si512(qnz, _mm512_load_si512(&r1.nonzero[off])))
            );
            sum_occ2 = _mm512_add_epi64(
                sum_occ2,
                _mm512_popcnt_epi64(_mm512_or_si512(qnz, _mm512_load_si512(&r2.nonzero[off])))
            );
            sum_occ3 = _mm512_add_epi64(
                sum_occ3,
                _mm512_popcnt_epi64(_mm512_or_si512(qnz, _mm512_load_si512(&r3.nonzero[off])))
            );
        }
    }

    uint32_t const lts[4] = {
        hsum_epi64(sum_lt0), hsum_epi64(sum_lt1), hsum_epi64(sum_lt2), hsum_epi64(sum_lt3)
    };
    uint32_t const nes[4] = {
        hsum_epi64(sum_ne0), hsum_epi64(sum_ne1), hsum_epi64(sum_ne2), hsum_epi64(sum_ne3)
    };
    uint32_t const occs[4] = {
        all_full ? kBuckets : hsum_epi64(sum_occ0),
        all_full ? kBuckets : hsum_epi64(sum_occ1),
        all_full ? kBuckets : hsum_epi64(sum_occ2),
        all_full ? kBuckets : hsum_epi64(sum_occ3),
    };
    for (int k = 0; k < 4; ++k) {
        out[k] = {lts[k], occs[k] - nes[k], nes[k] - lts[k], kBuckets - occs[k]};
    }
}

// Low-plane E bounds of 1 query against 4 references: ne |= q ^ r is one vpternlogd per plane.
inline void low_plane_bound_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    uint32_t out[4]
) noexcept {
    __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
    __m512i s2 = _mm512_setzero_si512(), s3 = _mm512_setzero_si512();
    for (uint32_t g = 0; g < 4U; ++g) {
        size_t const off = g * 8U;
        __m512i ne0 = _mm512_setzero_si512(), ne1 = _mm512_setzero_si512();
        __m512i ne2 = _mm512_setzero_si512(), ne3 = _mm512_setzero_si512();
        for (uint32_t p = 0; p < kFilterPlanes; ++p) {
            __m512i const qp = _mm512_load_si512(&q.planes[p][off]);
            ne0 = _mm512_ternarylogic_epi32(qp, _mm512_load_si512(&r0.planes[p][off]), ne0, 0xBE);
            ne1 = _mm512_ternarylogic_epi32(qp, _mm512_load_si512(&r1.planes[p][off]), ne1, 0xBE);
            ne2 = _mm512_ternarylogic_epi32(qp, _mm512_load_si512(&r2.planes[p][off]), ne2, 0xBE);
            ne3 = _mm512_ternarylogic_epi32(qp, _mm512_load_si512(&r3.planes[p][off]), ne3, 0xBE);
        }
        s0 = _mm512_add_epi64(s0, _mm512_popcnt_epi64(ne0));
        s1 = _mm512_add_epi64(s1, _mm512_popcnt_epi64(ne1));
        s2 = _mm512_add_epi64(s2, _mm512_popcnt_epi64(ne2));
        s3 = _mm512_add_epi64(s3, _mm512_popcnt_epi64(ne3));
    }
    out[0] = kBuckets - hsum_epi64(s0);
    out[1] = kBuckets - hsum_epi64(s1);
    out[2] = kBuckets - hsum_epi64(s2);
    out[3] = kBuckets - hsum_epi64(s3);
}

// Exhaustive comparison runs 2 + 15 * 2 logic ops per 512-bit chunk of each pair.
inline uint32_t logic_ops_per_pair() noexcept {
    return 4U * (2U + 15U * 2U);
}

// Twelve independent vpternlogd chains in named registers measure the cores' logic-op throughput;
// the empty asm keeps the compiler from folding the chains or moving them to memory.
inline uint64_t logic_peak(uint64_t iters) noexcept {
    __m512i x = _mm512_set1_epi64(0x5555555555555555LL);
    __m512i y = _mm512_set1_epi64(0x3333333333333333LL);
    asm volatile("" : "+v"(x), "+v"(y));
    __m512i a0 = _mm512_set1_epi64(0), a1 = _mm512_set1_epi64(1), a2 = _mm512_set1_epi64(2);
    __m512i a3 = _mm512_set1_epi64(3), a4 = _mm512_set1_epi64(4), a5 = _mm512_set1_epi64(5);
    __m512i a6 = _mm512_set1_epi64(6), a7 = _mm512_set1_epi64(7), a8 = _mm512_set1_epi64(8);
    __m512i a9 = _mm512_set1_epi64(9), a10 = _mm512_set1_epi64(10), a11 = _mm512_set1_epi64(11);
    // The chain is vpternlogd's destination operand, so no register copies enter the loop.
    auto step = [&](__m512i& a) {
        a = _mm512_ternarylogic_epi32(a, x, y, 0x96);
        asm volatile("" : "+v"(a));
    };
    for (uint64_t i = 0; i < iters; ++i) {
        step(a0), step(a1), step(a2), step(a3), step(a4), step(a5);
        step(a6), step(a7), step(a8), step(a9), step(a10), step(a11);
    }
    return hsum_epi64(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11);
}

// Compare 2 queries against 4 references (8 pairs) to cut loads/pair from 1.25 to 0.75 and double
// ILP.
inline void compare_2x4(
    BitSlicedSketch const& q0,
    BitSlicedSketch const& q1,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out0[4],
    Counts out1[4]
) noexcept {
    bool const all_full =
        (q0.all_full && q1.all_full) || (r0.all_full && r1.all_full && r2.all_full && r3.all_full);
    if (!all_full) {
        compare_1x4(q0, r0, r1, r2, r3, out0);
        compare_1x4(q1, r0, r1, r2, r3, out1);
        return;
    }
    __m512i sl0 = _mm512_setzero_si512(), sn0 = _mm512_setzero_si512();
    __m512i sl1 = _mm512_setzero_si512(), sn1 = _mm512_setzero_si512();
    __m512i sl2 = _mm512_setzero_si512(), sn2 = _mm512_setzero_si512();
    __m512i sl3 = _mm512_setzero_si512(), sn3 = _mm512_setzero_si512();
    __m512i sl4 = _mm512_setzero_si512(), sn4 = _mm512_setzero_si512();
    __m512i sl5 = _mm512_setzero_si512(), sn5 = _mm512_setzero_si512();
    __m512i sl6 = _mm512_setzero_si512(), sn6 = _mm512_setzero_si512();
    __m512i sl7 = _mm512_setzero_si512(), sn7 = _mm512_setzero_si512();

    for (uint32_t g = 0; g < 4U; ++g) {
        size_t const off = g * 8U;
        __m512i qa = _mm512_load_si512(&q0.planes[0][off]);
        __m512i qb = _mm512_load_si512(&q1.planes[0][off]);
        __m512i rp0 = _mm512_load_si512(&r0.planes[0][off]);
        __m512i rp1 = _mm512_load_si512(&r1.planes[0][off]);
        __m512i rp2 = _mm512_load_si512(&r2.planes[0][off]);
        __m512i rp3 = _mm512_load_si512(&r3.planes[0][off]);

        __m512i lt0 = _mm512_andnot_si512(qa, rp0), ne0 = _mm512_xor_si512(qa, rp0);
        __m512i lt1 = _mm512_andnot_si512(qa, rp1), ne1 = _mm512_xor_si512(qa, rp1);
        __m512i lt2 = _mm512_andnot_si512(qa, rp2), ne2 = _mm512_xor_si512(qa, rp2);
        __m512i lt3 = _mm512_andnot_si512(qa, rp3), ne3 = _mm512_xor_si512(qa, rp3);
        __m512i lt4 = _mm512_andnot_si512(qb, rp0), ne4 = _mm512_xor_si512(qb, rp0);
        __m512i lt5 = _mm512_andnot_si512(qb, rp1), ne5 = _mm512_xor_si512(qb, rp1);
        __m512i lt6 = _mm512_andnot_si512(qb, rp2), ne6 = _mm512_xor_si512(qb, rp2);
        __m512i lt7 = _mm512_andnot_si512(qb, rp3), ne7 = _mm512_xor_si512(qb, rp3);

        for (uint32_t p = 1; p < kPlanes; ++p) {
            qa = _mm512_load_si512(&q0.planes[p][off]);
            qb = _mm512_load_si512(&q1.planes[p][off]);
            rp0 = _mm512_load_si512(&r0.planes[p][off]);
            rp1 = _mm512_load_si512(&r1.planes[p][off]);
            rp2 = _mm512_load_si512(&r2.planes[p][off]);
            rp3 = _mm512_load_si512(&r3.planes[p][off]);

            lt0 = _mm512_ternarylogic_epi32(qa, rp0, lt0, 0x8E);
            ne0 = _mm512_ternarylogic_epi32(qa, rp0, ne0, 0xBE);
            lt1 = _mm512_ternarylogic_epi32(qa, rp1, lt1, 0x8E);
            ne1 = _mm512_ternarylogic_epi32(qa, rp1, ne1, 0xBE);
            lt2 = _mm512_ternarylogic_epi32(qa, rp2, lt2, 0x8E);
            ne2 = _mm512_ternarylogic_epi32(qa, rp2, ne2, 0xBE);
            lt3 = _mm512_ternarylogic_epi32(qa, rp3, lt3, 0x8E);
            ne3 = _mm512_ternarylogic_epi32(qa, rp3, ne3, 0xBE);

            lt4 = _mm512_ternarylogic_epi32(qb, rp0, lt4, 0x8E);
            ne4 = _mm512_ternarylogic_epi32(qb, rp0, ne4, 0xBE);
            lt5 = _mm512_ternarylogic_epi32(qb, rp1, lt5, 0x8E);
            ne5 = _mm512_ternarylogic_epi32(qb, rp1, ne5, 0xBE);
            lt6 = _mm512_ternarylogic_epi32(qb, rp2, lt6, 0x8E);
            ne6 = _mm512_ternarylogic_epi32(qb, rp2, ne6, 0xBE);
            lt7 = _mm512_ternarylogic_epi32(qb, rp3, lt7, 0x8E);
            ne7 = _mm512_ternarylogic_epi32(qb, rp3, ne7, 0xBE);
        }
        sl0 = _mm512_add_epi64(sl0, _mm512_popcnt_epi64(lt0));
        sn0 = _mm512_add_epi64(sn0, _mm512_popcnt_epi64(ne0));
        sl1 = _mm512_add_epi64(sl1, _mm512_popcnt_epi64(lt1));
        sn1 = _mm512_add_epi64(sn1, _mm512_popcnt_epi64(ne1));
        sl2 = _mm512_add_epi64(sl2, _mm512_popcnt_epi64(lt2));
        sn2 = _mm512_add_epi64(sn2, _mm512_popcnt_epi64(ne2));
        sl3 = _mm512_add_epi64(sl3, _mm512_popcnt_epi64(lt3));
        sn3 = _mm512_add_epi64(sn3, _mm512_popcnt_epi64(ne3));
        sl4 = _mm512_add_epi64(sl4, _mm512_popcnt_epi64(lt4));
        sn4 = _mm512_add_epi64(sn4, _mm512_popcnt_epi64(ne4));
        sl5 = _mm512_add_epi64(sl5, _mm512_popcnt_epi64(lt5));
        sn5 = _mm512_add_epi64(sn5, _mm512_popcnt_epi64(ne5));
        sl6 = _mm512_add_epi64(sl6, _mm512_popcnt_epi64(lt6));
        sn6 = _mm512_add_epi64(sn6, _mm512_popcnt_epi64(ne6));
        sl7 = _mm512_add_epi64(sl7, _mm512_popcnt_epi64(lt7));
        sn7 = _mm512_add_epi64(sn7, _mm512_popcnt_epi64(ne7));
    }
    uint32_t const l0[4] = {hsum_epi64(sl0), hsum_epi64(sl1), hsum_epi64(sl2), hsum_epi64(sl3)};
    uint32_t const n0[4] = {hsum_epi64(sn0), hsum_epi64(sn1), hsum_epi64(sn2), hsum_epi64(sn3)};
    uint32_t const l1[4] = {hsum_epi64(sl4), hsum_epi64(sl5), hsum_epi64(sl6), hsum_epi64(sl7)};
    uint32_t const n1[4] = {hsum_epi64(sn4), hsum_epi64(sn5), hsum_epi64(sn6), hsum_epi64(sn7)};
    for (int k = 0; k < 4; ++k) {
        out0[k] = {l0[k], kBuckets - n0[k], n0[k] - l0[k], 0U};
        out1[k] = {l1[k], kBuckets - n1[k], n1[k] - l1[k], 0U};
    }
}

#elif defined(__ARM_FEATURE_SVE2)
constexpr char const* kIsaName = "arm-sve2-bsl-cnt";

inline void compare_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out[4]
) noexcept {
    bool const all_full = q.all_full || (r0.all_full && r1.all_full && r2.all_full && r3.all_full);
    svbool_t const pg = svptrue_b64();
    uint64_t const vl = svcntd();
    svuint64_t sl0 = svdup_u64(0), sn0 = svdup_u64(0), so0 = svdup_u64(0);
    svuint64_t sl1 = svdup_u64(0), sn1 = svdup_u64(0), so1 = svdup_u64(0);
    svuint64_t sl2 = svdup_u64(0), sn2 = svdup_u64(0), so2 = svdup_u64(0);
    svuint64_t sl3 = svdup_u64(0), sn3 = svdup_u64(0), so3 = svdup_u64(0);

    for (uint64_t off = 0; off < kWords64; off += vl) {
        svuint64_t qp = svld1_u64(pg, &q.planes[0][off]);
        svuint64_t rp0 = svld1_u64(pg, &r0.planes[0][off]);
        svuint64_t rp1 = svld1_u64(pg, &r1.planes[0][off]);
        svuint64_t rp2 = svld1_u64(pg, &r2.planes[0][off]);
        svuint64_t rp3 = svld1_u64(pg, &r3.planes[0][off]);

        svuint64_t lt0 = svbic_u64_x(pg, rp0, qp), ne0 = sveor_u64_x(pg, qp, rp0);
        svuint64_t lt1 = svbic_u64_x(pg, rp1, qp), ne1 = sveor_u64_x(pg, qp, rp1);
        svuint64_t lt2 = svbic_u64_x(pg, rp2, qp), ne2 = sveor_u64_x(pg, qp, rp2);
        svuint64_t lt3 = svbic_u64_x(pg, rp3, qp), ne3 = sveor_u64_x(pg, qp, rp3);

        // SVE2 3-op/plane: t = q ^ r; ne |= t; lt = bsl(t, r, lt) = (r & t) | (lt & ~t)
        for (uint32_t p = 1; p < kPlanes; ++p) {
            qp = svld1_u64(pg, &q.planes[p][off]);
            rp0 = svld1_u64(pg, &r0.planes[p][off]);
            rp1 = svld1_u64(pg, &r1.planes[p][off]);
            rp2 = svld1_u64(pg, &r2.planes[p][off]);
            rp3 = svld1_u64(pg, &r3.planes[p][off]);

            svuint64_t t0 = sveor_u64_x(pg, qp, rp0);
            ne0 = svorr_u64_x(pg, ne0, t0);
            lt0 = svbsl_u64(rp0, lt0, t0);

            svuint64_t t1 = sveor_u64_x(pg, qp, rp1);
            ne1 = svorr_u64_x(pg, ne1, t1);
            lt1 = svbsl_u64(rp1, lt1, t1);

            svuint64_t t2 = sveor_u64_x(pg, qp, rp2);
            ne2 = svorr_u64_x(pg, ne2, t2);
            lt2 = svbsl_u64(rp2, lt2, t2);

            svuint64_t t3 = sveor_u64_x(pg, qp, rp3);
            ne3 = svorr_u64_x(pg, ne3, t3);
            lt3 = svbsl_u64(rp3, lt3, t3);
        }
        sl0 = svadd_u64_x(pg, sl0, svcnt_u64_x(pg, lt0));
        sn0 = svadd_u64_x(pg, sn0, svcnt_u64_x(pg, ne0));
        sl1 = svadd_u64_x(pg, sl1, svcnt_u64_x(pg, lt1));
        sn1 = svadd_u64_x(pg, sn1, svcnt_u64_x(pg, ne1));
        sl2 = svadd_u64_x(pg, sl2, svcnt_u64_x(pg, lt2));
        sn2 = svadd_u64_x(pg, sn2, svcnt_u64_x(pg, ne2));
        sl3 = svadd_u64_x(pg, sl3, svcnt_u64_x(pg, lt3));
        sn3 = svadd_u64_x(pg, sn3, svcnt_u64_x(pg, ne3));
        if (!all_full) {
            svuint64_t qnz = svld1_u64(pg, &q.nonzero[off]);
            so0 = svadd_u64_x(
                pg, so0, svcnt_u64_x(pg, svorr_u64_x(pg, qnz, svld1_u64(pg, &r0.nonzero[off])))
            );
            so1 = svadd_u64_x(
                pg, so1, svcnt_u64_x(pg, svorr_u64_x(pg, qnz, svld1_u64(pg, &r1.nonzero[off])))
            );
            so2 = svadd_u64_x(
                pg, so2, svcnt_u64_x(pg, svorr_u64_x(pg, qnz, svld1_u64(pg, &r2.nonzero[off])))
            );
            so3 = svadd_u64_x(
                pg, so3, svcnt_u64_x(pg, svorr_u64_x(pg, qnz, svld1_u64(pg, &r3.nonzero[off])))
            );
        }
    }
    uint32_t const lts[4] = {
        static_cast<uint32_t>(svaddv_u64(pg, sl0)),
        static_cast<uint32_t>(svaddv_u64(pg, sl1)),
        static_cast<uint32_t>(svaddv_u64(pg, sl2)),
        static_cast<uint32_t>(svaddv_u64(pg, sl3)),
    };
    uint32_t const nes[4] = {
        static_cast<uint32_t>(svaddv_u64(pg, sn0)),
        static_cast<uint32_t>(svaddv_u64(pg, sn1)),
        static_cast<uint32_t>(svaddv_u64(pg, sn2)),
        static_cast<uint32_t>(svaddv_u64(pg, sn3)),
    };
    uint32_t const occs[4] = {
        all_full ? kBuckets : static_cast<uint32_t>(svaddv_u64(pg, so0)),
        all_full ? kBuckets : static_cast<uint32_t>(svaddv_u64(pg, so1)),
        all_full ? kBuckets : static_cast<uint32_t>(svaddv_u64(pg, so2)),
        all_full ? kBuckets : static_cast<uint32_t>(svaddv_u64(pg, so3)),
    };
    for (int k = 0; k < 4; ++k) {
        out[k] = {lts[k], occs[k] - nes[k], nes[k] - lts[k], kBuckets - occs[k]};
    }
}

inline void compare_2x4(
    BitSlicedSketch const& q0,
    BitSlicedSketch const& q1,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out0[4],
    Counts out1[4]
) noexcept {
    bool const all_full =
        (q0.all_full && q1.all_full) || (r0.all_full && r1.all_full && r2.all_full && r3.all_full);
    if (!all_full) {
        compare_1x4(q0, r0, r1, r2, r3, out0);
        compare_1x4(q1, r0, r1, r2, r3, out1);
        return;
    }
    svbool_t const pg = svptrue_b64();
    uint64_t const vl = svcntd();
    uint32_t l0[4] = {0, 0, 0, 0}, n0[4] = {0, 0, 0, 0};
    uint32_t l1[4] = {0, 0, 0, 0}, n1[4] = {0, 0, 0, 0};

    for (uint64_t off = 0; off < kWords64; off += vl) {
        svuint64_t qa = svld1_u64(pg, &q0.planes[0][off]);
        svuint64_t qb = svld1_u64(pg, &q1.planes[0][off]);
        svuint64_t rp0 = svld1_u64(pg, &r0.planes[0][off]);
        svuint64_t rp1 = svld1_u64(pg, &r1.planes[0][off]);
        svuint64_t rp2 = svld1_u64(pg, &r2.planes[0][off]);
        svuint64_t rp3 = svld1_u64(pg, &r3.planes[0][off]);

        svuint64_t lt0 = svbic_u64_x(pg, rp0, qa), ne0 = sveor_u64_x(pg, qa, rp0);
        svuint64_t lt1 = svbic_u64_x(pg, rp1, qa), ne1 = sveor_u64_x(pg, qa, rp1);
        svuint64_t lt2 = svbic_u64_x(pg, rp2, qa), ne2 = sveor_u64_x(pg, qa, rp2);
        svuint64_t lt3 = svbic_u64_x(pg, rp3, qa), ne3 = sveor_u64_x(pg, qa, rp3);
        svuint64_t lt4 = svbic_u64_x(pg, rp0, qb), ne4 = sveor_u64_x(pg, qb, rp0);
        svuint64_t lt5 = svbic_u64_x(pg, rp1, qb), ne5 = sveor_u64_x(pg, qb, rp1);
        svuint64_t lt6 = svbic_u64_x(pg, rp2, qb), ne6 = sveor_u64_x(pg, qb, rp2);
        svuint64_t lt7 = svbic_u64_x(pg, rp3, qb), ne7 = sveor_u64_x(pg, qb, rp3);

        for (uint32_t p = 1; p < kPlanes; ++p) {
            qa = svld1_u64(pg, &q0.planes[p][off]);
            qb = svld1_u64(pg, &q1.planes[p][off]);
            rp0 = svld1_u64(pg, &r0.planes[p][off]);
            rp1 = svld1_u64(pg, &r1.planes[p][off]);
            rp2 = svld1_u64(pg, &r2.planes[p][off]);
            rp3 = svld1_u64(pg, &r3.planes[p][off]);

            svuint64_t t = sveor_u64_x(pg, qa, rp0);
            ne0 = svorr_u64_x(pg, ne0, t);
            lt0 = svbsl_u64(rp0, lt0, t);
            t = sveor_u64_x(pg, qa, rp1);
            ne1 = svorr_u64_x(pg, ne1, t);
            lt1 = svbsl_u64(rp1, lt1, t);
            t = sveor_u64_x(pg, qa, rp2);
            ne2 = svorr_u64_x(pg, ne2, t);
            lt2 = svbsl_u64(rp2, lt2, t);
            t = sveor_u64_x(pg, qa, rp3);
            ne3 = svorr_u64_x(pg, ne3, t);
            lt3 = svbsl_u64(rp3, lt3, t);

            t = sveor_u64_x(pg, qb, rp0);
            ne4 = svorr_u64_x(pg, ne4, t);
            lt4 = svbsl_u64(rp0, lt4, t);
            t = sveor_u64_x(pg, qb, rp1);
            ne5 = svorr_u64_x(pg, ne5, t);
            lt5 = svbsl_u64(rp1, lt5, t);
            t = sveor_u64_x(pg, qb, rp2);
            ne6 = svorr_u64_x(pg, ne6, t);
            lt6 = svbsl_u64(rp2, lt6, t);
            t = sveor_u64_x(pg, qb, rp3);
            ne7 = svorr_u64_x(pg, ne7, t);
            lt7 = svbsl_u64(rp3, lt7, t);
        }
        l0[0] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt0)));
        n0[0] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne0)));
        l0[1] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt1)));
        n0[1] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne1)));
        l0[2] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt2)));
        n0[2] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne2)));
        l0[3] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt3)));
        n0[3] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne3)));
        l1[0] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt4)));
        n1[0] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne4)));
        l1[1] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt5)));
        n1[1] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne5)));
        l1[2] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt6)));
        n1[2] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne6)));
        l1[3] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, lt7)));
        n1[3] += static_cast<uint32_t>(svaddv_u64(pg, svcnt_u64_x(pg, ne7)));
    }
    for (int k = 0; k < 4; ++k) {
        out0[k] = {l0[k], kBuckets - n0[k], n0[k] - l0[k], 0U};
        out1[k] = {l1[k], kBuckets - n1[k], n1[k] - l1[k], 0U};
    }
}

// Low-plane E bounds of 1 query against 4 references: EOR and ORR per plane.
inline void low_plane_bound_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    uint32_t out[4]
) noexcept {
    svbool_t const pg = svptrue_b64();
    uint64_t const vl = svcntd();
    svuint64_t s0 = svdup_u64(0), s1 = svdup_u64(0), s2 = svdup_u64(0), s3 = svdup_u64(0);
    for (uint64_t off = 0; off < kWords64; off += vl) {
        svuint64_t ne0 = svdup_u64(0), ne1 = svdup_u64(0), ne2 = svdup_u64(0), ne3 = svdup_u64(0);
        for (uint32_t p = 0; p < kFilterPlanes; ++p) {
            svuint64_t const qp = svld1_u64(pg, &q.planes[p][off]);
            ne0 = svorr_u64_x(pg, ne0, sveor_u64_x(pg, qp, svld1_u64(pg, &r0.planes[p][off])));
            ne1 = svorr_u64_x(pg, ne1, sveor_u64_x(pg, qp, svld1_u64(pg, &r1.planes[p][off])));
            ne2 = svorr_u64_x(pg, ne2, sveor_u64_x(pg, qp, svld1_u64(pg, &r2.planes[p][off])));
            ne3 = svorr_u64_x(pg, ne3, sveor_u64_x(pg, qp, svld1_u64(pg, &r3.planes[p][off])));
        }
        s0 = svadd_u64_x(pg, s0, svcnt_u64_x(pg, ne0));
        s1 = svadd_u64_x(pg, s1, svcnt_u64_x(pg, ne1));
        s2 = svadd_u64_x(pg, s2, svcnt_u64_x(pg, ne2));
        s3 = svadd_u64_x(pg, s3, svcnt_u64_x(pg, ne3));
    }
    out[0] = kBuckets - static_cast<uint32_t>(svaddv_u64(pg, s0));
    out[1] = kBuckets - static_cast<uint32_t>(svaddv_u64(pg, s1));
    out[2] = kBuckets - static_cast<uint32_t>(svaddv_u64(pg, s2));
    out[3] = kBuckets - static_cast<uint32_t>(svaddv_u64(pg, s3));
}

// Exhaustive comparison runs 2 + 15 * 3 logic ops per vector of each pair.
inline uint32_t logic_ops_per_pair() noexcept {
    return static_cast<uint32_t>(kWords64 / svcntd()) * (2U + 15U * 3U);
}

// Independent EOR chains measure the cores' logic-op throughput; the empty asm keeps the compiler
// from folding the chains.
inline uint64_t logic_peak(uint64_t iters) noexcept {
    svbool_t const pg = svptrue_b64();
    svuint64_t x = svdup_u64(0x5555555555555555ULL);
    asm volatile("" : "+w"(x));
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(1), a2 = svdup_u64(2), a3 = svdup_u64(3);
    svuint64_t a4 = svdup_u64(4), a5 = svdup_u64(5), a6 = svdup_u64(6), a7 = svdup_u64(7);
    svuint64_t a8 = svdup_u64(8), a9 = svdup_u64(9), a10 = svdup_u64(10), a11 = svdup_u64(11);
    auto step = [&](svuint64_t& a) {
        a = sveor_u64_x(pg, a, x);
        asm volatile("" : "+w"(a));
    };
    for (uint64_t i = 0; i < iters; ++i) {
        step(a0), step(a1), step(a2), step(a3), step(a4), step(a5);
        step(a6), step(a7), step(a8), step(a9), step(a10), step(a11);
    }
    svuint64_t s = sveor_u64_x(pg, sveor_u64_x(pg, sveor_u64_x(pg, a0, a1), a2), a3);
    s = sveor_u64_x(pg, sveor_u64_x(pg, sveor_u64_x(pg, sveor_u64_x(pg, s, a4), a5), a6), a7);
    s = sveor_u64_x(pg, sveor_u64_x(pg, sveor_u64_x(pg, sveor_u64_x(pg, s, a8), a9), a10), a11);
    return svaddv_u64(pg, s);
}

#elif defined(__AVX2__)
constexpr char const* kIsaName = "avx2-5op-popcnt";

inline uint32_t popcnt_m256(__m256i v) noexcept {
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), v);
    return static_cast<uint32_t>(
        __builtin_popcountll(tmp[0]) + __builtin_popcountll(tmp[1]) + __builtin_popcountll(tmp[2]) +
        __builtin_popcountll(tmp[3])
    );
}

inline void compare_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out[4]
) noexcept {
    bool const all_full = q.all_full || (r0.all_full && r1.all_full && r2.all_full && r3.all_full);
    uint32_t lts[4] = {0, 0, 0, 0};
    uint32_t nes[4] = {0, 0, 0, 0};
    uint32_t occs[4] = {
        all_full ? kBuckets : 0U,
        all_full ? kBuckets : 0U,
        all_full ? kBuckets : 0U,
        all_full ? kBuckets : 0U,
    };

    for (uint32_t g = 0; g < 8U; ++g) {
        size_t const off = g * 4U;
        __m256i qp = _mm256_load_si256(reinterpret_cast<__m256i const*>(&q.planes[0][off]));
        __m256i rp0 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r0.planes[0][off]));
        __m256i rp1 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r1.planes[0][off]));
        __m256i rp2 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r2.planes[0][off]));
        __m256i rp3 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r3.planes[0][off]));

        __m256i lt0 = _mm256_andnot_si256(qp, rp0);
        __m256i ne0 = _mm256_xor_si256(qp, rp0);
        __m256i lt1 = _mm256_andnot_si256(qp, rp1);
        __m256i ne1 = _mm256_xor_si256(qp, rp1);
        __m256i lt2 = _mm256_andnot_si256(qp, rp2);
        __m256i ne2 = _mm256_xor_si256(qp, rp2);
        __m256i lt3 = _mm256_andnot_si256(qp, rp3);
        __m256i ne3 = _mm256_xor_si256(qp, rp3);

        for (uint32_t p = 1; p < kPlanes; ++p) {
            qp = _mm256_load_si256(reinterpret_cast<__m256i const*>(&q.planes[p][off]));
            rp0 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r0.planes[p][off]));
            rp1 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r1.planes[p][off]));
            rp2 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r2.planes[p][off]));
            rp3 = _mm256_load_si256(reinterpret_cast<__m256i const*>(&r3.planes[p][off]));

            __m256i t0 = _mm256_xor_si256(qp, rp0);
            ne0 = _mm256_or_si256(ne0, t0);
            lt0 = _mm256_xor_si256(lt0, _mm256_and_si256(_mm256_xor_si256(lt0, rp0), t0));

            __m256i t1 = _mm256_xor_si256(qp, rp1);
            ne1 = _mm256_or_si256(ne1, t1);
            lt1 = _mm256_xor_si256(lt1, _mm256_and_si256(_mm256_xor_si256(lt1, rp1), t1));

            __m256i t2 = _mm256_xor_si256(qp, rp2);
            ne2 = _mm256_or_si256(ne2, t2);
            lt2 = _mm256_xor_si256(lt2, _mm256_and_si256(_mm256_xor_si256(lt2, rp2), t2));

            __m256i t3 = _mm256_xor_si256(qp, rp3);
            ne3 = _mm256_or_si256(ne3, t3);
            lt3 = _mm256_xor_si256(lt3, _mm256_and_si256(_mm256_xor_si256(lt3, rp3), t3));
        }

        lts[0] += popcnt_m256(lt0);
        nes[0] += popcnt_m256(ne0);
        lts[1] += popcnt_m256(lt1);
        nes[1] += popcnt_m256(ne1);
        lts[2] += popcnt_m256(lt2);
        nes[2] += popcnt_m256(ne2);
        lts[3] += popcnt_m256(lt3);
        nes[3] += popcnt_m256(ne3);
        if (!all_full) {
            __m256i qnz = _mm256_load_si256(reinterpret_cast<__m256i const*>(&q.nonzero[off]));
            occs[0] += popcnt_m256(_mm256_or_si256(
                qnz, _mm256_load_si256(reinterpret_cast<__m256i const*>(&r0.nonzero[off]))
            ));
            occs[1] += popcnt_m256(_mm256_or_si256(
                qnz, _mm256_load_si256(reinterpret_cast<__m256i const*>(&r1.nonzero[off]))
            ));
            occs[2] += popcnt_m256(_mm256_or_si256(
                qnz, _mm256_load_si256(reinterpret_cast<__m256i const*>(&r2.nonzero[off]))
            ));
            occs[3] += popcnt_m256(_mm256_or_si256(
                qnz, _mm256_load_si256(reinterpret_cast<__m256i const*>(&r3.nonzero[off]))
            ));
        }
    }
    for (int k = 0; k < 4; ++k) {
        out[k] = {lts[k], occs[k] - nes[k], nes[k] - lts[k], kBuckets - occs[k]};
    }
}

inline void compare_2x4(
    BitSlicedSketch const& q0,
    BitSlicedSketch const& q1,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out0[4],
    Counts out1[4]
) noexcept {
    compare_1x4(q0, r0, r1, r2, r3, out0);
    compare_1x4(q1, r0, r1, r2, r3, out1);
}

// Low-plane E bounds of 1 query against 4 references: two AVX2 ops per plane.
inline void low_plane_bound_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    uint32_t out[4]
) noexcept {
    BitSlicedSketch const* rs[4] = {&r0, &r1, &r2, &r3};
    uint32_t diff[4] = {0, 0, 0, 0};
    for (uint32_t g = 0; g < 8U; ++g) {
        size_t const off = g * 4U;
        __m256i ne[4] = {
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256()
        };
        for (uint32_t p = 0; p < kFilterPlanes; ++p) {
            __m256i const qp =
                _mm256_load_si256(reinterpret_cast<__m256i const*>(&q.planes[p][off]));
            for (int k = 0; k < 4; ++k) {
                __m256i const rp =
                    _mm256_load_si256(reinterpret_cast<__m256i const*>(&rs[k]->planes[p][off]));
                ne[k] = _mm256_or_si256(ne[k], _mm256_xor_si256(qp, rp));
            }
        }
        for (int k = 0; k < 4; ++k) diff[k] += popcnt_m256(ne[k]);
    }
    for (int k = 0; k < 4; ++k) out[k] = kBuckets - diff[k];
}

// Exhaustive comparison runs 2 + 15 * 5 logic ops per 256-bit chunk of each pair.
inline uint32_t logic_ops_per_pair() noexcept {
    return 8U * (2U + 15U * 5U);
}

// Twelve independent vpxor chains in named registers measure the cores' logic-op throughput; the
// empty asm keeps the compiler from folding the chains or moving them to memory.
inline uint64_t logic_peak(uint64_t iters) noexcept {
    __m256i x = _mm256_set1_epi64x(0x5555555555555555LL);
    asm volatile("" : "+x"(x));
    __m256i a0 = _mm256_set1_epi64x(0), a1 = _mm256_set1_epi64x(1), a2 = _mm256_set1_epi64x(2);
    __m256i a3 = _mm256_set1_epi64x(3), a4 = _mm256_set1_epi64x(4), a5 = _mm256_set1_epi64x(5);
    __m256i a6 = _mm256_set1_epi64x(6), a7 = _mm256_set1_epi64x(7), a8 = _mm256_set1_epi64x(8);
    __m256i a9 = _mm256_set1_epi64x(9), a10 = _mm256_set1_epi64x(10), a11 = _mm256_set1_epi64x(11);
    auto step = [&](__m256i& a) {
        a = _mm256_xor_si256(a, x);
        asm volatile("" : "+x"(a));
    };
    for (uint64_t i = 0; i < iters; ++i) {
        step(a0), step(a1), step(a2), step(a3), step(a4), step(a5);
        step(a6), step(a7), step(a8), step(a9), step(a10), step(a11);
    }
    return popcnt_m256(a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11);
}
#else
constexpr char const* kIsaName = "scalar-u64-5op";

inline void compare_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out[4]
) noexcept {
    BitSlicedSketch const* rs[4] = {&r0, &r1, &r2, &r3};
    for (int k = 0; k < 4; ++k) {
        uint32_t l = 0, ne_c = 0, occ = 0;
        bool const full = q.all_full || rs[k]->all_full;
        for (uint32_t w = 0; w < kWords64; ++w) {
            uint64_t lt = ~q.planes[0][w] & rs[k]->planes[0][w];
            uint64_t ne = q.planes[0][w] ^ rs[k]->planes[0][w];
            for (uint32_t p = 1; p < kPlanes; ++p) {
                uint64_t const t = q.planes[p][w] ^ rs[k]->planes[p][w];
                ne |= t;
                lt ^= (lt ^ rs[k]->planes[p][w]) & t;
            }
            l += static_cast<uint32_t>(__builtin_popcountll(lt));
            ne_c += static_cast<uint32_t>(__builtin_popcountll(ne));
            if (!full) {
                occ +=
                    static_cast<uint32_t>(__builtin_popcountll(q.nonzero[w] | rs[k]->nonzero[w]));
            }
        }
        uint32_t const o = full ? kBuckets : occ;
        out[k] = {l, o - ne_c, ne_c - l, kBuckets - o};
    }
}
inline void compare_2x4(
    BitSlicedSketch const& q0,
    BitSlicedSketch const& q1,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    Counts out0[4],
    Counts out1[4]
) noexcept {
    compare_1x4(q0, r0, r1, r2, r3, out0);
    compare_1x4(q1, r0, r1, r2, r3, out1);
}

inline void low_plane_bound_1x4(
    BitSlicedSketch const& q,
    BitSlicedSketch const& r0,
    BitSlicedSketch const& r1,
    BitSlicedSketch const& r2,
    BitSlicedSketch const& r3,
    uint32_t out[4]
) noexcept {
    BitSlicedSketch const* rs[4] = {&r0, &r1, &r2, &r3};
    for (int k = 0; k < 4; ++k) out[k] = low_plane_bound(q, *rs[k]);
}

// Exhaustive comparison runs 2 + 15 * 5 logic ops per 64-bit word of each pair.
inline uint32_t logic_ops_per_pair() noexcept {
    return kWords64 * (2U + 15U * 5U);
}

// Twelve independent XOR chains in named registers measure the cores' logic-op throughput; the
// empty asm keeps the compiler from folding the chains or moving them to memory.
inline uint64_t logic_peak(uint64_t iters) noexcept {
    uint64_t x = 0x5555555555555555ULL;
    asm volatile("" : "+r"(x));
    uint64_t a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5;
    uint64_t a6 = 6, a7 = 7, a8 = 8, a9 = 9, a10 = 10, a11 = 11;
    auto step = [&](uint64_t& a) {
        a ^= x;
        asm volatile("" : "+r"(a));
    };
    for (uint64_t i = 0; i < iters; ++i) {
        step(a0), step(a1), step(a2), step(a3), step(a4), step(a5);
        step(a6), step(a7), step(a8), step(a9), step(a10), step(a11);
    }
    return a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7 ^ a8 ^ a9 ^ a10 ^ a11;
}
#endif

struct FreeDeleter {
    void operator()(void* p) const noexcept {
        std::free(p);
    }
};

// Zeroes `bytes` at `p` with all OpenMP threads so first-touch spreads the pages over the NUMA
// nodes the threads run on.
void prefault(void* p, size_t bytes) {
    constexpr size_t chunk = 2ULL << 20;
    size_t const chunks = (bytes + chunk - 1U) / chunk;
    _Pragma("omp parallel for schedule(static)")
    for (size_t c = 0; c < chunks; ++c) {
        std::memset(static_cast<char*>(p) + c * chunk, 0, std::min(chunk, bytes - c * chunk));
    }
}

// Zero-copy overlapped O_DIRECT writer using a pre-allocated ring of 4 KiB-aligned buffers.
class AsyncDirectWriter {
   public:
    static constexpr size_t kAlign = 4096ULL;
    static constexpr size_t kChunk = 8ULL << 20;

    AsyncDirectWriter(
        std::string const& path,
        bool o_direct,
        bool do_fsync,
        size_t slot_bytes,
        size_t num_slots = 3U
    )
        : path_(path), o_direct_(o_direct), do_fsync_(do_fsync) {
        int flags = O_WRONLY | O_CREAT | O_TRUNC;
        if (o_direct_) flags |= O_DIRECT;
        fd_ = ::open(path.c_str(), flags, 0644);
        if (fd_ < 0) {
            std::perror("open async writer");
            std::abort();
        }
        size_t const alloc_bytes = (slot_bytes + kAlign - 1U) & ~(kAlign - 1U);
        slots_.reserve(num_slots);
        for (size_t s = 0; s < num_slots; ++s) {
            void* p = std::aligned_alloc(kAlign, alloc_bytes);
            if (!p) {
                std::perror("aligned_alloc slot");
                std::abort();
            }
            // Pre-fault once before timed work so OpenMP threads never take page faults. The
            // threads touch the pages, which spreads them over the NUMA nodes.
            prefault(p, alloc_bytes);
            slots_.push_back(p);
            free_slots_.push_back(p);
        }
        worker_ = std::thread([this]() { run(); });
    }

    ~AsyncDirectWriter() {
        finish();
        for (void* p : slots_) std::free(p);
    }

    void* acquire_slot() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]() { return !free_slots_.empty(); });
        void* p = free_slots_.front();
        free_slots_.pop_front();
        return p;
    }

    /// A free slot, or nullptr when the writer still holds them all.
    void* try_acquire_slot() {
        std::lock_guard<std::mutex> lk(mu_);
        if (free_slots_.empty()) {
            return nullptr;
        }
        void* p = free_slots_.front();
        free_slots_.pop_front();
        return p;
    }

    void enqueue(void* buf, size_t exact_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back({buf, exact_bytes});
        cv_.notify_all();
    }

    void finish() {
        if (finished_) return;
        finished_ = true;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (fd_ >= 0) {
            if (o_direct_ && total_bytes_ % kAlign != 0) {
                if (::ftruncate(fd_, static_cast<off_t>(total_bytes_)) != 0) {
                    std::perror("ftruncate");
                    std::abort();
                }
            }
            if (do_fsync_) {
                ::fsync(fd_);
                auto parent = std::filesystem::path(path_).parent_path();
                if (parent.empty()) parent = ".";
                int dfd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
                if (dfd >= 0) {
                    ::fsync(dfd);
                    ::close(dfd);
                }
            }
            ::close(fd_);
            fd_ = -1;
        }
    }

    size_t total_bytes() const noexcept {
        return total_bytes_;
    }

    /// Time the writer thread spent inside write() and the number of stripes it wrote; valid
    /// after finish().
    double busy_ms() const noexcept {
        return static_cast<double>(busy_ns_) / 1e6;
    }

    uint64_t writes() const noexcept {
        return writes_;
    }

   private:
    void run() {
        for (;;) {
            std::pair<void*, size_t> item{nullptr, 0};
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]() { return !queue_.empty() || stop_; });
                if (queue_.empty() && stop_) break;
                item = queue_.front();
                queue_.pop_front();
            }
            char* ptr = static_cast<char*>(item.first);
            size_t const exact = item.second;
            size_t rem = exact;
            if (o_direct_ && (exact % kAlign != 0)) {
                size_t const aligned = (exact + kAlign - 1U) & ~(kAlign - 1U);
                std::memset(ptr + exact, 0, aligned - exact);
                rem = aligned;
            }
            auto const write_start = std::chrono::steady_clock::now();
            while (rem > 0) {
                size_t const step = std::min(rem, kChunk);
                ssize_t const n = ::write(fd_, ptr, step);
                if (n <= 0) {
                    std::perror("async write");
                    std::abort();
                }
                ptr += n;
                rem -= static_cast<size_t>(n);
            }
            busy_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - write_start
                )
                    .count()
            );
            ++writes_;
            total_bytes_ += exact;
            {
                std::lock_guard<std::mutex> lk(mu_);
                free_slots_.push_back(item.first);
            }
            cv_.notify_all();
        }
    }

    uint64_t busy_ns_ = 0;
    uint64_t writes_ = 0;
    std::string path_;
    bool o_direct_ = false;
    bool do_fsync_ = false;
    bool finished_ = false;
    int fd_ = -1;
    size_t total_bytes_ = 0;
    std::vector<void*> slots_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<void*> free_slots_;
    std::deque<std::pair<void*, size_t>> queue_;
    bool stop_ = false;
    std::thread worker_;
};

bool load_cuddl_rows(std::string const& path, std::vector<uint16_t>& rows, uint32_t& ref_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        return false;
    }
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "CUDDLDB\0", 8) != 0) {
        std::fprintf(stderr, "invalid magic in %s\n", path.c_str());
        return false;
    }
    uint16_t version = 0;
    uint8_t kmer_len = 0, exp_bits = 0;
    uint16_t buckets = 0, key_mask = 0;
    uint32_t bl_count = 0;
    uint64_t bl_id = 0;
    in.read(reinterpret_cast<char*>(&version), 2);
    in.read(reinterpret_cast<char*>(&kmer_len), 1);
    in.read(reinterpret_cast<char*>(&exp_bits), 1);
    in.read(reinterpret_cast<char*>(&buckets), 2);
    in.read(reinterpret_cast<char*>(&key_mask), 2);
    in.read(reinterpret_cast<char*>(&ref_count), 4);
    in.read(reinterpret_cast<char*>(&bl_count), 4);
    in.read(reinterpret_cast<char*>(&bl_id), 8);
    if (!in || buckets != kBuckets) {
        std::fprintf(stderr, "unexpected header in %s (buckets=%u)\n", path.c_str(), buckets);
        return false;
    }
    in.seekg(static_cast<std::streamoff>(bl_count) * 8, std::ios::cur);
    for (uint32_t i = 0; i < ref_count; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), 4);
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
    }
    rows.resize(static_cast<size_t>(ref_count) * kBuckets);
    in.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(rows.size() * 2U));
    return static_cast<bool>(in);
}

static_assert(kBuckets == cpu_ingest::kBuckets);
static_assert(kBuckets == cpu_deltas::kBuckets);
static_assert(kBuckets == cpu_index::kBuckets);
constexpr uint32_t kIndexUnitRows = 8U;

/// Compares the first @p n CPU-sketched rows with those of the GPU-built database at @p path.
/// Prints the mismatching rows and buckets and returns whether every row is identical.
bool verify_against_db(std::string const& path, std::vector<uint16_t> const& rows, uint32_t n) {
    std::vector<uint16_t> gpu;
    uint32_t gpu_count = 0;
    if (!load_cuddl_rows(path, gpu, gpu_count)) {
        return false;
    }
    if (gpu_count < n) {
        std::fprintf(stderr, "verify: %s has %u rows, ingest produced %u\n", path.c_str(), gpu_count, n);
        return false;
    }
    uint64_t bad_rows = 0, bad_buckets = 0;
    for (uint32_t i = 0; i < n; ++i) {
        auto const* const cpu_row = rows.data() + static_cast<size_t>(i) * kBuckets;
        auto const* const gpu_row = gpu.data() + static_cast<size_t>(i) * kBuckets;
        uint32_t differing = 0;
        for (uint32_t b = 0; b < kBuckets; ++b) {
            if (cpu_row[b] != gpu_row[b]) {
                if (bad_buckets < 10U) {
                    std::fprintf(stderr, "verify: row %u bucket %u cpu=%u gpu=%u\n", i, b, cpu_row[b], gpu_row[b]);
                }
                ++differing;
                ++bad_buckets;
            }
        }
        bad_rows += differing != 0U;
    }
    std::printf(
        "{\"verify_db\":\"%s\",\"rows\":%u,\"mismatching_rows\":%llu,\"mismatching_buckets\":%llu}\n",
        path.c_str(),
        n,
        static_cast<unsigned long long>(bad_rows),
        static_cast<unsigned long long>(bad_buckets)
    );
    return bad_rows == 0U;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"CPU bit-sliced four-way DDL all-to-all comparator"};
    std::string db_path;
    std::string genomes_path;
    std::string verify_path;
    unsigned threads_opt = 0;
    uint32_t limit_n = 0;
    uint32_t tile_size = 64;
    int reps = 3;
    std::string out_path;
    bool do_fsync = false;
    bool o_direct = false;
    bool count_sparsity = false;
    bool use_deltas = false;
    bool use_index = false;
    uint32_t index_bitmap_divisor = 16;
    bool perturb_empty = false;
    bool profile_stripes = false;
    std::string format = "dense";
    uint32_t min_equal = 20;
    uint64_t peak_iters = 0;

    app.add_option("--db", db_path, "Path to v1 .cuddl reference database");
    app.add_option(
        "--genomes",
        genomes_path,
        "Genome folder, or a file listing one gzip/plain FASTA path per line (ordered as "
        "cuddl-build-reference-db orders references); ingested on the CPU instead of --db"
    );
    app.add_option(
        "--verify-db",
        verify_path,
        "With --genomes: compare every ingested row with this GPU-built .cuddl of the same list, "
        "then exit non-zero on any mismatch"
    );
    app.add_option(
        "--threads",
        threads_opt,
        "Ingest workers and OpenMP threads (default: all hardware threads for ingest, the OpenMP "
        "default for the comparison)"
    );
    app.add_option("--limit", limit_n, "Optional prefix of references to compare");
    app.add_option("--tile", tile_size, "Cache-blocking tile size")->default_val(64);
    app.add_option("--reps", reps, "Timed repetitions")->default_val(3);
    app.add_option("--output", out_path, "Optional binary 4-byte WKID output file");
    app.add_flag("--fsync", do_fsync, "Call fsync on the output file before exit");
    app.add_flag("--o-direct", o_direct, "Stream output via overlapped 8 MiB O_DIRECT writer");
    app.add_flag("--sparsity", count_sparsity, "Count E==0, E>=1, E>=5, E>=20 pairs on rep 0");
    app.add_flag(
        "--deltas",
        use_deltas,
        "Compare near-duplicate rows (children) through their base row instead of in full; with "
        "--format edges the threshold applies to the exact counts and the pre-filter is not used; "
        "deltas_build_ms is part of total_ms and end_to_end_ms"
    );
    app.add_flag(
        "--index",
        use_index,
        "Format edges: find candidate pairs through an inverted index over (bucket, score) instead "
        "of the 12-plane pre-filter, then refine them exactly; index_build_ms is part of total_ms "
        "and end_to_end_ms"
    );
    app.add_option(
           "--index-bitmaps",
           index_bitmap_divisor,
           "Testing: posting lists of at least n / this many entries also get bitmaps (0: none)"
    )
        ->default_val(index_bitmap_divisor)
        ->group("");
    app.add_flag(
        "--perturb-empty",
        perturb_empty,
        "Testing: zero 16 buckets in every 3rd row and 300 in every 5th before comparing, so empty "
        "buckets occur"
    )->group("");
    app.add_flag(
        "--profile-stripes",
        profile_stripes,
        "With compact output: print one JSON line per rep with the thread-seconds in compute, "
        "encode and waiting, the time stripes waited for a writer slot, and the writer's busy time"
    )->group("");
    app.add_option(
           "--format",
           format,
           "dense: 4 B float32 WKID per pair; sparse: per-row non-zero bitmap plus packed float32 "
           "WKID (lossless, like sparse_similarity); edges: 12 B (query, ref, WKID) for E >= "
           "--min-equal"
    )
        ->check(CLI::IsMember({"dense", "sparse", "edges"}));
    app.add_option("--min-equal", min_equal, "Edge threshold m_min for --format edges")
        ->default_val(20)
        ->check(CLI::Range(1U, kBuckets));
    app.add_option(
        "--logic-peak",
        peak_iters,
        "Measure the cores' peak logic-op rate with this many iterations per thread, then exit"
    );
    CLI11_PARSE(app, argc, argv);
    if (peak_iters > 0U) {
        int const threads = omp_get_max_threads();
        for (int rep = 0; rep < reps; ++rep) {
            uint64_t sum = 0;
            auto const start = std::chrono::steady_clock::now();
            _Pragma("omp parallel reduction(+ : sum)")
            sum += logic_peak(peak_iters);
            double const seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double const ops = static_cast<double>(peak_iters) * kPeakChains * threads;
            std::printf(
                "{\"isa\":\"%s\",\"threads\":%d,\"mode\":\"logic_peak\",\"rep\":%d,\"iters\":%llu,"
                "\"seconds\":%.4f,\"gops_s\":%.2f,\"logic_ops_per_pair\":%u,\"checksum\":%llu}\n",
                kIsaName,
                threads,
                rep,
                static_cast<unsigned long long>(peak_iters),
                seconds,
                ops / seconds / 1e9,
                logic_ops_per_pair(),
                static_cast<unsigned long long>(sum)
            );
            std::fflush(stdout);
        }
        return 0;
    }
    if (db_path.empty() == genomes_path.empty()) {
        std::fprintf(stderr, "exactly one of --db and --genomes is required\n");
        return 1;
    }
    if (!verify_path.empty() && genomes_path.empty()) {
        std::fprintf(stderr, "--verify-db needs --genomes\n");
        return 1;
    }
    if (threads_opt > 0U) {
        omp_set_num_threads(static_cast<int>(threads_opt));
    }
    bool const compact = format != "dense";
    if (compact && (out_path.empty() || !o_direct)) {
        std::fprintf(stderr, "--format %s needs --output and --o-direct\n", format.c_str());
        return 1;
    }
    if (use_deltas && use_index) {
        std::fprintf(stderr, "--deltas and --index are exclusive\n");
        return 1;
    }
    // Pairs below this E store WKID 0, so compaction keeps exactly the pairs the format emits.
    uint32_t const min_keep = format == "edges" ? min_equal : 0U;
    // Thresholded runs prune with the low-plane bound; sparsity counts need every pair's exact E.
    bool const filter = min_keep > 0U && !count_sparsity;
    if (use_index && !filter) {
        std::fprintf(stderr, "--index needs --format edges and no --sparsity\n");
        return 1;
    }

    std::vector<uint16_t> rows;
    uint32_t n = 0;
    cpu_ingest::IngestStats ingest;
    if (genomes_path.empty()) {
        if (!load_cuddl_rows(db_path, rows, n)) {
            return 1;
        }
    } else {
        std::vector<std::filesystem::path> paths;
        auto const list_start = std::chrono::steady_clock::now();
        if (!cpu_ingest::list_genomes(genomes_path, paths)) {
            return 1;
        }
        ingest.list_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - list_start).count();
        if (paths.empty()) {
            std::fprintf(stderr, "no genomes in %s\n", genomes_path.c_str());
            return 1;
        }
        if (limit_n > 0 && limit_n < paths.size()) {
            paths.resize(limit_n);
        }
        n = static_cast<uint32_t>(paths.size());
        unsigned const workers = threads_opt > 0U ? threads_opt : std::thread::hardware_concurrency();
        if (!cpu_ingest::ingest_genomes(paths, workers, rows, ingest)) {
            return 1;
        }
        if (!verify_path.empty() && !verify_against_db(verify_path, rows, n)) {
            return 4;
        }
    }
    if (limit_n > 0 && limit_n < n) {
        n = limit_n;
    }
    if (perturb_empty) {
        for (uint32_t r = 0; r < n; ++r) {
            uint32_t const zeroed = r % 5U == 0U ? 300U : r % 3U == 0U ? 16U : 0U;
            for (uint32_t k = 0; k < zeroed; ++k) {
                rows[static_cast<size_t>(r) * kBuckets + (k * 131U + r * 7U) % kBuckets] = 0U;
            }
        }
    }

    auto const t0 = std::chrono::steady_clock::now();
    std::vector<BitSlicedSketch> sketches(n);
    _Pragma("omp parallel for schedule(static)")
    for (uint32_t i = 0; i < n; ++i) {
        transpose_row(rows.data() + static_cast<size_t>(i) * kBuckets, sketches[i]);
    }
    double const transpose_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // Verify exact parity against scalar oracle on first 64 x 64 pairs for both 1x4 and 2x4.
    uint32_t const v_n = std::min(n, 64U);
    for (uint32_t i = 0; i + 1U < v_n; i += 2U) {
        for (uint32_t j = 0; j + 3U < v_n; j += 4U) {
            Counts got0[4], got1[4], g1x4[4];
            compare_2x4(
                sketches[i],
                sketches[i + 1],
                sketches[j],
                sketches[j + 1],
                sketches[j + 2],
                sketches[j + 3],
                got0,
                got1
            );
            compare_1x4(
                sketches[i], sketches[j], sketches[j + 1], sketches[j + 2], sketches[j + 3], g1x4
            );
            for (uint32_t k = 0; k < 4U; ++k) {
                Counts const ref0 = scalar_compare(
                    rows.data() + static_cast<size_t>(i) * kBuckets,
                    rows.data() + static_cast<size_t>(j + k) * kBuckets
                );
                Counts const ref1 = scalar_compare(
                    rows.data() + static_cast<size_t>(i + 1U) * kBuckets,
                    rows.data() + static_cast<size_t>(j + k) * kBuckets
                );
                if (low_plane_bound(sketches[i], sketches[j + k]) < ref0.equal ||
                    low_plane_bound(sketches[i + 1U], sketches[j + k]) < ref1.equal ||
                    got0[k].lower != ref0.lower || got0[k].equal != ref0.equal ||
                    got0[k].higher != ref0.higher || got0[k].empty != ref0.empty ||
                    g1x4[k].lower != ref0.lower || g1x4[k].equal != ref0.equal ||
                    got1[k].lower != ref1.lower || got1[k].equal != ref1.equal ||
                    got1[k].higher != ref1.higher || got1[k].empty != ref1.empty) {
                    std::fprintf(
                        stderr,
                        "MISMATCH (%u/%u,%u): got0=(%u,%u,%u,%u) ref0=(%u,%u,%u,%u)\n",
                        i,
                        i + 1U,
                        j + k,
                        got0[k].lower,
                        got0[k].equal,
                        got0[k].higher,
                        got0[k].empty,
                        ref0.lower,
                        ref0.equal,
                        ref0.higher,
                        ref0.empty
                    );
                    return 2;
                }
            }
        }
    }
    cpu_deltas::Model model;
    double deltas_build_ms = 0.0;
    if (use_deltas) {
        auto const td = std::chrono::steady_clock::now();
        model = cpu_deltas::build_model(rows.data(), n);
        deltas_build_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
    }
    cpu_index::Index index;
    double index_build_ms = 0.0;
    if (use_index) {
        auto const ti = std::chrono::steady_clock::now();
        if (!cpu_index::build(rows.data(), n, index_bitmap_divisor, index)) {
            std::fprintf(stderr, "--index: too many postings for 32-bit offsets\n");
            return 1;
        }
        index_build_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
    }
    if (!use_deltas && !use_index) {
        rows.clear();
        rows.shrink_to_fit();
    }

    uint64_t const total_pairs = static_cast<uint64_t>(n) * (n - 1U) / 2U;
    uint32_t const nblocks = (n + tile_size - 1U) / tile_size;
    std::vector<std::pair<uint32_t, uint32_t>> tiles;
    tiles.reserve(static_cast<size_t>(nblocks) * (nblocks + 1U) / 2U);
    // With --deltas a column block is a chunk of unit_bases bases and their children, not tile_size
    // consecutive references. At least 1,024 bases keep enough units for the threads when --tile
    // is large.
    uint32_t const unit_bases = std::max(1024U, 2U * tile_size);
    uint32_t const ncols =
        use_deltas ? static_cast<uint32_t>((model.bases.size() + unit_bases - 1U) / unit_bases)
                   : use_index ? (tile_size + kIndexUnitRows - 1U) / kIndexUnitRows : nblocks;
    // First column block that can hold a pair of row block rb: a base above the block's first row.
    auto first_col = [&](uint32_t rb) -> uint32_t {
        if (use_index) {
            return 0U;
        }
        if (!use_deltas) {
            return rb;
        }
        auto const above =
            std::upper_bound(model.bases.begin(), model.bases.end(), rb * tile_size);
        return static_cast<uint32_t>(above - model.bases.begin()) / unit_bases;
    };
    for (uint32_t bi = 0; bi < nblocks; ++bi) {
        for (uint32_t bj = first_col(bi); bj < ncols; ++bj) {
            tiles.emplace_back(bi, bj);
        }
    }

    // Compares the queries [bi, bi_end) with the bases of chunk cb and derives their children's
    // counts from the bases' counts. A step takes kSubBases bases and runs every group of
    // kGroup queries of the tile against them, so those bases' bit planes are read from memory
    // once and then stay in the L2 cache for the whole tile.
    constexpr uint32_t kSubBases = 64U;
    auto process_delta_unit = [&](uint32_t bi, uint32_t bi_end, uint32_t cb, auto const& emit) {
        thread_local std::vector<cpu_deltas::Lanes> scores;
        thread_local std::vector<cpu_deltas::Lanes> base_lower, base_higher, base_empty;
        uint32_t const sub_bases = std::min(tile_size, kSubBases);
        base_lower.resize(sub_bases);
        base_higher.resize(sub_bases);
        base_empty.resize(sub_bases);
        uint32_t const unit_begin = cb * unit_bases;
        uint32_t const unit_end =
            std::min(unit_begin + unit_bases, static_cast<uint32_t>(model.bases.size()));
        // Children are below their base, so a query at or above the unit's last base needs nothing.
        uint32_t const unit_last = model.bases[unit_end - 1U];
        uint32_t const groups =
            (std::min(bi_end, unit_last) - bi + cpu_deltas::kGroup - 1U) / cpu_deltas::kGroup;
        scores.resize(static_cast<size_t>(groups) * kBuckets);
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t const g0 = bi + g * cpu_deltas::kGroup;
            cpu_deltas::transpose_group(
                rows.data(),
                g0,
                std::min(g0 + cpu_deltas::kGroup, bi_end) - g0,
                scores.data() + static_cast<size_t>(g) * kBuckets
            );
        }
        for (uint32_t sub = unit_begin; sub < unit_end; sub += sub_bases) {
            uint32_t const sub_end = std::min(sub + sub_bases, unit_end);
            uint32_t const last = sub_end - 1U;
            for (uint32_t g = 0; g < groups; ++g) {
                uint32_t const g0 = bi + g * cpu_deltas::kGroup;
                uint32_t const g_end = std::min(g0 + cpu_deltas::kGroup, bi_end);
                if (model.bases[last] <= g0) {
                    break;
                }
                cpu_deltas::Lanes const* const group_scores =
                    scores.data() + static_cast<size_t>(g) * kBuckets;
                auto keep = [&](uint32_t q, uint32_t slot, uint32_t base, Counts const& cnt) {
                    base_lower[slot].v[q - g0] = static_cast<uint16_t>(cnt.lower);
                    base_higher[slot].v[q - g0] = static_cast<uint16_t>(cnt.higher);
                    base_empty[slot].v[q - g0] = static_cast<uint16_t>(cnt.empty);
                    if (q < base) {
                        emit(q, base, cnt);
                    }
                };
                for (uint32_t i = g0; i < g_end && i < model.bases[last]; i += 2U) {
                    uint32_t const i1 = std::min(i + 1U, g_end - 1U);
                    for (uint32_t s = sub; s < sub_end; s += 4U) {
                        uint32_t const id[4] = {
                            model.bases[s],
                            model.bases[std::min(s + 1U, last)],
                            model.bases[std::min(s + 2U, last)],
                            model.bases[std::min(s + 3U, last)],
                        };
                        Counts c0[4], c1[4];
                        compare_2x4(
                            sketches[i],
                            sketches[i1],
                            sketches[id[0]],
                            sketches[id[1]],
                            sketches[id[2]],
                            sketches[id[3]],
                            c0,
                            c1
                        );
                        for (uint32_t k = 0; k < 4U && s + k <= last; ++k) {
                            keep(i, s + k - sub, id[k], c0[k]);
                            if (i1 != i) {
                                keep(i1, s + k - sub, id[k], c1[k]);
                            }
                        }
                    }
                }
                for (uint32_t s = sub; s < sub_end; ++s) {
                    for (uint32_t ci = model.child_begin[s]; ci < model.child_begin[s + 1U]; ++ci) {
                        cpu_deltas::Child const& child = model.children[ci];
                        if (child.id <= g0) {
                            continue;
                        }
                        cpu_deltas::Correction corr;
                        cpu_deltas::correct(child, model.changes.data(), group_scores, corr);
                        uint32_t const slot = s - sub;
                        for (uint32_t k = 0; g0 + k < g_end && g0 + k < child.id; ++k) {
                            uint32_t const lower = base_lower[slot].v[k] + corr.lower[k];
                            uint32_t const higher = base_higher[slot].v[k] + corr.higher[k];
                            uint32_t const empty = base_empty[slot].v[k] + corr.empty[k];
                            emit(
                                g0 + k,
                                child.id,
                                Counts{lower, kBuckets - lower - higher - empty, higher, empty}
                            );
                        }
                    }
                }
            }
        }
    };

    // Looks up the queries [q0, q1) of a unit in the index, then refines every candidate pair with
    // the exact comparison; every other pair of the row stores WKID 0.
    std::atomic<uint64_t> index_candidates{0}, index_passing{0}, index_postings{0};
    std::atomic<uint64_t> index_bitmap_lists{0}, index_lookup_ns{0}, index_refine_ns{0};
    auto process_index_unit = [&](uint32_t bi,
                                  uint32_t bi_end,
                                  uint32_t cb,
                                  float* out_ptr,
                                  uint64_t base_offset,
                                  auto const& emit) {
        uint32_t const q0 = bi + cb * kIndexUnitRows;
        uint32_t const q1 = std::min(q0 + kIndexUnitRows, bi_end);
        thread_local cpu_index::Lookup lookup;
        thread_local std::vector<uint32_t> cand;
        cpu_index::Work work;
        uint64_t passing = 0, lookup_ns = 0, refine_ns = 0;
        auto const elapsed_ns = [](auto from, auto to) {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()
            );
        };
        for (uint32_t i = q0; i < q1; ++i) {
            auto const t0 = std::chrono::steady_clock::now();
            cand.clear();
            lookup.candidates(
                index, rows.data() + static_cast<size_t>(i) * kBuckets, i, min_keep, cand, work
            );
            auto const t1 = std::chrono::steady_clock::now();
            if (out_ptr != nullptr) {
                uint64_t const row_base =
                    static_cast<uint64_t>(i) * (2ULL * n - 1ULL - i) / 2ULL - base_offset;
                std::memset(out_ptr + row_base, 0, (n - 1U - i) * sizeof(float));
            }
            for (size_t k = 0; k < cand.size(); k += 4U) {
                size_t const left = std::min<size_t>(cand.size() - k, 4U);
                uint32_t const id[4] = {
                    cand[k], cand[k + std::min<size_t>(1U, left - 1U)],
                    cand[k + std::min<size_t>(2U, left - 1U)], cand[k + left - 1U]
                };
                Counts c[4];
                compare_1x4(
                    sketches[i], sketches[id[0]], sketches[id[1]], sketches[id[2]], sketches[id[3]], c
                );
                for (size_t m = 0; m < left; ++m) {
                    emit(i, id[m], c[m]);
                    passing += c[m].equal >= min_keep;
                }
            }
            lookup_ns += elapsed_ns(t0, t1);
            refine_ns += elapsed_ns(t1, std::chrono::steady_clock::now());
        }
        index_candidates += work.candidates;
        index_passing += passing;
        index_postings += work.list_postings;
        index_bitmap_lists += work.bitmap_lists;
        index_lookup_ns += lookup_ns;
        index_refine_ns += refine_ns;
    };

    auto process_tile = [&](uint32_t bi,
                            uint32_t bi_end,
                            uint32_t bj,
                            uint32_t bj_end,
                            float* out_ptr,
                            uint64_t base_offset,
                            uint64_t& e0,
                            uint64_t& e1,
                            uint64_t& e5,
                            uint64_t& e20,
                            double& checksum,
                            bool do_sp,
                            cpu_deltas::BlockedStripe const* blocked,
                            uint32_t* nnz_local,
                            uint32_t nnz_first_row) {
        auto emit = [&](uint32_t row, uint32_t col, Counts const& cnt) {
            float const w =
                cnt.equal >= min_keep ? wkid_from_counts(cnt.lower, cnt.equal, cnt.higher) : 0.0f;
            if (out_ptr == nullptr) {
                checksum += w;
            } else if (blocked != nullptr) {
                out_ptr[blocked->index(row, col)] = w;
            } else {
                uint64_t const row_base =
                    static_cast<uint64_t>(row) * (2ULL * n - 1ULL - row) / 2ULL - base_offset;
                out_ptr[row_base + (col - row - 1U)] = w;
            }
            // Compact output sizes each row from its non-zero results; counting them here
            // saves a pass over the stripe.
            if (nnz_local != nullptr) {
                nnz_local[row - nnz_first_row] += w != 0.0f;
            }
            if (do_sp) {
                if (cnt.equal == 0U) {
                    ++e0;
                } else {
                    ++e1;
                    if (cnt.equal >= 5U) ++e5;
                    if (cnt.equal >= 20U) ++e20;
                }
            }
        };

        if (use_deltas) {
            process_delta_unit(bi, bi_end, bj / tile_size, emit);
            return;
        }

        if (use_index) {
            process_index_unit(bi, bi_end, bj / tile_size, out_ptr, base_offset, emit);
            return;
        }

        if (filter) {
            // Survivors of the bound run the exact comparison four references at a time.
            for (uint32_t i = bi; i < bi_end; ++i) {
                uint32_t pending[4];
                uint32_t count = 0;
                auto flush = [&] {
                    Counts c[4];
                    compare_1x4(
                        sketches[i],
                        sketches[pending[0]],
                        sketches[pending[std::min(1U, count - 1U)]],
                        sketches[pending[std::min(2U, count - 1U)]],
                        sketches[pending[count - 1U]],
                        c
                    );
                    for (uint32_t k = 0; k < count; ++k) emit(i, pending[k], c[k]);
                    count = 0;
                };
                uint32_t j = std::max(i + 1U, bj);
                for (; j < bj_end; j += 4U) {
                    uint32_t const last = std::min(j + 3U, bj_end - 1U);
                    uint32_t bound[4];
                    low_plane_bound_1x4(
                        sketches[i],
                        sketches[j],
                        sketches[std::min(j + 1U, last)],
                        sketches[std::min(j + 2U, last)],
                        sketches[last],
                        bound
                    );
                    for (uint32_t k = 0; j + k <= last; ++k) {
                        if (bound[k] < min_keep) {
                            emit(i, j + k, Counts{0, 0, 0, 0});
                            continue;
                        }
                        pending[count++] = j + k;
                        if (count == 4U) flush();
                    }
                }
                if (count > 0U) flush();
            }
            return;
        }

        uint32_t i = bi;
        for (; i + 1U < bi_end; i += 2U) {
            if (i + 1U >= bj && i + 1U < bj_end) {
                Counts cd[4];
                compare_1x4(
                    sketches[i],
                    sketches[i + 1],
                    sketches[i + 1],
                    sketches[i + 1],
                    sketches[i + 1],
                    cd
                );
                emit(i, i + 1U, cd[0]);
            }
            uint32_t j = std::max(i + 2U, bj);
            for (; j + 3U < bj_end; j += 4U) {
                Counts c0[4], c1[4];
                compare_2x4(
                    sketches[i],
                    sketches[i + 1],
                    sketches[j],
                    sketches[j + 1],
                    sketches[j + 2],
                    sketches[j + 3],
                    c0,
                    c1
                );
                for (uint32_t k = 0; k < 4U; ++k) {
                    emit(i, j + k, c0[k]);
                    emit(i + 1U, j + k, c1[k]);
                }
            }
            if (j < bj_end) {
                uint32_t const rem = bj_end - j;
                uint32_t const j0 = j;
                uint32_t const j1 = std::min(j + 1U, bj_end - 1U);
                uint32_t const j2 = std::min(j + 2U, bj_end - 1U);
                Counts c0[4], c1[4];
                compare_2x4(
                    sketches[i],
                    sketches[i + 1],
                    sketches[j0],
                    sketches[j1],
                    sketches[j2],
                    sketches[j2],
                    c0,
                    c1
                );
                for (uint32_t k = 0; k < rem; ++k) {
                    emit(i, j + k, c0[k]);
                    emit(i + 1U, j + k, c1[k]);
                }
            }
        }
        if (i < bi_end) {
            uint32_t j = std::max(i + 1U, bj);
            for (; j + 3U < bj_end; j += 4U) {
                Counts c[4];
                compare_1x4(
                    sketches[i], sketches[j], sketches[j + 1], sketches[j + 2], sketches[j + 3], c
                );
                for (uint32_t k = 0; k < 4U; ++k) emit(i, j + k, c[k]);
            }
            if (j < bj_end) {
                uint32_t const rem = bj_end - j;
                uint32_t const j0 = j;
                uint32_t const j1 = std::min(j + 1U, bj_end - 1U);
                uint32_t const j2 = std::min(j + 2U, bj_end - 1U);
                Counts c[4];
                compare_1x4(sketches[i], sketches[j0], sketches[j1], sketches[j2], sketches[j2], c);
                for (uint32_t k = 0; k < rem; ++k) emit(i, j + k, c[k]);
            }
        }
    };

    int const nthreads = omp_get_max_threads();
    for (int rep = 0; rep < reps; ++rep) {
        uint64_t e0 = 0, e1 = 0, e5 = 0, e20 = 0;
        index_candidates = 0;
        index_passing = 0;
        index_postings = 0;
        index_bitmap_lists = 0;
        index_lookup_ns = 0;
        index_refine_ns = 0;
        double checksum = 0.0;
        double compare_ms = 0.0;
        double write_ms = 0.0;
        uint64_t bytes_written = 0;
        bool const do_sp = count_sparsity && rep == 0;
        if (!out_path.empty() && o_direct) {
            // Stream stripes whose row counts are multiples of 2048 so every non-final stripe
            // has an exact 4 KiB-aligned byte size (zero-copy O_DIRECT, pre-allocated ring).
            uint32_t const rows_per_block = 2048U;
            uint32_t const blocks_per_stripe = std::max(1U, rows_per_block / tile_size);
            uint64_t max_stripe_pairs = 0;
            for (uint32_t r = 0; r < std::min(n, blocks_per_stripe * tile_size); ++r) {
                max_stripe_pairs += (n - 1U - r);
            }
            uint32_t const max_stripe_rows = std::min(n, blocks_per_stripe * tile_size);
            // Compacted stripes start with an 8-byte payload length and pad to 4 KiB, so every
            // O_DIRECT write stays aligned and the file decodes stripe by stripe.
            size_t const slot_bytes =
                !compact ? max_stripe_pairs * sizeof(float)
                         : 8U + AsyncDirectWriter::kAlign +
                               (format == "edges"
                                    ? max_stripe_pairs * 12U
                                    : max_stripe_pairs * sizeof(float) + max_stripe_rows * 8U);
            AsyncDirectWriter writer(out_path, true, do_fsync, slot_bytes, 3U);
            // Compact output with --deltas scatters its results over the columns, so its stripe is
            // stored in line-sized blocks of query rows (BlockedStripe) before compaction.
            bool const use_blocked = compact && use_deltas;
            constexpr uint32_t kBuffers = 2U;
            uint32_t const stripe_count = (nblocks + blocks_per_stripe - 1U) / blocks_per_stripe;
            // Compact stripes compute into scratch buffers that are encoded while the next stripe
            // computes; dense stripes compute straight into the writer's slot.
            struct ScratchBuffer {
                std::unique_ptr<float, FreeDeleter> data;
                cpu_deltas::BlockedStripe layout;
                std::vector<uint64_t> row_off;
                std::unique_ptr<std::atomic<uint32_t>[]> nnz;
            };
            std::vector<ScratchBuffer> scratch(compact ? kBuffers : 0U);
            for (ScratchBuffer& buffer : scratch) {
                uint64_t const floats =
                    use_blocked ? cpu_deltas::BlockedStripe::max_floats(
                                      n, max_stripe_rows, blocks_per_stripe * tile_size
                                  )
                                : max_stripe_pairs;
                size_t const bytes = (floats * sizeof(float) + 4095U) & ~size_t{4095U};
                buffer.data.reset(static_cast<float*>(std::aligned_alloc(4096U, bytes)));
                if (!buffer.data) {
                    std::perror("aligned_alloc scratch");
                    std::abort();
                }
                prefault(buffer.data.get(), bytes);
                buffer.row_off.assign(max_stripe_rows + 1U, 0U);
                buffer.nnz = std::make_unique<std::atomic<uint32_t>[]>(max_stripe_rows);
            }
            struct StripeInfo {
                uint32_t r0 = 0;
                uint32_t rows_in = 0;
                uint64_t base_offset = 0;
                std::vector<std::pair<uint32_t, uint32_t>> tiles;
            };
            std::vector<StripeInfo> infos(compact ? stripe_count : 0U);
            cpu_pipeline::Profile pipeline_profile;
            bool have_profile = false;
            auto const start = std::chrono::steady_clock::now();
            if (!compact) {
                uint32_t row_block = 0;
                while (row_block < nblocks) {
                    uint32_t const rb_end = std::min(nblocks, row_block + blocks_per_stripe);
                    uint32_t const r0 = row_block * tile_size;
                    uint64_t const base_offset =
                        static_cast<uint64_t>(r0) * (2ULL * n - 1ULL - r0) / 2ULL;
                    uint64_t stripe_pairs = 0;
                    for (uint32_t rb = row_block; rb < rb_end; ++rb) {
                        uint32_t const bi = rb * tile_size;
                        uint32_t const bi_end = std::min(bi + tile_size, n);
                        for (uint32_t r = bi; r < bi_end; ++r) {
                            stripe_pairs += (n - 1U - r);
                        }
                    }
                    std::vector<std::pair<uint32_t, uint32_t>> stripe_tiles;
                    for (uint32_t rb = row_block; rb < rb_end; ++rb) {
                        for (uint32_t cb = first_col(rb); cb < ncols; ++cb) {
                            stripe_tiles.emplace_back(rb, cb);
                        }
                    }
                    void* buf = writer.acquire_slot();
                    float* fbuf = static_cast<float*>(buf);
                    _Pragma("omp parallel reduction(+ : e0, e1, e5, e20, checksum)")
                    {
                        _Pragma("omp for schedule(dynamic, 1)")
                        for (size_t t = 0; t < stripe_tiles.size(); ++t) {
                            uint32_t const bi = stripe_tiles[t].first * tile_size;
                            uint32_t const bi_end = std::min(bi + tile_size, n);
                            uint32_t const bj = stripe_tiles[t].second * tile_size;
                            uint32_t const bj_end = std::min(bj + tile_size, n);
                            process_tile(
                                bi,
                                bi_end,
                                bj,
                                bj_end,
                                fbuf,
                                base_offset,
                                e0,
                                e1,
                                e5,
                                e20,
                                checksum,
                                do_sp,
                                nullptr,
                                nullptr,
                                0U
                            );
                        }
                    }
                    writer.enqueue(buf, stripe_pairs * sizeof(float));
                    row_block = rb_end;
                }
            } else {
                auto const row_start = [&](StripeInfo const& info, uint32_t k) {
                    uint64_t const r = info.r0 + k;
                    return r * (2ULL * n - 1ULL - r) / 2ULL - info.base_offset;
                };
                cpu_pipeline::StripePipeline::Hooks hooks;
                hooks.prepare = [&](size_t s, uint32_t b) -> size_t {
                    StripeInfo& info = infos[s];
                    uint32_t const row_block = static_cast<uint32_t>(s) * blocks_per_stripe;
                    uint32_t const rb_end = std::min(nblocks, row_block + blocks_per_stripe);
                    info.r0 = row_block * tile_size;
                    info.rows_in = std::min(n, rb_end * tile_size) - info.r0;
                    info.base_offset =
                        static_cast<uint64_t>(info.r0) * (2ULL * n - 1ULL - info.r0) / 2ULL;
                    info.tiles.clear();
                    for (uint32_t rb = row_block; rb < rb_end; ++rb) {
                        for (uint32_t cb = first_col(rb); cb < ncols; ++cb) {
                            info.tiles.emplace_back(rb, cb);
                        }
                    }
                    ScratchBuffer& buffer = scratch[b];
                    if (use_blocked) {
                        buffer.layout.init(info.r0, info.rows_in, n);
                    }
                    for (uint32_t k = 0; k < info.rows_in; ++k) {
                        buffer.nnz[k].store(0U, std::memory_order_relaxed);
                    }
                    return info.tiles.size();
                };
                // Each row stores a bitmap and its values, or 12-byte edge records, after the
                // rows before it; the byte offsets follow from the non-zero counts.
                hooks.finish_compute = [&](size_t s, uint32_t b) {
                    StripeInfo const& info = infos[s];
                    ScratchBuffer& buffer = scratch[b];
                    buffer.row_off[0] = 0;
                    for (uint32_t k = 0; k < info.rows_in; ++k) {
                        uint64_t const len = n - 1U - (info.r0 + k);
                        uint64_t const nnz = buffer.nnz[k].load(std::memory_order_relaxed);
                        buffer.row_off[k + 1U] =
                            buffer.row_off[k] +
                            (format == "edges" ? nnz * 12U : (len + 63U) / 64U * 8U + nnz * 4U);
                    }
                };
                hooks.begin_encode = [&](size_t s, uint32_t b, void* slot) -> uint32_t {
                    StripeInfo const& info = infos[s];
                    uint64_t const payload = scratch[b].row_off[info.rows_in];
                    std::memcpy(slot, &payload, 8);
                    return use_blocked ? scratch[b].layout.blocks() : (info.rows_in + 7U) / 8U;
                };
                hooks.encode = [&](size_t s, uint32_t b, void* slot, uint32_t chunk) {
                    StripeInfo const& info = infos[s];
                    ScratchBuffer const& buffer = scratch[b];
                    float const* const fbuf = buffer.data.get();
                    char* const out = static_cast<char*>(slot);
                    if (use_blocked) {
                        constexpr uint32_t block_rows = cpu_deltas::BlockedStripe::kBlockRows;
                        char* dst[block_rows] = {};
                        for (uint32_t j = 0; j < block_rows; ++j) {
                            uint32_t const k_row = chunk * block_rows + j;
                            if (k_row < info.rows_in) {
                                dst[j] = out + 8U + buffer.row_off[k_row];
                            }
                        }
                        if (format == "edges") {
                            buffer.layout.encode<true>(fbuf, chunk, dst);
                        } else {
                            buffer.layout.encode<false>(fbuf, chunk, dst);
                        }
                        return;
                    }
                    for (uint32_t k = chunk * 8U; k < std::min(info.rows_in, chunk * 8U + 8U); ++k) {
                        uint32_t const r = info.r0 + k;
                        uint64_t const len = n - 1U - r;
                        float const* src = fbuf + row_start(info, k);
                        char* dst = out + 8U + buffer.row_off[k];
                        if (format == "edges") {
                            for (uint64_t c = 0; c < len; ++c) {
                                if (src[c] == 0.0f) continue;
                                uint32_t const col = r + 1U + static_cast<uint32_t>(c);
                                std::memcpy(dst, &r, 4);
                                std::memcpy(dst + 4, &col, 4);
                                std::memcpy(dst + 8, &src[c], 4);
                                dst += 12;
                            }
                        } else {
                            uint64_t const words = (len + 63U) / 64U;
                            char* vals = dst + words * 8U;
                            for (uint64_t w = 0; w < words; ++w) {
                                uint64_t bits = 0;
                                uint64_t const end = std::min(len, (w + 1U) * 64U);
                                for (uint64_t c = w * 64U; c < end; ++c) {
                                    if (src[c] == 0.0f) continue;
                                    bits |= uint64_t{1} << (c & 63U);
                                    std::memcpy(vals, &src[c], 4);
                                    vals += 4;
                                }
                                std::memcpy(dst + w * 8U, &bits, 8);
                            }
                        }
                    }
                };
                hooks.finish_encode = [&](size_t s, uint32_t b, void* slot) -> size_t {
                    StripeInfo& info = infos[s];
                    uint64_t const payload = scratch[b].row_off[info.rows_in];
                    size_t const padded = (8U + payload + AsyncDirectWriter::kAlign - 1U) &
                                          ~(AsyncDirectWriter::kAlign - 1U);
                    std::memset(static_cast<char*>(slot) + 8U + payload, 0, padded - 8U - payload);
                    std::vector<std::pair<uint32_t, uint32_t>>().swap(info.tiles);
                    return padded;
                };
                hooks.try_acquire_slot = [&]() { return writer.try_acquire_slot(); };
                hooks.enqueue = [&](void* slot, size_t bytes) { writer.enqueue(slot, bytes); };
                cpu_pipeline::StripePipeline pipeline(stripe_count, kBuffers, std::move(hooks));
                _Pragma("omp parallel reduction(+ : e0, e1, e5, e20, checksum)")
                {
                    std::vector<uint32_t> nnz_local;
                    pipeline.work([&](size_t s, uint32_t b, size_t t) {
                        StripeInfo const& info = infos[s];
                        ScratchBuffer& buffer = scratch[b];
                        uint32_t const bi = info.tiles[t].first * tile_size;
                        uint32_t const bi_end = std::min(bi + tile_size, n);
                        uint32_t const bj = info.tiles[t].second * tile_size;
                        uint32_t const bj_end = std::min(bj + tile_size, n);
                        nnz_local.assign(bi_end - bi, 0U);
                        process_tile(
                            bi,
                            bi_end,
                            bj,
                            bj_end,
                            buffer.data.get(),
                            info.base_offset,
                            e0,
                            e1,
                            e5,
                            e20,
                            checksum,
                            do_sp,
                            use_blocked ? &buffer.layout : nullptr,
                            nnz_local.data(),
                            bi
                        );
                        for (uint32_t i = 0; i < bi_end - bi; ++i) {
                            if (nnz_local[i] != 0U) {
                                buffer.nnz[bi - info.r0 + i].fetch_add(
                                    nnz_local[i], std::memory_order_relaxed
                                );
                            }
                        }
                    });
                }
                pipeline_profile = pipeline.profile();
                have_profile = true;
            }
            auto const t_comp = std::chrono::steady_clock::now();
            compare_ms = std::chrono::duration<double, std::milli>(t_comp - start).count();
            writer.finish();
            auto const t_done = std::chrono::steady_clock::now();
            write_ms = std::chrono::duration<double, std::milli>(t_done - t_comp).count();
            bytes_written = writer.total_bytes();
            if (profile_stripes && have_profile) {
                std::printf(
                    "{\"profile\":{\"stripes\":%llu,\"threads\":%d,\"compare_ms\":%.1f,"
                    "\"compute_thread_s\":%.2f,\"encode_thread_s\":%.2f,"
                    "\"idle_buffer_thread_s\":%.2f,\"idle_slot_thread_s\":%.2f,"
                    "\"idle_tail_thread_s\":%.2f,\"serial_s\":%.3f,\"slot_wait_s\":%.2f,"
                    "\"slot_waits\":%llu,\"writer_busy_ms\":%.1f,\"writes\":%llu,"
                    "\"bytes\":%llu,\"drain_ms\":%.1f}}\n",
                    static_cast<unsigned long long>(pipeline_profile.stripes),
                    nthreads,
                    compare_ms,
                    pipeline_profile.compute_s,
                    pipeline_profile.encode_s,
                    pipeline_profile.idle_buffer_s,
                    pipeline_profile.idle_slot_s,
                    pipeline_profile.idle_tail_s,
                    pipeline_profile.serial_s,
                    pipeline_profile.slot_wait_s,
                    static_cast<unsigned long long>(pipeline_profile.slot_waits),
                    writer.busy_ms(),
                    static_cast<unsigned long long>(writer.writes()),
                    static_cast<unsigned long long>(bytes_written),
                    write_ms
                );
            }
        } else {
            std::vector<float> out_buf;
            if (!out_path.empty()) {
                out_buf.resize(total_pairs);
            }
            auto const start = std::chrono::steady_clock::now();
            _Pragma("omp parallel reduction(+ : e0, e1, e5, e20, checksum)")
            {
                _Pragma("omp for schedule(dynamic, 1)")
                for (size_t t = 0; t < tiles.size(); ++t) {
                    uint32_t const bi = tiles[t].first * tile_size;
                    uint32_t const bi_end = std::min(bi + tile_size, n);
                    uint32_t const bj = tiles[t].second * tile_size;
                    uint32_t const bj_end = std::min(bj + tile_size, n);
                    process_tile(
                        bi,
                        bi_end,
                        bj,
                        bj_end,
                        out_buf.empty() ? nullptr : out_buf.data(),
                        0ULL,
                        e0,
                        e1,
                        e5,
                        e20,
                        checksum,
                        do_sp,
                        nullptr,
                        nullptr,
                        0U
                    );
                }
            }
            compare_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();

            if (!out_path.empty()) {
                auto const ws = std::chrono::steady_clock::now();
                int const fd = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) {
                    std::perror("open output");
                    return 3;
                }
                char const* ptr = reinterpret_cast<char const*>(out_buf.data());
                size_t rem = out_buf.size() * sizeof(float);
                while (rem > 0) {
                    size_t const chunk = std::min<size_t>(rem, 8ULL << 20);
                    ssize_t const nwr = ::write(fd, ptr, chunk);
                    if (nwr <= 0) {
                        std::perror("write");
                        ::close(fd);
                        return 3;
                    }
                    ptr += nwr;
                    rem -= static_cast<size_t>(nwr);
                }
                if (do_fsync) {
                    ::fsync(fd);
                }
                ::close(fd);
                write_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ws)
                        .count();
                bytes_written = out_buf.size() * sizeof(float);
            }
        }

        double const total_ms = compare_ms + write_ms + deltas_build_ms + index_build_ms;
        double const gpairs_s = (static_cast<double>(total_pairs) / 1e9) /
                                ((compare_ms + deltas_build_ms + index_build_ms) / 1e3);
        std::printf(
            "{\"isa\":\"%s\",\"threads\":%d,\"n\":%u,\"pairs\":%llu,\"tile\":%u,\"rep\":%d,"
            "\"o_direct\":%s,\"fsync\":%s,\"format\":\"%s\",\"min_equal\":%u,"
            "\"low_plane_filter\":%s,\"bytes\":%llu,"
            "\"transpose_ms\":%.2f,\"compare_ms\":%.2f,\"write_ms\":%.2f,\"total_ms\":%.2f,"
            "\"gpairs_s\":%.3f,\"logic_ops_per_pair\":%u,\"checksum\":%.1f,\"deltas\":%s,"
            "\"deltas_build_ms\":%.2f,\"deltas_bases\":%llu,\"deltas_children\":%llu,"
            "\"deltas_changed\":%llu",
            kIsaName,
            nthreads,
            n,
            static_cast<unsigned long long>(total_pairs),
            tile_size,
            rep,
            o_direct ? "true" : "false",
            do_fsync ? "true" : "false",
            format.c_str(),
            format == "edges" ? min_equal : 0U,
            filter && !use_deltas ? "true" : "false",
            static_cast<unsigned long long>(bytes_written),
            transpose_ms,
            compare_ms,
            write_ms,
            total_ms,
            gpairs_s,
            logic_ops_per_pair(),
            checksum,
            use_deltas ? "true" : "false",
            deltas_build_ms,
            static_cast<unsigned long long>(use_deltas ? model.bases.size() : n),
            static_cast<unsigned long long>(model.children.size()),
            static_cast<unsigned long long>(model.changes.size())
        );
        if (use_index) {
            // Lookup and refine times are thread-seconds divided by the thread count.
            std::printf(
                ",\"index\":true,\"index_build_ms\":%.2f,\"index_keys\":%u,\"index_cells\":%llu,"
                "\"index_postings_total\":%llu,\"index_mean_posting\":%.2f,"
                "\"index_bitmap_cells\":%llu,\"index_bytes\":%llu,\"index_candidates\":%llu,"
                "\"index_passing\":%llu,\"index_list_postings\":%llu,\"index_bitmap_lists\":%llu,"
                "\"index_lookup_ms\":%.2f,\"index_refine_ms\":%.2f",
                index_build_ms,
                index.keys,
                static_cast<unsigned long long>(index.nonempty_cells),
                static_cast<unsigned long long>(index.postings.size()),
                static_cast<double>(index.postings.size()) /
                    static_cast<double>(std::max<uint64_t>(index.nonempty_cells, 1U)),
                static_cast<unsigned long long>(index.bitmap_cells),
                static_cast<unsigned long long>(index.bytes()),
                static_cast<unsigned long long>(index_candidates.load()),
                static_cast<unsigned long long>(index_passing.load()),
                static_cast<unsigned long long>(index_postings.load()),
                static_cast<unsigned long long>(index_bitmap_lists.load()),
                static_cast<double>(index_lookup_ns.load()) / 1e6 / nthreads,
                static_cast<double>(index_refine_ns.load()) / 1e6 / nthreads
            );
        } else {
            std::printf(",\"index\":false");
        }
        if (count_sparsity && rep == 0) {
            std::printf(
                ",\"e0\":%llu,\"e_ge1\":%llu,\"e_ge5\":%llu,\"e_ge20\":%llu",
                static_cast<unsigned long long>(e0),
                static_cast<unsigned long long>(e1),
                static_cast<unsigned long long>(e5),
                static_cast<unsigned long long>(e20)
            );
        }
        if (!genomes_path.empty()) {
            // End to end is ingest + transpose + compare/write; listing is reported separately.
            std::printf(
                ",\"input\":\"genomes\",\"ingest_workers\":%u,\"list_s\":%.4f,\"ingest_s\":%.4f,"
                "\"ingest_bases\":%llu,\"ingest_kmers\":%llu,\"ingest_input_bytes\":%llu,"
                "\"ingest_gb_s\":%.3f,\"end_to_end_ms\":%.2f",
                ingest.workers,
                ingest.list_s,
                ingest.ingest_s,
                static_cast<unsigned long long>(ingest.bases),
                static_cast<unsigned long long>(ingest.kmers),
                static_cast<unsigned long long>(ingest.input_bytes),
                static_cast<double>(ingest.input_bytes) / 1e9 / ingest.ingest_s,
                ingest.ingest_s * 1e3 + transpose_ms + total_ms
            );
        }
        std::printf("}\n");
        std::fflush(stdout);
    }
    return 0;
}
