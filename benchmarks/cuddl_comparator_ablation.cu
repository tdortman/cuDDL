// Comparator ablation: scalar, SWAR, tiled and bit-sliced all-pairs variants on synthetic or real
// sketches, plus a host estimate of the warp-uniform MSB-first early-exit rate (--exit-stats).
// Build from the repository root (needs `meson compile -C build` once for liblibdeflate.a):
//   nvcc -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda -Iinclude \
//     -Isubprojects/cccl/libcudacxx/include -Isubprojects/cccl/cub -Isubprojects/cccl/thrust \
//     -Isubprojects/libdeflate \
//     -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 \
//     -gencode=arch=compute_120,code=sm_120 \
//     benchmarks/cuddl_comparator_ablation.cu build/subprojects/libdeflate/liblibdeflate.a \
//     -o cuddl-comparator-ablation
//
// Usage:
//   cuddl-comparator-ablation [--n N] [--reps R]            synthetic sketches (default N=100000)
//   cuddl-comparator-ablation --database f.cuddl [--limit N] [--reps R]
//                                                           the first N references of a database
//   --variants scalar,packed16,tiled,tiled_swar,2input_lsb,msb,msb_exit1,msb_exit4,lsb
//   --exit-stats [--exit-samples S] [--database f.cuddl]    host-only early-exit estimate
//   --scale-1m, --ncu-small                                 as before
//
// Timing lines are JSON with "input" (synthetic or database), "source" (generator or path) and
// "n" (reference count).

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>
#include <cuddl/cuddl.cuh>
#include <cuddl/reference_database_file.cuh>
#include <cuddl/replicated_database.cuh>
#include <cuddl/result_file.cuh>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t k_buckets = 2048U;

enum class cmp_variant {
    scalar_row_major,
    packed16_row_major,
    tiled_packed16_row_major,
    tiled_swar16_row_major,
    bitsliced_2input_lsb,
    bitsliced_lop3_msb,
    bitsliced_lop3_msb_exit1,
    bitsliced_lop3_msb_exit4,
    bitsliced_lop3_lsb,
};

char const* variant_name(cmp_variant v) {
    switch (v) {
        case cmp_variant::scalar_row_major: return "scalar_row_major";
        case cmp_variant::packed16_row_major: return "packed16_row_major";
        case cmp_variant::tiled_packed16_row_major: return "tiled_packed16_row_major";
        case cmp_variant::tiled_swar16_row_major: return "tiled_swar16_row_major";
        case cmp_variant::bitsliced_2input_lsb: return "bitsliced_2input_lsb";
        case cmp_variant::bitsliced_lop3_msb: return "bitsliced_lop3_msb";
        case cmp_variant::bitsliced_lop3_msb_exit1: return "bitsliced_lop3_msb_exit1";
        case cmp_variant::bitsliced_lop3_msb_exit4: return "bitsliced_lop3_msb_exit4";
        case cmp_variant::bitsliced_lop3_lsb: return "bitsliced_lop3_lsb";
    }
    return "unknown";
}

std::vector<cmp_variant> all_variants() {
    return {
        cmp_variant::scalar_row_major,
        cmp_variant::packed16_row_major,
        cmp_variant::tiled_packed16_row_major,
        cmp_variant::tiled_swar16_row_major,
        cmp_variant::bitsliced_2input_lsb,
        cmp_variant::bitsliced_lop3_msb,
        cmp_variant::bitsliced_lop3_msb_exit1,
        cmp_variant::bitsliced_lop3_msb_exit4,
        cmp_variant::bitsliced_lop3_lsb,
    };
}

/// Short --variants names, indexed by cmp_variant.
constexpr char const* variant_alias[] = {
    "scalar", "packed16", "tiled", "tiled_swar", "2input_lsb",
    "msb", "msb_exit1", "msb_exit4", "lsb",
};

/// Parses a comma-separated list of short or full variant names.
std::vector<cmp_variant> parse_variants(std::string const& csv) {
    std::vector<cmp_variant> out;
    for (size_t pos = 0; pos <= csv.size();) {
        auto const end = std::min(csv.find(',', pos), csv.size());
        auto const token = csv.substr(pos, end - pos);
        pos = end + 1U;
        bool found = false;
        for (auto const v : all_variants()) {
            if (token == variant_alias[static_cast<size_t>(v)] || token == variant_name(v)) {
                out.push_back(v);
                found = true;
            }
        }
        if (!found) {
            std::fprintf(stderr, "unknown variant '%s'\n", token.c_str());
            std::exit(1);
        }
    }
    return out;
}

struct options {
    bool scale_1m = false;
    bool ncu_small = false;
    bool exit_stats = false;
    std::string database;
    uint32_t limit = 0U;
    uint32_t n = 100000U;
    int reps = 0;
    uint64_t exit_samples = 200000U;
    std::vector<cmp_variant> variants;
};

/// Row-major uint16 sketch rows, k_buckets per reference: the layout every variant consumes.
struct sketch_input {
    std::vector<uint16_t> rows;
    uint32_t count{};
    bool synthetic{true};
    std::string source;
};

std::vector<uint16_t> synthetic_rows(uint32_t n) {
    std::mt19937_64 rng(42ULL);
    std::vector<uint16_t> rows(static_cast<size_t>(n) * k_buckets);
    for (auto& row : rows) {
        uint64_t const r = rng();
        row = static_cast<uint16_t>(((9U + (r & 7U)) << 11U) | ((r >> 3U) & 0x7FFU));
    }
    return rows;
}

sketch_input make_synthetic_input(uint32_t n) {
    return {synthetic_rows(n), n, true, "mt19937_64(42)"};
}

/// Loads the first @p limit references (all when 0) of a .cuddl reference database.
sketch_input load_database(std::string const& path, uint32_t limit) {
    auto const file = CUDDL_UNWRAP(cuddl::reference_database_file::load(path));
    auto const metadata = file.metadata();
    if (metadata.compatibility.bucket_count != k_buckets) {
        std::fprintf(
            stderr, "%s has %u buckets, expected %zu\n", path.c_str(),
            static_cast<unsigned>(metadata.compatibility.bucket_count), k_buckets
        );
        std::exit(1);
    }
    uint32_t const n = limit != 0U ? std::min(limit, metadata.reference_count)
                                   : metadata.reference_count;
    if (n < 2U) {
        std::fprintf(stderr, "%s: need at least 2 references\n", path.c_str());
        std::exit(1);
    }
    auto const rows = file.rows();
    return {
        std::vector<uint16_t>(rows.begin(), rows.begin() + static_cast<size_t>(n) * k_buckets), n,
        false, path
    };
}

/// Quoted JSON string; paths hold no control characters.
std::string json_string(std::string const& s) {
    std::string out = "\"";
    for (char const c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out + '"';
}

/// Host estimate of how often the warp-uniform MSB-first exit of compare_plane_group can fire.
///
/// refine_all_pairs_kernel gives lane l the bucket groups j * 32 + l (32 consecutive buckets
/// each) and calls compare_plane_group once per j, so one sweep covers the 1,024 consecutive
/// buckets [1024 j, 1024 (j + 1)) of one pair. A sweep may stop after k planes (k a multiple
/// of the stride, k < 16) once every one of those buckets differs in its top k bits; an equal
/// bucket, empty ones included, forces all 16 planes.
void print_exit_stats(sketch_input const& in, uint64_t samples) {
    constexpr uint32_t planes = 16U;
    constexpr size_t sweep_buckets = 1024U;
    constexpr size_t sweeps_per_pair = k_buckets / sweep_buckets;
    // resolved_at[k]: sweeps whose buckets all differ within the top k planes, not fewer.
    std::array<uint64_t, planes + 1U> resolved_at{};
    uint64_t agree12 = 0U;
    uint64_t agree12_occupied = 0U;
    uint64_t occupied = 0U;
    std::mt19937_64 rng(0xE17ULL);
    for (uint64_t s = 0; s < samples; ++s) {
        auto const qi = rng() % in.count;
        auto ri = rng() % (in.count - 1U);
        if (ri >= qi) {
            ++ri;
        }
        auto const* q = in.rows.data() + qi * k_buckets;
        auto const* r = in.rows.data() + ri * k_buckets;
        for (size_t sweep = 0; sweep < sweeps_per_pair; ++sweep) {
            uint32_t min_width = planes;
            for (size_t b = sweep * sweep_buckets; b < (sweep + 1U) * sweep_buckets; ++b) {
                uint32_t const d = q[b] ^ r[b];
                bool const top12 = (d >> 4U) == 0U;
                bool const used = (q[b] | r[b]) != 0U;
                min_width = std::min(min_width, static_cast<uint32_t>(std::bit_width(d)));
                agree12 += top12;
                occupied += used;
                agree12_occupied += top12 && used;
            }
            // d's highest set bit sits in plane bit_width(d) - 1, so the MSB-first sweep sees a
            // difference after planes + 1 - bit_width(d) planes; the widest-needed bucket rules.
            uint32_t const need = std::min(planes, planes + 1U - min_width);
            ++resolved_at[need];
            if (s < 64U) {
                uint32_t slow = planes;
                for (uint32_t k = 1U; k <= planes && slow == planes; ++k) {
                    bool all_differ = true;
                    for (size_t b = sweep * sweep_buckets; b < (sweep + 1U) * sweep_buckets; ++b) {
                        all_differ &= ((q[b] ^ r[b]) >> (planes - k)) != 0U;
                    }
                    if (all_differ) {
                        slow = k;
                    }
                }
                if (slow != need) {
                    std::fprintf(stderr, "exit-stats self-check failed: %u != %u\n", slow, need);
                    std::exit(2);
                }
            }
        }
    }
    double const sweeps = static_cast<double>(samples * sweeps_per_pair);
    double const buckets = sweeps * static_cast<double>(sweep_buckets);
    // Exits happen at a multiple of the stride below the last plane.
    auto const saved = [&](uint32_t stride) {
        double total = 0.0;
        for (uint32_t need = 1U; need < planes; ++need) {
            uint32_t const exit_at = (need + stride - 1U) / stride * stride;
            if (exit_at < planes) {
                total += static_cast<double>(resolved_at[need]) * (planes - exit_at);
            }
        }
        return total / sweeps;
    };
    std::string cumulative;
    uint64_t running = 0U;
    for (uint32_t k = 1U; k < planes; ++k) {
        running += resolved_at[k];
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%s%.5f", k == 1U ? "" : ",", running / sweeps);
        cumulative += buf;
    }
    auto const cum = [&](uint32_t k) {
        uint64_t sum = 0U;
        for (uint32_t i = 1U; i <= k; ++i) {
            sum += resolved_at[i];
        }
        return static_cast<double>(sum) / sweeps;
    };
    std::printf(
        "{\"exit_stats\":{\"input\":\"%s\",\"source\":%s,\"n\":%u,\"samples\":%llu,"
        "\"sweeps\":%.0f,\"p_top12_agree\":%.6f,\"p_top12_agree_occupied\":%.6f,"
        "\"resolved_after_4\":%.5f,\"resolved_after_8\":%.5f,\"resolved_after_12\":%.5f,"
        "\"mean_planes_saved_stride1\":%.4f,\"mean_planes_saved_stride4\":%.4f,"
        "\"resolved_after_1_to_15\":[%s]}}\n",
        in.synthetic ? "synthetic" : "database", json_string(in.source).c_str(), in.count,
        static_cast<unsigned long long>(samples), sweeps, static_cast<double>(agree12) / buckets,
        occupied == 0U ? 0.0 : static_cast<double>(agree12_occupied) / static_cast<double>(occupied),
        cum(4U), cum(8U), cum(12U), saved(1U), saved(4U), cumulative.c_str()
    );
}

/// Unsigned q < r in each packed 16-bit lane, as 1 in bit 0 or 16, without __vcmpltu2 (which sm_90
/// emulates). Setting each lane's MSB in q and clearing it in r keeps the subtraction's borrows
/// inside the lane, so d's lane MSB is set iff q's low 15 bits are >= r's.
template <bool HandSwar>
__device__ __forceinline__ uint32_t lanes_lt16(uint32_t q, uint32_t r) {
    if constexpr (HandSwar) {
        constexpr uint32_t h = 0x80008000U;
        uint32_t const d = (q | h) - (r & ~h);
        return (((~q & r) | (~(q ^ r) & ~d)) & h) >> 15U;
    } else {
        return __vcmpltu2(q, r) & 0x00010001U;
    }
}

/// 1 in bit 0 or 16 for each zero 16-bit lane of x: adding 0x7fff to a lane's low 15 bits sets its
/// MSB iff they are nonzero.
__device__ __forceinline__ uint32_t swar_zero16(uint32_t x) {
    constexpr uint32_t h = 0x80008000U;
    return (~(((x & ~h) + ~h) | x) & h) >> 15U;
}

template <bool HandSwar>
__device__ __forceinline__ uint32_t lanes_eq16(uint32_t q, uint32_t r) {
    if constexpr (HandSwar) {
        return swar_zero16(q ^ r);
    } else {
        return __vcmpeq2(q, r) & 0x00010001U;
    }
}

/// Shared-memory query-tiled row-major packed 16-bit SIMD (__vcmpltu2/__vcmpeq2) baseline
/// matching refine_all_pairs_kernel's exact 24-query (96 KiB) tiling and 2-ref register reuse;
/// @p HandSwar replaces the intrinsics with hand-written borrow-isolated SWAR.
template <
    size_t BucketCount,
    typename SearchResult,
    bool UpperTriangle,
    uint32_t QueryGroup,
    bool HandSwar = false>
__global__ __launch_bounds__(cuddl::detail::all_pairs_block_size, 1)
void tiled_packed16_row_major_kernel(
    uint16_t const* rows,
    uint32_t query_id_offset,
    uint32_t query_count,
    uint32_t reference_count,
    SearchResult* results,
    uint32_t* cell_counter
) {
    constexpr uint32_t warp_width = 32U;
    constexpr uint32_t group = QueryGroup;
    constexpr uint32_t row_u32 = static_cast<uint32_t>(BucketCount / 2U);  // 1024 uint32s = 2048 u16
    constexpr uint32_t words_per_lane = row_u32 / warp_width;              // 32 uint32s = 64 buckets
    constexpr uint32_t max_block_words = static_cast<uint32_t>(
        cuddl::detail::refine_block_row_bytes / (BucketCount * sizeof(uint16_t)) / warp_width
    );
    constexpr uint32_t part_width = warp_width / cuddl::detail::all_pairs_word_parts;

    extern __shared__ uint32_t smem_queries[];
    __shared__ uint32_t next_word;
    __shared__ uint32_t claimed_cell;

    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const words_per_query = cuddl::detail::candidate_bit_words(reference_count);
    auto const cells = cuddl::detail::make_refine_cells<UpperTriangle>(
        query_id_offset, query_count, group, words_per_query, max_block_words, 1U
    );
    auto const groups = cells.groups;
    auto const* rows_u32 = reinterpret_cast<uint32_t const*>(rows);

    for (auto cell = cuddl::detail::next_cell(cell_counter, blockIdx.x - gridDim.x, claimed_cell);
         cell < cells.cell_count;
         cell = cuddl::detail::next_cell(cell_counter, cell, claimed_cell)) {
        auto const first_query = (cell % groups) * group;
        auto const group_size = cuda::std::min(group, query_count - first_query);
        auto const first_word = cells.first_word + (cell / groups) * cells.block_words;
        auto const last_word = cuda::std::min(first_word + cells.block_words, words_per_query);

        __syncthreads();
        if (threadIdx.x == 0U) {
            next_word = first_word * cuddl::detail::all_pairs_word_parts;
        }
        auto const* q_src =
            rows_u32 + static_cast<size_t>(query_id_offset + first_query) * row_u32;
        auto const total_u4 = group_size * (row_u32 / 4U);
        auto const* q_src4 = reinterpret_cast<uint4 const*>(q_src);
        auto* smem4 = reinterpret_cast<uint4*>(smem_queries);
        for (auto i = static_cast<uint32_t>(threadIdx.x); i < total_u4;
             i += cuddl::detail::all_pairs_block_size) {
            smem4[i] = q_src4[i];
        }
        __syncthreads();

        for (;;) {
            uint32_t word = 0U;
            if (lane == 0U) {
                word = atomicAdd(&next_word, 1U);
            }
            word = __shfl_sync(0xffffffffU, word, 0);
            auto const part = word % cuddl::detail::all_pairs_word_parts;
            word /= cuddl::detail::all_pairs_word_parts;
            if (word >= last_word) {
                break;
            }
            uint32_t bits = 0U;
            if (lane < group_size) {
                auto const first_ref = word * warp_width;
                auto const left = reference_count - first_ref;
                bits = left >= warp_width ? ~0U : (1U << left) - 1U;
                bits &= ((1U << part_width) - 1U) << (part * part_width);
                if constexpr (UpperTriangle) {
                    auto const qid = query_id_offset + first_query + lane;
                    if (first_ref <= qid) {
                        auto const skipped = qid - first_ref + 1U;
                        bits &= skipped >= warp_width ? 0U : ~0U << skipped;
                    }
                }
            }
            auto remaining = __reduce_or_sync(0xffffffffU, bits);
            while (remaining != 0U) {
                auto const bit_a = static_cast<uint32_t>(__ffs(remaining) - 1);
                remaining &= remaining - 1U;
                auto const has_b = remaining != 0U;
                auto const bit_b = has_b ? static_cast<uint32_t>(__ffs(remaining) - 1) : bit_a;
                remaining &= remaining - 1U;

                auto const ref_a = word * warp_width + bit_a;
                auto const ref_b = word * warp_width + bit_b;
                auto const* ra_ptr = rows_u32 + static_cast<size_t>(ref_a) * row_u32;
                auto const* rb_ptr = rows_u32 + static_cast<size_t>(ref_b) * row_u32;

                uint32_t ra[words_per_lane];
                uint32_t rb[words_per_lane];
                uint32_t a_zero = 0U;
                uint32_t b_zero = 0U;
                _Pragma("unroll")
                for (uint32_t w = 0; w < words_per_lane; ++w) {
                    uint32_t const idx = w * warp_width + lane;
                    ra[w] = __ldg(ra_ptr + idx);
                    a_zero |= __vcmpeq2(ra[w], 0U);
                    if (has_b) {
                        rb[w] = __ldg(rb_ptr + idx);
                        b_zero |= __vcmpeq2(rb[w], 0U);
                    }
                }
                bool const a_full = __all_sync(0xffffffffU, a_zero == 0U);
                bool const b_full = has_b ? __all_sync(0xffffffffU, b_zero == 0U) : true;

                auto const mask_a = __ballot_sync(0xffffffffU, (bits & (1U << bit_a)) != 0U);
                auto const mask_b =
                    has_b ? __ballot_sync(0xffffffffU, (bits & (1U << bit_b)) != 0U) : 0U;

                for (uint32_t m = mask_a | mask_b; m != 0U; m &= m - 1U) {
                    auto const q = static_cast<uint32_t>(__ffs(m) - 1);
                    auto const* q_ptr = smem_queries + static_cast<size_t>(q) * row_u32;
                    bool const do_a = (mask_a & (1U << q)) != 0U;
                    bool const do_b = (mask_b & (1U << q)) != 0U;

                    uint32_t lt_a = 0U, eq_a = 0U, bz_a = 0U;
                    uint32_t lt_b = 0U, eq_b = 0U, bz_b = 0U;
                    _Pragma("unroll")
                    for (uint32_t w = 0; w < words_per_lane; ++w) {
                        uint32_t const qw = q_ptr[w * warp_width + lane];
                        if (do_a) {
                            uint32_t const rw = ra[w];
                            lt_a += lanes_lt16<HandSwar>(qw, rw);
                            eq_a += lanes_eq16<HandSwar>(qw, rw);
                            if (!a_full) {
                                bz_a += lanes_eq16<HandSwar>(qw | rw, 0U);
                            }
                        }
                        if (do_b) {
                            uint32_t const rw = rb[w];
                            lt_b += lanes_lt16<HandSwar>(qw, rw);
                            eq_b += lanes_eq16<HandSwar>(qw, rw);
                            if (!b_full) {
                                bz_b += lanes_eq16<HandSwar>(qw | rw, 0U);
                            }
                        }
                    }
                    if (do_a) {
                        uint32_t const lo = (lt_a & 0xFFFFU) + (lt_a >> 16U);
                        uint32_t const bz = a_full ? 0U : ((bz_a & 0xFFFFU) + (bz_a >> 16U));
                        uint32_t const eq = (eq_a & 0xFFFFU) + (eq_a >> 16U) - bz;
                        uint32_t const hi = (2U * words_per_lane) - lo - eq - bz;
                        auto const tot_l = __reduce_add_sync(0xffffffffU, lo);
                        auto const tot_e = __reduce_add_sync(0xffffffffU, eq);
                        auto const tot_h = __reduce_add_sync(0xffffffffU, hi);
                        if (lane == 0U) {
                            results[cuddl::detail::batch_result_slot<UpperTriangle>(
                                first_query + q, ref_a, query_id_offset, reference_count
                            )] = SearchResult::pack(tot_l, tot_e, tot_h);
                        }
                    }
                    if (do_b) {
                        uint32_t const lo = (lt_b & 0xFFFFU) + (lt_b >> 16U);
                        uint32_t const bz = b_full ? 0U : ((bz_b & 0xFFFFU) + (bz_b >> 16U));
                        uint32_t const eq = (eq_b & 0xFFFFU) + (eq_b >> 16U) - bz;
                        uint32_t const hi = (2U * words_per_lane) - lo - eq - bz;
                        auto const tot_l = __reduce_add_sync(0xffffffffU, lo);
                        auto const tot_e = __reduce_add_sync(0xffffffffU, eq);
                        auto const tot_h = __reduce_add_sync(0xffffffffU, hi);
                        if (lane == 0U) {
                            results[cuddl::detail::batch_result_slot<UpperTriangle>(
                                first_query + q, ref_b, query_id_offset, reference_count
                            )] = SearchResult::pack(tot_l, tot_e, tot_h);
                        }
                    }
                }
            }
        }
    }
}

void launch_variant(
    cmp_variant v,
    uint32_t const* d_planes,
    uint16_t const* d_rows,
    uint32_t query_offset,
    uint32_t query_count,
    uint32_t ref_count,
    cuddl::packed_pairwise_counts* d_results,
    uint32_t* d_cell_counter,
    uint32_t sm_count,
    cuda::stream_ref stream
) {
    CUDDL_CUDA_CALL(cudaMemsetAsync(d_cell_counter, 0, sizeof(uint32_t), stream.get()));
    if (v == cmp_variant::scalar_row_major) {
        cuddl::detail::row_major_all_pairs_kernel<k_buckets, cuddl::packed_pairwise_counts, true, false>
            <<<sm_count * 8U, cuddl::detail::block_size, 0, stream.get()>>>(
                d_rows, query_offset, query_count, ref_count, d_results, nullptr
            );
    } else if (v == cmp_variant::packed16_row_major) {
        cuddl::detail::row_major_all_pairs_kernel<k_buckets, cuddl::packed_pairwise_counts, true, true>
            <<<sm_count * 8U, cuddl::detail::block_size, 0, stream.get()>>>(
                d_rows, query_offset, query_count, ref_count, d_results, nullptr
            );
    } else if (v == cmp_variant::tiled_packed16_row_major ||
               v == cmp_variant::tiled_swar16_row_major) {
        constexpr uint32_t group = cuddl::detail::all_pairs_query_group<k_buckets>;
        constexpr size_t smem = static_cast<size_t>(group) * k_buckets * sizeof(uint16_t);
        auto* k = v == cmp_variant::tiled_swar16_row_major
                      ? tiled_packed16_row_major_kernel<
                            k_buckets, cuddl::packed_pairwise_counts, true, group, true>
                      : tiled_packed16_row_major_kernel<
                            k_buckets, cuddl::packed_pairwise_counts, true, group, false>;
        CUDDL_CUDA_CALL(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(k),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)
        ));
        k<<<sm_count, cuddl::detail::all_pairs_block_size, smem, stream.get()>>>(
            d_rows, query_offset, query_count, ref_count, d_results, d_cell_counter
        );
    } else {
        constexpr uint32_t group = cuddl::detail::all_pairs_query_group<k_buckets>;
        constexpr size_t smem = cuddl::detail::all_pairs_dynamic_bytes<k_buckets>;
        uint32_t const* q_planes =
            d_planes + static_cast<size_t>(query_offset) * (k_buckets / 2U);
        auto launch_bs = [&](auto use_lop3, auto msb_first, auto exit_stride) {
            auto* k = cuddl::detail::refine_all_pairs_kernel<
                k_buckets,
                cuddl::packed_pairwise_counts,
                true,
                group,
                decltype(use_lop3)::value,
                decltype(msb_first)::value,
                decltype(exit_stride)::value>;
            CUDDL_CUDA_CALL(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(k),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)
            ));
            k<<<sm_count, cuddl::detail::all_pairs_block_size, smem, stream.get()>>>(
                q_planes, query_offset, query_count, d_planes, ref_count, d_results, nullptr,
                d_cell_counter, nullptr
            );
        };
        using no_exit = cuda::std::integral_constant<uint32_t, 0U>;
        using exit1 = cuda::std::integral_constant<uint32_t, 1U>;
        using exit4 = cuda::std::integral_constant<uint32_t, 4U>;
        if (v == cmp_variant::bitsliced_2input_lsb) {
            launch_bs(cuda::std::false_type{}, cuda::std::false_type{}, no_exit{});
        } else if (v == cmp_variant::bitsliced_lop3_msb) {
            launch_bs(cuda::std::true_type{}, cuda::std::true_type{}, no_exit{});
        } else if (v == cmp_variant::bitsliced_lop3_msb_exit1) {
            launch_bs(cuda::std::true_type{}, cuda::std::true_type{}, exit1{});
        } else if (v == cmp_variant::bitsliced_lop3_msb_exit4) {
            launch_bs(cuda::std::true_type{}, cuda::std::true_type{}, exit4{});
        } else {
            launch_bs(cuda::std::true_type{}, cuda::std::false_type{}, no_exit{});
        }
    }
    CUDDL_CUDA_CALL(cudaGetLastError());
}

/// Checks every variant against a host scalar oracle on the first @p n rows of @p rows.
void verify_variants(std::vector<uint16_t> const& rows, uint32_t n, char const* label) {
    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto d_planes = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), static_cast<size_t>(n) * (k_buckets / 2U), cuda::no_init
    );
    auto const plane_blocks = cuddl::detail::warp_grid_blocks(
        static_cast<size_t>(n) * (k_buckets / 32U)
    );
    cuddl::detail::build_score_planes_kernel<k_buckets>
        <<<plane_blocks, cuddl::detail::block_size, 0, stream.get()>>>(
            d_rows.data(), n, d_planes.data()
        );
    CUDDL_CUDA_CALL(cudaGetLastError());
    uint64_t const pairs = static_cast<uint64_t>(n) * (n - 1U) / 2ULL;
    auto d_res = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), pairs, cuda::no_init
    );
    auto d_ctr = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    auto const sm_count = static_cast<uint32_t>(
        stream.device().attribute(cuda::device_attributes::multiprocessor_count)
    );
    std::vector<cuddl::packed_pairwise_counts> h_res(pairs);

    for (auto v : all_variants()) {
        launch_variant(
            v, d_planes.data(), d_rows.data(), 0U, n, n, d_res.data(), d_ctr.data(), sm_count,
            stream
        );
        cuda::copy_bytes(
            stream, cuda::std::span{d_res.data(), pairs}, cuda::std::span{h_res.data(), pairs}
        );
        stream.sync();
        uint64_t slot = 0;
        for (uint32_t q = 0; q < n; ++q) {
            for (uint32_t r = q + 1U; r < n; ++r, ++slot) {
                uint32_t l = 0, e = 0, h = 0;
                for (size_t b = 0; b < k_buckets; ++b) {
                    uint16_t qs = rows[q * k_buckets + b];
                    uint16_t rs = rows[r * k_buckets + b];
                    if ((qs | rs) == 0U) continue;
                    if (qs < rs) ++l;
                    else if (qs == rs) ++e;
                    else ++h;
                }
                auto const got = h_res[slot].unpack(k_buckets);
                if (got.lower != l || got.equal != e || got.higher != h) {
                    std::fprintf(stderr, "VERIFY FAILED %s %s (%u,%u)\n",
                                 variant_name(v), label, q, r);
                    std::exit(2);
                }
            }
        }
        std::fprintf(stderr, "VERIFY OK %s %s pairs=%llu\n",
                     variant_name(v), label, static_cast<unsigned long long>(pairs));
    }
}

void verify_all_variants(bool sparse) {
    uint32_t const n = 64U;
    std::mt19937_64 rng(777ULL + (sparse ? 1U : 0U));
    std::vector<uint16_t> rows(static_cast<size_t>(n) * k_buckets, 0U);
    for (size_t i = 0; i < rows.size(); ++i) {
        uint64_t r = rng();
        if (sparse && (r & 3U) == 0U) continue;
        rows[i] = static_cast<uint16_t>(((9U + (r & 7U)) << 11U) | ((r >> 3U) & 0x7FFU));
    }
    verify_variants(rows, n, sparse ? "sparse" : "dense");
}

void verify_sparse_file_roundtrip() {
    uint32_t const n = 128U;
    std::mt19937_64 rng(999ULL);
    std::vector<uint16_t> rows(static_cast<size_t>(n) * k_buckets, 0U);
    for (uint32_t i = 0; i < n; ++i) {
        // Make ~25% of pairs share k-mers, ~75% disjoint so WKID == 0 is exercised heavily.
        uint32_t const cluster = i % 16U;
        for (size_t b = 0; b < k_buckets; ++b) {
            uint64_t const r = (b < 64U && (i / 16U) % 2U == 0U) ? (cluster * 10000ULL + b) : rng();
            rows[i * k_buckets + b] =
                static_cast<uint16_t>(((9U + (r & 7U)) << 11U) | ((r >> 3U) & 0x7FFU));
        }
    }
    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto const compat = cuddl::score_compatibility::current<25, k_buckets>();
    auto db = CUDDL_UNWRAP((cuddl::reference_database<25, k_buckets>::build_async(
        {d_rows.data(), d_rows.size()}, compat, stream
    )));
    stream.sync();
    std::vector<cuddl::reference_database<25, k_buckets>> dbs;
    dbs.push_back(std::move(db));
    auto group = CUDDL_UNWRAP((cuddl::replicated_database<25, k_buckets>::adopt(std::move(dbs))));

    auto const dir = std::filesystem::temp_directory_path();
    auto const dense_path = dir / "cuddl_test_dense_sim.bin";
    auto const sparse_path = dir / "cuddl_test_sparse_sim.bin";

    {
        auto f_dense = CUDDL_UNWRAP(cuddl::result_file::open(
            dense_path, {.format = cuddl::result_format::similarity, .streams = 1, .durable = false}
        ));
        CUDDL_UNWRAP(group.search_all_to_all(
            [&](cuddl::replica_tile const& t) { CUDDL_UNWRAP(f_dense.write(t)); }
        ));
        CUDDL_UNWRAP(f_dense.finish());
    }
    {
        auto f_sparse = CUDDL_UNWRAP(cuddl::result_file::open(
            sparse_path,
            {.format = cuddl::result_format::sparse_similarity, .streams = 1, .durable = false}
        ));
        CUDDL_UNWRAP(group.search_all_to_all(
            [&](cuddl::replica_tile const& t) { CUDDL_UNWRAP(f_sparse.write(t)); }
        ));
        CUDDL_UNWRAP(f_sparse.finish());
    }

    auto read_all = [](std::filesystem::path const& p) {
        std::ifstream in(p, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in), {});
    };
    auto const dense_bytes = read_all(dense_path);
    auto const sparse_bytes = read_all(sparse_path);
    std::filesystem::remove(dense_path);
    std::filesystem::remove(sparse_path);

    // Decode dense tiles and sparse tiles and verify bit-for-bit equality.
    std::vector<float> dense_vals;
    size_t pos = 0;
    while (pos < dense_bytes.size()) {
        uint32_t hdr[2];
        std::memcpy(hdr, dense_bytes.data() + pos, sizeof(hdr));
        pos += sizeof(hdr);
        uint32_t const q0 = hdr[0], qc = hdr[1];
        uint64_t const cnt =
            static_cast<uint64_t>(qc) * (n - q0 - 1U) - static_cast<uint64_t>(qc) * (qc - 1U) / 2U;
        size_t const old = dense_vals.size();
        dense_vals.resize(old + cnt);
        std::memcpy(dense_vals.data() + old, dense_bytes.data() + pos, cnt * sizeof(float));
        pos += cnt * sizeof(float);
    }

    std::vector<float> decoded_vals;
    pos = 0;
    uint64_t total_nz = 0;
    while (pos < sparse_bytes.size()) {
        uint32_t hdr[4];
        std::memcpy(hdr, sparse_bytes.data() + pos, sizeof(hdr));
        pos += sizeof(hdr);
        uint32_t const cnt = hdr[2], nz = hdr[3];
        total_nz += nz;
        uint32_t const bwords = (cnt + 31U) / 32U;
        auto const* bm = reinterpret_cast<uint32_t const*>(sparse_bytes.data() + pos);
        pos += bwords * sizeof(uint32_t);
        auto const* nzv = reinterpret_cast<float const*>(sparse_bytes.data() + pos);
        pos += nz * sizeof(float);
        uint32_t cursor = 0;
        for (uint32_t i = 0; i < cnt; ++i) {
            bool const bit = ((bm[i / 32U] >> (i % 32U)) & 1U) != 0U;
            decoded_vals.push_back(bit ? nzv[cursor++] : 0.0f);
        }
        if (cursor != nz) {
            std::fprintf(stderr, "SPARSE ROUNDTRIP COUNT MISMATCH %u != %u\n", cursor, nz);
            std::exit(2);
        }
    }
    if (dense_vals.size() != decoded_vals.size() ||
        std::memcmp(dense_vals.data(), decoded_vals.data(), dense_vals.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "SPARSE ROUNDTRIP BIT MISMATCH\n");
        std::exit(2);
    }
    std::fprintf(
        stderr,
        "VERIFY OK sparse_similarity roundtrip pairs=%zu nz=%llu dense_B=%zu sparse_B=%zu ratio=%.2fx\n",
        dense_vals.size(),
        static_cast<unsigned long long>(total_nz),
        dense_bytes.size(),
        sparse_bytes.size(),
        static_cast<double>(dense_bytes.size()) / static_cast<double>(sparse_bytes.size())
    );
}

void bench_variants(sketch_input const& in, int reps, std::vector<cmp_variant> const& list) {
    uint32_t const n_refs = in.count;
    auto const& rows = in.rows;
    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto d_planes = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), static_cast<size_t>(n_refs) * (k_buckets / 2U), cuda::no_init
    );
    auto const plane_blocks = cuddl::detail::warp_grid_blocks(
        static_cast<size_t>(n_refs) * (k_buckets / 32U)
    );
    cuddl::detail::build_score_planes_kernel<k_buckets>
        <<<plane_blocks, cuddl::detail::block_size, 0, stream.get()>>>(
            d_rows.data(), n_refs, d_planes.data()
        );
    CUDDL_CUDA_CALL(cudaGetLastError());
    uint64_t const max_tile_pairs = 1ULL << 25U;
    auto d_res = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), max_tile_pairs, cuda::no_init
    );
    auto d_ctr = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    auto const sm_count = static_cast<uint32_t>(
        stream.device().attribute(cuda::device_attributes::multiprocessor_count)
    );
    stream.sync();

    // Build full-row upper-triangle tiles of <= 2^25 pairs.
    std::vector<std::pair<uint32_t, uint32_t>> tiles;
    uint32_t q0 = 0;
    while (q0 + 1U < n_refs) {
        uint32_t qc = 0;
        uint64_t acc = 0;
        while (q0 + qc + 1U < n_refs) {
            uint64_t const row_pairs = n_refs - (q0 + qc) - 1U;
            if (qc > 0 && acc + row_pairs > max_tile_pairs) break;
            acc += row_pairs;
            ++qc;
        }
        tiles.emplace_back(q0, qc);
        q0 += qc;
    }
    uint64_t const total_pairs = static_cast<uint64_t>(n_refs) * (n_refs - 1U) / 2ULL;

    auto const source_json = json_string(in.source);
    for (auto v : list) {
        // Warmup first 4 tiles
        for (size_t t = 0; t < std::min<size_t>(4U, tiles.size()); ++t) {
            launch_variant(
                v, d_planes.data(), d_rows.data(), tiles[t].first, tiles[t].second, n_refs,
                d_res.data(), d_ctr.data(), sm_count, stream
            );
        }
        stream.sync();

        for (int rep = 0; rep < reps; ++rep) {
            auto const t0 = std::chrono::steady_clock::now();
            for (auto const& [first_q, q_cnt] : tiles) {
                launch_variant(
                    v, d_planes.data(), d_rows.data(), first_q, q_cnt, n_refs,
                    d_res.data(), d_ctr.data(), sm_count, stream
                );
            }
            stream.sync();
            auto const t1 = std::chrono::steady_clock::now();
            double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double const gpairs_s = (static_cast<double>(total_pairs) / 1e9) / (ms / 1e3);
            std::printf(
                "{\"variant\":\"%s\",\"input\":\"%s\",\"source\":%s,\"n\":%u,\"pairs\":%llu,"
                "\"rep\":%d,\"ms\":%.2f,\"gpairs_s\":%.3f}\n",
                variant_name(v), in.synthetic ? "synthetic" : "database", source_json.c_str(),
                n_refs, static_cast<unsigned long long>(total_pairs), rep, ms, gpairs_s
            );
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        std::string const arg = argv[i];
        auto const value = [&]() -> char const* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--scale-1m") {
            opt.scale_1m = true;
        } else if (arg == "--ncu-small") {
            opt.ncu_small = true;
        } else if (arg == "--exit-stats") {
            opt.exit_stats = true;
        } else if (arg == "--database") {
            opt.database = value();
        } else if (arg == "--limit") {
            opt.limit = static_cast<uint32_t>(std::stoul(value()));
        } else if (arg == "--n") {
            opt.n = static_cast<uint32_t>(std::stoul(value()));
        } else if (arg == "--reps") {
            opt.reps = std::stoi(value());
        } else if (arg == "--exit-samples") {
            opt.exit_samples = std::stoull(value());
        } else if (arg == "--variants") {
            opt.variants = parse_variants(value());
        } else {
            std::fprintf(
                stderr,
                "usage: %s [--database f.cuddl [--limit N]] [--n N] [--reps R] [--variants a,b]\n"
                "          [--exit-stats [--exit-samples S]] [--scale-1m] [--ncu-small]\n"
                "variants: scalar,packed16,tiled,tiled_swar,2input_lsb,msb,msb_exit1,msb_exit4,lsb\n",
                argv[0]
            );
            return 1;
        }
    }
    if (opt.ncu_small) {
        // Single-tile N=4096 launch for Nsight Compute dynamic instruction profiling.
        uint32_t const n = 4096U;
        auto const rows = synthetic_rows(n);
        cuda::stream stream{cuda::devices[0]};
        auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
        auto d_planes = cuda::make_device_buffer<uint32_t>(
            stream, stream.device(), static_cast<size_t>(n) * (k_buckets / 2U), cuda::no_init
        );
        auto const plane_blocks = cuddl::detail::warp_grid_blocks(
            static_cast<size_t>(n) * (k_buckets / 32U)
        );
        cuddl::detail::build_score_planes_kernel<k_buckets>
            <<<plane_blocks, cuddl::detail::block_size, 0, stream.get()>>>(
                d_rows.data(), n, d_planes.data()
            );
        uint64_t const pairs = static_cast<uint64_t>(n) * (n - 1U) / 2ULL;
        auto d_res = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
            stream, stream.device(), pairs, cuda::no_init
        );
        auto d_ctr = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
        auto const sm_count = static_cast<uint32_t>(
            stream.device().attribute(cuda::device_attributes::multiprocessor_count)
        );
        stream.sync();
        for (auto v : {cmp_variant::tiled_packed16_row_major, cmp_variant::bitsliced_lop3_msb,
                       cmp_variant::bitsliced_lop3_lsb}) {
            launch_variant(
                v, d_planes.data(), d_rows.data(), 0U, n, n, d_res.data(), d_ctr.data(), sm_count,
                stream
            );
            stream.sync();
        }
        return 0;
    }
    sketch_input db;
    if (!opt.database.empty()) {
        db = load_database(opt.database, opt.limit);
    }
    if (opt.exit_stats) {
        // Host only, no GPU needed. Synthetic rows are i.i.d., so a few thousand give the same
        // statistics as any larger count.
        if (!db.rows.empty()) {
            print_exit_stats(db, opt.exit_samples);
        }
        print_exit_stats(make_synthetic_input(4096U), opt.exit_samples);
        return 0;
    }

    verify_all_variants(false);
    verify_all_variants(true);
    if (!db.rows.empty()) {
        constexpr uint32_t verify_rows = 256U;
        auto const n = std::min(db.count, verify_rows);
        verify_variants(
            std::vector<uint16_t>(
                db.rows.begin(), db.rows.begin() + static_cast<size_t>(n) * k_buckets
            ),
            n,
            "database"
        );
    }
    verify_sparse_file_roundtrip();

    auto const list = !opt.variants.empty() ? opt.variants
                      : opt.scale_1m        ? std::vector{cmp_variant::bitsliced_lop3_lsb}
                                            : all_variants();
    auto const reps = [&](int fallback) { return opt.reps > 0 ? opt.reps : fallback; };
    if (!db.rows.empty()) {
        bench_variants(db, reps(3), list);
    } else if (opt.scale_1m) {
        for (auto const n : {1000000U, 1500000U}) {
            bench_variants(make_synthetic_input(n), reps(1), list);
        }
    } else {
        bench_variants(make_synthetic_input(opt.n), reps(3), list);
    }
    return 0;
}
