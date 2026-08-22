#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using cuddl::detail::hash_kmer;
using cuddl::detail::pack;
using cuddl::detail::restore;
using cuddl::detail::score;
using cuddl::detail::winner;

constexpr uint32_t k_default = 25;
constexpr size_t b_default = 2048;
constexpr uint16_t k_counter_max = 65535;

/// @brief Scalar, sequential CPU oracle for DDL construction and comparison.
template <size_t BucketCount>
struct scalar_sketch {
    std::vector<uint32_t> registers = std::vector<uint32_t>(BucketCount, 0U);
    std::vector<uint16_t> winners = std::vector<uint16_t>(BucketCount, 0U);
    std::vector<uint32_t> counts = std::vector<uint32_t>(BucketCount, 0U);
    bool saturated = false;

    void add(std::vector<uint64_t> const& inputs) {
        for (auto const input : inputs) {
            auto const h = hash_kmer(input);
            auto const b = h & (BucketCount - 1);
            auto const s = score(h);
            if (s > winners[b]) {
                winners[b] = s;
                counts[b] = 1;
            } else if (s == winners[b]) {
                if (counts[b] == k_counter_max) {
                    saturated = true;
                } else {
                    ++counts[b];
                }
            }
        }
    }

    void pack_registers() {
        for (size_t b = 0; b < BucketCount; ++b) {
            registers[b] = pack(winners[b], static_cast<uint16_t>(counts[b]));
        }
    }

    [[nodiscard]] cuddl::pairwise_counts compare(scalar_sketch const& other) const {
        cuddl::pairwise_counts out{};
        for (size_t b = 0; b < BucketCount; ++b) {
            auto const left = winner(this->registers[b]);
            auto const right = winner(other.registers[b]);
            if (left == 0 && right == 0) {
                ++out.both_empty;
            } else if (left < right) {
                ++out.lower;
            } else if (left > right) {
                ++out.higher;
            } else {
                ++out.equal;
            }
        }
        return out;
    }

    [[nodiscard]] double cardinality() const {
        uint64_t empty = 0;
        double sum = 0.0;
        for (size_t b = 0; b < BucketCount; ++b) {
            auto const w = winner(this->registers[b]);
            if (w == 0) {
                ++empty;
            } else {
                sum += restore(w);
            }
        }
        return cuddl::detail::cardinality(
            static_cast<double>(BucketCount), static_cast<double>(empty), sum
        );
    }
};

/// @brief Deterministic packed k-mers, generated in linear time.
///
/// Values come from a SplitMix64 stream over successive indices, masked into k-mer space. The
/// GPU and scalar oracles build from the same multiset, so byte-identical output does not depend
/// on uniqueness; duplicates only lower the effective cardinality slightly.
std::vector<uint64_t> make_inputs(size_t count, uint64_t seed = 0x1234'5678'9abc'def0ULL) {
    std::vector<uint64_t> out;
    out.reserve(count);
    auto const mask = (1ULL << (2 * k_default)) - 1;
    for (size_t i = 0; i < count; ++i) {
        out.push_back(cuddl::detail::splitmix64(seed + i) & mask);
    }
    return out;
}

class ReferenceDatabaseTest : public ::testing::Test {
   protected:
    void SetUp() override {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream_));
    }

    void TearDown() override {
        EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_));
    }

    cudaStream_t stream_{};
};

cuddl::pairwise_summary score_row_oracle(
    std::vector<uint16_t> const& query,
    std::vector<uint16_t> const& rows,
    size_t reference_id
) {
    cuddl::pairwise_summary summary{};
    auto const row_offset = reference_id * query.size();
    for (size_t bucket = 0; bucket < query.size(); ++bucket) {
        auto const query_score = query[bucket];
        auto const reference_score = rows[row_offset + bucket];
        if (query_score == 0U && reference_score == 0U) {
            ++summary.counts.both_empty;
        } else if (query_score < reference_score) {
            ++summary.counts.lower;
        } else if (query_score > reference_score) {
            ++summary.counts.higher;
        } else {
            ++summary.counts.equal;
        }
    }
    return summary;
}

TEST(SketchTest, RefIsTriviallyCopyableAndKBeforeBucket) {
    static_assert(std::is_trivially_copyable_v<cuddl::sketch_ref<25, 2048>>);
    static_assert(std::is_trivially_copyable_v<cuddl::sketch<25, 2048>> == false);
    using ref_t = cuddl::sketch_ref<25, 2048>;
    EXPECT_EQ(ref_t::bucket_count(), 2048);
    EXPECT_EQ(ref_t::kmer_length(), 25);
}

TEST(SketchTest, EmptyRegisterEncoding) {
    EXPECT_EQ(winner(0U), 0U);
    EXPECT_EQ(cuddl::detail::count(0U), 0U);
    // No non-empty score may collide with the empty sentinel.
    EXPECT_GT(cuddl::detail::score(hash_kmer(0)), 0U);
}

TEST(SketchTest, ScoreEncodingClampsExtremeNlz) {
    // A hash with the most significant bit set has NLZ 0 -> score in the lowest tier.
    uint16_t const low = cuddl::detail::score(0x8000'0000'0000'0000ULL);
    EXPECT_GT(low, 0U);
    // A zero hash has NLZ 62 (clamped) -> top tier, and restore stays finite (no shift underflow).
    uint16_t const high = cuddl::detail::score(0ULL);
    EXPECT_GE(high, 1U << cuddl::detail::mantissa_bits);
    uint64_t const restored = restore(high);
    EXPECT_GT(restored, 0ULL);
    (void)low;
}

TEST(SketchTest, GpuRegistersMatchScalarOracleByteIdentically) {
    auto const inputs = make_inputs(50000);
    cuddl::sketch<k_default, b_default> gpu;
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}).has_value());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    // Copy GPU registers out via the owning sketch's pointer.
    cuddl::sketch_ref<k_default, b_default> ref = gpu.ref();
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            gpu_regs.data(), ref.data().data(), b_default * sizeof(uint32_t), cudaMemcpyDeviceToHost
        )
    );
    EXPECT_EQ(gpu_regs, oracle.registers);
}

TEST(SketchTest, ComparisonCountsMatchScalarOracle) {
    auto const a = make_inputs(40000, 0x1111'1111'1111'1111ULL);
    auto const b = make_inputs(30000, 0x2222'2222'2222'2222ULL);
    // Overlap half of A into B so the pair shares a large common prefix of k-mers.
    std::vector<uint64_t> shared(a.begin(), a.begin() + 20000);
    std::vector<uint64_t> b_joined = shared;
    b_joined.insert(b_joined.end(), b.begin(), b.end());

    cuddl::sketch<k_default, b_default> gpu_a;
    cuddl::sketch<k_default, b_default> gpu_b;
    ASSERT_TRUE(gpu_a.add({a.data(), a.size()}).has_value());
    ASSERT_TRUE(gpu_b.add({b_joined.data(), b_joined.size()}).has_value());

    auto const gpu_summary = gpu_a.compare(gpu_b.ref());
    ASSERT_TRUE(gpu_summary.has_value());

    scalar_sketch<b_default> oracle_a;
    oracle_a.add(a);
    oracle_a.pack_registers();
    scalar_sketch<b_default> oracle_b;
    oracle_b.add(b_joined);
    oracle_b.pack_registers();
    auto const oracle_counts = oracle_a.compare(oracle_b);

    EXPECT_EQ(gpu_summary->counts, oracle_counts);
    EXPECT_EQ(
        gpu_summary->counts.lower + gpu_summary->counts.equal + gpu_summary->counts.higher +
            gpu_summary->counts.both_empty,
        static_cast<size_t>(b_default)
    );
}

TEST_F(ReferenceDatabaseTest, ExhaustiveSearchMatchesScalarOracle) {
    constexpr size_t reference_count = 4;
    auto const compatibility =
        cuddl::score_compatibility::current<k_default, b_default>();

    std::vector<uint16_t> query(b_default);
    std::vector<uint16_t> rows(reference_count * b_default);
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        query[bucket] = std::array<uint16_t, 4>{0U, 7U, 5U, 9U}[bucket % 4];
        rows[bucket] = 0U;
        rows[b_default + bucket] = query[bucket];
        rows[2 * b_default + bucket] = query[bucket] == 0U ? 3U : query[bucket] + 1U;
        rows[3 * b_default + bucket] = query[bucket] == 0U ? 0U : query[bucket] - 1U;
    }

    thrust::device_vector<uint16_t> device_rows(rows);
    thrust::device_vector<uint16_t> device_query(query);
    thrust::device_vector<cuddl::reference_search_result> device_results(reference_count);
    thrust::device_vector<uint8_t> workspace;
    auto const stream = cuda::stream_ref{stream_};

    auto built = cuddl::reference_database<k_default, b_default>::build_async(
        device_rows, compatibility, stream
    );
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    ASSERT_TRUE(
        database
            .search_async(device_query, compatibility, workspace, device_results, stream)
            .has_value()
    );

    std::vector<cuddl::reference_search_result> results(reference_count);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            results.data(),
            thrust::raw_pointer_cast(device_results.data()),
            results.size() * sizeof(results.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        EXPECT_EQ(results[reference_id].reference_id, reference_id);
        EXPECT_EQ(results[reference_id].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, EmptyDatabaseReportsSizesAndReturnsNoResults) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    static_assert(std::is_trivially_copyable_v<typename database_type::ref_type>);
    static_assert(!std::is_trivially_copyable_v<database_type>);

    auto const compatibility =
        cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};
    auto built = database_type::build_async(
        cuddl::device_span<uint16_t const>{}, compatibility, stream
    );
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    EXPECT_EQ(database.reference_count(), 0U);
    EXPECT_EQ(database.metadata().compatibility, compatibility);
    EXPECT_EQ(database.persistent_row_bytes(), 0U);
    EXPECT_EQ(database.single_query_workspace_bytes(), 0U);
    EXPECT_EQ(database.single_query_result_count(), 0U);
    EXPECT_EQ(database_type::persistent_row_bytes(3), 3 * b_default * sizeof(uint16_t));
    EXPECT_EQ(database_type::single_query_workspace_bytes(3), 0U);
    EXPECT_EQ(database_type::single_query_result_count(3), 3U);

    thrust::device_vector<uint16_t> query(b_default, 0U);
    auto const searched = database.search_async(
        {thrust::raw_pointer_cast(query.data()), query.size()}, compatibility, {}, {}, stream
    );
    ASSERT_TRUE(searched.has_value()) << searched.error().message();
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
}

TEST_F(ReferenceDatabaseTest, RejectsMalformedAndIncompatibleInputsWithoutOutput) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility =
        cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    thrust::device_vector<uint16_t> malformed_rows(b_default + 1, 1U);
    auto malformed = database_type::build_async(
        {thrust::raw_pointer_cast(malformed_rows.data()), malformed_rows.size()},
        compatibility,
        stream
    );
    ASSERT_FALSE(malformed.has_value());
    EXPECT_EQ(malformed.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto unsupported = compatibility;
    unsupported.key_mask = 0x7fffU;
    auto unsupported_build = database_type::build_async(
        cuddl::device_span<uint16_t const>{}, unsupported, stream
    );
    ASSERT_FALSE(unsupported_build.has_value());
    EXPECT_EQ(unsupported_build.error().category(), cuddl::ErrorCategory::invalid_argument);

    thrust::device_vector<uint16_t> rows(2 * b_default, 4U);
    thrust::device_vector<uint16_t> query(b_default, 4U);
    auto built = database_type::build_async(
        {thrust::raw_pointer_cast(rows.data()), rows.size()}, compatibility, stream
    );
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    cuddl::reference_search_result sentinel{};
    sentinel.reference_id = 99U;
    sentinel.summary.counts.lower = 123U;
    thrust::device_vector<cuddl::reference_search_result> results(1, sentinel);

    auto malformed_query = database.search_async(
        {thrust::raw_pointer_cast(query.data()), query.size() - 1},
        compatibility,
        {},
        {thrust::raw_pointer_cast(results.data()), results.size()},
        stream
    );
    ASSERT_FALSE(malformed_query.has_value());
    EXPECT_EQ(malformed_query.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto incompatible = compatibility;
    ++incompatible.hash_seed;
    auto incompatible_query = database.search_async(
        {thrust::raw_pointer_cast(query.data()), query.size()},
        incompatible,
        {},
        {thrust::raw_pointer_cast(results.data()), results.size()},
        stream
    );
    ASSERT_FALSE(incompatible_query.has_value());
    EXPECT_EQ(incompatible_query.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto insufficient = database.search_async(
        {thrust::raw_pointer_cast(query.data()), query.size()},
        compatibility,
        {},
        {thrust::raw_pointer_cast(results.data()), results.size()},
        stream
    );
    ASSERT_FALSE(insufficient.has_value());
    EXPECT_EQ(insufficient.error().category(), cuddl::ErrorCategory::resource);

    cuddl::reference_search_result observed{};
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &observed,
            thrust::raw_pointer_cast(results.data()),
            sizeof(observed),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    EXPECT_EQ(observed, sentinel);
}

TEST(SketchTest, CardinalityMatchesScalarOracleAcrossMagnitudes) {
    for (size_t n : {0U, 1U, 2U, 8U, 64U, 512U, 4096U, 1u << 16, 1u << 22}) {
        auto const inputs = make_inputs(n, 0x3333'3333'3333'3333ULL + n);
        cuddl::sketch<k_default, b_default> gpu;
        if (n > 0) {
            ASSERT_TRUE(gpu.add({inputs.data(), n}).has_value());
        }
        auto const gpu_cardinality = gpu.cardinality();
        ASSERT_TRUE(gpu_cardinality.has_value());

        scalar_sketch<b_default> oracle;
        oracle.add(inputs);
        oracle.pack_registers();
        auto const oracle_cardinality = oracle.cardinality();

        EXPECT_TRUE(std::isfinite(*gpu_cardinality));
        if (n == 0) {
            EXPECT_EQ(*gpu_cardinality, 0.0);
        }
        EXPECT_NEAR(*gpu_cardinality, oracle_cardinality, 1e-6 * std::max(1.0, oracle_cardinality));
    }
}

// Truth-anchored check: the estimator's absolute value must approach the true distinct cardinality,
// not the per-bucket ratio (`truth / BucketCount`). This guards the global bucket-count factor in
// MeanM, which the scalar-oracle test (shares MeanM) cannot catch.
TEST(SketchTest, CardinalityApproachesTrueDistinctCount) {
    for (size_t n : {1u << 16, 1u << 18, 1u << 20}) {
        auto const inputs = make_inputs(n, 0x7777'7777'7777'7777ULL + n);
        // make_inputs from a 64-bit SplitMix stream masked into 2^50 space; collisions are
        // negligible at these sizes, so the distinct count is ~n.
        cuddl::sketch<k_default, b_default> gpu;
        ASSERT_TRUE(gpu.add({inputs.data(), n}).has_value());
        auto const card = gpu.cardinality();
        ASSERT_TRUE(card.has_value());
        auto const estimate = *card;
        EXPECT_GT(estimate, 0.0);
        // Without the global bucket-count factor the estimate would be ~n / 2048.
        EXPECT_NEAR(estimate, static_cast<double>(n), 0.1 * static_cast<double>(n))
            << "cardinality should approach the true distinct count, got " << estimate;
        EXPECT_GT(estimate, static_cast<double>(n) / 2.0);
    }
}

TEST(SketchTest, CardinalityRemainsAccurateThroughSparseTransition) {
    for (size_t n : {512U, 1024U, 2048U, 4096U, 8192U}) {
        auto const inputs = make_inputs(n, 0x5555'5555'5555'5555ULL + n);
        cuddl::sketch<k_default, b_default> gpu;
        ASSERT_TRUE(gpu.add({inputs.data(), n}).has_value());
        auto const estimate = gpu.cardinality();
        ASSERT_TRUE(estimate.has_value());
        EXPECT_NEAR(*estimate, static_cast<double>(n), 0.1 * static_cast<double>(n));
    }
}

TEST(SketchTest, WinnerCountsAndSaturationMatchScalarOracle) {
    // A repeated k-mer so a single bucket sees many equal observations.
    auto const packed_kmer = 0x1234'5678'9abc'def0ULL & ((1ULL << (2 * k_default)) - 1);
    std::vector<uint64_t> inputs;
    auto const repeat = 65536U;
    inputs.assign(repeat, packed_kmer);

    cuddl::sketch<k_default, b_default> gpu;
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}).has_value());
    auto const gpu_wc = gpu.winner_counts();
    ASSERT_TRUE(gpu_wc.has_value());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    EXPECT_EQ(gpu_wc->second, oracle.saturated);
    ASSERT_EQ(gpu_wc->first.size(), oracle.registers.size());
    for (size_t b = 0; b < b_default; ++b) {
        EXPECT_EQ(gpu_wc->first[b], cuddl::detail::count(oracle.registers[b])) << "bucket " << b;
    }
}

TEST(SketchTest, SaturationFlagSetAtCounterOverflow) {
    auto const packed_kmer = 0x0000'0000'0000'00ffULL & ((1ULL << (2 * k_default)) - 1);

    cuddl::sketch<k_default, b_default> under;
    std::vector<uint64_t> few(1000, packed_kmer);
    ASSERT_TRUE(under.add({few.data(), few.size()}).has_value());
    auto const under_wc = under.winner_counts();
    ASSERT_TRUE(under_wc.has_value());
    EXPECT_FALSE(under_wc->second);

    cuddl::sketch<k_default, b_default> over;
    std::vector<uint64_t> many(65536, packed_kmer);
    ASSERT_TRUE(over.add({many.data(), many.size()}).has_value());
    auto const over_wc = over.winner_counts();
    ASSERT_TRUE(over_wc.has_value());
    EXPECT_TRUE(over_wc->second);
}

TEST(SketchTest, HostMetricsOnRawPair) {
    auto const a = make_inputs(40000, 0x4444'4444'4444'4444ULL);
    auto const c = make_inputs(40000, 0x5555'5555'5555'5555ULL);

    cuddl::sketch<k_default, b_default> sa;
    cuddl::sketch<k_default, b_default> sb;
    cuddl::sketch<k_default, b_default> sc;
    ASSERT_TRUE(sa.add({a.data(), a.size()}).has_value());
    ASSERT_TRUE(sb.add({a.data(), a.size()}).has_value());
    ASSERT_TRUE(sc.add({c.data(), c.size()}).has_value());

    auto const same_result = sa.compare(sb.ref());
    ASSERT_TRUE(same_result.has_value());
    auto const same = *same_result;
    auto const ref_same = sa.ref();
    auto const wkid_same = ref_same.wkid(same);
    auto const ani_same = ref_same.ani(same);
    auto const cont_same = ref_same.containment(same);
    auto const comp_same = ref_same.completeness(same);
    ASSERT_TRUE(wkid_same.has_value());
    ASSERT_TRUE(ani_same.has_value());
    ASSERT_TRUE(cont_same.has_value());
    ASSERT_TRUE(comp_same.has_value());
    EXPECT_DOUBLE_EQ(*wkid_same, 1.0);
    EXPECT_DOUBLE_EQ(*ani_same, 1.0);
    EXPECT_GT(*cont_same, 0.9);
    EXPECT_GT(*comp_same, 0.9);

    auto const disjoint_result = sa.compare(sc.ref());
    ASSERT_TRUE(disjoint_result.has_value());
    auto const disjoint = *disjoint_result;
    auto const wkid_disjoint = sa.ref().wkid(disjoint);
    ASSERT_TRUE(wkid_disjoint.has_value());
    EXPECT_LT(*wkid_disjoint, 0.05);
}

TEST(SketchTest, WkidPadsSparseDivisors) {
    cuddl::sketch<k_default, b_default> sketch;
    cuddl::pairwise_summary sparse{};
    sparse.counts.equal = 1;
    sparse.counts.lower = 2;
    sparse.counts.higher = 4;

    auto const wkid = sketch.ref().wkid(sparse);
    ASSERT_TRUE(wkid.has_value());
    EXPECT_DOUBLE_EQ(*wkid, 1.0 / 6.0);
}

TEST(ResultTest, ErrorCategoryEnumMatchesVariantOrder) {
    EXPECT_EQ(cuddl::Error{}.category(), cuddl::ErrorCategory::cuda);
    EXPECT_EQ(
        cuddl::Error::invalid_argument("x").category(), cuddl::ErrorCategory::invalid_argument
    );
    EXPECT_EQ(cuddl::Error::resource("x").category(), cuddl::ErrorCategory::resource);
    EXPECT_EQ(cuddl::Error::invalid_argument("x").as_invalid_argument()->message, "x");
}

std::string write_tmp_fasta(std::string const& content) {
    static int counter = 0;
    auto const path = std::string("/tmp/cuddl_fasta_test_") + std::to_string(++counter) + ".fa";
    std::ofstream out(path, std::ios::binary);
    out << content;
    out.close();
    return path;
}

TEST(FastaTest, ParsesSimpleSeedIntoExpectedKmers) {
    // 8 bases, k=3: valid windows = N-k+1 = 6.
    auto const path = write_tmp_fasta(">seq\nACGTACGT\n");
    auto const res = cuddl::parse_fasta_file(path, 3);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->bases, 8u);
    EXPECT_EQ(res->valid_kmers, 6u);
    EXPECT_EQ(res->invalid_windows, 0u);
    std::remove(path.c_str());
}

TEST(FastaTest, InvalidBaseBreaksRollingWindow) {
    // ACGTNACGT with k=3: the N follows a full window, so it breaks the run but not a partial
    // window. No k-mer ever spans the N.
    auto const path = write_tmp_fasta(">seq\nACGTNACGT\n");
    auto const res = cuddl::parse_fasta_file(path, 3);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->bases, 8u);        // two ACGT runs
    EXPECT_EQ(res->valid_kmers, 4u);  // one valid window per 4-base run: (4-3+1)=2 each
    EXPECT_EQ(res->invalid_windows, 0u);
    std::remove(path.c_str());
}

TEST(FastaTest, InvalidBaseBreaksPartialWindow) {
    // k=5, "ACGNNACGT": the first N breaks the partial ACG window (1 invalid); the longest valid
    // run (ACGT) is shorter than k, so no k-mer is emitted.
    auto const path = write_tmp_fasta(">seq\nACGNNACGT\n");
    auto const res = cuddl::parse_fasta_file(path, 5);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->bases, 7u);  // ACG + ACGT
    EXPECT_EQ(res->valid_kmers, 0u);
    EXPECT_EQ(res->invalid_windows, 1u);
    std::remove(path.c_str());
}

TEST(FastaTest, NoKmerSpansInvalidBase) {
    // k=2 over "ATNGT": the window resets at N; emitted k-mers are the canonical forms of AT and
    // GT.
    auto const path = write_tmp_fasta(">s\nATNGT\n");
    auto const res = cuddl::parse_fasta_file(path, 2);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->valid_kmers, 2u);
    // AT: forward (A=0,T=2)=2, RC("AT")="AT"=2, canonical=max=2.
    // GT: forward (G=3,T=2)=14, RC("GT")="AC"=(0<<2)|1=1, canonical=14.
    std::vector<uint64_t> const expected{2, 14};
    EXPECT_EQ(res->kmers, expected);
    std::remove(path.c_str());
}

TEST(FastaTest, CanonicalOrientationIsMax) {
    auto const path = write_tmp_fasta(">s\nACGTAC\n");
    auto const res = cuddl::parse_fasta_file(path, 3);
    ASSERT_TRUE(res.has_value());
    for (auto const kmer : res->kmers) {
        auto const rev = cuddl::detail::reverse_complement(kmer, 3);
        EXPECT_GE(kmer, rev);
    }
    std::remove(path.c_str());
}

TEST(FastaTest, HeaderAtEndOfFileWithoutNewline) {
    // A trailing '>' header as the final byte must not read past the end of the buffer.
    auto const path = write_tmp_fasta(">a\nACGT\n>");
    auto const res = cuddl::parse_fasta_file(path, 3);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->bases, 4u);
    EXPECT_EQ(res->valid_kmers, 2u);
    std::remove(path.c_str());
}

TEST(FastaTest, EmptyFileParsesToEmptyResult) {
    auto const path = write_tmp_fasta("");
    auto const res = cuddl::parse_fasta_file(path, 3);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res->bases, 0u);
    EXPECT_EQ(res->valid_kmers, 0u);
    std::remove(path.c_str());
}

}  // namespace
