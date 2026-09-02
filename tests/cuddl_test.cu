#include <cuddl/a48.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>
#include <cuddl/refseq_parity.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
using cuddl::detail::restore_midpoint;
using cuddl::detail::score;
using cuddl::detail::winner;

constexpr uint32_t k_default = 25;
constexpr size_t b_default = 2048;
constexpr uint16_t k_counter_max = 65535;

/// @brief Scalar, sequential CPU oracle for DDL construction and comparison.
template <size_t BucketCount, typename Layout = cuddl::default_register_layout>
struct scalar_sketch {
    std::vector<uint32_t> registers = std::vector<uint32_t>(BucketCount, 0U);
    std::vector<uint16_t> winners = std::vector<uint16_t>(BucketCount, 0U);
    std::vector<uint32_t> counts = std::vector<uint32_t>(BucketCount, 0U);
    bool saturated = false;

    void add(std::vector<uint64_t> const& inputs) {
        for (auto const input : inputs) {
            auto const h = hash_kmer(input);
            auto const b = h & (BucketCount - 1);
            auto const s = score<Layout>(h);
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
                sum += restore_midpoint<Layout>(w);
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

template <typename T>
bool copy_device_vector(thrust::device_vector<T> const& source, std::vector<T>& destination) {
    destination.resize(source.size());
    if (destination.empty()) {
        return true;
    }
    return cudaMemcpy(
               destination.data(),
               thrust::raw_pointer_cast(source.data()),
               destination.size() * sizeof(T),
               cudaMemcpyDeviceToHost
           ) == cudaSuccess;
}

TEST(SketchTest, PublicMetadataUsesKBeforeBucket) {
    using sketch_type = cuddl::sketch<25, 2048>;
    static_assert(!std::is_trivially_copyable_v<sketch_type>);
    EXPECT_EQ(sketch_type::bucket_count(), 2048);
    EXPECT_EQ(sketch_type::kmer_length(), 25);
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
    // The midpoint refinement shifts ordinary tiers upward; the clamped top tier is already
    // exact and must stay unchanged.
    EXPECT_GT(restore_midpoint(low), restore(low));
    EXPECT_EQ(restore_midpoint(high), restore(high));
    (void)low;
}

TEST(SketchTest, FiveExponentElevenMantissaLayoutHasNoRuntimeState) {
    using layout = cuddl::register_layout<5, 11>;
    using sketch_type = cuddl::sketch<k_default, b_default, layout>;
    static_assert(std::is_empty_v<layout>);
    static_assert(std::is_same_v<typename sketch_type::register_type, uint32_t>);
    EXPECT_EQ(score<layout>(0ULL), std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(restore<layout>(std::numeric_limits<uint16_t>::max()), 1ULL << 32U);

    auto const inputs = make_inputs(50000);
    cuddl::sketch<k_default, b_default, layout> gpu;
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}).has_value());

    scalar_sketch<b_default, layout> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            gpu_regs.data(), gpu.data().data(), b_default * sizeof(uint32_t), cudaMemcpyDeviceToHost
        )
    );
    EXPECT_EQ(gpu_regs, oracle.registers);

    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default, layout>();
    EXPECT_EQ(compatibility.exponent_bits, 5U);
    EXPECT_EQ(compatibility.mantissa_bits, 11U);
    EXPECT_TRUE(gpu.cardinality().has_value());
    EXPECT_TRUE(gpu.hybrid_cardinality().has_value());
    EXPECT_TRUE(gpu.summary<true>(gpu).has_value());
}

TEST(SketchTest, GpuRegistersMatchScalarOracleByteIdentically) {
    auto const inputs = make_inputs(50000);
    cuddl::sketch<k_default, b_default> gpu;
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}).has_value());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            gpu_regs.data(), gpu.data().data(), b_default * sizeof(uint32_t), cudaMemcpyDeviceToHost
        )
    );
    EXPECT_EQ(gpu_regs, oracle.registers);
}

TEST(SketchTest, UnalignedInputSpanningMultipleGridStrideIterationsMatchesScalarOracle) {
    // An element-aligned but not 32-byte-aligned span forces the scalar path of
    // add_shared_kernel; the span must exceed one grid-stride iteration so an early exit would
    // drop inputs and diverge from the oracle. To make a dropped tail observable, the first
    // stride's worth of items avoids one bucket entirely and the eight items beyond the stride
    // are guaranteed winners in that bucket: a buggy kernel that stops after its first chunk
    // leaves the bucket empty.
    int multiprocessors = 0;
    ASSERT_EQ(
        cudaSuccess, cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0)
    );
    ASSERT_GT(multiprocessors, 0);
    auto const stride = size_t{2} * static_cast<size_t>(multiprocessors) *
                        static_cast<size_t>(cuddl::detail::shared_construction_block_size) * 4U;
    auto const mask = (1ULL << (2 * k_default)) - 1;
    auto const tail_value = 0xDEAD'BEEF'C0FF'EE00ULL & mask;
    auto const tail_bucket = cuddl::detail::hash_kmer(tail_value) & (b_default - 1);
    std::vector<uint64_t> inputs;
    inputs.reserve(stride + 8U);
    for (size_t kept = 0, attempt = 0; kept < stride; ++attempt) {
        auto const value = cuddl::detail::splitmix64(0x1234'5678'9abc'def0ULL + attempt) & mask;
        if ((cuddl::detail::hash_kmer(value) & (b_default - 1)) != tail_bucket) {
            inputs.push_back(value);
            ++kept;
        }
    }
    for (size_t i = 0; i < 8U; ++i) {
        inputs.push_back(tail_value);
    }
    thrust::device_vector<uint64_t> padded(inputs.size() + 1U);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            thrust::raw_pointer_cast(padded.data()) + 1,
            inputs.data(),
            inputs.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice
        )
    );
    auto const unaligned = cuddl::device_span<uint64_t const>{
        thrust::raw_pointer_cast(padded.data()) + 1, inputs.size()
    };

    cuddl::sketch<k_default, b_default> gpu;
    ASSERT_TRUE(gpu.add_async(unaligned, cudaStream_t{nullptr}).has_value());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpy(
            gpu_regs.data(), gpu.data().data(), b_default * sizeof(uint32_t), cudaMemcpyDeviceToHost
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

    auto const gpu_summary = gpu_a.compare(gpu_b);
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

TEST(SketchTest, BatchComparisonMatchesScalarOracle) {
    constexpr size_t pair_count = 3;
    std::vector<uint16_t> left_scores(pair_count * b_default);
    std::vector<uint16_t> right_scores(pair_count * b_default);
    std::vector<uint32_t> left_registers(pair_count * b_default);
    std::vector<uint32_t> right_registers(pair_count * b_default);
    for (size_t pair = 0; pair < pair_count; ++pair) {
        for (size_t bucket = 0; bucket < b_default; ++bucket) {
            auto const index = pair * b_default + bucket;
            left_scores[index] = (bucket + pair) % 11U == 0U
                                     ? 0U
                                     : static_cast<uint16_t>((bucket * 13U + pair) % 251U + 1U);
            right_scores[index] =
                (bucket + 2U * pair) % 13U == 0U
                    ? 0U
                    : static_cast<uint16_t>((bucket * 17U + pair * 3U) % 251U + 1U);
            left_registers[index] = pack(left_scores[index], left_scores[index] == 0U ? 0U : 1U);
            right_registers[index] = pack(right_scores[index], right_scores[index] == 0U ? 0U : 1U);
        }
    }

    thrust::device_vector<uint32_t> device_left(left_registers);
    thrust::device_vector<uint32_t> device_right(right_registers);
    thrust::device_vector<cuddl::pairwise_summary> device_outputs(pair_count);
    auto const result = cuddl::compare_batch_async<b_default>(
        {thrust::raw_pointer_cast(device_left.data()), device_left.size()},
        {thrust::raw_pointer_cast(device_right.data()), device_right.size()},
        {thrust::raw_pointer_cast(device_outputs.data()), device_outputs.size()}
    );
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<cuddl::pairwise_summary> outputs;
    ASSERT_TRUE(copy_device_vector(device_outputs, outputs));
    for (size_t pair = 0; pair < pair_count; ++pair) {
        auto const begin = left_scores.begin() + static_cast<ptrdiff_t>(pair * b_default);
        std::vector<uint16_t> left_row(begin, begin + b_default);
        EXPECT_EQ(outputs[pair], score_row_oracle(left_row, right_scores, pair));
    }

    auto const unequal = cuddl::compare_batch_async<b_default>(
        {thrust::raw_pointer_cast(device_left.data()), device_left.size()},
        {thrust::raw_pointer_cast(device_right.data()), device_right.size() - 1U},
        {thrust::raw_pointer_cast(device_outputs.data()), device_outputs.size()}
    );
    EXPECT_FALSE(unequal.has_value());
    EXPECT_EQ(unequal.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto const undersized = cuddl::compare_batch_async<b_default>(
        {thrust::raw_pointer_cast(device_left.data()), device_left.size()},
        {thrust::raw_pointer_cast(device_right.data()), device_right.size()},
        {thrust::raw_pointer_cast(device_outputs.data()), pair_count - 1U}
    );
    EXPECT_FALSE(undersized.has_value());
    auto const null_input = cuddl::compare_batch_async<b_default>(
        cuddl::device_span<uint32_t const>{static_cast<uint32_t const*>(nullptr), b_default},
        {thrust::raw_pointer_cast(device_right.data()), b_default},
        {thrust::raw_pointer_cast(device_outputs.data()), 1U}
    );
    EXPECT_FALSE(null_input.has_value());
    EXPECT_TRUE(
        cuddl::compare_batch_async<b_default>(
            cuddl::device_span<uint32_t const>{},
            cuddl::device_span<uint32_t const>{},
            cuddl::device_span<cuddl::pairwise_summary>{}
        )
            .has_value()
    );
}

TEST_F(ReferenceDatabaseTest, ExhaustiveSearchMatchesScalarOracle) {
    constexpr size_t reference_count = 4;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

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
    ASSERT_TRUE(database
                    .search_async(device_query, compatibility, workspace, device_results, stream)
                    .has_value());

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

TEST_F(ReferenceDatabaseTest, PackedRowsPreserveMultiplicityWithoutChangingSearch) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 4;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    std::vector<uint16_t> query(b_default);
    std::vector<uint16_t> scores(reference_count * b_default);
    std::vector<uint32_t> packed(scores.size());
    std::vector<uint32_t> saturation{1U, 0U, 1U, 0U};
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        query[bucket] = std::array<uint16_t, 4>{0U, 7U, 5U, 9U}[bucket % 4];
        scores[bucket] = 0U;
        scores[b_default + bucket] = query[bucket];
        scores[2 * b_default + bucket] = query[bucket] == 0U ? 3U : query[bucket] + 1U;
        scores[3 * b_default + bucket] = query[bucket] == 0U ? 0U : query[bucket] - 1U;
    }
    for (size_t offset = 0; offset < scores.size(); ++offset) {
        auto const count =
            scores[offset] == 0U ? uint16_t{0} : static_cast<uint16_t>(offset % k_counter_max + 1U);
        packed[offset] = pack(scores[offset], count);
    }

    thrust::device_vector<uint16_t> device_query(query);
    thrust::device_vector<uint16_t> device_scores(scores);
    thrust::device_vector<uint32_t> device_packed(packed);
    thrust::device_vector<uint32_t> device_saturation(saturation);
    auto compact_built = database_type::build_indexed_async(device_scores, compatibility, stream);
    ASSERT_TRUE(compact_built.has_value()) << compact_built.error().message();
    auto packed_built =
        database_type::build_indexed_async(device_packed, device_saturation, compatibility, stream);
    ASSERT_TRUE(packed_built.has_value()) << packed_built.error().message();
    auto compact = std::move(*compact_built);
    auto packed_database = std::move(*packed_built);

    EXPECT_FALSE(compact.preserves_multiplicity());
    EXPECT_TRUE(packed_database.preserves_multiplicity());
    EXPECT_EQ(compact.persistent_row_bytes(), reference_count * b_default * sizeof(uint16_t));
    EXPECT_EQ(
        packed_database.persistent_row_bytes(),
        reference_count * (b_default * sizeof(uint32_t) + sizeof(uint32_t))
    );
    EXPECT_TRUE(compact.packed_data().empty());
    EXPECT_EQ(packed_database.packed_data().size(), packed.size());
    EXPECT_EQ(packed_database.saturation_states().size(), saturation.size());

    thrust::device_vector<cuddl::reference_search_result> compact_exhaustive(reference_count);
    thrust::device_vector<cuddl::reference_search_result> packed_exhaustive(reference_count);
    thrust::device_vector<cuddl::reference_search_result> compact_indexed(reference_count);
    thrust::device_vector<cuddl::reference_search_result> packed_indexed(reference_count);
    thrust::device_vector<uint32_t> compact_result_count(1);
    thrust::device_vector<uint32_t> packed_result_count(1);
    auto workspace_bytes = compact.indexed_single_query_workspace_bytes();
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    thrust::device_vector<uint8_t> compact_workspace(*workspace_bytes);
    thrust::device_vector<uint8_t> packed_workspace(*workspace_bytes);

    ASSERT_TRUE(compact
                    .search_async(
                        device_query, compatibility, compact_workspace, compact_exhaustive, stream
                    )
                    .has_value());
    ASSERT_TRUE(
        packed_database
            .search_async(device_query, compatibility, packed_workspace, packed_exhaustive, stream)
            .has_value()
    );
    ASSERT_TRUE(compact
                    .search_indexed_async(
                        device_query,
                        compatibility,
                        compact_workspace,
                        compact_indexed,
                        compact_result_count,
                        {},
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_indexed_async(
                        device_query,
                        compatibility,
                        packed_workspace,
                        packed_indexed,
                        packed_result_count,
                        {},
                        stream
                    )
                    .has_value());

    std::vector<cuddl::reference_search_result> compact_exhaustive_host(reference_count);
    std::vector<cuddl::reference_search_result> packed_exhaustive_host(reference_count);
    std::vector<cuddl::reference_search_result> compact_indexed_host(reference_count);
    std::vector<cuddl::reference_search_result> packed_indexed_host(reference_count);
    std::vector<uint32_t> recovered_packed(packed.size());
    std::vector<uint32_t> recovered_saturation(saturation.size());
    uint32_t compact_count = 0;
    uint32_t packed_count = 0;
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            compact_exhaustive_host.data(),
            thrust::raw_pointer_cast(compact_exhaustive.data()),
            compact_exhaustive_host.size() * sizeof(compact_exhaustive_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            packed_exhaustive_host.data(),
            thrust::raw_pointer_cast(packed_exhaustive.data()),
            packed_exhaustive_host.size() * sizeof(packed_exhaustive_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            compact_indexed_host.data(),
            thrust::raw_pointer_cast(compact_indexed.data()),
            compact_indexed_host.size() * sizeof(compact_indexed_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            packed_indexed_host.data(),
            thrust::raw_pointer_cast(packed_indexed.data()),
            packed_indexed_host.size() * sizeof(packed_indexed_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &compact_count,
            thrust::raw_pointer_cast(compact_result_count.data()),
            sizeof(compact_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &packed_count,
            thrust::raw_pointer_cast(packed_result_count.data()),
            sizeof(packed_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            recovered_packed.data(),
            packed_database.packed_data().data(),
            packed_database.packed_data().size_bytes(),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            recovered_saturation.data(),
            packed_database.saturation_states().data(),
            packed_database.saturation_states().size_bytes(),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    EXPECT_EQ(recovered_packed, packed);
    EXPECT_EQ(recovered_saturation, saturation);
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        auto const expected = score_row_oracle(query, scores, reference_id);
        EXPECT_EQ(compact_exhaustive_host[reference_id].reference_id, reference_id);
        EXPECT_EQ(compact_exhaustive_host[reference_id].summary, expected);
        EXPECT_EQ(packed_exhaustive_host[reference_id], compact_exhaustive_host[reference_id]);
    }
    ASSERT_EQ(compact_count, 1U);
    ASSERT_EQ(packed_count, compact_count);
    EXPECT_EQ(compact_indexed_host[0].reference_id, 1U);
    EXPECT_EQ(packed_indexed_host[0], compact_indexed_host[0]);
}

TEST_F(ReferenceDatabaseTest, BatchSearchMatchesRepeatedSingleQueriesForCompactAndPacked) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 3U;
    constexpr uint32_t query_count = 3U;
    constexpr uint32_t query_id_offset = 41U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    constexpr std::array<std::array<uint16_t, 4>, query_count> patterns{{
        {0U, 7U, 5U, 9U},
        {3U, 0U, 11U, 4U},
        {5U, 12U, 0U, 1U},
    }};
    std::vector<uint16_t> queries(query_count * b_default);
    std::vector<uint16_t> scores(reference_count * b_default);
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        for (size_t bucket = 0; bucket < b_default; ++bucket) {
            queries[query_id * b_default + bucket] = patterns[query_id][bucket % 4U];
            scores[query_id * b_default + bucket] = patterns[query_id][bucket % 4U];
        }
    }
    std::vector<uint32_t> packed(scores.size());
    for (size_t offset = 0; offset < scores.size(); ++offset) {
        auto const count =
            scores[offset] == 0U ? uint16_t{0} : static_cast<uint16_t>(offset % 31U + 1U);
        packed[offset] = pack(scores[offset], count);
    }

    thrust::device_vector<uint16_t> device_queries(queries);
    thrust::device_vector<uint16_t> device_scores(scores);
    thrust::device_vector<uint32_t> device_packed(packed);
    thrust::device_vector<uint32_t> device_saturation(reference_count, 0U);
    auto compact_built = database_type::build_indexed_async(device_scores, compatibility, stream);
    auto packed_built =
        database_type::build_indexed_async(device_packed, device_saturation, compatibility, stream);
    ASSERT_TRUE(compact_built.has_value()) << compact_built.error().message();
    ASSERT_TRUE(packed_built.has_value()) << packed_built.error().message();
    auto compact = std::move(*compact_built);
    auto packed_database = std::move(*packed_built);

    auto compact_exhaustive_requirements = compact.batch_search_requirements(query_count);
    auto packed_exhaustive_requirements = packed_database.batch_search_requirements(query_count);
    auto compact_indexed_requirements = compact.indexed_batch_search_requirements(query_count);
    auto packed_indexed_requirements =
        packed_database.indexed_batch_search_requirements(query_count);
    ASSERT_TRUE(compact_exhaustive_requirements.has_value());
    ASSERT_TRUE(packed_exhaustive_requirements.has_value());
    ASSERT_TRUE(compact_indexed_requirements.has_value());
    ASSERT_TRUE(packed_indexed_requirements.has_value());
    EXPECT_EQ(*compact_exhaustive_requirements, *packed_exhaustive_requirements);
    EXPECT_EQ(*compact_indexed_requirements, *packed_indexed_requirements);
    EXPECT_EQ(compact_exhaustive_requirements->maximum_pair_count, query_count * reference_count);
    EXPECT_EQ(
        compact_exhaustive_requirements->result_bytes,
        static_cast<size_t>(query_count * reference_count) * sizeof(cuddl::batch_search_result)
    );
    EXPECT_EQ(
        compact_exhaustive_requirements->match_count_bytes,
        static_cast<size_t>(query_count * reference_count) * sizeof(uint32_t)
    );
    EXPECT_EQ(compact_exhaustive_requirements->workspace_bytes, 0U);
    ASSERT_GT(compact_indexed_requirements->workspace_bytes, 0U);

    thrust::device_vector<uint8_t> compact_exhaustive_workspace(
        compact_exhaustive_requirements->workspace_bytes
    );
    thrust::device_vector<uint8_t> packed_exhaustive_workspace(
        packed_exhaustive_requirements->workspace_bytes
    );
    thrust::device_vector<uint8_t> compact_indexed_workspace(
        compact_indexed_requirements->workspace_bytes
    );
    thrust::device_vector<uint8_t> packed_indexed_workspace(
        packed_indexed_requirements->workspace_bytes
    );
    auto const pair_capacity = compact_exhaustive_requirements->maximum_pair_count;
    thrust::device_vector<cuddl::batch_search_result> compact_exhaustive(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> packed_exhaustive(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> compact_indexed(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> packed_indexed(pair_capacity);
    thrust::device_vector<uint32_t> compact_exhaustive_count(1U);
    thrust::device_vector<uint32_t> packed_exhaustive_count(1U);
    thrust::device_vector<uint32_t> compact_indexed_count(1U);
    thrust::device_vector<uint32_t> packed_indexed_count(1U);
    thrust::device_vector<uint32_t> compact_exhaustive_matches(pair_capacity);
    thrust::device_vector<uint32_t> packed_exhaustive_matches(pair_capacity);
    thrust::device_vector<uint32_t> compact_indexed_matches(pair_capacity);
    thrust::device_vector<uint32_t> packed_indexed_matches(pair_capacity);

    ASSERT_TRUE(compact
                    .search_batch_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        compact_exhaustive_workspace,
                        compact_exhaustive,
                        compact_exhaustive_count,
                        compact_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_batch_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        packed_exhaustive_workspace,
                        packed_exhaustive,
                        packed_exhaustive_count,
                        packed_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(compact
                    .search_batch_indexed_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        compact_indexed_workspace,
                        compact_indexed,
                        compact_indexed_count,
                        compact_indexed_matches,
                        {.minimum_matches = 1U},
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_batch_indexed_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        packed_indexed_workspace,
                        packed_indexed,
                        packed_indexed_count,
                        packed_indexed_matches,
                        {.minimum_matches = 1U},
                        stream
                    )
                    .has_value());

    thrust::device_vector<cuddl::batch_search_result> no_diagnostic_results(pair_capacity);
    thrust::device_vector<uint32_t> no_diagnostic_count(1U);
    ASSERT_TRUE(compact
                    .search_batch_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        compact_exhaustive_workspace,
                        no_diagnostic_results,
                        no_diagnostic_count,
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<cuddl::batch_search_result> compact_exhaustive_host;
    std::vector<cuddl::batch_search_result> packed_exhaustive_host;
    std::vector<cuddl::batch_search_result> compact_indexed_host;
    std::vector<cuddl::batch_search_result> packed_indexed_host;
    std::vector<cuddl::batch_search_result> no_diagnostic_host;
    std::vector<uint32_t> compact_exhaustive_matches_host;
    std::vector<uint32_t> packed_exhaustive_matches_host;
    std::vector<uint32_t> compact_indexed_matches_host;
    std::vector<uint32_t> packed_indexed_matches_host;
    std::vector<uint32_t> compact_exhaustive_count_host;
    std::vector<uint32_t> packed_exhaustive_count_host;
    std::vector<uint32_t> compact_indexed_count_host;
    std::vector<uint32_t> packed_indexed_count_host;
    std::vector<uint32_t> no_diagnostic_count_host;
    ASSERT_TRUE(copy_device_vector(compact_exhaustive, compact_exhaustive_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive, packed_exhaustive_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed, compact_indexed_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed, packed_indexed_host));
    ASSERT_TRUE(copy_device_vector(no_diagnostic_results, no_diagnostic_host));
    ASSERT_TRUE(copy_device_vector(compact_exhaustive_matches, compact_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive_matches, packed_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed_matches, compact_indexed_matches_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed_matches, packed_indexed_matches_host));
    ASSERT_TRUE(copy_device_vector(compact_exhaustive_count, compact_exhaustive_count_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive_count, packed_exhaustive_count_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed_count, compact_indexed_count_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed_count, packed_indexed_count_host));
    ASSERT_TRUE(copy_device_vector(no_diagnostic_count, no_diagnostic_count_host));

    std::vector<cuddl::batch_search_result> expected_results;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        std::vector<uint16_t> query(
            queries.begin() + query_id * b_default, queries.begin() + (query_id + 1U) * b_default
        );
        for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
            expected_results.push_back({
                .query_id = query_id_offset + static_cast<uint32_t>(query_id),
                .reference_id = static_cast<uint32_t>(reference_id),
                .summary = score_row_oracle(query, scores, reference_id),
            });
        }
    }
    std::vector<cuddl::batch_search_result> const expected_indexed_results{
        expected_results[0U],
        expected_results[4U],
        expected_results[8U],
    };
    std::vector<uint32_t> const expected_matches{
        1536U,
        0U,
        0U,
        0U,
        1536U,
        0U,
        0U,
        0U,
        1536U,
    };
    std::vector<uint32_t> const expected_indexed_matches{1536U, 1536U, 1536U};
    EXPECT_EQ(compact_exhaustive_count_host.front(), pair_capacity);
    EXPECT_EQ(packed_exhaustive_count_host.front(), pair_capacity);
    EXPECT_EQ(compact_indexed_count_host.front(), expected_indexed_results.size());
    EXPECT_EQ(packed_indexed_count_host.front(), expected_indexed_results.size());
    EXPECT_EQ(no_diagnostic_count_host.front(), pair_capacity);
    compact_indexed_host.resize(compact_indexed_count_host.front());
    packed_indexed_host.resize(packed_indexed_count_host.front());
    compact_indexed_matches_host.resize(compact_indexed_count_host.front());
    packed_indexed_matches_host.resize(packed_indexed_count_host.front());
    EXPECT_EQ(compact_exhaustive_host, expected_results);
    EXPECT_EQ(packed_exhaustive_host, expected_results);
    EXPECT_EQ(compact_indexed_host, expected_indexed_results);
    EXPECT_EQ(packed_indexed_host, expected_indexed_results);
    EXPECT_EQ(compact_exhaustive_host, packed_exhaustive_host);
    EXPECT_EQ(compact_exhaustive_matches_host, expected_matches);
    EXPECT_EQ(packed_exhaustive_matches_host, expected_matches);
    EXPECT_EQ(compact_indexed_matches_host, expected_indexed_matches);
    EXPECT_EQ(packed_indexed_matches_host, expected_indexed_matches);

    auto collect_single = [&](auto const& database, bool indexed) {
        std::vector<cuddl::batch_search_result> concatenated;
        for (size_t query_id = 0; query_id < query_count; ++query_id) {
            thrust::device_vector<uint16_t> one_query(
                queries.begin() + query_id * b_default,
                queries.begin() + (query_id + 1U) * b_default
            );
            thrust::device_vector<cuddl::reference_search_result> one_results(reference_count);
            thrust::device_vector<uint8_t> one_workspace;
            thrust::device_vector<uint32_t> one_count(1U);
            if (indexed) {
                auto workspace_bytes = database.indexed_single_query_workspace_bytes();
                if (!workspace_bytes.has_value()) {
                    ADD_FAILURE() << workspace_bytes.error().message();
                    return concatenated;
                }
                one_workspace.resize(*workspace_bytes);
                auto searched = database.search_indexed_async(
                    one_query,
                    compatibility,
                    one_workspace,
                    one_results,
                    one_count,
                    {.minimum_matches = 1U},
                    stream
                );
                if (!searched.has_value()) {
                    ADD_FAILURE() << searched.error().message();
                    return concatenated;
                }
            } else {
                auto searched = database.search_async(
                    one_query, compatibility, one_workspace, one_results, stream
                );
                if (!searched.has_value()) {
                    ADD_FAILURE() << searched.error().message();
                    return concatenated;
                }
            }
            if (cudaStreamSynchronize(stream_) != cudaSuccess) {
                ADD_FAILURE();
                return concatenated;
            }
            std::vector<cuddl::reference_search_result> one_host;
            if (!copy_device_vector(one_results, one_host)) {
                ADD_FAILURE();
                return concatenated;
            }
            if (indexed) {
                std::vector<uint32_t> one_count_host;
                if (!copy_device_vector(one_count, one_count_host)) {
                    ADD_FAILURE();
                    return concatenated;
                }
                EXPECT_EQ(one_count_host.front(), 1U);
                one_host.resize(one_count_host.front());
            }
            for (auto const& result : one_host) {
                concatenated.push_back({
                    .query_id = query_id_offset + static_cast<uint32_t>(query_id),
                    .reference_id = result.reference_id,
                    .summary = result.summary,
                });
            }
        }
        return concatenated;
    };
    EXPECT_EQ(compact_exhaustive_host, collect_single(compact, false));
    EXPECT_EQ(compact_indexed_host, collect_single(compact, true));
    EXPECT_EQ(packed_exhaustive_host, collect_single(packed_database, false));
    EXPECT_EQ(packed_indexed_host, collect_single(packed_database, true));
}

TEST_F(ReferenceDatabaseTest, IndexedBatchCapacityReportsRequiredPairsAndReusesWorkspace) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 2U;
    constexpr uint32_t query_count = 3U;
    constexpr uint32_t query_id_offset = 100U;
    constexpr uint32_t short_capacity = 2U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    std::vector<uint16_t> queries(query_count * b_default);
    std::vector<uint16_t> rows(reference_count * b_default);
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        queries[bucket] = std::array<uint16_t, 4>{0U, 1U, 2U, 3U}[bucket % 4U];
        queries[b_default + bucket] = std::array<uint16_t, 4>{4U, 0U, 6U, 7U}[bucket % 4U];
        queries[2U * b_default + bucket] = std::array<uint16_t, 4>{8U, 9U, 0U, 11U}[bucket % 4U];
        rows[bucket] = queries[bucket];
        rows[b_default + bucket] = queries[b_default + bucket];
    }
    thrust::device_vector<uint16_t> device_queries(queries);
    thrust::device_vector<uint16_t> device_rows(rows);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto requirements = database.indexed_batch_search_requirements(query_count);
    ASSERT_TRUE(requirements.has_value()) << requirements.error().message();
    ASSERT_EQ(requirements->maximum_pair_count, query_count * reference_count);
    thrust::device_vector<uint8_t> workspace(requirements->workspace_bytes);

    cuddl::batch_search_result const result_sentinel{
        .query_id = 0xdeadU,
        .reference_id = 0xbeefU,
        .summary = {.counts = {.lower = 7U, .equal = 8U, .higher = 9U, .both_empty = 10U}},
    };
    constexpr uint32_t match_sentinel = 0xabcdU;
    thrust::device_vector<cuddl::batch_search_result> short_results(
        short_capacity, result_sentinel
    );
    thrust::device_vector<uint32_t> short_matches(short_capacity, match_sentinel);
    thrust::device_vector<uint32_t> short_count(1U, 31337U);
    auto indexed_short = database.search_batch_indexed_async(
        device_queries,
        compatibility,
        query_id_offset,
        workspace,
        short_results,
        short_count,
        short_matches,
        {.minimum_matches = 0U},
        stream
    );
    ASSERT_TRUE(indexed_short.has_value()) << indexed_short.error().message();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<cuddl::batch_search_result> short_results_host;
    std::vector<uint32_t> short_matches_host;
    std::vector<uint32_t> short_count_host;
    ASSERT_TRUE(copy_device_vector(short_results, short_results_host));
    ASSERT_TRUE(copy_device_vector(short_matches, short_matches_host));
    ASSERT_TRUE(copy_device_vector(short_count, short_count_host));
    EXPECT_EQ(short_count_host.front(), requirements->maximum_pair_count);
    EXPECT_EQ(
        short_results_host, std::vector<cuddl::batch_search_result>(short_capacity, result_sentinel)
    );
    EXPECT_EQ(short_matches_host, std::vector<uint32_t>(short_capacity, match_sentinel));

    auto exhaustive_requirements = database.batch_search_requirements(query_count);
    ASSERT_TRUE(exhaustive_requirements.has_value());
    thrust::device_vector<uint8_t> exhaustive_workspace(exhaustive_requirements->workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> exhaustive_short_results(
        requirements->maximum_pair_count - 1U, result_sentinel
    );
    thrust::device_vector<uint32_t> exhaustive_short_count(1U, 2718U);
    thrust::device_vector<uint32_t> exhaustive_short_matches(
        requirements->maximum_pair_count - 1U, match_sentinel
    );
    auto exhaustive_short = database.search_batch_async(
        device_queries,
        compatibility,
        query_id_offset,
        exhaustive_workspace,
        exhaustive_short_results,
        exhaustive_short_count,
        exhaustive_short_matches,
        stream
    );
    ASSERT_FALSE(exhaustive_short.has_value());
    EXPECT_EQ(exhaustive_short.error().category(), cuddl::ErrorCategory::resource);
    std::vector<cuddl::batch_search_result> exhaustive_short_host;
    std::vector<uint32_t> exhaustive_short_matches_host;
    std::vector<uint32_t> exhaustive_short_count_host;
    ASSERT_TRUE(copy_device_vector(exhaustive_short_results, exhaustive_short_host));
    ASSERT_TRUE(copy_device_vector(exhaustive_short_matches, exhaustive_short_matches_host));
    ASSERT_TRUE(copy_device_vector(exhaustive_short_count, exhaustive_short_count_host));
    EXPECT_EQ(
        exhaustive_short_host,
        std::vector<cuddl::batch_search_result>(
            requirements->maximum_pair_count - 1U, result_sentinel
        )
    );
    EXPECT_EQ(
        exhaustive_short_matches_host,
        std::vector<uint32_t>(requirements->maximum_pair_count - 1U, match_sentinel)
    );
    EXPECT_EQ(exhaustive_short_count_host.front(), 2718U);

    thrust::device_vector<cuddl::batch_search_result> successful_results(
        requirements->maximum_pair_count
    );
    thrust::device_vector<uint32_t> successful_matches(requirements->maximum_pair_count);
    thrust::device_vector<uint32_t> successful_count(1U);
    ASSERT_TRUE(database
                    .search_batch_indexed_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        workspace,
                        successful_results,
                        successful_count,
                        successful_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<cuddl::batch_search_result> successful_host;
    std::vector<uint32_t> successful_matches_host;
    std::vector<uint32_t> successful_count_host;
    ASSERT_TRUE(copy_device_vector(successful_results, successful_host));
    ASSERT_TRUE(copy_device_vector(successful_matches, successful_matches_host));
    ASSERT_TRUE(copy_device_vector(successful_count, successful_count_host));
    std::vector<cuddl::batch_search_result> expected_results;
    for (size_t query_id = 0; query_id < query_count; ++query_id) {
        std::vector<uint16_t> query(
            queries.begin() + query_id * b_default, queries.begin() + (query_id + 1U) * b_default
        );
        for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
            expected_results.push_back({
                .query_id = query_id_offset + static_cast<uint32_t>(query_id),
                .reference_id = static_cast<uint32_t>(reference_id),
                .summary = score_row_oracle(query, rows, reference_id),
            });
        }
    }
    std::vector<uint32_t> const expected_matches{
        1536U,
        0U,
        0U,
        1536U,
        0U,
        0U,
    };
    EXPECT_EQ(successful_count_host.front(), requirements->maximum_pair_count);
    EXPECT_EQ(successful_host, expected_results);
    EXPECT_EQ(successful_matches_host, expected_matches);
}
TEST_F(ReferenceDatabaseTest, AllToAllSearchHasOneExactDirectionalOrientation) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 4U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        rows[2U * b_default + bucket] = std::array<uint16_t, 4>{1U, 0U, 3U, 0U}[bucket % 4U];
        rows[3U * b_default + bucket] = std::array<uint16_t, 4>{0U, 4U, 3U, 5U}[bucket % 4U];
    }
    std::vector<uint32_t> packed(rows.size());
    for (size_t offset = 0; offset < rows.size(); ++offset) {
        auto const count =
            rows[offset] == 0U ? uint16_t{0} : static_cast<uint16_t>(offset % 17U + 1U);
        packed[offset] = pack(rows[offset], count);
    }

    thrust::device_vector<uint16_t> device_rows(rows);
    thrust::device_vector<uint32_t> device_packed(packed);
    thrust::device_vector<uint32_t> saturation(reference_count, 0U);
    auto compact_built = database_type::build_indexed_async(device_rows, compatibility, stream);
    auto packed_built =
        database_type::build_indexed_async(device_packed, saturation, compatibility, stream);
    ASSERT_TRUE(compact_built.has_value()) << compact_built.error().message();
    ASSERT_TRUE(packed_built.has_value()) << packed_built.error().message();
    auto compact = std::move(*compact_built);
    auto packed_database = std::move(*packed_built);

    auto exhaustive_requirements = compact.all_to_all_search_requirements(0U, reference_count);
    auto indexed_requirements = compact.indexed_all_to_all_search_requirements(0U, reference_count);
    ASSERT_TRUE(exhaustive_requirements.has_value()) << exhaustive_requirements.error().message();
    ASSERT_TRUE(indexed_requirements.has_value()) << indexed_requirements.error().message();
    ASSERT_EQ(exhaustive_requirements->maximum_pair_count, 6U);
    EXPECT_EQ(exhaustive_requirements->counter_bytes, 0U);
    EXPECT_EQ(exhaustive_requirements->candidate_bytes, 0U);
    EXPECT_EQ(exhaustive_requirements->temporary_bytes, 0U);
    EXPECT_EQ(exhaustive_requirements->workspace_bytes, 0U);
    EXPECT_GT(indexed_requirements->counter_bytes, 0U);
    EXPECT_GT(indexed_requirements->candidate_bytes, 0U);
    EXPECT_EQ(indexed_requirements->maximum_pair_count, 6U);
    auto large_indexed_requirements = compact.indexed_batch_search_requirements(128U);
    ASSERT_TRUE(large_indexed_requirements.has_value())
        << large_indexed_requirements.error().message();
    auto const dense_counter_bytes = static_cast<size_t>(128U) * reference_count * sizeof(uint32_t);
    EXPECT_LE(
        large_indexed_requirements->counter_bytes, dense_counter_bytes + 2U * sizeof(uint32_t)
    );
    EXPECT_LE(large_indexed_requirements->candidate_bytes, dense_counter_bytes);

    thrust::device_vector<uint8_t> compact_exhaustive_workspace(
        exhaustive_requirements->workspace_bytes
    );
    thrust::device_vector<uint8_t> packed_exhaustive_workspace(
        exhaustive_requirements->workspace_bytes
    );
    thrust::device_vector<uint8_t> compact_indexed_workspace(indexed_requirements->workspace_bytes);
    thrust::device_vector<uint8_t> packed_indexed_workspace(indexed_requirements->workspace_bytes);
    auto const pair_capacity = exhaustive_requirements->maximum_pair_count;
    thrust::device_vector<cuddl::batch_search_result> compact_exhaustive(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> packed_exhaustive(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> compact_indexed(pair_capacity);
    thrust::device_vector<cuddl::batch_search_result> packed_indexed(pair_capacity);
    thrust::device_vector<uint32_t> compact_exhaustive_count(1U);
    thrust::device_vector<uint32_t> packed_exhaustive_count(1U);
    thrust::device_vector<uint32_t> compact_indexed_count(1U);
    thrust::device_vector<uint32_t> packed_indexed_count(1U);
    thrust::device_vector<uint32_t> compact_exhaustive_matches(pair_capacity);
    thrust::device_vector<uint32_t> packed_exhaustive_matches(pair_capacity);
    thrust::device_vector<uint32_t> compact_indexed_matches(pair_capacity);
    thrust::device_vector<uint32_t> packed_indexed_matches(pair_capacity);

    ASSERT_TRUE(compact
                    .search_all_to_all_async(
                        0U,
                        reference_count,
                        compact_exhaustive_workspace,
                        compact_exhaustive,
                        compact_exhaustive_count,
                        compact_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_all_to_all_async(
                        0U,
                        reference_count,
                        packed_exhaustive_workspace,
                        packed_exhaustive,
                        packed_exhaustive_count,
                        packed_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(compact
                    .search_all_to_all_indexed_async(
                        0U,
                        reference_count,
                        compact_indexed_workspace,
                        compact_indexed,
                        compact_indexed_count,
                        compact_indexed_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_all_to_all_indexed_async(
                        0U,
                        reference_count,
                        packed_indexed_workspace,
                        packed_indexed,
                        packed_indexed_count,
                        packed_indexed_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

    std::vector<cuddl::batch_search_result> compact_exhaustive_host;
    std::vector<cuddl::batch_search_result> packed_exhaustive_host;
    std::vector<cuddl::batch_search_result> compact_indexed_host;
    std::vector<cuddl::batch_search_result> packed_indexed_host;
    std::vector<uint32_t> compact_exhaustive_matches_host;
    std::vector<uint32_t> packed_exhaustive_matches_host;
    std::vector<uint32_t> compact_indexed_matches_host;
    std::vector<uint32_t> packed_indexed_matches_host;
    std::vector<uint32_t> compact_exhaustive_count_host;
    std::vector<uint32_t> packed_exhaustive_count_host;
    std::vector<uint32_t> compact_indexed_count_host;
    std::vector<uint32_t> packed_indexed_count_host;
    ASSERT_TRUE(copy_device_vector(compact_exhaustive, compact_exhaustive_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive, packed_exhaustive_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed, compact_indexed_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed, packed_indexed_host));
    ASSERT_TRUE(copy_device_vector(compact_exhaustive_matches, compact_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive_matches, packed_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed_matches, compact_indexed_matches_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed_matches, packed_indexed_matches_host));
    ASSERT_TRUE(copy_device_vector(compact_exhaustive_count, compact_exhaustive_count_host));
    ASSERT_TRUE(copy_device_vector(packed_exhaustive_count, packed_exhaustive_count_host));
    ASSERT_TRUE(copy_device_vector(compact_indexed_count, compact_indexed_count_host));
    ASSERT_TRUE(copy_device_vector(packed_indexed_count, packed_indexed_count_host));

    std::vector<cuddl::batch_search_result> expected_results;
    for (size_t query_id = 0; query_id < reference_count; ++query_id) {
        std::vector<uint16_t> query(
            rows.begin() + query_id * b_default, rows.begin() + (query_id + 1U) * b_default
        );
        for (size_t reference_id = query_id + 1U; reference_id < reference_count; ++reference_id) {
            expected_results.push_back({
                .query_id = static_cast<uint32_t>(query_id),
                .reference_id = static_cast<uint32_t>(reference_id),
                .summary = score_row_oracle(query, rows, reference_id),
            });
        }
    }
    std::vector<uint32_t> const expected_matches{0U, 0U, 0U, 0U, 0U, 512U};
    EXPECT_EQ(compact_exhaustive_count_host.front(), pair_capacity);
    EXPECT_EQ(packed_exhaustive_count_host.front(), pair_capacity);
    EXPECT_EQ(compact_indexed_count_host.front(), pair_capacity);
    EXPECT_EQ(packed_indexed_count_host.front(), pair_capacity);
    EXPECT_EQ(compact_exhaustive_host, expected_results);
    EXPECT_EQ(packed_exhaustive_host, expected_results);
    EXPECT_EQ(compact_indexed_host, expected_results);
    EXPECT_EQ(packed_indexed_host, expected_results);
    EXPECT_EQ(compact_exhaustive_matches_host, expected_matches);
    EXPECT_EQ(packed_exhaustive_matches_host, expected_matches);
    EXPECT_EQ(compact_indexed_matches_host, expected_matches);
    EXPECT_EQ(packed_indexed_matches_host, expected_matches);
    for (auto const& result : compact_exhaustive_host) {
        EXPECT_LT(result.query_id, result.reference_id);
    }
    ASSERT_EQ(expected_results.front().summary.counts.both_empty, b_default);
    EXPECT_EQ(expected_matches.front(), 0U);
    auto const& mixed = expected_results.back().summary.counts;
    EXPECT_EQ(mixed.lower, b_default / 2U);
    EXPECT_EQ(mixed.higher, b_default / 4U);
    EXPECT_EQ(mixed.equal, b_default / 4U);
    EXPECT_EQ(mixed.both_empty, 0U);

    thrust::device_vector<uint16_t> empty_queries;
    thrust::device_vector<uint8_t> empty_exhaustive_workspace;
    thrust::device_vector<cuddl::batch_search_result> empty_results;
    thrust::device_vector<uint32_t> empty_count(1U, 77U);
    thrust::device_vector<uint32_t> empty_matches;
    ASSERT_TRUE(compact
                    .search_batch_async(
                        empty_queries,
                        compatibility,
                        55U,
                        empty_exhaustive_workspace,
                        empty_results,
                        empty_count,
                        empty_matches,
                        stream
                    )
                    .has_value());
    auto empty_indexed_requirements = compact.indexed_batch_search_requirements(0U);
    ASSERT_TRUE(empty_indexed_requirements.has_value());
    thrust::device_vector<uint8_t> empty_indexed_workspace(
        empty_indexed_requirements->workspace_bytes
    );
    ASSERT_TRUE(compact
                    .search_batch_indexed_async(
                        empty_queries,
                        compatibility,
                        55U,
                        empty_indexed_workspace,
                        empty_results,
                        empty_count,
                        empty_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    std::vector<uint32_t> empty_count_host;
    ASSERT_TRUE(copy_device_vector(empty_count, empty_count_host));
    EXPECT_EQ(empty_count_host.front(), 0U);

    constexpr uint32_t partial_query_count = 2U;
    thrust::device_vector<uint16_t> partial_queries(rows.begin() + 2U * b_default, rows.end());
    auto partial_requirements = compact.batch_search_requirements(partial_query_count);
    ASSERT_TRUE(partial_requirements.has_value());
    thrust::device_vector<uint8_t> partial_workspace(partial_requirements->workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> partial_results(
        partial_requirements->maximum_pair_count
    );
    thrust::device_vector<uint32_t> partial_matches(partial_requirements->maximum_pair_count);
    thrust::device_vector<uint32_t> partial_count(1U);
    ASSERT_TRUE(compact
                    .search_batch_async(
                        partial_queries,
                        compatibility,
                        2U,
                        partial_workspace,
                        partial_results,
                        partial_count,
                        partial_matches,
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    std::vector<cuddl::batch_search_result> partial_host;
    std::vector<uint32_t> partial_matches_host;
    std::vector<uint32_t> partial_count_host;
    ASSERT_TRUE(copy_device_vector(partial_results, partial_host));
    ASSERT_TRUE(copy_device_vector(partial_matches, partial_matches_host));
    ASSERT_TRUE(copy_device_vector(partial_count, partial_count_host));
    std::vector<cuddl::batch_search_result> expected_partial;
    for (size_t query_id = 2U; query_id < reference_count; ++query_id) {
        std::vector<uint16_t> query(
            rows.begin() + query_id * b_default, rows.begin() + (query_id + 1U) * b_default
        );
        for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
            expected_partial.push_back({
                .query_id = static_cast<uint32_t>(query_id),
                .reference_id = static_cast<uint32_t>(reference_id),
                .summary = score_row_oracle(query, rows, reference_id),
            });
        }
    }
    std::vector<uint32_t> const expected_partial_matches{
        0U,
        0U,
        1024U,
        512U,
        0U,
        0U,
        512U,
        1536U,
    };
    EXPECT_EQ(partial_count_host.front(), partial_requirements->maximum_pair_count);
    EXPECT_EQ(partial_host, expected_partial);
    EXPECT_EQ(partial_matches_host, expected_partial_matches);
    auto partial_all_requirements = compact.all_to_all_search_requirements(2U, 2U);
    auto partial_indexed_all_requirements = compact.indexed_all_to_all_search_requirements(2U, 2U);
    ASSERT_TRUE(partial_all_requirements.has_value());
    ASSERT_TRUE(partial_indexed_all_requirements.has_value());
    ASSERT_EQ(partial_all_requirements->maximum_pair_count, 1U);
    ASSERT_EQ(partial_indexed_all_requirements->maximum_pair_count, 1U);
    thrust::device_vector<uint8_t> partial_all_workspace(partial_all_requirements->workspace_bytes);
    thrust::device_vector<uint8_t> partial_indexed_all_workspace(
        partial_indexed_all_requirements->workspace_bytes
    );
    thrust::device_vector<cuddl::batch_search_result> partial_all_results(1U);
    thrust::device_vector<cuddl::batch_search_result> partial_indexed_all_results(1U);
    thrust::device_vector<uint32_t> partial_all_matches(1U);
    thrust::device_vector<uint32_t> partial_indexed_all_matches(1U);
    thrust::device_vector<uint32_t> partial_all_count(1U);
    thrust::device_vector<uint32_t> partial_indexed_all_count(1U);
    ASSERT_TRUE(compact
                    .search_all_to_all_async(
                        2U,
                        2U,
                        partial_all_workspace,
                        partial_all_results,
                        partial_all_count,
                        partial_all_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(compact
                    .search_all_to_all_indexed_async(
                        2U,
                        2U,
                        partial_indexed_all_workspace,
                        partial_indexed_all_results,
                        partial_indexed_all_count,
                        partial_indexed_all_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    std::vector<cuddl::batch_search_result> partial_all_host;
    std::vector<cuddl::batch_search_result> partial_indexed_all_host;
    std::vector<uint32_t> partial_all_matches_host;
    std::vector<uint32_t> partial_indexed_all_matches_host;
    std::vector<uint32_t> partial_all_count_host;
    std::vector<uint32_t> partial_indexed_all_count_host;
    ASSERT_TRUE(copy_device_vector(partial_all_results, partial_all_host));
    ASSERT_TRUE(copy_device_vector(partial_indexed_all_results, partial_indexed_all_host));
    ASSERT_TRUE(copy_device_vector(partial_all_matches, partial_all_matches_host));
    ASSERT_TRUE(copy_device_vector(partial_indexed_all_matches, partial_indexed_all_matches_host));
    ASSERT_TRUE(copy_device_vector(partial_all_count, partial_all_count_host));
    ASSERT_TRUE(copy_device_vector(partial_indexed_all_count, partial_indexed_all_count_host));
    std::vector<uint16_t> const partial_query(
        rows.begin() + 2U * b_default, rows.begin() + 3U * b_default
    );
    std::vector<cuddl::batch_search_result> const expected_partial_all{
        {
            .query_id = 2U,
            .reference_id = 3U,
            .summary = score_row_oracle(partial_query, rows, 3U),
        },
    };
    std::vector<uint32_t> const expected_partial_all_matches{512U};
    EXPECT_EQ(partial_all_count_host.front(), 1U);
    EXPECT_EQ(partial_indexed_all_count_host.front(), 1U);
    EXPECT_EQ(partial_all_host, expected_partial_all);
    EXPECT_EQ(partial_indexed_all_host, expected_partial_all);
    EXPECT_EQ(partial_all_matches_host, expected_partial_all_matches);
    EXPECT_EQ(partial_indexed_all_matches_host, expected_partial_all_matches);
}
TEST_F(ReferenceDatabaseTest, PackedRowsShareConstructionFailureContract) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    thrust::device_vector<uint16_t> malformed_scores(b_default + 1U, 1U);
    thrust::device_vector<uint32_t> malformed_packed(b_default + 1U, pack(1U, 1U));
    thrust::device_vector<uint32_t> saturation(1U, 0U);
    auto compact_malformed = database_type::build_async(
        cuddl::device_span<uint16_t const>{
            thrust::raw_pointer_cast(malformed_scores.data()), malformed_scores.size()
        },
        compatibility,
        stream
    );
    auto packed_malformed = database_type::build_async(
        cuddl::device_span<uint32_t const>{
            thrust::raw_pointer_cast(malformed_packed.data()), malformed_packed.size()
        },
        cuddl::device_span<uint32_t const>{
            thrust::raw_pointer_cast(saturation.data()), saturation.size()
        },
        compatibility,
        stream
    );
    ASSERT_FALSE(compact_malformed.has_value());
    ASSERT_FALSE(packed_malformed.has_value());
    EXPECT_EQ(packed_malformed.error().category(), compact_malformed.error().category());
    EXPECT_EQ(packed_malformed.error().message(), compact_malformed.error().message());

    auto incompatible = compatibility;
    incompatible.key_mask = 0x7fffU;
    auto compact_incompatible =
        database_type::build_async(cuddl::device_span<uint16_t const>{}, incompatible, stream);
    auto packed_incompatible = database_type::build_async(
        cuddl::device_span<uint32_t const>{},
        cuddl::device_span<uint32_t const>{},
        incompatible,
        stream
    );
    ASSERT_FALSE(compact_incompatible.has_value());
    ASSERT_FALSE(packed_incompatible.has_value());
    EXPECT_EQ(packed_incompatible.error().category(), compact_incompatible.error().category());
    EXPECT_EQ(packed_incompatible.error().message(), compact_incompatible.error().message());

    thrust::device_vector<uint32_t> complete_packed(2U * b_default, pack(1U, 1U));
    auto mismatched_saturation = database_type::build_async(
        cuddl::device_span<uint32_t const>{
            thrust::raw_pointer_cast(complete_packed.data()), complete_packed.size()
        },
        cuddl::device_span<uint32_t const>{
            thrust::raw_pointer_cast(saturation.data()), saturation.size()
        },
        compatibility,
        stream
    );
    ASSERT_FALSE(mismatched_saturation.has_value());
    EXPECT_EQ(mismatched_saturation.error().category(), cuddl::ErrorCategory::invalid_argument);
}

TEST_F(ReferenceDatabaseTest, EmptyDatabaseReportsSizesAndReturnsNoResults) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    static_assert(!std::is_trivially_copyable_v<database_type>);

    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};
    auto built =
        database_type::build_async(cuddl::device_span<uint16_t const>{}, compatibility, stream);
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
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
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
    auto unsupported_build =
        database_type::build_async(cuddl::device_span<uint16_t const>{}, unsupported, stream);
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

TEST_F(ReferenceDatabaseTest, IndexedSearchDefaultThresholdMatchesOracle) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 6;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    constexpr std::array<size_t, reference_count> matches{6, 1, 5, 2, 4, 3};
    std::vector<uint16_t> query(b_default, 0U);
    for (size_t bucket = 0; bucket < reference_count; ++bucket) {
        query[bucket] = bucket + 1U == reference_count ? std::numeric_limits<uint16_t>::max()
                                                       : static_cast<uint16_t>(bucket + 1U);
    }
    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        for (size_t bucket = 0; bucket < reference_count; ++bucket) {
            rows[reference_id * b_default + bucket] =
                bucket < matches[reference_id] ? query[bucket]
                                               : static_cast<uint16_t>(query[bucket] - 1U);
        }
    }

    thrust::device_vector<uint16_t> device_rows(rows);
    thrust::device_vector<uint16_t> device_query(query);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto workspace_bytes = database.indexed_single_query_workspace_bytes();
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    thrust::device_vector<uint8_t> workspace(*workspace_bytes);
    thrust::device_vector<cuddl::reference_search_result> device_results(reference_count);
    thrust::device_vector<uint32_t> device_result_count(1);

    auto searched = database.search_indexed_async(
        device_query, compatibility, workspace, device_results, device_result_count, {}, stream
    );
    ASSERT_TRUE(searched.has_value()) << searched.error().message();

    uint32_t result_count = 0;
    std::vector<cuddl::reference_search_result> results(reference_count);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &result_count,
            thrust::raw_pointer_cast(device_result_count.data()),
            sizeof(result_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
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
    ASSERT_EQ(result_count, 2U);
    constexpr std::array<uint32_t, 2> default_ids{0U, 2U};
    for (size_t index = 0; index < result_count; ++index) {
        auto const reference_id = default_ids[index];
        EXPECT_EQ(results[index].reference_id, reference_id);
        EXPECT_EQ(results[index].summary, score_row_oracle(query, rows, reference_id));
    }

    auto threshold_search = database.search_indexed_async(
        device_query,
        compatibility,
        workspace,
        device_results,
        device_result_count,
        {.minimum_matches = 3U},
        stream
    );
    ASSERT_TRUE(threshold_search.has_value()) << threshold_search.error().message();
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &result_count,
            thrust::raw_pointer_cast(device_result_count.data()),
            sizeof(result_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
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
    ASSERT_EQ(result_count, 4U);
    constexpr std::array<uint32_t, 4> threshold_ids{0U, 2U, 4U, 5U};
    for (size_t index = 0; index < result_count; ++index) {
        auto const reference_id = threshold_ids[index];
        EXPECT_EQ(results[index].reference_id, reference_id);
        EXPECT_EQ(results[index].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, IndexedSearchIsIndependentOfPostingOrder) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 6;
    constexpr std::array<std::array<uint32_t, reference_count>, 2> row_orders{{
        {0U, 1U, 2U, 3U, 4U, 5U},
        {5U, 2U, 4U, 0U, 3U, 1U},
    }};
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};
    std::vector<uint16_t> query(b_default, 0U);
    for (size_t bucket = 0; bucket < reference_count; ++bucket) {
        query[bucket] = static_cast<uint16_t>(bucket + 1U);
    }
    thrust::device_vector<uint16_t> device_query(query);
    std::array<std::vector<cuddl::reference_search_result>, row_orders.size()> canonical_results;

    for (size_t run = 0; run < row_orders.size(); ++run) {
        std::vector<uint16_t> rows(reference_count * b_default, 0U);
        for (size_t physical_id = 0; physical_id < reference_count; ++physical_id) {
            auto const logical_id = row_orders[run][physical_id];
            for (size_t bucket = 0; bucket < reference_count; ++bucket) {
                rows[physical_id * b_default + bucket] =
                    bucket <= logical_id ? query[bucket]
                                         : static_cast<uint16_t>(query[bucket] + 20U);
            }
        }

        thrust::device_vector<uint16_t> device_rows(rows);
        auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
        ASSERT_TRUE(built.has_value()) << built.error().message();
        auto database = std::move(*built);
        auto workspace_bytes = database.indexed_single_query_workspace_bytes();
        ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
        thrust::device_vector<uint8_t> workspace(*workspace_bytes);
        thrust::device_vector<cuddl::reference_search_result> device_results(reference_count);
        thrust::device_vector<uint32_t> device_result_count(1);
        auto searched = database.search_indexed_async(
            device_query, compatibility, workspace, device_results, device_result_count, {}, stream
        );
        ASSERT_TRUE(searched.has_value()) << searched.error().message();

        uint32_t result_count = 0;
        std::vector<cuddl::reference_search_result> results(reference_count);
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpyAsync(
                &result_count,
                thrust::raw_pointer_cast(device_result_count.data()),
                sizeof(result_count),
                cudaMemcpyDeviceToHost,
                stream_
            )
        );
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
        ASSERT_EQ(result_count, 2U);
        results.resize(result_count);
        for (size_t index = 1; index < results.size(); ++index) {
            EXPECT_LT(results[index - 1U].reference_id, results[index].reference_id);
        }
        for (auto& result : results) {
            auto const physical_id = result.reference_id;
            auto const logical_id = row_orders[run][physical_id];
            EXPECT_EQ(result.summary, score_row_oracle(query, rows, physical_id));
            result.reference_id = logical_id;
        }
        std::ranges::sort(results, {}, &cuddl::reference_search_result::reference_id);
        canonical_results[run] = std::move(results);
    }

    EXPECT_EQ(canonical_results[0], canonical_results[1]);
}

TEST_F(ReferenceDatabaseTest, IndexedSearchZeroThresholdExactlyMatchesExhaustive) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 3;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};

    std::vector<uint16_t> query(b_default, 0U);
    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    rows[b_default] = 123U;
    rows[2U * b_default + 7U] = std::numeric_limits<uint16_t>::max();
    thrust::device_vector<uint16_t> device_rows(rows);
    thrust::device_vector<uint16_t> device_query(query);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    thrust::device_vector<uint8_t> workspace;
    thrust::device_vector<cuddl::reference_search_result> indexed_results(reference_count);
    thrust::device_vector<cuddl::reference_search_result> exhaustive_results(reference_count);
    thrust::device_vector<uint32_t> device_result_count(1);
    auto indexed = database.search_indexed_async(
        device_query,
        compatibility,
        workspace,
        indexed_results,
        device_result_count,
        {.minimum_matches = 0U},
        stream
    );
    ASSERT_TRUE(indexed.has_value()) << indexed.error().message();
    auto exhaustive =
        database.search_async(device_query, compatibility, workspace, exhaustive_results, stream);
    ASSERT_TRUE(exhaustive.has_value()) << exhaustive.error().message();

    uint32_t result_count = 0;
    std::vector<cuddl::reference_search_result> indexed_host(reference_count);
    std::vector<cuddl::reference_search_result> exhaustive_host(reference_count);
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &result_count,
            thrust::raw_pointer_cast(device_result_count.data()),
            sizeof(result_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            indexed_host.data(),
            thrust::raw_pointer_cast(indexed_results.data()),
            indexed_host.size() * sizeof(indexed_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            exhaustive_host.data(),
            thrust::raw_pointer_cast(exhaustive_results.data()),
            exhaustive_host.size() * sizeof(exhaustive_host.front()),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    ASSERT_EQ(result_count, reference_count);
    EXPECT_EQ(indexed_host, exhaustive_host);
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        EXPECT_EQ(indexed_host[reference_id].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, IndexedSearchModesMatchExhaustiveAcrossThresholds) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 7U;
    auto const full = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};
    EXPECT_EQ(full.indexed_bucket_count, b_default);
    EXPECT_EQ(full.key_mask, std::numeric_limits<uint16_t>::max());

    auto partial = full;
    partial.indexed_bucket_count = b_default / 2U;
    auto masked = full;
    masked.key_mask = 0x7fffU;
    auto combined = partial;
    combined.key_mask = masked.key_mask;
    std::array modes{full, partial, masked, combined};

    constexpr std::array<size_t, reference_count> matches{6U, 1U, 5U, 2U, 4U, 3U, 0U};
    std::vector<uint16_t> query(b_default, 0U);
    for (size_t bucket = 0; bucket < 6U; ++bucket) {
        query[bucket] = static_cast<uint16_t>(bucket + 1U);
    }
    query[b_default / 2U + 1U] = 1000U;
    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        for (size_t bucket = 0; bucket < 6U; ++bucket) {
            rows[reference_id * b_default + bucket] =
                bucket < matches[reference_id] ? query[bucket]
                                               : static_cast<uint16_t>(query[bucket] + 32U);
        }
        rows[reference_id * b_default + b_default / 2U + 1U] =
            static_cast<uint16_t>(2000U + reference_id);
    }
    thrust::device_vector<uint16_t> device_query(query);
    thrust::device_vector<uint16_t> device_rows(rows);

    for (auto const& compatibility : modes) {
        SCOPED_TRACE(compatibility.indexed_bucket_count);
        SCOPED_TRACE(compatibility.key_mask);
        auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
        ASSERT_TRUE(built.has_value()) << built.error().message();
        auto database = std::move(*built);
        EXPECT_EQ(database.metadata().compatibility, compatibility);

        auto workspace_bytes = database.indexed_single_query_workspace_bytes();
        ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
        thrust::device_vector<uint8_t> workspace(*workspace_bytes);
        thrust::device_vector<uint8_t> exhaustive_workspace;
        thrust::device_vector<cuddl::reference_search_result> device_results(reference_count);
        thrust::device_vector<cuddl::reference_search_result> device_exhaustive(reference_count);
        thrust::device_vector<uint32_t> device_result_count(1U);
        auto exhaustive = database.search_async(
            device_query, compatibility, exhaustive_workspace, device_exhaustive, stream
        );
        ASSERT_TRUE(exhaustive.has_value()) << exhaustive.error().message();

        for (uint32_t threshold : {0U, 1U, 5U, 6U}) {
            SCOPED_TRACE(threshold);
            auto searched = database.search_indexed_async(
                device_query,
                compatibility,
                workspace,
                device_results,
                device_result_count,
                {.minimum_matches = threshold},
                stream
            );
            ASSERT_TRUE(searched.has_value()) << searched.error().message();
            ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

            uint32_t result_count = 0;
            ASSERT_EQ(
                cudaSuccess,
                cudaMemcpy(
                    &result_count,
                    thrust::raw_pointer_cast(device_result_count.data()),
                    sizeof(result_count),
                    cudaMemcpyDeviceToHost
                )
            );
            std::vector<cuddl::reference_search_result> results;
            std::vector<cuddl::reference_search_result> exhaustive_results;
            ASSERT_TRUE(copy_device_vector(device_results, results));
            ASSERT_TRUE(copy_device_vector(device_exhaustive, exhaustive_results));
            if (threshold == 0U) {
                ASSERT_EQ(result_count, reference_count);
                EXPECT_EQ(results, exhaustive_results);
                continue;
            }

            std::vector<uint32_t> expected_ids;
            for (uint32_t reference_id = 0; reference_id < reference_count; ++reference_id) {
                if (matches[reference_id] >= threshold) {
                    expected_ids.push_back(reference_id);
                }
            }
            ASSERT_EQ(result_count, expected_ids.size());
            for (size_t index = 0; index < result_count; ++index) {
                auto const reference_id = expected_ids[index];
                EXPECT_EQ(results[index].reference_id, reference_id);
                EXPECT_EQ(results[index].summary, exhaustive_results[reference_id].summary);
                EXPECT_EQ(results[index].summary, score_row_oracle(query, rows, reference_id));
            }
        }
    }
}

TEST_F(ReferenceDatabaseTest, MaskedKeysPreserveZeroAndExactCollisionSemantics) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 4U;
    auto const full = cuddl::score_compatibility::current<k_default, b_default>();
    auto masked = full;
    masked.key_mask = 0x7fffU;
    auto const stream = cuda::stream_ref{stream_};

    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    rows[0] = 0x8000U;
    rows[2U * b_default] = 0x0001U;
    rows[3U * b_default] = 0x8001U;
    thrust::device_vector<uint16_t> device_rows(rows);
    auto full_built = database_type::build_indexed_async(device_rows, full, stream);
    ASSERT_TRUE(full_built.has_value()) << full_built.error().message();
    auto full_database = std::move(*full_built);
    auto masked_built = database_type::build_indexed_async(device_rows, masked, stream);
    ASSERT_TRUE(masked_built.has_value()) << masked_built.error().message();
    auto masked_database = std::move(*masked_built);

    auto search =
        [&](database_type const& database,
            cuddl::score_compatibility compatibility,
            std::vector<uint16_t> const& query,
            std::vector<cuddl::reference_search_result>& results) -> ::testing::AssertionResult {
        thrust::device_vector<uint16_t> device_query(query);
        auto workspace_bytes = database.indexed_single_query_workspace_bytes();
        if (!workspace_bytes) {
            return ::testing::AssertionFailure() << workspace_bytes.error().message();
        }
        thrust::device_vector<uint8_t> workspace(*workspace_bytes);
        thrust::device_vector<cuddl::reference_search_result> device_results(reference_count);
        thrust::device_vector<uint32_t> device_result_count(1U);
        auto searched = database.search_indexed_async(
            device_query,
            compatibility,
            workspace,
            device_results,
            device_result_count,
            {.minimum_matches = 1U},
            stream
        );
        if (!searched) {
            return ::testing::AssertionFailure() << searched.error().message();
        }
        if (auto const status = cudaStreamSynchronize(stream_); status != cudaSuccess) {
            return ::testing::AssertionFailure() << cudaGetErrorString(status);
        }

        uint32_t result_count = 0;
        if (auto const status = cudaMemcpy(
                &result_count,
                thrust::raw_pointer_cast(device_result_count.data()),
                sizeof(result_count),
                cudaMemcpyDeviceToHost
            );
            status != cudaSuccess) {
            return ::testing::AssertionFailure() << cudaGetErrorString(status);
        }
        if (!copy_device_vector(device_results, results)) {
            return ::testing::AssertionFailure() << "failed to copy indexed search results";
        }
        results.resize(result_count);
        return ::testing::AssertionSuccess();
    };

    std::vector<uint16_t> empty_query(b_default, 0U);
    std::vector<cuddl::reference_search_result> empty_results;
    ASSERT_TRUE(search(masked_database, masked, empty_query, empty_results));
    EXPECT_TRUE(empty_results.empty());

    auto folded_zero_query = empty_query;
    folded_zero_query[0] = 0x8000U;
    std::vector<cuddl::reference_search_result> folded_zero_results;
    ASSERT_TRUE(search(masked_database, masked, folded_zero_query, folded_zero_results));
    ASSERT_EQ(folded_zero_results.size(), 1U);
    EXPECT_EQ(folded_zero_results[0].reference_id, 0U);
    EXPECT_EQ(folded_zero_results[0].summary, score_row_oracle(folded_zero_query, rows, 0U));

    auto collision_query = empty_query;
    collision_query[0] = 0x0001U;
    std::vector<cuddl::reference_search_result> full_results;
    std::vector<cuddl::reference_search_result> masked_results;
    ASSERT_TRUE(search(full_database, full, collision_query, full_results));
    ASSERT_TRUE(search(masked_database, masked, collision_query, masked_results));
    ASSERT_EQ(full_results.size(), 1U);
    EXPECT_EQ(full_results[0].reference_id, 2U);
    ASSERT_EQ(masked_results.size(), 2U);
    EXPECT_EQ(masked_results[0].reference_id, 2U);
    EXPECT_EQ(masked_results[1].reference_id, 3U);
    for (auto const& result : masked_results) {
        EXPECT_EQ(result.summary, score_row_oracle(collision_query, rows, result.reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, IndexedSearchRejectsInvalidInputsWithoutOutput) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 2;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const stream = cuda::stream_ref{stream_};
    thrust::device_vector<uint16_t> rows(reference_count * b_default, 17U);
    thrust::device_vector<uint16_t> query(b_default, 17U);

    auto incompatible = compatibility;
    ++incompatible.blacklist_version;
    auto invalid_build_compatibility = compatibility;
    --invalid_build_compatibility.indexed_bucket_count;
    auto incompatible_build =
        database_type::build_indexed_async(rows, invalid_build_compatibility, stream);
    ASSERT_FALSE(incompatible_build.has_value());
    EXPECT_EQ(incompatible_build.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto built = database_type::build_indexed_async(rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto workspace_bytes = database.indexed_single_query_workspace_bytes();
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    ASSERT_GT(*workspace_bytes, 0U);
    thrust::device_vector<uint8_t> workspace(*workspace_bytes);
    thrust::device_vector<uint8_t> short_workspace(*workspace_bytes - 1U);
    cuddl::reference_search_result const sentinel{
        .reference_id = 123U,
        .summary = {.counts = {.lower = 4U, .equal = 5U, .higher = 6U}},
    };
    thrust::device_vector<cuddl::reference_search_result> results(reference_count, sentinel);
    thrust::device_vector<cuddl::reference_search_result> short_results(reference_count - 1U);
    thrust::device_vector<uint32_t> result_count(1, 77U);
    thrust::device_vector<uint32_t> no_result_count;

    auto incompatible_query = database.search_indexed_async(
        query, incompatible, workspace, results, result_count, {}, stream
    );
    ASSERT_FALSE(incompatible_query.has_value());
    EXPECT_EQ(incompatible_query.error().category(), cuddl::ErrorCategory::invalid_argument);
    auto invalid_options = database.search_indexed_async(
        query,
        compatibility,
        workspace,
        results,
        result_count,
        {.minimum_matches = static_cast<uint32_t>(b_default + 1U)},
        stream
    );
    ASSERT_FALSE(invalid_options.has_value());
    EXPECT_EQ(invalid_options.error().category(), cuddl::ErrorCategory::invalid_argument);
    auto insufficient_workspace = database.search_indexed_async(
        query, compatibility, short_workspace, results, result_count, {}, stream
    );
    ASSERT_FALSE(insufficient_workspace.has_value());
    EXPECT_EQ(insufficient_workspace.error().category(), cuddl::ErrorCategory::resource);
    auto insufficient_results = database.search_indexed_async(
        query, compatibility, workspace, short_results, result_count, {}, stream
    );
    ASSERT_FALSE(insufficient_results.has_value());
    EXPECT_EQ(insufficient_results.error().category(), cuddl::ErrorCategory::resource);
    auto missing_count = database.search_indexed_async(
        query, compatibility, workspace, results, no_result_count, {}, stream
    );
    ASSERT_FALSE(missing_count.has_value());
    EXPECT_EQ(missing_count.error().category(), cuddl::ErrorCategory::resource);

    uint32_t observed_count = 0;
    cuddl::reference_search_result observed_result{};
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &observed_count,
            thrust::raw_pointer_cast(result_count.data()),
            sizeof(observed_count),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(
        cudaSuccess,
        cudaMemcpyAsync(
            &observed_result,
            thrust::raw_pointer_cast(results.data()),
            sizeof(observed_result),
            cudaMemcpyDeviceToHost,
            stream_
        )
    );
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
    EXPECT_EQ(observed_count, 77U);
    EXPECT_EQ(observed_result, sentinel);

    auto unindexed_build = database_type::build_async(rows, compatibility, stream);
    ASSERT_TRUE(unindexed_build.has_value()) << unindexed_build.error().message();
    auto unindexed = std::move(*unindexed_build);
    auto missing_index = unindexed.search_indexed_async(
        query, compatibility, workspace, results, result_count, {}, stream
    );
    ASSERT_FALSE(missing_index.has_value());
    EXPECT_EQ(missing_index.error().category(), cuddl::ErrorCategory::invalid_argument);
}

TEST_F(ReferenceDatabaseTest, IndexedBuildRejectsUnrepresentableBounds) {
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const maximum = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    auto const posting_scores = (maximum / b_default + 1U) * b_default;
    auto const reference_scores = (maximum + 1U) * b_default;
    auto const* fake_device_rows = reinterpret_cast<uint16_t const*>(uintptr_t{1});

    auto posting_bound = database_type::build_indexed_async(
        cuddl::device_span<uint16_t const>{fake_device_rows, posting_scores}, compatibility
    );
    ASSERT_FALSE(posting_bound.has_value());
    EXPECT_EQ(posting_bound.error().category(), cuddl::ErrorCategory::resource);

    auto reference_bound = database_type::build_indexed_async(
        cuddl::device_span<uint16_t const>{fake_device_rows, reference_scores}, compatibility
    );
    ASSERT_FALSE(reference_bound.has_value());
    EXPECT_EQ(reference_bound.error().category(), cuddl::ErrorCategory::resource);
}

TEST(SketchTest, CardinalityMatchesScalarOracleAcrossMagnitudes) {
    for (size_t n : {0U, 1U, 2U, 8U, 64U, 512U, 4096U, 1u << 16, 1u << 22}) {
        auto const inputs = make_inputs(n, 0x3333'3333'3333'3333ULL + n);
        thrust::device_vector<uint64_t> device_inputs(inputs.begin(), inputs.end());
        cuddl::sketch<k_default, b_default> gpu;
        if (n > 0) {
            ASSERT_TRUE(gpu.add(device_inputs).has_value());
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
        thrust::device_vector<uint64_t> device_inputs(inputs.begin(), inputs.end());
        cuddl::sketch<k_default, b_default> gpu;
        ASSERT_TRUE(gpu.add(device_inputs).has_value());
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
        thrust::device_vector<uint64_t> device_inputs(inputs.begin(), inputs.end());
        cuddl::sketch<k_default, b_default> gpu;
        ASSERT_TRUE(gpu.add(device_inputs).has_value());
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

    auto const same_result = sa.compare(sb);
    ASSERT_TRUE(same_result.has_value());
    auto const same = *same_result;
    auto const wkid_same = decltype(sa)::wkid(same);
    auto const ani_same = decltype(sa)::ani(same);
    auto const cont_same = decltype(sa)::containment(same);
    auto const comp_same = decltype(sa)::completeness(same);
    ASSERT_TRUE(wkid_same.has_value());
    ASSERT_TRUE(ani_same.has_value());
    ASSERT_TRUE(cont_same.has_value());
    ASSERT_TRUE(comp_same.has_value());
    EXPECT_DOUBLE_EQ(*wkid_same, 1.0);
    EXPECT_DOUBLE_EQ(*ani_same, 1.0);
    EXPECT_GT(*cont_same, 0.9);
    EXPECT_GT(*comp_same, 0.9);

    auto const disjoint_result = sa.compare(sc);
    ASSERT_TRUE(disjoint_result.has_value());
    auto const disjoint = *disjoint_result;
    auto const wkid_disjoint = decltype(sa)::wkid(disjoint);
    ASSERT_TRUE(wkid_disjoint.has_value());
    EXPECT_LT(*wkid_disjoint, 0.05);
}

TEST(SketchTest, WkidPadsSparseDivisors) {
    cuddl::pairwise_summary sparse{};
    sparse.counts.equal = 1;
    sparse.counts.lower = 2;
    sparse.counts.higher = 4;

    auto const wkid = cuddl::sketch<k_default, b_default>::wkid(sparse);
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

/// @brief Encodes one absolute 16-bit bucket value relative to a global NLZ offset.
///
/// Mirrors BBTools' `DynamicDemiLog.toBytesRelative`. Callers must guarantee every nonzero @p abs
/// has `abs >> (16 - exponent_bits) >= global_nlz`, the invariant a well-formed A48 file preserves.
std::string encode_a48_relative(uint16_t abs, int32_t global_nlz, uint32_t exponent_bits) {
    if (abs == 0U) {
        return "0";
    }
    auto const mantissa_bits = 16U - exponent_bits;
    auto const mask = (1U << mantissa_bits) - 1U;
    auto const abs_nlz = static_cast<uint32_t>(abs >> mantissa_bits);
    auto const rel_nlz = abs_nlz - static_cast<uint32_t>(global_nlz);
    auto const rel = static_cast<uint16_t>((rel_nlz << mantissa_bits) | (abs & mask));
    return cuddl::a48::encode_a48_token(rel);
}

/// @brief Builds a small A48 TSV using relative (`#offset`) encoding.
std::string build_a48_relative_tsv(
    std::vector<std::vector<uint16_t>> const& rows,
    uint32_t exponent_bits,
    int32_t global_nlz,
    uint32_t k,
    uint64_t seed
) {
    std::string out;
    out += "#k\t" + std::to_string(k) + "\n";
    out += "#seed\t" + std::to_string(seed) + "\n";
    out += "#exponent\t" + std::to_string(exponent_bits) + "\n";
    for (size_t r = 0; r < rows.size(); ++r) {
        out += "#id\t" + std::to_string(1000 + static_cast<int>(r)) + "\n";
        out += "#tid\t" + std::to_string(200 + static_cast<int>(r)) + "\n";
        out += "#name\tref" + std::to_string(r) + "\n";
        out += "#len\t" + std::to_string(rows[r].size()) + "\n";
        out += "#offset\t" + std::to_string(global_nlz) + "\n";
        for (size_t b = 0; b < rows[r].size(); ++b) {
            if (b != 0U) {
                out += '\t';
            }
            out += encode_a48_relative(rows[r][b], global_nlz, exponent_bits);
        }
        out += '\n';
    }
    return out;
}

/// @brief Builds a small A48 TSV using absolute (legacy, no `#offset`) encoding.
std::string build_a48_absolute_tsv(
    std::vector<std::vector<uint16_t>> const& rows,
    uint32_t exponent_bits,
    uint32_t k,
    uint64_t seed
) {
    std::string out;
    out += "#k\t" + std::to_string(k) + "\n";
    out += "#seed\t" + std::to_string(seed) + "\n";
    out += "#exponent\t" + std::to_string(exponent_bits) + "\n";
    for (size_t r = 0; r < rows.size(); ++r) {
        out += "#id\t" + std::to_string(3000 + static_cast<int>(r)) + "\n";
        out += "#name\tref" + std::to_string(r) + "\n";
        out += "#len\t" + std::to_string(rows[r].size()) + "\n";
        for (size_t b = 0; b < rows[r].size(); ++b) {
            if (b != 0U) {
                out += '\t';
            }
            out += cuddl::a48::encode_a48_token(rows[r][b]);
        }
        out += '\n';
    }
    return out;
}

TEST(A48Decoder, DecodesRelativeHeaderRowsAndMetadata) {
    constexpr uint32_t k = 3U;
    constexpr uint32_t exponent = 2U;
    constexpr int32_t global_nlz = 1;
    constexpr uint64_t seed = 12345ULL;
    // All nonzero absolute scores have `abs >> (16-2) >= 1`, satisfying the relative invariant.
    std::vector<std::vector<uint16_t>> const rows{
        {0, 3000, 4000, 0, 5000, 6000, 0, 7000},
        {0, 3000, 0, 0, 5000, 0, 8000, 0},
        {0, 0, 4000, 0, 0, 6000, 0, 0},
    };

    auto const tsv = build_a48_relative_tsv(rows, exponent, global_nlz, k, seed);
    auto const decoded = cuddl::a48::decode_a48_tsv(tsv);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    auto const& db = *decoded;

    auto const parallel = cuddl::a48::decode_a48_tsv_parallel(tsv, 4U);
    ASSERT_TRUE(parallel.has_value()) << parallel.error().message();
    EXPECT_EQ(parallel->metadata.kmer_length, db.metadata.kmer_length);
    EXPECT_EQ(parallel->metadata.seed, db.metadata.seed);
    EXPECT_EQ(parallel->metadata.exponent_bits, db.metadata.exponent_bits);
    ASSERT_EQ(parallel->records.size(), db.records.size());
    for (size_t r = 0; r < db.records.size(); ++r) {
        EXPECT_EQ(parallel->records[r].ordinal, db.records[r].ordinal);
        EXPECT_EQ(parallel->records[r].metadata.name, db.records[r].metadata.name);
        EXPECT_EQ(parallel->records[r].metadata.offset, db.records[r].metadata.offset);
        EXPECT_EQ(parallel->records[r].scores, db.records[r].scores);
    }

    EXPECT_EQ(db.metadata.kmer_length, k);
    EXPECT_TRUE(db.metadata.has_kmer_length);
    EXPECT_EQ(db.metadata.seed, seed);
    EXPECT_TRUE(db.metadata.has_seed);
    EXPECT_EQ(db.metadata.exponent_bits, exponent);
    EXPECT_TRUE(db.metadata.has_exponent);
    EXPECT_EQ(db.records.size(), rows.size());

    for (size_t r = 0; r < rows.size(); ++r) {
        EXPECT_EQ(db.records[r].ordinal, r);
        EXPECT_EQ(db.records[r].metadata.id, static_cast<int64_t>(1000 + r));
        EXPECT_EQ(db.records[r].metadata.tax_id, static_cast<int64_t>(200 + r));
        EXPECT_EQ(db.records[r].metadata.name, "ref" + std::to_string(r));
        EXPECT_EQ(db.records[r].metadata.offset, global_nlz);
        ASSERT_EQ(db.records[r].scores.size(), rows[r].size());
        EXPECT_EQ(db.records[r].scores, rows[r]);
    }
}

TEST(A48Decoder, DecodesAbsoluteLegacyRows) {
    constexpr uint32_t k = 3U;
    constexpr uint32_t exponent = 2U;
    constexpr uint64_t seed = 7ULL;
    std::vector<std::vector<uint16_t>> const rows{
        {0, 100, 200, 0, 300, 400, 0, 500},
        {0, 0, 0, 0, 0, 0, 0, 0},
    };
    auto const tsv = build_a48_absolute_tsv(rows, exponent, k, seed);
    auto const decoded = cuddl::a48::decode_a48_tsv(tsv);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    auto const& db = *decoded;
    EXPECT_EQ(db.records.size(), rows.size());
    EXPECT_EQ(db.records[0].scores, rows[0]);
    EXPECT_EQ(db.records[1].scores, rows[1]);
}

TEST(A48Decoder, HostDdlIndexOracleCountsMatchCounts) {
    std::vector<std::vector<uint16_t>> const rows{
        {0, 100, 200, 0, 300, 400, 0, 500},
        {0, 100, 0, 0, 300, 0, 600, 0},
        {0, 0, 200, 0, 0, 400, 0, 0},
    };
    std::vector<uint16_t> const query = rows[0];
    auto const oracle = cuddl::a48::exhaustive_oracle(rows, query, 1U);
    auto const& counts = oracle.match_counts;
    EXPECT_EQ(counts.size(), rows.size());
    // query = row0: nonzero buckets {1,2,4,5,7}. row0 matches all five; row1 matches {1,4};
    // row2 matches {2,5}.
    EXPECT_EQ(counts[0], 5U);
    EXPECT_EQ(counts[1], 2U);
    EXPECT_EQ(counts[2], 2U);
    // The fused oracle also classifies the query against every retained reference exactly:
    // query == row0 itself, so all five nonzero buckets are equal, two buckets are both empty,
    // and no bucket is lower or higher.
    auto const& self = oracle.summaries[0];
    EXPECT_EQ(self.equal, 5U);
    EXPECT_EQ(self.lower, 0U);
    EXPECT_EQ(self.higher, 0U);
    EXPECT_EQ(self.both_empty, 3U);
}

TEST(A48Decoder, RejectsMalformedInput) {
    // Character outside the A48 alphabet (ASCII '!' == 33 < 48).
    {
        std::string const tsv = "#name\tbad\n0\t0\t!\n";
        EXPECT_FALSE(cuddl::a48::decode_a48_tsv(tsv).has_value());
    }
    // `#len` disagrees with the data-row width.
    {
        std::string const tsv = "#name\tbad\n#len\t4\n0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_NE(res.error().message().find("#len"), std::string::npos);
    }
    // Bucket count changes between records.
    {
        std::string const tsv = "#name\ta\n0\t0\t0\n#name\tb\n0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_NE(res.error().message().find("width"), std::string::npos);
    }
    // Relative-encoded record without an `#exponent` header.
    {
        std::string const tsv = "#name\tbad\n#offset\t1\n0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_NE(res.error().message().find("exponent"), std::string::npos);
    }
    // Data row before any record header.
    {
        std::string const tsv = "\n0\t0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_EQ(res.error().message().find("record header") != std::string::npos, true);
    }
    // An exponent outside the valid 1..15 range.
    {
        std::string const tsv = "#exponent\t0\n#name\tbad\n0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_NE(res.error().message().find("exponent"), std::string::npos);
    }
    // A non-numeric header field is malformed, not a thrown exception.
    {
        std::string const tsv = "#name\tbad\n#len\tabc\n0\t0\n";
        auto const res = cuddl::a48::decode_a48_tsv(tsv);
        ASSERT_FALSE(res.has_value());
        EXPECT_NE(res.error().message().find("len"), std::string::npos);
    }
    // Empty input decodes to no records.
    { EXPECT_FALSE(cuddl::a48::decode_a48_tsv("").has_value()); }
}

TEST_F(ReferenceDatabaseTest, A48DecodedRowsMatchDdlIndexOracle) {
    using layout = cuddl::register_layout<5, 11>;
    using database_type = cuddl::reference_database<k_default, b_default, layout>;
    constexpr uint32_t exponent = layout::exponent_bits;
    constexpr uint64_t seed = 42ULL;
    constexpr uint32_t minimum_matches = 3U;
    constexpr uint32_t reference_count = 4U;
    constexpr uint32_t external_id = 3U;

    // Deterministic sparse rows over b_default buckets. Nonzero absolute scores live in a
    // realistic DDL range; the pattern yields distinct match counts per reference.
    std::vector<std::vector<uint16_t>> rows(reference_count, std::vector<uint16_t>(b_default, 0U));
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        auto const base = static_cast<uint16_t>(
            (bucket % 4U == 0U) ? 0U : static_cast<uint16_t>(3000U * (bucket % 4U))
        );
        rows[0][bucket] = base;
        rows[1][bucket] =
            static_cast<uint16_t>(base == 0U ? 0U : (bucket % 8U == 5U ? base + 1000U : base));
        rows[2][bucket] = static_cast<uint16_t>(base == 0U ? 0U : base + 1U);
        rows[3][bucket] = static_cast<uint16_t>(bucket % 11U == 0U ? base : 0U);
    }

    auto const tsv = build_a48_absolute_tsv(rows, exponent, k_default, seed);
    auto const decoded = cuddl::a48::decode_a48_tsv(tsv);
    ASSERT_TRUE(decoded.has_value()) << decoded.error().message();
    ASSERT_EQ(decoded->records.size(), reference_count);
    for (uint32_t r = 0; r < reference_count; ++r) {
        EXPECT_EQ(decoded->records[r].scores, rows[r]);
    }

    // Mark row 3 external, mirroring the parity benchmark policy: the index is built from the
    // remaining records in original order, and row 3 becomes an out-of-database query whose
    // candidate IDs address the reduced database.
    std::vector<uint32_t> const db_ids{0U, 1U, 2U};
    std::vector<uint16_t> scores(db_ids.size() * b_default);
    for (size_t i = 0; i < db_ids.size(); ++i) {
        std::copy(
            decoded->records[db_ids[i]].scores.begin(),
            decoded->records[db_ids[i]].scores.end(),
            scores.begin() + i * b_default
        );
    }
    auto const compatibility = cuddl::decoded_compatibility(k_default, b_default, exponent, seed);
    auto const stream = cuda::stream_ref{stream_};
    thrust::device_vector<uint16_t> device_scores(scores);
    auto built = database_type::build_indexed_async(device_scores, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    // Exhaustive reference database over the same subset, for the GPU exact-summary contract.
    auto exhaustive_built = database_type::build_async(device_scores, compatibility, stream);
    ASSERT_TRUE(exhaustive_built.has_value());
    auto exhaustive_database = std::move(*exhaustive_built);

    auto indexed_workspace_bytes = database.indexed_single_query_workspace_bytes();
    ASSERT_TRUE(indexed_workspace_bytes.has_value()) << indexed_workspace_bytes.error().message();
    thrust::device_vector<uint8_t> indexed_workspace(*indexed_workspace_bytes);

    // Resident queries and the external row must agree with the same host oracle.
    std::vector<uint32_t> const query_ids{0U, 1U, 2U, external_id};
    for (auto const query_id : query_ids) {
        SCOPED_TRACE(query_id);
        auto const& query_row = rows[query_id];
        thrust::device_vector<uint16_t> device_query(query_row);
        thrust::device_vector<cuddl::reference_search_result> indexed(db_ids.size());
        thrust::device_vector<cuddl::reference_search_result> exhaustive(db_ids.size());
        thrust::device_vector<uint32_t> indexed_count(1U);
        thrust::device_vector<uint8_t> exhaustive_workspace;

        ASSERT_TRUE(database
                        .search_indexed_async(
                            device_query,
                            compatibility,
                            indexed_workspace,
                            indexed,
                            indexed_count,
                            {.minimum_matches = minimum_matches},
                            stream
                        )
                        .has_value());
        ASSERT_TRUE(
            exhaustive_database
                .search_async(device_query, compatibility, exhaustive_workspace, exhaustive, stream)
                .has_value()
        );
        ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));

        std::vector<cuddl::reference_search_result> indexed_host(db_ids.size());
        std::vector<cuddl::reference_search_result> exhaustive_host(db_ids.size());
        ASSERT_TRUE(copy_device_vector(indexed, indexed_host));
        ASSERT_TRUE(copy_device_vector(exhaustive, exhaustive_host));
        uint32_t count = 0;
        ASSERT_EQ(
            cudaSuccess,
            cudaMemcpy(
                &count,
                thrust::raw_pointer_cast(indexed_count.data()),
                sizeof(count),
                cudaMemcpyDeviceToHost
            )
        );

        // Independent host oracle: BBTools DDLIndex/CSR2 match counts over the same subset,
        // plus exhaustive lower/equal/higher/both-empty summaries for every retained candidate.
        auto const oracle =
            cuddl::a48::exhaustive_oracle(scores, query_row, b_default, minimum_matches);
        std::vector<uint32_t> expected_candidates;
        for (uint32_t r = 0; r < db_ids.size(); ++r) {
            if (oracle.match_counts[r] >= minimum_matches) {
                expected_candidates.push_back(r);
            }
        }

        EXPECT_EQ(count, expected_candidates.size());
        ASSERT_LE(count, expected_candidates.size());
        for (uint32_t i = 0; i < count; ++i) {
            auto const reference_id = indexed_host[i].reference_id;
            EXPECT_EQ(reference_id, expected_candidates[i]);
            // Index match count equals the DDLIndex oracle count and the exact `equal` class.
            EXPECT_EQ(indexed_host[i].summary.counts.equal, oracle.match_counts[reference_id]);
            // Exact refinement == the independent exhaustive comparison over decoded rows.
            auto const& expected_summary = oracle.summaries[reference_id];
            EXPECT_EQ(indexed_host[i].summary.counts.lower, expected_summary.lower);
            EXPECT_EQ(indexed_host[i].summary.counts.equal, expected_summary.equal);
            EXPECT_EQ(indexed_host[i].summary.counts.higher, expected_summary.higher);
            EXPECT_EQ(indexed_host[i].summary.counts.both_empty, expected_summary.both_empty);
            // Exact refinement == the exhaustive GPU comparison over the same subset.
            EXPECT_EQ(indexed_host[i].summary, exhaustive_host[reference_id].summary);
            EXPECT_EQ(reference_id, exhaustive_host[reference_id].reference_id);
        }
    }
}

}  // namespace
