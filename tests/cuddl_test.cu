#include <cuddl/a48.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>
#include <cuddl/refseq_parity.hpp>

#include <gtest/gtest.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
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
    cuda::stream stream_{cuda::devices[0]};
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
bool copy_device_buffer(cuda::device_buffer<T> const& source, std::vector<T>& destination) {
    destination.resize(source.size());
    if (destination.empty()) {
        return true;
    }
    return cuddl::cuda_try(
               [&] {
                   cuda::copy_bytes(
                       source.stream(),
                       cuda::std::span{source.data(), destination.size()},
                       cuda::std::span{destination.data(), destination.size()}
                   );
                   source.stream().sync();
               }
    ).has_value();
}

TEST(SketchTest, PublicMetadataUsesKBeforeBucket) {
    using sketch_type = cuddl::sketch<25, 2048>;
    static_assert(!std::is_trivially_copyable_v<sketch_type>);
    static_assert(!std::is_default_constructible_v<sketch_type>);
    EXPECT_EQ(sketch_type::bucket_count(), 2048);
    EXPECT_EQ(sketch_type::kmer_length(), 25);
}

TEST(CudaRuntime, ConvertsExceptionsAndPreservesValues) {
    auto value = cuddl::cuda_try([] { return std::make_unique<int>(42); });
    ASSERT_TRUE(value);
    EXPECT_EQ(**value, 42);
    EXPECT_TRUE(cuddl::cuda_try([] {}));
    EXPECT_TRUE(cuddl::cuda_try([] { return cudaSuccess; }));

    auto failure =
        cuddl::cuda_try([] { throw cuda::cuda_error(cudaErrorInvalidDevice, "invalid device"); });
    ASSERT_FALSE(failure);
    ASSERT_NE(failure.error().as_cuda(), nullptr);
    EXPECT_EQ(failure.error().as_cuda()->code, cudaErrorInvalidDevice);
    EXPECT_FALSE(failure.error().as_cuda()->location.file.empty());

    auto exhausted = cuddl::cuda_try([] { throw std::bad_alloc{}; });
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.error().category(), cuddl::ErrorCategory::resource);
}

TEST(SketchTest, BufferMovesAndResultsAcrossNonblockingStreams) {
    cuda::stream producer{cuda::devices[0]};
    cuda::stream consumer{cuda::devices[0]};
    auto const host = make_inputs(65536);
    auto input = cuda::make_device_buffer<uint64_t>(producer, producer.device(), host);
    using sketch_type = cuddl::sketch<k_default, b_default>;
    sketch_type original(producer);
    sketch_type assigned(producer);
    ASSERT_TRUE(original.add_async({input.data(), input.size()}, producer));
    consumer.wait(producer);

    sketch_type moved(std::move(original));
    EXPECT_EQ(original.data().data(), nullptr);
    assigned = std::move(moved);
    EXPECT_EQ(moved.data().data(), nullptr);

    auto counts = assigned.winner_counts(consumer);
    ASSERT_TRUE(counts);
    EXPECT_FALSE(counts->second);
    scalar_sketch<b_default> oracle;
    oracle.add(host);
    oracle.pack_registers();
    for (size_t i = 0; i < b_default; ++i) {
        EXPECT_EQ(counts->first[i], static_cast<uint16_t>(oracle.registers[i]));
    }
    ASSERT_TRUE(assigned.clear(consumer));
    auto cleared = assigned.winner_counts(consumer);
    ASSERT_TRUE(cleared);
    EXPECT_EQ(cleared->first, std::vector<uint16_t>(b_default, 0));
    producer.wait(consumer);
    producer.sync();
}

TEST_F(ReferenceDatabaseTest, BufferMoveAssignmentRetainsRowsAndMetadata) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto rows =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), 2U * b_default, uint16_t{7});
    auto source = database_type::build_async(rows, compatibility, stream_);
    auto destination = database_type::build_async(
        cuddl::device_span<uint16_t const>{rows.data(), b_default}, compatibility, stream_
    );
    ASSERT_TRUE(source);
    ASSERT_TRUE(destination);
    auto const* original_data = source->data().data();
    *destination = std::move(*source);
    EXPECT_EQ(destination->data().data(), original_data);
    EXPECT_EQ(destination->reference_count(), 2U);
    EXPECT_EQ(source->reference_count(), 0U);
    EXPECT_TRUE(source->data().empty());
    std::vector<uint16_t> copied(rows.size());
    cuda::copy_bytes(stream_, destination->data(), copied);
    stream_.sync();
    EXPECT_EQ(copied, std::vector<uint16_t>(rows.size(), uint16_t{7}));
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
    cuda::stream stream{cuda::devices[0]};
    using layout = cuddl::register_layout<5, 11>;
    using sketch_type = cuddl::sketch<k_default, b_default, layout>;
    static_assert(std::is_empty_v<layout>);
    static_assert(std::is_same_v<typename sketch_type::register_type, uint32_t>);
    EXPECT_EQ(score<layout>(0ULL), std::numeric_limits<uint16_t>::max());
    EXPECT_EQ(restore<layout>(std::numeric_limits<uint16_t>::max()), 1ULL << 32U);

    auto const inputs = make_inputs(50000);
    cuddl::sketch<k_default, b_default, layout> gpu(stream);
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}, stream).has_value());

    scalar_sketch<b_default, layout> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            stream,
            cuda::std::span{gpu.data().data(), b_default},
            cuda::std::span{gpu_regs.data(), b_default}
        );
        stream.sync();
    }()));
    EXPECT_EQ(gpu_regs, oracle.registers);

    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default, layout>();
    EXPECT_EQ(compatibility.exponent_bits, 5U);
    EXPECT_EQ(compatibility.mantissa_bits, 11U);
    EXPECT_TRUE(gpu.cardinality(stream).has_value());
    EXPECT_TRUE(gpu.hybrid_cardinality(stream).has_value());
    EXPECT_TRUE(gpu.summary<true>(gpu, stream).has_value());
}

TEST(SketchTest, GpuRegistersMatchScalarOracleByteIdentically) {
    cuda::stream stream{cuda::devices[0]};
    auto const inputs = make_inputs(50000);
    cuddl::sketch<k_default, b_default> gpu(stream);
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}, stream).has_value());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            stream,
            cuda::std::span{gpu.data().data(), b_default},
            cuda::std::span{gpu_regs.data(), b_default}
        );
        stream.sync();
    }()));
    EXPECT_EQ(gpu_regs, oracle.registers);
}

// Sequential aggregation isolates the block histogram/reduction from the estimator formulas.
template <typename Layout>
__global__ void scalar_hybrid_cardinality(
    uint32_t const* registers,
    size_t size,
    cuddl::hybrid_cardinality_estimates* result
) {
    uint32_t bins[cuddl::detail::nlz_bins]{};
    float restored = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        auto const stored = winner(registers[i]);
        ++bins[stored == 0U ? 0U : (stored >> Layout::mantissa_bits) + 1U];
        if (stored != 0U) {
            restored += static_cast<float>(restore<Layout>(stored));
        }
    }
    *result = cuddl::detail::hybrid_estimates_f32(bins, static_cast<float>(size), restored);
}

TEST(SketchTest, HybridBlockPrimitivesMatchSequentialAggregation) {
    cuda::stream stream{cuda::devices[0]};
    auto check_layout = [&]<typename Layout> {
        cuddl::sketch<k_default, b_default, Layout> sketch(stream);
        auto expected = cuda::make_device_buffer<cuddl::hybrid_cardinality_estimates>(
            stream, stream.device(), 1, cuda::no_init
        );
        auto variants = cuda::make_device_buffer<double>(stream, stream.device(), 2, cuda::no_init);
        for (int pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint32_t> registers(b_default);
            for (size_t i = 0; i < registers.size(); ++i) {
                if (pattern == 0 || (pattern == 1 && i % 3 == 0)) {
                    continue;
                }
                auto const max_exponent =
                    std::min(63U - Layout::mantissa_bits, (1U << Layout::exponent_bits) - 1U);
                auto const exponent = i % (max_exponent + 1U);
                registers[i] =
                    pack(static_cast<uint16_t>((exponent << Layout::mantissa_bits) | 1U), 1U);
            }
            cuda::copy_bytes(stream, registers, sketch.data());
            scalar_hybrid_cardinality<Layout>
                <<<1, 1, 0, stream.get()>>>(sketch.data().data(), b_default, expected.data());
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            cuddl::detail::hybrid_cardinality_variant_kernel<
                b_default,
                cuddl::detail::hybrid_variant::bbtools,
                Layout><<<1, cuddl::detail::block_size, 0, stream.get()>>>(
                sketch.data().data(), variants.data()
            );
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            cuddl::detail::hybrid_cardinality_variant_kernel<
                b_default,
                cuddl::detail::hybrid_variant::paper,
                Layout><<<1, cuddl::detail::block_size, 0, stream.get()>>>(
                sketch.data().data(), variants.data() + 1
            );
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            auto actual = sketch.hybrid_cardinality(stream);
            ASSERT_TRUE(actual);
            std::vector<cuddl::hybrid_cardinality_estimates> oracle;
            std::vector<double> separate;
            ASSERT_TRUE(copy_device_buffer(expected, oracle));
            ASSERT_TRUE(copy_device_buffer(variants, separate));
            for (auto member :
                 {&cuddl::hybrid_cardinality_estimates::bbtools,
                  &cuddl::hybrid_cardinality_estimates::paper,
                  &cuddl::hybrid_cardinality_estimates::lc,
                  &cuddl::hybrid_cardinality_estimates::dlc,
                  &cuddl::hybrid_cardinality_estimates::mean_m_raw}) {
                EXPECT_NEAR(
                    (*actual).*member,
                    oracle[0].*member,
                    1e-5 * std::max(1.0, std::abs(oracle[0].*member))
                );
            }
            EXPECT_DOUBLE_EQ(separate[0], actual->bbtools);
            EXPECT_DOUBLE_EQ(separate[1], actual->paper);
        }
    };
    check_layout.template operator()<cuddl::default_register_layout>();
    check_layout.template operator()<cuddl::register_layout<5, 11>>();
}

TEST(SketchTest, UnalignedInputSpanningMultipleGridStrideIterationsMatchesScalarOracle) {
    cuda::stream stream{cuda::devices[0]};
    // An element-aligned but not 32-byte-aligned span forces the scalar path of
    // add_shared_kernel; the span must exceed one grid-stride iteration so an early exit would
    // drop inputs and diverge from the oracle. To make a dropped tail observable, the first
    // stride's worth of items avoids one bucket entirely and the eight items beyond the stride
    // are guaranteed winners in that bucket: a buggy kernel that stops after its first chunk
    // leaves the bucket empty.
    auto const multiprocessors =
        cuda::devices[0].attribute(cuda::device_attributes::multiprocessor_count);
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
    auto padded =
        cuda::make_device_buffer<uint64_t>(stream, stream.device(), inputs.size() + 1U, uint64_t{});
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            stream,
            cuda::std::span{inputs.data(), inputs.size()},
            cuda::std::span{padded.data() + 1, inputs.size()}
        );
        stream.sync();
    }()));
    auto const unaligned = cuddl::device_span<uint64_t const>{padded.data() + 1, inputs.size()};

    cuddl::sketch<k_default, b_default> gpu(stream);
    ASSERT_TRUE(gpu.add_async(unaligned, stream).has_value());
    ASSERT_NO_THROW(stream.sync());

    scalar_sketch<b_default> oracle;
    oracle.add(inputs);
    oracle.pack_registers();

    std::vector<uint32_t> gpu_regs(b_default);
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            stream,
            cuda::std::span{gpu.data().data(), b_default},
            cuda::std::span{gpu_regs.data(), b_default}
        );
        stream.sync();
    }()));
    EXPECT_EQ(gpu_regs, oracle.registers);
}

TEST(SketchTest, ComparisonCountsMatchScalarOracle) {
    cuda::stream stream{cuda::devices[0]};
    auto const a = make_inputs(40000, 0x1111'1111'1111'1111ULL);
    auto const b = make_inputs(30000, 0x2222'2222'2222'2222ULL);
    // Overlap half of A into B so the pair shares a large common prefix of k-mers.
    std::vector<uint64_t> shared(a.begin(), a.begin() + 20000);
    std::vector<uint64_t> b_joined = shared;
    b_joined.insert(b_joined.end(), b.begin(), b.end());

    cuddl::sketch<k_default, b_default> gpu_a(stream);
    cuddl::sketch<k_default, b_default> gpu_b(stream);
    ASSERT_TRUE(gpu_a.add({a.data(), a.size()}, stream).has_value());
    ASSERT_TRUE(gpu_b.add({b_joined.data(), b_joined.size()}, stream).has_value());

    auto const gpu_summary = gpu_a.compare(gpu_b, stream);
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
    cuda::stream stream{cuda::devices[0]};
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

    auto device_left = cuda::make_device_buffer<uint32_t>(stream, stream.device(), left_registers);
    auto device_right =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), right_registers);
    auto device_outputs = cuda::make_device_buffer<cuddl::pairwise_summary>(
        stream, stream.device(), pair_count, cuddl::pairwise_summary{}
    );
    auto const result = cuddl::compare_batch_async<b_default>(
        {device_left.data(), device_left.size()},
        {device_right.data(), device_right.size()},
        {device_outputs.data(), device_outputs.size()},
        stream
    );
    ASSERT_TRUE(result.has_value());
    ASSERT_NO_THROW(stream.sync());

    std::vector<cuddl::pairwise_summary> outputs;
    ASSERT_TRUE(copy_device_buffer(device_outputs, outputs));
    for (size_t pair = 0; pair < pair_count; ++pair) {
        auto const begin = left_scores.begin() + static_cast<ptrdiff_t>(pair * b_default);
        std::vector<uint16_t> left_row(begin, begin + b_default);
        EXPECT_EQ(outputs[pair], score_row_oracle(left_row, right_scores, pair));
    }

    auto const unequal = cuddl::compare_batch_async<b_default>(
        {device_left.data(), device_left.size()},
        {device_right.data(), device_right.size() - 1U},
        {device_outputs.data(), device_outputs.size()},
        stream
    );
    EXPECT_FALSE(unequal.has_value());
    EXPECT_EQ(unequal.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto const undersized = cuddl::compare_batch_async<b_default>(
        {device_left.data(), device_left.size()},
        {device_right.data(), device_right.size()},
        {device_outputs.data(), pair_count - 1U},
        stream
    );
    EXPECT_FALSE(undersized.has_value());
    auto const null_input = cuddl::compare_batch_async<b_default>(
        cuddl::device_span<uint32_t const>{static_cast<uint32_t const*>(nullptr), b_default},
        {device_right.data(), b_default},
        {device_outputs.data(), 1U},
        stream
    );
    EXPECT_FALSE(null_input.has_value());
    EXPECT_TRUE(
        cuddl::compare_batch_async<b_default>(
            cuddl::device_span<uint32_t const>{},
            cuddl::device_span<uint32_t const>{},
            cuddl::device_span<cuddl::pairwise_summary>{},
            stream
        )
            .has_value()
    );
}

TEST_F(ReferenceDatabaseTest, SparseAndDenseIndexesAgreeAcrossLayoutsAndKeyWidths) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t references = 33U;
    std::array<uint16_t, 7> const values{0U, 1U, 9U, 0x4000U, 0x8000U, 0xc000U, 0xffffU};
    std::vector<uint16_t> rows(references * b_default);
    std::vector<uint32_t> packed(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = values[(i / b_default + i % b_default) % values.size()];
        packed[i] = pack(rows[i], rows[i] == 0U ? 0U : 3U);
    }
    std::vector<uint16_t> query(rows.begin(), rows.begin() + b_default);
    auto compact_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto packed_rows = cuda::make_device_buffer<uint32_t>(stream, stream.device(), packed);
    auto saturation =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), references, uint32_t{});
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto dense_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), references, cuda::no_init
    );
    auto sparse_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), references, cuda::no_init
    );
    auto dense_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    auto sparse_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    for (bool use_packed : {false, true}) {
        for (uint16_t mask : {uint16_t{0x7fffU}, uint16_t{0xffffU}}) {
            for (uint32_t buckets : {uint32_t{b_default / 2U}, uint32_t{b_default}}) {
                auto compatibility = cuddl::score_compatibility::current<k_default, b_default>();
                compatibility.key_mask = mask;
                compatibility.indexed_bucket_count = buckets;
                auto build = [&](cuddl::index_storage storage) {
                    return use_packed ? database_type::build_indexed_async(
                                            packed_rows, saturation, compatibility, stream, storage
                                        )
                                      : database_type::build_indexed_async(
                                            compact_rows, compatibility, stream, storage
                                        );
                };
                auto dense_built = build(cuddl::index_storage::dense);
                auto sparse_built = build(cuddl::index_storage::sparse);
                ASSERT_TRUE(dense_built) << dense_built.error().message();
                ASSERT_TRUE(sparse_built) << sparse_built.error().message();
                auto dense = std::move(*dense_built);
                auto sparse = std::move(*sparse_built);
                // Exercise move assignment with a live index allocation on both sides.
                auto replacement = build(cuddl::index_storage::sparse);
                ASSERT_TRUE(replacement);
                sparse = std::move(*replacement);
                EXPECT_EQ(
                    sparse.persistent_index_bytes(),
                    size_t{buckets} * references * (sizeof(uint16_t) + sizeof(uint32_t))
                );
                EXPECT_LT(sparse.persistent_index_bytes(), dense.persistent_index_bytes());
                auto required = dense.indexed_single_query_workspace_bytes(stream);
                ASSERT_TRUE(required);
                auto workspace = cuda::make_device_buffer<uint8_t>(
                    stream, stream.device(), *required, cuda::no_init
                );
                for (uint32_t threshold : {0U, 1U, 5U, buckets}) {
                    ASSERT_TRUE(dense.search_indexed_async(
                        device_query,
                        compatibility,
                        workspace,
                        dense_results,
                        dense_count,
                        {.minimum_matches = threshold},
                        stream
                    ));
                    ASSERT_TRUE(sparse.search_indexed_async(
                        device_query,
                        compatibility,
                        workspace,
                        sparse_results,
                        sparse_count,
                        {.minimum_matches = threshold},
                        stream
                    ));
                    std::vector<uint32_t> dc, sc;
                    std::vector<cuddl::reference_search_result> dr, sr;
                    ASSERT_TRUE(copy_device_buffer(dense_count, dc));
                    ASSERT_TRUE(copy_device_buffer(sparse_count, sc));
                    ASSERT_EQ(dc, sc);
                    // Copy only the initialized result prefix.
                    dr.resize(dc[0]);
                    sr.resize(sc[0]);
                    cuda::copy_bytes(stream, cuda::std::span{dense_results.data(), dr.size()}, dr);
                    cuda::copy_bytes(stream, cuda::std::span{sparse_results.data(), sr.size()}, sr);
                    stream.sync();
                    EXPECT_EQ(dr, sr);
                }
            }
        }
    }
}

TEST_F(ReferenceDatabaseTest, IndexCountsInitializeTrailingScanElement) {
    auto const stream = cuda::stream_ref{stream_};
    constexpr uint32_t buckets = 3U;
    constexpr uint32_t keys = 1U << 15U;
    std::array<uint16_t, buckets> const host_scores{0U, 0x8000U, 0xffffU};
    auto scores = cuda::make_device_buffer<uint16_t>(stream, stream.device(), host_scores);
    auto counts = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), buckets * keys + 1U, uint32_t{42}
    );
    cuddl::detail::count_index_cells_bucket_kernel<<<
        2, 1024, keys / 4U * sizeof(uint32_t), stream.get()>>>(
        scores.data(), buckets, 1U, 0x7fffU, counts.data()
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    uint32_t trailing = 42U;
    cuda::copy_bytes(
        stream,
        cuda::std::span{counts.data() + buckets * keys, size_t{1}},
        cuda::std::span{&trailing, size_t{1}}
    );
    stream.sync();
    EXPECT_EQ(trailing, 0U);
}

TEST_F(ReferenceDatabaseTest, ExhaustiveSearchMatchesScalarOracle) {
    auto const stream = cuda::stream_ref{stream_};
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

    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto device_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto workspace = cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);

    auto built = cuddl::reference_database<k_default, b_default>::build_async(
        device_rows, compatibility, stream
    );
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    ASSERT_TRUE(database
                    .search_async(device_query, compatibility, workspace, device_results, stream)
                    .has_value());

    std::vector<cuddl::reference_search_result> results(reference_count);
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                device_results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            },
            cuda::std::span{
                results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        EXPECT_EQ(results[reference_id].reference_id, reference_id);
        EXPECT_EQ(results[reference_id].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, PackedRowsPreserveMultiplicityWithoutChangingSearch) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 4;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

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

    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto device_scores = cuda::make_device_buffer<uint16_t>(stream, stream.device(), scores);
    auto device_packed = cuda::make_device_buffer<uint32_t>(stream, stream.device(), packed);
    auto device_saturation =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), saturation);
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

    auto compact_exhaustive = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto packed_exhaustive = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto compact_indexed = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto packed_indexed = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto compact_result_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, uint32_t{});
    auto packed_result_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, uint32_t{});
    auto workspace_bytes = compact.indexed_single_query_workspace_bytes(stream);
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    auto compact_workspace =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
    auto packed_workspace =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});

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
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                compact_exhaustive.data(),
                (compact_exhaustive_host.size() * sizeof(compact_exhaustive_host.front())) /
                    sizeof(*(compact_exhaustive.data()))
            },
            cuda::std::span{
                compact_exhaustive_host.data(),
                (compact_exhaustive_host.size() * sizeof(compact_exhaustive_host.front())) /
                    sizeof(*(compact_exhaustive.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                packed_exhaustive.data(),
                (packed_exhaustive_host.size() * sizeof(packed_exhaustive_host.front())) /
                    sizeof(*(packed_exhaustive.data()))
            },
            cuda::std::span{
                packed_exhaustive_host.data(),
                (packed_exhaustive_host.size() * sizeof(packed_exhaustive_host.front())) /
                    sizeof(*(packed_exhaustive.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                compact_indexed.data(),
                (compact_indexed_host.size() * sizeof(compact_indexed_host.front())) /
                    sizeof(*(compact_indexed.data()))
            },
            cuda::std::span{
                compact_indexed_host.data(),
                (compact_indexed_host.size() * sizeof(compact_indexed_host.front())) /
                    sizeof(*(compact_indexed.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                packed_indexed.data(),
                (packed_indexed_host.size() * sizeof(packed_indexed_host.front())) /
                    sizeof(*(packed_indexed.data()))
            },
            cuda::std::span{
                packed_indexed_host.data(),
                (packed_indexed_host.size() * sizeof(packed_indexed_host.front())) /
                    sizeof(*(packed_indexed.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{compact_result_count.data(), size_t{1}},
            cuda::std::span{&compact_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{packed_result_count.data(), size_t{1}},
            cuda::std::span{&packed_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                packed_database.packed_data().data(),
                (packed_database.packed_data().size_bytes()) /
                    sizeof(*(packed_database.packed_data().data()))
            },
            cuda::std::span{
                recovered_packed.data(),
                (packed_database.packed_data().size_bytes()) /
                    sizeof(*(packed_database.packed_data().data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                packed_database.saturation_states().data(),
                (packed_database.saturation_states().size_bytes()) /
                    sizeof(*(packed_database.saturation_states().data()))
            },
            cuda::std::span{
                recovered_saturation.data(),
                (packed_database.saturation_states().size_bytes()) /
                    sizeof(*(packed_database.saturation_states().data()))
            }
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());

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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 3U;
    constexpr uint32_t query_count = 3U;
    constexpr uint32_t query_id_offset = 41U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

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

    auto device_queries = cuda::make_device_buffer<uint16_t>(stream, stream.device(), queries);
    auto device_scores = cuda::make_device_buffer<uint16_t>(stream, stream.device(), scores);
    auto device_packed = cuda::make_device_buffer<uint32_t>(stream, stream.device(), packed);
    auto device_saturation =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), reference_count, 0U);
    auto compact_built = database_type::build_indexed_async(device_scores, compatibility, stream);
    auto packed_built =
        database_type::build_indexed_async(device_packed, device_saturation, compatibility, stream);
    ASSERT_TRUE(compact_built.has_value()) << compact_built.error().message();
    ASSERT_TRUE(packed_built.has_value()) << packed_built.error().message();
    auto compact = std::move(*compact_built);
    auto packed_database = std::move(*packed_built);

    auto compact_exhaustive_requirements = compact.batch_search_requirements(query_count, stream);
    auto packed_exhaustive_requirements =
        packed_database.batch_search_requirements(query_count, stream);
    auto compact_indexed_requirements =
        compact.indexed_batch_search_requirements(query_count, stream);
    auto packed_indexed_requirements =
        packed_database.indexed_batch_search_requirements(query_count, stream);
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

    auto compact_exhaustive_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), compact_exhaustive_requirements->workspace_bytes, uint8_t{}
    );
    auto packed_exhaustive_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), packed_exhaustive_requirements->workspace_bytes, uint8_t{}
    );
    auto compact_indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), compact_indexed_requirements->workspace_bytes, uint8_t{}
    );
    auto packed_indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), packed_indexed_requirements->workspace_bytes, uint8_t{}
    );
    auto const pair_capacity = compact_exhaustive_requirements->maximum_pair_count;
    auto compact_exhaustive = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto packed_exhaustive = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto compact_indexed = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto packed_indexed = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto compact_exhaustive_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto packed_exhaustive_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto compact_indexed_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto packed_indexed_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto compact_exhaustive_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto packed_exhaustive_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto compact_indexed_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto packed_indexed_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});

    ASSERT_TRUE(compact
                    .search_batch_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        compact_exhaustive_workspace,
                        compact_exhaustive,
                        compact_exhaustive_count,
                        [](uint32_t) {},
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
                        [](uint32_t) {},
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
                        [](uint32_t) {},
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
                        [](uint32_t) {},
                        packed_indexed_matches,
                        {.minimum_matches = 1U},
                        stream
                    )
                    .has_value());

    auto no_diagnostic_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto no_diagnostic_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    ASSERT_TRUE(compact
                    .search_batch_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        compact_exhaustive_workspace,
                        no_diagnostic_results,
                        no_diagnostic_count,
                        [](uint32_t) {},
                        {},
                        stream
                    )
                    .has_value());
    ASSERT_NO_THROW(stream_.sync());

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
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive, compact_exhaustive_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive, packed_exhaustive_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed, compact_indexed_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed, packed_indexed_host));
    ASSERT_TRUE(copy_device_buffer(no_diagnostic_results, no_diagnostic_host));
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive_matches, compact_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive_matches, packed_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed_matches, compact_indexed_matches_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed_matches, packed_indexed_matches_host));
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive_count, compact_exhaustive_count_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive_count, packed_exhaustive_count_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed_count, compact_indexed_count_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed_count, packed_indexed_count_host));
    ASSERT_TRUE(copy_device_buffer(no_diagnostic_count, no_diagnostic_count_host));

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
            auto one_query = cuda::make_device_buffer<uint16_t>(
                stream,
                stream.device(),
                cuda::std::span{
                    queries.begin() + query_id * b_default,
                    queries.begin() + (query_id + 1U) * b_default
                }
            );
            auto one_results = cuda::make_device_buffer<cuddl::reference_search_result>(
                stream, stream.device(), reference_count, cuddl::reference_search_result{}
            );
            auto one_workspace =
                cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);
            auto one_count =
                cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
            if (indexed) {
                auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
                if (!workspace_bytes.has_value()) {
                    ADD_FAILURE() << workspace_bytes.error().message();
                    return concatenated;
                }
                one_workspace = cuda::make_device_buffer<uint8_t>(
                    stream, stream.device(), *workspace_bytes, uint8_t{}
                );
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
            if (!cuddl::cuda_try([&] { stream_.sync(); })) {
                ADD_FAILURE();
                return concatenated;
            }
            std::vector<cuddl::reference_search_result> one_host;
            if (!copy_device_buffer(one_results, one_host)) {
                ADD_FAILURE();
                return concatenated;
            }
            if (indexed) {
                std::vector<uint32_t> one_count_host;
                if (!copy_device_buffer(one_count, one_count_host)) {
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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 2U;
    constexpr uint32_t query_count = 3U;
    constexpr uint32_t query_id_offset = 100U;
    constexpr uint32_t short_capacity = 2U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

    std::vector<uint16_t> queries(query_count * b_default);
    std::vector<uint16_t> rows(reference_count * b_default);
    for (size_t bucket = 0; bucket < b_default; ++bucket) {
        queries[bucket] = std::array<uint16_t, 4>{0U, 1U, 2U, 3U}[bucket % 4U];
        queries[b_default + bucket] = std::array<uint16_t, 4>{4U, 0U, 6U, 7U}[bucket % 4U];
        queries[2U * b_default + bucket] = std::array<uint16_t, 4>{8U, 9U, 0U, 11U}[bucket % 4U];
        rows[bucket] = queries[bucket];
        rows[b_default + bucket] = queries[b_default + bucket];
    }
    auto device_queries = cuda::make_device_buffer<uint16_t>(stream, stream.device(), queries);
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto requirements = database.indexed_batch_search_requirements(query_count, stream);
    ASSERT_TRUE(requirements.has_value()) << requirements.error().message();
    ASSERT_EQ(requirements->maximum_pair_count, query_count * reference_count);
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements->workspace_bytes, uint8_t{}
    );

    cuddl::batch_search_result const result_sentinel{
        .query_id = 0xdeadU,
        .reference_id = 0xbeefU,
        .summary = {.counts = {.lower = 7U, .equal = 8U, .higher = 9U, .both_empty = 10U}},
    };
    constexpr uint32_t match_sentinel = 0xabcdU;
    auto short_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), short_capacity, result_sentinel
    );
    auto short_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), short_capacity, match_sentinel);
    auto short_count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, 31337U);
    auto indexed_short = database.search_batch_indexed_async(
        device_queries,
        compatibility,
        query_id_offset,
        workspace,
        short_results,
        short_count,
        [](uint32_t) {},
        short_matches,
        {.minimum_matches = 0U},
        stream
    );
    ASSERT_TRUE(indexed_short.has_value()) << indexed_short.error().message();
    ASSERT_NO_THROW(stream_.sync());

    std::vector<cuddl::batch_search_result> short_results_host;
    std::vector<uint32_t> short_matches_host;
    std::vector<uint32_t> short_count_host;
    ASSERT_TRUE(copy_device_buffer(short_results, short_results_host));
    ASSERT_TRUE(copy_device_buffer(short_matches, short_matches_host));
    ASSERT_TRUE(copy_device_buffer(short_count, short_count_host));
    EXPECT_EQ(short_count_host.front(), requirements->maximum_pair_count);
    EXPECT_EQ(
        short_results_host, std::vector<cuddl::batch_search_result>(short_capacity, result_sentinel)
    );
    EXPECT_EQ(short_matches_host, std::vector<uint32_t>(short_capacity, match_sentinel));

    auto exhaustive_requirements = database.batch_search_requirements(query_count, stream);
    ASSERT_TRUE(exhaustive_requirements.has_value());
    auto exhaustive_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), exhaustive_requirements->workspace_bytes, uint8_t{}
    );
    auto exhaustive_short_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), requirements->maximum_pair_count - 1U, result_sentinel
    );
    auto exhaustive_short_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, 2718U);
    auto exhaustive_short_matches = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), requirements->maximum_pair_count - 1U, match_sentinel
    );
    auto exhaustive_short = database.search_batch_async(
        device_queries,
        compatibility,
        query_id_offset,
        exhaustive_workspace,
        exhaustive_short_results,
        exhaustive_short_count,
        [](uint32_t) {},
        exhaustive_short_matches,
        stream
    );
    ASSERT_FALSE(exhaustive_short.has_value());
    EXPECT_EQ(exhaustive_short.error().category(), cuddl::ErrorCategory::resource);
    std::vector<cuddl::batch_search_result> exhaustive_short_host;
    std::vector<uint32_t> exhaustive_short_matches_host;
    std::vector<uint32_t> exhaustive_short_count_host;
    ASSERT_TRUE(copy_device_buffer(exhaustive_short_results, exhaustive_short_host));
    ASSERT_TRUE(copy_device_buffer(exhaustive_short_matches, exhaustive_short_matches_host));
    ASSERT_TRUE(copy_device_buffer(exhaustive_short_count, exhaustive_short_count_host));
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

    auto successful_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), requirements->maximum_pair_count, cuddl::batch_search_result{}
    );
    auto successful_matches = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), requirements->maximum_pair_count, uint32_t{}
    );
    auto successful_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    ASSERT_TRUE(database
                    .search_batch_indexed_async(
                        device_queries,
                        compatibility,
                        query_id_offset,
                        workspace,
                        successful_results,
                        successful_count,
                        [](uint32_t) {},
                        successful_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_NO_THROW(stream_.sync());

    std::vector<cuddl::batch_search_result> successful_host;
    std::vector<uint32_t> successful_matches_host;
    std::vector<uint32_t> successful_count_host;
    ASSERT_TRUE(copy_device_buffer(successful_results, successful_host));
    ASSERT_TRUE(copy_device_buffer(successful_matches, successful_matches_host));
    ASSERT_TRUE(copy_device_buffer(successful_count, successful_count_host));
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
TEST_F(ReferenceDatabaseTest, BatchSearchOwnsBoundedTraversal) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 2U;
    constexpr uint32_t query_count = 130U;
    constexpr uint32_t query_id_offset = 1000U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    std::vector<uint16_t> host_rows(static_cast<size_t>(reference_count) * b_default, 7U);
    std::vector<uint16_t> host_queries(static_cast<size_t>(query_count) * b_default, 7U);
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), host_rows);
    auto device_queries = cuda::make_device_buffer<uint16_t>(stream, stream.device(), host_queries);
    auto database = *database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_NO_THROW(stream_.sync());

    auto const exhaustive_requirements = *database.batch_search_requirements(query_count, stream);
    auto const indexed_requirements =
        *database.indexed_batch_search_requirements(query_count, stream);
    EXPECT_EQ(exhaustive_requirements.maximum_pair_count, 128U * reference_count);
    EXPECT_EQ(indexed_requirements.maximum_pair_count, 128U * reference_count);

    auto collect = [&](bool indexed) {
        auto const& requirements = indexed ? indexed_requirements : exhaustive_requirements;
        auto workspace = cuda::make_device_buffer<uint8_t>(
            stream, stream.device(), requirements.workspace_bytes, uint8_t{}
        );
        auto results = cuda::make_device_buffer<cuddl::batch_search_result>(
            stream, stream.device(), requirements.maximum_pair_count, cuddl::batch_search_result{}
        );
        auto result_count =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
        std::vector<cuddl::batch_search_result> collected;
        std::vector<uint32_t> tile_capacities;
        auto on_tile = [&](uint32_t capacity) {
            ASSERT_NO_THROW(stream_.sync());
            uint32_t count = 0U;
            ASSERT_NO_THROW(([&] {
                cuda::copy_bytes(
                    stream,
                    cuda::std::span{result_count.data(), size_t{1}},
                    cuda::std::span{&count, size_t{1}}
                );
                stream.sync();
            }()));
            ASSERT_LE(count, capacity);
            std::vector<cuddl::batch_search_result> tile(count);
            if (count != 0U) {
                ASSERT_NO_THROW(([&] {
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{results.data(), count},
                        cuda::std::span{tile.data(), count}
                    );
                    stream.sync();
                }()));
            }
            tile_capacities.push_back(capacity);
            collected.insert(collected.end(), tile.begin(), tile.end());
        };
        auto const search = indexed ? database.search_batch_indexed_async(
                                          device_queries,
                                          compatibility,
                                          query_id_offset,
                                          workspace,
                                          results,
                                          result_count,
                                          on_tile,
                                          {},
                                          {.minimum_matches = 0U},
                                          cuda::stream_ref{stream_}
                                      )
                                    : database.search_batch_async(
                                          device_queries,
                                          compatibility,
                                          query_id_offset,
                                          workspace,
                                          results,
                                          result_count,
                                          on_tile,
                                          {},
                                          cuda::stream_ref{stream_}
                                      );
        EXPECT_TRUE(search.has_value()) << search.error().message();
        EXPECT_EQ(tile_capacities, (std::vector<uint32_t>{256U, 4U}));
        EXPECT_EQ(collected.size(), static_cast<size_t>(query_count) * reference_count);
        std::sort(collected.begin(), collected.end(), [](auto const& left, auto const& right) {
            return std::tie(left.query_id, left.reference_id) <
                   std::tie(right.query_id, right.reference_id);
        });
        for (uint32_t query = 0U; query < query_count; ++query) {
            for (uint32_t reference = 0U; reference < reference_count; ++reference) {
                auto const& result =
                    collected[static_cast<size_t>(query) * reference_count + reference];
                EXPECT_EQ(result.query_id, query_id_offset + query);
                EXPECT_EQ(result.reference_id, reference);
            }
        }
    };
    collect(false);
    collect(true);
}

TEST_F(ReferenceDatabaseTest, AllToAllSearchHasOneExactDirectionalOrientation) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 4U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

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

    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto device_packed = cuda::make_device_buffer<uint32_t>(stream, stream.device(), packed);
    auto saturation =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), reference_count, 0U);
    auto compact_built = database_type::build_indexed_async(device_rows, compatibility, stream);
    auto packed_built =
        database_type::build_indexed_async(device_packed, saturation, compatibility, stream);
    ASSERT_TRUE(compact_built.has_value()) << compact_built.error().message();
    ASSERT_TRUE(packed_built.has_value()) << packed_built.error().message();
    auto compact = std::move(*compact_built);
    auto packed_database = std::move(*packed_built);

    auto exhaustive_requirements = compact.all_to_all_search_requirements(stream);
    auto indexed_requirements = compact.indexed_all_to_all_search_requirements(stream);
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
    auto large_indexed_requirements = compact.indexed_batch_search_requirements(128U, stream);
    ASSERT_TRUE(large_indexed_requirements.has_value())
        << large_indexed_requirements.error().message();
    auto const dense_counter_bytes = static_cast<size_t>(128U) * reference_count * sizeof(uint32_t);
    EXPECT_LE(
        large_indexed_requirements->counter_bytes, dense_counter_bytes + 2U * sizeof(uint32_t)
    );
    EXPECT_LE(large_indexed_requirements->candidate_bytes, dense_counter_bytes);

    auto compact_exhaustive_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), exhaustive_requirements->workspace_bytes, uint8_t{}
    );
    auto packed_exhaustive_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), exhaustive_requirements->workspace_bytes, uint8_t{}
    );
    auto compact_indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), indexed_requirements->workspace_bytes, uint8_t{}
    );
    auto packed_indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), indexed_requirements->workspace_bytes, uint8_t{}
    );
    auto const pair_capacity = exhaustive_requirements->maximum_pair_count;
    auto compact_exhaustive = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto packed_exhaustive = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto compact_indexed = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto packed_indexed = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), pair_capacity, cuddl::batch_search_result{}
    );
    auto compact_exhaustive_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto packed_exhaustive_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto compact_indexed_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto packed_indexed_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    auto compact_exhaustive_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto packed_exhaustive_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto compact_indexed_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});
    auto packed_indexed_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), pair_capacity, uint32_t{});

    ASSERT_TRUE(compact
                    .search_all_to_all_async(
                        compact_exhaustive_workspace,
                        compact_exhaustive,
                        compact_exhaustive_count,
                        [](uint32_t) {},
                        compact_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_all_to_all_async(
                        packed_exhaustive_workspace,
                        packed_exhaustive,
                        packed_exhaustive_count,
                        [](uint32_t) {},
                        packed_exhaustive_matches,
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(compact
                    .search_all_to_all_indexed_async(
                        compact_indexed_workspace,
                        compact_indexed,
                        compact_indexed_count,
                        [](uint32_t) {},
                        compact_indexed_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_TRUE(packed_database
                    .search_all_to_all_indexed_async(
                        packed_indexed_workspace,
                        packed_indexed,
                        packed_indexed_count,
                        [](uint32_t) {},
                        packed_indexed_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_NO_THROW(stream_.sync());

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
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive, compact_exhaustive_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive, packed_exhaustive_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed, compact_indexed_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed, packed_indexed_host));
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive_matches, compact_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive_matches, packed_exhaustive_matches_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed_matches, compact_indexed_matches_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed_matches, packed_indexed_matches_host));
    ASSERT_TRUE(copy_device_buffer(compact_exhaustive_count, compact_exhaustive_count_host));
    ASSERT_TRUE(copy_device_buffer(packed_exhaustive_count, packed_exhaustive_count_host));
    ASSERT_TRUE(copy_device_buffer(compact_indexed_count, compact_indexed_count_host));
    ASSERT_TRUE(copy_device_buffer(packed_indexed_count, packed_indexed_count_host));

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

    auto empty_queries =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), 0, cuda::no_init);
    auto empty_exhaustive_workspace =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);
    auto empty_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), 0, cuda::no_init
    );
    auto empty_count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, 77U);
    auto empty_matches =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 0, cuda::no_init);
    ASSERT_TRUE(compact
                    .search_batch_async(
                        empty_queries,
                        compatibility,
                        55U,
                        empty_exhaustive_workspace,
                        empty_results,
                        empty_count,
                        [](uint32_t) {},
                        empty_matches,
                        stream
                    )
                    .has_value());
    auto empty_indexed_requirements = compact.indexed_batch_search_requirements(0U, stream);
    ASSERT_TRUE(empty_indexed_requirements.has_value());
    auto empty_indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), empty_indexed_requirements->workspace_bytes, uint8_t{}
    );
    ASSERT_TRUE(compact
                    .search_batch_indexed_async(
                        empty_queries,
                        compatibility,
                        55U,
                        empty_indexed_workspace,
                        empty_results,
                        empty_count,
                        [](uint32_t) {},
                        empty_matches,
                        {.minimum_matches = 0U},
                        stream
                    )
                    .has_value());
    ASSERT_NO_THROW(stream_.sync());
    std::vector<uint32_t> empty_count_host;
    ASSERT_TRUE(copy_device_buffer(empty_count, empty_count_host));
    EXPECT_EQ(empty_count_host.front(), 0U);

    constexpr uint32_t partial_query_count = 2U;
    auto partial_queries = cuda::make_device_buffer<uint16_t>(
        stream, stream.device(), cuda::std::span{rows.begin() + 2U * b_default, rows.end()}
    );
    auto partial_requirements = compact.batch_search_requirements(partial_query_count, stream);
    ASSERT_TRUE(partial_requirements.has_value());
    auto partial_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), partial_requirements->workspace_bytes, uint8_t{}
    );
    auto partial_results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream,
        stream.device(),
        partial_requirements->maximum_pair_count,
        cuddl::batch_search_result{}
    );
    auto partial_matches = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), partial_requirements->maximum_pair_count, uint32_t{}
    );
    auto partial_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    ASSERT_TRUE(compact
                    .search_batch_async(
                        partial_queries,
                        compatibility,
                        2U,
                        partial_workspace,
                        partial_results,
                        partial_count,
                        [](uint32_t) {},
                        partial_matches,
                        stream
                    )
                    .has_value());
    ASSERT_NO_THROW(stream_.sync());
    std::vector<cuddl::batch_search_result> partial_host;
    std::vector<uint32_t> partial_matches_host;
    std::vector<uint32_t> partial_count_host;
    ASSERT_TRUE(copy_device_buffer(partial_results, partial_host));
    ASSERT_TRUE(copy_device_buffer(partial_matches, partial_matches_host));
    ASSERT_TRUE(copy_device_buffer(partial_count, partial_count_host));
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
}

TEST_F(ReferenceDatabaseTest, ExhaustiveAllToAllOwnsBoundedTraversal) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr uint32_t reference_count = 130U;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto rows = cuda::make_device_buffer<uint16_t>(
        stream, stream.device(), static_cast<size_t>(reference_count) * b_default, 0U
    );
    auto built = database_type::build_async(rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto requirements = database.all_to_all_search_requirements(stream);
    ASSERT_TRUE(requirements.has_value()) << requirements.error().message();

    auto const total_pairs = static_cast<uint64_t>(reference_count) * (reference_count - 1U) / 2U;
    ASSERT_LT(requirements->maximum_pair_count, total_pairs);
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements->workspace_bytes, uint8_t{}
    );
    auto results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), requirements->maximum_pair_count, cuddl::batch_search_result{}
    );
    auto result_count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
    uint64_t emitted_pairs = 0U;
    uint32_t callback_count = 0U;
    auto searched = database.search_all_to_all_async(
        workspace,
        results,
        result_count,
        [&](uint32_t pair_count) {
            emitted_pairs += pair_count;
            ++callback_count;
        },
        {},
        stream
    );
    ASSERT_TRUE(searched.has_value()) << searched.error().message();
    ASSERT_NO_THROW(stream_.sync());
    EXPECT_GT(callback_count, 1U);
    EXPECT_EQ(emitted_pairs, total_pairs);
}

TEST_F(ReferenceDatabaseTest, PackedRowsShareConstructionFailureContract) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

    auto malformed_scores =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), b_default + 1U, 1U);
    auto malformed_packed =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), b_default + 1U, pack(1U, 1U));
    auto saturation = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, 0U);
    auto compact_malformed = database_type::build_async(
        cuddl::device_span<uint16_t const>{malformed_scores.data(), malformed_scores.size()},
        compatibility,
        stream
    );
    auto packed_malformed = database_type::build_async(
        cuddl::device_span<uint32_t const>{malformed_packed.data(), malformed_packed.size()},
        cuddl::device_span<uint32_t const>{saturation.data(), saturation.size()},
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

    auto complete_packed =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 2U * b_default, pack(1U, 1U));
    auto mismatched_saturation = database_type::build_async(
        cuddl::device_span<uint32_t const>{complete_packed.data(), complete_packed.size()},
        cuddl::device_span<uint32_t const>{saturation.data(), saturation.size()},
        compatibility,
        stream
    );
    ASSERT_FALSE(mismatched_saturation.has_value());
    EXPECT_EQ(mismatched_saturation.error().category(), cuddl::ErrorCategory::invalid_argument);
}

TEST_F(ReferenceDatabaseTest, EmptyDatabaseReportsSizesAndReturnsNoResults) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    static_assert(!std::is_trivially_copyable_v<database_type>);

    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
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

    auto query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), b_default, 0U);
    auto const searched =
        database.search_async({query.data(), query.size()}, compatibility, {}, {}, stream);
    ASSERT_TRUE(searched.has_value()) << searched.error().message();
    EXPECT_NO_THROW(stream_.sync());
}

TEST_F(ReferenceDatabaseTest, RejectsMalformedAndIncompatibleInputsWithoutOutput) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

    auto malformed_rows =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), b_default + 1, 1U);
    auto malformed = database_type::build_async(
        {malformed_rows.data(), malformed_rows.size()}, compatibility, stream
    );
    ASSERT_FALSE(malformed.has_value());
    EXPECT_EQ(malformed.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto unsupported = compatibility;
    unsupported.key_mask = 0x7fffU;
    auto unsupported_build =
        database_type::build_async(cuddl::device_span<uint16_t const>{}, unsupported, stream);
    ASSERT_FALSE(unsupported_build.has_value());
    EXPECT_EQ(unsupported_build.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), 2 * b_default, 4U);
    auto query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), b_default, 4U);
    auto built = database_type::build_async({rows.data(), rows.size()}, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    cuddl::reference_search_result sentinel{};
    sentinel.reference_id = 99U;
    sentinel.summary.counts.lower = 123U;
    auto results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), 1, sentinel
    );

    auto malformed_query = database.search_async(
        {query.data(), query.size() - 1},
        compatibility,
        {},
        {results.data(), results.size()},
        stream
    );
    ASSERT_FALSE(malformed_query.has_value());
    EXPECT_EQ(malformed_query.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto incompatible = compatibility;
    ++incompatible.hash_seed;
    auto incompatible_query = database.search_async(
        {query.data(), query.size()}, incompatible, {}, {results.data(), results.size()}, stream
    );
    ASSERT_FALSE(incompatible_query.has_value());
    EXPECT_EQ(incompatible_query.error().category(), cuddl::ErrorCategory::invalid_argument);

    auto insufficient = database.search_async(
        {query.data(), query.size()}, compatibility, {}, {results.data(), results.size()}, stream
    );
    ASSERT_FALSE(insufficient.has_value());
    EXPECT_EQ(insufficient.error().category(), cuddl::ErrorCategory::resource);

    cuddl::reference_search_result observed{};
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{results.data(), size_t{1}},
            cuda::std::span{&observed, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
    EXPECT_EQ(observed, sentinel);
}

TEST_F(ReferenceDatabaseTest, IndexedSearchDefaultThresholdMatchesOracle) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 6;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

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

    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);
    auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    auto workspace =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
    auto device_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto device_result_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, uint32_t{});

    auto searched = database.search_indexed_async(
        device_query, compatibility, workspace, device_results, device_result_count, {}, stream
    );
    ASSERT_TRUE(searched.has_value()) << searched.error().message();

    uint32_t result_count = 0;
    std::vector<cuddl::reference_search_result> results(reference_count);
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{device_result_count.data(), size_t{1}},
            cuda::std::span{&result_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                device_results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            },
            cuda::std::span{
                results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
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
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{device_result_count.data(), size_t{1}},
            cuda::std::span{&result_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                device_results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            },
            cuda::std::span{
                results.data(),
                (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
    ASSERT_EQ(result_count, 4U);
    constexpr std::array<uint32_t, 4> threshold_ids{0U, 2U, 4U, 5U};
    for (size_t index = 0; index < result_count; ++index) {
        auto const reference_id = threshold_ids[index];
        EXPECT_EQ(results[index].reference_id, reference_id);
        EXPECT_EQ(results[index].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, IndexedSearchIsIndependentOfPostingOrder) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 6;
    constexpr std::array<std::array<uint32_t, reference_count>, 2> row_orders{{
        {0U, 1U, 2U, 3U, 4U, 5U},
        {5U, 2U, 4U, 0U, 3U, 1U},
    }};
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    std::vector<uint16_t> query(b_default, 0U);
    for (size_t bucket = 0; bucket < reference_count; ++bucket) {
        query[bucket] = static_cast<uint16_t>(bucket + 1U);
    }
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
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

        auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
        auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
        ASSERT_TRUE(built.has_value()) << built.error().message();
        auto database = std::move(*built);
        auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
        ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
        auto workspace =
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
        auto device_results = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), reference_count, cuddl::reference_search_result{}
        );
        auto device_result_count =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, uint32_t{});
        auto searched = database.search_indexed_async(
            device_query, compatibility, workspace, device_results, device_result_count, {}, stream
        );
        ASSERT_TRUE(searched.has_value()) << searched.error().message();

        uint32_t result_count = 0;
        std::vector<cuddl::reference_search_result> results(reference_count);
        ASSERT_NO_THROW(([&] {
            cuda::copy_bytes(
                cuda::stream_ref{stream_},
                cuda::std::span{device_result_count.data(), size_t{1}},
                cuda::std::span{&result_count, size_t{1}}
            );
        }()));
        ASSERT_NO_THROW(([&] {
            cuda::copy_bytes(
                cuda::stream_ref{stream_},
                cuda::std::span{
                    device_results.data(),
                    (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
                },
                cuda::std::span{
                    results.data(),
                    (results.size() * sizeof(results.front())) / sizeof(*(device_results.data()))
                }
            );
        }()));
        ASSERT_NO_THROW(stream_.sync());
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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 3;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();

    std::vector<uint16_t> query(b_default, 0U);
    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    rows[b_default] = 123U;
    rows[2U * b_default + 7U] = std::numeric_limits<uint16_t>::max();
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    auto workspace = cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);
    auto indexed_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto exhaustive_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, cuddl::reference_search_result{}
    );
    auto device_result_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, uint32_t{});
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
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{device_result_count.data(), size_t{1}},
            cuda::std::span{&result_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                indexed_results.data(),
                (indexed_host.size() * sizeof(indexed_host.front())) /
                    sizeof(*(indexed_results.data()))
            },
            cuda::std::span{
                indexed_host.data(),
                (indexed_host.size() * sizeof(indexed_host.front())) /
                    sizeof(*(indexed_results.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{
                exhaustive_results.data(),
                (exhaustive_host.size() * sizeof(exhaustive_host.front())) /
                    sizeof(*(exhaustive_results.data()))
            },
            cuda::std::span{
                exhaustive_host.data(),
                (exhaustive_host.size() * sizeof(exhaustive_host.front())) /
                    sizeof(*(exhaustive_results.data()))
            }
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
    ASSERT_EQ(result_count, reference_count);
    EXPECT_EQ(indexed_host, exhaustive_host);
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        EXPECT_EQ(indexed_host[reference_id].summary, score_row_oracle(query, rows, reference_id));
    }
}

TEST_F(ReferenceDatabaseTest, IndexedSearchModesMatchExhaustiveAcrossThresholds) {
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 7U;
    auto const full = cuddl::score_compatibility::current<k_default, b_default>();
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
    auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);

    for (auto const& compatibility : modes) {
        SCOPED_TRACE(compatibility.indexed_bucket_count);
        SCOPED_TRACE(compatibility.key_mask);
        auto built = database_type::build_indexed_async(device_rows, compatibility, stream);
        ASSERT_TRUE(built.has_value()) << built.error().message();
        auto database = std::move(*built);
        EXPECT_EQ(database.metadata().compatibility, compatibility);

        auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
        ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
        auto workspace =
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
        auto exhaustive_workspace =
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);
        auto device_results = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), reference_count, cuddl::reference_search_result{}
        );
        auto device_exhaustive = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), reference_count, cuddl::reference_search_result{}
        );
        auto device_result_count =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
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
            ASSERT_NO_THROW(stream_.sync());

            uint32_t result_count = 0;
            ASSERT_NO_THROW(([&] {
                cuda::copy_bytes(
                    stream,
                    cuda::std::span{device_result_count.data(), size_t{1}},
                    cuda::std::span{&result_count, size_t{1}}
                );
                stream.sync();
            }()));
            std::vector<cuddl::reference_search_result> results;
            std::vector<cuddl::reference_search_result> exhaustive_results;
            ASSERT_TRUE(copy_device_buffer(device_results, results));
            ASSERT_TRUE(copy_device_buffer(device_exhaustive, exhaustive_results));
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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 4U;
    auto const full = cuddl::score_compatibility::current<k_default, b_default>();
    auto masked = full;
    masked.key_mask = 0x7fffU;

    std::vector<uint16_t> rows(reference_count * b_default, 0U);
    rows[0] = 0x8000U;
    rows[2U * b_default] = 0x0001U;
    rows[3U * b_default] = 0x8001U;
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
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
        auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query);
        auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
        if (!workspace_bytes) {
            return ::testing::AssertionFailure() << workspace_bytes.error().message();
        }
        auto workspace =
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
        auto device_results = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), reference_count, cuddl::reference_search_result{}
        );
        auto device_result_count =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
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
        if (auto const status = cuddl::cuda_try([&] { stream_.sync(); }); !status) {
            return ::testing::AssertionFailure() << status.error().message();
        }

        uint32_t result_count = 0;
        if (auto const status = cuddl::cuda_try([&] {
                cuda::copy_bytes(
                    stream,
                    cuda::std::span{device_result_count.data(), size_t{1}},
                    cuda::std::span{&result_count, size_t{1}}
                );
                stream.sync();
            });
            !status) {
            return ::testing::AssertionFailure() << status.error().message();
        }
        if (!copy_device_buffer(device_results, results)) {
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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    constexpr size_t reference_count = 2;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto rows = cuda::make_device_buffer<uint16_t>(
        stream, stream.device(), reference_count * b_default, 17U
    );
    auto query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), b_default, 17U);

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
    auto workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
    ASSERT_TRUE(workspace_bytes.has_value()) << workspace_bytes.error().message();
    ASSERT_GT(*workspace_bytes, 0U);
    auto workspace =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), *workspace_bytes, uint8_t{});
    auto short_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), *workspace_bytes - 1U, uint8_t{}
    );
    cuddl::reference_search_result const sentinel{
        .reference_id = 123U,
        .summary = {.counts = {.lower = 4U, .equal = 5U, .higher = 6U}},
    };
    auto results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count, sentinel
    );
    auto short_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), reference_count - 1U, cuddl::reference_search_result{}
    );
    auto result_count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, 77U);
    auto no_result_count =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 0, cuda::no_init);

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
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{result_count.data(), size_t{1}},
            cuda::std::span{&observed_count, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(([&] {
        cuda::copy_bytes(
            cuda::stream_ref{stream_},
            cuda::std::span{results.data(), size_t{1}},
            cuda::std::span{&observed_result, size_t{1}}
        );
    }()));
    ASSERT_NO_THROW(stream_.sync());
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
    auto const stream = cuda::stream_ref{stream_};
    using database_type = cuddl::reference_database<k_default, b_default>;
    auto const compatibility = cuddl::score_compatibility::current<k_default, b_default>();
    auto const maximum = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    auto const posting_scores = (maximum / b_default + 1U) * b_default;
    auto const reference_scores = (maximum + 1U) * b_default;
    auto const* fake_device_rows = reinterpret_cast<uint16_t const*>(uintptr_t{1});

    auto posting_bound = database_type::build_indexed_async(
        cuddl::device_span<uint16_t const>{fake_device_rows, posting_scores}, compatibility, stream
    );
    ASSERT_FALSE(posting_bound.has_value());
    EXPECT_EQ(posting_bound.error().category(), cuddl::ErrorCategory::resource);

    auto reference_bound = database_type::build_indexed_async(
        cuddl::device_span<uint16_t const>{fake_device_rows, reference_scores},
        compatibility,
        stream
    );
    ASSERT_FALSE(reference_bound.has_value());
    EXPECT_EQ(reference_bound.error().category(), cuddl::ErrorCategory::resource);
}

TEST(SketchTest, CardinalityMatchesScalarOracleAcrossMagnitudes) {
    cuda::stream stream{cuda::devices[0]};
    for (size_t n : {0U, 1U, 2U, 8U, 64U, 512U, 4096U, 1u << 16, 1u << 22}) {
        auto const inputs = make_inputs(n, 0x3333'3333'3333'3333ULL + n);
        auto device_inputs = cuda::make_device_buffer<uint64_t>(
            stream, stream.device(), cuda::std::span{inputs.begin(), inputs.end()}
        );
        cuddl::sketch<k_default, b_default> gpu(stream);
        if (n > 0) {
            ASSERT_TRUE(gpu.add(device_inputs, stream).has_value());
        }
        auto const gpu_cardinality = gpu.cardinality(stream);
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
    cuda::stream stream{cuda::devices[0]};
    for (size_t n : {1u << 16, 1u << 18, 1u << 20}) {
        auto const inputs = make_inputs(n, 0x7777'7777'7777'7777ULL + n);
        // make_inputs from a 64-bit SplitMix stream masked into 2^50 space; collisions are
        // negligible at these sizes, so the distinct count is ~n.
        auto device_inputs = cuda::make_device_buffer<uint64_t>(
            stream, stream.device(), cuda::std::span{inputs.begin(), inputs.end()}
        );
        cuddl::sketch<k_default, b_default> gpu(stream);
        ASSERT_TRUE(gpu.add(device_inputs, stream).has_value());
        auto const card = gpu.cardinality(stream);
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
    cuda::stream stream{cuda::devices[0]};
    for (size_t n : {512U, 1024U, 2048U, 4096U, 8192U}) {
        auto const inputs = make_inputs(n, 0x5555'5555'5555'5555ULL + n);
        auto device_inputs = cuda::make_device_buffer<uint64_t>(
            stream, stream.device(), cuda::std::span{inputs.begin(), inputs.end()}
        );
        cuddl::sketch<k_default, b_default> gpu(stream);
        ASSERT_TRUE(gpu.add(device_inputs, stream).has_value());
        auto const estimate = gpu.cardinality(stream);
        ASSERT_TRUE(estimate.has_value());
        EXPECT_NEAR(*estimate, static_cast<double>(n), 0.1 * static_cast<double>(n));
    }
}

TEST(SketchTest, WinnerCountsAndSaturationMatchScalarOracle) {
    cuda::stream stream{cuda::devices[0]};
    // A repeated k-mer so a single bucket sees many equal observations.
    auto const packed_kmer = 0x1234'5678'9abc'def0ULL & ((1ULL << (2 * k_default)) - 1);
    std::vector<uint64_t> inputs;
    auto const repeat = 65536U;
    inputs.assign(repeat, packed_kmer);

    cuddl::sketch<k_default, b_default> gpu(stream);
    ASSERT_TRUE(gpu.add({inputs.data(), inputs.size()}, stream).has_value());
    auto const gpu_wc = gpu.winner_counts(stream);
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
    cuda::stream stream{cuda::devices[0]};
    auto const packed_kmer = 0x0000'0000'0000'00ffULL & ((1ULL << (2 * k_default)) - 1);

    cuddl::sketch<k_default, b_default> under(stream);
    std::vector<uint64_t> few(1000, packed_kmer);
    ASSERT_TRUE(under.add({few.data(), few.size()}, stream).has_value());
    auto const under_wc = under.winner_counts(stream);
    ASSERT_TRUE(under_wc.has_value());
    EXPECT_FALSE(under_wc->second);

    cuddl::sketch<k_default, b_default> over(stream);
    std::vector<uint64_t> many(65536, packed_kmer);
    ASSERT_TRUE(over.add({many.data(), many.size()}, stream).has_value());
    auto const over_wc = over.winner_counts(stream);
    ASSERT_TRUE(over_wc.has_value());
    EXPECT_TRUE(over_wc->second);
}

TEST(SketchTest, HostMetricsOnRawPair) {
    cuda::stream stream{cuda::devices[0]};
    auto const a = make_inputs(40000, 0x4444'4444'4444'4444ULL);
    auto const c = make_inputs(40000, 0x5555'5555'5555'5555ULL);

    cuddl::sketch<k_default, b_default> sa(stream);
    cuddl::sketch<k_default, b_default> sb(stream);
    cuddl::sketch<k_default, b_default> sc(stream);
    ASSERT_TRUE(sa.add({a.data(), a.size()}, stream).has_value());
    ASSERT_TRUE(sb.add({a.data(), a.size()}, stream).has_value());
    ASSERT_TRUE(sc.add({c.data(), c.size()}, stream).has_value());

    auto const same_result = sa.compare(sb, stream);
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

    auto const disjoint_result = sa.compare(sc, stream);
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
    auto const stream = cuda::stream_ref{stream_};
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
    auto device_scores = cuda::make_device_buffer<uint16_t>(stream, stream.device(), scores);
    auto built = database_type::build_indexed_async(device_scores, compatibility, stream);
    ASSERT_TRUE(built.has_value()) << built.error().message();
    auto database = std::move(*built);

    // Exhaustive reference database over the same subset, for the GPU exact-summary contract.
    auto exhaustive_built = database_type::build_async(device_scores, compatibility, stream);
    ASSERT_TRUE(exhaustive_built.has_value());
    auto exhaustive_database = std::move(*exhaustive_built);

    auto indexed_workspace_bytes = database.indexed_single_query_workspace_bytes(stream);
    ASSERT_TRUE(indexed_workspace_bytes.has_value()) << indexed_workspace_bytes.error().message();
    auto indexed_workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), *indexed_workspace_bytes, uint8_t{}
    );

    // Resident queries and the external row must agree with the same host oracle.
    std::vector<uint32_t> const query_ids{0U, 1U, 2U, external_id};
    for (auto const query_id : query_ids) {
        SCOPED_TRACE(query_id);
        auto const& query_row = rows[query_id];
        auto device_query = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query_row);
        auto indexed = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), db_ids.size(), cuddl::reference_search_result{}
        );
        auto exhaustive = cuda::make_device_buffer<cuddl::reference_search_result>(
            stream, stream.device(), db_ids.size(), cuddl::reference_search_result{}
        );
        auto indexed_count =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, uint32_t{});
        auto exhaustive_workspace =
            cuda::make_device_buffer<uint8_t>(stream, stream.device(), 0, cuda::no_init);

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
        ASSERT_NO_THROW(stream_.sync());

        std::vector<cuddl::reference_search_result> indexed_host(db_ids.size());
        std::vector<cuddl::reference_search_result> exhaustive_host(db_ids.size());
        ASSERT_TRUE(copy_device_buffer(indexed, indexed_host));
        ASSERT_TRUE(copy_device_buffer(exhaustive, exhaustive_host));
        uint32_t count = 0;
        ASSERT_NO_THROW(([&] {
            cuda::copy_bytes(
                stream,
                cuda::std::span{indexed_count.data(), size_t{1}},
                cuda::std::span{&count, size_t{1}}
            );
            stream.sync();
        }()));

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
