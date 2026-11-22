// Checks that GenomeSketch::add_run (vector lanes) fills the same registers as the scalar
// reference add_windows on synthetic packed runs: lengths at k - 1 + 1, around the scalar prefix
// and the lane chunk boundaries, and random ones, over random, low-complexity and periodic bases,
// with a fresh sketch per run and with one sketch accumulated over all runs. Exits non-zero on a
// mismatch. Build from the repository root with
//   g++ -O3 -march=native -std=c++20 -DCUDDL_ABLATIONS=0 -Iinclude -Isubprojects/libdeflate
//       -Isubprojects/cccl/libcudacxx/include -I"$(dirname "$(dirname "$(which nvcc)")")/include"
//       benchmarks/cuddl_cpu_sketch_check.cpp -o cuddl-cpu-sketch-check
// (-mcpu=native on aarch64). The headers are only parsed; nothing is linked.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "cuddl_cpu_ingest.hpp"

namespace {

using cpu_ingest::GenomeSketch;
using cpu_ingest::kBuckets;
using cpu_ingest::kKmer;
using cpu_ingest::kLanes;

std::vector<uint64_t> run_lengths(std::mt19937_64& rng) {
    std::vector<uint64_t> windows = {1, 2, 8, 9, 15, 16, 17, 135, 136, 137, 1000, 5000};
    uint64_t const step = 16U * (kLanes == 0U ? 1U : kLanes);
    for (uint64_t const prefix : {uint64_t{8}, GenomeSketch::kWarmup + 8U}) {
        for (uint64_t const chunks : {0U, 1U, 2U, 17U}) {
            for (uint64_t const extra : {uint64_t{0}, uint64_t{1}, uint64_t{15}, step - 1U, step, step + 1U}) {
                windows.push_back(prefix + chunks * step + extra);
            }
        }
    }
    for (uint32_t i = 0; i < 24; ++i) {
        windows.push_back(1U + rng() % 400000U);
    }
    windows.push_back(2000003);
    std::vector<uint64_t> bases;
    for (auto const w : windows) {
        bases.push_back(w + kKmer - 1U);
    }
    return bases;
}

}  // namespace

int main() {
    std::mt19937_64 rng(7);
    auto const lengths = run_lengths(rng);
    uint64_t mismatches = 0;
    for (uint32_t mode = 0; mode < 3; ++mode) {
        std::vector<std::vector<uint32_t>> runs;
        for (auto const bases : lengths) {
            std::vector<uint32_t> words((bases + 15U) / 16U + 2U);
            for (size_t i = 0; i < words.size(); ++i) {
                auto const random = static_cast<uint32_t>(rng());
                // 0 random bases, 1 only A and C, 2 a short period with a rare change.
                words[i] = mode == 0 ? random : mode == 1 ? random & 0x55555555U : 0x1B1B1B1BU ^ (i % 977U == 0U ? 0xFFFFU : 0U);
            }
            runs.push_back(std::move(words));
        }
        std::vector<uint16_t> vector_registers(kBuckets), scalar_registers(kBuckets);
        GenomeSketch vector_sketch{vector_registers.data()}, scalar_sketch{scalar_registers.data()};
        for (size_t r = 0; r < runs.size(); ++r) {
            vector_sketch.add_run(runs[r].data(), lengths[r]);
            scalar_sketch.add_windows(runs[r].data(), 0, lengths[r] - (kKmer - 1U));
            std::vector<uint16_t> fresh_vector(kBuckets), fresh_scalar(kBuckets);
            GenomeSketch fresh_vector_sketch{fresh_vector.data()}, fresh_scalar_sketch{fresh_scalar.data()};
            fresh_vector_sketch.add_run(runs[r].data(), lengths[r]);
            fresh_scalar_sketch.add_windows(runs[r].data(), 0, lengths[r] - (kKmer - 1U));
            for (uint32_t b = 0; b < kBuckets; ++b) {
                mismatches += fresh_vector[b] != fresh_scalar[b];
            }
        }
        for (uint32_t b = 0; b < kBuckets; ++b) {
            mismatches += vector_registers[b] != scalar_registers[b];
        }
    }
    std::printf("lanes %u runs %zu mismatching buckets %llu\n", kLanes, lengths.size(), static_cast<unsigned long long>(mismatches));
    return mismatches != 0U;
}
