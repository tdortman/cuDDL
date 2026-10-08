#include "resident_sequence_batches.hpp"

#include <gtest/gtest.h>

#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct genome_files {
    std::filesystem::path dir;
    std::vector<std::string> paths;

    explicit genome_files(std::vector<size_t> const& sizes) {
        dir = std::filesystem::temp_directory_path() /
              ("resident-sequence-test-" + std::to_string(::getpid()));
        std::filesystem::create_directories(dir);
        for (size_t g = 0; g < sizes.size(); ++g) {
            auto const path = dir / ("g" + std::to_string(g) + ".fa");
            std::ofstream out(path);
            out << ">g" << g << "\n";
            for (size_t i = 0; i < sizes[g]; ++i) out << "ACGT"[(i * 7 + g) % 4];
            out << "\n";
            paths.push_back(path.string());
        }
    }

    ~genome_files() {
        std::filesystem::remove_all(dir);
    }
};

}  // namespace

TEST(ResidentSequenceTest, WholeGenomesNeverStraddleBatches) {
    uint32_t const k = 5;
    size_t const cap = 100;
    // 60 + 60 overflows one batch; 250 exceeds the cap on its own.
    genome_files files({60, 60, 250, 30});
    std::map<size_t, std::set<size_t>> batches_of;
    std::map<size_t, size_t> bases_of;
    size_t batch_index = 0;
    resident_sequence::for_each_batch(
        files.paths,
        k,
        cap,
        [&](resident_sequence::batch const& batch) {
            for (auto const& chunk : batch.chunks) {
                batches_of[chunk.genome].insert(batch_index);
                bases_of[chunk.genome] += chunk.size;
            }
            ++batch_index;
        },
        1,
        true
    );
    ASSERT_EQ(batches_of.size(), 4U);
    for (auto const& [genome, batches] : batches_of) {
        EXPECT_EQ(batches.size(), 1U) << "genome " << genome << " straddles batches";
    }
    EXPECT_NE(*batches_of[0].begin(), *batches_of[1].begin());
    EXPECT_NE(*batches_of[2].begin(), *batches_of[3].begin());
    EXPECT_GE(bases_of[2], 250U);
}
