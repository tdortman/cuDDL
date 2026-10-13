// Distributed sketching and search through include/cuddl/distributed.cuh. Run under mpirun
// with any number of ranks; every rank may share one GPU.

#include <gtest/gtest.h>

#include <cuddl/distributed.cuh>
#include <cuddl/replicated_database.cuh>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t k = 25;
constexpr size_t buckets = 2048;

int world_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int world_size() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

/// A fresh directory shared by every rank, with one random genome per entry of @p lengths.
std::filesystem::path
genome_directory(std::string const& name, std::vector<size_t> const& lengths) {
    auto const directory = std::filesystem::temp_directory_path() / ("cuddl_distributed_" + name);
    if (world_rank() == 0) {
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        std::mt19937 random(7);
        for (size_t g = 0; g < lengths.size(); ++g) {
            std::ofstream out(directory / ("genome" + std::to_string(10 + g) + ".fna"));
            out << ">g" << g << '\n';
            for (size_t i = 0; i < lengths[g]; ++i) out << "ACGT"[random() % 4];
            out << '\n';
        }
        // Not a genome: listing must skip it.
        std::ofstream(directory / "notes.txt") << "skip\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return directory;
}

std::vector<size_t> all_firsts(cuddl::genome_slice const& slice) {
    std::vector<size_t> firsts(world_size());
    uint64_t const mine = slice.first;
    MPI_Allgather(&mine, 1, MPI_UINT64_T, firsts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    return firsts;
}

void expect_distributed_build_matches_single_process(std::filesystem::path const& directory) {
    cuda::stream stream{cuda::devices[0]};
    auto const slice = CUDDL_UNWRAP(cuddl::slice_genomes(directory, MPI_COMM_WORLD, 3));

    // Slices are contiguous, cover every genome once, and follow rank order.
    auto const firsts = all_firsts(slice);
    EXPECT_EQ(firsts.front(), 0U);
    EXPECT_EQ(slice.first, firsts[world_rank()]);
    EXPECT_EQ(
        slice.last, world_rank() + 1 == world_size() ? slice.paths.size() : firsts[world_rank() + 1]
    );

    auto const mine =
        CUDDL_UNWRAP((cuddl::reference_database_file::build<k, buckets>(slice.mine(), stream)));
    auto const gathered = CUDDL_UNWRAP(cuddl::gather_sketches(mine, slice, MPI_COMM_WORLD));

    auto const listed = CUDDL_UNWRAP(cuddl::genome_paths(directory));
    auto const single =
        CUDDL_UNWRAP((cuddl::reference_database_file::build<k, buckets>(listed, stream)));
    EXPECT_EQ(gathered.metadata(), single.metadata());
    EXPECT_TRUE(std::ranges::equal(gathered.names(), single.names()));
    EXPECT_TRUE(std::ranges::equal(gathered.rows(), single.rows()));
}

TEST(Distributed, GatheredSketchesEqualSingleProcessBuild) {
    auto const directory =
        genome_directory("equal", {4000, 90000, 1200, 30000, 60000, 2500, 45000, 800});
    expect_distributed_build_matches_single_process(directory);
}

TEST(Distributed, RanksWithoutGenomesStillGather) {
    auto const directory = genome_directory("one", {5000});
    expect_distributed_build_matches_single_process(directory);
}

TEST(Distributed, SlicesHoldAboutEqualBytes) {
    std::vector<size_t> lengths;
    for (size_t g = 0; g < 40; ++g) lengths.push_back(1000 + 997 * (g % 9));
    auto const directory = genome_directory("balance", lengths);
    auto const slice = CUDDL_UNWRAP(cuddl::slice_genomes(directory, MPI_COMM_WORLD, 2));
    ASSERT_EQ(slice.paths.size(), lengths.size());
    uint64_t total = 0, largest = 0, mine = 0;
    for (size_t g = 0; g < slice.bytes.size(); ++g) {
        EXPECT_EQ(slice.bytes[g], std::filesystem::file_size(slice.paths[g]));
        total += slice.bytes[g];
        largest = std::max(largest, slice.bytes[g]);
        if (g >= slice.first && g < slice.last) mine += slice.bytes[g];
    }
    auto const fair = static_cast<double>(total) / world_size();
    EXPECT_LE(std::abs(static_cast<double>(mine) - fair), static_cast<double>(largest));
}

TEST(Distributed, MissingGenomeFailsOnEveryRank) {
    auto const directory = genome_directory("missing", {3000, 3000, 3000, 3000});
    auto const list = directory / "list.txt";
    if (world_rank() == 0) {
        std::ofstream out(list);
        for (auto const& path : CUDDL_UNWRAP(cuddl::genome_paths(directory)))
            out << path.string() << '\n';
        out << (directory / "absent.fna").string() << '\n';
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto const slice = cuddl::slice_genomes(list, MPI_COMM_WORLD, 2);
    ASSERT_FALSE(slice);
    // The last rank looks up the last path, the missing one, and reports it; others name a rank.
    if (world_rank() + 1 == world_size()) {
        EXPECT_NE(slice.error().message().find("absent.fna"), std::string::npos)
            << slice.error().message();
    }
}

TEST(Distributed, DifferentSketchSettingsFailOnEveryRank) {
    if (world_size() < 2) GTEST_SKIP() << "needs two ranks";
    auto const directory = genome_directory("settings", {3000, 4000, 5000, 6000});
    cuda::stream stream{cuda::devices[0]};
    auto const slice = CUDDL_UNWRAP(cuddl::slice_genomes(directory, MPI_COMM_WORLD, 2));
    auto const mine =
        world_rank() == 0
            ? CUDDL_UNWRAP(
                  (cuddl::reference_database_file::build<k, buckets>(slice.mine(), stream))
              )
            : CUDDL_UNWRAP(
                  (cuddl::reference_database_file::build<21, buckets>(slice.mine(), stream))
              );
    auto const gathered = cuddl::gather_sketches(mine, slice, MPI_COMM_WORLD);
    EXPECT_FALSE(gathered);
}

using group_type = cuddl::replicated_database<k, buckets>;
constexpr uint32_t references = 6000U;
constexpr uint32_t group_size = 4U;
constexpr uint32_t minimum = 100U;

/// 6,000 rows in groups of four whose index matches reach @ref minimum only within a group, so
/// a threshold search reports exactly the 9,000 in-group pairs. Enough rows that the search
/// spans several tiles per rank.
group_type grouped_rows(cuda::stream_ref stream) {
    constexpr uint32_t shared = 48U;
    auto compatibility = cuddl::score_compatibility::current<k, buckets>();
    compatibility.indexed_bucket_count = buckets / 2U;
    compatibility.key_mask = 0x7fffU;
    std::vector<uint16_t> rows(static_cast<size_t>(references) * buckets);
    for (uint32_t reference = 0U; reference < references; ++reference) {
        for (uint32_t bucket = 0U; bucket < buckets; ++bucket) {
            if (bucket >= shared && bucket % 50U == reference % group_size) continue;
            auto const score =
                bucket < shared ? 7U : 1U + reference / group_size + 1600U * (bucket % 16U);
            rows[static_cast<size_t>(reference) * buckets + bucket] = static_cast<uint16_t>(score);
        }
    }
    auto device_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    std::vector<group_type::database_type> databases;
    databases.push_back(
        CUDDL_UNWRAP(group_type::database_type::build_async(device_rows, compatibility, stream))
    );
    stream.sync();
    auto group = CUDDL_UNWRAP(group_type::adopt(std::move(databases)));
    CUDDL_UNWRAP(group.build_index(cuddl::index_storage::dense));
    return group;
}

uint64_t pair_hash(uint32_t query, uint32_t reference) {
    auto key = (uint64_t{query} << 32U | reference) * 0x9e3779b97f4a7c15ULL;
    return key ^ (key >> 29U);
}

struct claimed {
    uint64_t pairs = 0;
    uint64_t hash = 0;
    uint64_t tiles = 0;
};

/// Occupies the stream it runs on for @p nanoseconds.
__global__ void spin_kernel(uint64_t nanoseconds) {
    uint64_t start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < nanoseconds);
}

/// Searches collectively across the world, exhaustively when @p minimum_matches is 0. With
/// @p download, counts this rank's pairs. With @p gpu_delay, every tile also occupies this
/// rank's GPU that long while its host thread runs on, as a slower GPU would.
claimed search_shared(
    group_type& group,
    uint32_t minimum_matches,
    bool download,
    std::chrono::nanoseconds gpu_delay = {}
) {
    claimed mine;
    std::mutex lock;
    auto const on_tile = [&](cuddl::replica_tile const& tile) {
        std::vector<cuddl::batch_search_result> passing;
        if (download) passing = CUDDL_UNWRAP(tile.download())->passing();
        if (gpu_delay.count() > 0) {
            spin_kernel<<<1, 1, 0, tile.stream.get()>>>(static_cast<uint64_t>(gpu_delay.count()));
        }
        std::lock_guard const guard(lock);
        ++mine.tiles;
        for (auto const& result : passing) {
            ++mine.pairs;
            mine.hash += pair_hash(result.query_id, result.reference_id);
        }
    };
    if (minimum_matches == 0) {
        CUDDL_UNWRAP(cuddl::search_all_to_all(group, on_tile, MPI_COMM_WORLD));
    } else {
        CUDDL_UNWRAP(
            cuddl::search_all_to_all(
                group, on_tile, {.minimum_matches = minimum_matches}, MPI_COMM_WORLD
            )
        );
    }
    return mine;
}

TEST(Distributed, SharedTilesCoverEveryPairOnce) {
    cuda::stream stream{cuda::devices[0]};
    auto group = grouped_rows(stream);
    for (uint32_t const threshold : {0U, minimum}) {
        auto const mine = search_shared(group, threshold, true);
        std::array<uint64_t, 2> const local{mine.pairs, mine.hash};
        std::array<uint64_t, 2> total{};
        MPI_Allreduce(local.data(), total.data(), 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        uint64_t expected_pairs = 0, expected_hash = 0;
        for (uint32_t query = 0U; query < references; ++query) {
            auto const end = threshold == 0U ? references : (query / group_size + 1U) * group_size;
            for (uint32_t reference = query + 1U; reference < end; ++reference) {
                ++expected_pairs;
                expected_hash += pair_hash(query, reference);
            }
        }
        EXPECT_EQ(total[0], expected_pairs) << "threshold " << threshold;
        EXPECT_EQ(total[1], expected_hash) << "threshold " << threshold;
    }
}

// The slow rank's host thread never waits for its GPU, so only claiming that follows the GPU's
// progress keeps it from taking tiles it cannot search soon.
TEST(Distributed, SlowerGpuClaimsFewerTiles) {
    if (world_size() < 2) GTEST_SKIP() << "needs two ranks";
    cuda::stream stream{cuda::devices[0]};
    auto group = grouped_rows(stream);
    for (uint32_t const threshold : {0U, minimum}) {
        auto const delay = std::chrono::milliseconds{world_rank() == 0 ? 200 : 0};
        auto const mine = search_shared(group, threshold, false, delay);
        std::vector<uint64_t> tiles(world_size());
        MPI_Allgather(&mine.tiles, 1, MPI_UINT64_T, tiles.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
        EXPECT_LT(tiles[0], *std::max_element(tiles.begin() + 1, tiles.end()))
            << "threshold " << threshold;
    }
}

TEST(Distributed, SharedTilesRejectQueryOrder) {
    cuda::stream stream{cuda::devices[0]};
    auto group = grouped_rows(stream);
    auto const result = group.search_all_to_all(
        [](cuddl::replica_tile const&) {},
        {.in_query_order = true, .next_tile = [] { return uint64_t{0}; }}
    );
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().category(), cuddl::ErrorCategory::invalid_argument);
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    testing::InitGoogleTest(&argc, argv);
    int const failed = RUN_ALL_TESTS();
    int any = 0;
    MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any;
}
