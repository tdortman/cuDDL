#include <gtest/gtest.h>
#include <cuddl/query_sketch.cuh>
#include <cuddl/reference_index_file.cuh>

#include <libdeflate.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using index_file = cuddl::reference_index_file;
using database_file = cuddl::reference_database_file;
using database = cuddl::reference_database<25, 2048>;
using acceleration = cuddl::reference_index<25, 2048>;
static_assert(!std::is_base_of_v<database, acceleration>);
using query_batch = cuddl::query_sketch_batch<25, 2048>;
static_assert(!std::is_default_constructible_v<database>);

template <uint32_t K, size_t Buckets>
cuddl::Result<void> write_index(
    database_file const& database,
    std::filesystem::path const& path,
    cuda::stream_ref stream,
    cuddl::index_storage storage
) {
    auto rows = CUDDL_TRY((database.upload<K, Buckets>(stream)));
    auto index =
        CUDDL_TRY((cuddl::reference_index<K, Buckets>::build_async(rows, stream, storage)));
    return index_file::save(index, rows, path, stream);
}

TEST(ReferenceIndexFileTest, ByteRangeHashMatchesXxHash64AtChunkBoundaries) {
    std::array<uint8_t, 64> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i);
    }
    std::array<std::pair<size_t, uint64_t>, 7> cases{
        {{0, 0x98b1582b0977e704ULL},
         {7, 0x3ec2863aeaf013fdULL},
         {8, 0x9cde1e0fbac053a6ULL},
         {31, 0x8340e23e22f83759ULL},
         {32, 0x8809e1ca0be25072ULL},
         {33, 0x1c682d8884811fdbULL},
         {64, 0xce47892c1e53be8eULL}}
    };
    for (auto [size, expected] : cases) {
        EXPECT_EQ(cuddl::detail::xxhash64(bytes.data(), size, 42), expected) << size;
    }
    EXPECT_EQ(cuddl::detail::xxhash64(bytes, 42), cases.back().second);
}

struct temporary_directory {
    std::filesystem::path path;
    temporary_directory() {
        auto pattern =
            (std::filesystem::temp_directory_path() / "cuddl-index-test-XXXXXX").string();
        auto* result = ::mkdtemp(pattern.data());
        if (result == nullptr) throw std::runtime_error("cannot create test directory");
        path = result;
    }
    ~temporary_directory() {
        std::filesystem::remove_all(path);
    }
};

std::vector<cuddl::reference_search_result> search(
    database const& db,
    query_batch const& query,
    cuda::stream_ref stream,
    acceleration const* index = nullptr
) {
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream,
        stream.device(),
        CUDDL_UNWRAP(db.search_workspace_bytes(stream, index)),
        cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), db.single_query_result_count(), cuda::no_init
    );
    auto count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, cuda::no_init);
    CUDDL_UNWRAP(db.search_async(
        query.scores(),
        query_batch::compatibility(),
        {workspace.data(), workspace.size()},
        {results.data(), results.size()},
        {count.data(), count.size()},
        {.minimum_matches = 1},
        stream,
        index
    ));
    uint32_t found = 0;

    cuda::copy_bytes(stream, count, cuda::std::span{&found, size_t{1}});
    stream.sync();
    std::vector<cuddl::reference_search_result> hits(found);
    if (found == 0) return hits;
    cuda::copy_bytes(
        stream,
        cuddl::device_span<cuddl::reference_search_result const>{results.data(), found},
        cuda::std::span{hits.data(), hits.size()}
    );
    stream.sync();
    std::sort(hits.begin(), hits.end(), [](auto const& a, auto const& b) {
        return a.reference_id < b.reference_id;
    });
    return hits;
}

std::vector<cuddl::batch_search_result> search_batch(
    database const& db,
    query_batch const& query,
    cuda::stream_ref stream,
    acceleration const* index,
    bool all_to_all
) {
    auto requirements = CUDDL_UNWRAP(
        all_to_all ? db.all_to_all_search_requirements(index)
                   : db.batch_search_requirements(1U, stream, index)
    );
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), requirements.maximum_pair_count, cuda::no_init
    );
    std::vector<cuddl::batch_search_result> output;
    auto consume = [&](cuddl::batch_result_tile const& tile) {
        auto const tile_copy = CUDDL_UNWRAP(cuddl::download(tile, stream));
        auto const passing = tile_copy.passing();
        output.insert(output.end(), passing.begin(), passing.end());
    };
    if (all_to_all) {
        CUDDL_UNWRAP(db.search_all_to_all_async(
            workspace, results, consume, {}, {.minimum_matches = 1U}, stream, index
        ));
    } else {
        CUDDL_UNWRAP(db.search_batch_async(
            query.scores(),
            query_batch::compatibility(),
            0U,
            workspace,
            results,
            consume,
            {},
            {.minimum_matches = 1U},
            stream,
            index
        ));
    }
    return output;
}

TEST(ReferenceIndexFileTest, DenseAndSparseRoundTripsSupportRepeatedSearchesWithoutFiles) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    std::array<cuddl::sequence_record, 1> alpha{
        {{"ACGTTGCACTGATCGAGGCTAACGTTGCACTGATCGAGGCTAACGT"}}
    };
    std::array<cuddl::sequence_record, 1> beta{{{"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"}}};
    std::array<cuddl::sequence_genome, 3> genomes{
        {{alpha, "alpha"}, {beta, "beta"}, {alpha, "copy"}}
    };
    auto archive = CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>(genomes, stream)));
    auto query = CUDDL_UNWRAP(
        query_batch::sketch_from_sequences(
            std::span<cuddl::sequence_genome const>{genomes.data(), 1}, stream
        )
    );
    for (auto storage : {cuddl::index_storage::dense, cuddl::index_storage::sparse}) {
        auto db_path = temporary.path / "references.cuddl";
        auto index_path = temporary.path / "index.data";
        ASSERT_TRUE(archive.save(db_path));
        auto written = write_index<25, 2048>(archive, index_path, stream, storage);
        ASSERT_TRUE(written) << written.error().message();
        auto relocated = temporary.path / "relocated.cuddl";
        std::filesystem::rename(db_path, relocated);
        auto disk_archive = CUDDL_UNWRAP(database_file::load(relocated));
        auto disk_database = CUDDL_UNWRAP((disk_archive.upload<25, 2048>(stream)));
        auto loaded = (index_file::load<25, 2048>(index_path, disk_database, stream));
        ASSERT_TRUE(loaded) << loaded.error().message();
        EXPECT_EQ(loaded->storage(), storage);
        auto expected_database = CUDDL_UNWRAP((archive.upload<25, 2048>(stream)));
        auto built = CUDDL_UNWRAP(acceleration::build_async(expected_database, stream, storage));
        auto expected = search(expected_database, query, stream);
        EXPECT_EQ(search(expected_database, query, stream, &built), expected);
        EXPECT_FALSE(disk_database.search_workspace_bytes(stream, &built));
        ASSERT_EQ(expected.size(), 2U);
        EXPECT_EQ(expected[0].reference_id, 0U);
        EXPECT_EQ(expected[1].reference_id, 2U);
        ASSERT_TRUE(std::filesystem::remove(index_path));
        ASSERT_TRUE(std::filesystem::remove(relocated));
        EXPECT_EQ(search(disk_database, query, stream, &*loaded), expected);
        EXPECT_EQ(search(disk_database, query, stream, &*loaded), expected);
        for (bool all_to_all : {false, true}) {
            auto baseline = search_batch(expected_database, query, stream, nullptr, all_to_all);
            ASSERT_EQ(baseline.size(), all_to_all ? 1U : 2U);
            EXPECT_EQ(baseline.back().reference_id, 2U);
            EXPECT_EQ(search_batch(expected_database, query, stream, &built, all_to_all), baseline);
            EXPECT_EQ(search_batch(disk_database, query, stream, &*loaded, all_to_all), baseline);
        }
        auto moved = std::move(*loaded);
        EXPECT_EQ(search(disk_database, query, stream, &moved), expected);
        EXPECT_EQ(disk_database.names()[2], "copy");
        auto moved_database = std::move(disk_database);
        EXPECT_EQ(search(moved_database, query, stream, &moved), expected);
        EXPECT_FALSE(moved_database.search_workspace_bytes(stream, &*loaded));
    }
}

TEST(ReferenceIndexFileTest, LoadRejectsDifferentContentsLabelsAndReferenceOrder) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    std::array<cuddl::sequence_record, 1> alpha{
        {{"ACGTTGCACTGATCGAGGCTAACGTTGCACTGATCGAGGCTAACGT"}}
    };
    std::array<cuddl::sequence_record, 1> beta{{{"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"}}};
    std::array<cuddl::sequence_genome, 2> genomes{{{alpha, "alpha"}, {beta, "beta"}}};
    auto archive = CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>(genomes, stream)));
    auto path = temporary.path / "index";
    ASSERT_TRUE((write_index<25, 2048>(archive, path, stream, cuddl::index_storage::sparse)));
    std::array<cuddl::sequence_genome, 2> reordered{{genomes[1], genomes[0]}};
    std::array<cuddl::sequence_genome, 2> relabelled{{{alpha, "renamed"}, genomes[1]}};
    std::array<cuddl::sequence_genome, 2> replaced{{{beta, "alpha"}, genomes[1]}};
    for (auto const& different : {reordered, relabelled, replaced}) {
        auto other =
            CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>(different, stream)));
        EXPECT_FALSE((
            index_file::load<25, 2048>(path, CUDDL_UNWRAP((other.upload<25, 2048>(stream))), stream)
        ));
    }
    auto loaded = (index_file::load<25, 2048>(
        path, CUDDL_UNWRAP((archive.upload<25, 2048>(stream))), stream
    ));
    ASSERT_TRUE(loaded) << loaded.error().message();
}

TEST(ReferenceIndexFileTest, EmptyIndexesRoundTripAndMalformedFilesFailAtLoad) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    auto empty = CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>({}, stream)));
    auto disk_database = CUDDL_UNWRAP((empty.upload<25, 2048>(stream)));
    std::array<cuddl::sequence_record, 1> bases{
        {{"ACGTTGCACTGATCGAGGCTAACGTTGCACTGATCGAGGCTAACGT"}}
    };
    std::array<cuddl::sequence_genome, 1> genomes{{{bases, "query"}}};
    auto query = CUDDL_UNWRAP(query_batch::sketch_from_sequences(genomes, stream));
    for (auto storage : {cuddl::index_storage::dense, cuddl::index_storage::sparse}) {
        auto path = temporary.path / "empty.index";
        ASSERT_TRUE((write_index<25, 2048>(empty, path, stream, storage)));
        auto loaded = (index_file::load<25, 2048>(path, disk_database, stream));
        ASSERT_TRUE(loaded) << loaded.error().message();
        EXPECT_TRUE(search(disk_database, query, stream, &*loaded).empty());
        if (storage == cuddl::index_storage::dense) {
            std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
            corrupt.seekp(32 + sizeof(uint32_t));
            corrupt.write("\1\0\0\0", 4);
            corrupt.flush();
            corrupt.seekg(0);
            auto data_bytes = std::filesystem::file_size(path) - sizeof(uint32_t);
            auto remaining = data_bytes;
            std::array<char, 65536> buffer{};
            uint32_t checksum = 0;
            while (remaining != 0) {
                auto chunk = std::min<uint64_t>(remaining, buffer.size());
                corrupt.read(buffer.data(), static_cast<std::streamsize>(chunk));
                ASSERT_TRUE(corrupt);
                checksum = libdeflate_crc32(checksum, buffer.data(), chunk);
                remaining -= chunk;
            }
            corrupt.seekp(static_cast<std::streamoff>(data_bytes));
            for (unsigned shift = 0; shift < 32; shift += 8) {
                corrupt.put(static_cast<char>(checksum >> shift));
            }
            corrupt.close();
            ASSERT_TRUE(corrupt);
            EXPECT_FALSE((index_file::load<25, 2048>(path, disk_database, stream)));
        }
    }
    auto path = temporary.path / "empty.index";
    std::ifstream input(path, std::ios::binary);
    std::string bytes{std::istreambuf_iterator<char>{input}, {}};
    input.close();
    auto reject = [&](std::string const& bad) {
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(bad.data(), bad.size());
        }
        EXPECT_FALSE((index_file::load<25, 2048>(path, disk_database, stream)));
    };
    reject("");
    reject(bytes.substr(0, 12));
    reject(bytes.substr(0, bytes.size() - 1));
    reject(bytes + "trailing data");
    bytes.back() ^= 1;
    reject(bytes);
}

TEST(ReferenceIndexFileTest, ValidChecksumDoesNotPermitMalformedSparsePostings) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    std::array<cuddl::sequence_record, 1> ambiguous{{{"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"}}};
    std::array<cuddl::sequence_genome, 2> genomes{{{ambiguous, "empty-a"}, {ambiguous, "empty-b"}}};
    auto archive = CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>(genomes, stream)));
    auto path = temporary.path / "sparse.index";
    ASSERT_TRUE((write_index<25, 2048>(archive, path, stream, cuddl::index_storage::sparse)));
    std::ifstream input(path, std::ios::binary);
    std::string original{std::istreambuf_iterator<char>{input}, {}};
    input.close();
    auto reject = [&](size_t offset, uint32_t value) {
        auto bytes = original;
        auto put = [&](size_t at, uint32_t word) {
            for (size_t i = 0; i < 4; ++i) {
                bytes[at + i] = static_cast<char>(word >> (8 * i));
            }
        };
        put(offset, value);
        put(bytes.size() - 4, libdeflate_crc32(0, bytes.data(), bytes.size() - 4));
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(bytes.data(), bytes.size());
        }
        EXPECT_FALSE((index_file::load<25, 2048>(
            path, CUDDL_UNWRAP((archive.upload<25, 2048>(stream))), stream
        )));
    };
    reject(12, 2);            // Unknown storage kind.
    reject(24, 0xffffffffU);  // Posting extent exceeds the bound database.
    reject(32, 2);            // Reference ID is outside the database.
    uint32_t first_id = static_cast<unsigned char>(original[32]);
    reject(36, first_id);                      // Duplicate valid ID, with matching zero keys.
    reject(original.size() - 8, 0x00010000U);  // Sorted nonzero key for an empty reference.
}

TEST(ReferenceIndexFileTest, PostingsAscendByReferenceWithinEveryKey) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    // Identical genomes share every key, so each posting list spans many warps of the build.
    std::array<cuddl::sequence_record, 1> bases{
        {{"ACGTTGCACTGATCGAGGCTAACGTTGCACTGATCGAGGCTAACGT"}}
    };
    std::vector<cuddl::sequence_genome> genomes;
    std::vector<std::string> names;
    for (size_t i = 0; i < 4096; ++i) names.push_back(std::to_string(i));
    for (auto const& name : names) genomes.push_back({bases, name});
    auto archive = CUDDL_UNWRAP((database_file::build_from_sequences<25, 2048>(genomes, stream)));
    auto database = CUDDL_UNWRAP((archive.upload<25, 2048>(stream)));
    for (auto storage : {cuddl::index_storage::dense, cuddl::index_storage::sparse}) {
        auto path = temporary.path / "shared.index";
        ASSERT_TRUE((write_index<25, 2048>(archive, path, stream, storage)));
        auto loaded = index_file::load<25, 2048>(path, database, stream);
        ASSERT_TRUE(loaded) << loaded.error().message();

        // The same postings out of order must not load.
        std::ifstream input(path, std::ios::binary);
        std::string bytes{std::istreambuf_iterator<char>{input}, {}};
        input.close();
        uint64_t count = 0;
        std::memcpy(&count, bytes.data() + 24, sizeof(count));
        auto key_bytes = storage == cuddl::index_storage::sparse ? count * sizeof(uint16_t) : 0;
        auto postings = bytes.size() - 4 - key_bytes - count * 4;
        if (storage == cuddl::index_storage::sparse) {
            // Skip the leading empty-key run so the swap lands inside one nonzero key.
            auto const* keys = bytes.data() + bytes.size() - 4 - key_bytes;
            size_t first = 0;
            while (keys[2 * first] == 0 && keys[2 * first + 1] == 0) ++first;
            postings += first * 4;
        }
        std::swap_ranges(
            bytes.begin() + static_cast<std::ptrdiff_t>(postings),
            bytes.begin() + static_cast<std::ptrdiff_t>(postings + 4),
            bytes.begin() + static_cast<std::ptrdiff_t>(postings + 4)
        );
        auto checksum = libdeflate_crc32(0, bytes.data(), bytes.size() - 4);
        for (size_t i = 0; i < 4; ++i) {
            bytes[bytes.size() - 4 + i] = static_cast<char>(checksum >> (8 * i));
        }
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        EXPECT_FALSE((index_file::load<25, 2048>(path, database, stream)));
    }
}

}  // namespace
