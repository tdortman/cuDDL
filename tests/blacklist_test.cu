#include <gtest/gtest.h>
#include <cuddl/cuddl.cuh>

#include <array>
#include <fstream>
#include <map>
#include <random>
#include <ranges>

namespace {
constexpr uint32_t k = 25;
constexpr size_t buckets = 2048;
using file_type = cuddl::reference_database_file;
using query_type = cuddl::query_sketch_batch<k, buckets>;

struct temporary_directory {
    std::filesystem::path path;
    temporary_directory() {
        auto pattern = (std::filesystem::temp_directory_path() / "cuddl-blacklist-XXXXXX").string();
        if (!::mkdtemp(pattern.data())) {
            throw std::runtime_error("cannot create temporary directory");
        }
        path = pattern;
    }
    ~temporary_directory() {
        std::filesystem::remove_all(path);
    }
};

std::string dna(uint64_t word) {
    std::string result(k, 'A');
    for (size_t i = k; i > 0; --i) {
        result[i - 1] = "ACTG"[word & 3];
        word >>= 2;
    }
    return result;
}

std::string reverse_complement(std::string const& bases) {
    std::string result;
    for (char base : std::views::reverse(bases)) {
        result.push_back("TGAC"[cuddl::detail::encode_base(base)]);
    }
    return result;
}

template <typename T>
std::vector<T> download(T const* data, size_t size, cuda::stream_ref stream) {
    std::vector<T> result(size);
    cuda::copy_bytes(
        stream, cuddl::device_span<T const>{data, size}, cuda::std::span{result.data(), size}
    );
    stream.sync();
    return result;
}

TEST(Blacklist, ExactMembershipSurvivesPersistenceAndIndexing) {
    temporary_directory temporary;
    cuda::stream stream{cuda::devices[0]};
    // Find different canonical literals with the same bucket and score, then a lower winner.
    std::map<std::pair<size_t, uint32_t>, uint64_t> seen;
    std::mt19937_64 random(934);
    uint64_t blocked = 0, collision = 0, runner = 0;
    auto next = [&] {
        return cuddl::kmer_blacklist::canonical(random() & ((uint64_t{1} << (2 * k)) - 1), k);
    };
    for (size_t attempts = 0; attempts < 1000000 && blocked == 0; ++attempts) {
        auto word = next();
        auto hash = cuddl::detail::hash_kmer(word);
        auto key = std::pair{
            cuddl::detail::bucket_of<buckets>(hash), uint32_t{cuddl::detail::score(hash)}
        };
        auto [it, inserted] = seen.emplace(key, word);
        if (!inserted && it->second != word) {
            blocked = it->second;
            collision = word;
        }
    }
    ASSERT_NE(blocked, 0);
    auto const bucket = cuddl::detail::bucket_of<buckets>(cuddl::detail::hash_kmer(blocked));
    auto const winning_score = cuddl::detail::score(cuddl::detail::hash_kmer(blocked));
    for (size_t attempts = 0; attempts < 1000000 && runner == 0; ++attempts) {
        auto word = next();
        auto hash = cuddl::detail::hash_kmer(word);
        if (cuddl::detail::bucket_of<buckets>(hash) == bucket &&
            cuddl::detail::score(hash) < winning_score) {
            runner = word;
        }
    }
    ASSERT_NE(runner, 0);
    auto list_path = temporary.path / "blacklist.txt";
    {
        std::ofstream out(list_path);
        out << dna(blocked) << '\n' << reverse_complement(dna(blocked)) << "\r\n";
    }
    auto list = CUDDL_UNWRAP(cuddl::kmer_blacklist::load(list_path, k));
    EXPECT_EQ(list.keys(), std::vector<uint64_t>{blocked});
    auto lookup = std::make_shared<cuddl::device_blacklist>(list, buckets, stream);
    std::array<std::string, 3> bases{
        dna(blocked) + 'N' + dna(runner), dna(collision), reverse_complement(dna(blocked))
    };
    std::array<cuddl::sequence_record, 3> records{{{bases[0]}, {bases[1]}, {bases[2]}}};
    std::array<cuddl::sequence_genome, 3> genomes;
    for (size_t i = 0; i < genomes.size(); ++i) {
        genomes[i] = {std::span{records}.subspan(i, 1), "reference"};
    }
    auto file = CUDDL_UNWRAP((file_type::build_from_sequences<k, buckets>(
        genomes, stream, {.blacklist = std::cref(*lookup)}
    )));
    std::vector<uint16_t> expected(3 * buckets, 0);
    expected[bucket] = cuddl::detail::score(cuddl::detail::hash_kmer(runner));
    expected[buckets + bucket] = winning_score;
    EXPECT_EQ(std::vector<uint16_t>(file.rows().begin(), file.rows().end()), expected);

    std::vector<std::filesystem::path> paths;
    for (size_t i = 0; i < bases.size(); ++i) {
        paths.push_back(temporary.path / (std::to_string(i) + ".fa"));
        std::ofstream out(paths.back());
        out << ">sequence\n" << bases[i] << '\n';
    }
    auto staged = CUDDL_UNWRAP(
        (file_type::build<k, buckets>(paths, stream, {.blacklist = std::cref(*lookup)}))
    );
    EXPECT_EQ(std::vector<uint16_t>(staged.rows().begin(), staged.rows().end()), expected);
    auto store = CUDDL_UNWRAP(
        (cuddl::build_sketch_store<k, buckets>(paths, stream, {.blacklist = std::cref(*lookup)}))
    );
    std::vector<std::string> names{"a", "b", "c"};
    auto adopted = CUDDL_UNWRAP(
        (file_type::from_store<k, buckets>({store.data(), store.size()}, names, stream, list))
    );
    EXPECT_EQ(adopted.metadata(), file.metadata());
    EXPECT_EQ(std::vector<uint16_t>(adopted.rows().begin(), adopted.rows().end()), expected);

    auto database_path = temporary.path / "references.cuddl";
    CUDDL_UNWRAP(file.save(database_path));
    std::filesystem::remove(list_path);
    auto loaded = CUDDL_UNWRAP(file_type::load(database_path));
    // Recompute CRC after corrupting the literal so semantic validation, not CRC, rejects it.
    {
        std::ifstream input(database_path, std::ios::binary);
        std::string bytes{std::istreambuf_iterator<char>{input}, {}};
        auto reader = cuddl::detail::database_file_reader{
            std::ifstream(database_path, std::ios::binary), bytes.size()
        };
        std::array<char, 10> prefix{};
        CUDDL_UNWRAP(reader.bytes(prefix.data(), prefix.size()));
        cuddl::reference_database_metadata metadata;
        uint32_t count = 0;
        CUDDL_UNWRAP(cuddl::detail::database_file_metadata(reader, metadata, count));
        auto key_offset = static_cast<size_t>(reader.input.tellg());
        bytes[key_offset + 7] = static_cast<char>(0xff);
        auto checksum = libdeflate_crc32(0, bytes.data(), bytes.size() - 4);
        for (size_t i = 0; i < 4; ++i) {
            bytes[bytes.size() - 4 + i] = static_cast<char>(checksum >> (8 * i));
        }
        auto corrupt_path = temporary.path / "corrupt.cuddl";
        {
            std::ofstream out(corrupt_path, std::ios::binary);
            out.write(bytes.data(), bytes.size());
        }
        EXPECT_FALSE(file_type::load(corrupt_path));
    }
    cuddl::device_blacklist query_filter(loaded.blacklist(), buckets, stream);
    auto query =
        CUDDL_UNWRAP(query_type::sketch(paths, stream, {.blacklist = std::cref(query_filter)}));
    EXPECT_EQ(download(query.scores().data(), query.scores().size(), stream), expected);
    auto moved = std::move(query);
    EXPECT_EQ(moved.compatibility(), file.metadata().compatibility);
    auto db = CUDDL_UNWRAP((loaded.upload<k, buckets>(stream)));
    auto index = CUDDL_UNWRAP(
        (cuddl::reference_index<k, buckets>::build_async(db, stream, cuddl::index_storage::sparse))
    );
    auto index_path = temporary.path / "references.index";
    CUDDL_UNWRAP(cuddl::reference_index_file::save(index, db, index_path, stream));
    auto restored_index = CUDDL_UNWRAP(cuddl::reference_index_file::load(index_path, db, stream));
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream,
        stream.device(),
        CUDDL_UNWRAP(db.search_workspace_bytes(stream, &restored_index)),
        cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), db.single_query_result_count(), cuda::no_init
    );
    auto count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, cuda::no_init);
    auto search = [&](cuddl::score_compatibility compatibility) {
        return db.search_async(
            {moved.scores().data(), buckets},
            compatibility,
            {workspace.data(), workspace.size()},
            {results.data(), results.size()},
            {count.data(), 1},
            {.minimum_matches = 1},
            stream,
            &restored_index
        );
    };
    EXPECT_FALSE(search(cuddl::score_compatibility::current<k, buckets>()));
    CUDDL_UNWRAP(search(moved.compatibility()));
    EXPECT_EQ(download(count.data(), 1, stream)[0], 1);
    EXPECT_EQ(download(results.data(), 1, stream)[0].reference_id, 0);

    cuddl::sketch<k, buckets> packed(stream, lookup), raw(stream, lookup), unfiltered(stream);
    std::vector<uint64_t> words{blocked, runner};
    auto input = cuda::make_device_buffer<uint64_t>(stream, stream.device(), words);
    CUDDL_UNWRAP(packed.add(input, stream));
    auto ascii = cuda::make_device_buffer<char>(stream, stream.device(), bases[0]);
    CUDDL_UNWRAP(raw.add_sequence(ascii, stream));
    EXPECT_EQ(
        download(std::as_const(packed).data().data(), buckets, stream),
        download(std::as_const(raw).data().data(), buckets, stream)
    );
    EXPECT_FALSE(packed.compare(unfiltered, stream));
    EXPECT_FALSE(packed.assign_async(std::as_const(unfiltered).data(), stream, {}));
    EXPECT_EQ(
        download(std::as_const(packed).data().data(), buckets, stream)[bucket], expected[bucket]
    );
}

TEST(Blacklist, RejectsInvalidInputAndNormalizesEquivalentLists) {
    temporary_directory temporary;
    auto path = temporary.path / "list.txt";
    auto load = [&](std::string const& text) {
        {
            std::ofstream out(path);
            out << text;
        }
        return cuddl::kmer_blacklist::load(path, k);
    };
    EXPECT_FALSE(load("ACGT\n"));
    EXPECT_FALSE(load(std::string(k, 'N') + '\n'));
    EXPECT_FALSE(cuddl::kmer_blacklist::load(path, 32));
    auto empty = CUDDL_UNWRAP(load("\n"));
    EXPECT_EQ(empty.identity(), 0);
    auto a = dna(cuddl::kmer_blacklist::canonical(12345, k));
    auto b = dna(cuddl::kmer_blacklist::canonical(45678, k));
    auto first = CUDDL_UNWRAP(load(a + '\n' + b + '\n'));
    auto second = CUDDL_UNWRAP(load(reverse_complement(b) + '\n' + a + '\n' + b + '\n'));
    EXPECT_EQ(first.identity(), second.identity());
    EXPECT_EQ(first.keys(), second.keys());
}

TEST(Blacklist, BBToolsFastaAndFusedGzipProduceTheSameLiteralSet) {
    temporary_directory temporary;
    std::string const fused = "ACGTTGCACTGATCGAGGCTAACGTTGCAC";
    auto fasta = temporary.path / "fused.fa";
    auto lines = temporary.path / "literal.txt";
    std::string const content = ">kmer_HEX raw=100 g=80\n" + fused.substr(0, 12) + "\n" +
                                fused.substr(12) + "NNN" + reverse_complement(fused.substr(0, k)) +
                                "\n>short-a\nACGT\n>short-b\nACGT\n";
    {
        std::ofstream out(fasta);
        out << content;
    }
    {
        std::ofstream out(lines);
        for (size_t i = 0; i + k <= fused.size(); ++i) {
            out << fused.substr(i, k) << '\n';
        }
    }
    auto expected = CUDDL_UNWRAP(cuddl::kmer_blacklist::load(lines, k));
    auto actual = CUDDL_UNWRAP(cuddl::kmer_blacklist::load(fasta, k));
    EXPECT_EQ(actual.keys(), expected.keys());
    EXPECT_EQ(actual.identity(), expected.identity());
    // A stored DEFLATE block needs only the decompressor linked by the library.
    ASSERT_LE(content.size(), 0xffffU);
    std::string compressed{"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x01", 11};
    auto put16 = [&](uint32_t value) {
        compressed.push_back(static_cast<char>(value & 0xffU));
        compressed.push_back(static_cast<char>(value >> 8));
    };
    put16(static_cast<uint32_t>(content.size()));
    put16(static_cast<uint32_t>(~content.size() & 0xffffU));
    compressed += content;
    for (auto word :
         {libdeflate_crc32(0, content.data(), content.size()),
          static_cast<uint32_t>(content.size())}) {
        put16(word & 0xffffU);
        put16(word >> 16);
    }
    auto gzip = temporary.path / "fused.fa.gz";
    {
        std::ofstream out(gzip, std::ios::binary);
        out.write(compressed.data(), compressed.size());
    }
    auto decoded = CUDDL_UNWRAP(cuddl::kmer_blacklist::load(gzip, k));
    EXPECT_EQ(decoded.keys(), expected.keys());
    EXPECT_EQ(decoded.identity(), expected.identity());
}

TEST(Blacklist, LongSequenceFloorPreservesFilteredAndUnfilteredWinners) {
    cuda::stream stream{cuda::devices[0]};
    std::mt19937_64 random(715);
    std::string bases(32 * 1024 * 1024, 'A');
    for (auto& base : bases) {
        base = "ACTG"[random() & 3];
    }
    std::vector<uint64_t> blocked;
    for (size_t i = 0; i < 23020; ++i) {
        uint64_t word = 0;
        for (size_t j = 0; j < k; ++j) {
            word = (word << 2) | cuddl::detail::encode_base(bases[i * 64 + j]);
        }
        blocked.push_back(cuddl::kmer_blacklist::canonical(word, k));
    }
    auto lookup = std::make_shared<cuddl::device_blacklist>(
        cuddl::kmer_blacklist(k, std::move(blocked)), buckets, stream
    );
    auto ascii = cuda::make_device_buffer<char>(stream, stream.device(), bases);
    for (bool filtered : {false, true}) {
        std::vector<uint32_t> expected(buckets, 0);
        uint64_t forward = 0, reverse = 0;
        for (size_t i = 0; i < bases.size(); ++i) {
            auto code = cuddl::detail::encode_base(bases[i]);
            forward = ((forward << 2) | code) & ((uint64_t{1} << 50) - 1);
            reverse = (reverse >> 2) | (uint64_t{code ^ 2U} << 48);
            if (i + 1 < k) continue;
            auto word = std::max(forward, reverse);
            auto const& keys = lookup->source().keys();
            if (filtered && std::binary_search(keys.begin(), keys.end(), word)) continue;
            auto hash = cuddl::detail::hash_kmer(word);
            auto& winner = expected[cuddl::detail::bucket_of<buckets>(hash)];
            winner = std::max(winner, uint32_t{cuddl::detail::score(hash)});
        }
        auto baseline =
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), buckets, uint32_t{0});
        CUDDL_UNWRAP(
            (cuddl::detail::launch_sequence_add<buckets, cuddl::default_register_layout, false>(
                ascii,
                k,
                {baseline.data(), buckets},
                stream,
                filtered ? lookup->view() : cuda::std::nullopt
            ))
        );
        EXPECT_EQ(download(baseline.data(), buckets, stream), expected);
        cuddl::sketch<k, buckets> sketch(
            stream,
            filtered ? std::optional<std::shared_ptr<cuddl::device_blacklist const>>{lookup}
                     : std::nullopt
        );
        CUDDL_UNWRAP(sketch.add_sequence(ascii, stream));
        EXPECT_EQ(download(std::as_const(sketch).data().data(), buckets, stream), expected);
        std::array<cuddl::sequence_record, 1> records{{{bases}}};
        std::array<cuddl::sequence_genome, 1> genomes{{{records, "long"}}};
        auto file = CUDDL_UNWRAP((file_type::build_from_sequences<k, buckets>(
            genomes,
            stream,
            {.blacklist = filtered ? std::optional{std::cref(*lookup)} : std::nullopt}
        )));
        EXPECT_EQ(std::vector<uint32_t>(file.rows().begin(), file.rows().end()), expected);
    }
}
}  // namespace
