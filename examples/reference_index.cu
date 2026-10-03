#include <CLI/CLI.hpp>

#include <cuddl/query_sketch.cuh>
#include <cuddl/reference_index_file.cuh>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "cli_config.hpp"

namespace {

// Binary output rows are raw little-endian batch_search_result records, documented in README.md.
static_assert(sizeof(cuddl::batch_search_result) == 24);
static_assert(std::is_trivially_copyable_v<cuddl::batch_search_result>);

/// Query genomes sketched per batch; each batch then runs as one batched search.
constexpr size_t query_batch_size = 4096;

using database_type = cuddl::reference_database<cli::kmer_length, cli::buckets, cli::layout>;
using index_type = cuddl::reference_index<cli::kmer_length, cli::buckets, cli::layout>;
using query_type = cuddl::query_sketch_batch<cli::kmer_length, cli::buckets, cli::layout>;

void search(
    database_type const& database,
    index_type const* index,
    std::vector<std::filesystem::path> const& paths,
    std::filesystem::path const& output,
    bool all_to_all,
    uint32_t minimum_matches,
    unsigned workers,
    cuddl::decompression_backend decompression,
    std::optional<std::reference_wrapper<cuddl::device_blacklist const>> blacklist,
    cuda::stream_ref stream
) {
    auto const batch_capacity = std::min(query_batch_size, paths.size());
    auto const requirements = CUDDL_UNWRAP(
        all_to_all ? database.all_to_all_search_requirements(index)
                   : database.batch_search_requirements(
                         static_cast<uint32_t>(batch_capacity), stream, index
                     )
    );
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), requirements.maximum_pair_count, cuda::no_init
    );
    std::ofstream binary;
    if (output.empty()) {
        std::cout << "query_id\treference_id\tlower\tequal\thigher\tboth_empty\n";
    } else {
        binary.open(output, std::ios::binary | std::ios::trunc);
        if (!binary) throw std::runtime_error("cannot open " + output.string());
    }
    // Each tile's results are written before the next tile reuses the storage.
    auto write_tile = [&](cuddl::batch_result_tile const& tile) {
        auto const tile_copy = CUDDL_UNWRAP(cuddl::download(tile, stream));
        auto const passing = tile_copy.passing();
        if (binary.is_open()) {
            binary.write(
                reinterpret_cast<char const*>(passing.data()),
                static_cast<std::streamsize>(passing.size() * sizeof(cuddl::batch_search_result))
            );
            return;
        }
        for (auto const& result : passing) {
            auto const& counts = result.counts;
            std::cout << result.query_id << '\t' << result.reference_id << '\t' << counts.lower
                      << '\t' << counts.equal << '\t' << counts.higher << '\t' << counts.both_empty
                      << '\n';
        }
    };
    // Without an index or a threshold every pair is reported, so the dedicated exhaustive
    // kernels run instead of counting matches only to discard nothing.
    bool const exhaustive = index == nullptr && minimum_matches == 0;
    if (all_to_all) {
        // Database rows are the queries; only pairs with query_id < reference_id are searched.
        if (exhaustive) {
            CUDDL_UNWRAP(database.search_all_to_all_async(
                {workspace.data(), workspace.size()},
                {results.data(), results.size()},
                write_tile,
                {},
                stream
            ));
        } else {
            CUDDL_UNWRAP(database.search_all_to_all_async(
                {workspace.data(), workspace.size()},
                {results.data(), results.size()},
                write_tile,
                {},
                {.minimum_matches = minimum_matches},
                stream,
                index
            ));
        }
    } else {
        for (size_t first = 0; first < paths.size(); first += query_batch_size) {
            auto batch_size = std::min(query_batch_size, paths.size() - first);
            auto queries = CUDDL_UNWRAP(
                query_type::sketch(
                    std::span{paths.data() + first, batch_size},
                    stream,
                    {.parser_workers = workers,
                     .decompression = decompression,
                     .blacklist = blacklist}
                )
            );
            if (exhaustive) {
                CUDDL_UNWRAP(database.search_batch_async(
                    queries.scores(),
                    queries.compatibility(),
                    static_cast<uint32_t>(first),
                    {workspace.data(), workspace.size()},
                    {results.data(), results.size()},
                    write_tile,
                    {},
                    stream
                ));
            } else {
                CUDDL_UNWRAP(database.search_batch_async(
                    queries.scores(),
                    queries.compatibility(),
                    static_cast<uint32_t>(first),
                    {workspace.data(), workspace.size()},
                    {results.data(), results.size()},
                    write_tile,
                    {},
                    {.minimum_matches = minimum_matches},
                    stream,
                    index
                ));
            }
        }
    }
    if (binary.is_open() ? !binary.flush() : !std::cout) {
        throw std::runtime_error("cannot write search results");
    }
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Build persistent retrieval indexes or search a validated index file"};
    app.require_subcommand(1);
    std::string database_path, index_path;
    auto format = cuddl::index_storage::automatic;
    std::vector<std::filesystem::path> queries;
    std::filesystem::path output;
    uint32_t minimum_matches = 1;
    unsigned workers = cuddl::default_parser_workers();
    auto decompression = cuddl::decompression_backend::automatic;
    auto* build = app.add_subcommand(
        "build", "Create a dense or sparse index file from a reference database"
    );
    build->add_option("database", database_path)->required()->check(CLI::ExistingFile);
    build->add_option("-o,--output", index_path, "Index file destination (replaces existing file)")
        ->required();
    build
        ->add_option(
            "--format", format, "Index storage; automatic picks the one using less device memory"
        )
        ->transform(
            CLI::CheckedTransformer(
                std::map<std::string, cuddl::index_storage>{
                    {"automatic", cuddl::index_storage::automatic},
                    {"dense", cuddl::index_storage::dense},
                    {"sparse", cuddl::index_storage::sparse}
                }
            )
        )
        ->default_str("automatic");
    auto* query = app.add_subcommand(
        "search", "Load once and search all query genomes; format is detected from the index"
    );
    query->add_option("database", database_path)->required()->check(CLI::ExistingFile);
    query->add_option("--index", index_path, "Optional acceleration index file")
        ->check(CLI::ExistingFile);
    auto* query_option =
        query
            ->add_option(
                "--query", queries, "One query genome per FASTA/FASTQ file, in supplied order"
            )
            ->check(CLI::ExistingFile);
    bool all_to_all = false;
    query
        ->add_flag(
            "--all-to-all",
            all_to_all,
            "Search database rows against each other, each unordered pair once, instead of --query"
        )
        ->excludes(query_option);
    query
        ->add_option(
            "--minimum-matches",
            minimum_matches,
            "Required matching indexed buckets; zero searches all references"
        )
        ->default_val(minimum_matches);
    query->add_option(
        "-o,--output",
        output,
        "Write binary result records here instead of TSV on standard output (replaces file)"
    );
    query->add_option("--workers", workers, "Concurrent query genome loaders")
        ->check(CLI::PositiveNumber);
    (*query)
        .add_option(
            "--decompression", decompression, "Decompression backend; GPU allows format fallbacks"
        )
        ->transform(
            CLI::CheckedTransformer(
                std::map<std::string, cuddl::decompression_backend>{
                    {"automatic", cuddl::decompression_backend::automatic},
                    {"cpu", cuddl::decompression_backend::cpu},
                    {"gpu", cuddl::decompression_backend::gpu},
                    {"coherent", cuddl::decompression_backend::coherent}
                }
            )
        )
        ->default_str("automatic");
    app.set_config(
        "--config", "", "TOML options file; long query lists go under [search] as query = [...]"
    );
    CLI11_PARSE(app, argc, argv);
    try {
        if (*query && !all_to_all && queries.empty()) {
            throw std::invalid_argument("search needs --query or --all-to-all");
        }
        if (*build && std::filesystem::exists(index_path) &&
            std::filesystem::equivalent(database_path, index_path)) {
            throw std::invalid_argument("index output must not overwrite the reference database");
        }
        auto const file = CUDDL_UNWRAP(cuddl::reference_database_file::load(database_path));
        auto const compatibility = file.metadata().compatibility;
        cli::require_compatible(compatibility);
        if (minimum_matches > compatibility.indexed_bucket_count) {
            throw std::invalid_argument("minimum matches exceeds indexed bucket count");
        }
        cuda::stream stream{cuda::devices[0]};
        auto database =
            CUDDL_UNWRAP((file.upload<cli::kmer_length, cli::buckets, cli::layout>(stream)));
        if (*build) {
            auto index = CUDDL_UNWRAP(index_type::build_async(database, stream, format));
            CUDDL_UNWRAP(cuddl::reference_index_file::save(index, database, index_path, stream));
            std::cout << "Saved "
                      << (index.storage() == cuddl::index_storage::dense ? "dense" : "sparse")
                      << " index to " << index_path << '\n';
            return 0;
        }
        std::optional<index_type> index;
        if (!index_path.empty()) {
            index.emplace(
                CUDDL_UNWRAP(cuddl::reference_index_file::load(index_path, database, stream))
            );
        }
        std::optional<cuddl::device_blacklist> blacklist;
        if (!all_to_all && !file.blacklist().keys().empty()) {
            blacklist.emplace(file.blacklist(), cli::buckets, stream);
        }
        search(
            database,
            index ? &*index : nullptr,
            queries,
            output,
            all_to_all,
            minimum_matches,
            workers,
            decompression,
            blacklist ? std::optional{std::cref(*blacklist)} : std::nullopt,
            stream
        );
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
