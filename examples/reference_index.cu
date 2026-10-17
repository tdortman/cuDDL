#include <CLI/CLI.hpp>

#include <cuddl/query_sketch.cuh>
#include <cuddl/reference_index_file.cuh>
#include <cuddl/replicated_database.cuh>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
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
using replicated_type = cuddl::replicated_database<cli::kmer_length, cli::buckets, cli::layout>;

/// Writes result rows as TSV on standard output, or as binary records to a file.
class result_writer {
   public:
    explicit result_writer(std::filesystem::path const& output) {
        if (output.empty()) {
            std::cout << "query_id\treference_id\tlower\tequal\thigher\tboth_empty\n";
            return;
        }
        binary_.open(output, std::ios::binary | std::ios::trunc);
        if (!binary_) throw std::runtime_error("cannot open " + output.string());
    }

    void write(std::span<cuddl::batch_search_result const> rows) {
        if (binary_.is_open()) {
            binary_.write(
                reinterpret_cast<char const*>(rows.data()),
                static_cast<std::streamsize>(rows.size() * sizeof(cuddl::batch_search_result))
            );
            return;
        }
        for (auto const& result : rows) {
            auto const& counts = result.counts;
            std::cout << result.query_id << '\t' << result.reference_id << '\t' << counts.lower
                      << '\t' << counts.equal << '\t' << counts.higher << '\t' << counts.both_empty
                      << '\n';
        }
    }

    void finish() {
        if (binary_.is_open() ? !binary_.flush() : !std::cout) {
            throw std::runtime_error("cannot write search results");
        }
    }

   private:
    std::ofstream binary_;
};

void search(
    database_type const& database,
    index_type const* index,
    std::vector<std::filesystem::path> const& paths,
    result_writer& writer,
    uint32_t minimum_matches,
    unsigned workers,
    cuddl::decompression_backend decompression,
    std::optional<std::reference_wrapper<cuddl::device_blacklist const>> blacklist,
    cuda::stream_ref stream
) {
    auto const batch_capacity = std::min(query_batch_size, paths.size());
    auto const requirements = CUDDL_UNWRAP(
        database.batch_search_requirements(static_cast<uint32_t>(batch_capacity), stream, index)
    );
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), requirements.maximum_pair_count, cuda::no_init
    );
    // Each tile's results are written before the next tile reuses the storage; the pinned host
    // copy is reused too.
    cuddl::host_batch_results host(stream);
    auto write_tile = [&](cuddl::batch_result_tile const& tile) {
        CUDDL_UNWRAP(cuddl::download(tile, stream, host));
        writer.write(host.passing());
    };
    // Without an index or a threshold every pair is reported, so the dedicated exhaustive
    // kernels run instead of counting matches only to discard nothing.
    bool const exhaustive = index == nullptr && minimum_matches == 0;
    for (size_t first = 0; first < paths.size(); first += query_batch_size) {
        auto batch_size = std::min(query_batch_size, paths.size() - first);
        auto queries = CUDDL_UNWRAP(
            query_type::sketch(
                std::span{paths.data() + first, batch_size},
                stream,
                {.parser_workers = workers, .decompression = decompression, .blacklist = blacklist}
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
    writer.finish();
}

/// Searches every unique database-row pair (query_id < reference_id) on @p devices, each holding
/// a replica. Tiles are written in query order, so the output does not depend on the devices.
void search_all_to_all(
    cuddl::reference_database_file const& file,
    std::filesystem::path const& index_path,
    result_writer& writer,
    uint32_t minimum_matches,
    std::span<cuda::device_ref const> devices
) {
    auto group = CUDDL_UNWRAP(replicated_type::upload(file, devices));
    if (!index_path.empty()) CUDDL_UNWRAP(group.load_index(index_path));
    auto const write_tile = [&](cuddl::replica_tile const& tile) {
        writer.write(CUDDL_UNWRAP(tile.download())->passing());
    };
    auto const split = cuddl::replica_search{.in_query_order = true};
    // Without an index or a threshold every pair is reported, so the exhaustive search runs.
    if (index_path.empty() && minimum_matches == 0) {
        CUDDL_UNWRAP(group.search_all_to_all(write_tile, split));
    } else {
        CUDDL_UNWRAP(
            group.search_all_to_all(write_tile, {.minimum_matches = minimum_matches}, split)
        );
    }
    writer.finish();
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
    auto* all_to_all_flag =
        query
            ->add_flag(
                "--all-to-all",
                all_to_all,
                "Search database rows against each other, each unordered pair once, instead of "
                "--query"
            )
            ->excludes(query_option);
    std::vector<int> device_ids{0};
    query
        ->add_option(
            "--devices",
            device_ids,
            "GPU ordinals sharing an --all-to-all search, each holding a replica"
        )
        ->needs(all_to_all_flag)
        ->check(CLI::NonNegativeNumber)
        ->default_str("0");
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
                    {"gpu-cores", cuddl::decompression_backend::gpu_cores},
                    {"hybrid", cuddl::decompression_backend::hybrid}
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
        if (*query && all_to_all) {
            std::vector<cuda::device_ref> devices;
            for (auto const id : device_ids) {
                if (id >= static_cast<int>(cuda::devices.size())) {
                    throw std::invalid_argument(
                        "--devices names GPU " + std::to_string(id) + " but only " +
                        std::to_string(cuda::devices.size()) + " are visible"
                    );
                }
                devices.push_back(cuda::devices[id]);
            }
            result_writer writer(output);
            search_all_to_all(file, index_path, writer, minimum_matches, devices);
            return 0;
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
        if (!file.blacklist().keys().empty()) {
            blacklist.emplace(file.blacklist(), cli::buckets, stream);
        }
        result_writer writer(output);
        search(
            database,
            index ? &*index : nullptr,
            queries,
            writer,
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
