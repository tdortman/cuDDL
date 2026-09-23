#include <CLI/CLI.hpp>

#include <iostream>
#include <string>

#include "reference_index_command.hpp"

namespace {

template <size_t Buckets, uint32_t K>
void dispatch_layout(reference_index_command const& command, uint32_t exponent_bits) {
    if (exponent_bits == 5) return dispatch_reference_index<Buckets, K, 5>(command);
    if (exponent_bits == 6) return dispatch_reference_index<Buckets, K, 6>(command);
    throw std::invalid_argument("unsupported register exponent width");
}

template <size_t Buckets, uint32_t K = 1>
void dispatch_kmers(
    reference_index_command const& command,
    cuddl::score_compatibility const& compatibility
) {
    if (compatibility.kmer_length == K) {
        return dispatch_layout<Buckets, K>(command, compatibility.exponent_bits);
    }
    if constexpr (K < 31) return dispatch_kmers<Buckets, K + 1>(command, compatibility);
    throw std::invalid_argument("unsupported k-mer length");
}

/// Bucket counts are the powers of two from 2048 through 131072.
template <size_t Buckets = 2048>
void dispatch_buckets(
    reference_index_command const& command,
    cuddl::score_compatibility const& compatibility
) {
    if (compatibility.bucket_count == Buckets) {
        return dispatch_kmers<Buckets>(command, compatibility);
    }
    if constexpr (Buckets < 131072) return dispatch_buckets<Buckets * 2>(command, compatibility);
    throw std::invalid_argument("unsupported sketch bucket count");
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Build persistent retrieval indexes or search a validated index file"};
    app.require_subcommand(1);
    std::string database_path, index_path, format = "dense";
    std::vector<std::filesystem::path> queries;
    std::filesystem::path output;
    uint32_t minimum_matches = 1;
    unsigned workers = cuddl::default_parser_workers();
    auto* build = app.add_subcommand(
        "build", "Create a dense or sparse index file from a reference database"
    );
    build->add_option("database", database_path)->required()->check(CLI::ExistingFile);
    build->add_option("-o,--output", index_path, "Index file destination (replaces existing file)")
        ->required();
    build->add_option("--format", format)
        ->check(CLI::IsMember({"dense", "sparse"}))
        ->default_val(format);
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
        auto database = CUDDL_UNWRAP(cuddl::reference_database_file::load(database_path));
        auto const compatibility = database.metadata().compatibility;
        if (minimum_matches > compatibility.indexed_bucket_count) {
            throw std::invalid_argument("minimum matches exceeds indexed bucket count");
        }
        reference_index_command command{
            database,
            index_path,
            queries,
            output,
            all_to_all,
            minimum_matches,
            workers,
            bool(*build),
            format == "dense" ? cuddl::index_storage::dense : cuddl::index_storage::sparse
        };
        dispatch_buckets(command, compatibility);
        if (*build) std::cout << "Saved " << format << " index to " << index_path << '\n';

    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
