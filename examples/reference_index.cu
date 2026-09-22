#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "reference_index_command.hpp"

int main(int argc, char** argv) {
    CLI::App app{"Build persistent retrieval indexes or search a validated index file"};
    app.require_subcommand(1);
    std::string database_path, index_path, format = "dense";
    std::vector<std::filesystem::path> queries;
    uint32_t minimum_matches = 1;
    unsigned workers = cuddl::default_parser_workers;
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
    query
        ->add_option("--query", queries, "One query genome per FASTA/FASTQ file, in supplied order")
        ->required()
        ->check(CLI::ExistingFile);
    query
        ->add_option(
            "--minimum-matches",
            minimum_matches,
            "Required matching indexed buckets; zero searches all references"
        )
        ->default_val(minimum_matches);
    query->add_option("--workers", workers, "Concurrent query genome loaders")
        ->check(CLI::PositiveNumber);
    CLI11_PARSE(app, argc, argv);
    try {
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
            minimum_matches,
            workers,
            bool(*build),
            format == "dense" ? cuddl::index_storage::dense : cuddl::index_storage::sparse
        };
        switch (compatibility.bucket_count) {
            case 2048:
                dispatch_reference_index_2048(command);
                break;
            case 4096:
                dispatch_reference_index_4096(command);
                break;
            case 8192:
                dispatch_reference_index_8192(command);
                break;
            case 16384:
                dispatch_reference_index_16384(command);
                break;
            case 32768:
                dispatch_reference_index_32768(command);
                break;
            case 65536:
                dispatch_reference_index_65536(command);
                break;
            case 131072:
                dispatch_reference_index_131072(command);
                break;
            default:
                throw std::invalid_argument("unsupported sketch bucket count");
        }
        if (*build) std::cout << "Saved " << format << " index to " << index_path << '\n';

    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
