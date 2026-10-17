#include <CLI/CLI.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cli_config.hpp"

int main(int argc, char** argv) {
    std::string folder;
    std::string output = "references.cuddl";
    std::string blacklist_path;
    unsigned workers = cuddl::default_parser_workers();
    auto decompression = cuddl::decompression_backend::automatic;
    CLI::App app{
        "Build one reference sketch per FASTA/FASTQ file in a folder, recursing into "
        "subdirectories. "
        "Recognized extensions: .fa, .fna, .fasta, .ffn, .frn, .fq, .fastq (case-insensitive). "
        "Gzip/BGZF inputs may additionally end in .gz, .bgz, or .bgzf. "
        "Files are sorted by path to assign reference IDs. Sketches use k=" +
        std::to_string(cli::kmer_length) + ", " + std::to_string(cli::buckets) + " buckets and " +
        std::to_string(cli::exponent_bits) +
        " exponent bits, set by the cli_* Meson options at build time."
    };
    app.add_option("folder", folder, "Genome folder")->required()->check(CLI::ExistingDirectory);
    app.add_option(
           "--blacklist",
           blacklist_path,
           "BBTools DDL FASTA (plain/gzip, fused supported), or exact DNA k-mer lines"
    )
        ->check(CLI::ExistingFile);
    app.add_option("-o,--output", output, "Binary database destination (replaces existing file)")
        ->default_val(output);
    app.add_option(
        "--workers",
        workers,
        "Concurrent genome loaders (defaults to machine threads); 1 minimizes RAM"
    );
    app.add_option(
           "--decompression", decompression, "Decompression backend; GPU allows format fallbacks"
    )
        ->transform(
            CLI::CheckedTransformer(
                std::map<std::string, cuddl::decompression_backend>{
                    {"automatic", cuddl::decompression_backend::automatic},
                    {"hybrid", cuddl::decompression_backend::hybrid},
                    {"cpu", cuddl::decompression_backend::cpu},
                    {"gpu", cuddl::decompression_backend::gpu},
                    {"gpu-cores", cuddl::decompression_backend::gpu_cores},
                }
            )
        )
        ->default_str("automatic");
    CLI11_PARSE(app, argc, argv);

    try {
        auto paths = CUDDL_UNWRAP(cuddl::genome_paths(folder));
        if (!blacklist_path.empty()) {
            std::erase_if(paths, [&](auto const& path) {
                return std::filesystem::equivalent(path, blacklist_path);
            });
        }
        if (std::filesystem::exists(output)) {
            // Each lookup of an input on a network file system waits for the server: 10,000
            // genomes linked from an NFS directory took 3.7 s one at a time.
            std::atomic<size_t> next{0};
            std::atomic<bool> clash{false};
            std::vector<std::jthread> threads;
            for (size_t t = 0; t < std::min<size_t>(64, paths.size()); ++t) {
                threads.emplace_back([&] {
                    std::error_code error;
                    for (auto i = next++; i < paths.size() && !clash; i = next++) {
                        if (std::filesystem::equivalent(paths[i], output, error)) clash = true;
                    }
                });
            }
            threads.clear();
            if (clash) throw std::invalid_argument("output must not overwrite an input genome");
        }
        if (paths.empty()) {
            throw std::invalid_argument("folder contains no supported FASTA/FASTQ files");
        }
        cuda::stream stream{cuda::devices[0]};
        std::optional<cuddl::device_blacklist> blacklist;
        if (!blacklist_path.empty()) {
            if (std::filesystem::exists(output) &&
                std::filesystem::equivalent(blacklist_path, output)) {
                throw std::invalid_argument("output must not overwrite the blacklist");
            }
            blacklist.emplace(
                CUDDL_UNWRAP(cuddl::kmer_blacklist::load(blacklist_path, cli::kmer_length)),
                cli::buckets,
                stream
            );
        }
        auto file = CUDDL_UNWRAP(
            (cuddl::reference_database_file::build<cli::kmer_length, cli::buckets, cli::layout>(
                paths,
                stream,
                {.parser_workers = workers,
                 .decompression = decompression,
                 .blacklist = blacklist ? std::optional{std::cref(*blacklist)} : std::nullopt}
            ))
        );
        CUDDL_UNWRAP(file.save(output));
        std::cout << "Saved " << paths.size() << " reference sketches to " << output << '\n';
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
