#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "cli_config.hpp"

int main(int argc, char** argv) {
    std::string folder;
    std::string output = "references.cuddl";
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
                    {"cpu", cuddl::decompression_backend::cpu},
                    {"gpu", cuddl::decompression_backend::gpu},
                    {"coherent", cuddl::decompression_backend::coherent}
                }
            )
        )
        ->default_str("automatic");
    CLI11_PARSE(app, argc, argv);

    try {
        std::vector<std::filesystem::path> paths;
        for (auto const& entry : std::filesystem::recursive_directory_iterator(
                 folder, std::filesystem::directory_options::skip_permission_denied
             )) {
            if (!entry.is_regular_file()) continue;
            auto filename = entry.path().filename().string();
            std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            auto extension = std::filesystem::path(filename).extension().string();
            if (extension == ".gz" || extension == ".bgz" || extension == ".bgzf") {
                extension = std::filesystem::path(filename).stem().extension().string();
            }
            if (extension == ".fa" || extension == ".fna" || extension == ".fasta" ||
                extension == ".ffn" || extension == ".frn" || extension == ".fq" ||
                extension == ".fastq") {
                if (std::filesystem::exists(output) &&
                    std::filesystem::equivalent(entry.path(), output)) {
                    throw std::invalid_argument("output must not overwrite an input genome");
                }
                paths.push_back(entry.path());
            }
        }
        if (paths.empty()) {
            throw std::invalid_argument("folder contains no supported FASTA/FASTQ files");
        }
        std::sort(paths.begin(), paths.end());
        cuda::stream stream{cuda::devices[0]};
        auto file = CUDDL_UNWRAP(
            (cuddl::reference_database_file::build<cli::kmer_length, cli::buckets, cli::layout>(
                paths, stream, {.parser_workers = workers, .decompression = decompression}
            ))
        );
        CUDDL_UNWRAP(file.save(output));
        std::cout << "Saved " << paths.size() << " reference sketches to " << output << '\n';
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
