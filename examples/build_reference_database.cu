#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <cuddl/reference_database_file.cuh>

namespace {

template <uint32_t K = 1>
void build_database(
    uint32_t k,
    uint32_t buckets,
    std::vector<std::filesystem::path> const& paths,
    std::filesystem::path const& output,
    cuda::stream_ref stream,
    unsigned workers
) {
    if (k != K) {
        if constexpr (K < 31) {
            return build_database<K + 1>(k, buckets, paths, output, stream, workers);
        }
        throw std::invalid_argument("k must be between 1 and 31");
    }
    auto save = [&]<size_t BucketCount>() {
        auto file = CUDDL_UNWRAP(
            (cuddl::reference_database_file::build<K, BucketCount>(paths, stream, workers))
        );
        CUDDL_UNWRAP(file.save(output));
    };
    switch (buckets) {
        case 2048:
            return save.template operator()<2048>();
        case 4096:
            return save.template operator()<4096>();
        case 8192:
            return save.template operator()<8192>();
        case 16384:
            return save.template operator()<16384>();
        case 32768:
            return save.template operator()<32768>();
        case 65536:
            return save.template operator()<65536>();
        case 131072:
            return save.template operator()<131072>();
        default:
            throw std::invalid_argument("unsupported bucket count");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string folder;
    std::string output = "references.cuddl";
    uint32_t k{};
    uint32_t buckets{};
    unsigned workers = 0;
    CLI::App app{
        "Build one reference sketch per FASTA/FASTQ file directly in a folder. "
        "Recognized extensions: .fa, .fna, .fasta, .ffn, .frn, .fq, .fastq (case-insensitive). "
        "Gzip/BGZF inputs may additionally end in .gz, .bgz, or .bgzf. "
        "Files are sorted by path to assign reference IDs."
    };
    app.add_option("folder", folder, "Genome folder")->required()->check(CLI::ExistingDirectory);
    app.add_option("-k,--k", k, "K-mer length")->required()->check(CLI::Range(1, 31));
    app.add_option("-b,--buckets", buckets, "Sketch bucket count")
        ->required()
        ->check(CLI::IsMember({2048, 4096, 8192, 16384, 32768, 65536, 131072}));
    app.add_option("-o,--output", output, "Binary database destination (replaces existing file)")
        ->default_val(output);
    app.add_option(
           "--workers",
           workers,
           "Concurrent genomes in host memory; 0 selects up to 8, 1 minimizes RAM"
    )
        ->default_val(workers);
    CLI11_PARSE(app, argc, argv);

    try {
        std::vector<std::filesystem::path> paths;
        for (auto const& entry : std::filesystem::directory_iterator(folder)) {
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
        build_database(k, buckets, paths, output, stream, workers);
        std::cout << "Saved " << paths.size() << " reference sketches to " << output << '\n';
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
