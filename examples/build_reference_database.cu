#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <cuddl/reference_database_file.cuh>

namespace {

template <uint32_t K = 1>
void build_database(
    uint32_t k,
    uint32_t buckets,
    uint32_t exponent_bits,
    std::vector<std::filesystem::path> const& paths,
    std::filesystem::path const& output,
    cuda::stream_ref stream,
    unsigned workers
) {
    if (k != K) {
        if constexpr (K < 31) {
            return build_database<K + 1>(k, buckets, exponent_bits, paths, output, stream, workers);
        }
        throw std::invalid_argument("k must be between 1 and 31");
    }
    auto save = [&]<size_t BucketCount, uint32_t ExponentBits>() {
        using layout = cuddl::register_layout<ExponentBits, 16U - ExponentBits>;
        auto file = CUDDL_UNWRAP((cuddl::reference_database_file::build<K, BucketCount, layout>(
            paths, stream, {.parser_workers = workers}
        )));
        CUDDL_UNWRAP(file.save(output));
    };
    auto with_layout = [&]<size_t BucketCount>() {
        if (exponent_bits == 5) return save.template operator()<BucketCount, 5>();
        return save.template operator()<BucketCount, 6>();
    };
    switch (buckets) {
        case 2048:
            return with_layout.template operator()<2048>();
        case 4096:
            return with_layout.template operator()<4096>();
        case 8192:
            return with_layout.template operator()<8192>();
        case 16384:
            return with_layout.template operator()<16384>();
        case 32768:
            return with_layout.template operator()<32768>();
        case 65536:
            return with_layout.template operator()<65536>();
        case 131072:
            return with_layout.template operator()<131072>();
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
    uint32_t exponent_bits = 6;
    unsigned workers = cuddl::default_parser_workers();
    CLI::App app{
        "Build one reference sketch per FASTA/FASTQ file in a folder, recursing into "
        "subdirectories. "
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
           "--exponent-bits",
           exponent_bits,
           "Register exponent width: 6 (+10-bit mantissa) handles unbounded cardinality, 5 "
           "(+11-bit mantissa) halves false register matches for genome comparison"
    )
        ->check(CLI::IsMember({5, 6}))
        ->capture_default_str();
    app.add_option(
        "--workers",
        workers,
        "Concurrent genome loaders (defaults to machine threads); 1 minimizes RAM"
    );
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
        build_database(k, buckets, exponent_bits, paths, output, stream, workers);
        std::cout << "Saved " << paths.size() << " reference sketches to " << output << '\n';
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
