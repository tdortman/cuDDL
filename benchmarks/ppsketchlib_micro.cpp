// pp-sketchlib lanes for the micro comparison (scripts/run_micro_comparison.py), driven through
// pp-sketchlib's own classes: the sketching its CLI runs on the CPU for assemblies and the
// distances it runs on the GPU.
//
//   sketch LIST OUT THREADS [K,K...]    one sketch per listed FASTX path, with the given k-mer
//                                       lengths, by default 15, 19, 23, 27 and 31 as its CLI
//                                       uses, and 2048 bins per length; writes OUT.h5, the names
//                                       to OUT.names, the k-mer lengths to OUT.kmers, and the
//                                       random match chances its database construction adds
//   compare REFS QUERIES|- OUT THREADS  core and accessory distances on GPU 0, the fit across
//                                       the k-mer lengths that its GPU path computes (at least
//                                       two); writes one little-endian f32 per pair to OUT, the
//                                       Jaccard index at k = 25 that the fit predicts,
//                                       (1 - a)(1 - c)^25, and the core distance c to OUT.core:
//                                       row-major query x reference, or the strict upper triangle
//                                       of REFS for '-'
//   knn REFS K OUT THREADS              each sketch's K nearest neighbours among REFS by core
//                                       distance, through the sparse search on GPU 0; writes
//                                       one `row<TAB>neighbour<TAB>core distance` line each
//
// REFS and QUERIES are prefixes written by `sketch`. With CUDDL_RESIDENT_TIMINGS set, every
// command writes its in-memory time: hashing sequences already read, or computing the
// distances of loaded sketches up to the host matrix or neighbour lists the library returns.

#include "api.hpp"
#include "database/database.hpp"
#include "reference.hpp"
#include "sketch/seqio.hpp"
#include "sketch/sketch.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<size_t> const default_kmer_lengths{15, 19, 23, 27, 31};
constexpr size_t sketch_bins = 2048;
/// FASTX bytes on disk read per sketching batch, so reading stays outside the hashing time.
constexpr uintmax_t batch_bytes = uintmax_t{1} << 30;

std::vector<std::string> read_lines(std::filesystem::path const& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

/// Strictly ascending k-mer lengths, at least two: pp-sketchlib's GPU distances fit a line
/// across them.
std::vector<size_t> parse_kmers(std::string const& text) {
    std::vector<size_t> kmers;
    for (size_t start = 0; start <= text.size();) {
        auto const end = std::min(text.find(',', start), text.size());
        kmers.push_back(std::stoul(text.substr(start, end - start)));
        start = end + 1;
    }
    if (kmers.size() < 2 || !std::is_sorted(kmers.begin(), kmers.end(), std::less_equal<>{})) {
        throw std::invalid_argument(
            "k-mer lengths must be two or more, strictly ascending: " + text
        );
    }
    return kmers;
}

std::vector<size_t> read_kmers(std::string const& prefix) {
    auto const lines = read_lines(prefix + ".kmers");
    if (lines.size() != 1) throw std::runtime_error("expected one line in " + prefix + ".kmers");
    return parse_kmers(lines[0]);
}

void write_resident(double milliseconds, char const* device, char const* input) {
    char const* path = std::getenv("CUDDL_RESIDENT_TIMINGS");
    if (path == nullptr) return;
    std::ofstream out(path);
    out << "{\"resident_ms\":" << milliseconds << R"(,"source":"steady_clock_cpu_wall","device":")"
        << device << R"(","input":")" << input << "\"}";
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

void sketch(
    std::filesystem::path const& list,
    std::string const& out,
    int threads,
    std::vector<size_t> const& kmer_lengths
) {
    auto const files = read_lines(list);
    std::vector<std::string> names(files.size());
    for (size_t i = 0; i < files.size(); ++i) names[i] = std::to_string(i);
    auto const seeds = generate_seeds(kmer_lengths, false);
    std::vector<Reference> sketches(files.size());
    double hashing_ms = 0;
    for (size_t first = 0; first < files.size();) {
        size_t last = first;
        for (uintmax_t bytes = 0; last < files.size() && (last == first || bytes < batch_bytes);
             ++last) {
            bytes += std::filesystem::file_size(files[last]);
        }
        std::vector<SeqBuf> sequences(last - first);

        _Pragma("omp parallel for schedule(dynamic, 1) num_threads(threads)")
        for (size_t i = first; i < last; ++i) {
            sequences[i - first] = SeqBuf({files[i]}, kmer_lengths.back());
        }
        auto const start = std::chrono::steady_clock::now();

        _Pragma("omp parallel for schedule(dynamic, 5) num_threads(threads)")
        for (size_t i = first; i < last; ++i) {
            sketches[i] = Reference(
                names[i], sequences[i - first], seeds, sketch_bins / 64, false, true, 0, false
            );
        }
        hashing_ms += elapsed_ms(start);
        first = last;
    }
    auto database = new_db(out + ".h5", true, false);
    for (auto const& reference : sketches) database.add_sketch(reference);
    if (sketches.size() >= default_n_clusters) {
        calculate_random(sketches, out, default_n_clusters, default_n_MC, false, true, threads);
    }
    std::ofstream(out + ".names") << [&] {
        std::string text;
        for (auto const& name : names) text += name + "\n";
        return text;
    }();
    std::ofstream(out + ".kmers") << [&] {
        std::string text;
        for (auto const k : kmer_lengths) text += (text.empty() ? "" : ",") + std::to_string(k);
        return text + "\n";
    }();
    write_resident(hashing_ms, "cpu", "fastx sequences read into memory");
}

void compare(
    std::string const& references,
    std::string const& queries,
    std::filesystem::path const& out,
    int threads
) {
    auto const kmer_lengths = read_kmers(references);
    auto ref_sketches =
        load_sketches(references, read_lines(references + ".names"), kmer_lengths, false);
    auto const self = queries == "-";
    if (!self && read_kmers(queries) != kmer_lengths) {
        throw std::runtime_error(
            queries + " was sketched at other k-mer lengths than " + references
        );
    }
    auto query_sketches =
        self ? std::vector<Reference>{}
             : load_sketches(queries, read_lines(queries + ".names"), kmer_lengths, false);
    auto random = get_random(references, true);
    auto const start = std::chrono::steady_clock::now();
    auto const dists = query_db_cuda(
        ref_sketches, self ? ref_sketches : query_sketches, kmer_lengths, random, 0, threads
    );
    write_resident(elapsed_ms(start), "gpu", "loaded sketches to host distance matrix");
    std::vector<float> jaccard(static_cast<size_t>(dists.rows()));
    std::vector<float> core(jaccard.size());
    for (size_t i = 0; i < jaccard.size(); ++i) {
        auto const row = static_cast<Eigen::Index>(i);
        core[i] = dists(row, 0);
        jaccard[i] = (1.0f - dists(row, 1)) * std::pow(1.0f - core[i], 25.0f);
    }
    for (auto const& [path, values] :
         {std::pair{out, &jaccard},
          std::pair{std::filesystem::path{out.string() + ".core"}, &core}}) {
        std::ofstream file(path, std::ios::binary);
        file.write(
            reinterpret_cast<char const*>(values->data()),
            static_cast<std::streamsize>(values->size() * sizeof(float))
        );
        if (!file) throw std::runtime_error("cannot write " + path.string());
    }
}

void knn(std::string const& references, int k, std::filesystem::path const& out, int threads) {
    auto const kmer_lengths = read_kmers(references);
    auto ref_sketches =
        load_sketches(references, read_lines(references + ".names"), kmer_lengths, false);
    auto random = get_random(references, true);
    auto const start = std::chrono::steady_clock::now();
    auto const [rows, columns, distances] =
        query_db_sparse_cuda(ref_sketches, kmer_lengths, random, k, 0, 0, threads);
    write_resident(elapsed_ms(start), "gpu", "loaded sketches to host neighbour lists");
    std::ofstream file(out);
    for (size_t i = 0; i < rows.size(); ++i) {
        file << rows[i] << '\t' << columns[i] << '\t' << distances[i] << '\n';
    }
    if (!file) throw std::runtime_error("cannot write " + out.string());
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> const args(argv + 1, argv + argc);
    try {
        if ((args.size() == 4 || args.size() == 5) && args[0] == "sketch") {
            sketch(
                args[1],
                args[2],
                std::stoi(args[3]),
                args.size() == 5 ? parse_kmers(args[4]) : default_kmer_lengths
            );
        } else if (args.size() == 5 && args[0] == "compare") {
            compare(args[1], args[2], args[3], std::stoi(args[4]));
        } else if (args.size() == 5 && args[0] == "knn") {
            knn(args[1], std::stoi(args[2]), args[3], std::stoi(args[4]));
        } else {
            std::cerr << "usage: " << argv[0]
                      << " sketch LIST OUT THREADS [K,K...] | compare REFS QUERIES|- OUT THREADS"
                         " | knn REFS K OUT THREADS\n";
            return 2;
        }
    } catch (std::exception const& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
