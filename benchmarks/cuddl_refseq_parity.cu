#include <zlib.h>
#include <CLI/CLI.hpp>
#include <cuddl/a48.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/refseq_parity.hpp>
#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono;
using ddl_t = cuddl::reference_database<25, 4096>;
using json = nlohmann::json;

/// @brief The official BBTools RefSeq DDL sketch asset pinned for parity validation.
///
/// This is the v40.00 GitHub release asset (unmerged RefSeq records, one per sequence). The
/// release URL and on-disk byte size are pinned; the SHA-256 is computed over that exact asset.
/// The header parameters, record count, and merged/unmerged policy are checked against the
/// decoded asset itself before any result is accepted.
struct asset_prerequisite {
    std::string release = "BBTools v40.00";
    std::string resource = "refseqSketchDDL_k25e5b4096.tsv.gz";
    std::string url =
        "https://github.com/bbushnell/BBTools/releases/download/v40.00/"
        "refseqSketchDDL_k25e5b4096.tsv.gz";
    uint64_t size_bytes = 1270805218;
    std::string sha256 = "4f7181c94c32e2e778ea1f4ec4edb8ba485a62549cc9460d842a5a7acaa8acc5";
    uint32_t k = 25;
    uint32_t buckets = 4096;
    uint32_t exponent = 5;
    uint32_t records = 148108;
    bool merged = false;
};

/// @brief One query selected by the parity manifest.
struct parity_query {
    uint32_t ordinal{};
    bool external{};
    std::vector<uint16_t> row;
    std::string name;
};

/// @brief Per-query parity report assembled from all three comparison surfaces.
struct query_report {
    parity_query query;
    uint64_t row_checksum{};
    uint64_t row_nonzero{};

    std::vector<uint32_t> oracle_candidates;
    std::vector<cuddl::a48::row_summary> oracle_summaries;
    std::vector<uint32_t> oracle_counts;  // nonzero entries as {ref, count} pairs flattened
    double exact_refinement_ms{};

    bool gpu_ran{};
    uint32_t cuddl_candidate_count{};
    bool summaries_match_exhaustive{true};

    bool bbtools_ran{};
    std::vector<uint32_t> bbtools_candidates;
    std::vector<cuddl::a48::row_summary> bbtools_summaries;
    std::vector<uint32_t> bbtools_counts;  // flattened {ref, count} pairs for nonzero counts
    uint64_t bbtools_row_checksum{};
    uint64_t bbtools_row_nonzero{};
    std::vector<double> bbtools_query_runs;
    double bbtools_query_seconds{};
    std::vector<double> bbtools_query_csr_runs;
    double bbtools_query_csr_seconds{};

    bool counts_match_bbtools{};
    bool candidates_match_bbtools{};
    bool summaries_match_bbtools{};
    bool decode_matches_bbtools{};
    // Initialised true and only falsified on a measured-iteration mismatch, so skipped surfaces
    // (no GPU, --skip-bbtools) stay neutral rather than failing.
    bool candidates_match_oracle{true};
    bool summaries_match_oracle{true};
};

/// @brief Median of a non-empty sample (sorts a copy).
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto const size = values.size();
    return (values[size / 2U] + values[(size - 1U) / 2U]) / 2.0;
}

double now_ms() {
    return static_cast<double>(
               duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count()
           ) /
           1000.0;
}

struct bgzf_block {
    uint64_t offset{};
    uint32_t size{};
};

/// @brief Discovers BGZF member boundaries. Returns empty when the file is not BGZF.
std::vector<bgzf_block> find_bgzf_blocks(std::string const& path, uint64_t file_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    std::vector<bgzf_block> blocks;
    uint64_t offset = 0;
    while (offset < file_size) {
        file.seekg(static_cast<std::streamoff>(offset));
        std::array<char, 12> header{};
        file.read(header.data(), header.size());
        if (!file || static_cast<uint8_t>(header[0]) != 0x1fU ||
            static_cast<uint8_t>(header[1]) != 0x8bU) {
            return {};
        }
        auto const flags = static_cast<uint8_t>(header[3]);
        if ((flags & 0x04U) == 0U) {
            return {};
        }
        auto const extra_size = static_cast<uint16_t>(
            static_cast<uint8_t>(header[10]) | (static_cast<uint8_t>(header[11]) << 8U)
        );
        std::string extra(extra_size, '\0');
        file.read(extra.data(), extra.size());
        if (!file) {
            return {};
        }
        bool found = false;
        uint32_t block_size = 0;
        size_t cursor = 0;
        while (cursor + 4U <= extra.size()) {
            auto const sub_size = static_cast<uint16_t>(
                static_cast<uint8_t>(extra[cursor + 2U]) |
                (static_cast<uint8_t>(extra[cursor + 3U]) << 8U)
            );
            if (cursor + 4U + sub_size > extra.size()) {
                return {};
            }
            if (extra[cursor] == 'B' && extra[cursor + 1U] == 'C' && sub_size >= 2U) {
                block_size = static_cast<uint16_t>(
                    static_cast<uint8_t>(extra[cursor + 4U]) |
                    (static_cast<uint8_t>(extra[cursor + 5U]) << 8U)
                );
                found = true;
                break;
            }
            cursor += 4U + sub_size;
        }
        if (!found || block_size == 0U) {
            return {};
        }
        auto const total = static_cast<uint64_t>(block_size) + 1U;
        if (total > file_size - offset) {
            return {};
        }
        blocks.push_back(bgzf_block{offset, static_cast<uint32_t>(total)});
        offset += total;
    }
    return offset == file_size ? blocks : std::vector<bgzf_block>{};
}

/// @brief Decompresses one BGZF member (a complete gzip member).
std::string inflate_bgzf_block(std::ifstream& file, bgzf_block const& block) {
    std::string compressed(block.size, '\0');
    file.seekg(static_cast<std::streamoff>(block.offset));
    file.read(compressed.data(), compressed.size());
    if (!file) {
        throw std::runtime_error("cannot read compressed BGZF block");
    }

    z_stream stream{};
    if (inflateInit2(&stream, 15 + 16) != Z_OK) {
        throw std::runtime_error("cannot initialise gzip inflater");
    }
    stream.next_in = reinterpret_cast<Bytef*>(compressed.data());
    stream.avail_in = compressed.size();

    std::string output;
    std::vector<char> buffer(1U << 20);
    while (true) {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = buffer.size();
        auto const status = inflate(&stream, Z_NO_FLUSH);
        auto const produced = buffer.size() - stream.avail_out;
        if (produced != 0U) {
            output.append(buffer.data(), produced);
        }
        if (status == Z_STREAM_END) {
            break;
        }
        if (status != Z_OK || (stream.avail_in == 0U && produced == 0U)) {
            inflateEnd(&stream);
            throw std::runtime_error("BGZF block inflate failed");
        }
    }
    inflateEnd(&stream);
    return output;
}

/// @brief Reads an entire file (plain, ordinary gzip, or parallel BGZF) into a byte buffer.
std::string read_file_any(std::string const& path) {
    auto const compressed_size = std::filesystem::file_size(path);
    auto const blocks = find_bgzf_blocks(path, compressed_size);
    if (!blocks.empty()) {
        auto const requested = std::thread::hardware_concurrency();
        auto const worker_count = std::min<size_t>(
            requested == 0U ? 1U : requested, std::max<size_t>(1U, blocks.size())
        );
        std::vector<std::string> outputs(worker_count);
        std::vector<std::exception_ptr> errors(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            auto const begin = blocks.size() * worker / worker_count;
            auto const end = blocks.size() * (worker + 1U) / worker_count;
            if (begin == end) {
                continue;
            }
            workers.emplace_back([&, worker, begin, end]() {
                try {
                    std::ifstream file(path, std::ios::binary);
                    for (size_t i = begin; i < end; ++i) {
                        outputs[worker].append(inflate_bgzf_block(file, blocks[i]));
                    }
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        for (auto const& error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        size_t total_size = 0;
        for (auto const& output : outputs) {
            total_size += output.size();
        }
        std::string contents;
        contents.reserve(total_size);
        for (auto& output : outputs) {
            contents.append(output);
            output.clear();
            output.shrink_to_fit();
        }
        return contents;
    }

    gzFile handle = gzopen(path.c_str(), "rb");
    if (handle == nullptr) {
        throw std::runtime_error("cannot open asset (plain or .gz expected): " + path);
    }
    gzbuffer(handle, 1U << 20);
    std::string contents;
    contents.reserve(static_cast<size_t>(compressed_size) * 2U + (1U << 20));
    std::vector<char> buffer(1U << 20);
    while (true) {
        auto const read = gzread(handle, buffer.data(), static_cast<unsigned>(buffer.size()));
        if (read <= 0) {
            break;
        }
        contents.append(buffer.data(), static_cast<size_t>(read));
    }
    gzclose(handle);
    return contents;
}

/// @brief Quotes a shell argument with single quotes.
std::string shell_quote(std::string const& value) {
    std::string quoted;
    quoted.push_back('\'');
    for (char const character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(character);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

/// @brief Runs a shell command, capturing merged output and returning the exit status.
int run_command(std::string const& command, std::string& output) {
    auto const full = command + " 2>&1";
    FILE* pipe = popen(full.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("cannot invoke: " + command);
    }
    output.clear();
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    return pclose(pipe);
}

/// @brief Computes the lowercase hex SHA-256 of the file at @p path via the platform utility.
std::string sha256_of_file(std::string const& path) {
    std::string output;
    auto const status = run_command(std::string("sha256sum ") + shell_quote(path), output);
    if (status != 0) {
        throw std::runtime_error("`sha256sum` failed for " + path + ": " + output);
    }
    std::string hex;
    for (char const character : output) {
        if (hex.size() == 64U) {
            break;
        }
        if (std::isxdigit(static_cast<unsigned char>(character))) {
            hex.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }
    }
    if (hex.size() != 64U) {
        throw std::runtime_error("unexpected `sha256sum` output: " + output);
    }
    return hex;
}

/// @brief FNV-1a 64-bit fingerprint of one decoded row; must equal the Java harness'
/// `rowChecksum` over BBTools' absolute maxArray for the same record.
struct row_fingerprint {
    uint64_t checksum{};
    uint64_t nonzero{};
};

row_fingerprint fingerprint_row(std::vector<uint16_t> const& row) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    uint64_t nonzero = 0;
    for (auto const score : row) {
        hash ^= score;
        hash *= 0x100000001b3ULL;
        if (score != 0U) {
            ++nonzero;
        }
    }
    return {hash, nonzero};
}

/// @brief True when the CUDA runtime can enumerate at least one device.
bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return false;
    }
    return true;
}

/// @brief Prints the exact external prerequisite when the official asset is not available.
void report_prerequisite(asset_prerequisite const& prereq, bool asset_present) {
    std::fprintf(
        stderr,
        "\nBBTools RefSeq DDL parity requires the official precomputed asset.\n"
        "  Resource:      %s\n"
        "  Release:       %s\n"
        "  URL:           %s\n"
        "  Byte size:     %s\n"
        "  SHA-256:       %s\n"
        "  Header:        k=%u, buckets=%u, exponent=%u\n"
        "  Merged/unmerged: %s\n",
        prereq.resource.c_str(),
        prereq.release.empty() ? "(unspecified)" : prereq.release.c_str(),
        prereq.url.empty() ? "(provide --asset-url)" : prereq.url.c_str(),
        prereq.size_bytes == 0 ? "(provide --size-bytes)"
                               : std::to_string(prereq.size_bytes).c_str(),
        prereq.sha256.empty() ? "(provide --sha256)" : prereq.sha256.c_str(),
        prereq.k,
        prereq.buckets,
        prereq.exponent,
        prereq.merged ? "merged (taxonomy-merged)" : "unmerged"
    );
    std::fputs(
        !asset_present
            ? "  Status:        ASSET_NOT_AVAILABLE -- run where the asset is reachable, or\n"
              "                 set --asset to a locally cached copy of the above resource.\n"
            : "  Status:        asset present but provenance/header checks did not pass.\n",
        stderr
    );
}

/// @brief Parses a query-selection manifest.
///
/// Every line selects one query ordinal. `external:` ordinals are removed from the database
/// before the index is built, so they arrive as out-of-database queries; `resident:` ordinals
/// stay in the database. The legacy `holdout:` / `query:` spelling is accepted as an alias.
struct manifest_selection {
    std::set<uint32_t> external;
    std::vector<uint32_t> queries;
};

manifest_selection parse_manifest(std::string const& path, size_t record_count) {
    std::ifstream manifest(path);
    if (!manifest) {
        throw std::runtime_error("cannot open manifest: " + path);
    }
    manifest_selection selection;
    std::string line;
    size_t line_no = 0;
    while (std::getline(manifest, line)) {
        ++line_no;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        auto const split = line.find(':');
        if (split == std::string::npos) {
            throw std::runtime_error(
                "unrecognized manifest line at line " + std::to_string(line_no)
            );
        }
        auto const key = line.substr(0, split);
        uint32_t ordinal = 0;
        try {
            ordinal = static_cast<uint32_t>(std::stoul(line.substr(split + 1)));
        } catch (std::exception const&) {
            throw std::runtime_error(
                "non-numeric manifest ordinal at line " + std::to_string(line_no)
            );
        }
        if (ordinal >= record_count) {
            throw std::runtime_error(
                "manifest ordinal out of range at line " + std::to_string(line_no)
            );
        }
        if (key == "external") {
            selection.external.insert(ordinal);
            selection.queries.push_back(ordinal);
        } else if (key == "holdout") {
            selection.external.insert(ordinal);
        } else if (key == "resident" || key == "query") {
            selection.queries.push_back(ordinal);
        } else {
            throw std::runtime_error(
                "unrecognized manifest key at line " + std::to_string(line_no) + ": " + key
            );
        }
    }
    return selection;
}

/// @brief The pinned default parity selection, generated so no manifest file is required.
///
/// Ordinals are zero-based positions in the pinned v40.00 unmerged RefSeq asset's 148,108
/// records. The selection alternates equal numbers of external and resident queries evenly
/// across the file, so both query classes stress every source partition.
manifest_selection pinned_manifest_selection(size_t record_count, uint32_t queries_per_class) {
    if (queries_per_class == 0U) {
        throw std::runtime_error("--queries-per-class must be positive");
    }
    auto const total_queries = static_cast<uint64_t>(queries_per_class) * 2U;
    if (total_queries > record_count) {
        throw std::runtime_error("--queries-per-class exceeds the decoded record count");
    }

    manifest_selection selection;
    for (uint64_t i = 0; i < total_queries; ++i) {
        auto const ordinal = static_cast<uint32_t>(
            total_queries == 1U
                ? 0U
                : (i * (static_cast<uint64_t>(record_count) - 1U)) / (total_queries - 1U)
        );
        if (i % 2U == 0U) {
            selection.external.insert(ordinal);
        }
        selection.queries.push_back(ordinal);
    }
    return selection;
}

/// @brief Compiles and runs the BBTools DDLIndex/CSR2 Java harness, returning its JSON output.
json run_bbtools_harness(
    std::string const& asset_path,
    std::string const& selections_path,
    uint32_t min_hits,
    uint32_t threads,
    uint32_t warmup,
    uint32_t runs,
    std::string const& java_heap,
    std::string const& bbtools_dir,
    std::string const& harness_source,
    std::string const& workdir
) {
    auto const jar = bbtools_dir + "/bbtools.jar";
    if (!std::filesystem::exists(jar)) {
        throw std::runtime_error("BBTools jar not found: " + jar);
    }
    if (!std::filesystem::exists(harness_source)) {
        throw std::runtime_error("BBTools parity harness source not found: " + harness_source);
    }
    std::filesystem::create_directories(workdir);
    auto const classes = workdir + "/classes";
    std::filesystem::create_directories(classes);

    std::string output;
    auto const javac_command = std::string("javac -cp ") + shell_quote(jar) + " -d " +
                               shell_quote(classes) + " " + shell_quote(harness_source);
    auto const javac_status = run_command(javac_command, output);
    if (javac_status != 0) {
        throw std::runtime_error("javac failed for BBTools parity harness:\n" + output);
    }

    auto const out_json = workdir + "/bbtools-oracle.json";
    auto const java_command =
        std::string("java -Xmx") + java_heap + " -cp " + shell_quote(classes + ":" + jar) +
        " BBToolsRefSeqParity " + shell_quote(asset_path) + " " + shell_quote(selections_path) +
        " " + std::to_string(min_hits) + " " + std::to_string(threads) + " " +
        std::to_string(warmup) + " " + std::to_string(runs) + " " + shell_quote(out_json);
    auto const java_status = run_command(java_command, output);
    if (java_status != 0) {
        throw std::runtime_error("BBTools parity harness failed:\n" + output);
    }

    std::ifstream result_file(out_json);
    if (!result_file) {
        throw std::runtime_error("BBTools parity harness produced no output at " + out_json);
    }
    json result;
    result_file >> result;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{
            "BBTools RefSeq DDL parity validation for cuDDL (decoded-row index mechanics)."
        };

        std::string asset_path;
        asset_prerequisite prereq;
        std::string manifest_path;
        uint32_t queries_per_class = 64;
        uint32_t min_hits = 5;
        uint64_t seed = 42;
        std::string evidence_path;
        std::string bbtools_dir = "subprojects/bbmap";
        std::string harness_source = "benchmarks/BBToolsRefSeqParity.java";
        std::string java_heap = "24g";
        uint32_t bbtools_threads = 8;
        std::string workdir = "results/refseq-parity";
        uint32_t warmup = 3;
        uint32_t runs = 10;
        bool skip_verify = false;
        bool skip_bbtools = false;
        bool no_gpu = false;

        app.add_option(
            "--asset", asset_path, "Path to the official RefSeq DDL A48 .tsv(.gz) asset"
        );
        app.add_option("--asset-url", prereq.url, "Official release URL of the asset");
        app.add_option("--release", prereq.release, "Official release tag");
        app.add_option(
            "--size-bytes", prereq.size_bytes, "Expected on-disk (compressed) asset byte size"
        );
        app.add_option(
            "--sha256", prereq.sha256, "Expected asset SHA-256 checksum (hex), verified"
        );
        // The parity command is pinned to one 25/4096/e5 asset family (unmerged and merged
        // variants of refseqSketchDDL_k25e5b4096), so the construction parameters are
        // provenance constants, not options: the GPU search type is `reference_database<25,
        // 4096>`, and accepting other k/bucket/exponent values would only fail later.
        app.add_option(
            "--expect-records",
            prereq.records,
            "Expected decoded record count (content-based merged/unmerged policy check)"
        );
        app.add_flag("--expect-merged", prereq.merged, "Asset must be the taxonomy-merged variant");
        app.add_option(
            "--manifest",
            manifest_path,
            "Optional query-selection manifest override (`external: N` / `resident: N` ordinal "
            "lines). Defaults to the built-in pinned selection."
        );
        app.add_option(
            "--queries-per-class",
            queries_per_class,
            "External and resident queries selected by the built-in default (ignored with "
            "--manifest)"
        )->check(CLI::PositiveNumber);
        app.add_option("--min-hits", min_hits, "Minimum matching buckets to retain a candidate")
            ->check(CLI::NonNegativeNumber);
        app.add_option("--seed", seed, "Hash seed fallback when the asset has no #seed header");
        app.add_option("--evidence", evidence_path, "Output JSON evidence path (default: stdout)");
        app.add_option(
            "--bbtools-dir", bbtools_dir, "Vendored bbmap directory holding bbtools.jar"
        );
        app.add_option("--bbtools-harness", harness_source, "Path to BBToolsRefSeqParity.java");
        app.add_option("--java-heap", java_heap, "BBTools Java heap (--Xmx value)");
        app.add_option("--bbtools-threads", bbtools_threads, "BBTools loader/index thread count")
            ->check(CLI::PositiveNumber);
        app.add_option("--workdir", workdir, "Scratch directory for the Java harness build");
        app.add_option("--warmup", warmup, "Timed warm-up iterations (discarded)")
            ->check(CLI::NonNegativeNumber);
        app.add_option("--runs", runs, "Measured iterations; medians are reported")
            ->check(CLI::PositiveNumber);
        app.add_flag(
            "--skip-verify",
            skip_verify,
            "Skip provenance/header verification (pipeline tests only)"
        );
        app.add_flag("--skip-bbtools", skip_bbtools, "Skip the BBTools DDLIndex/CSR2 comparison");
        app.add_flag("--no-gpu", no_gpu, "Skip the cuDDL GPU build and search phases");

        CLI11_PARSE(app, argc, argv);

        // A default location under ./data is tried before reporting the prerequisite, so a plain
        // run works after the official asset is downloaded there.
        auto const default_asset = std::string("data/refseqSketchDDL_k25e5b4096.tsv.gz");
        if (asset_path.empty() && std::filesystem::exists(default_asset)) {
            asset_path = default_asset;
        }
        if (!skip_verify && asset_path.find("_merged") != std::string::npos) {
            prereq.merged = true;  // default policy from filename unless --expect-merged overrides
        }

        FILE* evidence_file = nullptr;
        auto open_evidence = [&]() {
            if (!evidence_path.empty()) {
                evidence_file = std::fopen(evidence_path.c_str(), "w");
                if (evidence_file == nullptr) {
                    throw std::runtime_error("cannot open evidence output: " + evidence_path);
                }
            }
        };

        if (asset_path.empty()) {
            open_evidence();
            report_prerequisite(prereq, false);
            json evidence;
            evidence["ok"] = false;
            evidence["reason"] = "ASSET_NOT_AVAILABLE";
            evidence["prerequisite"] = {
                {"resource", prereq.resource},
                {"release", prereq.release},
                {"url", prereq.url},
                {"size_bytes", prereq.size_bytes},
                {"sha256", prereq.sha256},
                {"k", prereq.k},
                {"buckets", prereq.buckets},
                {"exponent", prereq.exponent},
                {"merged", prereq.merged},
            };
            auto const text = evidence.dump(2);
            FILE* out = evidence_file != nullptr ? evidence_file : stdout;
            std::fwrite(text.data(), 1, text.size(), out);
            std::fputc('\n', out);
            if (evidence_file != nullptr) {
                std::fclose(evidence_file);
            }
            return 0;
        }

        // Load the asset and verify on-disk size + checksum before any results are accepted
        auto const t_load_start = now_ms();
        std::string opaque = read_file_any(asset_path);
        auto const t_load_end = now_ms();

        if (!skip_verify) {
            auto const on_disk_size = std::filesystem::file_size(asset_path);
            if (prereq.size_bytes != 0 && on_disk_size != prereq.size_bytes) {
                throw std::runtime_error(
                    "asset byte size mismatch: expected " + std::to_string(prereq.size_bytes) +
                    ", got " + std::to_string(on_disk_size)
                );
            }
            if (!prereq.sha256.empty()) {
                auto const computed = sha256_of_file(asset_path);
                if (computed != prereq.sha256) {
                    throw std::runtime_error(
                        "asset SHA-256 mismatch: expected " + prereq.sha256 + ", got " + computed
                    );
                }
            }
        }

        // Decode A48
        auto const t_parse_start = now_ms();
        auto decoded = cuddl::a48::decode_a48_tsv_parallel(
            opaque, std::thread::hardware_concurrency()
        );
        auto const t_parse_end = now_ms();
        if (!decoded) {
            throw std::runtime_error("A48 decode failed: " + decoded.error().message());
        }
        auto const& db = *decoded;
        if (db.records.empty()) {
            throw std::runtime_error("A48 asset contains no records");
        }

        if (!skip_verify) {
            if (!db.metadata.has_kmer_length || db.metadata.kmer_length != prereq.k) {
                throw std::runtime_error("A48 #k header missing or mismatched");
            }
            if (!db.metadata.has_exponent || db.metadata.exponent_bits != prereq.exponent) {
                throw std::runtime_error("A48 #exponent header missing or mismatched");
            }
            if (db.records[0].scores.size() != prereq.buckets) {
                throw std::runtime_error("A48 bucket count mismatch");
            }
            // Content-based merged/unmerged policy check: the pinned unmerged asset has exactly
            // the pinned record count, so a merged variant (or any other asset) cannot pass
            // silently with the unmerged provenance.
            if (prereq.records != 0U && db.records.size() != prereq.records) {
                throw std::runtime_error(
                    "decoded record count " + std::to_string(db.records.size()) +
                    " does not match the pinned asset record count " +
                    std::to_string(prereq.records)
                );
            }
            // Every record of the pinned asset carries its own construction origin; the header
            // omits the file extension, so compare against the resource's stem.
            auto expected_origin = prereq.resource;
            constexpr std::string_view compressed_suffix = ".tsv.gz";
            if (expected_origin.size() > compressed_suffix.size() &&
                expected_origin.ends_with(compressed_suffix)) {
                expected_origin.resize(expected_origin.size() - compressed_suffix.size());
            }
            if (!db.records.front().metadata.origin.empty() &&
                db.records.front().metadata.origin != expected_origin) {
                throw std::runtime_error(
                    "asset #origin is '" + db.records.front().metadata.origin + "', expected '" +
                    expected_origin + "'"
                );
            }
        }
        auto const compatibility_seed = db.metadata.has_seed ? db.metadata.seed : seed;
        auto const compatibility = cuddl::decoded_compatibility(
            prereq.k, prereq.buckets, prereq.exponent, compatibility_seed
        );

        // Require both external and resident queries. External queries exercise the
        // reduced-database id mapping; resident queries exercise ordinary in-database lookup.
        // The pinned default selection is generated in the binary; `--manifest` is an override.
        auto const selection = manifest_path.empty()
                                   ? pinned_manifest_selection(db.records.size(), queries_per_class)
                                   : parse_manifest(manifest_path, db.records.size());
        auto const has_external_query = std::any_of(
            selection.queries.begin(), selection.queries.end(), [&](uint32_t ordinal) {
                return selection.external.contains(ordinal);
            }
        );
        auto const has_database_query = std::any_of(
            selection.queries.begin(), selection.queries.end(), [&](uint32_t ordinal) {
                return !selection.external.contains(ordinal);
            }
        );
        if (selection.external.empty() || !has_external_query || !has_database_query) {
            throw std::runtime_error(
                "manifest must select external and resident queries"
            );
        }

        // Build the database subset (every record except the external query rows)
        std::vector<uint32_t> db_ordinals;
        std::vector<uint32_t> ordinal_to_db_id(db.records.size(), UINT32_MAX);
        for (uint32_t ordinal = 0; ordinal < db.records.size(); ++ordinal) {
            if (selection.external.count(ordinal) == 0U) {
                ordinal_to_db_id[ordinal] = static_cast<uint32_t>(db_ordinals.size());
                db_ordinals.push_back(ordinal);
            }
        }
        std::vector<uint16_t> flat_scores(db_ordinals.size() * prereq.buckets);
        for (size_t i = 0; i < db_ordinals.size(); ++i) {
            auto const& row = db.records[db_ordinals[i]].scores;
            std::copy(row.begin(), row.end(), flat_scores.begin() + i * prereq.buckets);
        }

        std::vector<query_report> reports;
        for (auto const ordinal : selection.queries) {
            query_report report;
            report.query.ordinal = ordinal;
            report.query.external = selection.external.count(ordinal) != 0U;
            report.query.row = db.records[ordinal].scores;
            report.query.name = db.records[ordinal].metadata.name;
            reports.push_back(std::move(report));
        }

        // Independent host oracle: index match counts + exhaustive exact summaries
        double oracle_total_ms = 0;
        for (auto& report : reports) {
            auto const t_start = now_ms();
            auto oracle = cuddl::a48::exhaustive_oracle(
                flat_scores, report.query.row, prereq.buckets, min_hits
            );
            report.exact_refinement_ms = now_ms() - t_start;
            oracle_total_ms += report.exact_refinement_ms;

            auto const fingerprint = fingerprint_row(report.query.row);
            report.row_checksum = fingerprint.checksum;
            report.row_nonzero = fingerprint.nonzero;
            report.oracle_summaries = std::move(oracle.summaries);
            for (uint32_t id = 0; id < oracle.match_counts.size(); ++id) {
                if (oracle.match_counts[id] >= min_hits) {
                    report.oracle_candidates.push_back(id);
                }
                if (oracle.match_counts[id] >= 1U) {
                    report.oracle_counts.push_back(id);
                    report.oracle_counts.push_back(oracle.match_counts[id]);
                }
            }
        }

        // cuDDL GPU build and indexed search (skipped without a CUDA device)
        bool gpu_available = !no_gpu && cuda_available();
        double host_to_device_ms = 0;
        std::vector<double> index_build_runs;
        double index_build_ms = 0;
        std::vector<double> in_database_query_runs;
        std::vector<double> external_query_runs;
        double in_database_query_ms = 0;
        double external_query_ms = 0;
        std::vector<double> exhaustive_batch_runs;
        double exhaustive_batch_ms = 0;
        size_t device_free_before = 0, device_free_low = 0, device_total = 0;
        size_t device_peak_bytes = 0;
        size_t device_rows_bytes = 0, device_index_bytes = 0;
        if (gpu_available) {
            CUDDL_CUDA_CALL(cudaMemGetInfo(&device_free_before, &device_total));
            device_free_low = device_free_before;

            // One-shot upload: the rows are identical across iterations, so the copy is not a
            // measured phase.
            auto const t_upload_start = now_ms();
            thrust::device_vector<uint16_t> device_scores(flat_scores);
            auto const t_upload_end = now_ms();
            host_to_device_ms = t_upload_end - t_upload_start;

            auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};

            // Batched buffers. The exhaustive pass still runs over all selected queries in
            // one row-major tile; the timed indexed search runs two equal-sized tiles so the
            // evidence can report external and resident query classes separately.
            std::vector<uint16_t> flat_queries;
            flat_queries.reserve(reports.size() * prereq.buckets);
            std::vector<uint16_t> flat_in_database_queries;
            std::vector<uint16_t> flat_external_queries;
            std::vector<uint32_t> in_database_query_to_report;
            std::vector<uint32_t> external_query_to_report;
            flat_in_database_queries.reserve(reports.size() * prereq.buckets);
            flat_external_queries.reserve(reports.size() * prereq.buckets);
            for (uint32_t qi = 0; qi < reports.size(); ++qi) {
                auto const& report = reports[qi];
                flat_queries.insert(
                    flat_queries.end(), report.query.row.begin(), report.query.row.end()
                );
                auto& flat_class = report.query.external ? flat_external_queries
                                                         : flat_in_database_queries;
                auto& class_to_report =
                    report.query.external ? external_query_to_report : in_database_query_to_report;
                flat_class.insert(flat_class.end(), report.query.row.begin(), report.query.row.end());
                class_to_report.push_back(qi);
            }
            thrust::device_vector<uint16_t> device_queries(flat_queries);
            thrust::device_vector<uint16_t> device_in_database_queries(flat_in_database_queries);
            thrust::device_vector<uint16_t> device_external_queries(flat_external_queries);
            thrust::device_vector<uint32_t> batch_count(1U);
            thrust::device_vector<uint32_t> batch_exhaustive_count(1U);

            thrust::device_vector<uint8_t> batch_workspace;
            using batch_result_t = ddl_t::ref_type::batch_result_type;
            thrust::device_vector<batch_result_t> batch_results;
            thrust::device_vector<batch_result_t> batch_exhaustive_results;
            std::vector<batch_result_t> batch_host;
            std::vector<batch_result_t> batch_exhaustive_host;

            for (uint32_t iteration = 0; iteration < warmup + runs; ++iteration) {
                bool const measured = iteration >= warmup;
                // A background sampler tracks the free-memory low-water across the whole
                // construction: the build's transient transpose and CUB scan scratch peak
                // before the build returns, so post-sync sampling would undercount it.
                std::atomic<bool> sampler_stop{false};
                size_t sample_free_low = std::numeric_limits<size_t>::max();
                std::thread sampler;
                if (measured) {
                    sampler = std::thread([&] {
                        for (;;) {
                            size_t free_now = 0;
                            size_t total_now = 0;
                            if (cudaMemGetInfo(&free_now, &total_now) != cudaSuccess) {
                                return;
                            }
                            sample_free_low = std::min(sample_free_low, free_now);
                            if (sampler_stop.load(std::memory_order_relaxed)) {
                                return;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    });
                }
                auto const t_index_start = now_ms();
                auto built = ddl_t::build_indexed_async(device_scores, compatibility, stream);
                if (!built) {
                    throw std::runtime_error("index build failed: " + built.error().message());
                }
                auto database = std::move(*built);
                CUDDL_CUDA_CALL(cudaStreamSynchronize(stream.get()));
                auto const t_index_end = now_ms();
                if (measured) {
                    index_build_runs.push_back(t_index_end - t_index_start);
                    sampler_stop.store(true, std::memory_order_relaxed);
                    sampler.join();
                    if (sample_free_low <= device_free_before) {
                        device_peak_bytes = std::max(
                            device_peak_bytes, device_free_before - sample_free_low
                        );
                    }
                }

                if (iteration == 0U) {
                    device_rows_bytes = database.persistent_row_bytes();
                    device_index_bytes = database.persistent_index_bytes();

                    auto const max_class_queries = static_cast<uint32_t>(std::max(
                        in_database_query_to_report.size(), external_query_to_report.size()
                    ));
                    auto const batch_requirements =
                        database.indexed_batch_search_requirements(max_class_queries);
                    auto const exhaustive_requirements = database.indexed_batch_search_requirements(
                        static_cast<uint32_t>(reports.size())
                    );
                    if (!batch_requirements || !exhaustive_requirements) {
                        throw std::runtime_error("batch workspace sizing failed");
                    }
                    batch_workspace =
                        thrust::device_vector<uint8_t>(batch_requirements->workspace_bytes);
                    batch_results = thrust::device_vector<batch_result_t>(
                        batch_requirements->maximum_pair_count
                    );
                    batch_exhaustive_results = thrust::device_vector<batch_result_t>(
                        exhaustive_requirements->maximum_pair_count
                    );
                }

                // Batched exhaustive pass: the exact-refinement bound for the whole tile.
                thrust::device_vector<uint8_t> batch_exhaustive_workspace;
                thrust::device_vector<uint32_t> batch_exhaustive_match_counts;
                auto const t_exhaustive_start = now_ms();
                auto const exhaustive_batch_res = database.search_batch_async(
                    device_queries,
                    compatibility,
                    0U,
                    batch_exhaustive_workspace,
                    batch_exhaustive_results,
                    batch_exhaustive_count,
                    batch_exhaustive_match_counts,
                    stream
                );
                CUDDL_CUDA_CALL(cudaStreamSynchronize(stream.get()));
                auto const t_exhaustive_end = now_ms();
                if (!exhaustive_batch_res) {
                    throw std::runtime_error(
                        "batched exhaustive search failed: " +
                        exhaustive_batch_res.error().message()
                    );
                }

                uint32_t exhaustive_pair_count = 0;
                CUDDL_CUDA_CALL(cudaMemcpy(
                    &exhaustive_pair_count,
                    thrust::raw_pointer_cast(batch_exhaustive_count.data()),
                    sizeof(exhaustive_pair_count),
                    cudaMemcpyDeviceToHost
                ));
                batch_exhaustive_host.resize(exhaustive_pair_count);
                if (exhaustive_pair_count != 0U) {
                    CUDDL_CUDA_CALL(cudaMemcpy(
                        batch_exhaustive_host.data(),
                        thrust::raw_pointer_cast(batch_exhaustive_results.data()),
                        exhaustive_pair_count * sizeof(batch_result_t),
                        cudaMemcpyDeviceToHost
                    ));
                }

                if (measured) {
                    exhaustive_batch_runs.push_back(t_exhaustive_end - t_exhaustive_start);
                }

                // Timed indexed search for one query class. The class tile is answered by one
                // stream-ordered call; local query IDs are mapped back to full report order.
                auto run_indexed_batch = [&](
                                             thrust::device_vector<uint16_t> const& class_queries,
                                             std::vector<uint32_t> const& class_to_report,
                                             std::vector<double>& runs
                                         ) {
                    thrust::device_vector<uint32_t> batch_match_counts;
                    auto const t_batch_start = now_ms();
                    auto const batch_res = database.search_batch_indexed_async(
                        class_queries,
                        compatibility,
                        0U,
                        batch_workspace,
                        batch_results,
                        batch_count,
                        batch_match_counts,
                        {.minimum_matches = min_hits},
                        stream
                    );
                    CUDDL_CUDA_CALL(cudaStreamSynchronize(stream.get()));
                    auto const t_batch_end = now_ms();
                    if (!batch_res) {
                        throw std::runtime_error(
                            "batched indexed search failed: " + batch_res.error().message()
                        );
                    }

                    uint32_t batch_pair_count = 0;
                    CUDDL_CUDA_CALL(cudaMemcpy(
                        &batch_pair_count,
                        thrust::raw_pointer_cast(batch_count.data()),
                        sizeof(batch_pair_count),
                        cudaMemcpyDeviceToHost
                    ));
                    batch_host.resize(batch_pair_count);
                    if (batch_pair_count != 0U) {
                        CUDDL_CUDA_CALL(cudaMemcpy(
                            batch_host.data(),
                            thrust::raw_pointer_cast(batch_results.data()),
                            batch_pair_count * sizeof(batch_result_t),
                            cudaMemcpyDeviceToHost
                        ));
                    }

                    if (!measured) {
                        return;
                    }
                    runs.push_back(t_batch_end - t_batch_start);
                    for (auto const& pair : batch_host) {
                        auto const report_index = class_to_report[pair.query_id];
                        auto& report = reports[report_index];
                        auto const reference_id = pair.reference_id;
                        auto const& oracle_summary = report.oracle_summaries[reference_id];
                        auto const counts_match =
                            pair.summary.counts.lower == oracle_summary.lower &&
                            pair.summary.counts.equal == oracle_summary.equal &&
                            pair.summary.counts.higher == oracle_summary.higher &&
                            pair.summary.counts.both_empty == oracle_summary.both_empty;
                        if (!counts_match) {
                            report.summaries_match_oracle = false;
                        }
                        auto const dense_index =
                            static_cast<size_t>(report_index) * db_ordinals.size() + reference_id;
                        if (dense_index >= batch_exhaustive_host.size() ||
                            pair.summary != batch_exhaustive_host[dense_index].summary) {
                            report.summaries_match_exhaustive = false;
                        }
                    }
                    // The batched pair set must equal the oracle candidate set per query:
                    // pairs arrive in ascending query-major order, so each query owns one
                    // contiguous run of the emitted list.
                    size_t batch_index = 0;
                    for (size_t local_query = 0; local_query < class_to_report.size(); ++local_query) {
                        auto const report_index = class_to_report[local_query];
                        auto& report = reports[report_index];
                        std::vector<uint32_t> batch_candidates;
                        while (batch_index < batch_host.size() &&
                               batch_host[batch_index].query_id == local_query) {
                            batch_candidates.push_back(batch_host[batch_index].reference_id);
                            ++batch_index;
                        }
                        report.gpu_ran = true;
                        report.cuddl_candidate_count =
                            static_cast<uint32_t>(batch_candidates.size());
                        if (batch_candidates != report.oracle_candidates) {
                            report.candidates_match_oracle = false;
                        }
                    }
                };

                run_indexed_batch(
                    device_in_database_queries, in_database_query_to_report, in_database_query_runs
                );
                run_indexed_batch(
                    device_external_queries, external_query_to_report, external_query_runs
                );
                // Post-sync free memory at this sample point (context only); the build
                // high-water mark for `peak_allocated_bytes` is sampled during the
                // construction by a background thread.
                size_t free_now = 0;
                CUDDL_CUDA_CALL(cudaMemGetInfo(&free_now, &device_total));
                device_free_low = std::min(device_free_low, free_now);
            }

            index_build_ms = median(index_build_runs);
            in_database_query_ms = median(in_database_query_runs);
            external_query_ms = median(external_query_runs);
            exhaustive_batch_ms = median(exhaustive_batch_runs);
        } else {
            std::fputs(
                "No CUDA device available (or --no-gpu): skipping the cuDDL GPU phases.\n", stderr
            );
        }

        // BBTools DDLIndex/CSR2 oracle via the first-party Java harness
        bool bbtools_ran = false;
        json bbtools;
        if (!skip_bbtools) {
            std::filesystem::create_directories(workdir);
            auto const selections_path = workdir + "/selections.txt";
            {
                std::ofstream selections_file(selections_path);
                for (auto const ordinal : selection.queries) {
                    selections_file << (selection.external.contains(ordinal) ? "external: " : "resident: ")
                                     << ordinal << "\n";
                }
            }
            bbtools = run_bbtools_harness(
                asset_path,
                selections_path,
                min_hits,
                bbtools_threads,
                warmup,
                runs,
                java_heap,
                bbtools_dir,
                harness_source,
                workdir
            );
            bbtools_ran = true;

            auto const bbtools_records = bbtools.value("records", size_t{0});
            auto const bbtools_db_records = bbtools.value("db_records", size_t{0});
            if (bbtools_records != db.records.size() || bbtools_db_records != db_ordinals.size()) {
                throw std::runtime_error(
                    "BBTools harness decoded " + std::to_string(bbtools_records) +
                    " records into a " + std::to_string(bbtools_db_records) +
                    "-record database; the A48 decoder produced " +
                    std::to_string(db.records.size()) + " records into a " +
                    std::to_string(db_ordinals.size()) +
                    "-record database -- decoders disagree on the asset structure"
                );
            }

            auto const& bbtools_queries = bbtools.at("queries");
            if (bbtools_queries.size() != reports.size()) {
                throw std::runtime_error("BBTools harness query count disagrees with the manifest");
            }
            auto const& query_runs_json = bbtools.at("timings_seconds").at("query_runs");
            auto const& query_csr_runs_json = bbtools.at("timings_seconds").at("query_csr_runs");
            if (query_runs_json.size() != reports.size() ||
                query_csr_runs_json.size() != reports.size()) {
                throw std::runtime_error(
                    "BBTools harness query-run count disagrees with the manifest"
                );
            }
            for (size_t i = 0; i < reports.size(); ++i) {
                auto& report = reports[i];
                auto const& bq = bbtools_queries.at(i);
                report.bbtools_ran = true;
                report.bbtools_row_checksum = bq.value("row_checksum", uint64_t{0});
                report.bbtools_row_nonzero = bq.value("row_nonzero", uint64_t{0});
                for (auto const& seconds : query_runs_json.at(i)) {
                    report.bbtools_query_runs.push_back(seconds.get<double>());
                }
                report.bbtools_query_seconds = median(report.bbtools_query_runs);
                for (auto const& seconds : query_csr_runs_json.at(i)) {
                    report.bbtools_query_csr_runs.push_back(seconds.get<double>());
                }
                report.bbtools_query_csr_seconds = median(report.bbtools_query_csr_runs);
                for (auto const& entry : bq.at("counts")) {
                    auto const id = entry.at("id").get<uint32_t>();
                    auto const count = entry.at("count").get<uint32_t>();
                    report.bbtools_counts.push_back(id);
                    report.bbtools_counts.push_back(count);
                }
                for (auto const& entry : bq.at("candidates")) {
                    report.bbtools_candidates.push_back(entry.get<uint32_t>());
                }
                for (auto const& entry : bq.at("summaries")) {
                    report.bbtools_summaries.push_back(
                        cuddl::a48::row_summary{
                            entry.at("lower").get<uint32_t>(),
                            entry.at("equal").get<uint32_t>(),
                            entry.at("higher").get<uint32_t>(),
                            entry.at("both_empty").get<uint32_t>(),
                        }
                    );
                }
            }
        } else {
            std::fputs("--skip-bbtools: skipping the BBTools DDLIndex/CSR2 comparison.\n", stderr);
        }

        // Compare every surface
        bool ok = !skip_verify;
        for (auto& report : reports) {
            report.decode_matches_bbtools = report.bbtools_ran &&
                                            report.row_checksum == report.bbtools_row_checksum &&
                                            report.row_nonzero == report.bbtools_row_nonzero;

            report.counts_match_bbtools =
                report.bbtools_ran && report.bbtools_counts == report.oracle_counts;

            report.candidates_match_bbtools =
                report.bbtools_ran && report.bbtools_candidates == report.oracle_candidates;

            report.summaries_match_bbtools = true;
            if (report.bbtools_ran &&
                report.bbtools_summaries.size() != report.bbtools_candidates.size()) {
                report.summaries_match_bbtools = false;
            }
            if (report.bbtools_ran) {
                for (size_t i = 0; i < report.bbtools_summaries.size(); ++i) {
                    auto const reference_id = report.bbtools_candidates[i];
                    if (report.bbtools_summaries[i] != report.oracle_summaries[reference_id]) {
                        report.summaries_match_bbtools = false;
                        break;
                    }
                }
            }

            // cuDDL candidate/summary comparisons against the oracle and the exhaustive pass are
            // performed inside the measured GPU loop, so they hold across every measured run.

            auto const report_ok =
                report.gpu_ran && report.candidates_match_oracle &&
                report.summaries_match_oracle && report.summaries_match_exhaustive &&
                report.decode_matches_bbtools && report.counts_match_bbtools &&
                report.candidates_match_bbtools && report.summaries_match_bbtools;
            ok = ok && report_ok;
        }

        // Compose evidence
        open_evidence();

        json queries_json = json::array();
        for (auto const& report : reports) {
            queries_json.push_back({
                {"ordinal", report.query.ordinal},
                {"name", report.query.name},
                {"external", report.query.external},
                {"row_checksum", report.row_checksum},
                {"row_nonzero", report.row_nonzero},
                {"oracle_candidate_count", report.oracle_candidates.size()},
                {"cuddl_candidate_count", report.gpu_ran ? report.cuddl_candidate_count : 0},
                {"bbtools_candidate_count", report.bbtools_candidates.size()},
                {"decode_matches_bbtools", report.decode_matches_bbtools},
                {"counts_match_bbtools", report.counts_match_bbtools},
                {"candidates_match_bbtools", report.candidates_match_bbtools},
                {"summaries_match_bbtools", report.summaries_match_bbtools},
                {"candidates_match_oracle", report.candidates_match_oracle},
                {"summaries_match_oracle", report.summaries_match_oracle},
                {"summaries_match_exhaustive", report.summaries_match_exhaustive},
                {"gpu_ran", report.gpu_ran},
                {"bbtools_ran", report.bbtools_ran},
                {"exact_refinement_ms", report.exact_refinement_ms},
                {"bbtools_query_seconds", report.bbtools_query_seconds},
                {"bbtools_query_runs", report.bbtools_query_runs},
                {"bbtools_query_csr_seconds", report.bbtools_query_csr_seconds},
                {"bbtools_query_csr_runs", report.bbtools_query_csr_runs},
            });
        }

        json evidence;
        evidence["ok"] = ok;
        evidence["min_hits"] = min_hits;
        evidence["measurement"] = {
            {"warmup", warmup},
            {"runs", runs},
            {"query_mode", "batch_by_class"},
        };
        evidence["manifest"] = manifest_path.empty() ? "built-in" : manifest_path;
        evidence["selection"] = {
            {"external", std::vector<uint32_t>(selection.external.begin(), selection.external.end())},
            {"queries", selection.queries},
        };
        evidence["asset"] = {
            {"path", asset_path},
            {"resource", prereq.resource},
            {"release", prereq.release},
            {"url", prereq.url},
            {"sha256_expected", prereq.sha256},
            {"sha256_computed", sha256_of_file(asset_path)},
            {"size_bytes_expected", prereq.size_bytes},
            {"size_bytes_actual", std::filesystem::file_size(asset_path)},
            {"merged", prereq.merged},
            {"k", prereq.k},
            {"buckets", prereq.buckets},
            {"exponent", prereq.exponent},
            {"records_expected", prereq.records},
            {"seed", db.metadata.has_seed ? db.metadata.seed : seed},
            {"blacklist", db.metadata.blacklist},
        };
        evidence["records"] = db.records.size();
        evidence["db_records"] = db_ordinals.size();
        evidence["external_records"] = selection.external.size();
        evidence["phases_ms"] = {
            {"asset_read", t_load_end - t_load_start},
            {"a48_parse", t_parse_end - t_parse_start},
            {"host_oracle_total", oracle_total_ms},
            {"host_to_device", host_to_device_ms},
            {"index_build", index_build_ms},
            {"index_build_runs", index_build_runs},
            {"in_database_query_total", in_database_query_ms},
            {"in_database_query_runs", in_database_query_runs},
            {"external_query_total", external_query_ms},
            {"external_query_runs", external_query_runs},
            {"exhaustive_gpu_total", exhaustive_batch_ms},
            {"exhaustive_gpu_runs", exhaustive_batch_runs},
        };
        int cuda_runtime_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        evidence["gpu"] = {
            {"available", gpu_available},
            {"cuda_runtime_version", cuda_runtime_version},
            {"device_free_before", device_free_before},
            {"device_free_low", device_free_low},
            {"device_total", device_total},
            {"persistent_rows_bytes", device_rows_bytes},
            {"persistent_index_bytes", device_index_bytes},
            {"peak_allocated_bytes", device_peak_bytes},
        };
        if (bbtools_ran) {
            evidence["bbtools"] = {
                {"harness", harness_source},
                {"java_heap", java_heap},
                {"threads", bbtools_threads},
                {"timings_seconds", bbtools.value("timings_seconds", json::object())},
                {"memory_bytes", bbtools.value("memory_bytes", json::object())},
                {"db_records", bbtools.value("db_records", size_t{0})},
            };
        } else {
            evidence["bbtools"] = json::object();
        }
        evidence["queries"] = std::move(queries_json);

        auto const text = evidence.dump(2);
        FILE* out = evidence_file != nullptr ? evidence_file : stdout;
        std::fwrite(text.data(), 1, text.size(), out);
        std::fputc('\n', out);
        if (evidence_file != nullptr) {
            std::fclose(evidence_file);
        }

        if (!ok) {
            std::fprintf(stderr, "Parity validation FAILED -- see evidence for details.\n");
            return 1;
        }
        std::fprintf(
            stderr,
            "Parity validation OK: %zu query report(s)%s%s.\n",
            reports.size(),
            gpu_available ? " (GPU)" : " (no GPU)",
            bbtools_ran ? " (BBTools DDLIndex/CSR2)" : ""
        );
        return 0;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "RefSeq parity error: %s\n", error.what());
        return 1;
    }
}
