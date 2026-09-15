// cub-exact-pairwise: exact k-mer set baseline from CUB device primitives.
//
// Parses FASTA files with the cuDDL parser (k=25 canonical packed k-mers)
// one genome at a time. Genomes touched by evaluated pairs keep their
// packed arrays; the rest stream through for distinct counts only, so
// host memory stays bounded by the evaluated set. Per genome: upload,
// CUB radix sort, run-length encode; the run count is the exact
// distinct cardinality. Per evaluated pair: concatenate on device, sort,
// run-length encode; the run count is the exact
// union cardinality, so shared = |A| + |B| - union. No custom kernels:
// sort and encode are CUB device-wide calls. The only downloads are one
// integer per stage; pair metrics are host math.
//
// --max-pairs stride-samples the evaluated pair space (first and last pair
// always measured) and --match-rows stride-samples the emitted rows the same
// way; 0 disables either cap. --sketch-only skips pair intersection for
// full-corpus parse timing; pair metrics come from the subset run instead.
// matching the CLI tools, with the parse subtotal reported separately.
// Allocation and temp-storage sizing stay outside timing; the report carries
// buffer sizes instead.
#include <cuddl/cuda_error.hpp>
#include <cuddl/error.hpp>
#include <cuddl/fastx.hpp>

#include <CLI/CLI.hpp>
#include <cub/cub.cuh>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

// Packing one genome into k-mers is a serial pass that costs an order of magnitude more than
// that genome's device stages, so the sketch loop parses ahead on a bounded pool instead of
// stalling the GPU. Depth bounds host memory: each worker holds one genome's packed k-mers.
class genome_parse_pool {
   public:
    genome_parse_pool(
        std::vector<std::string> const& names,
        size_t depth,
        size_t max_kmers
    )
        : names_(names), depth_(std::max<size_t>(1, depth)), max_kmers_(max_kmers),
          results_(names.size()), errors_(names.size()) {
        workers_.reserve(depth_);
        for (size_t i = 0; i < depth_; ++i) {
            workers_.emplace_back([this] { work(); });
        }
    }
    ~genome_parse_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        assign_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    genome_parse_pool(genome_parse_pool const&) = delete;
    genome_parse_pool& operator=(genome_parse_pool const&) = delete;

    /// @brief Packed k-mers for genome @p index, in input order.
    [[nodiscard]] std::vector<uint64_t> take(size_t index) {
        std::unique_lock lock(mutex_);
        filled_.wait(lock, [&] { return results_[index].has_value() || errors_[index] != nullptr; });
        ++taken_;
        assign_.notify_all();
        lock.unlock();
        if (errors_[index] != nullptr) std::rethrow_exception(errors_[index]);
        return std::move(*results_[index]);
    }

   private:
    void work() {
        while (true) {
            size_t index;
            {
                std::unique_lock lock(mutex_);
                assign_.wait(lock, [&] {
                    return stop_ || next_ >= names_.size() || next_ - taken_ < depth_;
                });
                if (stop_ || next_ >= names_.size()) return;
                if (next_ - taken_ >= depth_) continue;
                index = next_++;
            }
            try {
                // One thread per genome: the pool supplies the concurrency.
                auto parsed = cuddl::parse_fasta_file(names_[index], 25, 1);
                if (!parsed) throw std::runtime_error(parsed.error().message());
                if (parsed->kmers.size() > max_kmers_) {
                    throw std::runtime_error(
                        "genome exceeds --max-kmers, refusing: " + names_[index]
                    );
                }
                std::lock_guard lock(mutex_);
                results_[index] = std::move(parsed->kmers);
            } catch (...) {
                std::lock_guard lock(mutex_);
                errors_[index] = std::current_exception();
            }
            filled_.notify_all();
        }
    }

    std::vector<std::string> const& names_;
    size_t depth_;
    size_t max_kmers_;
    std::vector<std::optional<std::vector<uint64_t>>> results_;
    std::vector<std::exception_ptr> errors_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable assign_, filled_;
    size_t next_ = 0, taken_ = 0;
    bool stop_ = false;
};

/// @brief Host memory a run may spend on resident k-mer arrays, or 0 when it cannot be read.
[[nodiscard]] size_t available_host_bytes() noexcept {
    size_t available = 0;
    if (std::FILE* info = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, info) != nullptr) {
            unsigned long long kib = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
                available = static_cast<size_t>(kib) * 1024;
                break;
            }
        }
        std::fclose(info);
    }
    return available;
}

[[nodiscard]] unsigned parse_worker_count(size_t genomes) noexcept {
    auto const hardware = std::max(1U, std::thread::hardware_concurrency());
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(genomes, std::min<unsigned>(8U, hardware))));
}

using clock_type = std::chrono::steady_clock;

double median_of(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("no timing samples");
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2) return *middle;
    return (*std::max_element(values.begin(), middle) + *middle) / 2;
}

struct device_buffer {
    void* data = nullptr;
    size_t bytes = 0;
    void reset(size_t size) {
        if (size <= bytes) return;
        if (data) CUDDL_CUDA_CALL(cudaFree(data));
        CUDDL_CUDA_CALL(cudaMalloc(&data, size));
        bytes = size;
    }
    ~device_buffer() {
        if (data) CUDDL_CUDA_ABORT(cudaFree(data));
    }
};

size_t sort_temp_bytes(size_t count) {
    size_t bytes = 0;
    if (count) {
        CUDDL_CUDA_CALL(
            cub::DeviceRadixSort::SortKeys(
                nullptr,
                bytes,
                static_cast<uint64_t*>(nullptr),
                static_cast<uint64_t*>(nullptr),
                count
            )
        );
    }
    return bytes;
}

size_t encode_temp_bytes(size_t count) {
    size_t bytes = 0;
    if (count) {
        CUDDL_CUDA_CALL(
            cub::DeviceRunLengthEncode::Encode(
                nullptr,
                bytes,
                static_cast<uint64_t const*>(nullptr),
                static_cast<uint64_t*>(nullptr),
                static_cast<int*>(nullptr),
                static_cast<size_t*>(nullptr),
                count
            )
        );
    }
    return bytes;
}

int run_main(
    std::vector<std::string> const& references,
    std::vector<std::string> const& queries,
    bool all_to_all,
    int samples,
    int warmups,
    size_t max_kmers,
    size_t max_pairs,
    size_t match_rows,
    bool sketch_only,
    unsigned parse_workers,
    size_t stash_bytes,
    json& report
) {
    cudaStream_t stream = nullptr;
    CUDDL_CUDA_CALL(cudaStreamCreate(&stream));
    struct stream_guard {
        cudaStream_t stream;
        ~stream_guard() {
            CUDDL_CUDA_ABORT(cudaStreamDestroy(stream));
        }
    } guard{stream};

    std::vector<std::string> names;
    for (auto const& path : references) names.push_back(path);
    size_t const reference_count = names.size();
    for (auto const& path : queries) names.push_back(path);
    size_t const genomes = names.size();
    size_t const query_base = all_to_all ? 0 : reference_count;
    size_t const query_count = all_to_all ? genomes : genomes - reference_count;

    auto parse_one = [&](size_t g) {
        auto parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(names[g], 25));
        if (parsed.kmers.size() > max_kmers) {
            throw std::runtime_error("genome exceeds --max-kmers, refusing: " + names[g]);
        }
        return parsed.kmers;
    };
    // Parsing runs ahead of the GPU loop; a serial loop leaves the device idle between packs.
    std::optional<genome_parse_pool> parsers;
    if (parse_workers > 1) parsers.emplace(names, parse_workers, max_kmers);

    // Pair space enumeration needs no packed data, only indices.
    size_t const total_pairs = [&] {
        if (all_to_all) return genomes * (genomes - 1) / 2;
        return query_count * reference_count;
    }();
    size_t const pair_stride =
        max_pairs && total_pairs > max_pairs ? (total_pairs + max_pairs - 1) / max_pairs : 1;
    auto pair_at = [&](size_t ordinal) {
        if (all_to_all) {
            // Triangular index: ordinal -> (q, r) with q < r.
            size_t q = 0, lo = 0;
            while (lo + (genomes - 1 - q) <= ordinal) {
                lo += genomes - 1 - q;
                ++q;
            }
            return std::pair{q, q + 1 + (ordinal - lo)};
        }
        return std::pair{query_base + ordinal / reference_count, ordinal % reference_count};
    };
    // Sketch-only runs never evaluate a pair. Enumerating a full-corpus
    // pair space here would materialize billions of ordinals.
    std::vector<size_t> evaluated;
    if (!sketch_only) {
        for (size_t i = 0; i < total_pairs; i += pair_stride) evaluated.push_back(i);
        if (!evaluated.empty() && evaluated.back() != total_pairs - 1) {
            evaluated.push_back(total_pairs - 1);
        }
    }
    // Genomes touched by evaluated pairs keep their packed arrays across the
    // pair loop; the rest stream through for distinct counts only.
    std::vector<char> needed(genomes, 0);
    for (size_t ordinal : evaluated) {
        auto const [qa, rb] = pair_at(ordinal);
        needed[qa] = 1;
        needed[rb] = 1;
    }
    // Retaining a genome's k-mers is what makes a pair cheap, but every genome touched by an
    // evaluated pair would be kept: over a full corpus that is terabytes, which is how a run
    // sets the machine's memory alight. Keep a budget and re-parse whatever falls outside it.
    std::vector<std::vector<uint64_t>> stashed(genomes);
    std::vector<char> resident(genomes, 0);
    std::vector<uint64_t> reparsed_a, reparsed_r;
    size_t stashed_bytes = 0;
    size_t reparsed_genomes = 0;

    device_buffer work, uniques, run_counts, concat, pair_uniques, pair_counts, num_runs_dev, temp;
    size_t max_keys = 0, max_pair = 0, temp_bytes = 0;
    num_runs_dev.reset(sizeof(size_t));
    auto fetch_runs = [&] {
        int runs = 0;
        CUDDL_CUDA_CALL(
            cudaMemcpyAsync(&runs, num_runs_dev.data, sizeof(runs), cudaMemcpyDeviceToHost, stream)
        );
        CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
        return runs;
    };

    std::vector<size_t> kmers_of(genomes, 0), distinct(genomes, 0);
    std::vector<double> parse_ms, sketch_ms, compare_ms, end_to_end_ms;
    // Emitted rows stride-sample the evaluated pairs, first and last always kept. Applying the
    // stride as rows are produced keeps only what will be reported: a full corpus evaluates
    // hundreds of millions of pairs, and holding one JSON object each needs far more memory
    // than the k-mer arrays this benchmark reads.
    json emitted = json::array();
    size_t const emit_stride = match_rows && evaluated.size() > match_rows
                                   ? (evaluated.size() + match_rows - 1) / match_rows
                                   : 1;

    for (int rep = -warmups; rep < samples; ++rep) {
        auto const sample_tick = clock_type::now();
        auto parse_tick = clock_type::now();
        // Sketch streams one genome at a time; only counts are retained.
        for (size_t g = 0; g < genomes; ++g) {
            auto packed = parsers ? parsers->take(g) : parse_one(g);
            size_t const count = packed.size();
            kmers_of[g] = count;
            max_keys = std::max(max_keys, count);
            work.reset(count * sizeof(uint64_t));
            uniques.reset(count * sizeof(uint64_t));
            run_counts.reset(count * sizeof(int));
            auto* keys = static_cast<uint64_t*>(work.data);
            temp_bytes =
                std::max(temp_bytes, std::max(sort_temp_bytes(count), encode_temp_bytes(count)));
            temp.reset(temp_bytes);
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                keys, packed.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream
            ));
            if (count) {
                CUDDL_CUDA_CALL(
                    cub::DeviceRadixSort::SortKeys(
                        temp.data, temp.bytes, keys, keys, count, 0, 64, stream
                    )
                );
                CUDDL_CUDA_CALL(
                    cub::DeviceRunLengthEncode::Encode(
                        temp.data,
                        temp.bytes,
                        keys,
                        static_cast<uint64_t*>(uniques.data),
                        static_cast<int*>(run_counts.data),
                        static_cast<int*>(num_runs_dev.data),
                        count,
                        stream
                    )
                );
            }
            CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
            distinct[g] = count ? fetch_runs() : 0;
            if (needed[g]) {
                auto const bytes = packed.size() * sizeof(uint64_t);
                // Leave room for the pair working set as well as the resident arrays.
                if (stashed_bytes + bytes <= stash_bytes) {
                    stashed_bytes += bytes;
                    resident[g] = 1;
                    stashed[g] = std::move(packed);
                }
            }
        }
        // Pair buffers sized from the largest evaluated pair actually measured.
        for (size_t ordinal : evaluated) {
            auto const [qa, rb] = pair_at(ordinal);
            max_pair = std::max(max_pair, kmers_of[qa] + kmers_of[rb]);
        }
        concat.reset(max_pair * sizeof(uint64_t));
        pair_uniques.reset(max_pair * sizeof(uint64_t));
        pair_counts.reset(max_pair * sizeof(int));
        temp_bytes =
            std::max(temp_bytes, std::max(sort_temp_bytes(max_pair), encode_temp_bytes(max_pair)));
        temp.reset(temp_bytes);
        auto const parse_done = clock_type::now();
        auto const compare_tick = clock_type::now();
        emitted = json::array();
        reparsed_a.clear();
        reparsed_r.clear();
        size_t pair_index = 0;
        for (size_t ordinal : evaluated) {
            auto const [a, r] = pair_at(ordinal);
            // Resident arrays come from the sketch pass; anything outside the budget is packed
            // again here, which costs a parse but keeps memory bounded.
            if (!resident[a]) {
                reparsed_a = parse_one(a);
                ++reparsed_genomes;
            }
            if (!resident[r]) {
                reparsed_r = parse_one(r);
                ++reparsed_genomes;
            }
            auto const& packed_a = resident[a] ? stashed[a] : reparsed_a;
            auto const& packed_r = resident[r] ? stashed[r] : reparsed_r;
            size_t shared = 0;
            if (distinct[a] && distinct[r]) {
                auto* keys = static_cast<uint64_t*>(concat.data);
                CUDDL_CUDA_CALL(cudaMemcpyAsync(
                    keys,
                    packed_a.data(),
                    packed_a.size() * sizeof(uint64_t),
                    cudaMemcpyHostToDevice,
                    stream
                ));
                CUDDL_CUDA_CALL(cudaMemcpyAsync(
                    keys + packed_a.size(),
                    packed_r.data(),
                    packed_r.size() * sizeof(uint64_t),
                    cudaMemcpyHostToDevice,
                    stream
                ));
                size_t const total = packed_a.size() + packed_r.size();
                CUDDL_CUDA_CALL(
                    cub::DeviceRadixSort::SortKeys(
                        temp.data, temp.bytes, keys, keys, total, 0, 64, stream
                    )
                );
                CUDDL_CUDA_CALL(
                    cub::DeviceRunLengthEncode::Encode(
                        temp.data,
                        temp.bytes,
                        keys,
                        static_cast<uint64_t*>(pair_uniques.data),
                        static_cast<int*>(pair_counts.data),
                        static_cast<int*>(num_runs_dev.data),
                        total,
                        stream
                    )
                );
                size_t const union_size = fetch_runs();
                shared = distinct[a] + distinct[r] - union_size;
            }
            size_t const pair_union = distinct[a] + distinct[r] - shared;
            double const jaccard = pair_union ? static_cast<double>(shared) / pair_union : 0.0;
            double mash_ani = 0.0;
            if (jaccard > 0 && jaccard <= 1) {
                mash_ani = (1.0 + std::log(2 * jaccard / (1 + jaccard)) / 25.0) * 100.0;
            }
            if (pair_index % emit_stride == 0 || pair_index + 1 == evaluated.size()) {
                emitted.push_back(
                    {{"query", names[a]},
                 {"reference", names[r]},
                 {"distinct_a", distinct[a]},
                 {"distinct_b", distinct[r]},
                 {"intersection", shared},
                 {"union", pair_union},
                 {"jaccard", jaccard},
                 {"containment_a_in_b",
                  distinct[a] ? static_cast<double>(shared) / distinct[a] : 0.0},
                 {"containment_b_in_a",
                  distinct[r] ? static_cast<double>(shared) / distinct[r] : 0.0},
                     {"mash_ani", mash_ani}}
                );
            }
            ++pair_index;
        }
        auto const done = clock_type::now();
        if (rep >= 0) {
            using ms = std::chrono::duration<double, std::milli>;
            parse_ms.push_back(ms(parse_done - sample_tick).count());
            sketch_ms.push_back(ms(compare_tick - sample_tick).count());
            compare_ms.push_back(ms(done - compare_tick).count());
            end_to_end_ms.push_back(ms(done - sample_tick).count());
        }
    }

    json genome_rows = json::array();
    for (size_t g = 0; g < genomes; ++g) {
        genome_rows.push_back(
            {{"path", names[g]}, {"kmers", kmers_of[g]}, {"distinct", distinct[g]}}
        );
    }
    auto summarize = [](std::vector<double> const& values) {
        return json{
            {"samples", values.size()},
            {"median_ms", median_of(values)},
            {"min_ms", *std::min_element(values.begin(), values.end())},
            {"max_ms", *std::max_element(values.begin(), values.end())},
            {"source", "steady_clock_cpu_wall"},
        };
    };
    report = {
        {"implementation", {{"name", "cub-exact"}}},
        {"case",
         {{"k", 25},
          {"topology", all_to_all ? "all-to-all" : "batch"},
          {"references", reference_count},
          {"queries", query_count},
          {"samples", samples},
          {"warmups", warmups},
          {"pairs_total", total_pairs},
          {"pairs_evaluated", evaluated.size()},
          {"parse_workers", parse_workers},
          {"stash_mb_allowed", stash_bytes >> 20},
          {"stashed_genomes", static_cast<size_t>(std::count(resident.begin(), resident.end(), 1))},
          {"stashed_mb", stashed_bytes >> 20},
          {"reparsed_genomes", reparsed_genomes},
          {"pair_stride", pair_stride},
          {"pairs_emitted", emitted.size()}}},
        {"genomes", genome_rows},
        {"phases_ms",
         {{"parse_wall", summarize(parse_ms)},
          {"device_buffers",
           {{"genome_keys", max_keys * sizeof(uint64_t)},
            {"pair_keys", max_pair * sizeof(uint64_t)},
            {"pair_runs", max_pair * (sizeof(uint64_t) + sizeof(int))},
            {"temp", temp.bytes}}},
          {"sketch", summarize(sketch_ms)},
          {"compare", summarize(compare_ms)},
          {"end_to_end", summarize(end_to_end_ms)}}},
        {"pairs", emitted},
    };
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<std::string> references, queries;
    std::string topology = "batch", output;
    int samples = 5, warmups = 1;
    size_t max_kmers = 32ULL << 20, max_pairs = 0, match_rows = 0;
    unsigned workers = 0;
    size_t stash_mb = 0;
    CLI::App app{"Exact k-mer set baseline from CUB primitives (k=25)"};
    app.add_option("--reference", references)->required()->check(CLI::ExistingFile);
    app.add_option("--query", queries)->check(CLI::ExistingFile);
    app.add_option("--topology", topology)->check(CLI::IsMember({"batch", "all-to-all"}));
    app.add_option("--samples", samples)->check(CLI::Range(1, 1000));
    app.add_option("--warmups", warmups)->check(CLI::Range(0, 100));
    app.add_option("--max-kmers", max_kmers);
    app.add_option("--max-pairs", max_pairs, "Evaluated pairs cap, even stride (0 disables)");
    app.add_option("--match-rows", match_rows, "Emitted pair rows cap, even stride (0 disables)");
    app.add_option(
        "--workers",
        workers,
        "Concurrent genome parsers ahead of the GPU loop; 0 selects up to 8"
    );
    app.add_option(
        "--stash-mb",
        stash_mb,
        "Host memory for resident k-mer arrays; 0 uses half of MemAvailable. Arrays outside the "
        "budget are packed again per pair instead of being retained."
    );
    app.add_option("--output", output)->required();
    app.set_config("--config", "TOML file with options, e.g. reference = [...]");
    bool sketch_only = false;
    app.add_flag("--sketch-only", sketch_only, "Parse plus per-genome device work only, no pairs");
    CLI11_PARSE(app, argc, argv);
    if (topology == "batch" && (references.empty() || queries.empty())) {
        throw std::runtime_error("batch needs nonempty --reference and --query");
    }
    if (topology == "all-to-all" && references.size() < 2) {
        throw std::runtime_error("all-to-all needs at least two --reference files");
    }
    json report;
    run_main(
        references,
        queries,
        topology == "all-to-all",
        samples,
        warmups,
        max_kmers,
        max_pairs,
        match_rows,
        sketch_only,
        workers == 0 ? parse_worker_count(references.size() + queries.size()) : workers,
        stash_mb ? stash_mb << 20
                 : std::max<size_t>(available_host_bytes() / 2, size_t{1} << 30),
        report
    );
    FILE* stream = std::fopen(output.c_str(), "w");
    if (!stream) throw std::runtime_error("cannot write " + output);
    auto const text = report.dump();
    if (std::fwrite(text.data(), 1, text.size(), stream) != text.size() ||
        std::fclose(stream) != 0) {
        throw std::runtime_error("cannot write " + output);
    }
    std::cout << "wrote " << output << " with " << report["pairs"].size() << " pairs\n";
    return 0;
} catch (std::exception const& error) {
    std::cerr << "cub-exact-pairwise: " << error.what() << '\n';
    return 1;
}
