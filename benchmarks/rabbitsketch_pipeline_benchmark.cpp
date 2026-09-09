#include <api/RuntimeInfo.h>
#include <api/SketchBuilder.h>
#include <api/Version.h>
#include <fastkmv.h>
#include <omp.h>
#include <rank/RankStream.h>
#include <CLI/CLI.hpp>
#include <cuddl/fastx.hpp>
#include <exception>
#include <functional>
#include <nvbench/nvbench.cuh>
#include <optional>
#include <utility>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "host_result_json.hpp"

namespace {
namespace api = Sketch::API;
using collection = std::vector<api::BuildResult>;
using records = std::vector<std::vector<Sketch::IO::FastxRecord>>;
using json = nlohmann::json;

struct options {
    std::vector<std::string> references, queries;
    std::string output, name = "RabbitSketch CPU pipeline", topology = "batch";
    std::string ingest = "packed";
    int k = 25, sketch_size = 4096, samples = 20, warmups = 3;
    uint64_t seed = 42;
    int threads = omp_get_num_procs();
    size_t match_rows = 20000, dataset_hashes = 8, all_to_all_pairs = 50000000;
};

template <typename Function>
void parallel_for(size_t count, Function&& function) {
    if (count == 0) return;
    int const workers =
        static_cast<int>(std::min(count, static_cast<size_t>(omp_get_max_threads())));
    std::exception_ptr failure;
    _Pragma("omp parallel for num_threads(workers) schedule(dynamic)")
    for (size_t i = 0; i < count; ++i) {
        try {
            function(i);
        } catch (...) {
            _Pragma("omp critical(rabbitsketch_failure)")
            {
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
}

api::SketchConfig config(options const& opts) {
    api::SketchConfig value;
    value.algorithm = api::Algorithm::FastKMV;
    value.sampling = api::SamplingMode::BottomK;
    value.aggregation = api::AggregationMode::OneSketchPerFile;
    value.ambiguous_policy = api::AmbiguousPolicy::SkipKmer;
    value.kmer_size = opts.k;
    value.resolution = opts.sketch_size;
    value.seed = opts.seed;
    value.canonical = true;
    value.validate();
    return value;
}

// All application stages consume their outputs before the measured interval ends.
volatile size_t consumed_size = 0;
json measure(
    options const& opts,
    const std::function<void()>& function,
    const std::function<void()>& prepare = {}
) {
    auto run = [&](nvbench::state& state, nvbench::type_list<>) {
        // CPU measurements need a launch placeholder, not an allocated CUDA stream.
        state.set_cuda_stream(nvbench::make_cuda_stream_view(nullptr));
        bool warming = true;
        state.exec(
            nvbench::exec_tag::no_gpu | nvbench::exec_tag::timer,
            [&](nvbench::launch&, auto& timer) {
                // CPU mode invokes one warmup callback; expand it to the requested count.
                auto const repetitions = std::exchange(warming, false) ? opts.warmups : 1;
                if (repetitions == 0) {
                    timer.start();
                    timer.stop();
                }
                for (int i = 0; i < repetitions; ++i) {
                    if (prepare) prepare();
                    timer.start();
                    function();
                    timer.stop();
                }
            }
        );
    };
    nvbench::benchmark<decltype(run)> benchmark(run);
    benchmark.set_is_cpu_only(true)
        .set_stopping_criterion("sample-count")
        .set_min_samples(opts.samples)
        .set_criterion_param_int64("target-samples", opts.samples)
        .set_skip_batched(true);
    benchmark.run();
    auto const& state = benchmark.get_states().front();
    if (state.is_skipped()) throw std::runtime_error(state.get_skip_reason());
    std::string const prefix = "nv/cpu_only/time/cpu";
    json result = {
        {"samples", state.get_summary("nv/cpu_only/sample_size").get_int64("value")},
        {"median_ms", state.get_summary(prefix + "/median").get_float64("value") * 1000},
        {"min_ms", state.get_summary(prefix + "/min").get_float64("value") * 1000},
        {"max_ms", state.get_summary(prefix + "/max").get_float64("value") * 1000},
        {"source", "nvbench_cpu_wall"},
    };
    auto const noise = state.get_summary(prefix + "/stdev/relative").get_float64("value");
    if (std::isfinite(noise)) result["relative_stddev_percent"] = noise * 100;
    return result;
}

records parse(std::vector<std::string> const& paths) {
    records result(paths.size());
    parallel_for(paths.size(), [&](size_t i) {
        auto& file = result[i];
        Sketch::IO::FastxReader reader(paths[i]);
        Sketch::IO::FastxRecord record;
        while (reader.next(record)) {
            file.push_back(std::move(record));
        }
    });
    return result;
}

template <typename Function>
collection build_parallel(size_t count, Function&& function) {
    std::vector<std::optional<api::BuildResult>> built(count);
    parallel_for(count, [&](size_t i) { built[i].emplace(function(i)); });
    collection result;
    result.reserve(count);
    for (auto& value : built) {
        result.push_back(std::move(*value));
    }
    return result;
}

collection construct(records const& files, api::SketchConfig const& cfg) {
    return build_parallel(files.size(), [&](size_t i) {
        api::MultiSketchBuilder builder({cfg});
        for (auto const& record : files[i]) {
            builder.update(record);
        }
        auto built = builder.finish("genome");
        return std::move(built.at(0));
    });
}

collection build_files(std::vector<std::string> const& paths, api::SketchConfig const& cfg) {
    return build_parallel(paths.size(), [&](size_t i) {
        auto built = api::buildFastxFiles({paths[i]}, {cfg});
        return std::move(built.at(0));
    });
}

struct match {
    size_t query_id, reference_id;
    Sketch::Query::Result result;
};

std::vector<match> search(collection const& refs, collection const& queries, bool all) {
    std::vector<match> result;
    auto const& left = all ? refs : queries;
    if (!refs.empty() && left.size() > result.max_size() / refs.size()) {
        throw std::length_error("pair results exceed addressable storage");
    }
    result.resize(all ? refs.size() * (refs.size() - 1) / 2 : left.size() * refs.size());
    parallel_for(left.size(), [&](size_t q) {
        for (size_t r = all ? q + 1 : 0; r < refs.size(); ++r) {
            auto value = left[q].sketch.query(refs[r].sketch);
            if (!value.estimate_available || !std::isfinite(value.jaccard)) {
                throw std::runtime_error("RabbitSketch query unavailable: " + value.plan.reason);
            }
            auto const position =
                all ? q * (2 * refs.size() - q - 1) / 2 + r - q - 1 : q * refs.size() + r;
            result[position] = {q, r, std::move(value)};
        }
    });
    return result;
}

json query_metrics(Sketch::Query::Result const& value) {
    json result = {
        {"estimate_available", value.estimate_available},
        {"method", Sketch::Query::methodName(value.plan.method)},
        {"effective_samples", value.effective_samples},
    };
    auto add = [&](char const* name, double metric) {
        result[std::string(name) + "_defined"] = std::isfinite(metric);
        if (std::isfinite(metric)) {
            result[name] = metric;
        }
    };
    add("jaccard", value.jaccard);
    add("jaccard_distance", value.jaccard_distance);
    add("mash_distance", value.mash_distance);
    add("ani", value.ani);
    add("intersection", value.intersection);
    add("union", value.union_size);
    add("query_cardinality", value.left_cardinality);
    add("reference_cardinality", value.right_cardinality);
    add("query_containment", value.left_containment);
    add("reference_containment", value.right_containment);
    std::string warnings;
    for (auto const& warning : value.warnings) {
        if (!warnings.empty()) {
            warnings += "; ";
        }
        warnings += warning;
    }
    result["warnings"] = warnings;
    return result;
}

json measurements(
    collection const& refs,
    collection const& queries,
    std::vector<match> const& matches,
    size_t match_limit = 0,
    size_t* emitted = nullptr
) {
    json result = json::array();
    size_t count = 0;
    auto add_match = [&](match const& row) {
        result.push_back({
            {"implementation", {{"name", "rabbitsketch"}}},
            {"case",
             {{"measurement", "match"},
              {"query_id", row.query_id},
              {"reference_id", row.reference_id}}},
            {"metrics", query_metrics(row.result)},
        });
        ++count;
    };
    // One evenly spread row sample, always including the last pair.
    size_t const limit = match_limit ? match_limit : matches.size();
    size_t const stride = matches.size() > limit ? (matches.size() + limit - 1) / limit : 1;
    for (size_t i = 0; i < matches.size(); i += stride) {
        add_match(matches[i]);
    }
    if (matches.size() && (matches.size() - 1) % stride != 0) {
        add_match(matches.back());
    }
    if (emitted != nullptr) {
        *emitted = count;
    }
    for (auto const& [role, group] : std::vector<std::pair<std::string, collection const*>>{
             {"reference", &refs}, {"query", &queries}
         }) {
        for (size_t i = 0; i < group->size(); ++i) {
            auto const& genome = group->at(i);
            auto metrics = query_metrics(genome.sketch.query(genome.sketch));
            metrics.update({
                {"records", genome.stats.records},
                {"bases", genome.stats.input_bases},
                {"candidate_kmers", genome.stats.candidate_kmers},
                {"accepted_kmers", genome.stats.accepted_kmers},
                {"skipped_ambiguous_kmers", genome.stats.skipped_ambiguous_kmers},
            });
            result.push_back({
                {"implementation", {{"name", "rabbitsketch"}}},
                {"case", {{"measurement", "genome"}, {"role", role}, {"genome_id", i}}},
                {"metrics", std::move(metrics)},
            });
        }
    }
    return result;
}

json resident_timings(options const& opts) {
    std::vector<std::vector<uint64_t>> inputs;
    for (auto const* paths : {&opts.references, &opts.queries}) {
        for (auto const& path : *paths) {
            auto parsed = cuddl::parse_fasta_file(path, opts.k);
            if (!parsed) {
                throw std::runtime_error("cannot prepare packed input: " + path);
            }
            inputs.push_back(std::move(parsed.value().kmers));
        }
    }
    std::vector<Sketch::FastKMV> sketches;
    sketches.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        sketches.emplace_back(opts.sketch_size, opts.k, opts.seed);
    }
    auto const n = opts.references.size();
    auto const all = opts.topology == "all-to-all";
    auto const q = all ? n : opts.queries.size();
    if (n && q > std::numeric_limits<size_t>::max() / n) {
        throw std::runtime_error("resident result size overflow");
    }
    std::vector<double> cardinalities(inputs.size()), similarities(n * q);
    auto reset = [&] {
        parallel_for(sketches.size(), [&](size_t i) { sketches[i].clear(); });
    };
    auto construct = [&] {
        parallel_for(inputs.size(), [&](size_t i) {
            sketches[i].updatePacked(inputs[i].data(), inputs[i].size());
        });
    };
    auto finalize = [&] {
        parallel_for(sketches.size(), [&](size_t i) { sketches[i].finalize(); });
    };
    auto cardinality = [&] {
        parallel_for(sketches.size(), [&](size_t i) {
            cardinalities[i] = sketches[i].cardinality();
        });
    };
    auto compare = [&] {
        parallel_for(similarities.size(), [&](size_t position) {
            auto const i = position / n, j = position % n;
            if (all && j <= i) return;
            similarities[position] = sketches[all ? i : n + i].jaccard(sketches[j]);
        });
        consumed_size = similarities.size();
    };
    auto pipeline = [&] {
        reset();
        construct();
        finalize();
        cardinality();
        compare();
    };
    pipeline();
    // Validate packed SIMD ingestion against scalar hash/sort bottom-k, outside timing.
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<uint64_t> expected;
        expected.reserve(inputs[i].size());
        for (auto value : inputs[i]) {
            expected.push_back(Sketch::Rank::RankStream::fmix64(value, opts.seed) >> 11);
        }
        std::sort(expected.begin(), expected.end());
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
        expected.resize(std::min(expected.size(), static_cast<size_t>(opts.sketch_size)));
        if (sketches[i].size() != expected.size() ||
            !std::equal(expected.begin(), expected.end(), sketches[i].getRegisters())) {
            throw std::runtime_error("packed FastKMV differs from scalar bottom-k oracle");
        }
    }
    auto const expected = similarities;
    json timings;
    timings["resident_total_wall"] = measure(opts, pipeline);
    if (similarities != expected) {
        throw std::runtime_error("resident replay changed pair results");
    }
    timings["resident_reset_wall"] = measure(opts, reset);
    timings["resident_construct_wall"] = measure(opts, construct, reset);
    timings["resident_finalize_wall"] = measure(opts, finalize, [&] {
        reset();
        construct();
    });
    timings["resident_cardinality_wall"] = measure(opts, cardinality);
    timings["resident_search_wall"] = measure(opts, compare);
    return timings;
}

json run(options const& opts) {
    auto const cfg = config(opts);
    auto const all = opts.topology == "all-to-all";
    bool const streamed = opts.ingest == "sequence";
    auto const reference_count = opts.references.size();
    uint64_t const all_to_all_pairs =
        reference_count > 1 ? uint64_t{reference_count} * (reference_count - 1) / 2 : 0;
    if (all && opts.all_to_all_pairs && all_to_all_pairs > opts.all_to_all_pairs) {
        throw std::runtime_error(
            "all-to-all over " + std::to_string(reference_count) + " references retains " +
            std::to_string(all_to_all_pairs) + " pair results, above --all-to-all-pairs (" +
            std::to_string(opts.all_to_all_pairs) +
            "); use --topology batch with a small --query set"
        );
    }
    auto build_queries = [&] {
        return build_files(opts.queries, cfg);
    };
    // The untimed run warms input caches and checks file ingest against record construction.
    auto refs = build_files(opts.references, cfg);
    auto queries = build_queries();
    auto matches = search(refs, queries, all);
    size_t match_rows_emitted = 0;
    auto rows = measurements(refs, queries, matches, opts.match_rows, &match_rows_emitted);
    std::string resident_scope = "all_files";
    if (streamed) {
        // Bounded check: build one genome per role both ways and compare sketch and parse
        // metrics. The type-erased BuiltSketch exposes no registers.
        resident_scope = "first_file_per_role";
        for (auto const* paths : {&opts.references, &opts.queries}) {
            if (paths->empty()) {
                continue;
            }
            auto const resident = construct(parse({paths->front()}), cfg);
            auto const ingested = build_files({paths->front()}, cfg);
            if (measurements(resident, {}, {}, 0) != measurements(ingested, {}, {}, 0)) {
                throw std::runtime_error(
                    "file ingest differs from record construction: " + paths->front()
                );
            }
        }
    } else {
        auto reference_records = parse(opts.references);
        auto query_records = parse(opts.queries);
        auto resident_refs = construct(reference_records, cfg);
        auto resident_queries = construct(query_records, cfg);
        if (measurements(
                resident_refs,
                resident_queries,
                search(resident_refs, resident_queries, all),
                opts.match_rows
            ) != rows) {
            throw std::runtime_error("resident and streaming RabbitSketch results differ");
        }
    }

    json timings = measure_pipeline(opts.samples, opts.warmups, [&](auto mark) {
        auto r = build_files(opts.references, cfg);
        auto q = build_queries();
        mark();
        auto hits = search(r, q, all);
        auto output = measurements(r, q, hits, opts.match_rows).dump();
        consumed_size = output.size();
        mark();
    });
    // The record and packed-input stages materialize the whole corpus, so a streamed run skips
    // them. RabbitSketch's own file ingest is the bounded path.
    if (!streamed) {
        auto reference_records = parse(opts.references);
        auto query_records = parse(opts.queries);
        timings["parse_fastx"] = measure(opts, [&] {
            consumed_size = reference_records.size() + query_records.size();
        });
        timings["construct_resident"] = measure(opts, [&] {
            auto r = construct(reference_records, cfg);
            auto q = construct(query_records, cfg);
            consumed_size = r.size() + q.size();
        });
    }
    timings[all ? "search_all_to_all_exhaustive" : "search_batch_exhaustive"] = measure(opts, [&] {
        auto hits = search(refs, queries, all);
        consumed_size = hits.size();
    });
    timings["metrics_and_serialize"] = measure(opts, [&] {
        auto output = measurements(refs, queries, matches, opts.match_rows).dump();
        consumed_size = output.size();
    });

    auto const& runtime = Sketch::Runtime::runtimeInfo();
    if (!streamed) {
        timings.update(resident_timings(opts));
    }
    rows.insert(
        rows.begin(),
        json{
            {"implementation",
             {{"name", "rabbitsketch"},
              {"version", api::VERSION},
              {"revision", RABBITSKETCH_REVISION},
              {"variant", "FastKMV"}}},
            {"case",
             {{"measurement", "pipeline"},
              {"k", opts.k},
              {"sketch_size", opts.sketch_size},
              {"hash_seed", opts.seed},
              {"canonical", true},
              {"ambiguous_policy", "SkipKmer"},
              {"aggregation", "OneSketchPerFile"},
              {"index", "none"},
              {"topology", opts.topology},
              {"references", refs.size()},
              {"queries", queries.size()},
              {"samples", opts.samples},
              {"warmups", opts.warmups},
              {"input_cache", "warm_os_cache"},
              {"orchestration_threads", opts.threads},
              {"resident_parallelism", "independent_sketches_and_pairs"},
              {"end_to_end_output", "host_metrics_and_in_memory_json"},
              {"ingest", opts.ingest},
              {"resident_input", streamed ? "fastx_files" : "packed_u64_actg_max"},
              {"resident_output", "host_cardinalities_and_jaccard_matrix"},
              {"resident_minimum_matches", 0},
              {"packed_input_patch", "FastKMV::updatePacked+clear"},
              {"native_arch", !runtime.portable_baseline},
              {"simd_byte_path", runtime.selected_byte_path},
              {"simd_u64_path", runtime.selected_u64_path},
              {"simd_rank_path", runtime.selected_rank_path},
              {"simd_override", runtime.environment_override},
              {"simd_override_honored", runtime.environment_override_honored}}},
            {"metrics",
             {{"exhaustive_pairs", matches.size()},
              {"match_rows_total", matches.size()},
              {"match_rows_emitted", match_rows_emitted},
              {"resident_streaming_equal", true},
              {"resident_streaming_scope", resident_scope},
              {"all_to_all_pairs", all_to_all_pairs}}},
            {"timings", timings},
        }
    );
    json datasets = json::object();
    for (auto const& [role, paths] :
         std::vector<std::pair<std::string, std::vector<std::string> const*>>{
             {"reference", &opts.references}, {"query", &opts.queries}
         }) {
        datasets.update(dataset_entries(role, *paths, opts.dataset_hashes));
    }
    return {
        {"schema", "cuddl-benchmark/v1"},
        {"name", opts.name},
        {"operation", "pipeline"},
        {"scope", "end_to_end"},
        {"datasets", std::move(datasets)},
        {"system", benchmark_host_system()},
        {"measurements", std::move(rows)}
    };
}
}  // namespace

int main(int argc, char** argv) try {
    options opts;
    CLI::App app{
        "RabbitSketch FastKMV CPU FASTA/FASTQ-to-results pipeline. Each file is one genome."
    };
    app.add_option("--reference", opts.references, "Reference FASTX files, repeatable")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--query", opts.queries, "Query FASTX files, repeatable; batch topology only")
        ->check(CLI::ExistingFile);
    app.add_option("--topology", opts.topology)->check(CLI::IsMember({"batch", "all-to-all"}));
    app.add_option("-k,--k", opts.k)->check(CLI::Range(1, 32));
    app.add_option("--sketch-size", opts.sketch_size)->check(CLI::Range(2, 100000000));
    app.add_option("--seed", opts.seed);
    app.add_option("--threads", opts.threads, "CPU workers; defaults to all available processors")
        ->check(CLI::PositiveNumber);
    app.add_option("--samples", opts.samples)->check(CLI::Range(2, 10000));
    app.add_option("--warmups", opts.warmups)->check(CLI::Range(0, 1000));
    app.add_option(
           "--ingest",
           opts.ingest,
           "packed: add the record and packed-input stages. sequence: RabbitSketch file ingest "
           "only, required for a large corpus"
    )
        ->check(CLI::IsMember({"packed", "sequence"}));
    app.add_option("--match-rows", opts.match_rows, "Match measurement rows emitted")
        ->check(CLI::Range(size_t{0}, size_t{1} << 40));
    app.add_option("--dataset-hashes", opts.dataset_hashes, "Per-file dataset digests")
        ->check(CLI::Range(size_t{0}, size_t{1} << 20));
    app.add_option(
           "--all-to-all-pairs",
           opts.all_to_all_pairs,
           "Reference pairs above which all-to-all is refused"
    )
        ->check(CLI::Range(size_t{0}, size_t{1} << 40));
    app.add_option("--output", opts.output, "Shared-schema JSON output, stdout if omitted");
    app.add_option("--name", opts.name);
    app.set_config("--config", "", "Read benchmark options from a configuration file");
    CLI11_PARSE(app, argc, argv);
    omp_set_dynamic(0);
    omp_set_num_threads(opts.threads);
    if (opts.topology == "batch" && opts.queries.empty()) {
        throw std::runtime_error("batch topology requires --query");
    }
    if (opts.topology == "all-to-all" && (opts.references.size() < 2 || !opts.queries.empty())) {
        throw std::runtime_error("all-to-all requires at least two references and no queries");
    }
    if (!opts.output.empty()) {
        for (auto const* paths : {&opts.references, &opts.queries}) {
            for (auto const& path : *paths) {
                if (std::filesystem::exists(opts.output) &&
                    std::filesystem::equivalent(path, opts.output)) {
                    throw std::runtime_error("output must not overwrite an input genome");
                }
            }
        }
    }
    auto report = run(opts).dump(2) + '\n';
    if (opts.output.empty()) {
        std::cout << report;
    } else {
        std::ofstream output(opts.output);
        if (!output) {
            throw std::runtime_error("cannot open output " + opts.output);
        }
        output << report;
        output.close();
        if (!output) {
            throw std::runtime_error("cannot write output " + opts.output);
        }
    }
    return 0;
} catch (std::exception const& error) {
    std::cerr << "rabbitsketch-pipeline: " << error.what() << '\n';
    return 1;
}
