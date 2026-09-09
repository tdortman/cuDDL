#include <api/RuntimeInfo.h>
#include <api/SketchBuilder.h>
#include <api/Version.h>
#include <fastkmv.h>
#include <omp.h>
#include <rank/CanonicalKmer.h>
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
#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

#include "host_result_json.hpp"
#include "resident_sequence_batches.hpp"

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
    size_t resident_bytes = 64ULL << 20;
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

struct search_result {
    // One evenly spread pair sample, always including the last pair.
    std::vector<match> matches;
    uint64_t pairs = 0;
};

// @p retain_limit bounds retained results (0 keeps every pair). All pairs are still queried, so
// the pair count and the timing stay complete while a large all-to-all matrix stays bounded.
search_result search(
    collection const& refs,
    collection const& queries,
    bool all,
    size_t retain_limit
) {
    auto const& left = all ? refs : queries;
    auto const pair_count = all ? uint64_t{refs.size()} * (refs.size() - 1) / 2
                                : uint64_t{left.size()} * refs.size();
    size_t const limit = retain_limit ? retain_limit : (pair_count ? pair_count : 1);
    size_t const stride = pair_count > limit ? (pair_count + limit - 1) / limit : 1;
    bool const extra_last = pair_count > 0 && (pair_count - 1) % stride != 0;
    size_t const slots =
        pair_count == 0 ? 0 : (pair_count - 1) / stride + 1 + (extra_last ? 1 : 0);
    std::vector<match> retained(slots);
    parallel_for(left.size(), [&](size_t q) {
        for (size_t r = all ? q + 1 : 0; r < refs.size(); ++r) {
            auto value = left[q].sketch.query(refs[r].sketch);
            if (!value.estimate_available || !std::isfinite(value.jaccard)) {
                throw std::runtime_error("RabbitSketch query unavailable: " + value.plan.reason);
            }
            auto const position =
                all ? q * (2 * refs.size() - q - 1) / 2 + r - q - 1 : q * refs.size() + r;
            if (position % stride == 0) {
                retained[position / stride] = {q, r, std::move(value)};
            } else if (extra_last && position + 1 == pair_count) {
                retained[slots - 1] = {q, r, std::move(value)};
            }
        }
    });
    return {std::move(retained), pair_count};
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

// Bounded resident ASCII-sequence path. Staging (parse, whitespace strip, chunking) uses the
// shared resident_sequence batch rule and stays outside every timed region; only staged batch
// bytes are consumed inside NVBench. Sketches and pair results stay corpus-sized, and every
// nonempty record with at least k bases contributes its windows exactly once.
char normalize_sequence_base(char base) {
    switch (base) {
        case 'A':
        case 'a':
            return 'A';
        case 'C':
        case 'c':
            return 'C';
        case 'G':
        case 'g':
            return 'G';
        case 'T':
        case 't':
            return 'T';
        default:
            return 'N';
    }
}

bool is_staged_whitespace(char base) {
    return base == '\n' || base == '\r' || base == ' ' || base == '\t';
}

// Groups one staged batch's chunks by genome so chunks sharing a sketch run serially while
// distinct genomes run on the existing OpenMP workers.
struct sequence_batch_groups {
    std::vector<size_t> genomes;
    std::vector<std::vector<std::pair<size_t, size_t>>> ranges;
};

sequence_batch_groups group_sequence_batch(resident_sequence::batch const& staged) {
    sequence_batch_groups result;
    for (auto const& piece : staged.chunks) {
        size_t slot = result.genomes.size();
        for (size_t i = 0; i < result.genomes.size(); ++i) {
            if (result.genomes[i] == piece.genome) {
                slot = i;
                break;
            }
        }
        if (slot == result.genomes.size()) {
            result.genomes.push_back(piece.genome);
            result.ranges.emplace_back();
        }
        result.ranges[slot].emplace_back(piece.offset, piece.size);
    }
    return result;
}

void update_sequence_groups(
    std::vector<Sketch::FastKMV>& sketches,
    resident_sequence::batch const& staged,
    sequence_batch_groups const& groups
) {
    parallel_for(groups.genomes.size(), [&](size_t slot) {
        auto& sketch = sketches[groups.genomes[slot]];
        for (auto const& [offset, size] : groups.ranges[slot]) {
            std::string sequence;
            sequence.resize(size);
            for (size_t i = 0; i < size; ++i) {
                sequence[i] = normalize_sequence_base(staged.bases[offset + i]);
            }
            sketch.update(sequence.data(), sequence.size());
        }
    });
    consumed_size = staged.bases.size();
}

// One NVBench observation: with a single sample the summary median is the raw time, so
// per-replay batch segments sum to an aligned raw total without a raw-sample export.
double measure_single_ms(options const& opts, const std::function<void()>& function) {
    options single = opts;
    single.samples = 1;
    single.warmups = 0;
    return measure(single, function).at("median_ms").get<double>();
}

json summarize_replays(std::vector<double> values) {
    if (values.empty()) {
        throw std::runtime_error("no resident replay samples to summarize");
    }
    std::sort(values.begin(), values.end());
    size_t const middle = values.size() / 2;
    json result = {
        {"samples", values.size()},
        {"median_ms",
         values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2},
        {"min_ms", values.front()},
        {"max_ms", values.back()},
        {"source", "nvbench_cpu_wall"},
    };
    if (values.size() > 1) {
        double mean = 0;
        for (auto value : values) {
            mean += value;
        }
        mean /= static_cast<double>(values.size());
        double variance = 0;
        for (auto value : values) {
            variance += (value - mean) * (value - mean);
        }
        variance /= static_cast<double>(values.size());
        double const noise = mean != 0 ? std::sqrt(variance) / std::abs(mean) : 0;
        if (std::isfinite(noise)) {
            result["relative_stddev_percent"] = noise * 100;
        }
    }
    return result;
}

// Validates staged ASCII against the native FastxReader, rebuilds per-genome stats, and
// checks whole-record sketches against the scalar hash/sort bottom-k oracle. One file at a
// time: parser storage never spans the corpus.
struct sequence_reference {
    std::vector<api::BuildStats> stats;
    std::vector<std::vector<uint64_t>> registers;
};

sequence_reference
check_sequence_reference(std::vector<std::string> const& genome_paths, options const& opts) {
    sequence_reference result;
    result.stats.resize(genome_paths.size());
    result.registers.resize(genome_paths.size());
    size_t const k = static_cast<size_t>(opts.k);
    for (size_t genome = 0; genome < genome_paths.size(); ++genome) {
        auto const& path = genome_paths[genome];
        auto loaded = cuddl::detail::load_fastx_sequence_file(path);
        if (!loaded) {
            throw std::runtime_error(path + ": " + loaded.error().message());
        }
        std::vector<std::string> parsed;
        {
            Sketch::IO::FastxReader reader(path);
            Sketch::IO::FastxRecord record;
            while (reader.next(record)) {
                parsed.push_back(record.sequence);
            }
        }
        auto const& extents = (*loaded)->extents;
        if (extents.size() != parsed.size()) {
            throw std::runtime_error("staged record count differs from FastxReader for " + path);
        }
        Sketch::FastKMV sketch(opts.sketch_size, opts.k, opts.seed);
        auto& stats = result.stats[genome];
        std::vector<uint64_t> keys;
        for (size_t record = 0; record < extents.size(); ++record) {
            std::string sequence;
            for (auto cursor = extents[record].begin; cursor != extents[record].end; ++cursor) {
                if (is_staged_whitespace(*cursor)) {
                    continue;
                }
                sequence.push_back(normalize_sequence_base(*cursor));
            }
            std::string expected;
            for (char base : parsed[record]) {
                if (is_staged_whitespace(base)) {
                    continue;
                }
                expected.push_back(normalize_sequence_base(base));
            }
            if (sequence != expected) {
                throw std::runtime_error("staged ASCII differs from FastxReader for " + path);
            }
            ++stats.records;
            stats.input_bases += parsed[record].size();
            if (!sequence.empty()) {
                sketch.update(sequence.data(), sequence.size());
            }
            if (sequence.size() < k) {
                continue;
            }
            stats.candidate_kmers += sequence.size() - k + 1;
            size_t ambiguous = 0;
            for (size_t i = 0; i < k; ++i) {
                ambiguous += sequence[i] == 'N';
            }
            for (size_t start = 0; start + k <= sequence.size(); ++start) {
                if (ambiguous != 0) {
                    ++stats.skipped_ambiguous_kmers;
                } else {
                    ++stats.accepted_kmers;
                }
                if (start + k < sequence.size()) {
                    ambiguous -= sequence[start] == 'N';
                    ambiguous += sequence[start + k] == 'N';
                }
            }
            Sketch::Rank::CanonicalKmerIterator windows(
                sequence.data(), sequence.size(), static_cast<uint8_t>(opts.k)
            );
            uint64_t code = 0;
            while (windows.next(code)) {
                keys.push_back(Sketch::Rank::RankStream::fmix64(code, opts.seed) >> 11);
            }
        }
        sketch.finalize();
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        keys.resize(std::min(keys.size(), static_cast<size_t>(opts.sketch_size)));
        auto const* registers = sketch.getRegisters();
        if (sketch.size() != keys.size() || !std::equal(keys.begin(), keys.end(), registers)) {
            throw std::runtime_error(
                "sequence FastKMV differs from scalar bottom-k oracle: " + path
            );
        }
        result.registers[genome].assign(registers, registers + sketch.size());
    }
    return result;
}

struct sequence_timings {
    json timings;
    size_t batches = 0;
};

sequence_timings sequence_resident_timings(
    options const& opts,
    api::SketchConfig const& cfg,
    uint64_t expected_pairs,
    json const& rows
) {
    bool const all = opts.topology == "all-to-all";
    if (opts.resident_bytes < static_cast<size_t>(opts.k)) {
        throw std::runtime_error("--resident-bytes must be at least k to stage overlapping chunks");
    }
    std::vector<std::string> genome_paths = opts.references;
    if (!all) {
        genome_paths.insert(genome_paths.end(), opts.queries.begin(), opts.queries.end());
    }
    auto const reference_size = opts.references.size();
    auto const n = reference_size;
    auto const q = all ? n : opts.queries.size();
    if (n && q > std::numeric_limits<size_t>::max() / n) {
        throw std::runtime_error("resident result size overflow");
    }
    auto const reference = check_sequence_reference(genome_paths, opts);
    uint32_t const k = static_cast<uint32_t>(opts.k);
    std::vector<Sketch::FastKMV> staged;
    staged.reserve(genome_paths.size());
    for (size_t i = 0; i < genome_paths.size(); ++i) {
        staged.emplace_back(opts.sketch_size, opts.k, opts.seed);
    }
    // Untimed chunked build over the shared batch stream; register equality against the
    // whole-record reference proves the k-1 overlap loses and duplicates no window.
    // A one-batch corpus keeps its staged batch for the timed replays below; a
    // multi-batch corpus keeps streaming so staging stays bounded by the cap.
    resident_sequence::batch retained;
    size_t seen = 0;
    size_t const batches = resident_sequence::for_each_batch(
        genome_paths, k, opts.resident_bytes, [&](resident_sequence::batch& batch) {
            update_sequence_groups(staged, batch, group_sequence_batch(batch));
            // consume() passes a nonconst batch; move the first one into retained
            // storage. flush() clears the moved-from vectors safely.
            if (seen++ == 0) {
                retained = std::move(batch);
            }
        }
    );
    if (batches != 1) {
        retained = resident_sequence::batch{};
    }
    sequence_batch_groups retained_groups;
    if (batches == 1) {
        retained_groups = group_sequence_batch(retained);
    }
    for (size_t i = 0; i < staged.size(); ++i) {
        staged[i].finalize();
        auto const* registers = staged[i].getRegisters();
        if (staged[i].size() != reference.registers[i].size() ||
            !std::equal(reference.registers[i].begin(), reference.registers[i].end(), registers)) {
            throw std::runtime_error(
                "batched ASCII chunks differ from whole-record sketch: " + genome_paths[i]
            );
        }
    }
    std::vector<double> expected(n * q);
    parallel_for(expected.size(), [&](size_t position) {
        auto const i = position / n, j = position % n;
        if (all && j <= i) {
            return;
        }
        expected[position] = staged[all ? i : n + i].jaccard(staged[j]);
    });
    std::vector<Sketch::FastKMV> sketches;
    sketches.reserve(genome_paths.size());
    for (size_t i = 0; i < genome_paths.size(); ++i) {
        sketches.emplace_back(opts.sketch_size, opts.k, opts.seed);
    }
    std::vector<double> cardinalities(genome_paths.size()), similarities(n * q);
    auto reset = [&] {
        parallel_for(sketches.size(), [&](size_t i) { sketches[i].clear(); });
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
            if (all && j <= i) {
                return;
            }
            similarities[position] = sketches[all ? i : n + i].jaccard(sketches[j]);
        });
        consumed_size = similarities.size();
    };
    // Timed replays sum the per-replay batch segments (each a single raw NVBench observation),
    // then summarize across replays while warmup replays are discarded. A one-batch corpus
    // reuses its retained staging; multi-batch corpora restage outside the timers per replay
    // so staging stays bounded by the cap. Downstream stages stay whole-corpus singles.
    std::vector<double> reset_ms, construct_ms, finalize_ms, cardinality_ms, search_ms, total_ms;
    for (int replay = -opts.warmups; replay < opts.samples; ++replay) {
        double const reset_time = measure_single_ms(opts, reset);
        double construct_time = 0;
        if (batches == 1) {
            construct_time += measure_single_ms(opts, [&] {
                update_sequence_groups(sketches, retained, retained_groups);
            });
        } else {
            size_t const observed = resident_sequence::for_each_batch(
                genome_paths, k, opts.resident_bytes, [&](resident_sequence::batch const& batch) {
                    auto const groups = group_sequence_batch(batch);
                    construct_time += measure_single_ms(opts, [&] {
                        update_sequence_groups(sketches, batch, groups);
                    });
                }
            );
            if (observed != batches) {
                throw std::runtime_error("resident batch plan changed between replays");
            }
        }
        double const finalize_time = measure_single_ms(opts, finalize);
        double const cardinality_time = measure_single_ms(opts, cardinality);
        double const search_time = measure_single_ms(opts, compare);
        if (similarities != expected) {
            throw std::runtime_error("resident replay changed pair results");
        }
        if (replay >= 0) {
            reset_ms.push_back(reset_time);
            construct_ms.push_back(construct_time);
            finalize_ms.push_back(finalize_time);
            cardinality_ms.push_back(cardinality_time);
            search_ms.push_back(search_time);
            total_ms.push_back(
                reset_time + construct_time + finalize_time + cardinality_time + search_time
            );
        }
    }
    for (size_t i = 0; i < sketches.size(); ++i) {
        auto const* registers = sketches[i].getRegisters();
        if (sketches[i].size() != reference.registers[i].size() ||
            !std::equal(reference.registers[i].begin(), reference.registers[i].end(), registers)) {
            throw std::runtime_error("timed resident replay differs from untimed chunks");
        }
    }
    json timings;
    timings["resident_total_wall"] = summarize_replays(total_ms);
    timings["resident_reset_wall"] = summarize_replays(reset_ms);
    timings["resident_construct_wall"] = summarize_replays(construct_ms);
    timings["resident_finalize_wall"] = summarize_replays(finalize_ms);
    timings["resident_cardinality_wall"] = summarize_replays(cardinality_ms);
    timings["resident_search_wall"] = summarize_replays(search_ms);
    collection resident_refs, resident_queries;
    resident_refs.reserve(reference_size);
    if (!all) {
        resident_queries.reserve(opts.queries.size());
    }
    for (size_t i = 0; i < genome_paths.size(); ++i) {
        api::BuiltSketch sketch = api::BuiltSketch::fromFastKMV(std::move(sketches[i]));
        api::BuildResult built(
            std::move(sketch), "genome", cfg, reference.stats[i], genome_paths[i], ""
        );
        if (i < reference_size) {
            resident_refs.push_back(std::move(built));
        } else {
            resident_queries.push_back(std::move(built));
        }
    }
    auto const resident = search(resident_refs, resident_queries, all, opts.match_rows);
    if (resident.pairs != expected_pairs ||
        measurements(resident_refs, resident_queries, resident.matches) != rows) {
        throw std::runtime_error("resident sequence chunks differ from file ingest results");
    }
    return {std::move(timings), batches};
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
    auto const searched = search(refs, queries, all, opts.match_rows);
    auto const& matches = searched.matches;
    size_t const match_rows_emitted = matches.size();
    auto rows = measurements(refs, queries, matches);
    std::string resident_scope = "all_files";
    bool sequence_oracle_equal = false;
    size_t resident_batches = 0;
    json chunked_timings = json::object();
    if (streamed) {
        // Full-corpus check: bounded chunked resident sketches must match file ingest exactly.
        auto resident = sequence_resident_timings(opts, cfg, searched.pairs, rows);
        chunked_timings = std::move(resident.timings);
        resident_batches = resident.batches;
        sequence_oracle_equal = true;
    } else {
        auto reference_records = parse(opts.references);
        auto query_records = parse(opts.queries);
        auto resident_refs = construct(reference_records, cfg);
        auto resident_queries = construct(query_records, cfg);
        auto const resident = search(resident_refs, resident_queries, all, opts.match_rows);
        if (resident.pairs != searched.pairs ||
            measurements(resident_refs, resident_queries, resident.matches) != rows) {
            throw std::runtime_error("resident and streaming RabbitSketch results differ");
        }
    }

    json timings = measure_pipeline(opts.samples, opts.warmups, [&](auto mark) {
        auto r = build_files(opts.references, cfg);
        auto q = build_queries();
        mark();
        auto hits = search(r, q, all, opts.match_rows);
        auto output = measurements(r, q, hits.matches).dump();
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
        auto hits = search(refs, queries, all, opts.match_rows);
        consumed_size = hits.matches.size();
    });
    timings["metrics_and_serialize"] = measure(opts, [&] {
        auto output = measurements(refs, queries, matches).dump();
        consumed_size = output.size();
    });

    auto const& runtime = Sketch::Runtime::runtimeInfo();
    if (!streamed) {
        timings.update(resident_timings(opts));
    } else {
        timings.update(chunked_timings);
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
              {"resident_input", streamed ? "sequence_ascii" : "packed_u64_actg_max"},
              {"resident_output", "host_cardinalities_and_jaccard_matrix"},
              {"resident_minimum_matches", 0},
              {"resident_batch_bytes", streamed ? opts.resident_bytes : 0},
              {"resident_batches", resident_batches},
              {"resident_timing_scope",
               streamed ? "batched_resident_segments" : "resident_pipeline"},
              {"resident_construct_path",
               streamed ? "ascii_chunks+FastKMV-update" : "packed_u64+FastKMV-updatePacked"},
              {"packed_input_patch", "FastKMV::updatePacked+clear"},
              {"native_arch", !runtime.portable_baseline},
              {"simd_byte_path", runtime.selected_byte_path},
              {"simd_u64_path", runtime.selected_u64_path},
              {"simd_rank_path", runtime.selected_rank_path},
              {"simd_override", runtime.environment_override},
              {"simd_override_honored", runtime.environment_override_honored}}},
            {"metrics",
             {{"exhaustive_pairs", searched.pairs},
              {"match_rows_total", searched.pairs},
              {"match_rows_emitted", match_rows_emitted},
              {"resident_streaming_equal", true},
              {"resident_streaming_scope", resident_scope},
              {"resident_sequence_chunk_oracle_equal", sequence_oracle_equal},
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
           "packed: add the record and packed-input stages. sequence: bounded ASCII-chunk "
           "resident path limited by --resident-bytes, required for a large corpus"
    )
        ->check(CLI::IsMember({"packed", "sequence"}));
    app.add_option(
           "--resident-bytes",
           opts.resident_bytes,
           "Staged ASCII byte cap for --ingest sequence, including k-1 overlap; must be at "
           "least k. Sketches and pair results stay corpus-sized"
    )
        ->check(CLI::Range(size_t{0}, std::numeric_limits<size_t>::max()));
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
    if (opts.ingest == "sequence" && opts.resident_bytes < static_cast<size_t>(opts.k)) {
        throw std::runtime_error("--resident-bytes must be at least k for --ingest sequence");
    }
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
