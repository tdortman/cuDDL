#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <CLI/CLI.hpp>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuddl/a48.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>
#include <cuddl/reference_database_file.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.cuh"
#include "resident_sequence_batches.hpp"
#include "result_json.hpp"

namespace {
constexpr uint32_t k = 25;
constexpr size_t buckets = 4096;
using sketch = cuddl::sketch<k, buckets>;
using database = cuddl::reference_database<k, buckets>;

struct options {
    std::vector<std::string> references, queries;
    std::string output, name = "cuDDL pipeline";
    std::string rows = "compact", index = "sparse", topology = "batch";
    std::string ingest = "packed";
    uint32_t minimum_matches = 5, indexed_buckets = buckets / 2, key_bits = 15;
    unsigned workers = 0;
    size_t resident_bytes = 0;  // 0 selects the batch budget from free GPU memory.
    bool resident_plan = false;
    int samples = 20, warmups = 3;
    size_t oracle_pairs = 1000000, match_rows = 20000, dataset_hashes = 8,
           all_to_all_pairs = 50000000;
};

struct winner_score {
    __device__ uint16_t operator()(uint32_t reg) const {
        return cuddl::detail::winner(reg);
    }
};

json measure(
    options const& opts,
    std::string const& name,
    bool host,
    std::function<void(cuda::stream_ref)> function,
    std::function<void(cuda::stream_ref)> prepare = {},
    json* wall = nullptr,
    std::function<void()> finish = {},
    cuda::stream_ref const* external_stream = nullptr
) {
    // CPU-only NVBench timing supplies wall time for host work and synchronized E2E runs.
    auto run = [&](nvbench::state& state, nvbench::type_list<>) {
        if (external_stream) {
            state.set_cuda_stream(nvbench::make_cuda_stream_view(external_stream->get()));
        }
        // Stateful segments must execute exactly once; corpus replays provide their warmups.
        state.set_run_once(opts.samples == 1 && opts.warmups == 0);
        if (host) {
            state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
                timer.start();
                function(cuda::stream_ref{cudaStream_t{nullptr}});
                timer.stop();
            });
        } else {
            state.exec(
                nvbench::exec_tag::sync | nvbench::exec_tag::timer,
                [&](nvbench::launch& launch, auto& timer) {
                    auto const stream = cuda::stream_ref{launch.get_stream()};
                    if (prepare) {
                        prepare(stream);
                    }
                    timer.start();
                    function(stream);
                    timer.stop();
                }
            );
        }
        // NVBench destroys the state stream when this kernel-generator returns.
        if (finish) finish();
    };
    nvbench::benchmark<decltype(run)> benchmark(run);
    benchmark.set_name(name)
        .set_stopping_criterion("sample-count")
        .set_min_samples(opts.samples)
        .set_criterion_param_int64("target-samples", opts.samples)
        .set_cold_warmup_runs(opts.warmups)
        .set_skip_batched(true);
    if (host) {
        benchmark.set_is_cpu_only(true);
    } else {
        benchmark.add_device(cuda::devices[0].get());
    }
    benchmark.run();
    auto const& state = benchmark.get_states().front();
    if (state.is_skipped()) {
        throw std::runtime_error(state.get_skip_reason());
    }
    std::string const base = host ? "nv/cpu_only" : "nv/cold";
    auto summarize = [&](bool cpu) {
        auto const prefix = base + (cpu ? "/time/cpu" : "/time/gpu");
        auto result = json{
            {"samples", state.get_summary(base + "/sample_size").get_int64("value")},
            {"median_ms", state.get_summary(prefix + "/median").get_float64("value") * 1000},
            {"min_ms", state.get_summary(prefix + "/min").get_float64("value") * 1000},
            {"max_ms", state.get_summary(prefix + "/max").get_float64("value") * 1000},
            {"source", cpu ? "nvbench_cpu_wall" : "nvbench_gpu_events"},
        };
        auto const noise = state.get_summary(prefix + "/stdev/relative").get_float64("value");
        if (std::isfinite(noise)) {
            result["relative_stddev_percent"] = noise * 100;
        }
        return result;
    };
    if (wall) {
        *wall = summarize(true);
    }
    return summarize(host);
}

using parsed_files = std::vector<cuddl::fasta_parse_result>;
parsed_files parse(std::vector<std::string> const& paths) {
    parsed_files files;
    for (auto const& path : paths) {
        files.push_back(CUDDL_UNWRAP(cuddl::parse_fasta_file(path, k)));
    }
    return files;
}

// One genome per input file, streamed through the tile builder: bases are encoded and rolled
// on the GPU, so neither a host nor a device k-mer array is materialized. Footprint is rows
// (genomes * buckets) instead of bases, which is what makes a full RefSeq collection fit.
struct genome_rows {
    // One contiguous run per genome: buckets registers followed by the saturation word.
    std::vector<uint32_t> registers;
    std::vector<uint32_t> saturation;
    uint64_t input_bytes = 0;
};

genome_rows stream_genomes(
    std::vector<std::string> const& paths,
    options const& opts,
    cuda::stream_ref stream
) {
    std::vector<std::filesystem::path> files{paths.begin(), paths.end()};
    auto file = CUDDL_UNWRAP(
        (cuddl::reference_database_file::build<k, buckets>(files, stream, opts.workers))
    );
    genome_rows result;
    auto const count = file.saturation().size();
    result.saturation.assign(file.saturation().begin(), file.saturation().end());
    result.registers.resize(count * (buckets + 1));
    for (size_t i = 0; i < count; ++i) {
        std::copy_n(
            file.rows().data() + i * buckets, buckets, result.registers.data() + i * (buckets + 1)
        );
        result.registers[i * (buckets + 1) + buckets] = file.saturation()[i];
    }
    for (auto const& path : paths) {
        std::error_code error;
        auto const size = std::filesystem::file_size(path, error);
        if (!error) {
            result.input_bytes += size;
        }
    }
    return result;
}

// Read-only rows of a streamed store, for the library's batch operations.
cuddl::device_span<uint32_t const> stored_rows(cuddl::device_span<uint32_t> rows) noexcept {
    return {rows.data(), rows.size()};
}

// Each input file is a genome. Records within a file retain parser boundary semantics.
struct collection {
    std::vector<cuda::device_buffer<uint64_t>> inputs;
    // Streamed register source: (buckets + 1) words per genome, kept alive for the copies.
    cuda::device_buffer<uint32_t> registers;
    // Rows this collection owns inside a shared streamed store; empty for per-file sketches.
    cuddl::device_span<uint32_t> store;
    std::vector<sketch> sketches;
    cuda::device_buffer<uint16_t> scores, counts;
    cuda::device_buffer<uint32_t> packed, saturated;
    cuda::device_buffer<uint64_t> empty;
    cuda::device_buffer<double> cardinalities;

    collection(size_t genomes, cuda::stream_ref stream, bool compact_rows, bool packed_rows)
        : scores(
              cuda::make_device_buffer<uint16_t>(
                  stream,
                  stream.device(),
                  compact_rows ? genomes * buckets : 0,
                  cuda::no_init
              )
          ),
          counts(
              cuda::make_device_buffer<uint16_t>(
                  stream,
                  stream.device(),
                  genomes * buckets,
                  cuda::no_init
              )
          ),
          packed(
              cuda::make_device_buffer<uint32_t>(
                  stream,
                  stream.device(),
                  packed_rows ? genomes * buckets : 0,
                  cuda::no_init
              )
          ),
          saturated(
              cuda::make_device_buffer<uint32_t>(stream, stream.device(), genomes, cuda::no_init)
          ),
          empty(
              cuda::make_device_buffer<uint64_t>(stream, stream.device(), genomes, cuda::no_init)
          ),
          cardinalities(
              cuda::make_device_buffer<double>(stream, stream.device(), genomes, cuda::no_init)
          ),
          registers(cuda::make_device_buffer<uint32_t>(stream, stream.device(), 0, cuda::no_init)) {
    }

    collection(
        parsed_files const& files,
        cuda::stream_ref stream,
        bool compact_rows = true,
        bool packed_rows = true
    )
        : collection(files.size(), stream, compact_rows, packed_rows) {
        for (auto const& file : files) {
            inputs.push_back(
                cuda::make_device_buffer<uint64_t>(stream, stream.device(), file.kmers)
            );
            sketches.emplace_back(stream);
        }
    }

    // Rows are copied into sketches; the caller keeps @p rows alive until the stream completes.
    collection(
        genome_rows const& rows,
        cuda::stream_ref stream,
        bool compact_rows = true,
        bool packed_rows = true
    )
        : collection(rows.saturation.size(), stream, compact_rows, packed_rows) {
        if (saturated.size()) {
            cuda::copy_bytes(stream, rows.saturation, saturated);
        }
        registers = cuda::make_device_buffer<uint32_t>(stream, stream.device(), rows.registers);
        for (size_t i = 0; i < rows.saturation.size(); ++i) {
            sketches.emplace_back(stream);
            CUDDL_UNWRAP(
                sketches[i].assign_async(
                    {registers.data() + i * (buckets + 1), buckets + 1}, stream
                )
            );
        }
    }
    /// @brief Stored sketches this collection contributes: shared store rows or per-file sketches.
    [[nodiscard]] size_t rows() const noexcept {
        return store.size() ? store.size() / (buckets + 1) : sketches.size();
    }

    /// @brief Binds the shared store rows written by the streamed construct kernel.
    void bind_store(cuddl::device_span<uint32_t> rows) noexcept {
        store = rows;
    }

    void clear(cuda::stream_ref stream) {
        if (store.size()) {
            cuda::fill_bytes(stream, cuda::std::span{store.data(), store.size()}, 0);
            return;
        }
        for (auto const& s : sketches) {
            CUDDL_UNWRAP(s.clear_async(stream));
        }
    }
    void add(cuda::stream_ref stream, bool chunked = false) {
        if (inputs.size() != sketches.size()) {
            throw std::runtime_error("streamed collection has no k-mer input to add");
        }
        for (size_t i = 0; i < sketches.size(); ++i) {
            auto const n = inputs[i].size();
            if (chunked && n > 1) {
                CUDDL_UNWRAP(sketches[i].add_async({inputs[i].data(), n / 2}, stream));
                CUDDL_UNWRAP(sketches[i].add_async({inputs[i].data() + n / 2, n - n / 2}, stream));
            } else {
                CUDDL_UNWRAP(sketches[i].add_async(inputs[i], stream));
            }
        }
    }
    void extract_scores(cuda::stream_ref stream) {
        if (!scores.size()) {
            return;
        }
        if (store.size()) {
            CUDDL_UNWRAP(
                cuddl::extract_scores_batch_async<buckets>(stored_rows(store), scores, stream)
            );
            return;
        }
        for (size_t i = 0; i < sketches.size(); ++i) {
            CUDDL_CUDA_CALL(
                cub::DeviceTransform::Transform(
                    sketches[i].data().data(),
                    scores.data() + i * buckets,
                    buckets,
                    winner_score{},
                    stream
                )
            );
        }
    }
    void extract_packed(cuda::stream_ref stream) {
        if (!packed.size()) {
            return;
        }
        if (store.size()) {
            CUDDL_UNWRAP(
                cuddl::extract_packed_rows_batch_async<buckets>(stored_rows(store), packed, stream)
            );
            return;
        }
        for (size_t i = 0; i < sketches.size(); ++i) {
            cuda::copy_bytes(
                stream,
                cuda::std::span{sketches[i].data().data(), buckets},
                cuda::std::span{packed.data() + i * buckets, buckets}
            );
        }
    }
    void winner_counts(cuda::stream_ref stream) {
        if (store.size()) {
            CUDDL_UNWRAP(
                cuddl::winner_counts_batch_async<buckets>(
                    stored_rows(store), counts, saturated, stream
                )
            );
            return;
        }
        for (size_t i = 0; i < sketches.size(); ++i) {
            CUDDL_UNWRAP(
                sketches[i].winner_counts_async(
                    counts.data() + i * buckets, saturated.data() + i, stream
                )
            );
        }
    }
    void extract(cuda::stream_ref stream) {
        if (scores.size()) extract_scores(stream);
        if (packed.size()) extract_packed(stream);

        winner_counts(stream);
    }
    void cardinality(cuda::stream_ref stream) {
        if (store.size()) {
            CUDDL_UNWRAP(
                cuddl::cardinality_batch_async<buckets>(
                    stored_rows(store), empty, cardinalities, stream
                )
            );
            return;
        }
        for (size_t i = 0; i < sketches.size(); ++i) {
            CUDDL_UNWRAP(
                sketches[i].cardinality_async(empty.data() + i, cardinalities.data() + i, stream)
            );
        }
    }
};

template <typename T>
std::vector<T> download(cuda::device_buffer<T> const& data, cuda::stream_ref stream) {
    std::vector<T> host(data.size());
    cuda::copy_bytes(stream, data, host);
    stream.sync();
    return host;
}

cuddl::score_compatibility compatibility(options const& opts) {
    auto result = cuddl::score_compatibility::current<k, buckets>();
    result.indexed_bucket_count = opts.indexed_buckets;
    result.key_mask = static_cast<uint16_t>((uint32_t{1} << opts.key_bits) - 1);
    return result;
}
database build(collection const& refs, options const& opts, cuda::stream_ref stream, bool indexed) {
    auto const compat =
        indexed ? compatibility(opts) : cuddl::score_compatibility::current<k, buckets>();
    auto const storage =
        opts.index == "dense" ? cuddl::index_storage::dense : cuddl::index_storage::sparse;
    if (opts.rows == "packed") {
        if (indexed) {
            return CUDDL_UNWRAP(
                database::build_indexed_async(refs.packed, refs.saturated, compat, stream, storage)
            );
        }
        return CUDDL_UNWRAP(database::build_async(refs.packed, refs.saturated, compat, stream));
    }
    if (indexed) {
        return CUDDL_UNWRAP(database::build_indexed_async(refs.scores, compat, stream, storage));
    }
    return CUDDL_UNWRAP(database::build_async(refs.scores, compat, stream));
}

struct search_buffers {
    cuda::device_buffer<uint8_t> workspace;
    cuda::device_buffer<cuddl::batch_search_result> results;
    cuda::device_buffer<uint32_t> count, matches;

    search_buffers(
        database const& db,
        uint32_t queries,
        cuda::stream_ref stream,
        std::string const& topology = "both"
    )
        : workspace(stream, cuda::device_default_memory_pool(stream.device())),
          results(stream, cuda::device_default_memory_pool(stream.device())),
          count(cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, cuda::no_init)),
          matches(stream, cuda::device_default_memory_pool(stream.device())) {
        size_t workspace_bytes = 0;
        uint32_t size = 1;
        auto include = [&](cuddl::batch_search_requirements const& requirement) {
            workspace_bytes = std::max(workspace_bytes, requirement.workspace_bytes);
            size = std::max(size, requirement.maximum_pair_count);
        };
        if (topology != "all-to-all") {
            include(CUDDL_UNWRAP(db.indexed_batch_search_requirements(queries, stream)));
            include(CUDDL_UNWRAP(db.batch_search_requirements(queries, stream)));
        }
        if (topology != "batch") {
            include(CUDDL_UNWRAP(db.indexed_all_to_all_search_requirements(stream)));
            include(CUDDL_UNWRAP(db.all_to_all_search_requirements(stream)));
        }
        workspace = cuda::make_device_buffer<uint8_t>(
            stream, stream.device(), workspace_bytes, cuda::no_init
        );
        results = cuda::make_device_buffer<cuddl::batch_search_result>(
            stream, stream.device(), size, cuda::no_init
        );
        matches = cuda::make_device_buffer<uint32_t>(stream, stream.device(), size, cuda::no_init);
    }
};

struct host_results {
    std::vector<cuddl::batch_search_result> rows;
    std::vector<uint32_t> matches;
};

void search(
    database const& db,
    collection const& queries,
    search_buffers& buffers,
    options const& opts,
    cuda::stream_ref stream,
    bool indexed,
    bool all,
    host_results* output = nullptr,
    std::function<void(uint32_t)> device_consume = {}
) {
    auto consume = [&](uint32_t capacity) {
        if (device_consume) device_consume(capacity);
        if (!output) {
            return;  // A device consumer can retain results without host downloads.
        }
        auto count = download(buffers.count, stream).front();
        if (!(count <= buffers.results.size())) {
            throw std::runtime_error("invalid search result count");
        }
        if (count == 0) {
            return;
        }
        auto const offset = output->rows.size();
        output->rows.resize(offset + count);
        output->matches.resize(offset + count);
        cuda::copy_bytes(
            stream,
            cuda::std::span{buffers.results.data(), size_t{count}},
            cuda::std::span{output->rows.data() + offset, size_t{count}}
        );
        cuda::copy_bytes(
            stream,
            cuda::std::span{buffers.matches.data(), size_t{count}},
            cuda::std::span{output->matches.data() + offset, size_t{count}}
        );
        stream.sync();  // Tile storage is reused by the next callback.
    };
    auto const config = cuddl::indexed_search_options{.minimum_matches = opts.minimum_matches};
    if (all) {
        if (indexed) {
            CUDDL_UNWRAP(db.search_all_to_all_indexed_async(
                buffers.workspace,
                buffers.results,
                buffers.count,
                consume,
                buffers.matches,
                config,
                stream
            ));
        } else {
            CUDDL_UNWRAP(db.search_all_to_all_async(
                buffers.workspace, buffers.results, buffers.count, consume, buffers.matches, stream
            ));
        }
    } else {
        if (indexed) {
            CUDDL_UNWRAP(db.search_batch_indexed_async(
                queries.scores,
                compatibility(opts),
                0,
                buffers.workspace,
                buffers.results,
                buffers.count,
                consume,
                buffers.matches,
                config,
                stream
            ));
        } else {
            CUDDL_UNWRAP(db.search_batch_async(
                queries.scores,
                compatibility(opts),
                0,
                buffers.workspace,
                buffers.results,
                buffers.count,
                consume,
                buffers.matches,
                stream
            ));
        }
    }
}

cuddl::pairwise_counts oracle(uint16_t const* left, uint16_t const* right) {
    cuddl::pairwise_counts c;
    for (size_t i = 0; i < buckets; ++i) {
        if (left[i] == 0 && right[i] == 0) {
            ++c.both_empty;
        } else if (left[i] < right[i]) {
            ++c.lower;
        } else if (left[i] > right[i]) {
            ++c.higher;
        } else {
            ++c.equal;
        }
    }
    return c;
}
uint32_t hits(uint16_t const* left, uint16_t const* right, options const& opts) {
    auto const mask = compatibility(opts).key_mask;
    uint32_t count = 0;
    for (size_t i = 0; i < opts.indexed_buckets; ++i) {
        count += left[i] != 0 && right[i] != 0 && (left[i] & mask) == (right[i] & mask);
    }
    return count;
}

struct validation {
    size_t candidates = 0, expected = 0, checked = 0;
};

// Checks every expected candidate's presence and order, then compares a deterministic spread
// against the scalar oracle. @p oracle_limit caps the expensive per-pair oracle work (0 checks
// every pair); the first and last reference of each query are always compared.
validation validate(
    host_results const& output,
    std::vector<uint16_t> const& refs,
    std::vector<uint16_t> const& queries,
    options const& opts,
    bool indexed,
    bool all,
    size_t oracle_limit
) {
    auto const& left = all ? refs : queries;
    auto const reference_count = refs.size() / buckets;
    auto const query_count = left.size() / buckets;
    size_t const pair_space = all ? (query_count > 1 ? query_count * (query_count - 1) / 2 : 0)
                                  : query_count * reference_count;
    size_t const limit = oracle_limit ? oracle_limit : std::numeric_limits<size_t>::max();
    size_t const stride = pair_space > limit ? (pair_space + limit - 1) / limit : 1;
    validation result;
    size_t cursor = 0;
    for (size_t q = 0; q < query_count; ++q) {
        for (size_t r = all ? q + 1 : 0; r < reference_count; ++r) {
            auto* a = left.data() + q * buckets;
            auto* b = refs.data() + r * buckets;
            bool const sampled = result.candidates % stride == 0 || r + 1 == reference_count;
            uint32_t match_count = 0;
            if (opts.minimum_matches != 0 || sampled) {
                match_count = hits(a, b, opts);
            }
            ++result.candidates;
            if (indexed && match_count < opts.minimum_matches) {
                continue;
            }
            ++result.expected;
            if (!(cursor < output.rows.size())) {
                throw std::runtime_error("search omitted an expected pair");
            }
            auto const& row = output.rows[cursor];
            if (!(row.query_id == q && row.reference_id == r)) {
                throw std::runtime_error("search IDs/order differ from oracle");
            }
            if (sampled) {
                if (!(row.summary.counts == oracle(a, b))) {
                    throw std::runtime_error("search counts differ from scalar oracle");
                }
                if (!(output.matches[cursor] ==
                      (indexed ? match_count : row.summary.counts.equal))) {
                    throw std::runtime_error("search match diagnostics differ from oracle");
                }
                ++result.checked;
            }
            ++cursor;
        }
    }
    if (!(cursor == output.rows.size())) {
        throw std::runtime_error("search returned unexpected pairs");
    }
    return result;
}

json metrics(cuddl::pairwise_summary const& summary) {
    json result = {
        {"lower", summary.counts.lower},
        {"equal", summary.counts.equal},
        {"higher", summary.counts.higher},
        {"both_empty", summary.counts.both_empty}
    };
    auto add = [&](char const* name, std::optional<double> value) {
        // Scalar-map schema has no null type. Validity is explicit; absent estimates stay absent.
        result[std::string(name) + "_defined"] = value.has_value();
        if (value) {
            result[name] = *value;
        }
    };
    add("wkid", sketch::wkid(summary));
    add("ani", sketch::ani(summary));
    add("containment", sketch::containment(summary));
    add("completeness", sketch::completeness(summary));
    return result;
}

json collection_metrics(collection const& group, cuda::stream_ref stream) {
    json output = json::array();
    auto all_counts = download(group.counts, stream);
    auto saturation = download(group.saturated, stream);
    for (size_t i = 0; i < group.sketches.size(); ++i) {
        auto const& s = group.sketches[i];
        auto estimates = CUDDL_UNWRAP(s.hybrid_cardinality(stream));
        auto const saturated = saturation[i] != 0;
        auto value = json{
            {"cardinality", CUDDL_UNWRAP(s.cardinality(stream))},
            {"hybrid_bbtools", estimates.bbtools},
            {"hybrid_paper", estimates.paper},
            {"hybrid_lc", estimates.lc},
            {"hybrid_dlc", estimates.dlc},
            {"hybrid_mean_m_raw", estimates.mean_m_raw},
            {"saturated", saturated}
        };
        std::map<uint16_t, uint32_t> histogram;
        for (size_t j = 0; j < buckets; ++j) {
            ++histogram[all_counts[i * buckets + j]];
        }
        for (auto [c, n] : histogram) {
            value["q_" + std::to_string(c)] = n;
        }
        value["q_0"] = histogram[0];
        value["q_65535"] = histogram[65535];
        output.push_back(std::move(value));
    }
    return output;
}

// Counts FASTX record extents one file at a time for the descriptor bound. No base
// bytes are copied; only the current file's parser storage is alive at a time.
size_t count_sequence_records(std::vector<std::string> const& paths) {
    size_t records = 0;
    for (auto const& path : paths) {
        auto loaded = cuddl::detail::load_fastx_sequence_file(path);
        if (!loaded) {
            throw std::runtime_error("cannot load FASTX file: " + path);
        }
        records += (*loaded)->extents.size();
    }
    return records;
}

struct resident_budget {
    size_t cap = 0;  // Effective ASCII staging bytes per batch.
    size_t free_bytes = 0;
    size_t reusable_pool_bytes = 0;
    size_t available_bytes = 0;
    size_t future_peak_bytes = 0;  // Index-rebuild transients incl. exact CUB scratch.
    size_t reserve_bytes = 0;      // Automatic sizing keeps 10% availability headroom.
    size_t total_bytes = 0;
    size_t total_records = 0;
    size_t metadata_bytes = 0;  // 24-byte chunk descriptors bound at the resolved cap.
};

// Chunk-descriptor bound at @p cap: every staged piece holds at least k bytes, and
// split pieces beyond record starts are bounded by the header's per-chunk UINT32
// window cap, so the staged count never exceeds either term.
size_t batch_metadata_bound(size_t cap, size_t records) {
    size_t const dense = cap / k;
    if (records >= dense) {
        return dense;  // Records alone exceed the per-byte bound; sums below stay safe.
    }
    // records < dense <= SIZE_MAX/k here, so records + 2 + splits cannot wrap.
    size_t const splits = cap / (static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - k + 1);
    return std::min(dense, records + 2 + splits);
}

// Exact CUB scratch for the index-rebuild transients, queried without launching.
size_t index_rebuild_scratch(
    options const& opts,
    uint64_t postings,
    uint64_t cells,
    uint32_t references,
    uint32_t indexed_buckets,
    cuda::stream_ref stream
) {
    if (opts.index == "sparse") {
        size_t bytes = 0;
        auto const segment_offsets = cuda::make_transform_iterator(
            cuda::make_counting_iterator(uint32_t{0}),
            cuddl::detail::sparse_segment_offset{references}
        );
        CUDDL_CUDA_CALL(
            cub::DeviceSegmentedSort::SortPairs(
                nullptr,
                bytes,
                static_cast<uint16_t const*>(nullptr),
                static_cast<uint16_t*>(nullptr),
                static_cast<uint32_t const*>(nullptr),
                static_cast<uint32_t*>(nullptr),
                static_cast<int64_t>(postings),
                static_cast<int64_t>(indexed_buckets),
                segment_offsets,
                segment_offsets + 1,
                stream.get()
            )
        );
        return bytes;
    }
    size_t bytes = 0;
    CUDDL_CUDA_CALL(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            bytes,
            static_cast<uint32_t const*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            static_cast<int64_t>(cells + 1),
            stream.get()
        )
    );
    return bytes;
}

// Device bytes a fresh index generation needs while the resident generation is
// still alive: new rows, the new index, build temporaries, and exact CUB scratch.
size_t index_generation_peak(options const& opts, uint32_t references, cuda::stream_ref stream) {
    auto const compat = compatibility(opts);
    uint64_t const postings = cuddl::detail::indexed_posting_count(references, compat);
    // Build arguments evaluate before optional::emplace destroys the old generation,
    // so a rebuild holds old and new rows plus the new index and temporaries.
    // Counting the full new generation is conservative but safe against pool reuse.
    size_t const rows = (opts.rows == "packed") ? database::persistent_packed_row_bytes(references)
                                                : database::persistent_row_bytes(references);
    size_t const row_size = (opts.rows == "packed") ? sizeof(uint32_t) : sizeof(uint16_t);
    size_t index = 0;
    size_t transients = static_cast<size_t>(postings) * row_size;
    if (opts.index == "sparse") {
        index = static_cast<size_t>(postings) * sizeof(uint16_t) +
                static_cast<size_t>(postings) * sizeof(uint32_t);
        transients += static_cast<size_t>(postings) * sizeof(uint16_t);
        transients += static_cast<size_t>(postings) * sizeof(uint32_t);
    } else {
        uint64_t const cells = cuddl::detail::indexed_cell_count(compat);
        index = static_cast<size_t>(cells + 1U) * sizeof(uint32_t) +
                static_cast<size_t>(postings) * sizeof(uint32_t);
    }
    uint64_t const cells = cuddl::detail::indexed_cell_count(compat);
    return rows + index + transients +
           index_rebuild_scratch(
               opts, postings, cells, references, compat.indexed_bucket_count, stream
           );
}

// Resolves the staging cap against actual remaining GPU memory: pool-aware
// available bytes minus a 10% reserve and the future rebuild peak fund cap input
// bytes plus 24-byte descriptors per staged chunk. Explicit caps are honored when
// affordable, otherwise rejected before any staging allocation.
resident_budget resolve_resident_budget(
    options const& opts,
    size_t references,
    size_t records,
    cuda::stream_ref setup
) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDDL_CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
    auto const& pool = cuda::device_default_memory_pool(setup.device());
    auto const reserved = pool.attribute(cuda::memory_pool_attributes::reserved_mem_current);
    auto const used = pool.attribute(cuda::memory_pool_attributes::used_mem_current);
    // cudaMemGetInfo excludes cached pool storage, which the next allocation reuses.
    size_t const reusable = reserved - std::min(reserved, used);
    size_t const available = free_bytes + reusable;
    if (references > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("resident reference count exceeds 32-bit index capacity");
    }
    size_t const peak = index_generation_peak(opts, static_cast<uint32_t>(references), setup);
    // A resolved cap already includes headroom; allow it to absorb probe-to-run changes.
    size_t const reserve = opts.resident_bytes == 0 ? available / 10U : 0;
    // Staging fits when the input bytes plus bounded descriptors fit the funded base.
    // Both comparisons avoid forming the overflowing sum for huge explicit caps, and
    // the predicate is monotone so binary search finds the largest affordable cap.
    auto const fits = [&](size_t cap, size_t base) {
        return cap <= base && size_t{24} * batch_metadata_bound(cap, records) <= base - cap;
    };
    auto const insufficient = [&](char const* what) {
        throw std::runtime_error(
            std::string("insufficient free device memory for resident sequence ") + what +
            " (free " + std::to_string(free_bytes >> 20) + " MiB, peak " +
            std::to_string(peak >> 20) + " MiB)"
        );
    };
    if (reserve + peak >= available) {
        insufficient("staging budget");
    }
    size_t const base = available - reserve - peak;
    size_t cap = 0;
    if (opts.resident_bytes != 0) {
        cap = opts.resident_bytes;
        if (!fits(cap, base)) {
            insufficient("staging budget");
        }
    } else {
        size_t lo = 0, hi = base;
        while (lo < hi) {
            size_t const mid = lo + (hi - lo + 1) / 2;
            if (fits(mid, base)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        cap = lo;
        if (cap < k) {
            insufficient("staging budget");
        }
    }
    return {
        cap,
        free_bytes,
        reusable,
        available,
        peak,
        reserve,
        total_bytes,
        records,
        size_t{24} * batch_metadata_bound(cap, records)
    };
}

json resident_timings(
    options opts,
    parsed_files const& reference_files,
    parsed_files const& query_files,
    std::vector<uint16_t> const& ref_scores,
    std::vector<uint16_t> const& query_scores,
    size_t* resident_batches = nullptr,
    genome_rows const* expected_references = nullptr,
    genome_rows const* expected_queries = nullptr,
    size_t* resolved_bytes = nullptr,
    json* plan = nullptr,
    json* memory = nullptr
) {
    opts.minimum_matches = 0;
    bool const all = opts.topology == "all-to-all";
    bool const sequence = opts.ingest == "sequence";
    cuda::stream setup{cuda::devices[0]};
    // Streamed rows are hashed straight into one store of (buckets + 1)-word sketches, shared by
    // the reference and query collections.
    auto paths = opts.references;
    if (!all) {
        paths.insert(paths.end(), opts.queries.begin(), opts.queries.end());
    }
    auto store = cuda::make_device_buffer<uint32_t>(
        setup, setup.device(), sequence ? paths.size() * (buckets + 1) : 0, cuda::no_init
    );
    collection refs =
        sequence
            ? collection(
                  opts.references.size(), setup, opts.rows == "compact", opts.rows == "packed"
              )
            : collection(reference_files, setup, opts.rows == "compact", opts.rows == "packed");
    collection queries = sequence
                             ? collection(all ? 0 : opts.queries.size(), setup, true, false)
                             : collection(all ? parsed_files{} : query_files, setup, true, false);
    if (sequence) {
        auto const reference_words = opts.references.size() * (buckets + 1);
        refs.bind_store({store.data(), reference_words});
        queries.bind_store({store.data() + reference_words, store.size() - reference_words});
        refs.clear(setup);
        queries.clear(setup);
    } else {
        refs.add(setup);
        queries.add(setup);
    }
    refs.extract(setup);
    queries.extract(setup);
    std::optional<database> db{build(refs, opts, setup, true)};
    search_buffers buffers(*db, static_cast<uint32_t>(queries.rows()), setup, opts.topology);
    auto const n = refs.rows();
    auto const q = all ? n : queries.rows();
    if (n && q > std::numeric_limits<size_t>::max() / n) {
        throw std::runtime_error("resident result size overflow");
    }
    // All-to-all keeps the strict upper triangle in row-major order; batch stays rectangular.
    size_t const pairs =
        all ? (n < 2 ? 0 : (n % 2 == 0 ? (n / 2) * (n - 1) : n * ((n - 1) / 2))) : n * q;
    auto retained = cuda::make_device_buffer<cuddl::batch_search_result>(
        setup, setup.device(), pairs, cuda::no_init
    );
    auto retained_matches =
        cuda::make_device_buffer<uint32_t>(setup, setup.device(), pairs, cuda::no_init);
    setup.sync();
    auto reset = [&](cuda::stream_ref s) {
        refs.clear(s);
        queries.clear(s);
    };
    auto construct = [&](cuda::stream_ref s) {
        refs.add(s);
        queries.add(s);
    };
    auto statistics = [&](cuda::stream_ref s) {
        refs.cardinality(s);
        queries.cardinality(s);
        refs.winner_counts(s);
        queries.winner_counts(s);
    };
    auto rows = [&](cuda::stream_ref s) {
        if (opts.rows == "compact") {
            refs.extract_scores(s);
        } else {
            refs.extract_packed(s);
        }
        queries.extract_scores(s);
    };
    auto index = [&](cuda::stream_ref s) {
        db.emplace(build(refs, opts, s, true));
    };
    auto query = [&](cuda::stream_ref s) {
        auto consume = [&](uint32_t capacity) {
            auto const* source = buffers.results.data();
            auto const* source_matches = buffers.matches.data();
            auto const* count = buffers.count.data();
            auto* target = retained.data();
            auto* matches = retained_matches.data();
            thrust::for_each_n(
                thrust::cuda::par_nosync.on(s.get()),
                thrust::counting_iterator<size_t>{0},
                capacity,
                [=] __device__(size_t i) {
                    if (i < *count) {
                        auto const row = source[i];
                        auto const qid = static_cast<size_t>(row.query_id);
                        auto const rid = static_cast<size_t>(row.reference_id);
                        auto const position =
                            all ? qid * (2 * n - qid - 1) / 2 + rid - qid - 1 : qid * n + rid;
                        target[position] = row;
                        matches[position] = source_matches[i];
                    }
                }
            );
        };
        search(*db, queries, buffers, opts, s, true, all, nullptr, consume);
    };
    if (sequence) {
        auto single = opts;
        single.samples = 1;
        single.warmups = 0;
        cuda::stream_ref const stream = setup;
        auto const device = setup.device();
        auto const per_chunk_blocks =
            static_cast<size_t>(device.attribute(cuda::device_attributes::multiprocessor_count)) *
            2U;
        auto const max_grid =
            static_cast<size_t>(device.attribute(cuda::device_attributes::max_grid_dim_x));
        // Budget against actual remaining memory now that every persistent allocation
        // above is resident. The untimed record count tightens the 24-byte descriptor
        // bound for long-record corpora.
        setup.sync();
        size_t const records = count_sequence_records(paths);
        // Only references are indexed; queries never enter the database peak.
        size_t const indexed = all ? paths.size() : opts.references.size();
        resident_budget const budget = resolve_resident_budget(opts, indexed, records, setup);
        size_t const cap = budget.cap;
        if (resolved_bytes) {
            *resolved_bytes = cap;
        }
        if (plan) {
            // Probe mode: same persistent allocations as a regular run, then report the
            // effective budget without staging input, streaming batches, or running
            // timings. Batch counts come from real runs at the probed budget.
            *plan = json{
                {"resident_batch_bytes", cap},
                {"requested_bytes", opts.resident_bytes},
                {"free_bytes", budget.free_bytes},
                {"reusable_pool_bytes", budget.reusable_pool_bytes},
                {"available_bytes", budget.available_bytes},
                {"future_peak_bytes", budget.future_peak_bytes},
                {"reserve_bytes", budget.reserve_bytes},
                {"metadata_bytes", budget.metadata_bytes},
                {"total_records", records},
                {"total_bytes", budget.total_bytes},
            };
            return json{};
        }
        using batch_chunk = cuddl::detail::sequence_batch_chunk;
        std::optional<cuda::device_buffer<char>> input;
        std::optional<cuda::device_buffer<batch_chunk>> staged;
        std::vector<batch_chunk> host;
        size_t block_end = 0;
        auto construct_sequence = [&](cuda::stream_ref s) {
            auto const grid = static_cast<uint32_t>(std::min(block_end, max_grid));
            cuddl::detail::add_sequence_batch_kernel<buckets, cuddl::default_register_layout>
                <<<grid, 256, 0, s.get()>>>(
                    input->data(), staged->data(), host.size(), block_end, k, store.data()
                );
            CUDDL_CUDA_CALL(cudaGetLastError());
        };
        std::map<std::string, std::vector<double>> samples;
        for (int sample = -opts.warmups; sample < opts.samples; ++sample) {
            std::map<std::string, double> elapsed;
            auto segment = [&](std::string const& name, auto function) {
                json wall;
                json gpu = measure(single, name, false, function, {}, &wall, {}, &stream);
                if (gpu["samples"] != 1 || wall["samples"] != 1) {
                    throw std::runtime_error(
                        "resident segment requires exactly one NVBench sample"
                    );
                }
                elapsed[name] += gpu["median_ms"].get<double>();
                elapsed[name + "_wall"] += wall["median_ms"].get<double>();
            };
            db.reset();
            setup.sync();
            segment("resident_reset", reset);
            elapsed["resident_construct"] = 0;
            elapsed["resident_construct_wall"] = 0;
            size_t count = 0;
            if (sample != -opts.warmups && *resident_batches <= 1) {
                count = *resident_batches;
                if (count) segment("resident_construct", construct_sequence);
            } else {
                count = resident_sequence::for_each_batch(
                    paths, k, cap, [&](resident_sequence::batch const& batch) {
                        // Host windows arithmetic stays size_t; the header caps chunk windows
                        // at UINT32_MAX so the narrowing cast below cannot wrap.
                        host.clear();
                        host.reserve(batch.chunks.size());
                        block_end = 0;
                        for (auto const& chunk : batch.chunks) {
                            if (chunk.size < k) {
                                continue;  // No complete window; the header omits these.
                            }
                            size_t const windows = chunk.size - k + 1;
                            if (windows > std::numeric_limits<uint32_t>::max() ||
                                chunk.genome > std::numeric_limits<uint32_t>::max()) {
                                throw std::runtime_error(
                                    "resident chunk exceeds batch kernel range"
                                );
                            }
                            block_end +=
                                std::min(per_chunk_blocks, (windows + size_t{2047}) / size_t{2048});
                            host.push_back({
                                chunk.offset,
                                block_end,
                                static_cast<uint32_t>(chunk.genome),
                                static_cast<uint32_t>(windows),
                            });
                        }
                        if (!input || input->size() < batch.bases.size()) {
                            // Free before growing so two near-capacity allocations cannot overlap.
                            input.reset();
                            setup.sync();
                            input.emplace(
                                cuda::make_device_buffer<char>(
                                    setup, setup.device(), batch.bases.size(), cuda::no_init
                                )
                            );
                        }
                        if (!staged || staged->size() < host.size()) {
                            staged.reset();
                            setup.sync();
                            staged.emplace(
                                cuda::make_device_buffer<batch_chunk>(
                                    setup, setup.device(), host.size(), cuda::no_init
                                )
                            );
                        }
                        cuda::copy_bytes(
                            setup,
                            cuda::std::span{batch.bases.data(), batch.bases.size()},
                            cuddl::device_span<char>{input->data(), batch.bases.size()}
                        );
                        cuda::copy_bytes(
                            setup,
                            cuda::std::span{host.data(), host.size()},
                            cuddl::device_span<batch_chunk>{staged->data(), host.size()}
                        );
                        setup.sync();
                        segment("resident_construct", construct_sequence);
                    }
                );
            }
            if (sample == -opts.warmups) *resident_batches = count;
            if (*resident_batches != count) {
                throw std::runtime_error("resident input changed between replays");
            }
            segment("resident_statistics", statistics);
            segment("resident_rows", rows);
            segment("resident_index", index);
            segment("resident_search", query);
            // Validate the timed resident construction, including multiplicities and saturation.
            auto const observed_registers = download(store, setup);
            for (size_t i = 0; i < observed_registers.size(); ++i) {
                auto const expected =
                    i < expected_references->registers.size()
                        ? expected_references->registers[i]
                        : expected_queries->registers[i - expected_references->registers.size()];
                if (observed_registers[i] != expected) {
                    throw std::runtime_error(
                        "resident sequence register " + std::to_string(i) + " expected " +
                        std::to_string(expected) + ", observed " +
                        std::to_string(observed_registers[i])
                    );
                }
            }
            double total_gpu = 0, total_wall = 0;
            for (auto const& [name, value] : elapsed) {
                (name.ends_with("_wall") ? total_wall : total_gpu) += value;
            }
            elapsed["resident_total"] = total_gpu;
            elapsed["resident_total_wall"] = total_wall;
            if (sample >= 0) {
                for (auto const& [name, value] : elapsed) {
                    samples[name].push_back(value);
                }
            }
        }
        // Compact downloads already arrive in validation row order.
        host_results observed{download(retained, setup), download(retained_matches, setup)};
        validate(observed, ref_scores, query_scores, opts, true, all, opts.oracle_pairs);
        json result;
        if (memory) {
            *memory = {
                {"resident_input", input ? input->size() : 0},
                {"resident_input_metadata", staged ? staged->size() * sizeof(batch_chunk) : 0},
            };
        }
        for (auto& [name, values] : samples) {
            std::sort(values.begin(), values.end());
            auto const middle = values.size() / 2;
            result[name] = {
                {"samples", values.size()},
                {"min_ms", values.front()},
                {"max_ms", values.back()},
                {"median_ms",
                 values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2},
                {"source", name.ends_with("_wall") ? "nvbench_cpu_wall" : "nvbench_gpu_events"},
            };
        }
        return result;
    }
    json timings;
    auto gpu = [&](char const* name,
                   auto function,
                   std::function<void(cuda::stream_ref)> prepare = {}) {
        json wall;
        timings[name] = measure(opts, name, false, function, prepare, &wall, [&] { db.reset(); });
        timings[std::string(name) + "_wall"] = std::move(wall);
    };
    gpu(
        "resident_total",
        [&](cuda::stream_ref s) {
            reset(s);
            construct(s);
            statistics(s);
            rows(s);
            index(s);
            query(s);
        },
        [&](cuda::stream_ref) { db.reset(); }
    );
    // Download only after measurement; every pair must survive device tile reuse.
    // Compact downloads already arrive in validation row order.
    host_results observed{download(retained, setup), download(retained_matches, setup)};
    validate(observed, ref_scores, query_scores, opts, true, all, opts.oracle_pairs);
    gpu("resident_reset", reset);
    gpu("resident_construct", construct, reset);
    gpu("resident_statistics", statistics);
    gpu("resident_rows", rows);
    gpu("resident_index", index, [&](cuda::stream_ref) { db.reset(); });
    db.emplace(build(refs, opts, setup, true));
    setup.sync();
    gpu("resident_search", query);
    return timings;
}

// Per-genome metrics, a bounded match sample, and in-memory JSON serialization: the output
// phase. Serializing one object per pair is not a realistic application output for a large
// collection, and the DOM costs about 2 KiB per pair, so the sample follows --match-rows.
void application_output(
    options const& opts,
    collection const& refs,
    collection const& queries,
    host_results const& output,
    cuda::stream_ref stream
) {
    json result = {
        {"references", collection_metrics(refs, stream)},
        {"queries", collection_metrics(queries, stream)},
        {"matches", json::array()},
        {"matches_total", output.rows.size()}
    };
    size_t const limit = opts.match_rows ? opts.match_rows : output.rows.size();
    size_t const stride = output.rows.size() > limit ? (output.rows.size() + limit - 1) / limit : 1;
    size_t emitted = 0;
    auto add = [&](cuddl::batch_search_result const& row) {
        result["matches"].push_back(
            {{"query_id", row.query_id},
             {"reference_id", row.reference_id},
             {"metrics", metrics(row.summary)}}
        );
        ++emitted;
    };
    // One evenly spread row sample, always including the last pair.
    for (size_t i = 0; i < output.rows.size(); i += stride) {
        add(output.rows[i]);
    }
    if (output.rows.size() && (output.rows.size() - 1) % stride != 0) {
        add(output.rows.back());
    }
    result["matches_emitted"] = emitted;
    stream.sync();
    auto serialized = result.dump();
    do_not_optimise(serialized);
}

template <typename Mark>
void end_to_end(options const& opts, cuda::stream_ref stream, Mark&& mark) {
    auto reference_files = parse(opts.references);
    auto query_files = opts.topology == "batch" ? parse(opts.queries) : parsed_files{};
    collection refs(reference_files, stream, opts.rows == "compact", opts.rows == "packed");
    collection queries(query_files, stream, true, false);
    refs.add(stream);
    queries.add(stream);
    refs.extract(stream);
    queries.extract(stream);
    auto db = build(refs, opts, stream, true);
    search_buffers buffers(
        db, static_cast<uint32_t>(queries.sketches.size()), stream, opts.topology
    );
    host_results output;
    stream.sync();
    mark();
    search(db, queries, buffers, opts, stream, true, opts.topology == "all-to-all", &output);
    application_output(opts, refs, queries, output, stream);
    mark();
}

// Streamed application run: one bounded tile pass per genome replaces parse, host allocation,
// and upload. Every other stage matches the packed-input path.
template <typename Mark>
void end_to_end_streamed(options const& opts, cuda::stream_ref stream, Mark&& mark) {
    auto reference_rows = stream_genomes(opts.references, opts, stream);
    auto query_rows =
        opts.topology == "batch" ? stream_genomes(opts.queries, opts, stream) : genome_rows{};
    collection refs(reference_rows, stream, opts.rows == "compact", opts.rows == "packed");
    collection queries(query_rows, stream, true, false);
    refs.extract(stream);
    queries.extract(stream);
    auto db = build(refs, opts, stream, true);
    search_buffers buffers(
        db, static_cast<uint32_t>(queries.sketches.size()), stream, opts.topology
    );
    host_results output;
    stream.sync();
    mark();
    search(db, queries, buffers, opts, stream, true, opts.topology == "all-to-all", &output);
    application_output(opts, refs, queries, output, stream);
    mark();
}

json run(options const& opts) {
    bool const streamed = opts.ingest == "sequence";
    cuda::stream stream{cuda::devices[0]};
    // All-to-all result storage grows as N*(N-1)/2 rows. Refuse a corpus that cannot hold them
    // instead of failing inside the first timed sample.
    if (opts.topology == "all-to-all") {
        auto const count = opts.references.size();
        uint64_t const pairs = count > 1 ? uint64_t{count} * (count - 1) / 2 : 0;
        uint64_t const required = pairs * (sizeof(cuddl::batch_search_result) + sizeof(uint32_t));
        size_t free_bytes = 0, total_bytes = 0;
        CUDDL_CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (required > free_bytes / 2) {
            throw std::runtime_error(
                "all-to-all over " + std::to_string(count) + " references needs " +
                std::to_string(required >> 30) + " GiB of result rows, more than half of the " +
                std::to_string(free_bytes >> 30) +
                " GiB free; use --topology batch with a small --query set"
            );
        }
    }
    if (opts.resident_plan && !streamed) {
        throw std::runtime_error("--resident-plan requires --ingest sequence");
    }
    // Probe mode skips the end-to-end run; the outer setup below still stages the
    // realistic memory state the batch budget is measured against.
    json timings = opts.resident_plan
                       ? json::object()
                       : measure_pipeline(opts.samples, opts.warmups, [&](auto mark) {
                             if (streamed) {
                                 end_to_end_streamed(opts, stream, mark);
                             } else {
                                 end_to_end(opts, stream, mark);
                             }
                             stream.sync();
                         });
    // Streamed rows must outlive their collections: register copies are stream-ordered.
    parsed_files reference_files, query_files;
    genome_rows reference_rows, query_rows;
    std::optional<collection> refs_holder, queries_holder;
    if (streamed) {
        reference_rows = stream_genomes(opts.references, opts, stream);
        query_rows = stream_genomes(opts.queries, opts, stream);
        // Both row formats are allocated so the stage suite matches the packed-input path.
        refs_holder.emplace(reference_rows, stream);
        queries_holder.emplace(query_rows, stream);
    } else {
        reference_files = parse(opts.references);
        query_files = parse(opts.queries);
        refs_holder.emplace(reference_files, stream);
        queries_holder.emplace(query_files, stream);
        refs_holder->add(stream);
        queries_holder->add(stream);
    }
    auto& refs = *refs_holder;
    auto& queries = *queries_holder;
    refs.extract(stream);
    queries.extract(stream);
    stream.sync();
    auto const ref_scores = download(refs.scores, stream),
               query_scores = download(queries.scores, stream);
    if (!streamed) {
        auto const original_packed = download(refs.packed, stream);
        auto const original_saturation = download(refs.saturated, stream);
        refs.clear(stream);
        refs.add(stream, true);
        refs.extract(stream);
        if (!(download(refs.packed, stream) == original_packed &&
              download(refs.saturated, stream) == original_saturation)) {
            throw std::runtime_error("incremental construction differs from one-shot construction");
        }
    }

    if (opts.resident_plan) {
        // The outer database and buffers above stage the realistic memory state;
        // the probe reuses them only as resident pressure, then reports the flat
        // batch plan without running any timing.
        json plan;
        size_t plan_batches = 0, plan_bytes = 0;
        resident_timings(
            opts,
            parsed_files{},
            parsed_files{},
            {},
            {},
            &plan_batches,
            nullptr,
            nullptr,
            &plan_bytes,
            &plan
        );
        return plan;
    }

    auto db = build(refs, opts, stream, true);
    search_buffers buffers(db, static_cast<uint32_t>(queries.sketches.size()), stream);
    stream.sync();
    json measurements = json::array();
    auto gpu = [&](std::string const& name,
                   std::function<void(cuda::stream_ref)> function,
                   std::function<void(cuda::stream_ref)> prepare = {}) {
        json wall;
        timings[name] = measure(opts, name, false, std::move(function), std::move(prepare), &wall);
        timings[name + "_wall"] = std::move(wall);
    };
    auto host = [&](std::string const& name, auto function) {
        timings[name] = measure(opts, name, true, function);
    };
    // Packed-input stages need a materialized k-mer stream; a streamed corpus skips them.
    if (!streamed) {
        std::string a48_text = "#k\t25\n#exponent\t6\n";
        for (size_t r = 0; r < refs.sketches.size(); ++r) {
            a48_text += "#id\t" + std::to_string(r) + "\n#len\t4096\n";
            for (size_t b = 0; b < buckets; ++b) {
                if (b != 0) {
                    a48_text += '\t';
                }
                a48_text += cuddl::a48::encode_a48_token(ref_scores[r * buckets + b]);
            }
            a48_text += '\n';
        }
        for (bool parallel : {false, true}) {
            auto decode = [&] {
                return parallel ? CUDDL_UNWRAP(cuddl::a48::decode_a48_tsv_parallel(a48_text))
                                : CUDDL_UNWRAP(cuddl::a48::decode_a48_tsv(a48_text));
            };
            auto decoded = decode();
            if (!(decoded.records.size() == refs.sketches.size())) {
                throw std::runtime_error("A48 record count differs");
            }
            for (size_t r = 0; r < decoded.records.size(); ++r) {
                if (!(decoded.records[r].ordinal == r &&
                      decoded.records[r].scores.size() == buckets &&
                      std::equal(
                          decoded.records[r].scores.begin(),
                          decoded.records[r].scores.end(),
                          ref_scores.data() + r * buckets
                      ))) {
                    throw std::runtime_error("A48 round trip differs");
                }
            }
            host(parallel ? "a48_decode_parallel" : "a48_decode_serial", [&](cuda::stream_ref) {
                auto value = decode();
                do_not_optimise(value);
            });
        }
        host("parse_and_canonicalize", [&](cuda::stream_ref) {
            auto r = parse(opts.references), q = parse(opts.queries);
            do_not_optimise(r);
            do_not_optimise(q);
        });
        gpu("host_to_device", [&](cuda::stream_ref s) {
            for (size_t i = 0; i < refs.inputs.size(); ++i) {
                cuda::copy_bytes(s, reference_files[i].kmers, refs.inputs[i]);
            }
            for (size_t i = 0; i < queries.inputs.size(); ++i) {
                cuda::copy_bytes(s, query_files[i].kmers, queries.inputs[i]);
            }
        });
        gpu("clear", [&](cuda::stream_ref s) {
            refs.clear(s);
            queries.clear(s);
        });
        gpu(
            "construct_resident",
            [&](cuda::stream_ref s) {
                refs.add(s);
                queries.add(s);
            },
            [&](cuda::stream_ref s) {
                refs.clear(s);
                queries.clear(s);
            }
        );
        // Clear is included and labelled here: every NVBench replay starts with identical state.
        gpu("clear_and_construct", [&](cuda::stream_ref s) {
            refs.clear(s);
            queries.clear(s);
            refs.add(s);
            queries.add(s);
        });
        gpu("clear_and_incremental_construct", [&](cuda::stream_ref s) {
            refs.clear(s);
            queries.clear(s);
            refs.add(s, true);
            queries.add(s, true);
        });
    }
    gpu("extract_compact_rows", [&](cuda::stream_ref s) {
        refs.extract_scores(s);
        queries.extract_scores(s);
    });
    gpu("extract_packed_rows", [&](cuda::stream_ref s) {
        refs.extract_packed(s);
        queries.extract_packed(s);
    });
    gpu("winner_counts", [&](cuda::stream_ref s) {
        refs.winner_counts(s);
        queries.winner_counts(s);
    });
    gpu("cardinality", [&](cuda::stream_ref s) {
        refs.cardinality(s);
        queries.cardinality(s);
    });
    host("hybrid_cardinality_host_result", [&](cuda::stream_ref) {
        for (auto const& c : {&refs, &queries}) {
            for (auto const& s : c->sketches) {
                auto value = CUDDL_UNWRAP(s.hybrid_cardinality(stream));
                do_not_optimise(value);
            }
        }
    });
    auto pair_output = cuda::make_device_buffer<cuddl::pairwise_summary>(
        stream, stream.device(), 1, cuddl::pairwise_summary{}
    );
    auto compare = [&](cuda::stream_ref s, bool cardinality) {
        if (cardinality) {
            CUDDL_UNWRAP(queries.sketches.front().summary_async<true>(
                refs.sketches.front(), *pair_output.data(), s
            ));
        } else {
            CUDDL_UNWRAP(queries.sketches.front().compare_async(
                refs.sketches.front(), *pair_output.data(), s
            ));
        }
    };
    compare(stream, false);
    auto summary = download(pair_output, stream).front();
    if (!(summary.counts == oracle(query_scores.data(), ref_scores.data()))) {
        throw std::runtime_error("pairwise summary differs from oracle");
    }
    gpu("pairwise_summary", [&](cuda::stream_ref s) { compare(s, false); });
    gpu("pairwise_summary_with_cardinality", [&](cuda::stream_ref s) { compare(s, true); });
    auto const fused = download(pair_output, stream).front();
    // The standalone reduction uses FP32; the fused summary retains FP64 accumulation.
    if (!(fused.counts == summary.counts &&
          std::abs(
              fused.cardinality - CUDDL_UNWRAP(queries.sketches.front().cardinality(stream))
          ) <= 2e-6 * std::max(1.0, fused.cardinality))) {
        throw std::runtime_error("fused summary differs from separate reductions");
    }
    auto pair_metrics = metrics(fused);
    pair_metrics["cardinality"] = fused.cardinality;
    measurements.push_back(
        {{"implementation", {{"name", "cuddl"}}},
         {"case",
          {{"measurement", "pairwise"},
           {"query_id", 0},
           {"reference_id", 0},
           {"include_cardinality", true}}},
         {"metrics", pair_metrics}}
    );
    auto pair_count = std::min(refs.sketches.size(), queries.sketches.size());
    auto batch_output = cuda::make_device_buffer<cuddl::pairwise_summary>(
        stream, stream.device(), pair_count, cuda::no_init
    );
    auto batch_compare = [&](cuda::stream_ref s) {
        CUDDL_UNWRAP(
            cuddl::compare_batch_async<buckets>(
                {queries.packed.data(), pair_count * buckets},
                {refs.packed.data(), pair_count * buckets},
                batch_output,
                s
            )
        );
    };
    batch_compare(stream);
    auto batch_host = download(batch_output, stream);
    for (size_t i = 0; i < pair_count; ++i) {
        if (!(batch_host[i].counts ==
              oracle(query_scores.data() + i * buckets, ref_scores.data() + i * buckets))) {
            throw std::runtime_error("corresponding-row batch compare differs from oracle");
        }
    }
    gpu("compare_corresponding_rows", batch_compare);
    host("derive_pair_metrics", [&](cuda::stream_ref) {
        auto m = metrics(summary);
        do_not_optimise(m);
    });
    gpu("database_build", [&](cuda::stream_ref s) {
        auto value = build(refs, opts, s, false);
        do_not_optimise(value);
    });
    gpu("database_and_index_build", [&](cuda::stream_ref s) {
        auto value = build(refs, opts, s, true);
        do_not_optimise(value);
    });

    auto single_results = cuda::make_device_buffer<cuddl::reference_search_result>(
        stream, stream.device(), refs.sketches.size(), cuda::no_init
    );
    auto single_workspace = cuda::make_device_buffer<uint8_t>(
        stream,
        stream.device(),
        CUDDL_UNWRAP(db.indexed_single_query_workspace_bytes(stream)),
        cuda::no_init
    );
    auto single = [&](cuda::stream_ref s, bool indexed) {
        cuddl::device_span<uint16_t const> query{queries.scores.data(), buckets};
        if (indexed) {
            CUDDL_UNWRAP(db.search_indexed_async(
                query,
                compatibility(opts),
                single_workspace,
                single_results,
                buffers.count,
                {.minimum_matches = opts.minimum_matches},
                s
            ));
        } else {
            CUDDL_UNWRAP(db.search_async(query, compatibility(opts), {}, single_results, s));
        }
    };
    for (bool indexed : {false, true}) {
        single(stream, indexed);
        auto rows = download(single_results, stream);
        auto size =
            indexed ? download(buffers.count, stream).front() : static_cast<uint32_t>(rows.size());
        size_t expected = 0;
        for (size_t r = 0; r < refs.sketches.size(); ++r) {
            if (indexed && hits(query_scores.data(), ref_scores.data() + r * buckets, opts) <
                               opts.minimum_matches) {
                continue;
            }
            if (!(expected < size && rows[expected].reference_id == r &&
                  rows[expected].summary.counts ==
                      oracle(query_scores.data(), ref_scores.data() + r * buckets))) {
                throw std::runtime_error("single-query search differs from oracle");
            }
            ++expected;
        }
        if (!(expected == size)) {
            throw std::runtime_error("single-query search count differs from oracle");
        }
        gpu(indexed ? "search_single_indexed" : "search_single_exhaustive",
            [&](cuda::stream_ref s) { single(s, indexed); });
    }
    // The all-to-all suite is coverage for small corpora. A large collection needs more pair
    // rows than the buffers hold, so it runs the selected topology only.
    auto const all_to_all_fits =
        CUDDL_UNWRAP(db.all_to_all_search_requirements(stream)).maximum_pair_count <=
        buffers.results.size();
    uint64_t selected = 0, exhaustive_pairs = 0, match_rows_emitted = 0, match_rows_total = 0;
    validation selected_validation;
    auto emit_match = [&](cuddl::batch_search_result const& row) {
        measurements.push_back(
            {{"implementation", {{"name", "cuddl"}}},
             {"case",
              {{"measurement", "match"},
               {"query_id", row.query_id},
               {"reference_id", row.reference_id}}},
             {"metrics", metrics(row.summary)}}
        );
    };
    for (bool all : {false, true}) {
        if (all && !all_to_all_fits) {
            continue;
        }
        for (bool indexed : {false, true}) {
            host_results output;
            search(db, queries, buffers, opts, stream, indexed, all, &output);
            auto const observed =
                validate(output, ref_scores, query_scores, opts, indexed, all, opts.oracle_pairs);
            auto name = std::string(all ? "search_all_to_all_" : "search_batch_") +
                        (indexed ? "indexed" : "exhaustive");
            if (!all || refs.sketches.size() > 1) {
                gpu(name, [&](cuda::stream_ref s) {
                    search(db, queries, buffers, opts, s, indexed, all);
                });
            }
            // Retain the selected application results, independently checked before timing.
            if (all == (opts.topology == "all-to-all")) {
                if (!indexed) {
                    exhaustive_pairs = output.rows.size();
                } else {
                    selected = output.rows.size();
                    selected_validation = observed;
                    match_rows_total = output.rows.size();
                    // One evenly spread row sample, always including the last pair.
                    size_t const limit = opts.match_rows ? opts.match_rows : output.rows.size();
                    size_t const stride =
                        output.rows.size() > limit ? (output.rows.size() + limit - 1) / limit : 1;
                    for (size_t i = 0; i < output.rows.size(); i += stride) {
                        emit_match(output.rows[i]);
                        ++match_rows_emitted;
                    }
                    if (output.rows.size() && (output.rows.size() - 1) % stride != 0) {
                        emit_match(output.rows.back());
                        ++match_rows_emitted;
                    }
                }
            }
            // Threshold zero must exhaustively refine all pairs, even when the requested threshold
            // is nonzero.
            if (indexed) {
                auto zero = opts;
                zero.minimum_matches = 0;
                host_results exhaustive_indexed;
                search(db, queries, buffers, zero, stream, true, all, &exhaustive_indexed);
                validate(
                    exhaustive_indexed, ref_scores, query_scores, zero, true, all, zero.oracle_pairs
                );
            }
        }
    }
    host("search_and_download", [&](cuda::stream_ref) {
        host_results output;
        search(db, queries, buffers, opts, stream, true, opts.topology == "all-to-all", &output);
        do_not_optimise(output);
    });
    for (auto const& [label, group] : std::vector<std::pair<std::string, collection const*>>{
             {"reference", &refs}, {"query", &queries}
         }) {
        auto values = collection_metrics(*group, stream);
        for (size_t i = 0; i < values.size(); ++i) {
            measurements.push_back(
                {{"implementation", {{"name", "cuddl"}}},
                 {"case", {{"measurement", "genome"}, {"role", label}, {"genome_id", i}}},
                 {"metrics", values[i]}}
            );
        }
    }
    uint64_t input_bytes = 0, input_files = 0, bases = 0, kmers = 0;
    if (streamed) {
        input_files = reference_rows.input_bytes + query_rows.input_bytes;
    } else {
        for (auto const* files : {&reference_files, &query_files}) {
            for (auto const& f : *files) {
                input_bytes += f.kmers.size() * sizeof(uint64_t);
                bases += f.bases;
                kmers += f.valid_kmers;
            }
        }
    }
    size_t resident_batches = 0, resident_resolved = 0;
    json resident_memory = json::object();
    timings.update(resident_timings(
        opts,
        reference_files,
        query_files,
        ref_scores,
        query_scores,
        &resident_batches,
        &reference_rows,
        &query_rows,
        &resident_resolved,
        nullptr,
        &resident_memory
    ));
    measurements.insert(
        measurements.begin(),
        json{
            {"implementation",
             {{"name", "cuddl"},
              {"revision", command_output("git rev-parse HEAD")},
              {"variant", opts.rows + "_" + opts.index}}},
            {"case",
             {{"measurement", "pipeline"},
              {"k", k},
              {"buckets", buckets},
              {"rows", opts.rows},
              {"index", opts.index},
              {"topology", opts.topology},
              {"ingest", opts.ingest},
              {"indexed_buckets", opts.indexed_buckets},
              {"key_bits", opts.key_bits},
              {"minimum_matches", opts.minimum_matches},
              {"hash_identity", compatibility(opts).hash_identity},
              {"hash_seed", compatibility(opts).hash_seed},
              {"canonicalisation_policy", compatibility(opts).canonicalisation_policy},
              {"exponent_bits", compatibility(opts).exponent_bits},
              {"mantissa_bits", compatibility(opts).mantissa_bits},
              {"parser_threads", streamed ? static_cast<int>(opts.workers) : 0},
              {"application_queries", opts.topology == "batch" ? queries.sketches.size() : 0},
              {"references", refs.sketches.size()},
              {"queries", queries.sketches.size()},
              {"bases", bases},
              {"kmers", kmers},
              {"samples", opts.samples},
              {"warmups", opts.warmups},
              {"input_cache", "warm_os_cache"},
              {"end_to_end_output", "host_metrics_and_bounded_json"},
              {"resident_input", streamed ? "sequence_ascii" : "packed_u64_actg_max"},
              {"resident_batch_bytes", streamed ? resident_resolved : 0},
              {"resident_batches", resident_batches},
              {"resident_timing_scope",
               streamed ? "batched_resident_segments" : "resident_pipeline"},
              {"resident_output", "device_cardinalities_winners_and_pair_summaries"},
              {"resident_minimum_matches", 0},
              {"worktree_dirty",
               !command_output("git status --porcelain --untracked-files=no").empty()}}},
            {"metrics",
             {{"oracle_passed", true},
              {"candidates", selected},
              {"exhaustive_pairs", exhaustive_pairs},
              {"oracle_pairs_total", selected_validation.expected},
              {"oracle_pairs_checked", selected_validation.checked},
              {"match_rows_total", match_rows_total},
              {"match_rows_emitted", match_rows_emitted},
              {"all_to_all_suite", all_to_all_fits},
              {"all_to_all_nonempty", refs.sketches.size() > 1},
              {"corresponding_pairs", pair_count},
              {"preserves_multiplicity", db.preserves_multiplicity()}}},
            {"timings", timings},
            {"memory_bytes",
             {{"input_kmers", input_bytes},
              {"input_files", input_files},
              {"persistent_rows", db.persistent_row_bytes()},
              {"persistent_index", db.persistent_index_bytes()},
              {"search_workspace", buffers.workspace.size()},
              {"search_result_capacity",
               buffers.results.size() * sizeof(cuddl::batch_search_result)},
              {"search_match_capacity", buffers.matches.size() * sizeof(uint32_t)}}},
        }
    );
    measurements.front()["memory_bytes"].update(resident_memory);
    json datasets = json::object();
    for (auto const& [role, paths] : std::vector<std::pair<std::string, std::vector<std::string>>>{
             {"reference", opts.references}, {"query", opts.queries}
         }) {
        datasets.update(dataset_entries(role, paths, opts.dataset_hashes));
    }
    return make_benchmark_result(opts.name, "pipeline", "end_to_end", measurements, datasets);
}
}  // namespace

int main(int argc, char** argv) try {
    options opts;
    CLI::App app{
        "cuDDL FASTA-to-results pipeline with isolated NVBench stages. Each file is one genome."
    };
    app.add_option("--reference", opts.references, "Reference FASTA files, repeatable")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--query", opts.queries, "Query FASTA files, repeatable")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--output", opts.output, "Shared-schema JSON output, stdout if omitted");
    app.add_option("--name", opts.name);
    app.add_option("--rows", opts.rows)->check(CLI::IsMember({"compact", "packed"}));
    app.add_option("--index", opts.index)->check(CLI::IsMember({"dense", "sparse"}));
    app.add_option("--topology", opts.topology, "Search used in the E2E total")
        ->check(CLI::IsMember({"batch", "all-to-all"}));
    app.add_option("--minimum-matches", opts.minimum_matches)->check(CLI::Range(0, int(buckets)));
    app.add_option("--indexed-buckets", opts.indexed_buckets)
        ->check(CLI::IsMember({int(buckets / 2), int(buckets)}));
    app.add_option("--key-bits", opts.key_bits)->check(CLI::IsMember({15, 16}));
    app.add_option("--samples", opts.samples)->check(CLI::Range(2, 10000));
    app.add_option("--warmups", opts.warmups)->check(CLI::Range(0, 1000));
    app.add_option(
           "--ingest",
           opts.ingest,
           "packed: parse to host uint64 k-mer arrays. sequence: stream tiles per genome with "
           "bounded memory, required for a large corpus"
    )
        ->check(CLI::IsMember({"packed", "sequence"}));
    app.add_option("--workers", opts.workers, "File loading workers for --ingest sequence")
        ->check(CLI::Range(0u, 64u));
    app.add_option(
           "--resident-bytes",
           opts.resident_bytes,
           "Resident ASCII input batch byte budget for --ingest sequence "
           "(0 selects automatically from free GPU memory)"
    )
        ->check(
            CLI::Validator(
                [](std::string& value) {
                    size_t parsed = 0;
                    try {
                        parsed = std::stoull(value);
                    } catch (std::exception const&) {
                        return std::string{"must be a byte count"};
                    }
                    if (parsed != 0 && parsed < size_t{k}) {
                        return std::string{"must be 0 (auto) or at least k bytes"};
                    }
                    return std::string{};
                },
                "0 or >= k"
            )
        );
    app.add_flag(
        "--resident-plan",
        opts.resident_plan,
        "Sequence-only: print the resident batch plan as flat JSON without timings"
    );
    app.add_option("--oracle-pairs", opts.oracle_pairs, "Scalar-oracle pair comparisons per suite")
        ->check(CLI::Range(size_t{0}, size_t{1} << 40));
    app.add_option("--match-rows", opts.match_rows, "Match measurement rows emitted")
        ->check(CLI::Range(size_t{0}, size_t{1} << 40));
    app.add_option("--dataset-hashes", opts.dataset_hashes, "Per-file dataset digests")
        ->check(CLI::Range(size_t{0}, size_t{1} << 20));
    app.set_config("--config", "", "Read benchmark options from a configuration file");
    CLI11_PARSE(app, argc, argv);
    if (!(opts.topology != "all-to-all" || opts.references.size() >= 2)) {
        throw std::runtime_error("all-to-all E2E needs at least two reference genomes");
    }
    if (!(opts.minimum_matches <= opts.indexed_buckets)) {
        throw std::runtime_error("minimum matches exceeds indexed buckets");
    }
    auto report = run(opts).dump(2) + "\n";
    if (opts.output.empty()) {
        std::cout << report;
    } else {
        std::ofstream output(opts.output);
        if (!output) {
            throw std::runtime_error("cannot open output " + opts.output);
        }
        output << report;
        if (!output) {
            throw std::runtime_error("cannot write output " + opts.output);
        }
    }
    return 0;
} catch (std::exception const& e) {
    std::cerr << "cuddl-pipeline: " << e.what() << '\n';
    return 1;
}
