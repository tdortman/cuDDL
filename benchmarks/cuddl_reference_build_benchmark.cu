#include <CLI/CLI.hpp>
#include <cuddl/reference_database_file.cuh>
#include <nlohmann/json.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

int main(int argc, char** argv) try {
    std::vector<std::string> references;
    std::string output;
    unsigned threads = std::thread::hardware_concurrency();
    // One loader by default: run_micro_comparison.py measures ingest this way.
    unsigned workers = 1;
    std::optional<size_t> staging_bytes;
    int copies = 1, samples = 5;
    bool parse_only = false;
    std::string transfer = "automatic";
    CLI::App app{
        "NVBench wall timing of FASTX -> reference sketches -> binary file (k=25, buckets=4096)"
    };
    app.add_option("--reference", references)->required()->check(CLI::ExistingFile);
    app.add_option("--database", output, "Temporary benchmark database destination")->required();
    app.add_option(
        "--threads", threads, "CPU oracle parser threads (--parse-only); defaults to all cores"
    );
    app.add_option("--workers", workers, "Concurrent genome loaders (default: 1)");
    app.add_option(
        "--staging-bytes",
        staging_bytes,
        "Cap the device staging arena; unset sizes it from the inputs and free memory"
    );
    app.add_option("--copies", copies, "Repeat inputs for an explicitly synthetic collection")
        ->check(CLI::Range(1, 100000));
    app.add_option("--samples", samples)->check(CLI::Range(2, 1000));
    app.add_flag(
        "--parse-only", parse_only, "Time the serial CPU parser oracle, excluding GPU construction"
    );
    app.add_option(
           "--transfer",
           transfer,
           "How the build gets bytes to the device: automatic asks the device, pinned uses "
           "page-locked buffers, staged copies through a heap buffer, in-place lets the kernels "
           "read the heap buffer"
    )
        ->check(CLI::IsMember({"automatic", "pinned", "staged", "in-place"}))
        ->capture_default_str();
    app.set_config("--config", "TOML file with options, e.g. reference = [...]");
    CLI11_PARSE(app, argc, argv);
    auto const requested_transfer = [&] {
        if (transfer == "pinned") return cuddl::transfer_mode::pinned;
        if (transfer == "staged") return cuddl::transfer_mode::staged;
        if (transfer == "in-place") return cuddl::transfer_mode::in_place;
        return cuddl::transfer_mode::automatic;
    }();
    std::vector<std::filesystem::path> paths;
    uint64_t bytes = 0;
    for (int copy = 0; copy < copies; ++copy) {
        for (auto const& path : references) {
            paths.emplace_back(path);
            bytes += std::filesystem::file_size(path);
        }
    }
    cuda::stream stream{cuda::devices[0]};
    cuddl::reference_build_statistics statistics;
    std::vector<double> resident_ms;
    int invocation = -1;
    auto run = [&](nvbench::state& state, nvbench::type_list<>) {
        state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
            timer.start();
            if (parse_only) {
                for (auto const& path : paths) {
                    auto parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(path.string(), 25, threads));
                    if (parsed.valid_kmers != parsed.kmers.size()) {
                        throw std::runtime_error("incorrect parser count");
                    }
                }
            } else {
                statistics = {};
                statistics.measure_resident = true;
                auto file = CUDDL_UNWRAP((cuddl::reference_database_file::build<25, 4096>(
                    paths,
                    stream,
                    {.statistics = &statistics,
                     .parser_workers = workers,
                     .staging_bytes = staging_bytes,
                     .transfer = requested_transfer}
                )));
                CUDDL_UNWRAP(file.save(output));
            }
            timer.stop();
            if (invocation++ >= 0) {
                if (!parse_only) resident_ms.push_back(statistics.resident_compute_ms);
            }
        });
    };
    nvbench::benchmark<decltype(run)> benchmark(run);
    // Corpus-sized samples must finish by count, not NVBench's default wall timeout.
    benchmark.set_name(parse_only ? "reference-parse" : "reference-build-and-save")
        .set_stopping_criterion("sample-count")
        .set_min_samples(samples)
        .set_criterion_param_int64("target-samples", samples)
        .set_timeout(std::numeric_limits<double>::max())
        .set_cold_warmup_runs(1)
        .set_skip_batched(true)
        .set_is_cpu_only(true);
    benchmark.run();
    auto const& state = benchmark.get_states().front();
    if (state.is_skipped()) throw std::runtime_error(state.get_skip_reason());
    auto const measured_samples = state.get_summary("nv/cpu_only/sample_size").get_int64("value");
    if (measured_samples != samples ||
        (!parse_only && resident_ms.size() != static_cast<size_t>(measured_samples))) {
        throw std::runtime_error("reference build did not collect the requested timing samples");
    }
    if (!parse_only) {
        auto file = CUDDL_UNWRAP(cuddl::reference_database_file::load(output));
        if (file.names().size() != paths.size()) {
            throw std::runtime_error("incorrect reference count");
        }
    }
    auto const median = state.get_summary("nv/cpu_only/time/cpu/median").get_float64("value");
    auto const minimum = state.get_summary("nv/cpu_only/time/cpu/min").get_float64("value");
    auto const maximum = state.get_summary("nv/cpu_only/time/cpu/max").get_float64("value");
    auto summarize = [](std::vector<double> values, std::string const& source) {
        std::sort(values.begin(), values.end());
        auto const middle = values.size() / 2;
        auto const median_ms =
            values.size() % 2 != 0 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
        return nlohmann::json{
            {"samples", values.size()},
            {"median_ms", median_ms},
            {"min_ms", values.front()},
            {"max_ms", values.back()},
            {"source", source},
        };
    };
    nlohmann::json report{
        {"stage", parse_only ? "parse" : "build-and-save"},
        {"source", "nvbench_cpu_wall"},
        {"references", paths.size()},
        {"input_paths", references},
        {"copies", copies},
        {"input_bytes", bytes},
        {"parser_threads", threads},
        {"parser_workers", workers},
        {"samples", measured_samples},
        {"median_seconds", median},
        {"input_MB_per_second", bytes / median / 1e6},
        {"direct_bytes", statistics.direct_bytes},
        {"staged_bytes", statistics.staged_bytes},
        {"direct_chunks", statistics.direct_chunks},
        {"staged_chunks", statistics.staged_chunks},
        {"pinned_buffers", statistics.pinned_buffers},
        {"transfer_requested", transfer},
        {"in_place", statistics.in_place},
        {"parsers", static_cast<int>(statistics.workers)},
        {"staging_bytes", statistics.staging_bytes},
        {"batches", statistics.batches},
        {"transfers", statistics.transfers}
    };
    report["wall"] = {
        {"samples", measured_samples},
        {"median_ms", median * 1000},
        {"min_ms", minimum * 1000},
        {"max_ms", maximum * 1000},
        {"source", "nvbench_cpu_wall"},
    };
    if (!parse_only) report["resident"] = summarize(resident_ms, "cuda_events");
    std::cout << report.dump(2) << '\n';
} catch (std::exception const& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
