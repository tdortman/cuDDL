#include <CLI/CLI.hpp>
#include <cuddl/reference_database_file.cuh>
#include <nlohmann/json.hpp>
#include <nvbench/nvbench.cuh>

#include <filesystem>
#include <iostream>
#include <vector>

int main(int argc, char** argv) try {
    std::vector<std::string> references;
    std::string output;
    unsigned threads = 0;
    unsigned workers = 1;
    int copies = 1, samples = 5;
    bool parse_only = false;
    CLI::App app{
        "NVBench wall timing of FASTX -> reference sketches -> binary file (k=25, buckets=2048)"
    };
    app.add_option("--reference", references)->required()->check(CLI::ExistingFile);
    app.add_option("--database", output, "Temporary benchmark database destination")->required();
    app.add_option(
        "--threads", threads, "CPU oracle parser threads (--parse-only); 0 selects automatic"
    );
    app.add_option("--workers", workers, "Concurrent genome loaders; 0 selects automatic");
    app.add_option("--copies", copies, "Repeat inputs for an explicitly synthetic collection")
        ->check(CLI::Range(1, 100000));
    app.add_option("--samples", samples)->check(CLI::Range(2, 1000));
    app.add_flag(
        "--parse-only", parse_only, "Time the serial CPU parser oracle, excluding GPU construction"
    );
    CLI11_PARSE(app, argc, argv);
    std::vector<std::filesystem::path> paths;
    uint64_t bytes = 0;
    for (int copy = 0; copy < copies; ++copy) {
        for (auto const& path : references) {
            paths.emplace_back(path);
            bytes += std::filesystem::file_size(path);
        }
    }
    cuda::stream stream{cuda::devices[0]};
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
                auto file = CUDDL_UNWRAP(
                    (cuddl::reference_database_file::build<25, 2048>(paths, stream, workers))
                );
                CUDDL_UNWRAP(file.save(output));
            }
            timer.stop();
        });
    };
    nvbench::benchmark<decltype(run)> benchmark(run);
    benchmark.set_name(parse_only ? "reference-parse" : "reference-build-and-save")
        .set_stopping_criterion("sample-count")
        .set_min_samples(samples)
        .set_criterion_param_int64("target-samples", samples)
        .set_cold_warmup_runs(1)
        .set_skip_batched(true)
        .set_is_cpu_only(true);
    benchmark.run();
    auto const& state = benchmark.get_states().front();
    if (state.is_skipped()) throw std::runtime_error(state.get_skip_reason());
    if (!parse_only) {
        auto file = CUDDL_UNWRAP(cuddl::reference_database_file::load(output));
        if (file.names().size() != paths.size()) {
            throw std::runtime_error("incorrect reference count");
        }
    }
    auto const median = state.get_summary("nv/cpu_only/time/cpu/median").get_float64("value");
    std::cout << nlohmann::json{
        {"stage", parse_only ? "parse" : "build-and-save"},
        {"source", "nvbench_cpu_wall"},
        {"references", paths.size()}, {"input_paths", references}, {"copies", copies},
        {"input_bytes", bytes}, {"parser_threads", threads}, {"parser_workers", workers}, {"samples", samples},
        {"median_seconds", median}, {"input_MB_per_second", bytes / median / 1e6}
    }.dump(2) << '\n';
} catch (std::exception const& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
