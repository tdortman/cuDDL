// cuddl-benchmark-fasta: reproducible raw-FASTA end-to-end benchmark for cuDDL.
//
// Reads every record in each FASTA file as one complete genome, builds a DDL sketch per genome
// on the selected backend, compares the single pair, and emits one CSV row per warm-up or measured
// run. GPU totals run from first input byte read through final host-visible WKID/ANI; the CPU
// backend covers the same work with a single thread as the exact scalar baseline.
//
// This is the bespoke end-to-end gate tooling the acceptance contract prescribes; it intentionally
// is not google-benchmark or nvbench, which measure resident kernels rather than a raw-FASTA
// pipeline.

#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>

#include <cuda_runtime.h>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using steady_clock_t = std::chrono::steady_clock;

/// @brief Per-run timing breakdown and result, emitted as one CSV row.
struct run_result {
    double preprocess_ms = 0;
    double allocation_ms = 0;
    double h2d_ms = 0;
    double construction_ms = 0;
    double comparison_ms = 0;
    double d2h_ms = 0;
    double metric_ms = 0;
    double total_ms = 0;
    uint32_t lower = 0;
    uint32_t equal = 0;
    uint32_t higher = 0;
    bool valid_metric = false;
    double wkid = 0;
    double ani = 0;
};

/// @brief Fills metric fields of @p r from raw pairwise counts via the library metric methods.
void fill_metrics(run_result& r, cuddl::pairwise_counts const& counts, uint32_t k) {
    uint32_t saturation{};
    cuddl::sketch_ref<25, 2048> metrics{{}, saturation};
    cuddl::pairwise_summary summary;
    summary.counts = counts;
    auto const wkid = metrics.wkid(summary);
    if (wkid) {
        r.valid_metric = true;
        r.wkid = *wkid;
        auto const ani = metrics.ani(summary);
        if (ani) {
            r.ani = *ani;
        }
    }
    (void)k;
}

/// @brief Scalar CPU DDL sketch used by the CPU backend (host-only, exact).
template <size_t BucketCount>
struct host_sketch {
    std::vector<uint32_t> registers = std::vector<uint32_t>(BucketCount, 0U);

    void add(std::vector<uint64_t> const& inputs) {
        for (auto const input : inputs) {
            auto const hash = cuddl::detail::hash_kmer(input);
            auto const bucket = cuddl::detail::bucket_of<BucketCount>(hash);
            auto const s = cuddl::detail::score(hash);
            auto const observed = registers[bucket];
            auto const w = cuddl::detail::winner(observed);
            auto const c = cuddl::detail::count(observed);
            if (s > w) {
                registers[bucket] = cuddl::detail::pack(s, 1U);
            } else if (s == w && c < cuddl::detail::max_winner_count) {
                registers[bucket] = cuddl::detail::pack(s, static_cast<uint16_t>(c + 1U));
            }
        }
    }
};

/// @brief Size cap (bytes) per device chunk during construction, below 80% of free memory.
size_t construction_chunk_bytes() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    auto const cap = free_bytes * 4U / 5U;
    // Leave room for the two sketches (registers + flags) plus a modest headroom.
    auto const target = cap / 3;
    // Bound each chunk to a sane maximum so huge inputs stream without one giant allocation.
    auto const chunk_cap = (size_t{1} << 26);  // 64 Mi k-mers = 512 MiB
    return target < chunk_cap ? target : chunk_cap;
}

/// @brief Runs one measured GPU pass, constructing sketches in bounded memory chunks.
template <size_t BucketCount>
run_result run_gpu(
    cuddl::sketch<25, BucketCount>& query_sketch,
    cuddl::sketch<25, BucketCount>& ref_sketch,
    std::vector<uint64_t> const& query_kmers,
    std::vector<uint64_t> const& ref_kmers
) {
    run_result r;

    auto t0 = steady_clock_t::now();
    CUDDL_UNWRAP(query_sketch.clear());
    CUDDL_UNWRAP(ref_sketch.clear());
    auto const sketch_chunk_elements = construction_chunk_bytes() / sizeof(uint64_t);
    if (sketch_chunk_elements == 0) {
        throw std::runtime_error("insufficient free device memory for construction");
    }
    double h2d = 0.0, construction = 0.0;
    // Transfer + construct each genome in chunks so a whole-genome stream fits any device.
    auto build = [&](cuddl::sketch<25, BucketCount>& target, std::vector<uint64_t> const& kmers) {
        for (size_t offset = 0; offset < kmers.size(); offset += sketch_chunk_elements) {
            auto const n = std::min(sketch_chunk_elements, kmers.size() - offset);
            uint64_t* d = nullptr;
            auto t = steady_clock_t::now();
            CUDDL_CUDA_CALL(cudaMalloc(&d, n * sizeof(uint64_t)));
            CUDDL_CUDA_CALL(
                cudaMemcpy(d, kmers.data() + offset, n * sizeof(uint64_t), cudaMemcpyHostToDevice)
            );
            h2d += std::chrono::duration<double, std::milli>(steady_clock_t::now() - t).count();
            t = steady_clock_t::now();
            CUDDL_UNWRAP(target.add({d, n}));
            construction +=
                std::chrono::duration<double, std::milli>(steady_clock_t::now() - t).count();
            CUDDL_CUDA_CALL(cudaFree(d));
        }
    };
    build(query_sketch, query_kmers);
    build(ref_sketch, ref_kmers);
    r.h2d_ms = h2d;
    r.construction_ms = construction;

    t0 = steady_clock_t::now();
    auto const summary = CUDDL_UNWRAP(query_sketch.compare(ref_sketch.ref()));
    r.comparison_ms = std::chrono::duration<double, std::milli>(steady_clock_t::now() - t0).count();

    r.lower = summary.counts.lower;
    r.equal = summary.counts.equal;
    r.higher = summary.counts.higher;
    fill_metrics(r, summary.counts, 25);
    return r;
}

/// @brief Runs one measured CPU pass.
template <size_t BucketCount>
run_result
run_cpu(std::vector<uint64_t> const& query_kmers, std::vector<uint64_t> const& ref_kmers) {
    run_result r;
    auto t0 = steady_clock_t::now();
    host_sketch<BucketCount> query;
    host_sketch<BucketCount> ref_sketch;
    query.add(query_kmers);
    ref_sketch.add(ref_kmers);
    r.construction_ms =
        std::chrono::duration<double, std::milli>(steady_clock_t::now() - t0).count();

    t0 = steady_clock_t::now();
    cuddl::pairwise_counts counts{};
    for (size_t b = 0; b < BucketCount; ++b) {
        auto const l = cuddl::detail::winner(query.registers[b]);
        auto const rr = cuddl::detail::winner(ref_sketch.registers[b]);
        if (l == 0 && rr == 0) {
            ++counts.both_empty;
        } else if (l < rr) {
            ++counts.lower;
        } else if (l > rr) {
            ++counts.higher;
        } else {
            ++counts.equal;
        }
    }
    r.comparison_ms = std::chrono::duration<double, std::milli>(steady_clock_t::now() - t0).count();
    r.lower = counts.lower;
    r.equal = counts.equal;
    r.higher = counts.higher;
    fill_metrics(r, counts, 25);
    return r;
}

/// @brief Emits one complete CSV row matching the 60-column acceptance schema.
///
/// Machine/tool provenance columns are left blank; the acceptance runner fills them. Empty
/// numeric columns are blank, never zero. @p is_warmup distinguishes warm-up from measured rows.
void emit_row(
    std::ofstream& csv,
    run_result const& r,
    uint32_t run_index,
    bool is_warmup,
    uint32_t k,
    size_t buckets,
    std::string const& backend,
    std::string const& ref_path,
    std::string const& query_path,
    std::string const& ref_sha,
    std::string const& query_sha,
    uint64_t seq_bases,
    uint64_t valid_kmers,
    uint64_t invalid_windows,
    std::string const& status,
    std::string const& error
) {
    auto dur = [&](double v) -> std::string {
        return v > 0.0 ? std::to_string(v) : "";
    };
    auto metric = [&](double v) -> std::string {
        return r.valid_metric ? std::to_string(v) : "";
    };
    csv << "1," << run_index << ","                                 // schema_version,run_id
        << "" << "," << (is_warmup ? "warmup" : "measured") << ","  // timestamp_utc,phase
        << "" << ","                                                // dataset
        << ref_path << "," << ref_sha << ","                 // reference_path,reference_sha256
        << query_path << "," << query_sha << ","             // query_path,query_sha256
        << backend << "," << k << "," << buckets << ","      // backend,k,buckets
        << "42," << "" << "," << (is_warmup ? 1 : 0) << ","  // root_seed,derived_seed,warmup
        << run_index << "," << status << ","                 // run_index,status
        << seq_bases << "," << valid_kmers << "," << invalid_windows << ","  // counts
        << dur(r.preprocess_ms) << "," << dur(r.allocation_ms) << "," << dur(r.h2d_ms) << ","
        << dur(r.construction_ms) << "," << dur(r.comparison_ms) << "," << dur(r.d2h_ms) << ","
        << dur(r.metric_ms) << "," << dur(r.total_ms) << ","               // timings
        << r.lower << "," << r.equal << "," << r.higher << ","             // counts
        << metric(r.wkid) << "," << metric(r.ani) << "," << error << ",";  // metrics,error
    // 19 provenance placeholders (git_commit through kernel); 26 total remaining columns.
    for (int i = 0; i < 25; ++i) {
        csv << "" << ",";
    }
    csv << "" << "\n";
}

/// @brief Normalises and validates the backend name.
///
/// The library selects direct global packed CAS as the sole construction mapping; the CTA-local
/// candidate was dropped and is rejected. `gpu` is accepted as an alias for `gpu-global`.
std::string normalise_backend(const std::string& backend) {
    if (backend == "gpu" || backend == "gpu-global") {
        return "gpu";
    }
    if (backend == "cpu") {
        return "cpu";
    }
    if (backend == "gpu-cta") {
        throw std::runtime_error(
            "--backend gpu-cta is unsupported: the library uses direct global construction only"
        );
    }
    throw std::runtime_error("--backend must be gpu-global, gpu, or cpu");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"cuDDL  raw-FASTA end-to-end benchmark"};
        std::string reference, query, backend = "gpu", csv_path;
        uint32_t k = 25;
        size_t buckets = 2048;
        uint32_t runs = 5;
        uint32_t warmup = 1;
        app.add_option("--reference", reference, "Reference FASTA file")->required();
        app.add_option("--query", query, "Query FASTA file")->required();
        app.add_option("--k", k, "K-mer length")->check(CLI::Range(1U, 31U));
        app.add_option("--buckets", buckets, "Bucket count (power of two)");
        app.add_option("--backend", backend, "Backend: gpu or cpu");
        app.add_option("--runs", runs, "Number of measured runs");
        app.add_option("--warmup", warmup, "Number of untimed warm-up runs");
        app.add_option("--csv", csv_path, "Output CSV path")->required();
        CLI11_PARSE(app, argc, argv);

        backend = normalise_backend(backend);
        // 's acceptance anchor is fixed at k=25, BucketCount=2048; the binary instantiates only
        // that geometry, so other values are rejected rather than silently ignored.
        if (k != 25) {
            throw std::runtime_error(
                "--k must be 25 (the acceptance anchor); other geometries are unsupported"
            );
        }
        if (buckets != 2048) {
            throw std::runtime_error("--buckets must be 2048 (the acceptance anchor)");
        }

        // Parsing + checksum happens once; per-run timing starts after (preprocess fixed cost).
        auto t0 = steady_clock_t::now();
        auto const query_parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(query, k));
        auto const ref_parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(reference, k));
        auto const preprocess_ms =
            std::chrono::duration<double, std::milli>(steady_clock_t::now() - t0).count();
        auto const seq_bases = query_parsed.bases + ref_parsed.bases;
        auto const valid_kmers = query_parsed.valid_kmers + ref_parsed.valid_kmers;
        auto const invalid_windows = query_parsed.invalid_windows + ref_parsed.invalid_windows;

        std::ofstream csv(csv_path);
        if (!csv) {
            throw std::runtime_error("cannot open CSV output: " + csv_path);
        }
        csv << "schema_version,run_id,timestamp_utc,phase,dataset,reference_path,reference_sha256,"
               "query_path,query_sha256,backend,k,buckets,root_seed,derived_seed,warmup,run_index,"
               "status,sequence_bases,valid_kmers,invalid_windows,preprocess_ms,allocation_ms,"
               "h2d_ms,construction_ms,comparison_ms,d2h_ms,metric_ms,total_ms,lower,equal,higher,"
               "wkid,ani,error,git_commit,manifest_sha256,bbtools_revision,cuda_version,"
               "driver_version,compiler_version,nvbench_revision,cccl_revision,cuco_revision,"
               "hostname,gpu_name,gpu_uuid,compute_capability,gpu_memory_bytes,"
               "free_gpu_memory_bytes,power_limit_w,persistence_mode,application_clocks,"
               "gpu_temperature_c,other_gpu_processes,cpu_model,cpu_logical_count,cpu_affinity,"
               "host_memory_bytes,os,kernel\n";

        if (backend == "gpu") {
            cuddl::sketch<25, 2048> query_gpu;
            cuddl::sketch<25, 2048> ref_gpu;
            auto const total_runs = warmup + runs;
            for (uint32_t idx = 0; idx < total_runs; ++idx) {
                auto const is_warmup = idx < warmup;
                auto r = run_gpu<2048>(query_gpu, ref_gpu, query_parsed.kmers, ref_parsed.kmers);
                r.preprocess_ms = preprocess_ms;
                r.allocation_ms = 0;
                r.d2h_ms = 0;
                r.metric_ms = 0;
                r.total_ms = r.preprocess_ms + r.allocation_ms + r.h2d_ms + r.construction_ms +
                             r.comparison_ms + r.d2h_ms + r.metric_ms;
                emit_row(
                    csv,
                    r,
                    idx,
                    is_warmup,
                    k,
                    buckets,
                    backend,
                    reference,
                    query,
                    "",
                    "",
                    seq_bases,
                    valid_kmers,
                    invalid_windows,
                    "ok",
                    ""
                );
            }
        } else {
            for (uint32_t idx = 0; idx < warmup + runs; ++idx) {
                auto const is_warmup = idx < warmup;
                auto r = run_cpu<2048>(query_parsed.kmers, ref_parsed.kmers);
                r.preprocess_ms = preprocess_ms;
                r.total_ms = r.preprocess_ms + r.allocation_ms + r.h2d_ms + r.construction_ms +
                             r.comparison_ms + r.d2h_ms + r.metric_ms;
                emit_row(
                    csv,
                    r,
                    idx,
                    is_warmup,
                    k,
                    buckets,
                    backend,
                    reference,
                    query,
                    "",
                    "",
                    seq_bases,
                    valid_kmers,
                    invalid_windows,
                    "ok",
                    ""
                );
            }
        }

        csv.close();
        return 0;
    } catch (std::exception const& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
