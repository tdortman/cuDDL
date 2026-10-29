// Compressed FASTA to filtered all-to-all results on one or more nodes, one MPI rank per GPU,
// through include/cuddl/distributed.cuh.
//
// Each phase is one library step: `list` is cuddl::slice_genomes, `sketch` builds this rank's
// slice, `allgather` is cuddl::gather_sketches, `build` uploads the gathered file to this
// rank's GPU, `index` builds the index for a threshold search, and `search` is
// cuddl::search_all_to_all across the ranks, writing this rank's tiles through a
// cuddl::result_file to `rank-<r>.bin`. `download` is the part of the search spent copying
// finished tiles to the host, converting them on the GPU first for similarity records, and
// `write` the part spent waiting for the file: for a free host buffer while earlier tiles are
// written, and for the flush at the end. The file's threads expand and write records while the
// search continues. With `--write-streams S` above 1, stream s writes `rank-<r>.bin.<s>`, and
// each tile goes whole to the next stream in turn.
//
// `--records similarity` (the default) writes cuddl::result_format::similarity, each pair's
// weighted k-mer identity as a 4-byte float, and `--records counts` the 24-byte
// cuddl::batch_search_result records with the raw bucket counts. Phases are separated by
// barriers; each rank records its own time per phase.

#include "multi_gpu.cuh"

#include <CLI/CLI.hpp>
#include <cuddl/distributed.cuh>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace multi_gpu;
namespace fs = std::filesystem;

/// Per-phase wall time of this rank, between barriers of the world.
class phase_meter {
   public:
    phase_meter() {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD));
        first_ = wall_clock::now();
    }

    template <typename Body>
    void run(std::string const& name, Body&& body) {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD));
        auto const start = wall_clock::now();
        body();
        ms_[name] = ms_since(start);
    }

    /// Wall time from the first barrier to now, after a closing barrier.
    double total_ms() {
        mpi_check(MPI_Barrier(MPI_COMM_WORLD));
        return ms_since(first_);
    }

    std::map<std::string, double> const& ms() const noexcept {
        return ms_;
    }

   private:
    wall_clock::time_point first_;
    std::map<std::string, double> ms_;
};

unsigned default_workers(ranks const& r) {
    if (auto const* cpus = std::getenv("SLURM_CPUS_PER_TASK")) return std::stoul(cpus);
    return std::max(1U, std::thread::hardware_concurrency() / static_cast<unsigned>(r.local_size));
}

int run(int argc, char** argv, ranks const& r) {
    CLI::App app{"Sketch, gather and search all genomes against each other on every rank's GPU"};
    std::string input, output_dir, output;
    uint32_t minimum_matches = 0;
    unsigned workers = default_workers(r);
    bool keep_output = false;
    bool direct_reads = false;
    unsigned write_streams = 3;
    std::string records = "similarity";
    app.add_option("--records", records, "Result records: similarity or raw bucket counts")
        ->check(CLI::IsMember({"similarity", "counts"}));
    app.add_option("genomes", input, "Genome directory, or a file listing one path per line")
        ->required()
        ->check(CLI::ExistingPath);
    app.add_option(
        "--minimum-matches", minimum_matches, "Index-match threshold; 0 reports every pair"
    );
    app.add_option("--workers", workers, "Parser threads per rank")->check(CLI::PositiveNumber);
    std::map<std::string, cuddl::decompression_backend> const backends{
        {"automatic", cuddl::decompression_backend::automatic},
        {"hybrid", cuddl::decompression_backend::hybrid},
        {"cpu", cuddl::decompression_backend::cpu},
        {"gpu", cuddl::decompression_backend::gpu},
        {"gpu-cores", cuddl::decompression_backend::gpu_cores},
    };
    std::string decompression_name = "automatic";
    app.add_option("--decompression", decompression_name, "Decompression backend")
        ->check(CLI::IsMember(backends));
    app.add_flag("--direct-reads", direct_reads, "Read genomes with O_DIRECT, past the page cache");
    app.add_option("--output-dir", output_dir, "Directory for the per-rank result files")
        ->required();
    app.add_flag("--keep-output", keep_output, "Keep the result files instead of deleting them");
    app.add_option(
           "--write-streams",
           write_streams,
           "Result files this rank writes in parallel; a shared file system needs several"
    )
        ->check(CLI::PositiveNumber);
    app.add_option("-o,--output", output, "JSON report destination (written by rank 0)");
    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError const& error) {
        return r.rank == 0 ? app.exit(error) : (error.get_exit_code() == 0 ? 0 : 1);
    }
    auto const decompression = backends.at(decompression_name);

    nvml_check(nvmlInit_v2());
    auto const device = rank_device(r);
    cuda::stream stream{device};
    phase_meter meter;
    std::optional<cuddl::genome_slice> slice;
    meter.run("list", [&] {
        slice.emplace(CUDDL_UNWRAP(cuddl::slice_genomes(input, MPI_COMM_WORLD, workers)));
    });
    auto const& bytes = slice->bytes;
    uint64_t const slice_bytes =
        std::accumulate(bytes.begin() + slice->first, bytes.begin() + slice->last, uint64_t{0});

    std::optional<cuddl::reference_database_file> sketches;
    meter.run("sketch", [&] {
        sketches.emplace(CUDDL_UNWRAP((cuddl::reference_database_file::build<k, buckets, layout>(
            slice->mine(),
            stream,
            {.parser_workers = workers,
             .decompression = decompression,
             .direct_reads = direct_reads}
        ))));
    });

    auto const n = static_cast<uint32_t>(slice->paths.size());
    std::optional<cuddl::reference_database_file> gathered;
    meter.run("allgather", [&] {
        gathered.emplace(CUDDL_UNWRAP(cuddl::gather_sketches(*sketches, *slice, MPI_COMM_WORLD)));
    });

    std::optional<group_type> group;
    meter.run("build", [&] {
        group.emplace(CUDDL_UNWRAP(group_type::upload(*gathered, std::array{device})));
    });
    if (minimum_matches > 0 && !cuddl::detail::ablated("inverted-index")) {
        meter.run("index", [&] { CUDDL_UNWRAP(group->build_index()); });
    }

    auto const path = fs::path(output_dir) / ("rank-" + std::to_string(r.rank) + ".bin");
    cuddl::result_file_statistics written;
    meter.run("search", [&] {
        auto file = CUDDL_UNWRAP(
            cuddl::result_file::open(
                path,
                {.format = records == "counts" ? cuddl::result_format::counts
                                               : cuddl::result_format::similarity,
                 .streams = write_streams,
                 .durable = true,
                 .threads = workers}
            )
        );
        auto const on_tile = [&](cuddl::replica_tile const& tile) {
            // Waiting for the tile's search first keeps that wait out of the download time.
            tile.stream.sync();
            CUDDL_UNWRAP(file.write(tile));
        };
        if (minimum_matches == 0) {
            CUDDL_UNWRAP(cuddl::search_all_to_all(*group, on_tile, MPI_COMM_WORLD));
        } else {
            CUDDL_UNWRAP(
                cuddl::search_all_to_all(
                    *group, on_tile, {.minimum_matches = minimum_matches}, MPI_COMM_WORLD
                )
            );
        }
        CUDDL_UNWRAP(file.finish());
        written = file.statistics();
    });
    auto const wall_ms = meter.total_ms();
    if (!keep_output) {
        for (auto const& file : cuddl::result_file::files(path, write_streams)) fs::remove(file);
    }

    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    json mine{
        {"rank", r.rank},
        {"host", host},
        {"gpu_index", device.get()},
        {"files", slice->last - slice->first},
        {"input_bytes", slice_bytes},
        {"download_ms", written.download_seconds * 1e3},
        {"write_ms", written.stall_seconds * 1e3},
        {"pairs", written.pairs},
        {"bytes_written", written.bytes},
    };
    for (auto const& [name, ms] : meter.ms()) mine[name + "_ms"] = ms;
    auto detail = gather_json(mine, MPI_COMM_WORLD);
    if (r.rank != 0) return 0;

    auto phases = json::object();
    uint64_t pairs_total = 0;
    uint64_t bytes_total = 0;
    for (auto const& rank : detail) {
        for (auto const* name :
             {"list", "sketch", "allgather", "build", "index", "search", "download", "write"}) {
            auto const key = std::string(name) + "_ms";
            if (!rank.contains(key)) continue;
            phases[name] = std::max(phases.value(name, 0.0), rank[key].get<double>());
        }
        pairs_total += rank["pairs"].get<uint64_t>();
        bytes_total += rank["bytes_written"].get<uint64_t>();
    }
    json report{
        {"genomes", n},
        {"input_bytes", std::accumulate(bytes.begin(), bytes.end(), uint64_t{0})},
        {"nodes", (r.size + r.local_size - 1) / r.local_size},
        {"ranks", r.size},
        {"ranks_per_node", r.local_size},
        {"gpu", std::string(device.name())},
        {"workers_per_rank", workers},
        {"decompression", decompression_name},
        {"minimum_matches", minimum_matches},
        {"indexed", minimum_matches > 0},
        {"records", records},
        {"write_streams", write_streams},
        {"pairs_total", pairs_total},
        {"bytes_written_total", bytes_total},
        {"wall_ms", wall_ms},
        {"phases_wall_ms", phases},
        {"ranks_detail", detail},
    };
    std::cerr << "genomes=" << n << " ranks=" << r.size << " pairs=" << pairs_total
              << " wall_ms=" << wall_ms << " phases=" << phases.dump() << '\n';
    if (!output.empty()) std::ofstream(output) << report.dump(2) << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int status = 1;
    ranks r;
    try {
        r = world_ranks();
        status = run(argc, argv, r);
    } catch (std::exception const& error) {
        std::cerr << "rank " << r.rank << ": Error: " << error.what() << '\n';
        // Other ranks may be blocked in a collective; take them down instead of hanging.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
