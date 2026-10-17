// Exhaustive all-to-all search of a reference database file with one MPI rank per GPU, on one
// node or across several. The first rank of each node reads the file into memory the node's
// ranks share; every rank uploads it, builds a replica and searches the tiles it claims from a
// cuddl::shared_tile_counter, so faster ranks search more. One launch sweeps
// the requested rank counts by splitting the world communicator; ranks outside a count wait.
// With --weak, a count of P ranks searches n*sqrt(P/P_max) of the n references, so the pairs
// per GPU stay about constant (weak scaling). The references are the ones whose hashed IDs come
// first: a pseudorandom sample, each smaller count's a subset of the larger ones'. That keeps the
// share of near-duplicate references, which sets the cost per pair, close to the whole file's.
// Accession order clusters species, so a prefix or a fixed stride can hold far more or fewer
// near-duplicates.
//
// Every rank count is verified before it is timed: each rank downloads its results and reduces
// them to a pair count and an order-independent fingerprint, MPI sums those, and rank 0 checks
// the total pair count and that the fingerprint equals the first count's. A deterministic spread
// of pairs is also compared with the scalar oracle on host rows. NVBench times the search in CPU
// wall time between two barriers, with results left on the devices (`device`) or downloaded per
// tile (`host`); rank 0 reports its measurement.
//
// Each count also records, from untimed runs: every rank's per-tile GPU completion times and its
// wait at the closing barrier (`rank_timeline`, `load_balance`), its search, download and wait
// times (`phases`; a download starts once its tile's search finished), and MPI broadcast and
// barrier costs over its ranks (`communication`). `setup` holds every rank's load, build and
// index times. --thresholds sweeps
// minimum index matches (0: every pair, without the index) at --sweep-ranks ranks and reports
// pairs emitted, bytes written as raw 24-byte records to --write-dir, and output times.

#include "multi_gpu.cuh"

#include <CLI/CLI.hpp>
#include <cuda/buffer>
#include <cuddl/cuddl.cuh>
#include <cuddl/reference_database_file.cuh>
#include <nlohmann/json.hpp>
#include <nvbench/nvbench.cuh>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace multi_gpu;

/// MPI shared memory of one node, freed with the window.
class node_window {
   public:
    node_window() = default;
    explicit node_window(MPI_Win window) : window_(window) {}
    node_window(node_window&& other) noexcept : window_(std::exchange(other.window_, MPI_WIN_NULL)) {}
    node_window& operator=(node_window&&) = delete;
    ~node_window() {
        if (window_ != MPI_WIN_NULL) MPI_Win_free(&window_);
    }

   private:
    MPI_Win window_ = MPI_WIN_NULL;
};

/// Rows of the database file in memory shared by the ranks of a node, and on this rank's device.
struct loaded_rows {
    cuddl::score_compatibility compatibility{};
    uint32_t reference_count = 0;
    node_window window;
    std::span<uint16_t const> host;
    cuda::device_buffer<uint16_t> rows;
    double load_ms = 0, upload_ms = 0;
};

/// MPI_Bcast of @p rows from rank 0 over @p comm, one row per element so counts fit an int.
void broadcast(std::span<uint16_t> rows, MPI_Comm comm) {
    auto row = CUDDL_UNWRAP(cuddl::detail::row_type(buckets));
    mpi_check(MPI_Bcast(rows.data(), static_cast<int>(rows.size() / buckets), row, 0, comm));
    mpi_check(MPI_Type_free(&row));
}

/// The first rank of each node reads the file into memory the node's ranks share, and every rank
/// uploads from there. With 100,000 genomes (410 MB of rows) on an 8-GPU node, a broadcast of
/// the rows took 620-780 ms per rank, and 8 ranks each loading the file took 170-630 ms: 8 reads
/// and 16 copies of 410 MB contend for host memory bandwidth.
loaded_rows load_database(std::string const& path, cuda::stream_ref stream) {
    MPI_Comm node;
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int local = 0;
    mpi_check(MPI_Comm_rank(node, &local));
    mpi_check(MPI_Barrier(MPI_COMM_WORLD));
    auto const start = wall_clock::now();
    std::optional<cuddl::reference_database_file> file;
    struct {
        cuddl::score_compatibility compatibility;
        uint32_t reference_count;
    } header{};
    if (local == 0) {
        file.emplace(CUDDL_UNWRAP(cuddl::reference_database_file::load(path)));
        header = {file->metadata().compatibility, file->metadata().reference_count};
    }
    mpi_check(MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, node));
    auto const count = size_t{header.reference_count} * buckets;
    uint16_t* base = nullptr;
    MPI_Win window = MPI_WIN_NULL;
    mpi_check(MPI_Win_allocate_shared(
        static_cast<MPI_Aint>(local == 0 ? count * sizeof(uint16_t) : 0),
        sizeof(uint16_t),
        MPI_INFO_NULL,
        node,
        &base,
        &window
    ));
    node_window owner(window);
    if (local != 0) {
        MPI_Aint size = 0;
        int unit = 0;
        mpi_check(MPI_Win_shared_query(window, 0, &size, &unit, &base));
    }
    if (file) std::ranges::copy(file->rows(), base);
    file.reset();
    mpi_check(MPI_Barrier(node));
    mpi_check(MPI_Comm_free(&node));
    auto const load_ms = ms_since(start);
    auto const uploaded = wall_clock::now();
    std::span<uint16_t const> const host{base, count};
    auto device_rows =
        cuda::make_device_buffer<uint16_t>(stream, stream.device(), count, cuda::no_init);
    cuda::copy_bytes(stream, host, device_rows);
    stream.sync();
    return {
        header.compatibility,
        header.reference_count,
        std::move(owner),
        host,
        std::move(device_rows),
        load_ms,
        ms_since(uploaded)
    };
}

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
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

struct verification {
    uint64_t pairs = 0, fingerprint = 0, oracle_checked = 0, oracle_mismatches = 0;
};

/// This rank's verification sums, reduced over @p comm onto its rank 0.
verification verify(
    group_type& group,
    tile_claims& claims,
    std::span<uint16_t const> rows,
    uint64_t sample_every,
    MPI_Comm comm
) {
    // One sum per replica, since replicas deliver tiles concurrently.
    std::vector<verification> parts(group.size());
    CUDDL_UNWRAP(group.search_all_to_all(
        [&](cuddl::replica_tile const& tile) {
            auto& part = parts[tile.replica];
            CUDDL_UNWRAP(tile.download())
                ->tile()
                .for_each_passing([&](cuddl::batch_search_result const& result) {
                    auto const pair = (uint64_t{result.query_id} << 32U) | result.reference_id;
                    auto const& c = result.counts;
                    auto const packed =
                        cuddl::packed_pairwise_counts::pack(c.lower, c.equal, c.higher);
                    ++part.pairs;
                    part.fingerprint += mix(pair ^ mix(packed.bits));
                    if (mix(pair) % sample_every == 0) {
                        ++part.oracle_checked;
                        part.oracle_mismatches +=
                            oracle(
                                rows.data() + size_t{result.query_id} * buckets,
                                rows.data() + size_t{result.reference_id} * buckets
                            ) != c;
                    }
                });
        },
        claims.next()
    ));
    verification local;
    for (auto const& part : parts) {
        local.pairs += part.pairs;
        local.fingerprint += part.fingerprint;
        local.oracle_checked += part.oracle_checked;
        local.oracle_mismatches += part.oracle_mismatches;
    }
    verification total;
    // Unsigned sums wrap modulo 2^64, so the fingerprint stays independent of the split.
    mpi_check(MPI_Reduce(&local, &total, 4, MPI_UINT64_T, MPI_SUM, 0, comm));
    return total;
}

/// The @p q quantile of ascending @p values, interpolated linearly between neighbouring samples.
double quantile(std::vector<double> const& values, double q) {
    auto const position = q * static_cast<double>(values.size() - 1U);
    auto const below = static_cast<size_t>(position);
    auto const above = std::min(below + 1U, values.size() - 1U);
    return values[below] +
           (position - static_cast<double>(below)) * (values[above] - values[below]);
}

/// Median, 5th and 95th percentile of @p values, with keys ending in @p unit.
json spread(std::vector<double> values, std::string const& unit = "ms") {
    std::ranges::sort(values);
    return {
        {"median_" + unit, quantile(values, 0.5)},
        {"p5_" + unit, quantile(values, 0.05)},
        {"p95_" + unit, quantile(values, 0.95)},
    };
}

/// NVBench CPU-wall samples of @p function run together by every rank of @p comm. Each sample
/// starts after a barrier, so time other ranks spend between samples stays out of it, and ends
/// after a closing barrier, so it lasts until the slowest rank finishes. NVBench reports no
/// percentiles, so each sample is also timed here; the last `sample_size` of them, after the
/// warmups, give the spread.
json measure(
    std::string const& name,
    int samples,
    int warmups,
    MPI_Comm comm,
    std::function<void()> const& function
) {
    std::vector<double> times;
    auto run = [&](nvbench::state& state, nvbench::type_list<>) {
        state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
            mpi_check(MPI_Barrier(comm));
            timer.start();
            auto const start = wall_clock::now();
            function();
            mpi_check(MPI_Barrier(comm));
            times.push_back(ms_since(start));
            timer.stop();
        });
    };
    nvbench::benchmark<decltype(run)> benchmark(run);
    // A fixed sample count keeps every rank's iterations, and so its barriers, in step.
    benchmark.set_name(name)
        .set_is_cpu_only(true)
        .set_stopping_criterion("sample-count")
        .set_min_samples(samples)
        .set_criterion_param_int64("target-samples", samples)
        .set_timeout(std::numeric_limits<double>::max())
        .set_cold_warmup_runs(warmups)
        .set_skip_batched(true);
    benchmark.run();
    auto const& state = benchmark.get_states().front();
    if (state.is_skipped()) throw std::runtime_error(state.get_skip_reason());
    auto const count = state.get_summary("nv/cpu_only/sample_size").get_int64("value");
    if (count < 1 || static_cast<size_t>(count) > times.size()) {
        throw std::runtime_error("NVBench reported more samples than were timed");
    }
    auto result = spread({times.end() - count, times.end()});
    result["samples"] = count;
    auto const noise =
        state.get_summary("nv/cpu_only/time/cpu/stdev/relative").get_float64("value");
    if (std::isfinite(noise)) result["relative_stddev_percent"] = noise * 100;
    return result;
}

double ms_between(cuda::timed_event const& end, cuda::timed_event const& start) {
    return static_cast<double>((end - start).count()) / 1e6;
}

/// Pairs (query, reference > query) of the all-to-all tile with queries [first, first + count).
uint64_t tile_pairs(uint32_t first, uint32_t count, uint32_t n) {
    return uint64_t{count} * (n - 1U) - uint64_t{count} * first -
           uint64_t{count} * (count - 1U) / 2U;
}

/// When each of this rank's tiles finished on its GPU, from one device-consumer search started
/// together with the other ranks of @p comm. Each callback runs after its tile's search, so an
/// event recorded there marks that tile's end on the device timeline.
json rank_timeline(
    group_type& group,
    tile_claims& claims,
    uint32_t n,
    cuda::stream_ref stream,
    MPI_Comm comm,
    json identity
) {
    struct mark {
        uint32_t first, count;
        cuda::timed_event done;
    };
    std::vector<mark> marks;
    auto split = claims.next();
    mpi_check(MPI_Barrier(comm));
    auto const start = stream.record_timed_event();
    CUDDL_UNWRAP(group.search_all_to_all(
        [&](cuddl::replica_tile const& tile) {
            marks.push_back(
                {tile.tile.first_query_id, tile.tile.query_count, tile.stream.record_timed_event()}
            );
        },
        std::move(split)
    ));
    auto const wait_ms = timed_barrier(comm);
    auto tiles = json::array();
    double previous = 0;
    for (auto const& m : marks) {
        auto const end = ms_between(m.done, start);
        tiles.push_back({
            {"first_query", m.first},
            {"query_count", m.count},
            {"pairs", tile_pairs(m.first, m.count, n)},
            {"start_ms", previous},
            {"end_ms", end},
        });
        previous = end;
    }
    identity["tiles"] = std::move(tiles);
    identity["search_ms"] = previous;
    identity["wait_ms"] = wait_ms;
    return identity;
}

/// This rank's search, download and closing-barrier wait from one host-consumer search.
json rank_phases(group_type& group, tile_claims& claims, MPI_Comm comm, json identity) {
    double download_ms = 0;
    auto split = claims.next();
    mpi_check(MPI_Barrier(comm));
    auto const start = wall_clock::now();
    CUDDL_UNWRAP(group.search_all_to_all(
        [&](cuddl::replica_tile const& tile) {
            // Waiting for the tile's search first keeps that wait out of the download time.
            tile.stream.sync();
            auto const begin = wall_clock::now();
            CUDDL_UNWRAP(tile.download());
            download_ms += ms_since(begin);
        },
        std::move(split)
    ));
    identity["search_ms"] = ms_since(start);
    identity["download_ms"] = download_ms;
    identity["wait_ms"] = timed_barrier(comm);
    return identity;
}

/// MPI broadcast of @p rows from rank 0, and MPI barrier latency, over @p comm, each sample taken
/// as the slowest rank's time.
json communication(std::span<uint16_t> rows, MPI_Comm comm, int samples) {
    constexpr int barriers = 200;
    std::vector<double> broadcast(samples), barrier(barriers);
    for (int i = -1; i < samples; ++i) {
        mpi_check(MPI_Barrier(comm));
        auto const start = wall_clock::now();
        ::broadcast(rows, comm);
        if (i >= 0) broadcast[i] = ms_since(start);
    }
    for (auto& sample : barrier) sample = timed_barrier(comm) * 1000;
    auto const slowest = [&](std::vector<double>& values) {
        mpi_check(MPI_Allreduce(
            MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_MAX, comm
        ));
    };
    slowest(broadcast);
    slowest(barrier);
    auto const median = spread(broadcast)["median_ms"].get<double>();
    return {
        {"broadcast_bytes", rows.size_bytes()},
        {"broadcast", spread(broadcast)},
        {"broadcast_gbps", static_cast<double>(rows.size_bytes()) / (median / 1000) / 1e9},
        {"barrier_us", spread(barrier, "us")},
    };
}

/// Runs @p on_tile over the tiles this rank claims with @p split: every pair when
/// @p minimum_matches is 0, otherwise the pairs with at least that many index matches, filtered
/// on the GPU.
template <typename OnTile>
void search_threshold(
    group_type& group,
    cuddl::replica_search split,
    uint32_t minimum_matches,
    OnTile&& on_tile
) {
    if (minimum_matches == 0) {
        CUDDL_UNWRAP(group.search_all_to_all(on_tile, std::move(split)));
    } else {
        CUDDL_UNWRAP(
            group.search_all_to_all(on_tile, {.minimum_matches = minimum_matches}, std::move(split))
        );
    }
}

/// One instrumented threshold search that downloads every passing pair and writes it to
/// @p path as raw 24-byte records, flushed to the device before the clock stops. The writer's
/// thread writes while the search continues; node-local NVMe gains nothing from more files.
json rank_output(
    group_type& group,
    tile_claims& claims,
    uint32_t minimum_matches,
    std::filesystem::path const& path,
    MPI_Comm comm
) {
    parallel_writer file(path, 1);
    double download_ms = 0, write_ms = 0;
    uint64_t pairs = 0;
    auto split = claims.next();
    mpi_check(MPI_Barrier(comm));
    auto const start = wall_clock::now();
    search_threshold(
        group, std::move(split), minimum_matches, [&](cuddl::replica_tile const& tile) {
            auto const begin = wall_clock::now();
            auto const rows = CUDDL_UNWRAP(tile.download())->passing();
            auto const downloaded = wall_clock::now();
            download_ms += std::chrono::duration<double, std::milli>(downloaded - begin).count();
            file.append(rows.data(), rows.size() * sizeof(rows[0]));
            write_ms += ms_since(downloaded);
            pairs += rows.size();
        }
    );
    auto const flush = wall_clock::now();
    file.finish();
    write_ms += ms_since(flush);
    json result{
        {"pairs", pairs},
        {"download_ms", download_ms},
        {"write_ms", write_ms},
        {"total_ms", ms_since(start)},
    };
    std::filesystem::remove(path);
    mpi_check(MPI_Barrier(comm));
    return result;
}

int run(int argc, char** argv, ranks const& r) {
    CLI::App app{"Exhaustive all-to-all search with one MPI rank per GPU"};
    std::string database_path, output, write_dir;
    std::vector<int> rank_counts;
    std::vector<std::string> consumers{"device", "host"};
    std::vector<uint32_t> thresholds;
    int samples = 5, warmups = 1, sweep_ranks = 0;
    uint64_t oracle_pairs = 1000000;
    bool skip_verify = false, weak = false;
    app.add_option("database", database_path, "Database from cuddl-build-reference-db")->required();
    app.add_option("--ranks", rank_counts, "Rank counts to run (default 1, 2, 4, ... all)");
    app.add_option("--consume", consumers, "Tile consumers to time")
        ->check(CLI::IsMember({"device", "host"}));
    app.add_option("--samples", samples)->check(CLI::Range(1, 1000));
    app.add_option("--warmups", warmups)->check(CLI::Range(0, 100));
    app.add_option("--oracle-pairs", oracle_pairs, "Pairs compared with the host scalar oracle")
        ->check(CLI::PositiveNumber);
    app.add_flag("--skip-verify", skip_verify, "Only time; skip downloading and checking results");
    app.add_flag(
        "--weak",
        weak,
        "Weak scaling: P ranks search a hashed sample of n*sqrt(P/P_max) references, so each "
        "GPU's pair count stays about constant"
    );
    auto* write_option = app.add_option(
        "--write-dir", write_dir, "Node-local directory for the threshold sweep's result files"
    );
    app.add_option(
           "--thresholds",
           thresholds,
           "Minimum index matches to sweep; 0 reports every pair without the index"
    )
        ->needs(write_option);
    app.add_option("--sweep-ranks", sweep_ranks, "Ranks for the threshold sweep (default: most)");
    app.add_option("-o,--output", output, "JSON report destination (written by rank 0)");
    try {
        app.parse(argc, argv);
    } catch (CLI::ParseError const& error) {
        return r.rank == 0 ? app.exit(error) : (error.get_exit_code() == 0 ? 0 : 1);
    }
    if (rank_counts.empty()) {
        for (int n = 1; n < r.size; n *= 2) rank_counts.push_back(n);
        rank_counts.push_back(r.size);
    }
    std::ranges::sort(rank_counts);
    if (sweep_ranks == 0) sweep_ranks = rank_counts.back();
    if (rank_counts.front() < 1 || rank_counts.back() > r.size || sweep_ranks < 1 ||
        sweep_ranks > r.size) {
        throw std::invalid_argument(
            "--ranks and --sweep-ranks must be between 1 and the " + std::to_string(r.size) +
            " MPI ranks"
        );
    }

    nvml_check(nvmlInit_v2());
    auto const device = rank_device(r);
    cuda::stream stream{device};
    auto source = load_database(database_path, stream);
    auto const rows = std::span{source.rows.data(), source.rows.size()};
    auto const build = [&](std::span<uint16_t const> device_rows) {
        std::vector<database> replicas;
        replicas.push_back(
            CUDDL_UNWRAP(database::build_async(device_rows, source.compatibility, stream))
        );
        stream.sync();
        // One replica per rank; the group owns its storage and NUMA-local pinned memory.
        return CUDDL_UNWRAP(group_type::adopt(std::move(replicas)));
    };
    auto const n = source.reference_count;
    auto const build_start = wall_clock::now();
    auto group = build(rows);
    auto const build_ms = ms_since(build_start);
    auto const host_rows = source.host;
    // Broadcast timing writes into this; the node-shared rows must stay as loaded.
    std::vector<uint16_t> broadcast_buffer(host_rows.begin(), host_rows.end());

    uint64_t const pair_count = n > 1 ? uint64_t{n} * (n - 1) / 2 : 0;
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    std::cerr << "rank " << r.rank << "/" << r.size << " on " << host << " GPU " << device.get()
              << " (" << device.name() << ", host NUMA node " << host_numa_node(device)
              << "), references=" << n << '\n';
    json const identity{{"rank", r.rank}, {"host", host}, {"gpu_index", device.get()}};
    auto const nodes_of = [&](int count) {
        // mpirun fills nodes slot by slot, so the first count ranks span these nodes.
        return (count + r.local_size - 1) / r.local_size;
    };
    // Reference IDs in hash order; a weak-scaling count searches a prefix of it.
    std::vector<uint32_t> hash_order(n);
    std::iota(hash_order.begin(), hash_order.end(), 0U);
    std::ranges::sort(hash_order, [](uint32_t a, uint32_t b) {
        return std::pair{mix(a), a} < std::pair{mix(b), b};
    });

    json report{
        {"database", database_path},
        {"mode", weak ? "weak" : "strong"},
        {"references", n},
        {"pairs", pair_count},
        {"gpu", std::string(device.name())},
        {"world_ranks", r.size},
        {"ranks_per_node", r.local_size},
        {"runs", json::array()},
        {"communication", json::array()},
    };
    std::optional<uint64_t> first_fingerprint;
    std::map<std::string, double> first_run_ms;
    for (auto const count : rank_counts) {
        bool const active = r.rank < count;
        MPI_Comm comm;
        mpi_check(MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, r.rank, &comm));
        if (active) {
            auto const references =
                weak ? std::max<uint32_t>(
                           2U,
                           static_cast<uint32_t>(std::lround(
                               n * std::sqrt(static_cast<double>(count) / rank_counts.back())
                           ))
                       )
                     : n;
            uint64_t const pairs = uint64_t{references} * (references - 1) / 2;
            // The first references of hash_order, in ID order, uploaded for this count only.
            std::vector<uint16_t> sampled;
            std::optional<cuda::device_buffer<uint16_t>> sampled_rows;
            std::optional<group_type> sample;
            if (references != n) {
                std::vector<uint32_t> ids(hash_order.begin(), hash_order.begin() + references);
                std::ranges::sort(ids);
                sampled.reserve(size_t{references} * buckets);
                for (auto const id : ids) {
                    auto const row = std::span{host_rows}.subspan(size_t{id} * buckets, buckets);
                    sampled.insert(sampled.end(), row.begin(), row.end());
                }
                sampled_rows.emplace(cuda::make_device_buffer<uint16_t>(stream, device, sampled));
                sample.emplace(build(std::span{sampled_rows->data(), sampled_rows->size()}));
            }
            auto& searched = sample ? *sample : group;
            std::span<uint16_t const> const searched_rows =
                sample ? std::span<uint16_t const>{sampled} : std::span<uint16_t const>{host_rows};
            auto claims = tile_claims::make(comm);
            json entry{
                {"ranks", count},
                {"nodes", nodes_of(count)},
                {"references", references},
                {"pairs", pairs},
                {"tile_queries",
                 cuddl::detail::all_to_all_tile_queries(references, claims.shard.count)},
            };
            if (!skip_verify) {
                auto const sample_every = std::max<uint64_t>(1, pairs / oracle_pairs);
                auto const v = verify(searched, claims, searched_rows, sample_every, comm);
                if (r.rank == 0) {
                    // Weak-scaling runs search different references, so only strong runs share
                    // a fingerprint.
                    if (!weak && !first_fingerprint) first_fingerprint = v.fingerprint;
                    json const matches =
                        weak ? json(nullptr) : json(v.fingerprint == *first_fingerprint);
                    bool const passed = v.pairs == pairs && v.oracle_mismatches == 0 &&
                                        v.oracle_checked > 0 && matches != json(false);
                    entry["verification"] = {
                        {"pairs", v.pairs},
                        {"fingerprint", v.fingerprint},
                        {"matches_first_run_fingerprint", matches},
                        {"oracle_checked", v.oracle_checked},
                        {"oracle_mismatches", v.oracle_mismatches},
                        {"passed", passed},
                    };
                    std::cerr << "ranks=" << count << " verification " << entry["verification"]
                              << '\n';
                    if (!passed) throw std::runtime_error("verification failed");
                }
            }
            for (auto const& consumer : consumers) {
                bool const download = consumer == "host";
                auto timing = measure(
                    "all_to_all_" + consumer + "_" + std::to_string(count) + "_ranks",
                    samples,
                    warmups,
                    comm,
                    [&] {
                        CUDDL_UNWRAP(searched.search_all_to_all(
                            [&](cuddl::replica_tile const& tile) {
                                if (download) CUDDL_UNWRAP(tile.download());
                            },
                            claims.next()
                        ));
                    }
                );
                auto const ms = timing["median_ms"].get<double>();
                auto const base = first_run_ms.try_emplace(consumer, ms).first->second;
                timing["pairs_per_second"] = static_cast<double>(pairs) / (ms / 1000);
                timing["speedup_vs_first_run"] = base / ms;
                // Weak scaling keeps the work per GPU, so its ideal time stays constant.
                timing["parallel_efficiency"] =
                    weak ? base / ms : base / ms * rank_counts.front() / count;
                entry[consumer] = timing;
            }
            auto timeline = gather_json(
                rank_timeline(searched, claims, references, stream, comm, identity), comm
            );
            auto phases = gather_json(rank_phases(searched, claims, comm, identity), comm);
            auto traffic = communication(broadcast_buffer, comm, samples);
            if (r.rank == 0) {
                double longest = 0, total = 0;
                for (auto const& rank : timeline) {
                    longest = std::max(longest, rank["search_ms"].get<double>());
                    total += rank["search_ms"].get<double>();
                }
                auto const mean = total / count;
                entry["load_balance"] = {
                    {"max_rank_ms", longest},
                    {"mean_rank_ms", mean},
                    {"ratio", longest / mean},
                };
                entry["rank_timeline"] = std::move(timeline);
                entry["phases"] = std::move(phases);
                std::cerr << "ranks=" << count << " load_balance " << entry["load_balance"] << '\n';
                traffic["ranks"] = count;
                traffic["nodes"] = nodes_of(count);
                report["communication"].push_back(traffic);
                report["runs"].push_back(entry);
            }
            mpi_check(MPI_Comm_free(&comm));
        }
        mpi_check(MPI_Barrier(MPI_COMM_WORLD));
    }

    double index_ms = 0;
    if (!thresholds.empty()) {
        if (std::ranges::any_of(thresholds, [](uint32_t m) { return m > 0; })) {
            auto const start = wall_clock::now();
            CUDDL_UNWRAP(group.build_index());
            index_ms = ms_since(start);
        }
        report["threshold_sweep"] = json::array();
        bool const active = r.rank < sweep_ranks;
        MPI_Comm comm;
        mpi_check(MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, r.rank, &comm));
        if (active) {
            auto claims = tile_claims::make(comm);
            auto const path =
                std::filesystem::path(write_dir) / ("rank-" + std::to_string(r.rank) + ".bin");
            for (auto const minimum : thresholds) {
                auto timing =
                    measure("threshold_" + std::to_string(minimum), samples, warmups, comm, [&] {
                        search_threshold(
                            group, claims.next(), minimum, [](cuddl::replica_tile const&) {}
                        );
                    });
                auto const outputs =
                    gather_json(rank_output(group, claims, minimum, path, comm), comm);
                if (r.rank == 0) {
                    uint64_t emitted = 0;
                    double download_ms = 0, write_ms = 0, total_ms = 0;
                    for (auto const& rank : outputs) {
                        emitted += rank["pairs"].get<uint64_t>();
                        download_ms = std::max(download_ms, rank["download_ms"].get<double>());
                        write_ms = std::max(write_ms, rank["write_ms"].get<double>());
                        total_ms = std::max(total_ms, rank["total_ms"].get<double>());
                    }
                    json entry{
                        {"ranks", sweep_ranks},
                        {"nodes", nodes_of(sweep_ranks)},
                        {"minimum_matches", minimum},
                        {"indexed", minimum > 0},
                        {"pairs_emitted", emitted},
                        {"bytes_written", emitted * sizeof(cuddl::batch_search_result)},
                        {"search", timing},
                        {"download_ms", download_ms},
                        {"write_ms", write_ms},
                        {"total_ms", total_ms},
                    };
                    std::cerr << "threshold " << entry.dump() << '\n';
                    report["threshold_sweep"].push_back(entry);
                }
            }
            mpi_check(MPI_Comm_free(&comm));
        }
        mpi_check(MPI_Barrier(MPI_COMM_WORLD));
    }

    auto setup = identity;
    setup["load_ms"] = source.load_ms;
    setup["upload_ms"] = source.upload_ms;
    setup["build_ms"] = build_ms;
    setup["index_ms"] = index_ms;
    report["setup"] = gather_json(setup, MPI_COMM_WORLD);
    if (r.rank == 0) {
        if (!output.empty()) std::ofstream(output) << report.dump(2) << '\n';
        std::cout << report.dump(2) << '\n';
    }
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
