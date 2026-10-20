#include <CLI/CLI.hpp>

#include <cuddl/detail/worker_pool.hpp>
#include <cuddl/query_sketch.cuh>
#include <cuddl/reference_index_file.cuh>
#include <cuddl/replicated_database.cuh>
#include <cuddl/top_k.cuh>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "cli_config.hpp"

namespace {

// Binary output rows are raw little-endian batch_search_result records, documented in README.md.
static_assert(sizeof(cuddl::batch_search_result) == 24);
static_assert(std::is_trivially_copyable_v<cuddl::batch_search_result>);

/// Query genomes sketched per batch; each batch then runs as one batched search.
constexpr size_t query_batch_size = 4096;

using database_type = cuddl::reference_database<cli::kmer_length, cli::buckets, cli::layout>;
using index_type = cuddl::reference_index<cli::kmer_length, cli::buckets, cli::layout>;
using query_type = cuddl::query_sketch_batch<cli::kmer_length, cli::buckets, cli::layout>;
using replicated_type = cuddl::replicated_database<cli::kmer_length, cli::buckets, cli::layout>;

/// Writes result rows as TSV on standard output, or as binary records to a file.
class result_writer {
   public:
    explicit result_writer(std::filesystem::path const& output) {
        if (output.empty()) {
            std::cout << "query_id\treference_id\tlower\tequal\thigher\tboth_empty\n";
            return;
        }
        binary_.open(output, std::ios::binary | std::ios::trunc);
        if (!binary_) throw std::runtime_error("cannot open " + output.string());
    }

    /// Writes the passing results of a downloaded tile. Binary records are expanded from the
    /// packed results by a pool of threads that lives as long as the writer, each taking runs of
    /// query rows, into one buffer reused across tiles, so neither one thread nor first-touch
    /// page faults bound the output.
    void write(cuddl::host_batch_results const& host) {
        auto const tile = host.tile();
        if (!binary_.is_open()) {
            tile.for_each_passing([](cuddl::batch_search_result const& result) {
                auto const& counts = result.counts;
                std::cout << result.query_id << '\t' << result.reference_id << '\t' << counts.lower
                          << '\t' << counts.equal << '\t' << counts.higher << '\t'
                          << counts.both_empty << '\n';
            });
            return;
        }
        auto const total = tile.count();
        if (total > capacity_) {
            records_ = std::make_unique_for_overwrite<cuddl::batch_search_result[]>(total);
            capacity_ = total;
        }
        // Row runs of about equal result counts, several per thread for upper-triangle tiles.
        auto const runs = std::min<uint64_t>(uint64_t{threads} * 8U, tile.query_count);
        bounds_.assign(runs + 1U, tile.query_count);
        auto const rows = std::views::iota(uint32_t{0}, tile.query_count);
        for (uint64_t r = 0; r < runs; ++r) {
            bounds_[r] = *std::ranges::partition_point(rows, [&](uint32_t query) {
                return tile.passing_before(query) * runs < total * r;
            });
        }
        pool_.for_each(runs, [&](size_t r) {
            auto* out = records_.get() + tile.passing_before(bounds_[r]);
            tile.for_each_passing(
                bounds_[r], bounds_[r + 1U], [&](cuddl::batch_search_result const& result) {
                    *out++ = result;
                }
            );
        });
        binary_.write(
            reinterpret_cast<char const*>(records_.get()),
            static_cast<std::streamsize>(total * sizeof(cuddl::batch_search_result))
        );
    }

    /// Writes result rows already expanded on the host, such as each genome's top-k neighbours.
    void write(std::span<cuddl::batch_search_result const> rows) {
        if (binary_.is_open()) {
            binary_.write(
                reinterpret_cast<char const*>(rows.data()),
                static_cast<std::streamsize>(rows.size_bytes())
            );
            return;
        }
        for (auto const& result : rows) {
            auto const& counts = result.counts;
            std::cout << result.query_id << '\t' << result.reference_id << '\t' << counts.lower
                      << '\t' << counts.equal << '\t' << counts.higher << '\t' << counts.both_empty
                      << '\n';
        }
    }

    void finish() {
        if (binary_.is_open() ? !binary_.flush() : !std::cout) {
            throw std::runtime_error("cannot write search results");
        }
    }

   private:
    unsigned const threads = std::max(1U, std::thread::hardware_concurrency());
    std::ofstream binary_;
    std::unique_ptr<cuddl::batch_search_result[]> records_;
    uint64_t capacity_ = 0;
    std::vector<uint32_t> bounds_;
    cuddl::detail::worker_pool pool_{threads};
};

/// @brief Writes all-to-all tiles on a thread of its own, so a replica searches its next tile
/// while the host expands and writes the previous one.
///
/// Each replica downloads into two pinned buffers of this writer's, alternately: the replica's
/// own buffer is reused by its next download, which would otherwise wait for the write. Tiles are
/// written in the order they are pushed.
class overlapped_writer {
   public:
    explicit overlapped_writer(result_writer& writer)
        : writer_(writer), thread_([this] { run(); }) {}
    overlapped_writer(overlapped_writer const&) = delete;
    overlapped_writer& operator=(overlapped_writer const&) = delete;
    ~overlapped_writer() {
        stop();
    }

    /// Downloads @p tile, once a buffer of its replica is free, and queues it for writing.
    void push(cuddl::replica_tile const& tile) {
        slot* free = nullptr;
        {
            std::unique_lock lock(mutex_);
            auto& [slots, next] = replicas_[tile.replica];
            free = &slots[next];
            next ^= 1U;
            written_.wait(lock, [&] { return !free->queued || failure_; });
            if (failure_) std::rethrow_exception(failure_);
            if (!free->host) free->host.emplace(tile.stream);
        }
        CUDDL_UNWRAP(cuddl::download(tile.tile, tile.stream, *free->host));
        {
            std::lock_guard lock(mutex_);
            free->queued = true;
            queue_.push_back(free);
        }
        queued_.notify_one();
    }

    /// Writes every queued tile and flushes the output.
    void finish() {
        stop();
        if (failure_) std::rethrow_exception(failure_);
        writer_.finish();
    }

   private:
    struct slot {
        std::optional<cuddl::host_batch_results> host;
        bool queued = false;
    };

    void stop() {
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        queued_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    void run() {
        while (true) {
            slot* next = nullptr;
            {
                std::unique_lock lock(mutex_);
                queued_.wait(lock, [&] { return done_ || !queue_.empty(); });
                if (queue_.empty()) return;
                next = queue_.front();
                queue_.pop_front();
            }
            try {
                writer_.write(*next->host);
            } catch (...) {
                std::lock_guard lock(mutex_);
                failure_ = std::current_exception();
                written_.notify_all();
                return;
            }
            {
                std::lock_guard lock(mutex_);
                next->queued = false;
            }
            written_.notify_all();
        }
    }

    result_writer& writer_;
    std::mutex mutex_;
    std::condition_variable queued_, written_;
    std::map<uint32_t, std::pair<std::array<slot, 2>, uint32_t>> replicas_;
    std::deque<slot*> queue_;
    std::exception_ptr failure_;
    bool done_ = false;
    std::thread thread_;
};

void search(
    database_type const& database,
    index_type const* index,
    std::vector<std::filesystem::path> const& paths,
    result_writer& writer,
    uint32_t minimum_matches,
    uint32_t top_k,
    unsigned workers,
    cuddl::decompression_backend decompression,
    std::optional<std::reference_wrapper<cuddl::device_blacklist const>> blacklist,
    cuda::stream_ref stream
) {
    auto const batch_capacity = std::min(query_batch_size, paths.size());
    auto const requirements = CUDDL_UNWRAP(
        database.batch_search_requirements(static_cast<uint32_t>(batch_capacity), stream, index)
    );
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), requirements.maximum_pair_count, cuda::no_init
    );
    // Each tile's results are written before the next tile reuses the storage; the pinned host
    // copy is reused too. A top-k search keeps each query's neighbours on the device instead.
    cuddl::host_batch_results host(stream);
    std::optional<cuddl::top_k_search> neighbours;
    if (top_k != 0U) {
        neighbours.emplace(CUDDL_UNWRAP(
            cuddl::top_k_search::make(
                static_cast<uint32_t>(paths.size()),
                database.metadata().reference_count,
                cli::buckets,
                top_k,
                stream
            )
        ));
    }
    auto write_tile = [&](cuddl::batch_result_tile const& tile) {
        if (neighbours) {
            CUDDL_UNWRAP(neighbours->add(tile, stream));
            return;
        }
        CUDDL_UNWRAP(cuddl::download(tile, stream, host));
        writer.write(host);
    };
    // Without an index or a threshold every pair is reported, so the dedicated exhaustive
    // kernels run instead of counting matches only to discard nothing.
    bool const exhaustive = index == nullptr && minimum_matches == 0;
    for (size_t first = 0; first < paths.size(); first += query_batch_size) {
        auto batch_size = std::min(query_batch_size, paths.size() - first);
        auto queries = CUDDL_UNWRAP(
            query_type::sketch(
                std::span{paths.data() + first, batch_size},
                stream,
                {.parser_workers = workers, .decompression = decompression, .blacklist = blacklist}
            )
        );
        if (exhaustive) {
            CUDDL_UNWRAP(database.search_batch_async(
                queries.scores(),
                queries.compatibility(),
                static_cast<uint32_t>(first),
                {workspace.data(), workspace.size()},
                {results.data(), results.size()},
                write_tile,
                {},
                stream
            ));
        } else {
            CUDDL_UNWRAP(database.search_batch_async(
                queries.scores(),
                queries.compatibility(),
                static_cast<uint32_t>(first),
                {workspace.data(), workspace.size()},
                {results.data(), results.size()},
                write_tile,
                {},
                {.minimum_matches = minimum_matches},
                stream,
                index
            ));
        }
    }
    if (neighbours) writer.write(CUDDL_UNWRAP(neighbours->download(stream)));
    writer.finish();
}

/// Searches every unique database-row pair (query_id < reference_id) on @p devices, each holding
/// a replica. Tiles are written in query order, so the output does not depend on the devices.
/// A top-k search instead keeps each row's neighbours on every replica's device and merges them.
void search_all_to_all(
    cuddl::reference_database_file const& file,
    std::filesystem::path const& index_path,
    result_writer& writer,
    uint32_t minimum_matches,
    uint32_t top_k,
    std::span<cuda::device_ref const> devices
) {
    auto group = CUDDL_UNWRAP(replicated_type::upload(file, devices));
    if (!index_path.empty()) CUDDL_UNWRAP(group.load_index(index_path));
    auto run = [&](auto const& on_tile, const cuddl::replica_search& split) {
        // Without an index or a threshold every pair is reported, so the exhaustive search runs.
        if (index_path.empty() && minimum_matches == 0) {
            CUDDL_UNWRAP(group.search_all_to_all(on_tile, split));
        } else {
            CUDDL_UNWRAP(
                group.search_all_to_all(on_tile, {.minimum_matches = minimum_matches}, split)
            );
        }
    };
    if (top_k == 0U) {
        overlapped_writer overlapped(writer);
        run([&](cuddl::replica_tile const& tile) { overlapped.push(tile); },
            {.in_query_order = true});
        overlapped.finish();
        return;
    }
    auto const rows = file.metadata().reference_count;
    std::vector<cuda::stream> streams;
    std::vector<cuddl::top_k_search> neighbours;
    streams.reserve(devices.size());
    for (auto const device : devices) {
        auto& stream = streams.emplace_back(device);
        neighbours.push_back(
            CUDDL_UNWRAP(cuddl::top_k_search::make(rows, rows, cli::buckets, top_k, stream))
        );
        stream.sync();
    }
    // Replicas call back concurrently, each with its own tiles and collector.
    run([&](
            cuddl::replica_tile const& tile
        ) { CUDDL_UNWRAP(neighbours[tile.replica].add(tile.tile, tile.stream)); },
        {});
    std::vector<cuddl::batch_search_result> kept;
    for (size_t replica = 0; replica < devices.size(); ++replica) {
        auto const part = CUDDL_UNWRAP(neighbours[replica].download(streams[replica]));
        kept.insert(kept.end(), part.begin(), part.end());
    }
    writer.write(cuddl::select_top_k(std::move(kept), top_k));
    writer.finish();
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Build persistent retrieval indexes or search a validated index file"};
    app.require_subcommand(1);
    std::string database_path, index_path;
    auto format = cuddl::index_storage::automatic;
    std::vector<std::filesystem::path> queries;
    std::filesystem::path output;
    uint32_t minimum_matches = 1;
    uint32_t top_k = 0;
    unsigned workers = cuddl::default_parser_workers();
    auto decompression = cuddl::decompression_backend::automatic;
    auto* build = app.add_subcommand(
        "build", "Create a dense or sparse index file from a reference database"
    );
    build->add_option("database", database_path)->required()->check(CLI::ExistingFile);
    build->add_option("-o,--output", index_path, "Index file destination (replaces existing file)")
        ->required();
    build
        ->add_option(
            "--format", format, "Index storage; automatic picks the one using less device memory"
        )
        ->transform(
            CLI::CheckedTransformer(
                std::map<std::string, cuddl::index_storage>{
                    {"automatic", cuddl::index_storage::automatic},
                    {"dense", cuddl::index_storage::dense},
                    {"sparse", cuddl::index_storage::sparse}
                }
            )
        )
        ->default_str("automatic");
    auto* query = app.add_subcommand(
        "search", "Load once and search all query genomes; format is detected from the index"
    );
    query->add_option("database", database_path)->required()->check(CLI::ExistingFile);
    query->add_option("--index", index_path, "Optional acceleration index file")
        ->check(CLI::ExistingFile);
    auto* query_option =
        query
            ->add_option(
                "--query", queries, "One query genome per FASTA/FASTQ file, in supplied order"
            )
            ->check(CLI::ExistingFile);
    bool all_to_all = false;
    auto* all_to_all_flag =
        query
            ->add_flag(
                "--all-to-all",
                all_to_all,
                "Search database rows against each other, each unordered pair once, instead of "
                "--query"
            )
            ->excludes(query_option);
    std::vector<int> device_ids{0};
    query
        ->add_option(
            "--devices",
            device_ids,
            "GPU ordinals sharing an --all-to-all search, each holding a replica"
        )
        ->needs(all_to_all_flag)
        ->check(CLI::NonNegativeNumber)
        ->default_str("0");
    query
        ->add_option(
            "--minimum-matches",
            minimum_matches,
            "Required matching indexed buckets; zero searches all references"
        )
        ->default_val(minimum_matches);
    query->add_flag(
        "--top-k{3}",
        top_k,
        "Keep each genome's K most similar genomes by weighted k-mer identity instead of every "
        "passing pair; --top-k alone keeps 3, --top-k=K keeps K"
    );
    query->add_option(
        "-o,--output",
        output,
        "Write binary result records here instead of TSV on standard output (replaces file)"
    );
    query->add_option("--workers", workers, "Concurrent query genome loaders")
        ->check(CLI::PositiveNumber);
    (*query)
        .add_option(
            "--decompression", decompression, "Decompression backend; GPU allows format fallbacks"
        )
        ->transform(
            CLI::CheckedTransformer(
                std::map<std::string, cuddl::decompression_backend>{
                    {"automatic", cuddl::decompression_backend::automatic},
                    {"cpu", cuddl::decompression_backend::cpu},
                    {"gpu", cuddl::decompression_backend::gpu},
                    {"gpu-cores", cuddl::decompression_backend::gpu_cores},
                    {"hybrid", cuddl::decompression_backend::hybrid}
                }
            )
        )
        ->default_str("automatic");
    app.set_config(
        "--config", "", "TOML options file; long query lists go under [search] as query = [...]"
    );
    CLI11_PARSE(app, argc, argv);
    try {
        if (*query && !all_to_all && queries.empty()) {
            throw std::invalid_argument("search needs --query or --all-to-all");
        }
        if (*build && std::filesystem::exists(index_path) &&
            std::filesystem::equivalent(database_path, index_path)) {
            throw std::invalid_argument("index output must not overwrite the reference database");
        }
        // Creating a CUDA context takes a few hundred milliseconds, so it overlaps loading the
        // file. A device that fails to initialise reports the error again where it is used.
        std::jthread initialise_devices([&] {
            try {
                for (auto const id : all_to_all ? device_ids : std::vector<int>{0}) {
                    if (id < static_cast<int>(cuda::devices.size())) cuda::devices[id].init();
                }
            } catch (cuda::cuda_error const&) {}
        });
        auto const file = CUDDL_UNWRAP(cuddl::reference_database_file::load(database_path));
        auto const compatibility = file.metadata().compatibility;
        cli::require_compatible(compatibility);
        if (minimum_matches > compatibility.indexed_bucket_count) {
            throw std::invalid_argument("minimum matches exceeds indexed bucket count");
        }
        if (*query && all_to_all) {
            std::vector<cuda::device_ref> devices;
            for (auto const id : device_ids) {
                if (id >= static_cast<int>(cuda::devices.size())) {
                    throw std::invalid_argument(
                        "--devices names GPU " + std::to_string(id) + " but only " +
                        std::to_string(cuda::devices.size()) + " are visible"
                    );
                }
                devices.push_back(cuda::devices[id]);
            }
            result_writer writer(output);
            search_all_to_all(file, index_path, writer, minimum_matches, top_k, devices);
            return 0;
        }
        cuda::stream stream{cuda::devices[0]};
        auto database =
            CUDDL_UNWRAP((file.upload<cli::kmer_length, cli::buckets, cli::layout>(stream)));
        if (*build) {
            auto index = CUDDL_UNWRAP(index_type::build_async(database, stream, format));
            CUDDL_UNWRAP(cuddl::reference_index_file::save(index, database, index_path, stream));
            std::cout << "Saved "
                      << (index.storage() == cuddl::index_storage::dense ? "dense" : "sparse")
                      << " index to " << index_path << '\n';
            return 0;
        }
        std::optional<index_type> index;
        if (!index_path.empty()) {
            index.emplace(
                CUDDL_UNWRAP(cuddl::reference_index_file::load(index_path, database, stream))
            );
        }
        std::optional<cuddl::device_blacklist> blacklist;
        if (!file.blacklist().keys().empty()) {
            blacklist.emplace(file.blacklist(), cli::buckets, stream);
        }
        result_writer writer(output);
        search(
            database,
            index ? &*index : nullptr,
            queries,
            writer,
            minimum_matches,
            top_k,
            workers,
            decompression,
            blacklist ? std::optional{std::cref(*blacklist)} : std::nullopt,
            stream
        );
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
