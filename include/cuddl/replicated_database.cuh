#pragma once

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <condition_variable>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include <cuddl/reference_index_file.cuh>

namespace cuddl {

/// @brief One all-to-all tile as a @ref replicated_database search delivers it.
struct replica_tile {
    /// Results in the replica's device memory; valid only during the callback.
    batch_result_tile tile;
    /// Stream ordering work on the tile's storage; work enqueued here finishes before reuse.
    cuda::stream_ref stream;
    /// Position of the replica's device in the devices the group was made for.
    uint32_t replica{};
    host_batch_results* host{};

    /// @brief Downloads the tile into the replica's reusable pinned buffer, which lives on its
    /// device's host NUMA node, and returns that buffer; valid until the replica's next tile.
    [[nodiscard]] Result<host_batch_results const*> download() const {
        CUDDL_TRY(cuddl::download(tile, stream, *host));
        return static_cast<host_batch_results const*>(host);
    }
};

/// @brief How a @ref replicated_database search splits and delivers its tiles.
struct replica_search {
    /// This process's part of the pairs, when several processes share the search.
    all_to_all_shard shard{};
    /// Deliver tiles one at a time in increasing query order, for example to write output
    /// that does not depend on the device count. Each replica then waits for its turn before
    /// its callback, so callbacks no longer overlap. Not available with `next_tile`.
    bool in_query_order = false;
    /// Claim tiles from a counter that several processes share instead of keeping the tiles
    /// `shard` assigns, so faster processes do more of the search; see
    /// @ref all_to_all_tile_queue. `shard.count` must still give the number of processes.
    std::function<uint64_t()> next_tile{};
};

/**
 * @brief One reference database replicated on explicitly chosen devices, searched all-to-all
 * by all of them at once.
 *
 * Each device gets a replica with its own stream, result storage, optional index, and pinned
 * host memory on the device's host NUMA node, so downloads stay off the socket interconnect.
 * A search runs one host thread per replica. Replicas claim tiles from one
 * @ref all_to_all_tile_queue, largest first, so devices of different speeds share the work:
 * a faster device simply claims more. Kernels pick their launch shapes from each device's own
 * properties, so the devices may differ in architecture and memory size; each needs room for
 * the rows and its result storage, and a device without it fails the search with its error.
 *
 * To split a search across processes, for example MPI ranks on several nodes, give each
 * process its own group and pass every process a different @ref all_to_all_shard. To let
 * processes of different speeds share the work, also pass every process the same
 * `next_tile` counter, such as a shared_tile_counter from distributed.cuh; there,
 * cuddl::search_all_to_all sets both up from an MPI communicator.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class replicated_database {
   public:
    using database_type = reference_database<K, BucketCount, Layout>;
    using index_type = reference_index<K, BucketCount, Layout>;

    /// @brief Uploads @p file to each of @p devices, concurrently.
    [[nodiscard]] static Result<replicated_database>
    upload(reference_database_file const& file, std::span<cuda::device_ref const> devices) {
        auto group = CUDDL_TRY(make(devices));
        CUDDL_TRY(group.on_each([&](replica& r, uint32_t) -> Result<void> {
            r.database.emplace(CUDDL_TRY((file.upload<K, BucketCount, Layout>(r.stream))));
            return Ok();
        }));
        return Result<replicated_database>::ok(std::move(group));
    }

    /// @brief Adopts databases that already hold identical rows, one per device, each searched
    /// on the device that built it (@ref reference_database::device). The streams that built
    /// them must outlive the group.
    [[nodiscard]] static Result<replicated_database> adopt(std::vector<database_type> databases) {
        std::vector<cuda::device_ref> devices;
        for (auto const& database : databases) {
            if (database.metadata() != databases.front().metadata()) {
                return Err(Error::invalid_argument("replicas must share their database metadata"));
            }
            devices.push_back(database.device());
        }
        auto group = CUDDL_TRY(make(devices));
        for (size_t i = 0; i < databases.size(); ++i) {
            group.replicas_[i]->database.emplace(std::move(databases[i]));
        }
        return Result<replicated_database>::ok(std::move(group));
    }

    /// @brief Builds an index on every replica.
    [[nodiscard]] Result<void> build_index(index_storage storage = index_storage::automatic) {
        return on_each([&](replica& r, uint32_t) -> Result<void> {
            r.index.emplace(CUDDL_TRY(index_type::build_async(*r.database, r.stream, storage)));
            r.stream.sync();
            return Ok();
        });
    }

    /// @brief Loads the index file at @p path onto every replica.
    [[nodiscard]] Result<void> load_index(std::filesystem::path const& path) {
        return on_each([&](replica& r, uint32_t) -> Result<void> {
            r.index.emplace(CUDDL_TRY(reference_index_file::load(path, *r.database, r.stream)));
            return Ok();
        });
    }

    /**
     * @brief Exhaustively searches every unique database-row pair in `split.shard` on all
     * replicas.
     *
     * @p on_tile receives each tile as a @ref replica_tile. By default replicas call it
     * concurrently from their own threads, in no particular order, so it must be thread-safe;
     * with `split.in_query_order` they call it one tile at a time in increasing query order.
     * Like the single-database search, it must consume the tile before returning or enqueue
     * dependent work on the tile's stream. An exception from @p on_tile stops every replica and
     * is rethrown here.
     */
    template <typename OnTile>
    [[nodiscard]] Result<void> search_all_to_all(OnTile&& on_tile, replica_search split = {}) {
        return search(on_tile, split, std::nullopt);
    }

    /// @brief Threshold search of every unique pair in `split.shard`: pairs with at least
    /// `options.minimum_matches` matches, through each replica's index when one is loaded.
    template <typename OnTile>
    [[nodiscard]] Result<void>
    search_all_to_all(OnTile&& on_tile, search_options options, replica_search split = {}) {
        return search(on_tile, split, options);
    }

    /// @brief Number of replicas, one per device.
    [[nodiscard]] size_t size() const noexcept {
        return replicas_.size();
    }

    /// @brief The devices, in replica order.
    [[nodiscard]] std::vector<cuda::device_ref> devices() const {
        std::vector<cuda::device_ref> devices;
        for (auto const& r : replicas_) devices.push_back(r->stream.device());
        return devices;
    }

    /// @brief The database every replica holds, as uploaded to replica 0.
    [[nodiscard]] database_type const& database() const noexcept {
        return *replicas_.front()->database;
    }

   private:
    struct replica {
        explicit replica(cuda::device_ref device)
            : stream(device),
              host(
                  stream,
                  detail::device_pinned_pool(device, detail::pinned_placement::device_node)
              ),
              workspace(stream, cuda::device_default_memory_pool(device)),
              results(stream, cuda::device_default_memory_pool(device)) {}

        // Declaration order is construction order: the stream outlives its users.
        cuda::stream stream;
        host_batch_results host;
        std::optional<database_type> database;
        std::optional<index_type> index;
        cuda::device_buffer<uint8_t> workspace;
        cuda::device_buffer<packed_pairwise_counts> results;

        /// @brief Grows the search storage to @p requirements, keeping it across searches.
        void reserve(batch_search_requirements const& requirements) {
            auto const device = stream.device();
            if (workspace.size() < requirements.workspace_bytes) {
                workspace = cuda::make_device_buffer<uint8_t>(
                    stream, device, requirements.workspace_bytes, cuda::no_init
                );
            }
            if (results.size() < requirements.maximum_pair_count) {
                results = cuda::make_device_buffer<packed_pairwise_counts>(
                    stream, device, requirements.maximum_pair_count, cuda::no_init
                );
            }
        }
    };

    replicated_database() = default;

    static Result<replicated_database> make(std::span<cuda::device_ref const> devices) {
        if (devices.empty()) {
            return Err(Error::invalid_argument("a replicated database needs at least one device"));
        }
        replicated_database group;
        try {
            for (auto const device : devices) {
                group.replicas_.push_back(std::make_unique<replica>(device));
            }
        } catch (std::exception const& error) {
            return Err(Error::resource(error.what()));
        }
        return Result<replicated_database>::ok(std::move(group));
    }

    /**
     * @brief Runs @p body(replica, index) on every replica concurrently and waits. The calling
     * thread runs replica 0 and each further replica gets its own thread, so a one-replica group
     * starts searching without waiting for a new thread to be scheduled.
     *
     * The first returned error, or CUDA exception from CCCL, is returned; the first other
     * exception, which a user callback may throw, is rethrown. Either one calls @p stop so the
     * other replicas finish early.
     */
    template <typename Body, typename Stop = void (*)()>
    Result<void> on_each(Body&& body, Stop&& stop = [] {}) {
        std::mutex mutex;
        std::optional<Error> error;
        std::exception_ptr exception;
        auto const run = [&](size_t i) {
            // An empty outcome means the replica stopped without an error to report.
            auto const outcome = [&]() -> std::optional<Result<void>> {
                try {
                    return body(*replicas_[i], static_cast<uint32_t>(i));
                } catch (cuda::cuda_error const& failure) {
                    return Err(Error::cuda(static_cast<cudaError_t>(failure.status())));
                } catch (search_stopped const&) {
                    // Another replica already recorded why the search stopped.
                    return std::nullopt;
                } catch (...) {
                    std::lock_guard lock(mutex);
                    if (!exception) exception = std::current_exception();
                    stop();
                    return std::nullopt;
                }
            }();
            if (outcome && !*outcome) {
                std::lock_guard lock(mutex);
                if (!error) error = outcome->error();
                stop();
            }
        };
        {
            std::vector<std::jthread> threads;
            for (size_t i = 1; i < replicas_.size(); ++i) threads.emplace_back(run, i);
            run(0);
        }
        if (exception) std::rethrow_exception(exception);
        if (error) return Err(*error);
        return Ok();
    }

    /// @brief Thrown into a replica waiting for its turn once the search stops elsewhere.
    struct search_stopped {};

    template <typename OnTile>
    Result<void>
    search(OnTile& on_tile, replica_search split, std::optional<search_options> options) {
        if (split.next_tile && split.in_query_order) {
            return Err(
                Error::invalid_argument(
                    "tiles claimed across processes cannot be delivered in query order"
                )
            );
        }
        auto tiles = CUDDL_TRY(
            all_to_all_tile_queue::make(
                database().reference_count(),
                split.shard,
                static_cast<uint32_t>(replicas_.size()),
                std::move(split.next_tile)
            )
        );
        std::mutex order;
        std::condition_variable turn;
        size_t delivered = 0U;
        bool stopped = false;
        return on_each(
            [&](replica& r, uint32_t replica_index) -> Result<void> {
                auto const* index = r.index ? &*r.index : nullptr;
                // Without an index or a threshold every pair is reported, so the dedicated
                // exhaustive kernels run instead of counting matches only to discard nothing.
                bool const exhaustive =
                    !options || (index == nullptr && options->minimum_matches == 0U);
                r.reserve(CUDDL_TRY(r.database->all_to_all_search_requirements(index)));
                auto const deliver = [&](batch_result_tile const& tile) {
                    replica_tile const delivered_tile{tile, r.stream, replica_index, &r.host};
                    if (!split.in_query_order) {
                        on_tile(delivered_tile);
                        return;
                    }
                    auto const position = tiles.position(tile.first_query_id);
                    std::unique_lock lock(order);
                    turn.wait(lock, [&] { return stopped || delivered == position; });
                    if (stopped) throw search_stopped{};
                    on_tile(delivered_tile);
                    ++delivered;
                    turn.notify_all();
                };
                auto const workspace = device_span<uint8_t>{r.workspace.data(), r.workspace.size()};
                auto const results =
                    device_span<packed_pairwise_counts>{r.results.data(), r.results.size()};
                if (exhaustive) {
                    CUDDL_TRY(r.database->search_all_to_all_async(
                        workspace, results, deliver, {}, r.stream, tiles
                    ));
                } else {
                    CUDDL_TRY(r.database->search_all_to_all_async(
                        workspace, results, deliver, {}, *options, r.stream, index, tiles
                    ));
                }
                CUDDL_CUDA_TRY(r.stream.sync());
                return Ok();
            },
            [&] {
                tiles.cancel();
                {
                    std::lock_guard lock(order);
                    stopped = true;
                }
                turn.notify_all();
            }
        );
    }

    std::vector<std::unique_ptr<replica>> replicas_;
};

}  // namespace cuddl
