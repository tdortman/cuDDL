#pragma once

#include <cuddl/detail/file_writer.hpp>
#include <cuddl/detail/worker_pool.hpp>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>
#include <cuddl/reference_database.cuh>
#include <cuddl/replicated_database.cuh>

#include <cub/device/device_transform.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cuddl {

/// @brief How a @ref result_file stores each pair.
enum class result_format : uint8_t {
    /// One 24-byte @ref batch_search_result per pair: query ID, reference ID and the lower,
    /// equal, higher and both-empty bucket counts, six little-endian `uint32`.
    counts,
    /// The pair's @ref weighted_kmer_identity as a 4-byte float, computed on the GPU. A tile
    /// that holds every pair is written as its first query ID and query count (two `uint32`)
    /// followed by one float per pair in the tile's order: query-major, each query against every
    /// later reference in an all-to-all search and against every reference otherwise. Every
    /// other pair is a 12-byte @ref similarity_record.
    similarity,
};

/// @brief One pair of a thresholded search in @ref result_format::similarity.
struct similarity_record {
    uint32_t query_id{};
    uint32_t reference_id{};
    float similarity{};
};

struct result_file_options {
    result_format format = result_format::counts;
    /// Files written in parallel: one writes the path given, more write `<path>.0` to
    /// `<path>.<streams - 1>`, each tile whole in one of them. A parallel file system gives its
    /// bandwidth only to many streams: four ranks on an A100 node wrote 20 GB to Lustre in
    /// 9.1 s with one file each and in 5.8 s with two.
    unsigned streams = 1;
    /// Whether @ref result_file::finish flushes the files to storage before returning.
    bool durable = false;
    /// Threads that expand results into records; 0 uses every hardware thread.
    unsigned threads = 0;
};

/// @brief What a @ref result_file wrote and how long its callers waited for it.
struct result_file_statistics {
    uint64_t pairs = 0;
    uint64_t bytes = 0;
    /// Seconds callers spent in @ref result_file::write downloading tiles and converting them on
    /// the GPU, summed over the threads that called it.
    double download_seconds = 0;
    /// Seconds callers waited in @ref result_file::write and @ref result_file::finish for earlier
    /// tiles to be written.
    double stall_seconds = 0;
};

namespace detail {

/// @brief Weighted k-mer identity of one packed result, for the GPU conversion.
struct packed_identity {
    uint32_t bucket_count;

    __host__ __device__ float operator()(packed_pairwise_counts packed) const noexcept {
        return weighted_kmer_identity(packed.unpack(bucket_count));
    }
};

}  // namespace detail

/**
 * @brief Writes search results to files while the GPUs search on.
 *
 * Pass every tile a search delivers to @ref write, from its `on_tile` callback. A write copies
 * the tile to pinned host memory, or in @ref result_format::similarity converts it on the GPU
 * first, and returns; a thread of the file's expands it into records on every core and writes
 * them, bypassing the page cache where the file system allows. Each stream that delivers tiles
 * gets two host buffers, so its next tile downloads while the last one is written; a write
 * waits only when both are still being written. Tiles reach the file in the order their writes
 * finish downloading, which is the search order when callbacks run one at a time. The buffers
 * are freed on those streams, so call @ref finish, or destroy the file, while they live.
 *
 * @code
 * auto file = CUDDL_TRY(cuddl::result_file::open("pairs.bin"));
 * CUDDL_TRY(group.search_all_to_all([&](cuddl::replica_tile const& tile) {
 *     CUDDL_UNWRAP(file.write(tile));
 * }));
 * CUDDL_TRY(file.finish());
 * @endcode
 */
class result_file {
   public:
    /// @brief The files a result file with @p streams streams writes for @p path.
    [[nodiscard]] static std::vector<std::filesystem::path>
    files(std::filesystem::path const& path, unsigned streams) {
        if (streams <= 1) return {path};
        std::vector<std::filesystem::path> paths;
        paths.reserve(streams);
        for (unsigned s = 0; s < streams; ++s)
            paths.emplace_back(path.string() + "." + std::to_string(s));
        return paths;
    }

    /// @brief Creates or truncates the files for @p path.
    [[nodiscard]] static Result<result_file>
    open(std::filesystem::path const& path, result_file_options options = {}) {
        if (options.streams == 0) {
            return Err(Error::invalid_argument("a result file needs at least one stream"));
        }
        result_file file;
        try {
            file.state_ = std::make_unique<state>(path, options);
        } catch (std::exception const& error) {
            return Err(Error::resource(error.what()));
        }
        return file;
    }

    /// @brief Downloads @p tile, whose results @p stream orders, and queues it for writing.
    ///
    /// Thread-safe. The tile's device storage is free for reuse when it returns.
    [[nodiscard]] Result<void> write(batch_result_tile const& tile, cuda::stream_ref stream) {
        return state_->write(tile, stream);
    }

    [[nodiscard]] Result<void> write(replica_tile const& tile) {
        return write(tile.tile, tile.stream);
    }

    /// @brief Queues results already on the host, such as each genome's top-k neighbours.
    [[nodiscard]] Result<void> write(std::span<batch_search_result const> rows) {
        return state_->write(rows);
    }

    /// @brief Writes everything queued and closes the files, flushing them to storage first if
    /// the file is durable. Reports the first error any write met.
    [[nodiscard]] Result<void> finish() {
        return state_->finish();
    }

    [[nodiscard]] result_file_statistics statistics() const {
        std::lock_guard lock(state_->mutex);
        return state_->statistics;
    }

   private:
    using clock = std::chrono::steady_clock;
    using pinned_floats =
        cuda::buffer<float, cuda::mr::host_accessible, cuda::mr::device_accessible>;

    /// One tile on its way to the file.
    struct slot {
        std::optional<host_batch_results> host;
        // A tile holding every pair, in the similarity format, converted on the GPU.
        std::optional<cuda::device_buffer<float>> device;
        std::optional<pinned_floats> floats;
        bool converted = false;
        uint32_t first_query_id = 0;
        uint32_t query_count = 0;
        uint64_t pairs = 0;
        bool queued = false;
    };

    /// A queued tile, or host rows copied in.
    struct job {
        slot* tile = nullptr;
        std::vector<batch_search_result> rows;
    };

    static double seconds_since(clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    }

    struct state {
        state(std::filesystem::path const& path, result_file_options const& options)
            : format(options.format),
              threads(
                  options.threads != 0 ? options.threads
                                       : std::max(1U, std::thread::hardware_concurrency())
              ),
              pool(threads) {
            auto const paths = files(path, options.streams);
            for (auto const& file : paths) {
                streams.push_back(
                    std::make_unique<detail::file_writer>(
                        file, options.durable, (size_t{1} << 30) / paths.size()
                    )
                );
            }
            thread = std::thread([this] { run(); });
        }
        state(state const&) = delete;
        state& operator=(state const&) = delete;
        ~state() {
            stop();
        }

        Result<void> write(batch_result_tile const& tile, cuda::stream_ref stream) {
            slot* free = nullptr;
            {
                std::unique_lock lock(mutex);
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                auto& [pair, next] = slots[stream.get()];
                free = &pair[next];
                next ^= 1U;
                auto const waiting = clock::now();
                written.wait(lock, [&] { return !free->queued || failure; });
                statistics.stall_seconds += seconds_since(waiting);
                if (failure) return Err(error_of(failure));
            }
            auto const start = clock::now();
            CUDDL_TRY(download(tile, stream, *free));
            {
                std::lock_guard lock(mutex);
                statistics.download_seconds += seconds_since(start);
                free->queued = true;
                queue.push_back({free, {}});
            }
            queued.notify_one();
            return Ok();
        }

        Result<void> write(std::span<batch_search_result const> rows) {
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                queue.push_back({nullptr, {rows.begin(), rows.end()}});
            }
            queued.notify_one();
            return Ok();
        }

        Result<void> finish() {
            auto const start = clock::now();
            stop();
            slots.clear();
            if (failure) return Err(error_of(failure));
            try {
                for (auto& file : streams) file->finish();
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            std::lock_guard lock(mutex);
            statistics.stall_seconds += seconds_since(start);
            return Ok();
        }

        /// Copies @p tile into @p into, converting it on the GPU first in the similarity format
        /// when it holds every pair; synchronises @p stream.
        Result<void>
        download(batch_result_tile const& tile, cuda::stream_ref stream, slot& into) const {
            into.converted = format == result_format::similarity && tile.pass_bits == nullptr;
            if (!into.converted) {
                if (!into.host) CUDDL_TRY(cuda_try([&] { into.host.emplace(stream); }));
                return cuddl::download(tile, stream, *into.host);
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            if (!into.device || into.device->size() < pairs) {
                CUDDL_TRY(cuda_try([&] {
                    into.device.emplace(
                        cuda::make_device_buffer<float>(
                            stream, stream.device(), pairs, cuda::no_init
                        )
                    );
                    into.floats.emplace(
                        stream,
                        detail::device_pinned_pool(
                            stream.device(), detail::pinned_placement::device_node
                        ),
                        pairs,
                        cuda::no_init
                    );
                }));
            }
            if (pairs != 0U) {
                CUDDL_CUDA_TRY(
                    cub::DeviceTransform::Transform(
                        tile.results,
                        into.device->data(),
                        pairs,
                        detail::packed_identity{tile.bucket_count},
                        stream.get()
                    )
                );
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        cuda::std::span{into.device->data(), pairs},
                        cuda::std::span{into.floats->data(), pairs}
                    )
                );
            }
            CUDDL_CUDA_TRY(stream.sync());
            into.first_query_id = tile.first_query_id;
            into.query_count = tile.query_count;
            into.pairs = pairs;
            return Ok();
        }

        static Error error_of(std::exception_ptr const& failure) {
            try {
                std::rethrow_exception(failure);
            } catch (std::exception const& error) {
                return Error::resource(error.what());
            } catch (...) {
                return Error::resource("result file write failed");
            }
        }

        void stop() {
            {
                std::lock_guard lock(mutex);
                done = true;
            }
            queued.notify_one();
            if (thread.joinable()) thread.join();
        }

        void run() {
            while (true) {
                job next;
                {
                    std::unique_lock lock(mutex);
                    queued.wait(lock, [&] { return done || !queue.empty(); });
                    if (queue.empty()) return;
                    next = std::move(queue.front());
                    queue.pop_front();
                }
                try {
                    if (next.tile != nullptr) {
                        write_tile(*next.tile);
                    } else {
                        write_rows(next.rows);
                    }
                } catch (...) {
                    std::lock_guard lock(mutex);
                    failure = std::current_exception();
                    written.notify_all();
                    return;
                }
                if (next.tile != nullptr) {
                    std::lock_guard lock(mutex);
                    next.tile->queued = false;
                }
                written.notify_all();
            }
        }

        detail::file_writer& next_stream() {
            return *streams[next_file++ % streams.size()];
        }

        void count(uint64_t pairs, uint64_t bytes) {
            std::lock_guard lock(mutex);
            statistics.pairs += pairs;
            statistics.bytes += bytes;
        }

        void write_tile(slot const& tile) {
            if (tile.converted) {
                auto& file = next_stream();
                auto const bytes = tile.pairs * sizeof(float);
                auto const out = file.reserve(2 * sizeof(uint32_t) + bytes);
                std::array const header{tile.first_query_id, tile.query_count};
                std::memcpy(out.data(), header.data(), sizeof(header));
                // A tile can hold gigabytes, more than one core copies in time.
                constexpr size_t piece = size_t{8} << 20;
                auto const* source = reinterpret_cast<char const*>(tile.floats->data());
                pool.for_each((bytes + piece - 1) / piece, [&](size_t p) {
                    auto const at = p * piece;
                    std::memcpy(
                        out.data() + sizeof(header) + at, source + at, std::min(piece, bytes - at)
                    );
                });
                file.commit();
                count(tile.pairs, out.size());
                return;
            }
            auto const results = tile.host->tile();
            if (format == result_format::counts) {
                expand(results, [](batch_search_result const& result) { return result; });
            } else {
                expand(results, [](batch_search_result const& result) {
                    return similarity_record{
                        result.query_id, result.reference_id, weighted_kmer_identity(result.counts)
                    };
                });
            }
        }

        /// Writes the passing results of @p tile as records, expanded on every core straight
        /// into the file's memory.
        template <typename Convert>
        void expand(batch_result_tile const& tile, Convert const& convert) {
            using record = std::invoke_result_t<Convert, batch_search_result const&>;
            auto const total = tile.count();
            if (total == 0) return;
            auto& file = next_stream();
            // Every record and header is a whole number of 4-byte words, and extents are
            // page-aligned, so records start on 4-byte boundaries. Stores through a char pointer
            // instead made the compiler reload the tile on every record.
            auto* const records =
                reinterpret_cast<record*>(file.reserve(total * sizeof(record)).data());
            if (reinterpret_cast<uintptr_t>(records) % alignof(record) != 0) {
                throw std::logic_error("result records are misaligned");
            }
            // Row runs of about equal result counts, several per thread for upper-triangle tiles.
            auto const runs = std::min<uint64_t>(uint64_t{threads} * 8U, tile.query_count);
            bounds.assign(runs + 1U, tile.query_count);
            auto const rows = std::views::iota(uint32_t{0}, tile.query_count);
            for (uint64_t r = 0; r < runs; ++r) {
                bounds[r] = *std::ranges::partition_point(rows, [&](uint32_t query) {
                    return tile.passing_before(query) * runs < total * r;
                });
            }
            pool.for_each(runs, [&](size_t r) {
                auto* out = records + tile.passing_before(bounds[r]);
                tile.for_each_passing(
                    bounds[r], bounds[r + 1U], [&](batch_search_result const& result) {
                        *out++ = convert(result);
                    }
                );
            });
            file.commit();
            count(total, total * sizeof(record));
        }

        void write_rows(std::vector<batch_search_result> const& rows) {
            auto& file = next_stream();
            if (format == result_format::counts) {
                file.append(
                    {reinterpret_cast<char const*>(rows.data()), rows.size() * sizeof(rows[0])}
                );
                count(rows.size(), rows.size() * sizeof(rows[0]));
                return;
            }
            std::vector<similarity_record> records;
            records.reserve(rows.size());
            for (auto const& row : rows) {
                records.push_back(
                    {row.query_id, row.reference_id, weighted_kmer_identity(row.counts)}
                );
            }
            file.append(
                {reinterpret_cast<char const*>(records.data()), records.size() * sizeof(records[0])}
            );
            count(records.size(), records.size() * sizeof(records[0]));
        }

        result_format format;
        std::vector<std::unique_ptr<detail::file_writer>> streams;
        size_t next_file = 0;
        size_t threads;
        detail::worker_pool pool;
        std::vector<uint32_t> bounds;
        std::map<cudaStream_t, std::pair<std::array<slot, 2>, unsigned>> slots;
        std::deque<job> queue;
        std::mutex mutex;
        std::condition_variable queued, written;
        std::exception_ptr failure;
        result_file_statistics statistics;
        bool done = false;
        std::thread thread;
    };

    std::unique_ptr<state> state_;
};

static_assert(sizeof(batch_search_result) == 24);
static_assert(sizeof(similarity_record) == 12);

}  // namespace cuddl
