#pragma once

#include <cuddl/detail/file_writer.hpp>
#include <cuddl/detail/uplink_lock.hpp>
#include <cuddl/detail/worker_pool.hpp>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>
#include <cuddl/reference_database.cuh>
#include <cuddl/replicated_database.cuh>

#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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

namespace detail {

struct packed_to_ranked {
    __host__ __device__ ranked_pairwise_counts
    operator()(packed_pairwise_counts packed) const noexcept {
        return ranked_pairwise_counts::from_packed(packed);
    }
};

struct is_nonzero_float {
    __host__ __device__ bool operator()(float value) const noexcept {
        return value != 0.0f;
    }
};

__global__ void
pack_nonzero_bitmap_kernel(float const* values, size_t count, uint32_t* bitmap) {
    constexpr uint32_t warp_width = 32U;
    auto const words = (count + warp_width - 1U) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const stride = static_cast<size_t>(gridDim.x) * (blockDim.x / warp_width);
    for (auto w = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_width;
         w < words;
         w += stride) {
        auto const idx = w * warp_width + lane;
        auto const nz = (idx < count) && (values[idx] != 0.0f);
        auto const mask = __ballot_sync(0xffffffffU, nz);
        if (lane == 0U) {
            bitmap[w] = mask;
        }
    }
}

/// @brief A WKID in [0, 1] as `round(wkid * 65535)`. The product of a float and 65535 is exact
/// in double.
__host__ __device__ inline uint16_t quantize_similarity(float similarity) noexcept {
    return static_cast<uint16_t>(static_cast<double>(similarity) * 65535.0 + 0.5);
}

struct similarity_to_u16 {
    __host__ __device__ uint16_t operator()(float similarity) const noexcept {
        return quantize_similarity(similarity);
    }
};

struct packed_identity_to_u16 {
    uint32_t bucket_count;

    __host__ __device__ uint16_t operator()(packed_pairwise_counts packed) const noexcept {
        return quantize_similarity(packed_identity{bucket_count}(packed));
    }
};

}  // namespace detail

/// @brief How a @ref result_file stores each pair.
enum class result_format : uint8_t {
    /// One 24-byte @ref batch_search_result per pair: query ID, reference ID and the lower,
    /// equal, higher and both-empty bucket counts, six little-endian `uint32`.
    counts,
    /// One 8-byte @ref packed_pairwise_counts per pair for exhaustive tiles: each tile writes
    /// its first query ID and query count (two `uint32`) followed by one packed `(L, E, H)` word
    /// per pair in the tile's order (`Z = B - L - E - H`). Thresholded tiles fall back to
    /// 24-byte @ref batch_search_result records.
    packed_counts,
    /// One 4-byte @ref ranked_pairwise_counts per pair for exhaustive tiles (`B <= 2952`):
    /// lossless combinatorial rank of `(L, E, H, Z)` prefixed by the 8-byte tile header.
    /// Thresholded tiles fall back to 24-byte @ref batch_search_result records.
    ranked_counts,
    /// The pair's @ref weighted_kmer_identity as a 4-byte float, computed on the GPU. A tile
    /// that holds every pair is written as its first query ID and query count (two `uint32`)
    /// followed by one float per pair in the tile's order: query-major, each query against every
    /// later reference in an all-to-all search and against every reference otherwise. Every
    /// other pair is a 12-byte @ref similarity_record.
    similarity,
    /// Lossless sparse encoding of the full float32 WKID tile: 16-byte header
    /// `(first_query_id, query_count, pair_count, nonzero_count)`, followed by
    /// `ceil(pair_count / 32)` `uint32` bitmap words (bit `i` set iff `wkid[i] != 0.0f`),
    /// followed by `nonzero_count` packed `float` values in tile order. Thresholded tiles fall
    /// back to 12-byte @ref similarity_record records.
    sparse_similarity,
    /// The pair's @ref weighted_kmer_identity as `round(wkid * 65535)`, a little-endian 2-byte
    /// `uint16`, converted on the GPU before the download. A tile that holds every pair is
    /// written as in @ref similarity, the 8-byte header followed by one `uint16` per pair in the
    /// same order. Every other pair is a 12-byte @ref similarity_record with the exact float.
    similarity16,
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

/**
 * @brief Writes search results to files while the GPUs search on.
 *
 * Pass every tile a search delivers to @ref write, from its `on_tile` callback. A write copies
 * the tile to pinned host memory and returns; a thread of the file's expands it into records on
 * every core and writes them, bypassing the page cache where the file system allows. Each stream
 * that delivers tiles gets two host buffers, so its next tile downloads while the last one is
 * written; a write waits only when both are still being written. In
 * @ref result_format::similarity, a tile that holds every pair is copied as similarities straight
 * into the file's page-locked write buffer instead, converted on the GPU unless the search wrote
 * them itself (exhaustive_output::similarities). Tiles reach the file in the
 * order their writes finish downloading, which is the search order when callbacks run one at a
 * time. The buffers are freed on those streams, so call @ref finish, or destroy the file, while
 * they live.
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

    /// One tile on its way to the file.
    struct slot {
        std::optional<host_batch_results> host;
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
            detail::file_extent_hooks hooks;
            if (format == result_format::similarity || format == result_format::sparse_similarity ||
                format == result_format::similarity16 || format == result_format::packed_counts ||
                format == result_format::ranked_counts) {
                // Direct-DMA tile writers copy from the GPU straight into the files' memory.
                hooks.allocated = [](char* data, size_t bytes) {
                    CUDDL_CUDA_CALL(cudaHostRegister(data, bytes, cudaHostRegisterPortable));
                };
                hooks.freeing = [](char* data) {
                    static_cast<void>(cudaHostUnregister(data));
                };
            }
            for (auto const& file : paths) {
                streams.push_back(
                    std::make_unique<detail::file_writer>(
                        file, options.durable, (size_t{1} << 30) / paths.size(), hooks
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
            if (tile.similarities != nullptr && !stores_similarities(format)) {
                return Err(Error::invalid_argument("a counts file needs tiles that hold counts"));
            }
            if (format == result_format::similarity16 && tile.pass_bits == nullptr) {
                return write_similarity16(tile, stream);
            }
            if (format == result_format::similarity && tile.pass_bits == nullptr) {
                return write_converted(tile, stream);
            }
            if (format == result_format::sparse_similarity && tile.pass_bits == nullptr) {
                return write_sparse_similarity(tile, stream);
            }
            if (format == result_format::packed_counts && tile.pass_bits == nullptr) {
                return write_packed_counts(tile, stream);
            }
            if (format == result_format::ranked_counts && tile.pass_bits == nullptr) {
                return write_ranked_counts(tile, stream);
            }
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
            converted.clear();
            sparse_bufs.clear();
            quantized.clear();
            ranked.clear();
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

        /// Copies @p tile into @p into; synchronises @p stream.
        static Result<void>
        download(batch_result_tile const& tile, cuda::stream_ref stream, slot& into) {
            if (!into.host) CUDDL_TRY(cuda_try([&] { into.host.emplace(stream); }));
            return cuddl::download(tile, stream, *into.host);
        }

        /// Copies the similarities of @p tile, which holds every pair, straight into the bytes a
        /// file reserves, converting its counts on the GPU first when it holds counts. Host
        /// threads copying a tile out of a download buffer would compete with the next download
        /// for memory bandwidth: on an RTX PRO 6000 node, 40,000 genomes searched and downloaded
        /// in 465-487 ms this way and in 812-844 ms with that copy. Synchronises @p stream.
        Result<void> write_converted(batch_result_tile const& tile, cuda::stream_ref stream) {
            std::optional<cuda::device_buffer<float>>* scratch = nullptr;
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                if (tile.similarities == nullptr) scratch = &converted[stream.get()];
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            if (scratch != nullptr && (!*scratch || (*scratch)->size() < pairs)) {
                CUDDL_TRY(cuda_try([&] {
                    scratch->emplace(
                        cuda::make_device_buffer<float>(
                            stream, stream.device(), pairs, cuda::no_init
                        )
                    );
                }));
            }
            float const* const source = scratch != nullptr ? (*scratch)->data() : tile.similarities;
            auto& file = next_stream();
            std::array const header{tile.first_query_id, tile.query_count};
            auto const waiting = clock::now();
            std::span<char> out;
            try {
                out = file.reserve(sizeof(header) + pairs * sizeof(float));
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            auto const start = clock::now();
            std::memcpy(out.data(), header.data(), sizeof(header));
            // Records and headers are whole 4-byte words, so the floats start word-aligned.
            auto* const floats = reinterpret_cast<float*>(out.data() + sizeof(header));
            auto copied = [&]() -> Result<void> {
                detail::uplink_lock const uplink(stream.device().get());
                if (pairs != 0U) {
                    if (scratch != nullptr) {
                        CUDDL_CUDA_TRY(
                            cub::DeviceTransform::Transform(
                                tile.results,
                                (*scratch)->data(),
                                pairs,
                                detail::packed_identity{tile.bucket_count},
                                stream.get()
                            )
                        );
                    }
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream, cuda::std::span{source, pairs}, cuda::std::span{floats, pairs}
                        )
                    );
                }
                CUDDL_CUDA_TRY(stream.sync());
                return Ok();
            }();
            if (!copied) {
                file.abandon();
                return copied;
            }
            file.commit();
            std::lock_guard lock(mutex);
            statistics.stall_seconds += std::chrono::duration<double>(start - waiting).count();
            statistics.download_seconds += seconds_since(start);
            statistics.pairs += pairs;
            statistics.bytes += out.size();
            return Ok();
        }

        /// Converts the similarities of @p tile, which holds every pair, to 16 bits on the GPU
        /// and copies those straight into the bytes a file reserves. Synchronises @p stream.
        Result<void> write_similarity16(batch_result_tile const& tile, cuda::stream_ref stream) {
            std::optional<cuda::device_buffer<uint16_t>>* scratch = nullptr;
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                scratch = &quantized[stream.get()];
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            if (!*scratch || (*scratch)->size() < pairs) {
                CUDDL_TRY(cuda_try([&] {
                    scratch->emplace(
                        cuda::make_device_buffer<uint16_t>(
                            stream, stream.device(), pairs, cuda::no_init
                        )
                    );
                }));
            }
            auto& file = next_stream();
            std::array const header{tile.first_query_id, tile.query_count};
            auto const waiting = clock::now();
            std::span<char> out;
            try {
                out = file.reserve(sizeof(header) + pairs * sizeof(uint16_t));
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            auto const start = clock::now();
            std::memcpy(out.data(), header.data(), sizeof(header));
            auto* const words = reinterpret_cast<uint16_t*>(out.data() + sizeof(header));
            auto copied = [&]() -> Result<void> {
                detail::uplink_lock const uplink(stream.device().get());
                if (pairs != 0U) {
                    if (tile.similarities != nullptr) {
                        CUDDL_CUDA_TRY(
                            cub::DeviceTransform::Transform(
                                tile.similarities,
                                (*scratch)->data(),
                                pairs,
                                detail::similarity_to_u16{},
                                stream.get()
                            )
                        );
                    } else {
                        CUDDL_CUDA_TRY(
                            cub::DeviceTransform::Transform(
                                tile.results,
                                (*scratch)->data(),
                                pairs,
                                detail::packed_identity_to_u16{tile.bucket_count},
                                stream.get()
                            )
                        );
                    }
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream,
                            cuda::std::span{(*scratch)->data(), pairs},
                            cuda::std::span{words, pairs}
                        )
                    );
                }
                CUDDL_CUDA_TRY(stream.sync());
                return Ok();
            }();
            if (!copied) {
                file.abandon();
                return copied;
            }
            file.commit();
            std::lock_guard lock(mutex);
            statistics.stall_seconds += std::chrono::duration<double>(start - waiting).count();
            statistics.download_seconds += seconds_since(start);
            statistics.pairs += pairs;
            statistics.bytes += out.size();
            return Ok();
        }

        struct sparse_scratch {
            std::optional<cuda::device_buffer<uint32_t>> bitmap;
            std::optional<cuda::device_buffer<float>> nonzeros;
            std::optional<cuda::device_buffer<uint8_t>> cub_temp;
            std::optional<cuda::device_buffer<uint32_t>> num_selected;
        };

        /// Losslessly packs an exhaustive tile's float32 WKIDs on the GPU into a 1-bit nonzero
        /// bitmap plus packed nonzero floats and copies only the compacted bytes straight into the
        /// file's pinned write buffer. Synchronises @p stream.
        Result<void> write_sparse_similarity(batch_result_tile const& tile, cuda::stream_ref stream) {
            std::optional<cuda::device_buffer<float>>* scratch = nullptr;
            sparse_scratch* sp = nullptr;
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                if (tile.similarities == nullptr) scratch = &converted[stream.get()];
                sp = &sparse_bufs[stream.get()];
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            auto const bitmap_words = (pairs + 31U) / 32U;
            if (scratch != nullptr && (!*scratch || (*scratch)->size() < pairs)) {
                CUDDL_TRY(cuda_try([&] {
                    scratch->emplace(
                        cuda::make_device_buffer<float>(
                            stream, stream.device(), pairs, cuda::no_init
                        )
                    );
                }));
            }
            if (!sp->bitmap || sp->bitmap->size() < bitmap_words) {
                CUDDL_TRY(cuda_try([&] {
                    sp->bitmap.emplace(
                        cuda::make_device_buffer<uint32_t>(
                            stream, stream.device(), std::max<size_t>(1U, bitmap_words), cuda::no_init
                        )
                    );
                }));
            }
            if (!sp->nonzeros || sp->nonzeros->size() < pairs) {
                CUDDL_TRY(cuda_try([&] {
                    sp->nonzeros.emplace(
                        cuda::make_device_buffer<float>(
                            stream, stream.device(), std::max<size_t>(1U, pairs), cuda::no_init
                        )
                    );
                }));
            }
            if (!sp->num_selected) {
                CUDDL_TRY(cuda_try([&] {
                    sp->num_selected.emplace(
                        cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init)
                    );
                }));
            }
            float const* const source = scratch != nullptr ? (*scratch)->data() : tile.similarities;
            uint32_t nonzero_count = 0U;
            auto const gpu_start = clock::now();
            if (pairs != 0U) {
                if (scratch != nullptr) {
                    CUDDL_CUDA_TRY(
                        cub::DeviceTransform::Transform(
                            tile.results,
                            (*scratch)->data(),
                            pairs,
                            detail::packed_identity{tile.bucket_count},
                            stream.get()
                        )
                    );
                }
                constexpr uint32_t threads_per_block = 256U;
                constexpr uint32_t warps_per_block = threads_per_block / 32U;
                auto const blocks = static_cast<uint32_t>(std::min<size_t>(
                    (bitmap_words + warps_per_block - 1U) / warps_per_block, 65535U
                ));
                detail::pack_nonzero_bitmap_kernel<<<blocks, threads_per_block, 0, stream.get()>>>(
                    source, pairs, sp->bitmap->data()
                );
                CUDDL_CUDA_TRY(cudaGetLastError());

                size_t temp_bytes = 0;
                CUDDL_CUDA_TRY(cub::DeviceSelect::If(
                    nullptr,
                    temp_bytes,
                    source,
                    sp->nonzeros->data(),
                    sp->num_selected->data(),
                    pairs,
                    detail::is_nonzero_float{},
                    stream.get()
                ));
                if (!sp->cub_temp || sp->cub_temp->size() < temp_bytes) {
                    CUDDL_TRY(cuda_try([&] {
                        sp->cub_temp.emplace(
                            cuda::make_device_buffer<uint8_t>(
                                stream, stream.device(), std::max<size_t>(1U, temp_bytes), cuda::no_init
                            )
                        );
                    }));
                }
                CUDDL_CUDA_TRY(cub::DeviceSelect::If(
                    sp->cub_temp->data(),
                    temp_bytes,
                    source,
                    sp->nonzeros->data(),
                    sp->num_selected->data(),
                    pairs,
                    detail::is_nonzero_float{},
                    stream.get()
                ));
                CUDDL_CUDA_TRY(cuda::copy_bytes(
                    stream,
                    cuda::std::span{sp->num_selected->data(), 1U},
                    cuda::std::span{&nonzero_count, 1U}
                ));
                CUDDL_CUDA_TRY(stream.sync());
            }
            auto const gpu_seconds = seconds_since(gpu_start);

            auto& file = next_stream();
            std::array<uint32_t, 4> const header{
                tile.first_query_id,
                tile.query_count,
                static_cast<uint32_t>(pairs),
                nonzero_count,
            };
            auto const bitmap_bytes = bitmap_words * sizeof(uint32_t);
            auto const nonzero_bytes = static_cast<size_t>(nonzero_count) * sizeof(float);
            auto const waiting = clock::now();
            std::span<char> out;
            try {
                out = file.reserve(sizeof(header) + bitmap_bytes + nonzero_bytes);
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            auto const dl_start = clock::now();
            std::memcpy(out.data(), header.data(), sizeof(header));
            auto* const dst_bitmap = reinterpret_cast<uint32_t*>(out.data() + sizeof(header));
            auto* const dst_nonzeros =
                reinterpret_cast<float*>(out.data() + sizeof(header) + bitmap_bytes);
            auto copied = [&]() -> Result<void> {
                detail::uplink_lock const uplink(stream.device().get());
                if (bitmap_words != 0U) {
                    CUDDL_CUDA_TRY(cuda::copy_bytes(
                        stream,
                        cuda::std::span{sp->bitmap->data(), bitmap_words},
                        cuda::std::span{dst_bitmap, bitmap_words}
                    ));
                }
                if (nonzero_count != 0U) {
                    CUDDL_CUDA_TRY(cuda::copy_bytes(
                        stream,
                        cuda::std::span{sp->nonzeros->data(), static_cast<size_t>(nonzero_count)},
                        cuda::std::span{dst_nonzeros, static_cast<size_t>(nonzero_count)}
                    ));
                }
                CUDDL_CUDA_TRY(stream.sync());
                return Ok();
            }();
            if (!copied) {
                file.abandon();
                return copied;
            }
            file.commit();
            std::lock_guard lock(mutex);
            statistics.stall_seconds += std::chrono::duration<double>(dl_start - waiting).count();
            statistics.download_seconds += gpu_seconds + seconds_since(dl_start);
            statistics.pairs += pairs;
            statistics.bytes += out.size();
            return Ok();
        }

        /// Copies the packed 8-byte `(L, E, H)` counts of an exhaustive @p tile straight into the
        /// bytes a file reserves, prefixed by the 8-byte `(first_query_id, query_count)` header.
        Result<void> write_packed_counts(batch_result_tile const& tile, cuda::stream_ref stream) {
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            auto& file = next_stream();
            std::array const header{tile.first_query_id, tile.query_count};
            auto const waiting = clock::now();
            std::span<char> out;
            try {
                out = file.reserve(sizeof(header) + pairs * sizeof(packed_pairwise_counts));
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            auto const start = clock::now();
            std::memcpy(out.data(), header.data(), sizeof(header));
            auto* const packed =
                reinterpret_cast<packed_pairwise_counts*>(out.data() + sizeof(header));
            auto copied = [&]() -> Result<void> {
                detail::uplink_lock const uplink(stream.device().get());
                if (pairs != 0U) {
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream,
                            cuda::std::span{tile.results, pairs},
                            cuda::std::span{packed, pairs}
                        )
                    );
                }
                CUDDL_CUDA_TRY(stream.sync());
                return Ok();
            }();
            if (!copied) {
                file.abandon();
                return copied;
            }
            file.commit();
            std::lock_guard lock(mutex);
            statistics.stall_seconds += std::chrono::duration<double>(start - waiting).count();
            statistics.download_seconds += seconds_since(start);
            statistics.pairs += pairs;
            statistics.bytes += out.size();
            return Ok();
        }

        /// Converts an exhaustive @p tile's counts on the GPU into 4-byte lossless combinatorial
        /// ranks and copies them straight into the file's pinned write buffer.
        Result<void> write_ranked_counts(batch_result_tile const& tile, cuda::stream_ref stream) {
            if (tile.bucket_count > ranked_pairwise_counts::max_bucket_count) {
                return Err(Error::invalid_argument(
                    "ranked_counts requires bucket_count <= ranked_pairwise_counts::max_bucket_count"
                ));
            }
            std::optional<cuda::device_buffer<ranked_pairwise_counts>>* scratch = nullptr;
            {
                std::lock_guard lock(mutex);
                if (failure) return Err(error_of(failure));
                if (done) return Err(Error::invalid_argument("the result file is finished"));
                scratch = &ranked[stream.get()];
            }
            detail::device_scope const scope{stream};
            auto const pairs = static_cast<size_t>(tile.slot_count());
            if (!*scratch || (*scratch)->size() < pairs) {
                CUDDL_TRY(cuda_try([&] {
                    scratch->emplace(
                        cuda::make_device_buffer<ranked_pairwise_counts>(
                            stream, stream.device(), pairs, cuda::no_init
                        )
                    );
                }));
            }
            auto& file = next_stream();
            std::array const header{tile.first_query_id, tile.query_count};
            auto const waiting = clock::now();
            std::span<char> out;
            try {
                out = file.reserve(sizeof(header) + pairs * sizeof(ranked_pairwise_counts));
            } catch (std::exception const& error) {
                return Err(Error::resource(error.what()));
            }
            auto const start = clock::now();
            std::memcpy(out.data(), header.data(), sizeof(header));
            auto* const dst =
                reinterpret_cast<ranked_pairwise_counts*>(out.data() + sizeof(header));
            auto copied = [&]() -> Result<void> {
                detail::uplink_lock const uplink(stream.device().get());
                if (pairs != 0U) {
                    CUDDL_CUDA_TRY(
                        cub::DeviceTransform::Transform(
                            tile.results,
                            (*scratch)->data(),
                            pairs,
                            detail::packed_to_ranked{},
                            stream.get()
                        )
                    );
                    CUDDL_CUDA_TRY(
                        cuda::copy_bytes(
                            stream,
                            cuda::std::span{(*scratch)->data(), pairs},
                            cuda::std::span{dst, pairs}
                        )
                    );
                }
                CUDDL_CUDA_TRY(stream.sync());
                return Ok();
            }();
            if (!copied) {
                file.abandon();
                return copied;
            }
            file.commit();
            std::lock_guard lock(mutex);
            statistics.stall_seconds += std::chrono::duration<double>(start - waiting).count();
            statistics.download_seconds += seconds_since(start);
            statistics.pairs += pairs;
            statistics.bytes += out.size();
            return Ok();
        }

        /// Whether the format's thresholded tiles and host rows are 12-byte similarity records.
        static bool stores_similarities(result_format format) noexcept {
            return format == result_format::similarity ||
                   format == result_format::sparse_similarity ||
                   format == result_format::similarity16;
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
            return *streams[next_file.fetch_add(1) % streams.size()];
        }

        void count(uint64_t pairs, uint64_t bytes) {
            std::lock_guard lock(mutex);
            statistics.pairs += pairs;
            statistics.bytes += bytes;
        }

        void write_tile(slot const& tile) {
            auto const results = tile.host->tile();
            if (!stores_similarities(format)) {
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
            if (!stores_similarities(format)) {
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
        std::atomic<size_t> next_file = 0;
        size_t threads;
        detail::worker_pool pool;
        std::vector<uint32_t> bounds;
        std::map<cudaStream_t, std::pair<std::array<slot, 2>, unsigned>> slots;
        /// Per stream, the GPU-side similarities of the last tile write_converted converted.
        std::map<cudaStream_t, std::optional<cuda::device_buffer<float>>> converted;
        /// Per stream, GPU scratch buffers for lossless sparse-bitmap similarity compaction.
        std::map<cudaStream_t, sparse_scratch> sparse_bufs;
        /// Per stream, the GPU-side 16-bit similarities of the last tile write_similarity16 converted.
        std::map<cudaStream_t, std::optional<cuda::device_buffer<uint16_t>>> quantized;
        /// Per stream, the GPU-side 4-byte ranked counts of the last tile write_ranked_counts converted.
        std::map<cudaStream_t, std::optional<cuda::device_buffer<ranked_pairwise_counts>>> ranked;
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
