// Helpers shared by the MPI benchmarks that run one rank per GPU.
#pragma once

#include <mpi.h>
#include <nvml.h>
#include <cuddl/cuddl.cuh>
#include <cuddl/distributed.cuh>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace multi_gpu {

using json = nlohmann::json;
constexpr uint32_t k = 25;
constexpr size_t buckets = 2048;
using layout = cuddl::register_layout<5, 11>;
using database = cuddl::reference_database<k, buckets, layout>;
using group_type = cuddl::replicated_database<k, buckets, layout>;

inline void mpi_check(int status) {
    if (status != MPI_SUCCESS) throw std::runtime_error("MPI call failed");
}

inline void nvml_check(nvmlReturn_t status) {
    if (status != NVML_SUCCESS) throw std::runtime_error(nvmlErrorString(status));
}

struct ranks {
    int rank = 0, size = 1, local_rank = 0, local_size = 1;
};

inline ranks world_ranks() {
    ranks r;
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &r.rank));
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &r.size));
    MPI_Comm node;
    mpi_check(
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, r.rank, MPI_INFO_NULL, &node)
    );
    mpi_check(MPI_Comm_rank(node, &r.local_rank));
    mpi_check(MPI_Comm_size(node, &r.local_size));
    mpi_check(MPI_Comm_free(&node));
    return r;
}

/// Host NUMA node closest to @p device, or -1 without NUMA. CCCL exposes no attribute for it.
inline int host_numa_node(cuda::device_ref device) {
    int node = -1;
    CUDDL_CUDA_CALL(cudaDeviceGetAttribute(&node, cudaDevAttrHostNumaId, device.get()));
    return node;
}

/// NVML's handle for @p device, matched by PCI address because NVML and CUDA number GPUs
/// independently. CCCL's device attributes lack the PCI domain, hence the runtime call.
inline nvmlDevice_t nvml_device(cuda::device_ref device) {
    char address[32];
    CUDDL_CUDA_CALL(cudaDeviceGetPCIBusId(address, sizeof(address), device.get()));
    nvmlDevice_t handle;
    nvml_check(nvmlDeviceGetHandleByPciBusId_v2(address, &handle));
    return handle;
}

/// The GPU this rank drives, or the only visible GPU when the launcher already restricted each
/// rank to one.
///
/// Node-local ranks take GPUs in farthest-first order over the PCIe topology NVML reports: each
/// next GPU has the most distant nearest common ancestor (switch, host bridge, CPU, system) to
/// the GPUs already taken. A run with fewer ranks than GPUs then shares as few PCIe links and
/// CPU sockets as the machine allows. Every rank derives the same order.
inline cuda::device_ref rank_device(ranks const& r) {
    auto const visible = static_cast<int>(cuda::devices.size());
    if (visible == 1) return cuda::devices[0];
    if (r.local_rank >= visible) {
        throw std::runtime_error(
            "node-local rank " + std::to_string(r.local_rank) + " has no GPU; " +
            std::to_string(visible) + " are visible"
        );
    }
    std::vector<nvmlDevice_t> handles;
    for (auto const device : cuda::devices) handles.push_back(nvml_device(device));
    auto const distance = [&](int a, int b) {
        nvmlGpuTopologyLevel_t level;
        nvml_check(nvmlDeviceGetTopologyCommonAncestor(handles[a], handles[b], &level));
        return static_cast<int>(level);
    };
    std::vector<int> order{0};
    std::vector<bool> taken(visible, false);
    taken[0] = true;
    while (static_cast<int>(order.size()) <= r.local_rank) {
        int best = -1, best_distance = -1;
        for (int d = 0; d < visible; ++d) {
            if (taken[d]) continue;
            int nearest = std::numeric_limits<int>::max();
            for (auto const t : order) nearest = std::min(nearest, distance(d, t));
            if (nearest > best_distance) best = d, best_distance = nearest;
        }
        taken[best] = true;
        order.push_back(best);
    }
    return cuda::devices[order[r.local_rank]];
}

using wall_clock = std::chrono::steady_clock;

inline double ms_since(wall_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(wall_clock::now() - start).count();
}

/// Time this rank waits in a barrier of @p comm, in milliseconds.
inline double timed_barrier(MPI_Comm comm) {
    auto const start = wall_clock::now();
    mpi_check(MPI_Barrier(comm));
    return ms_since(start);
}

/// Every rank's @p mine as an array on rank 0 of @p comm; empty on the other ranks.
inline json gather_json(json const& mine, MPI_Comm comm) {
    int rank = 0, size = 1;
    mpi_check(MPI_Comm_rank(comm, &rank));
    mpi_check(MPI_Comm_size(comm, &size));
    auto const text = mine.dump();
    auto const length = static_cast<int>(text.size());
    std::vector<int> lengths(size), offsets(size);
    mpi_check(MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, comm));
    std::exclusive_scan(lengths.begin(), lengths.end(), offsets.begin(), 0);
    std::string all(rank == 0 ? static_cast<size_t>(offsets.back() + lengths.back()) : 0U, '\0');
    mpi_check(MPI_Gatherv(
        text.data(), length, MPI_CHAR, all.data(), lengths.data(), offsets.data(), MPI_CHAR, 0, comm
    ));
    auto gathered = json::array();
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            gathered.push_back(json::parse(all.substr(offsets[i], lengths[i])));
        }
    }
    return gathered;
}

/// The files a @ref parallel_writer with @p streams streams writes for @p path.
inline std::vector<std::filesystem::path>
result_files(std::filesystem::path const& path, unsigned streams) {
    if (streams <= 1) return {path};
    std::vector<std::filesystem::path> files;
    for (unsigned s = 0; s < streams; ++s)
        files.emplace_back(path.string() + "." + std::to_string(s));
    return files;
}

/// Writes records to storage on its own threads, so the caller produces the next records while
/// earlier ones are written. With one stream, @p path holds the records in order. With more,
/// stream s writes `<path>.<s>`, and each @ref append goes whole to the next stream in turn.
/// A shared file system gives its bandwidth only to many streams, and on MOGON's Lustre several
/// threads writing one file were far slower than one. Four ranks on an A100 node wrote 20 GB in
/// 9.1 s with one synchronous stream each, in 5.8 s with two files each, and in 150 s with four
/// threads into each rank's one file; the node's NVMe took 18.7 s however it was written.
class parallel_writer {
   public:
    /// @param streams Files written in parallel.
    /// @param queued_limit Bytes appended but not yet written before @ref append waits.
    parallel_writer(
        std::filesystem::path const& path,
        unsigned streams,
        size_t queued_limit = size_t{1} << 30
    )
        : queued_limit_(queued_limit), streams_(std::max(1U, streams)) {
        auto const files = result_files(path, streams);
        for (size_t s = 0; s < streams_.size(); ++s) {
            streams_[s].fd = ::open(files[s].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (streams_[s].fd == -1) {
                close();
                throw std::runtime_error("cannot write " + files[s].string());
            }
        }
        for (auto& stream : streams_) {
            stream.thread = std::thread([this, &stream] { work(stream); });
        }
    }
    parallel_writer(parallel_writer const&) = delete;
    parallel_writer& operator=(parallel_writer const&) = delete;
    ~parallel_writer() {
        stop();
        close();
    }

    /// Queues a copy of @p chunks, written together; waits while too many bytes are queued.
    void append(std::initializer_list<std::span<char const>> chunks) {
        size_t bytes = 0;
        for (auto const& chunk : chunks) bytes += chunk.size();
        if (bytes == 0) return;
        std::unique_lock lock(mutex_);
        space_.wait(lock, [&] {
            return failed_ || queued_ == 0 || queued_ + bytes <= queued_limit_;
        });
        if (failed_) throw std::runtime_error("cannot write the result files");
        queued_ += bytes;
        auto& stream = streams_[next_++ % streams_.size()];
        lock.unlock();
        std::vector<char> copy;
        copy.reserve(bytes);
        for (auto const& chunk : chunks) copy.insert(copy.end(), chunk.begin(), chunk.end());
        lock.lock();
        stream.queue.push_back(std::move(copy));
        lock.unlock();
        ready_.notify_all();
    }

    void append(void const* data, size_t bytes) {
        append({std::span{static_cast<char const*>(data), bytes}});
    }

    /// Waits for every write, then flushes the files to storage and closes them.
    void finish() {
        {
            std::unique_lock lock(mutex_);
            space_.wait(lock, [&] { return failed_ || queued_ == 0; });
        }
        stop();
        bool synced = !failed_;
        for (auto& stream : streams_) synced = synced && ::fsync(stream.fd) == 0;
        close();
        if (!synced) throw std::runtime_error("cannot write the result files");
    }

   private:
    struct stream_state {
        int fd = -1;
        std::deque<std::vector<char>> queue;
        std::thread thread;
    };

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& stream : streams_) {
            if (stream.thread.joinable()) stream.thread.join();
        }
    }

    void close() {
        for (auto& stream : streams_) {
            if (stream.fd != -1) ::close(stream.fd);
            stream.fd = -1;
        }
    }

    void work(stream_state& stream) {
        while (true) {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || !stream.queue.empty(); });
            if (stream.queue.empty()) return;
            auto next = std::move(stream.queue.front());
            stream.queue.pop_front();
            lock.unlock();
            size_t done = 0;
            while (done < next.size()) {
                auto const wrote = ::write(stream.fd, next.data() + done, next.size() - done);
                if (wrote <= 0) break;
                done += static_cast<size_t>(wrote);
            }
            lock.lock();
            if (done != next.size()) failed_ = true;
            queued_ -= next.size();
            lock.unlock();
            space_.notify_all();
        }
    }

    size_t queued_limit_;
    std::vector<stream_state> streams_;
    std::mutex mutex_;
    std::condition_variable ready_, space_;
    size_t queued_ = 0, next_ = 0;
    bool stopping_ = false, failed_ = false;
};

/// The ranks of @p comm sharing one search: each claims tiles until none are left, so a faster
/// rank searches more of the pairs. With CUDDL_ABLATE=tile-queue each keeps the tiles its shard
/// assigns instead.
struct tile_claims {
    cuddl::all_to_all_shard shard;
    cuddl::shared_tile_counter counter;

    static tile_claims make(MPI_Comm comm) {
        int rank = 0, size = 1;
        mpi_check(MPI_Comm_rank(comm, &rank));
        mpi_check(MPI_Comm_size(comm, &size));
        return {
            {.index = static_cast<uint32_t>(rank), .count = static_cast<uint32_t>(size)},
            CUDDL_UNWRAP(cuddl::shared_tile_counter::make(comm)),
        };
    }

    /// The split for the next search. Resets the counter, two barriers, so every rank must call
    /// it before each search.
    cuddl::replica_search next() {
        CUDDL_UNWRAP(counter.reset());
        return {.shard = shard, .next_tile = [this] { return counter(); }};
    }
};

}  // namespace multi_gpu
