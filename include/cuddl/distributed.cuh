#pragma once

// Sketching and searching one genome collection across the ranks of an MPI communicator.
// Needs MPI, so cuddl.cuh leaves it out; link `cuddl_distributed_dep`.
//
// Every rank calls each function collectively, with the same arguments where noted:
//
//     auto slice = CUDDL_TRY(cuddl::slice_genomes(input, comm));
//     auto mine = CUDDL_TRY((cuddl::reference_database_file::build<K, B>(slice.mine(), stream)));
//     auto file = CUDDL_TRY(cuddl::gather_sketches(mine, slice, comm));
//
// `file` then holds every genome on every rank, in reference-ID order, the same rows a
// single-process build of `slice.paths` gives. To search it all-to-all across the ranks,
// upload it with replicated_database::upload on each rank and search collectively:
//
//     auto group = CUDDL_TRY((cuddl::replicated_database<K, B>::upload(file, devices)));
//     CUDDL_TRY(cuddl::search_all_to_all(group, on_tile, comm));
//
// Ranks claim tiles from a counter they share until none are left, so a faster GPU searches
// more of the pairs.
//
// A rank that fails inside one of these functions returns its error, and every other rank
// returns an error naming it, so no rank is left waiting in a collective. A rank that fails
// between them (for example in the build) must still reach the next call or abort the
// communicator.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <cuddl/reference_database_file.cuh>
#include <cuddl/replicated_database.cuh>

namespace cuddl {

/// @brief One rank's share of a genome collection split across an MPI communicator.
struct genome_slice {
    /// Every genome of the collection in reference-ID order; the same on every rank.
    std::vector<std::filesystem::path> paths;
    /// Size of each genome's file in bytes.
    std::vector<uint64_t> bytes;
    /// This rank's genomes are `paths[first, last)`.
    size_t first = 0;
    size_t last = 0;

    [[nodiscard]] std::span<std::filesystem::path const> mine() const noexcept {
        return std::span{paths}.subspan(first, last - first);
    }
};

namespace detail {

[[nodiscard]] inline Result<void> mpi_try(int status, char const* call) {
    if (status == MPI_SUCCESS) return Ok();
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    return Err(Error::resource(std::string(call) + ": " + std::string(text, length)));
}

/// Returns @p local on the rank that failed, and on every other rank an error naming the lowest
/// failing rank, so that all ranks leave a collective step together.
[[nodiscard]] inline Result<void> agree(Result<void> local, MPI_Comm comm) {
    int rank = 0, size = 1;
    CUDDL_TRY(mpi_try(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank"));
    CUDDL_TRY(mpi_try(MPI_Comm_size(comm, &size), "MPI_Comm_size"));
    int const mine = local ? size : rank;
    int failed = size;
    CUDDL_TRY(mpi_try(MPI_Allreduce(&mine, &failed, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce"));
    if (!local) return local;
    if (failed == size) return Ok();
    return Err(Error::resource("rank " + std::to_string(failed) + " failed"));
}

/// Rank 0's listing of @p input on every rank.
[[nodiscard]] inline Result<std::vector<std::filesystem::path>>
share_listing(std::filesystem::path const& input, int rank, MPI_Comm comm) {
    std::string joined;
    auto listed = [&]() -> Result<void> {
        if (rank != 0) return Ok();
        auto const paths = CUDDL_TRY(list_genomes(input));
        for (auto const& path : paths) joined += path.string() + '\n';
        return Ok();
    }();
    CUDDL_TRY(agree(std::move(listed), comm));
    uint64_t length = joined.size();
    CUDDL_TRY(mpi_try(MPI_Bcast(&length, 1, MPI_UINT64_T, 0, comm), "MPI_Bcast"));
    joined.resize(length);
    // MPI counts are int: send the listing in chunks below 2 GiB.
    constexpr uint64_t chunk = uint64_t{1} << 30U;
    for (uint64_t offset = 0; offset < length; offset += chunk) {
        auto const count = static_cast<int>(std::min(chunk, length - offset));
        CUDDL_TRY(
            mpi_try(MPI_Bcast(joined.data() + offset, count, MPI_CHAR, 0, comm), "MPI_Bcast")
        );
    }
    std::vector<std::filesystem::path> paths;
    for (size_t begin = 0; begin < joined.size();) {
        auto const end = joined.find('\n', begin);
        paths.emplace_back(joined.substr(begin, end - begin));
        begin = end + 1;
    }
    return paths;
}

/// First genome of rank @p rank's slice: contiguous slices of about equal bytes.
[[nodiscard]] inline size_t
slice_begin(std::span<uint64_t const> bytes, uint64_t total, int rank, int size) {
    auto const target = static_cast<uint64_t>(static_cast<double>(total) * rank / size);
    uint64_t sum = 0;
    size_t i = 0;
    while (i < bytes.size() && sum + bytes[i] / 2 < target) sum += bytes[i++];
    return i;
}

/// Builds a @ref reference_database_file from gathered parts; the file's members are private.
struct reference_file_assembly {
    static reference_database_file make(
        reference_database_file const& like,
        std::vector<std::filesystem::path> const& paths,
        std::vector<uint16_t> rows
    ) {
        reference_database_file file;
        file.blacklist_ = like.blacklist_;
        file.metadata_ = like.metadata_;
        file.metadata_.reference_count = static_cast<uint32_t>(paths.size());
        file.names_.reserve(paths.size());
        for (auto const& path : paths) file.names_.push_back(path.string());
        file.rows_ = std::move(rows);
        return file;
    }
};

}  // namespace detail

/**
 * @brief Lists a genome collection once and splits it into one contiguous slice per rank of
 * @p comm, of about equal file bytes. Collective.
 *
 * Rank 0 lists @p input, a directory (see @ref genome_paths) or a text file with one path per
 * line, and shares the list. Every rank then looks up the sizes of an equal share of the files
 * with @p workers threads, because on a parallel file system each lookup waits for a metadata
 * round trip, and the ranks exchange them. Slices follow path order, so concatenating them in
 * rank order gives the reference IDs of a single-process build. Bytes stand in for sketching
 * work; a rank may get no genomes when there are fewer genomes than ranks.
 */
[[nodiscard]] inline Result<genome_slice> slice_genomes(
    std::filesystem::path const& input,
    MPI_Comm comm,
    unsigned workers = default_parser_workers()
) {
    if (workers == 0U)
        return Err(Error::invalid_argument("slice_genomes needs at least one worker"));
    int rank = 0, size = 1;
    CUDDL_TRY(detail::mpi_try(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank"));
    CUDDL_TRY(detail::mpi_try(MPI_Comm_size(comm, &size), "MPI_Comm_size"));
    genome_slice slice;
    slice.paths = CUDDL_TRY(detail::share_listing(input, rank, comm));
    auto const& paths = slice.paths;
    if (paths.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Err(Error::resource("genome collection exceeds MPI counts"));
    }

    auto const share = [&](int r) {
        return paths.size() * static_cast<size_t>(r) / static_cast<size_t>(size);
    };
    auto const own = share(rank);
    std::vector<uint64_t> mine(share(rank + 1) - own);
    std::vector<std::error_code> errors(mine.size());
    {
        std::vector<std::jthread> threads;
        for (unsigned t = 0; t < workers && t < mine.size(); ++t) {
            threads.emplace_back([&, t] {
                for (size_t i = t; i < mine.size(); i += workers) {
                    mine[i] = std::filesystem::file_size(paths[own + i], errors[i]);
                }
            });
        }
    }
    auto sized = [&]() -> Result<void> {
        for (size_t i = 0; i < mine.size(); ++i) {
            if (errors[i]) {
                return Err(Error::resource(paths[own + i].string() + ": " + errors[i].message()));
            }
        }
        return Ok();
    }();
    CUDDL_TRY(detail::agree(std::move(sized), comm));

    std::vector<int> counts(size), offsets(size);
    for (int r = 0; r < size; ++r) {
        offsets[r] = static_cast<int>(share(r));
        counts[r] = static_cast<int>(share(r + 1) - share(r));
    }
    slice.bytes.resize(paths.size());
    CUDDL_TRY(
        detail::mpi_try(
            MPI_Allgatherv(
                mine.data(),
                static_cast<int>(mine.size()),
                MPI_UINT64_T,
                slice.bytes.data(),
                counts.data(),
                offsets.data(),
                MPI_UINT64_T,
                comm
            ),
            "MPI_Allgatherv"
        )
    );
    auto const total = std::accumulate(slice.bytes.begin(), slice.bytes.end(), uint64_t{0});
    slice.first = detail::slice_begin(slice.bytes, total, rank, size);
    slice.last =
        rank + 1 == size ? paths.size() : detail::slice_begin(slice.bytes, total, rank + 1, size);
    return slice;
}

namespace detail {

/// Every rank's genome count, after checking collectively that @p mine holds the sketches of
/// this rank's slice and that every rank built with rank 0's settings.
[[nodiscard]] inline Result<std::vector<uint32_t>>
slice_counts(reference_database_file const& mine, genome_slice const& slice, MPI_Comm comm) {
    auto const metadata = mine.metadata();
    auto reference = metadata.compatibility;
    CUDDL_TRY(mpi_try(MPI_Bcast(&reference, sizeof(reference), MPI_BYTE, 0, comm), "MPI_Bcast"));
    auto local = [&]() -> Result<void> {
        if (metadata.reference_count != slice.last - slice.first) {
            return Err(Error::invalid_argument("sketches do not match this rank's genome slice"));
        }
        if (!(reference == metadata.compatibility)) {
            return Err(Error::invalid_argument("ranks sketched with different settings"));
        }
        return Ok();
    }();
    CUDDL_TRY(agree(std::move(local), comm));
    int size = 1;
    CUDDL_TRY(mpi_try(MPI_Comm_size(comm, &size), "MPI_Comm_size"));
    uint32_t const count = metadata.reference_count;
    std::vector<uint32_t> counts(size);
    CUDDL_TRY(mpi_try(
        MPI_Allgather(&count, 1, MPI_UINT32_T, counts.data(), 1, MPI_UINT32_T, comm),
        "MPI_Allgather"
    ));
    if (std::accumulate(counts.begin(), counts.end(), uint64_t{0}) != slice.paths.size()) {
        return Err(Error::invalid_argument("ranks hold slices of different genome collections"));
    }
    return counts;
}

/// One row of @p buckets scores as a single MPI element, so counts of rows stay within int.
[[nodiscard]] inline Result<MPI_Datatype> row_type(uint32_t buckets) {
    MPI_Datatype row = MPI_DATATYPE_NULL;
    CUDDL_TRY(mpi_try(
        MPI_Type_contiguous(static_cast<int>(buckets), MPI_UINT16_T, &row), "MPI_Type_contiguous"
    ));
    CUDDL_TRY(mpi_try(MPI_Type_commit(&row), "MPI_Type_commit"));
    return row;
}

}  // namespace detail

/**
 * @brief Gathers every rank's sketches of its @ref genome_slice into one file on every rank.
 * Collective.
 *
 * @p mine must hold the sketches of `slice.mine()`, in order, and every rank must have built
 * with the same k, buckets, layout and blacklist; both are checked. The result equals a
 * single-process build of `slice.paths`: rows in reference-ID order, labelled with the paths.
 * Rows travel over MPI from host memory; every rank receives all of them.
 */
[[nodiscard]] inline Result<reference_database_file>
gather_sketches(reference_database_file const& mine, genome_slice const& slice, MPI_Comm comm) {
    auto const counts = CUDDL_TRY(detail::slice_counts(mine, slice, comm));
    std::vector<int> sizes(counts.begin(), counts.end()), offsets(counts.size());
    std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), 0);
    auto const buckets = mine.metadata().compatibility.bucket_count;
    auto row = CUDDL_TRY(detail::row_type(buckets));
    std::vector<uint16_t> rows(slice.paths.size() * buckets);
    auto const gathered = detail::mpi_try(
        MPI_Allgatherv(
            mine.rows().data(),
            static_cast<int>(mine.metadata().reference_count),
            row,
            rows.data(),
            sizes.data(),
            offsets.data(),
            row,
            comm
        ),
        "MPI_Allgatherv"
    );
    MPI_Type_free(&row);
    CUDDL_TRY(gathered);
    return detail::reference_file_assembly::make(mine, slice.paths, std::move(rows));
}

/**
 * @brief A tile counter shared by every rank of a communicator, for
 * replica_search::next_tile. Making, resetting and destroying it are collective.
 *
 * Rank 0 holds the count in an MPI window, and each claim is one remote fetch-and-add: about
 * 20 microseconds across nodes, against tiles that take milliseconds to search. Open MPI
 * completes a claim from another node only while rank 0 is inside MPI, even with hardware
 * atomics: while rank 0 waited on its GPU, claims from the other node took 70-130 ms. Rank 0
 * therefore runs a thread that drives MPI progress for as long as the counter lives, so MPI
 * must provide `MPI_THREAD_MULTIPLE`. The thread polls without sleeping, because Open MPI
 * drives one-sided operations only on some progress calls: with a 20 us sleep between polls,
 * claims took 10-21 ms. It yields, so it occupies a core only while no other thread wants one.
 * A counter serves one search; call @ref reset on every rank before the next. A failed claim
 * aborts the communicator, because a lost tile would silently drop pairs from the results.
 */
class shared_tile_counter {
   public:
    __host__ shared_tile_counter(shared_tile_counter&& other) noexcept
        : state_(std::move(other.state_)) {}
    __host__ shared_tile_counter& operator=(shared_tile_counter&& other) noexcept {
        state_ = std::move(other.state_);
        return *this;
    }
    __host__ ~shared_tile_counter() {}  // NOLINT(modernize-use-equals-default)

    [[nodiscard]] static Result<shared_tile_counter> make(MPI_Comm comm) {
        int provided = MPI_THREAD_SINGLE;
        CUDDL_TRY(detail::mpi_try(MPI_Query_thread(&provided), "MPI_Query_thread"));
        if (provided < MPI_THREAD_MULTIPLE) {
            return Err(
                Error::invalid_argument(
                    "shared_tile_counter needs MPI initialised with MPI_THREAD_MULTIPLE"
                )
            );
        }
        auto state = std::make_unique<shared_state>();
        state->comm = comm;
        CUDDL_TRY(detail::mpi_try(MPI_Comm_rank(comm, &state->rank), "MPI_Comm_rank"));
        auto const bytes = state->rank == 0 ? static_cast<MPI_Aint>(sizeof(uint64_t)) : 0;
        CUDDL_TRY(
            detail::mpi_try(
                MPI_Win_allocate(
                    bytes, sizeof(uint64_t), MPI_INFO_NULL, comm, &state->count, &state->window
                ),
                "MPI_Win_allocate"
            )
        );
        CUDDL_TRY(detail::mpi_try(MPI_Win_lock_all(0, state->window), "MPI_Win_lock_all"));
        state->locked = true;
        if (state->rank == 0) {
            *state->count = 0;
            CUDDL_TRY(detail::mpi_try(MPI_Win_sync(state->window), "MPI_Win_sync"));
            CUDDL_TRY(
                detail::mpi_try(MPI_Comm_dup(MPI_COMM_SELF, &state->progress_comm), "MPI_Comm_dup")
            );
            // Probing a communicator nobody sends on runs MPI's progress engine, which
            // completes the claims other ranks have in flight.
            if (!detail::ablated("progress-thread")) {
                state->progress =
                    std::jthread([comm = state->progress_comm](const std::stop_token& stop) {
                        while (!stop.stop_requested()) {
                            int flag = 0;
                            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
                            std::this_thread::yield();
                        }
                    });
            }
        }
        CUDDL_TRY(detail::mpi_try(MPI_Barrier(comm), "MPI_Barrier"));
        shared_tile_counter counter;
        counter.state_ = std::move(state);
        return Result<shared_tile_counter>::ok(std::move(counter));
    }

    /// @brief The next tile number of the search, counting from 0 across every rank.
    [[nodiscard]] uint64_t operator()() {
        std::lock_guard const lock(state_->update);
        uint64_t const one = 1;
        uint64_t previous = 0;
        if (MPI_Fetch_and_op(&one, &previous, MPI_UINT64_T, 0, 0, MPI_SUM, state_->window) !=
                MPI_SUCCESS ||
            MPI_Win_flush(0, state_->window) != MPI_SUCCESS) {
            MPI_Abort(state_->comm, 1);
        }
        return previous;
    }

    /// @brief Starts the count again at 0 once every rank has finished the previous search.
    [[nodiscard]] Result<void> reset() {
        CUDDL_TRY(detail::mpi_try(MPI_Barrier(state_->comm), "MPI_Barrier"));
        if (state_->rank == 0) {
            uint64_t const zero = 0;
            CUDDL_TRY(
                detail::mpi_try(
                    MPI_Accumulate(
                        &zero, 1, MPI_UINT64_T, 0, 0, 1, MPI_UINT64_T, MPI_REPLACE, state_->window
                    ),
                    "MPI_Accumulate"
                )
            );
            CUDDL_TRY(detail::mpi_try(MPI_Win_flush(0, state_->window), "MPI_Win_flush"));
        }
        return detail::mpi_try(MPI_Barrier(state_->comm), "MPI_Barrier");
    }

   private:
    struct shared_state {
        MPI_Comm comm = MPI_COMM_NULL;
        MPI_Comm progress_comm = MPI_COMM_NULL;
        MPI_Win window = MPI_WIN_NULL;
        uint64_t* count = nullptr;
        int rank = 0;
        bool locked = false;
        std::mutex update;
        std::jthread progress;

        ~shared_state() {
            if (progress.joinable()) {
                progress.request_stop();
                progress.join();
            }
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (finalized) return;
            if (progress_comm != MPI_COMM_NULL) MPI_Comm_free(&progress_comm);
            if (window == MPI_WIN_NULL) return;
            if (locked) MPI_Win_unlock_all(window);
            MPI_Win_free(&window);
        }
    };

    shared_tile_counter() = default;
    std::unique_ptr<shared_state> state_;
};

namespace detail {

/// @brief Runs @p search with a replica_search that shares tiles across the ranks of @p comm.
template <typename Search>
[[nodiscard]] Result<void> shared_search(MPI_Comm comm, Search&& search) {
    int rank = 0;
    int size = 1;
    CUDDL_TRY(mpi_try(MPI_Comm_rank(comm, &rank), "MPI_Comm_rank"));
    CUDDL_TRY(mpi_try(MPI_Comm_size(comm, &size), "MPI_Comm_size"));
    auto counter = CUDDL_TRY(shared_tile_counter::make(comm));
    return search(
        replica_search{
            .shard = {.index = static_cast<uint32_t>(rank), .count = static_cast<uint32_t>(size)},
            .next_tile = detail::ablated("tile-queue") ? std::function<uint64_t()>{}
                                                       : [&counter] { return counter(); },
        }
    );
}

}  // namespace detail

/**
 * @brief Compares every pair of @p group's references, sharing the tiles among the ranks of
 * @p comm; collective.
 *
 * Every rank calls it with a replicated_database of the same rows. Each rank delivers the tiles
 * it claims to @p on_tile, as replicated_database::search_all_to_all does, and every pair is
 * delivered on exactly one rank. The ranks share one shared_tile_counter for the call.
 */
template <uint32_t K, size_t BucketCount, typename Layout, typename OnTile>
[[nodiscard]] Result<void> search_all_to_all(
    replicated_database<K, BucketCount, Layout>& group,
    OnTile&& on_tile,
    MPI_Comm comm
) {
    return detail::shared_search(comm, [&](replica_search split) {
        return group.search_all_to_all(on_tile, std::move(split));
    });
}

/// @brief The threshold form of @ref search_all_to_all: only pairs passing @p options reach
/// @p on_tile. @p group needs its index (replicated_database::build_index); collective.
template <uint32_t K, size_t BucketCount, typename Layout, typename OnTile>
[[nodiscard]] Result<void> search_all_to_all(
    replicated_database<K, BucketCount, Layout>& group,
    OnTile&& on_tile,
    search_options options,
    MPI_Comm comm
) {
    return detail::shared_search(comm, [&](replica_search split) {
        return group.search_all_to_all(on_tile, options, std::move(split));
    });
}

}  // namespace cuddl
