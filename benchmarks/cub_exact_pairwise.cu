// cub-exact-pairwise: exact k-mer set baseline from CUB device primitives.
//
// Parses FASTA files with the cuDDL parser (k=25 canonical packed k-mers). Genomes touched by
// evaluated pairs keep their packed arrays; the rest stream through for distinct counts only, so
// host memory stays bounded by the evaluated set. Sketching queues a batch of genomes' sorts and
// encodes behind each other and reads the whole batch's run counts back at once, so no genome
// waits on a host round trip. Per evaluated pair: blocks take equal tiles of the merge of two
// sorted, deduplicated arrays and count the shared k-mers, so shared = |A| + |B| - union. The
// only downloads are the per-batch counts and one integer per pair batch; pair metrics are host
// math.
//
// --max-pairs stride-samples the evaluated pair space (first and last pair
// always measured) and --match-rows stride-samples the emitted rows the same
// way; 0 disables either cap. --sketch-only skips pair intersection for
// full-corpus parse timing; pair metrics come from the subset run instead.
// matching the CLI tools, with the parse subtotal reported separately.
// phases_ms.resident_sketch sums per-batch device spans for sequence packing, compaction,
// sorting and dedup on staged bytes; phases_ms.resident_compare sums the pair kernels only.
// Compare rebuilds stay untimed, and transfers and host parsing stay in the wall phases.
// Allocation and temp-storage sizing stay outside timing; the report carries
#include <cuda_pipeline.h>
#include <vector_types.h>
#include <cuddl/cuda_error.hpp>
#include <cuddl/error.hpp>
#include <cuddl/fastx.hpp>

#include <CLI/CLI.hpp>
#include <cub/cub.cuh>
#include <nlohmann/json.hpp>

#include "resident_sequence_batches.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

// Canonical packed k-mer length, the same one the cuDDL parser this baseline stages from uses.
uint32_t constexpr kmer_length = 25;

/// @brief Host memory a run may spend on resident k-mer arrays, or 0 when it cannot be read.
[[nodiscard]] size_t available_host_bytes() noexcept {
    size_t available = 0;
    if (std::FILE* info = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, info) != nullptr) {
            unsigned long long kib = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
                available = static_cast<size_t>(kib) * 1024;
                break;
            }
        }
        std::fclose(info);
    }
    return available;
}

[[nodiscard]] unsigned parse_worker_count(size_t genomes) noexcept {
    auto const hardware = std::max(1U, std::thread::hardware_concurrency());
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(genomes, hardware)));
}

using clock_type = std::chrono::steady_clock;

double median_of(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("no timing samples");
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2) return *middle;
    return (*std::max_element(values.begin(), middle) + *middle) / 2;
}

struct device_buffer {
    void* data = nullptr;
    size_t bytes = 0;
    void reset(size_t size) {
        if (size <= bytes) return;
        if (data) CUDDL_CUDA_CALL(cudaFree(data));
        CUDDL_CUDA_CALL(cudaMalloc(&data, size));
        bytes = size;
    }
    ~device_buffer() {
        if (data) CUDDL_CUDA_ABORT(cudaFree(data));
    }
};
struct event_span {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

// Device-only time on one stream. Each span brackets device work while transfers and host
// work stay outside; elapsed times are read only at stream syncs the workload already
// performs, so timing adds no synchronization. Bounded spans are summed per replicate.
struct event_accum {
    cudaStream_t stream = nullptr;
    std::vector<event_span> pool;
    std::vector<size_t> pending;
    size_t cursor = 0;
    double ms = 0;

    event_accum() = default;

    explicit event_accum(cudaStream_t s) : stream(s) {}

    size_t begin() {
        if (cursor >= pool.size()) {
            event_span span;
            CUDDL_CUDA_CALL(cudaEventCreate(&span.start));
            CUDDL_CUDA_CALL(cudaEventCreate(&span.stop));
            pool.push_back(span);
        }
        CUDDL_CUDA_CALL(cudaEventRecord(pool[cursor].start, stream));
        pending.push_back(cursor);
        return cursor++;
    }

    void end(size_t idx) {
        CUDDL_CUDA_CALL(cudaEventRecord(pool[idx].stop, stream));
    }

    void resolve() {
        for (size_t idx : pending) {
            float span_ms = 0;
            CUDDL_CUDA_CALL(cudaEventElapsedTime(&span_ms, pool[idx].start, pool[idx].stop));
            ms += span_ms;
        }
        pending.clear();
    }

    void reset_sample() {
        cursor = 0;
        pending.clear();
        ms = 0;
    }

    ~event_accum() {
        for (auto& span : pool) {
            if (span.start) CUDDL_CUDA_ABORT(cudaEventDestroy(span.start));
            if (span.stop) CUDDL_CUDA_ABORT(cudaEventDestroy(span.stop));
        }
    }
};

size_t sort_temp_bytes(size_t count) {
    size_t bytes = 0;
    if (count) {
        CUDDL_CUDA_CALL(
            cub::DeviceRadixSort::SortKeys(
                nullptr,
                bytes,
                static_cast<uint64_t*>(nullptr),
                static_cast<uint64_t*>(nullptr),
                count
            )
        );
    }
    return bytes;
}

size_t encode_temp_bytes(size_t count) {
    size_t bytes = 0;
    if (count) {
        CUDDL_CUDA_CALL(
            cub::DeviceRunLengthEncode::Encode(
                nullptr,
                bytes,
                static_cast<uint64_t const*>(nullptr),
                static_cast<uint64_t*>(nullptr),
                static_cast<int*>(nullptr),
                static_cast<size_t*>(nullptr),
                count
            )
        );
    }
    return bytes;
}

size_t select_temp_bytes(size_t count) {
    size_t bytes = 0;
    if (count) {
        CUDDL_CUDA_CALL(
            cub::DeviceSelect::Flagged(
                nullptr,
                bytes,
                static_cast<uint64_t const*>(nullptr),
                static_cast<uint8_t const*>(nullptr),
                static_cast<uint64_t*>(nullptr),
                static_cast<int*>(nullptr),
                static_cast<int64_t>(count)
            )
        );
    }
    return bytes;
}

/// @brief Device memory this run may spend on sorted k-mer arrays, or 0 when it cannot be read.
[[nodiscard]] size_t available_device_bytes() noexcept {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    return free_bytes;
}

/// @brief One staged piece of one genome's sequence, and the windows it contributes.
///
/// The loader gives each piece the k - 1 bases it shares with the previous one, so a piece's
/// windows are disjoint from its neighbours' and never span a record: `size - k + 1` windows is
/// exactly the window count a host parse of that piece would have seen, counted once.
struct staged_chunk {
    size_t offset;       // first byte of the piece inside the staged batch
    size_t window_base;  // first window of the piece inside the flat window array
    uint32_t windows;    // complete windows the piece holds
};

/// Windows each emit block covers. A long piece spans many tiles and a tile can span short pieces,
/// so the grid follows the windows: one block per piece left a batch of a few genomes, whose
/// pieces are whole contigs, on a handful of SMs.
constexpr uint32_t emit_tile = 8192;

/// @brief Emits the canonical packed k-mers of every staged piece, emit_tile windows per block.
///
/// Packing k-mers on the host spent the corpus on 72 cores while the device idled. A window is
/// one k-mer unless an ambiguous base breaks it, which emits nothing and clears its flag for the
/// compaction to drop. A genome's pieces stay in order, so its keys stay together for the sort.
/// A piece's valid windows add up in `valid`, which starts at zero, across the tiles it spans.
__global__ void emit_chunk_kmers(
    char const* bases,
    staged_chunk const* chunks,
    uint32_t chunk_count,
    uint64_t window_count,
    uint32_t k,
    uint64_t* keys,
    uint8_t* flags,
    uint32_t* valid
) {
    auto const first = uint64_t{blockIdx.x} * emit_tile;
    auto const last = min(first + emit_tile, window_count);
    // The last piece that starts at or before the tile.
    uint32_t piece = 0;
    for (uint32_t hi = chunk_count - 1; piece < hi;) {
        auto const mid = piece + (hi - piece + 1) / 2;
        if (chunks[mid].window_base <= first) {
            piece = mid;
        } else {
            hi = mid - 1;
        }
    }
    using reduce_type = cub::BlockReduce<uint32_t, 256>;
    __shared__ typename reduce_type::TempStorage storage;
    for (; piece < chunk_count && chunks[piece].window_base < last; ++piece) {
        auto const chunk = chunks[piece];
        auto const end = min(last, chunk.window_base + chunk.windows);
        char const* const sequence = bases + chunk.offset;
        uint32_t mine = 0;
        for (auto w = max(first, chunk.window_base) + threadIdx.x; w < end; w += blockDim.x) {
            auto const i = w - chunk.window_base;
            uint64_t forward = 0;
            bool complete = true;
            for (uint32_t j = 0; j < k; ++j) {
                auto const symbol = cuddl::detail::encode_base(sequence[i + j]);
                complete = complete && symbol != 0xFFU;
                forward = (forward << 2U) | (symbol & 3U);
            }
            auto const reverse = cuddl::detail::reverse_complement(forward, k);
            keys[w] = forward > reverse ? forward : reverse;
            flags[w] = complete ? 1U : 0U;
            mine += complete ? 1U : 0U;
        }
        auto const total = reduce_type(storage).Sum(mine);
        if (threadIdx.x == 0 && total) atomicAdd(valid + piece, total);
        // The next piece's sum reuses the reduction storage.
        __syncthreads();
    }
}

/// @brief One pair of sorted, deduplicated k-mer arrays to count the intersection of.
struct pair_job {
    uint64_t const* a;
    uint32_t a_len;
    uint64_t const* b;
    uint32_t b_len;
};

/// Threads per merge block, merged keys each thread walks, and so the merged keys per tile.
constexpr uint32_t merge_threads = 256;
constexpr uint32_t merge_items = 16;
constexpr uint32_t merge_tile = merge_threads * merge_items;
/// Consecutive tiles a block walks: one job lookup for them all, and each tile's end in A is the
/// next one's start.
constexpr uint32_t merge_chunk = 8;
/// Above every 25-mer, so no key equals it: what a side reads once it has no keys left.
constexpr uint64_t no_key = ~uint64_t{0};

/// @brief A keys among the first @p diagonal keys of the merge of sorted @p a and @p b, which
/// takes A first on ties, so an equal pair always sits as A then B; called by a whole warp.
///
/// Key i is in the first @p diagonal while a[i] <= b[diagonal - 1 - i], which holds up to the
/// split and fails after it. Every lane probes one point of the remaining range at once, so the
/// range shrinks 32-fold per round of dependent loads instead of halving; more probes per round
/// would cost more random sector reads than the rounds they save.
__device__ __forceinline__ uint64_t warp_merge_split(
    uint64_t const* a,
    uint64_t a_len,
    uint64_t const* b,
    uint64_t b_len,
    uint64_t diagonal
) {
    auto const lane = threadIdx.x % 32;
    auto lo = diagonal > b_len ? diagonal - b_len : 0;
    auto hi = min(diagonal, a_len);
    while (lo < hi) {
        auto const step = (hi - lo + 31) / 32;
        auto const probe = lo + lane * step;
        bool const below = probe < hi && a[probe] <= b[diagonal - 1 - probe];
        auto const held = static_cast<uint64_t>(__popc(__ballot_sync(~0U, below)));
        // The split lies after the last probe that held and at or before the first that failed.
        auto const next_hi = min(hi, lo + held * step);
        lo = held != 0 ? lo + (held - 1) * step + 1 : lo;
        hi = next_hi;
    }
    return lo;
}

static_assert(merge_chunk == merge_threads / 32, "one warp finds each tile's end");

/// @brief Counts the k-mers two sorted, deduplicated sets share, merge_chunk tiles of their merges
/// per block.
///
/// Pair p's merge, |A| + |B| keys long, is cut into tiles of merge_tile keys, tiles
/// @p tile_first[p] onward. A block takes merge_chunk consecutive tiles, and its warps first find
/// where each of them ends in A, all at once by merge-path searches: a tile starts where the one
/// before it ended, or at 0 for a new pair, and the last tile of a pair ends at |A|. Each tile is
/// then staged in shared memory with coalesced copies, and every thread merges merge_items of its
/// merged order, so work is the same for every thread whatever the keys. The merge takes A first
/// on ties, so a shared k-mer is an A key directly followed by its equal B key: it counts when the
/// A key is taken while the B head equals it. The B key after a tile is staged too, for a pair
/// the tile boundary splits.
__global__ __launch_bounds__(merge_threads) void merge_intersection_kernel(
    pair_job const* jobs,
    uint64_t const* tile_first,
    uint32_t job_count,
    uint64_t first_block,
    uint32_t* counts
) {
    // Logical key x is slot x + x / 8: A keys from 0, B keys from na, the B key after the tile at
    // n. Threads start their merges about eight keys apart, which without the pad word every
    // eight keys lands a warp on the same banks.
    __shared__ uint64_t keys[merge_tile + merge_tile / 8 + 1];
    // Where the block's first tile starts in A, then where each tile ends.
    __shared__ uint64_t bounds[merge_chunk + 1];
    using reduce_type = cub::BlockReduce<uint32_t, merge_threads>;
    __shared__ typename reduce_type::TempStorage storage;
    auto const at = [&](uint32_t x) -> uint64_t& {
        return keys[x + (x >> 3)];
    };
    auto const first = (first_block + blockIdx.x) * merge_chunk;
    auto const last = min(first + merge_chunk, tile_first[job_count]);
    if (first >= last) return;
    // The last job whose tiles start at or before the first; jobs without tiles share its start.
    uint32_t start_job = 0;
    for (uint32_t hi = job_count - 1; start_job < hi;) {
        auto const mid = start_job + (hi - start_job + 1) / 2;
        if (tile_first[mid] <= first) {
            start_job = mid;
        } else {
            hi = mid - 1;
        }
    }
    // Tile `tile`'s job, found by walking on from @p job, and its merged keys [d0, d1).
    auto const place = [&](uint64_t tile, uint32_t& job, uint64_t& d0, uint64_t& d1) {
        while (tile >= tile_first[job + 1]) ++job;
        d0 = (tile - tile_first[job]) * merge_tile;
        d1 = min(d0 + merge_tile, uint64_t{jobs[job].a_len} + jobs[job].b_len);
    };
    {
        auto const warp = threadIdx.x / 32;
        auto const tile = first + warp;
        if (tile < last) {
            auto job = start_job;
            uint64_t d0 = 0, d1 = 0;
            place(tile, job, d0, d1);
            auto const& pair = jobs[job];
            auto const end = d1 == uint64_t{pair.a_len} + pair.b_len
                                 ? uint64_t{pair.a_len}
                                 : warp_merge_split(pair.a, pair.a_len, pair.b, pair.b_len, d1);
            if (threadIdx.x % 32 == 0) bounds[warp + 1] = end;
            if (warp == 0) {
                auto const begin =
                    d0 == 0 ? 0 : warp_merge_split(pair.a, pair.a_len, pair.b, pair.b_len, d0);
                if (threadIdx.x == 0) bounds[0] = begin;
            }
        }
    }
    __syncthreads();
    uint32_t matches = 0;
    auto job_index = start_job;
    auto counting = start_job;
    for (auto tile = first; tile < last; ++tile) {
        uint64_t d0 = 0, d1 = 0;
        place(tile, job_index, d0, d1);
        if (job_index != counting) {
            auto const sum = reduce_type(storage).Sum(matches);
            if (threadIdx.x == 0 && sum != 0) atomicAdd(counts + counting, sum);
            matches = 0;
            counting = job_index;
        }
        auto const& job = jobs[job_index];
        auto const a0 = d0 == 0 ? 0 : bounds[tile - first];
        auto const na = static_cast<uint32_t>(bounds[tile - first + 1] - a0);
        auto const n = static_cast<uint32_t>(d1 - d0);
        auto const nb = n - na;
        auto const b0 = d0 - a0;
        // Every copy in flight at once, straight into shared memory. Key e = threadIdx.x + 256k
        // has slot threadIdx.x + threadIdx.x / 8 + 288k, so the unrolled copies differ only by
        // constants.
        auto const* const from_a = job.a + a0;
        auto const* const from_b = job.b + b0;
        auto* const slot = keys + threadIdx.x + (threadIdx.x >> 3);
        _Pragma("unroll")
        for (uint32_t k = 0; k < merge_items; ++k) {
            auto const e = threadIdx.x + k * merge_threads;
            if (e < n) {
                __pipeline_memcpy_async(
                    slot + k * (merge_threads + merge_threads / 8),
                    e < na ? from_a + e : from_b + (e - na),
                    sizeof(uint64_t)
                );
            }
        }
        if (threadIdx.x == 0) {
            if (b0 + nb < job.b_len) {
                __pipeline_memcpy_async(&at(n), from_b + nb, sizeof(uint64_t));
            } else {
                at(n) = no_key;
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        auto const begin = min(threadIdx.x * merge_items, n);
        auto const end = min(begin + merge_items, n);
        uint32_t i = begin > nb ? begin - nb : 0;
        for (uint32_t high = min(begin, na); i < high;) {
            auto const mid = i + (high - i) / 2;
            if (at(mid) <= at(na + begin - 1 - mid)) {
                i = mid + 1;
            } else {
                high = mid;
            }
        }
        auto j = begin - i;
        // Both heads stay in registers; a step reads only the next key of the side it took, from
        // a slot that always exists: slot na holds a B key or the key after the tile, and an A
        // side that has run out reads it and drops it. The B head is at most the key after the
        // tile, which no step within it takes.
        auto head_a = i < na ? at(min(i, na)) : no_key;
        auto head_b = at(na + j);
        auto const advance = [&] {
            bool const take_a = head_a <= head_b;
            matches += take_a && head_a == head_b;
            i += take_a;
            j += !take_a;
            auto const read = at(take_a ? min(i, na) : na + j);
            auto const following = take_a && i >= na ? no_key : read;
            head_a = take_a ? following : head_a;
            head_b = take_a ? head_b : following;
        };
        if (n == merge_tile) {
            // Every tile but a pair's last is full, and then every thread takes all its steps.
            _Pragma("unroll")
            for (uint32_t step = 0; step < merge_items; ++step) advance();
        } else {
            _Pragma("unroll")
            for (uint32_t step = 0; step < merge_items; ++step) {
                if (begin + step < end) advance();
            }
        }
        // The next tile overwrites the keys.
        __syncthreads();
    }
    auto const sum = reduce_type(storage).Sum(matches);
    if (threadIdx.x == 0 && sum != 0) atomicAdd(counts + counting, sum);
}

int run_main(
    std::vector<std::string> const& references,
    std::vector<std::string> const& queries,
    bool all_to_all,
    int samples,
    int warmups,
    size_t max_kmers,
    size_t max_pairs,
    size_t match_rows,
    bool sketch_only,
    unsigned parse_workers,
    size_t stash_bytes,
    size_t device_stash_bytes,
    json& report
) {
    cudaStream_t stream = nullptr;
    CUDDL_CUDA_CALL(cudaStreamCreate(&stream));
    event_accum sketch_device{stream};
    event_accum compare_device{stream};
    struct stream_guard {
        cudaStream_t stream;
        ~stream_guard() {
            CUDDL_CUDA_ABORT(cudaStreamDestroy(stream));
        }
    } guard{stream};

    std::vector<std::string> names;
    names.reserve(references.size());
    for (auto const& path : references) names.push_back(path);
    size_t const reference_count = names.size();
    for (auto const& path : queries) names.push_back(path);
    size_t const genomes = names.size();
    size_t const query_base = all_to_all ? 0 : reference_count;
    size_t const query_count = all_to_all ? genomes : genomes - reference_count;

    // Pair space enumeration needs no packed data, only indices.
    size_t const total_pairs = [&] {
        if (all_to_all) return genomes * (genomes - 1) / 2;
        return query_count * reference_count;
    }();
    size_t const pair_stride =
        max_pairs && total_pairs > max_pairs ? (total_pairs + max_pairs - 1) / max_pairs : 1;
    auto pair_at = [&](size_t ordinal) {
        if (all_to_all) {
            // Triangular index: ordinal -> (q, r) with q < r. Row q starts at q (2N - q - 1) / 2,
            // so q solves a quadratic; the loops correct the floating-point estimate.
            auto const row_start = [&](size_t row) {
                return row * (2 * genomes - row - 1) / 2;
            };
            auto const n = static_cast<double>(genomes);
            auto q = static_cast<size_t>(std::max(
                0.0, (2 * n - 1 - std::sqrt((2 * n - 1) * (2 * n - 1) - 8.0 * ordinal)) / 2
            ));
            while (q > 0 && row_start(q) > ordinal) --q;
            while (row_start(q + 1) <= ordinal) ++q;
            return std::pair{q, q + 1 + (ordinal - row_start(q))};
        }
        return std::pair{query_base + ordinal / reference_count, ordinal % reference_count};
    };
    // Sketch-only runs never evaluate a pair. Enumerating a full-corpus
    // pair space here would materialize billions of ordinals.
    std::vector<size_t> evaluated;
    if (!sketch_only) {
        for (size_t i = 0; i < total_pairs; i += pair_stride) evaluated.push_back(i);
        if (!evaluated.empty() && evaluated.back() != total_pairs - 1) {
            evaluated.push_back(total_pairs - 1);
        }
    }
    std::vector<uint32_t> pair_a(evaluated.size()), pair_r(evaluated.size());
    for (size_t i = 0; i < evaluated.size(); ++i) {
        auto const [a, r] = pair_at(evaluated[i]);
        pair_a[i] = static_cast<uint32_t>(a);
        pair_r[i] = static_cast<uint32_t>(r);
    }
    // The query side is touched by every reference and the reference side by every query, so
    // whichever of them stays resident decides how often the other is parsed. Queries come first
    // because a batch run reuses each of them across the whole reference set.
    auto const query_side = [&](size_t g) {
        return !all_to_all && g >= query_base;
    };
    auto const stash_order = [&](size_t pass, size_t g) {
        return pass == 0 ? query_side(g) : !query_side(g);
    };
    // Genomes touched by evaluated pairs keep their packed arrays across the
    // pair loop; the rest stream through for distinct counts only.
    std::vector<char> needed(genomes, 0);
    for (size_t i = 0; i < evaluated.size(); ++i) {
        needed[pair_a[i]] = 1;
        needed[pair_r[i]] = 1;
    }
    // Retaining a genome's k-mers is what makes a pair cheap, but every genome touched by an
    // evaluated pair would be kept: over a full corpus that is terabytes, which is how a run
    // sets the machine's memory alight. What a rebuild needs is the sequence, so the budget holds
    // staged bytes, about a tenth of the packed keys.
    struct stashed_sequence {
        std::vector<char> bases;
        std::vector<staged_chunk> pieces;
    };
    std::vector<stashed_sequence> stashed(genomes);
    std::vector<char> resident(genomes, 0);
    // A pair needs two sorted arrays. Sorting each genome once and keeping the sorted copy on
    // the device turns the per-pair cost from an 8-pass sort of both sets into one merge pass
    // over them, and the genomes a run reuses most are the ones it packs first.
    device_buffer sorted_store;
    std::vector<size_t> sorted_offset(genomes, 0);
    std::vector<uint32_t> sorted_count(genomes, 0);
    size_t sorted_bytes = 0;
    if (device_stash_bytes >= sizeof(uint64_t)) sorted_store.reset(device_stash_bytes);
    size_t stashed_bytes = 0;
    size_t reparsed_genomes = 0, rebuilt_genomes = 0, compare_blocks = 0;

    device_buffer unique_regions, run_counts_dev, pair_counts, num_runs_dev, temp, staged_bases,
        staged_chunks, window_keys, window_flags, chunk_valid, compacted_keys;
    size_t max_keys = 0, max_pair = 0, temp_bytes = 0;
    num_runs_dev.reset(sizeof(size_t));

    // Packs staged sequence bytes into a compacted key array: the emit pass, then the compaction
    // that drops the windows an ambiguous base broke. The sketch and a rebuilt side both go
    // through here, so a rebuild returns to bytes rather than to the host parser.
    auto stage_and_compact = [&](std::vector<char> const& bases,
                                 std::vector<staged_chunk> const& pieces,
                                 size_t windows_total,
                                 event_accum* timed) {
        staged_bases.reset(bases.size());
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            staged_bases.data, bases.data(), bases.size(), cudaMemcpyHostToDevice, stream
        ));
        staged_chunks.reset(pieces.size() * sizeof(staged_chunk));
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            staged_chunks.data,
            pieces.data(),
            pieces.size() * sizeof(staged_chunk),
            cudaMemcpyHostToDevice,
            stream
        ));
        window_keys.reset(windows_total * sizeof(uint64_t));
        window_flags.reset(windows_total);
        chunk_valid.reset(pieces.size() * sizeof(uint32_t));
        CUDDL_CUDA_CALL(
            cudaMemsetAsync(chunk_valid.data, 0, pieces.size() * sizeof(uint32_t), stream)
        );
        compacted_keys.reset(windows_total * sizeof(uint64_t));
        temp_bytes = std::max(temp_bytes, select_temp_bytes(windows_total));
        temp.reset(temp_bytes);
        size_t emit_span = std::numeric_limits<size_t>::max();
        if (timed) emit_span = timed->begin();
        emit_chunk_kmers<<<
            static_cast<uint32_t>((windows_total + emit_tile - 1) / emit_tile),
            256,
            0,
            stream>>>(
            static_cast<char const*>(staged_bases.data),
            static_cast<staged_chunk const*>(staged_chunks.data),
            static_cast<uint32_t>(pieces.size()),
            windows_total,
            static_cast<uint32_t>(kmer_length),
            static_cast<uint64_t*>(window_keys.data),
            static_cast<uint8_t*>(window_flags.data),
            static_cast<uint32_t*>(chunk_valid.data)
        );
        CUDDL_CUDA_CALL(cudaGetLastError());
        if (timed) timed->end(emit_span);
        size_t compact_span = std::numeric_limits<size_t>::max();
        if (timed) compact_span = timed->begin();
        CUDDL_CUDA_CALL(
            cub::DeviceSelect::Flagged(
                temp.data,
                temp.bytes,
                static_cast<uint64_t const*>(window_keys.data),
                static_cast<uint8_t const*>(window_flags.data),
                static_cast<uint64_t*>(compacted_keys.data),
                static_cast<int*>(num_runs_dev.data),
                static_cast<int64_t>(windows_total),
                stream
            )
        );
        if (timed) timed->end(compact_span);
    };

    std::vector<size_t> kmers_of(genomes, 0), distinct(genomes, 0);
    std::vector<double> parse_ms, sketch_ms, compare_ms, end_to_end_ms, resident_sketch_ms,
        resident_compare_ms;
    // Emitted rows stride-sample the evaluated pairs, first and last always kept. Applying the
    // stride as rows are produced keeps only what will be reported: a full corpus evaluates
    // hundreds of millions of pairs, and holding one JSON object each needs far more memory
    // than the k-mer arrays this benchmark reads.
    json emitted = json::array();
    size_t const emit_stride = match_rows && evaluated.size() > match_rows
                                   ? (evaluated.size() + match_rows - 1) / match_rows
                                   : 1;

    for (int rep = -warmups; rep < samples; ++rep) {
        sketch_device.reset_sample();
        compare_device.reset_sample();
        auto const sample_tick = clock_type::now();
        auto parse_tick = clock_type::now();
        sorted_bytes = 0;
        std::fill(sorted_count.begin(), sorted_count.end(), 0);
        // Sketch order decides who wins the stash: the reused side is packed first, so a small
        // budget keeps the side that every pair touches instead of the first genomes in the
        // corpus.
        std::vector<size_t> sketch_order;
        sketch_order.reserve(genomes);
        for (size_t pass = 0; pass < 2; ++pass) {
            for (size_t g = 0; g < genomes; ++g) {
                if (stash_order(pass, g)) sketch_order.push_back(g);
            }
        }
        // Sketching stages sequence bytes and lets the device pack the k-mers: the host parser
        // spent the whole corpus on 72 cores while the device idled. The loader decompresses in
        // parallel and hands over complete windows per piece, so what the device packs is what a
        // host parse of the same piece would have produced.
        std::vector<std::string> sketch_paths;
        sketch_paths.reserve(sketch_order.size());
        for (size_t const g : sketch_order) sketch_paths.push_back(names[g]);
        // A staged batch holds its bytes, its windows and their flags at once, so it is sized by
        // the room left once the sorted store is placed.
        // A staged byte becomes a window, and a window costs eight bytes of keys twice over (the
        // emitted and the compacted arrays) plus a flag and the compaction and sort scratch.
        // Sizing the batch by the bytes alone overcommits a device that is also holding the
        // sorted store and the resident arrays.
        size_t const staging_bytes = std::max<size_t>(
            size_t{32} << 20, std::min<size_t>(available_device_bytes() / 80, size_t{256} << 20)
        );
        std::vector<staged_chunk> chunk_host;
        std::vector<size_t> chunk_genomes;
        std::vector<uint32_t> chunk_valid_host;
        std::vector<size_t> batch_genomes, batch_offsets;
        std::vector<int> batch_runs;
        // Lays a loaded batch's pieces out for the emitter, dropping those too short for a window.
        auto const chunk_batch = [&](resident_sequence::batch const& batch) {
            chunk_host.clear();
            chunk_genomes.clear();
            size_t windows_total = 0;
            for (auto const& chunk : batch.chunks) {
                if (chunk.size < kmer_length) continue;
                auto const windows = chunk.size - kmer_length + 1;
                chunk_host.push_back({chunk.offset, windows_total, static_cast<uint32_t>(windows)});
                chunk_genomes.push_back(chunk.genome);
                windows_total += windows;
            }
            return windows_total;
        };
        resident_sequence::for_each_batch(
            sketch_paths,
            static_cast<uint32_t>(kmer_length),
            staging_bytes,
            [&](resident_sequence::batch const& batch) {
                auto const windows_total = chunk_batch(batch);
                if (chunk_host.empty() || !windows_total) return;
                stage_and_compact(batch.bases, chunk_host, windows_total, &sketch_device);
                chunk_valid_host.resize(chunk_host.size());
                CUDDL_CUDA_CALL(cudaMemcpyAsync(
                    chunk_valid_host.data(),
                    chunk_valid.data,
                    chunk_host.size() * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
                sketch_device.resolve();
                // One genome can arrive as several pieces; the loader emits them in genome order
                // and never splits a genome across batches, so its valid windows add up here and
                // its keys stay together for the sort that follows.
                batch_genomes.clear();
                batch_offsets.clear();
                size_t elements = 0;
                for (size_t c = 0; c < chunk_host.size();) {
                    auto const piece = c;
                    auto const g = chunk_genomes[c];
                    size_t count = 0;
                    while (c < chunk_host.size() && chunk_genomes[c] == g) {
                        count += chunk_valid_host[c];
                        ++c;
                    }
                    batch_genomes.push_back(g);
                    batch_offsets.push_back(elements);
                    elements += count;
                    auto const name = sketch_order[g];
                    kmers_of[name] = count;
                    max_keys = std::max(max_keys, count);
                    if (count > max_kmers) {
                        throw std::runtime_error(
                            "genome exceeds --max-kmers, refusing: " + names[name]
                        );
                    }
                    // Held for the pair loop: a genome's pieces sit next to each other in
                    // the staged batch, so one copy takes its sequence and rebases the pieces
                    // onto it.
                    if (!needed[name]) continue;
                    auto const last = c - 1;
                    auto const begin = chunk_host[piece].offset;
                    auto const end =
                        chunk_host[last].offset + chunk_host[last].windows + kmer_length - 1;
                    if (stashed_bytes + (end - begin) > stash_bytes) continue;
                    auto& held = stashed[name];
                    held.bases.assign(
                        batch.bases.begin() + static_cast<ptrdiff_t>(begin),
                        batch.bases.begin() + static_cast<ptrdiff_t>(end)
                    );
                    held.pieces.clear();
                    for (size_t p = piece; p <= last; ++p) {
                        held.pieces.push_back({
                            chunk_host[p].offset - begin,
                            chunk_host[p].window_base - chunk_host[piece].window_base,
                            chunk_host[p].windows,
                        });
                    }
                    stashed_bytes += end - begin;
                    resident[name] = 1;
                }
                auto const segments = batch_genomes.size();
                unique_regions.reset(elements * sizeof(uint64_t));
                run_counts_dev.reset(segments * sizeof(int));
                // A genome with no k-mers is never encoded, so its slot has to start at zero
                // rather than at whatever the allocation held.
                CUDDL_CUDA_CALL(
                    cudaMemsetAsync(run_counts_dev.data, 0, segments * sizeof(int), stream)
                );
                // The encode writes one run length per k-mer, so the scratch is sized by the
                // largest genome seen.
                pair_counts.reset(max_keys * sizeof(int));
                temp_bytes = std::max(
                    temp_bytes, std::max(sort_temp_bytes(max_keys), encode_temp_bytes(max_keys))
                );
                temp.reset(temp_bytes);
                for (size_t i = 0; i < segments; ++i) {
                    auto const count =
                        (i + 1 < segments ? batch_offsets[i + 1] : elements) - batch_offsets[i];
                    if (!count) continue;
                    auto* keys = static_cast<uint64_t*>(compacted_keys.data) + batch_offsets[i];
                    auto const sort_span = sketch_device.begin();
                    CUDDL_CUDA_CALL(
                        cub::DeviceRadixSort::SortKeys(
                            temp.data, temp.bytes, keys, keys, count, 0, 64, stream
                        )
                    );
                    sketch_device.end(sort_span);
                    auto const encode_span = sketch_device.begin();
                    CUDDL_CUDA_CALL(
                        cub::DeviceRunLengthEncode::Encode(
                            temp.data,
                            temp.bytes,
                            keys,
                            static_cast<uint64_t*>(unique_regions.data) + batch_offsets[i],
                            static_cast<int*>(pair_counts.data),
                            static_cast<int*>(run_counts_dev.data) + i,
                            count,
                            stream
                        )
                    );
                    sketch_device.end(encode_span);
                }
                batch_runs.assign(segments, 0);
                CUDDL_CUDA_CALL(cudaMemcpyAsync(
                    batch_runs.data(),
                    run_counts_dev.data,
                    segments * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
                sketch_device.resolve();
                for (size_t i = 0; i < segments; ++i) {
                    distinct[sketch_order[batch_genomes[i]]] = static_cast<size_t>(batch_runs[i]);
                }
                // The store keeps what device memory has room for.
                for (size_t i = 0; i < segments; ++i) {
                    auto const g = sketch_order[batch_genomes[i]];
                    auto const runs = static_cast<size_t>(batch_runs[i]);
                    auto const bytes = runs * sizeof(uint64_t);
                    if (runs && needed[g] && sorted_store.bytes &&
                        sorted_bytes + bytes <= sorted_store.bytes) {
                        sorted_offset[g] = sorted_bytes / sizeof(uint64_t);
                        sorted_count[g] = static_cast<uint32_t>(runs);
                        CUDDL_CUDA_CALL(cudaMemcpyAsync(
                            static_cast<uint64_t*>(sorted_store.data) + sorted_offset[g],
                            static_cast<uint64_t const*>(unique_regions.data) + batch_offsets[i],
                            bytes,
                            cudaMemcpyDeviceToDevice,
                            stream
                        ));
                        sorted_bytes += bytes;
                    }
                }
            },
            parse_workers,
            /*whole_genomes=*/true
        );
        // Pair buffers sized from the largest evaluated pair actually measured.
        for (size_t i = 0; i < evaluated.size(); ++i) {
            max_pair = std::max(max_pair, kmers_of[pair_a[i]] + kmers_of[pair_r[i]]);
        }
        // A rebuilt side sorts one genome and run-length encodes it, so what it needs is sized by
        // the largest genome rather than by the largest pair.
        pair_counts.reset(max_keys * sizeof(int));
        temp_bytes = std::max({temp_bytes, sort_temp_bytes(max_keys), encode_temp_bytes(max_keys)});
        temp.reset(temp_bytes);
        auto const parse_done = clock_type::now();
        auto const compare_tick = clock_type::now();
        // Pairs are counted in batches: one launch and one host read for a whole batch, instead of
        // a merge, an encode and a pair-sized copy for each pair. A batched job points into the
        // stash or a scratch, so the batch is flushed as soon as a scratch is rebuilt.
        struct job_row {
            size_t a;
            size_t r;
            char emit;
        };
        std::vector<pair_job> jobs;
        std::vector<job_row> job_rows;
        device_buffer jobs_dev, counts_dev, tile_first_dev;
        std::vector<uint32_t> intersections;
        // First merge tile of each job, then the total: a job's tiles cover its merged keys.
        std::vector<uint64_t> tile_first;
        auto flush_jobs = [&]() {
            if (jobs.empty()) return;
            tile_first.assign(1, 0);
            for (auto const& job : jobs) {
                auto const merged = uint64_t{job.a_len} + job.b_len;
                auto const tiles =
                    job.a_len && job.b_len ? (merged + merge_tile - 1) / merge_tile : 0;
                tile_first.push_back(tile_first.back() + tiles);
            }
            jobs_dev.reset(jobs.size() * sizeof(pair_job));
            counts_dev.reset(jobs.size() * sizeof(uint32_t));
            tile_first_dev.reset(tile_first.size() * sizeof(uint64_t));
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                jobs_dev.data,
                jobs.data(),
                jobs.size() * sizeof(pair_job),
                cudaMemcpyHostToDevice,
                stream
            ));
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                tile_first_dev.data,
                tile_first.data(),
                tile_first.size() * sizeof(uint64_t),
                cudaMemcpyHostToDevice,
                stream
            ));
            CUDDL_CUDA_CALL(
                cudaMemsetAsync(counts_dev.data, 0, jobs.size() * sizeof(uint32_t), stream)
            );
            auto const kernel_span = compare_device.begin();
            // One block per merge_chunk tiles, in launches within the grid's x limit.
            auto const total_blocks = (tile_first.back() + merge_chunk - 1) / merge_chunk;
            constexpr uint64_t max_blocks = uint64_t{1} << 30;
            for (uint64_t first = 0; first < total_blocks; first += max_blocks) {
                auto const blocks =
                    static_cast<uint32_t>(std::min(max_blocks, total_blocks - first));
                merge_intersection_kernel<<<blocks, merge_threads, 0, stream>>>(
                    static_cast<pair_job const*>(jobs_dev.data),
                    static_cast<uint64_t const*>(tile_first_dev.data),
                    static_cast<uint32_t>(jobs.size()),
                    first,
                    static_cast<uint32_t*>(counts_dev.data)
                );
                CUDDL_CUDA_CALL(cudaGetLastError());
            }
            compare_device.end(kernel_span);
            intersections.resize(jobs.size());
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                intersections.data(),
                counts_dev.data,
                jobs.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream
            ));
            CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
            compare_device.resolve();
            for (size_t i = 0; i < jobs.size(); ++i) {
                if (!job_rows[i].emit) continue;
                auto const qa = job_rows[i].a, rb = job_rows[i].r;
                auto const shared = intersections[i];
                auto const pair_union = distinct[qa] + distinct[rb] - shared;
                auto const jaccard = pair_union ? static_cast<double>(shared) / pair_union : 0.0;
                auto mash_ani = 0.0;
                if (jaccard > 0 && jaccard <= 1) {
                    mash_ani = (1.0 + std::log(2 * jaccard / (1 + jaccard)) / 25.0) * 100.0;
                }
                emitted.push_back(
                    {{"query", names[qa]},
                     {"reference", names[rb]},
                     {"distinct_a", distinct[qa]},
                     {"distinct_b", distinct[rb]},
                     {"intersection", shared},
                     {"union", pair_union},
                     {"jaccard", jaccard},
                     {"containment_a_in_b",
                      distinct[qa] ? static_cast<double>(shared) / distinct[qa] : 0.0},
                     {"containment_b_in_a",
                      distinct[rb] ? static_cast<double>(shared) / distinct[rb] : 0.0},
                     {"mash_ani", mash_ani}}
                );
            }
            jobs.clear();
            job_rows.clear();
        };
        emitted = json::array();
        // Pairs read sorted, deduplicated k-mers from the store. When the sketch pass fitted every
        // genome a pair touches, they stay where it put them. Otherwise the store is split in two
        // regions and genomes are taken in blocks that fit one: block I is built into the first,
        // each block J >= I into the second, and every pair between them runs as one batch. A
        // genome is then rebuilt about once per block rather than once per pair, so the merges
        // rather than the rebuilds set the time.
        bool all_stored = true;
        for (size_t g = 0; g < genomes; ++g) {
            if (needed[g] && distinct[g] && !sorted_count[g]) all_stored = false;
        }
        size_t const region_words = sorted_store.bytes / 2 / sizeof(uint64_t);
        std::vector<uint32_t> block_of(genomes, 0);
        std::vector<std::vector<size_t>> blocks(1);
        size_t block_words = 0;
        for (size_t g = 0; g < genomes; ++g) {
            if (!needed[g]) continue;
            if (!all_stored) {
                if (distinct[g] > region_words) {
                    throw std::runtime_error(
                        "a genome's sorted k-mers exceed half the device store, raise "
                        "--device-stash-mb: " +
                        names[g]
                    );
                }
                if (block_words + distinct[g] > region_words) {
                    blocks.emplace_back();
                    block_words = 0;
                }
                block_words += distinct[g];
            }
            block_of[g] = static_cast<uint32_t>(blocks.size() - 1);
            blocks.back().push_back(g);
        }
        compare_blocks = blocks.size();
        // Pairs grouped by their unordered block pair, each group in evaluation order.
        auto const bucket_of = [&](size_t i) {
            auto lo = block_of[pair_a[i]], hi = block_of[pair_r[i]];
            if (lo > hi) std::swap(lo, hi);
            return static_cast<size_t>(lo) * blocks.size() + hi;
        };
        std::vector<size_t> bucket_start(blocks.size() * blocks.size() + 1, 0);
        for (size_t i = 0; i < evaluated.size(); ++i) ++bucket_start[bucket_of(i) + 1];
        for (size_t b = 1; b < bucket_start.size(); ++b) bucket_start[b] += bucket_start[b - 1];
        std::vector<size_t> order(evaluated.size());
        {
            auto next = bucket_start;
            for (size_t i = 0; i < evaluated.size(); ++i) order[next[bucket_of(i)]++] = i;
        }
        auto* const store = static_cast<uint64_t*>(sorted_store.data);
        std::vector<uint64_t const*> sorted_at(genomes, nullptr);
        if (all_stored) {
            for (size_t g = 0; g < genomes; ++g) sorted_at[g] = store + sorted_offset[g];
        }
        // Sorts and deduplicates a block's genomes into a region, from their held sequence or,
        // outside the host budget, from the files again through the sketch's parallel loader.
        auto build_block = [&](size_t block, size_t region) {
            // Pending jobs read the region this overwrites.
            flush_jobs();
            auto* out = store + region * region_words;
            auto const sort_into_region = [&](size_t g, uint64_t* keys) {
                sorted_at[g] = out;
                CUDDL_CUDA_CALL(
                    cub::DeviceRadixSort::SortKeys(
                        temp.data, temp.bytes, keys, keys, kmers_of[g], 0, 64, stream
                    )
                );
                CUDDL_CUDA_CALL(
                    cub::DeviceRunLengthEncode::Encode(
                        temp.data,
                        temp.bytes,
                        keys,
                        out,
                        static_cast<int*>(pair_counts.data),
                        static_cast<int*>(num_runs_dev.data),
                        kmers_of[g],
                        stream
                    )
                );
                out += distinct[g];
                ++rebuilt_genomes;
            };
            // A packed batch's valid windows sit in the compacted keys genome after genome, as many
            // as the sketch counted for each.
            auto const sort_batch = [&](std::vector<size_t> const& members) {
                auto* keys = static_cast<uint64_t*>(compacted_keys.data);
                for (size_t const g : members) {
                    sort_into_region(g, keys);
                    keys += kmers_of[g];
                }
            };
            // Held genomes are packed in batches like the sketch's, so a copy and an emit cover
            // many genomes rather than one: a pageable copy first waits for the work before it.
            std::vector<char> held_bases;
            std::vector<staged_chunk> held_pieces;
            std::vector<size_t> held_genomes, reload_genomes;
            std::vector<std::string> reload_paths;
            size_t held_windows = 0;
            auto const flush_held = [&] {
                if (held_genomes.empty()) return;
                stage_and_compact(held_bases, held_pieces, held_windows, nullptr);
                sort_batch(held_genomes);
                held_bases.clear();
                held_pieces.clear();
                held_genomes.clear();
                held_windows = 0;
            };
            for (size_t const g : blocks[block]) {
                if (!distinct[g]) {
                    sorted_at[g] = out;
                    continue;
                }
                if (!resident[g]) {
                    reload_paths.push_back(names[g]);
                    reload_genomes.push_back(g);
                    continue;
                }
                auto const& held = stashed[g];
                if (!held_bases.empty() && held_bases.size() + held.bases.size() > staging_bytes) {
                    flush_held();
                }
                auto const window_base = held_windows;
                for (auto const& piece : held.pieces) {
                    held_pieces.push_back({
                        held_bases.size() + piece.offset,
                        window_base + piece.window_base,
                        piece.windows,
                    });
                    held_windows += piece.windows;
                }
                held_bases.insert(held_bases.end(), held.bases.begin(), held.bases.end());
                held_genomes.push_back(g);
            }
            flush_held();
            if (reload_paths.empty()) return;
            reparsed_genomes += reload_paths.size();
            std::vector<size_t> members;
            resident_sequence::for_each_batch(
                reload_paths,
                static_cast<uint32_t>(kmer_length),
                staging_bytes,
                [&](resident_sequence::batch const& batch) {
                    auto const windows_total = chunk_batch(batch);
                    if (chunk_host.empty() || !windows_total) return;
                    members.clear();
                    for (size_t c = 0; c < chunk_genomes.size(); ++c) {
                        if (c == 0 || chunk_genomes[c] != chunk_genomes[c - 1]) {
                            members.push_back(reload_genomes[chunk_genomes[c]]);
                        }
                    }
                    stage_and_compact(batch.bases, chunk_host, windows_total, nullptr);
                    sort_batch(members);
                },
                parse_workers,
                /*whole_genomes=*/true
            );
        };
        for (size_t lo = 0; lo < blocks.size(); ++lo) {
            bool built = all_stored;
            for (size_t hi = lo; hi < blocks.size(); ++hi) {
                auto const first = bucket_start[lo * blocks.size() + hi];
                auto const last = bucket_start[lo * blocks.size() + hi + 1];
                if (first == last) continue;
                if (!built) {
                    build_block(lo, 0);
                    built = true;
                }
                if (!all_stored && hi != lo) build_block(hi, 1);
                for (auto k = first; k < last; ++k) {
                    auto const i = order[k];
                    auto const a = pair_a[i], r = pair_r[i];
                    // Both sides are sorted and deduplicated, so a pair is one linear pass over
                    // them and the union follows from distinct[a] + distinct[r] - shared. An
                    // empty side shares nothing, which the kernel handles without reading.
                    bool const both = distinct[a] && distinct[r];
                    jobs.push_back({
                        both ? sorted_at[a] : nullptr,
                        static_cast<uint32_t>(both ? distinct[a] : 0),
                        both ? sorted_at[r] : nullptr,
                        static_cast<uint32_t>(both ? distinct[r] : 0),
                    });
                    job_rows.push_back({
                        a,
                        r,
                        static_cast<char>(i % emit_stride == 0 || i + 1 == evaluated.size()),
                    });
                }
                flush_jobs();
            }
        }
        auto const done = clock_type::now();
        if (rep >= 0) {
            using ms = std::chrono::duration<double, std::milli>;
            parse_ms.push_back(ms(parse_done - sample_tick).count());
            sketch_ms.push_back(ms(compare_tick - sample_tick).count());
            compare_ms.push_back(ms(done - compare_tick).count());
            end_to_end_ms.push_back(ms(done - sample_tick).count());
            resident_sketch_ms.push_back(sketch_device.ms);
            resident_compare_ms.push_back(compare_device.ms);
        }
    }

    json genome_rows = json::array();
    for (size_t g = 0; g < genomes; ++g) {
        genome_rows.push_back(
            {{"path", names[g]}, {"kmers", kmers_of[g]}, {"distinct", distinct[g]}}
        );
    }
    auto summarize = [](std::vector<double> const& values) {
        return json{
            {"samples", values.size()},
            {"median_ms", median_of(values)},
            {"min_ms", *std::min_element(values.begin(), values.end())},
            {"max_ms", *std::max_element(values.begin(), values.end())},
            {"source", "steady_clock_cpu_wall"},
        };
    };
    // Bounded per-batch device spans are summed per replicate, so the resident summary is a
    // measured total, never wall time or a difference of medians.
    auto summarize_device = [](std::vector<double> const& values) {
        return json{
            {"samples", values.size()},
            {"median_ms", median_of(values)},
            {"min_ms", *std::min_element(values.begin(), values.end())},
            {"max_ms", *std::max_element(values.begin(), values.end())},
            {"source", "cuda_events"},
        };
    };
    report = {
        {"implementation", {{"name", "cub-exact"}}},
        {"case",
         {{"k", 25},
          {"topology", all_to_all ? "all-to-all" : "batch"},
          {"references", reference_count},
          {"queries", query_count},
          {"samples", samples},
          {"warmups", warmups},
          {"pairs_total", total_pairs},
          {"pairs_evaluated", evaluated.size()},
          {"parse_workers", parse_workers},
          {"stash_mb_allowed", stash_bytes >> 20},
          {"device_stash_mb", device_stash_bytes >> 20},
          {"stashed_genomes", static_cast<size_t>(std::count(resident.begin(), resident.end(), 1))},
          {"stashed_mb", stashed_bytes >> 20},
          {"reparsed_genomes", reparsed_genomes},
          {"compare_blocks", compare_blocks},
          {"rebuilt_genomes", rebuilt_genomes},
          {"pair_stride", pair_stride},
          {"pairs_emitted", emitted.size()}}},
        {"genomes", genome_rows},
        {"phases_ms",
         {{"parse_wall", summarize(parse_ms)},
          {"device_buffers",
           {{"genome_keys", max_keys * sizeof(uint64_t)},
            {"pair_keys", max_pair * sizeof(uint64_t)},
            {"pair_runs", max_pair * (sizeof(uint64_t) + sizeof(int))},
            {"temp", temp.bytes}}},
          {"sketch", summarize(sketch_ms)},
          {"resident_sketch", summarize_device(resident_sketch_ms)},
          {"compare", summarize(compare_ms)},
          {"resident_compare", summarize_device(resident_compare_ms)},
          {"end_to_end", summarize(end_to_end_ms)}}},
        {"pairs", emitted},
    };
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    std::vector<std::string> references, queries;
    std::string topology = "batch", output;
    int samples = 5, warmups = 1;
    size_t max_kmers = 32ULL << 20, max_pairs = 0, match_rows = 0;
    unsigned workers = 0;
    size_t stash_mb = 0, device_stash_mb = 0;
    CLI::App app{"Exact k-mer set baseline from CUB primitives (k=25)"};
    app.add_option("--reference", references)->required()->check(CLI::ExistingFile);
    app.add_option("--query", queries)->check(CLI::ExistingFile);
    app.add_option("--topology", topology)->check(CLI::IsMember({"batch", "all-to-all"}));
    app.add_option("--samples", samples)->check(CLI::Range(1, 1000));
    app.add_option("--warmups", warmups)->check(CLI::Range(0, 100));
    app.add_option("--max-kmers", max_kmers);
    app.add_option("--max-pairs", max_pairs, "Evaluated pairs cap, even stride (0 disables)");
    app.add_option("--match-rows", match_rows, "Emitted pair rows cap, even stride (0 disables)");
    app.add_option(
        "--workers",
        workers,
        "Concurrent genome parsers ahead of the GPU loop; 0 uses the hardware thread count"
    );
    app.add_option(
        "--stash-mb",
        stash_mb,
        "Host memory for resident k-mer arrays; 0 uses half of MemAvailable. Arrays outside the "
        "budget are packed again per pair instead of being retained."
    );
    app.add_option(
        "--device-stash-mb",
        device_stash_mb,
        "Device memory for sorted k-mer arrays; 0 uses half of what is free. A pair merges two "
        "sorted arrays instead of sorting both again, which is what that memory buys."
    );
    app.add_option("--output", output)->required();
    app.set_config("--config", "TOML file with options, e.g. reference = [...]");
    bool sketch_only = false;
    app.add_flag("--sketch-only", sketch_only, "Parse plus device sketch only, no pairs");
    CLI11_PARSE(app, argc, argv);
    if (topology == "batch" && (references.empty() || queries.empty())) {
        throw std::runtime_error("batch needs nonempty --reference and --query");
    }
    if (topology == "all-to-all" && references.size() < 2) {
        throw std::runtime_error("all-to-all needs at least two --reference files");
    }
    json report;
    run_main(
        references,
        queries,
        topology == "all-to-all",
        samples,
        warmups,
        max_kmers,
        max_pairs,
        match_rows,
        sketch_only,
        workers == 0 ? parse_worker_count(references.size() + queries.size()) : workers,
        stash_mb ? stash_mb << 20 : std::max<size_t>(available_host_bytes() / 2, size_t{1} << 30),
        device_stash_mb ? device_stash_mb << 20 : available_device_bytes() / 2,
        report
    );
    FILE* stream = std::fopen(output.c_str(), "w");
    if (!stream) throw std::runtime_error("cannot write " + output);
    auto const text = report.dump();
    if (std::fwrite(text.data(), 1, text.size(), stream) != text.size() ||
        std::fclose(stream) != 0) {
        throw std::runtime_error("cannot write " + output);
    }
    std::cout << "wrote " << output << " with " << report["pairs"].size() << " pairs\n";
    return 0;
} catch (std::exception const& error) {
    std::cerr << "cub-exact-pairwise: " << error.what() << '\n';
    return 1;
}
