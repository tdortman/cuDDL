// cub-exact-pairwise: exact k-mer set baseline from CUB device primitives.
//
// Parses FASTA files with the cuDDL parser (k=25 canonical packed k-mers). Genomes touched by
// evaluated pairs keep their packed arrays; the rest stream through for distinct counts only, so
// host memory stays bounded by the evaluated set. Sketching queues a batch of genomes' sorts and
// encodes behind each other and reads the whole batch's run counts back at once, so no genome
// waits on a host round trip. Per evaluated pair: one block counts the shared k-mers of two
// sorted, deduplicated arrays, so shared = |A| + |B| - union. The only downloads are the
// per-batch counts and one integer per pair batch; pair metrics are host math.
//
// --max-pairs stride-samples the evaluated pair space (first and last pair
// always measured) and --match-rows stride-samples the emitted rows the same
// way; 0 disables either cap. --sketch-only skips pair intersection for
// full-corpus parse timing; pair metrics come from the subset run instead.
// matching the CLI tools, with the parse subtotal reported separately.
// Allocation and temp-storage sizing stay outside timing; the report carries
// buffer sizes instead.
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

/// @brief Emits the canonical packed k-mers of every staged piece, one block per piece.
///
/// Packing k-mers on the host spent the corpus on 72 cores while the device idled. A window is
/// one k-mer unless an ambiguous base breaks it, which emits nothing and clears its flag for the
/// compaction to drop. A genome's pieces stay in order, so its keys stay together for the sort.
__global__ void emit_chunk_kmers(
    char const* bases,
    staged_chunk const* chunks,
    uint32_t k,
    uint64_t* keys,
    uint8_t* flags,
    uint32_t* valid
) {
    auto const chunk = chunks[blockIdx.x];
    char const* const sequence = bases + chunk.offset;
    uint32_t mine = 0;
    for (uint32_t i = threadIdx.x; i < chunk.windows; i += blockDim.x) {
        uint64_t forward = 0;
        bool complete = true;
        for (uint32_t j = 0; j < k; ++j) {
            auto const symbol = cuddl::detail::encode_base(sequence[i + j]);
            complete = complete && symbol != 0xFFU;
            forward = (forward << 2U) | (symbol & 3U);
        }
        auto const reverse = cuddl::detail::reverse_complement(forward, k);
        keys[chunk.window_base + i] = forward > reverse ? forward : reverse;
        flags[chunk.window_base + i] = complete ? 1U : 0U;
        mine += complete ? 1U : 0U;
    }
    using reduce_type = cub::BlockReduce<uint32_t, 256>;
    __shared__ typename reduce_type::TempStorage storage;
    auto const total = reduce_type(storage).Sum(mine);
    if (threadIdx.x == 0) valid[blockIdx.x] = total;
}

/// @brief One pair of sorted, deduplicated k-mer arrays to count the intersection of.
struct pair_job {
    uint64_t const* a;
    uint32_t a_len;
    uint64_t const* b;
    uint32_t b_len;
};

/// @brief Four keys of the sixteen-byte-aligned block that holds index @p pos.
///
/// An eight-byte key still costs a whole sector, so a four-key register block taken with one
/// aligned vector load is what keeps the walk off the load-request limit. @p keys is the start of
/// the array, @p pos the index to cover, and [@p lo, @p hi) the range the caller owns: a block
/// that reaches outside it, or past the end of the array, falls back to a scalar window. Keys
/// past the end repeat the last one, so the compare chain reads nothing it should not.
struct key_block {
    uint64_t keys[4];
    size_t base;
    size_t count;
};

__device__ __forceinline__ key_block
load_block(uint64_t const* keys, size_t pos, size_t lo, size_t hi) {
    auto const pad = ((16 - (reinterpret_cast<uintptr_t>(keys) & 15)) & 15) / sizeof(uint64_t);
    key_block block;
    if (pos >= pad) {
        auto const aligned = pad + ((pos - pad) / 4) * 4;
        if (aligned >= lo && aligned + 4 <= hi) {
            auto const value = *reinterpret_cast<ulonglong4 const*>(keys + aligned);
            block.keys[0] = value.x;
            block.keys[1] = value.y;
            block.keys[2] = value.z;
            block.keys[3] = value.w;
            block.base = aligned;
            block.count = 4;
            return block;
        }
    }
    block.base = pos;
    block.count = min(static_cast<size_t>(4), hi - pos);
    _Pragma("unroll")
    for (size_t i = 0; i < 4; ++i) {
        block.keys[i] = i < block.count ? keys[pos + i] : keys[hi - 1];
    }
    return block;
}

/// @brief Counts the k-mers two sorted, deduplicated sets share, one block per pair.
///
/// Both sides arrive sorted and deduplicated, so the intersection is one linear pass over them:
/// each thread takes a slice of one side and the range of the other covering it, found by two
/// binary searches, and the two walk together. Each side is taken four keys at a time, so a
/// thread has eight loads in flight instead of a chain of two, and an aligned four-key load costs
/// one request where four scalar loads cost four. Nothing is written but one integer per pair, and
/// a whole batch of pairs goes in one launch.
__global__ void exact_intersection_kernel(pair_job const* jobs, uint32_t* counts) {
    auto const job = jobs[blockIdx.x];
    if (job.a_len == 0 || job.b_len == 0) {
        if (threadIdx.x == 0) counts[blockIdx.x] = 0;
        return;
    }
    auto const bound = [&](uint64_t value, bool upper) {
        uint32_t lo = 0, hi = job.b_len;
        while (lo < hi) {
            auto const mid = lo + (hi - lo) / 2;
            if (job.b[mid] < value || (upper && job.b[mid] == value)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    };
    auto const per = (static_cast<size_t>(job.a_len) + blockDim.x - 1) / blockDim.x;
    auto const a_begin =
        min(static_cast<size_t>(threadIdx.x) * per, static_cast<size_t>(job.a_len));
    auto const a_end = min(a_begin + per, static_cast<size_t>(job.a_len));
    size_t matches = 0;
    if (a_begin < a_end) {
        size_t mine = a_begin;
        size_t theirs = bound(job.a[a_begin], false);
        auto const theirs_begin = theirs;
        auto const theirs_end = bound(job.a[a_end - 1], true);
        while (mine < a_end && theirs < theirs_end) {
            auto const a_block = load_block(job.a, mine, a_begin, a_end);
            auto const b_block = load_block(job.b, theirs, theirs_begin, theirs_end);
            // Both tiles sit in registers, so every key pair is one static comparison with no load
            // in the chain and no branch to diverge on. A tile may start before the walk position
            // when the walk is not block-aligned, and the keys it holds back are already counted,
            // so only the ones the walk has not passed are compared.
            _Pragma("unroll")
            for (size_t i = 0; i < 4; ++i) {
                _Pragma("unroll")
                for (size_t j = 0; j < 4; ++j) {
                    bool const live = i < a_block.count && j < b_block.count &&
                                      a_block.base + i >= mine && b_block.base + j >= theirs;
                    matches += static_cast<size_t>(live && a_block.keys[i] == b_block.keys[j]);
                }
            }
            // A tile's keys are all accounted for once the walk passes them, and a key is only
            // passed when the other side can no longer produce one that equals it. The step stops
            // at the end of the window: an aligned window can end before the walk's own block does,
            // and stepping past it would pass keys no window ever held.
            auto const a_step = min(a_block.base + a_block.count - mine, static_cast<size_t>(4));
            auto const b_step = min(b_block.base + b_block.count - theirs, static_cast<size_t>(4));
            // A short window repeats its last key, so index 3 is that key either way.
            if (a_block.keys[3] < b_block.keys[3]) {
                mine += a_step;
            } else if (b_block.keys[3] < a_block.keys[3]) {
                theirs += b_step;
            } else {
                mine += a_step;
                theirs += b_step;
            }
        }
    }
    using reduce_type = cub::BlockReduce<size_t, 256>;
    __shared__ typename reduce_type::TempStorage storage;
    auto const total = reduce_type(storage).Sum(matches);
    if (threadIdx.x == 0) counts[blockIdx.x] = static_cast<uint32_t>(total);
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

    // A rebuilt side re-parses a single genome, so it parses on one thread: the default would
    // spin up the machine's whole thread count for a file the device then waits on anyway.
    auto parse_one = [&](size_t g) {
        auto parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(names[g], kmer_length, 1));
        if (parsed.kmers.size() > max_kmers) {
            throw std::runtime_error("genome exceeds --max-kmers, refusing: " + names[g]);
        }
        return parsed.kmers;
    };
    // Pair space enumeration needs no packed data, only indices.
    size_t const total_pairs = [&] {
        if (all_to_all) return genomes * (genomes - 1) / 2;
        return query_count * reference_count;
    }();
    size_t const pair_stride =
        max_pairs && total_pairs > max_pairs ? (total_pairs + max_pairs - 1) / max_pairs : 1;
    auto pair_at = [&](size_t ordinal) {
        if (all_to_all) {
            // Triangular index: ordinal -> (q, r) with q < r.
            size_t q = 0, lo = 0;
            while (lo + (genomes - 1 - q) <= ordinal) {
                lo += genomes - 1 - q;
                ++q;
            }
            return std::pair{q, q + 1 + (ordinal - lo)};
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
    // The query side is touched by every reference and the reference side by every query, so
    // whichever of them stays resident decides how often the other is parsed. Queries come first
    // because a batch run reuses each of them across the whole reference set.
    auto const query_side = [&](size_t g) {
        return !all_to_all && g >= query_base;
    };
    auto const stash_order = [&](size_t pass, size_t g) {
        return pass == 0 ? query_side(g) : !query_side(g);
    };
    // Pairs are processed reference-major: one group per reference, every query against it. With
    // the queries resident that is one parse per reference instead of one per pair.
    struct scheduled_pair {
        size_t ordinal;
        size_t evaluated_index;
    };
    std::vector<scheduled_pair> schedule;
    schedule.reserve(evaluated.size());
    for (size_t i = 0; i < evaluated.size(); ++i) {
        schedule.push_back({evaluated[i], i});
    }
    std::stable_sort(schedule.begin(), schedule.end(), [&](auto const& left, auto const& right) {
        return pair_at(left.ordinal).second < pair_at(right.ordinal).second;
    });
    // Genomes touched by evaluated pairs keep their packed arrays across the
    // pair loop; the rest stream through for distinct counts only.
    std::vector<char> needed(genomes, 0);
    for (size_t ordinal : evaluated) {
        auto const [qa, rb] = pair_at(ordinal);
        needed[qa] = 1;
        needed[rb] = 1;
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
    size_t reparsed_genomes = 0;

    device_buffer unique_regions, run_counts_dev, pair_counts, num_runs_dev, temp, staged_bases,
        staged_chunks, window_keys, window_flags, chunk_valid, compacted_keys;
    size_t max_keys = 0, max_pair = 0, temp_bytes = 0;
    num_runs_dev.reset(sizeof(size_t));

    // Packs staged sequence bytes into a compacted key array: the emit pass, then the compaction
    // that drops the windows an ambiguous base broke. The sketch and a rebuilt side both go
    // through here, so a rebuild returns to bytes rather than to the host parser.
    auto stage_and_compact = [&](std::vector<char> const& bases,
                                 std::vector<staged_chunk> const& pieces,
                                 size_t windows_total) {
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
        compacted_keys.reset(windows_total * sizeof(uint64_t));
        temp_bytes = std::max(temp_bytes, select_temp_bytes(windows_total));
        temp.reset(temp_bytes);
        emit_chunk_kmers<<<static_cast<uint32_t>(pieces.size()), 256, 0, stream>>>(
            static_cast<char const*>(staged_bases.data),
            static_cast<staged_chunk const*>(staged_chunks.data),
            static_cast<uint32_t>(kmer_length),
            static_cast<uint64_t*>(window_keys.data),
            static_cast<uint8_t*>(window_flags.data),
            static_cast<uint32_t*>(chunk_valid.data)
        );
        CUDDL_CUDA_CALL(cudaGetLastError());
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
    };

    std::vector<size_t> kmers_of(genomes, 0), distinct(genomes, 0);
    std::vector<double> parse_ms, sketch_ms, compare_ms, end_to_end_ms;
    // Emitted rows stride-sample the evaluated pairs, first and last always kept. Applying the
    // stride as rows are produced keeps only what will be reported: a full corpus evaluates
    // hundreds of millions of pairs, and holding one JSON object each needs far more memory
    // than the k-mer arrays this benchmark reads.
    json emitted = json::array();
    size_t const emit_stride = match_rows && evaluated.size() > match_rows
                                   ? (evaluated.size() + match_rows - 1) / match_rows
                                   : 1;

    for (int rep = -warmups; rep < samples; ++rep) {
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
        resident_sequence::for_each_batch(
            sketch_paths,
            static_cast<uint32_t>(kmer_length),
            staging_bytes,
            [&](resident_sequence::batch const& batch) {
                chunk_host.clear();
                chunk_genomes.clear();
                size_t windows_total = 0;
                for (auto const& chunk : batch.chunks) {
                    if (chunk.size < kmer_length) continue;
                    auto const windows = chunk.size - kmer_length + 1;
                    chunk_host.push_back(
                        {chunk.offset, windows_total, static_cast<uint32_t>(windows)}
                    );
                    chunk_genomes.push_back(chunk.genome);
                    windows_total += windows;
                }
                if (chunk_host.empty() || !windows_total) return;
                stage_and_compact(batch.bases, chunk_host, windows_total);
                chunk_valid_host.resize(chunk_host.size());
                CUDDL_CUDA_CALL(cudaMemcpyAsync(
                    chunk_valid_host.data(),
                    chunk_valid.data,
                    chunk_host.size() * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
                // One genome can arrive as several pieces; the loader emits them in genome order,
                // so their valid windows add up in order and each genome's keys stay together for
                // the sort that follows.
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
                    CUDDL_CUDA_CALL(
                        cub::DeviceRadixSort::SortKeys(
                            temp.data, temp.bytes, keys, keys, count, 0, 64, stream
                        )
                    );
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
            parse_workers
        );
        // Pair buffers sized from the largest evaluated pair actually measured.
        for (size_t ordinal : evaluated) {
            auto const [qa, rb] = pair_at(ordinal);
            max_pair = std::max(max_pair, kmers_of[qa] + kmers_of[rb]);
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
        device_buffer jobs_dev, counts_dev;
        std::vector<uint32_t> intersections;
        auto flush_jobs = [&]() {
            if (jobs.empty()) return;
            jobs_dev.reset(jobs.size() * sizeof(pair_job));
            counts_dev.reset(jobs.size() * sizeof(uint32_t));
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                jobs_dev.data,
                jobs.data(),
                jobs.size() * sizeof(pair_job),
                cudaMemcpyHostToDevice,
                stream
            ));
            exact_intersection_kernel<<<static_cast<uint32_t>(jobs.size()), 256, 0, stream>>>(
                static_cast<pair_job const*>(jobs_dev.data), static_cast<uint32_t*>(counts_dev.data)
            );
            CUDDL_CUDA_CALL(cudaGetLastError());
            intersections.resize(jobs.size());
            CUDDL_CUDA_CALL(cudaMemcpyAsync(
                intersections.data(),
                counts_dev.data,
                jobs.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream
            ));
            CUDDL_CUDA_CALL(cudaStreamSynchronize(stream));
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
        // Sorted, deduplicated k-mers for the two sides of the current pair. A side the sketch
        // pass packed for the device already has them; anything else is uploaded and reduced once
        // per group, which the reference-major schedule makes one pass per genome.
        struct sorted_side {
            device_buffer keys;
            device_buffer unique;
            size_t genome = std::numeric_limits<size_t>::max();
        };
        sorted_side side_a, side_r;
        auto sorted_for = [&](sorted_side& state, size_t g) {
            if (sorted_count[g]) {
                return std::pair{
                    static_cast<uint64_t const*>(sorted_store.data) + sorted_offset[g],
                    static_cast<size_t>(sorted_count[g])
                };
            }
            if (state.genome != g) {
                // Pending jobs point at this scratch, so they are counted before it is filled
                // with another genome. Deferring the flush to the next iteration in the caller
                // is too late: the rebuild below overwrites or frees what they read.
                flush_jobs();
                // A held sequence is re-extracted on the device; anything outside the budget is
                // packed again here, which costs a parse but keeps memory bounded.
                std::vector<uint64_t> parsed;
                uint64_t* keys = nullptr;
                size_t count = 0;
                if (resident[g]) {
                    // The emitter fills a slot per window, broken ones included, so the array is
                    // sized by the genome's windows and the compaction is what leaves the
                    // valid ones behind.
                    size_t windows = 0;
                    for (auto const& piece : stashed[g].pieces) windows += piece.windows;
                    stage_and_compact(stashed[g].bases, stashed[g].pieces, windows);
                    count = kmers_of[g];
                    keys = static_cast<uint64_t*>(compacted_keys.data);
                } else {
                    parsed = parse_one(g);
                    ++reparsed_genomes;
                    count = parsed.size();
                    if (count) {
                        if (state.keys.bytes < count * sizeof(uint64_t)) {
                            state.keys.reset(count * sizeof(uint64_t));
                        }
                        CUDDL_CUDA_CALL(cudaMemcpyAsync(
                            state.keys.data,
                            parsed.data(),
                            count * sizeof(uint64_t),
                            cudaMemcpyHostToDevice,
                            stream
                        ));
                        keys = static_cast<uint64_t*>(state.keys.data);
                    }
                }
                if (state.unique.bytes < count * sizeof(uint64_t)) {
                    state.unique.reset(count * sizeof(uint64_t));
                }
                if (!count) {
                    state.genome = g;
                    return std::pair{
                        static_cast<uint64_t const*>(state.unique.data), static_cast<size_t>(0)
                    };
                }
                CUDDL_CUDA_CALL(
                    cub::DeviceRadixSort::SortKeys(
                        temp.data, temp.bytes, keys, keys, count, 0, 64, stream
                    )
                );
                CUDDL_CUDA_CALL(
                    cub::DeviceRunLengthEncode::Encode(
                        temp.data,
                        temp.bytes,
                        keys,
                        static_cast<uint64_t*>(state.unique.data),
                        static_cast<int*>(pair_counts.data),
                        static_cast<int*>(num_runs_dev.data),
                        count,
                        stream
                    )
                );
                state.genome = g;
            }
            return std::pair{
                static_cast<uint64_t const*>(state.unique.data), static_cast<size_t>(distinct[g])
            };
        };
        for (auto const& scheduled : schedule) {
            auto const [a, r] = pair_at(scheduled.ordinal);
            // Both sides are sorted and deduplicated, so a pair is one linear pass over them and
            // the union follows from distinct[a] + distinct[r] - shared. An empty side shares
            // nothing, which the kernel handles without reading anything.
            uint64_t const* left = nullptr;
            uint64_t const* right = nullptr;
            size_t left_len = 0, right_len = 0;
            if (distinct[a] && distinct[r]) {
                auto const packed_left = sorted_for(side_a, a);
                auto const packed_right = sorted_for(side_r, r);
                left = packed_left.first;
                left_len = packed_left.second;
                right = packed_right.first;
                right_len = packed_right.second;
            }
            jobs.push_back({
                left,
                static_cast<uint32_t>(left_len),
                right,
                static_cast<uint32_t>(right_len),
            });
            job_rows.push_back({
                a,
                r,
                static_cast<char>(
                    scheduled.evaluated_index % emit_stride == 0 ||
                    scheduled.evaluated_index + 1 == evaluated.size()
                ),
            });
        }
        flush_jobs();
        auto const done = clock_type::now();
        if (rep >= 0) {
            using ms = std::chrono::duration<double, std::milli>;
            parse_ms.push_back(ms(parse_done - sample_tick).count());
            sketch_ms.push_back(ms(compare_tick - sample_tick).count());
            compare_ms.push_back(ms(done - compare_tick).count());
            end_to_end_ms.push_back(ms(done - sample_tick).count());
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
          {"stashed_genomes", static_cast<size_t>(std::count(resident.begin(), resident.end(), 1))},
          {"stashed_mb", stashed_bytes >> 20},
          {"reparsed_genomes", reparsed_genomes},
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
          {"compare", summarize(compare_ms)},
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
