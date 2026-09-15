#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuddl/detail/fastx_sequence_file.hpp>

namespace resident_sequence {

struct chunk {
    size_t genome;
    size_t offset;
    size_t size;
};

struct batch {
    std::vector<char> bases;
    std::vector<chunk> chunks;
};

// The cap bounds staged ASCII payload, including overlap, not parser/sketch/result storage.
// Each chunk contains complete k-mer windows; overlap never crosses a record boundary.
// Batches may exceed UINT32_MAX bytes; every chunk's window count fits uint32. Records
// stream through one bounded piece buffer with plain vector growth: no cap-sized reserves
// and no whole-record stripped copy, so a huge budget over a tiny corpus stages tiny input.
//
// Files are decompressed and compacted by a loader pool rather than by the thread building
// batches: parsing every genome on one core otherwise dominates a full corpus, and the whole
// staging phase would sit on a single core while the workers wait.
template <typename Consume>
size_t for_each_batch(
    std::vector<std::string> const& paths,
    uint32_t k,
    size_t cap,
    Consume&& consume,
    unsigned workers = 0
) {
    if (k == 0 || cap < k) {
        throw std::invalid_argument("resident byte budget must be at least k");
    }
    // A chunk holds at most UINT32_MAX windows: size - k + 1 <= UINT32_MAX.
    uint64_t const chunk_limit64 = uint64_t{UINT32_MAX} + static_cast<uint64_t>(k) - 1U;
    size_t const chunk_limit =
        chunk_limit64 >= static_cast<uint64_t>(cap)
            ? cap
            : (chunk_limit64 >= static_cast<uint64_t>(std::numeric_limits<size_t>::max())
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>(chunk_limit64));
    batch current;
    size_t batches = 0;
    auto flush = [&] {
        if (current.bases.empty()) {
            return;
        }
        consume(current);
        ++batches;
        current.bases.clear();
        current.chunks.clear();
    };
    size_t piece_limit = 0;
    auto refill = [&] {
        // No useful piece fits the rest of this batch; seal it first.
        if (cap - current.bases.size() < k) {
            flush();
        }
        size_t const room = cap - current.bases.size();
        piece_limit = chunk_limit < room ? chunk_limit : room;
    };
    auto append = [&](size_t genome, char const* data, size_t size) {
        if (size < k) {
            return;
        }
        if (size > cap - current.bases.size()) {
            flush();
        }
        current.chunks.push_back({genome, current.bases.size(), size});
        current.bases.insert(current.bases.end(), data, data + size);
    };

    std::vector<std::filesystem::path> file_paths;
    file_paths.reserve(paths.size());
    for (auto const& path : paths) file_paths.emplace_back(path);
    auto const depth = std::max<size_t>(
        1,
        std::min(
            paths.size(),
            size_t{workers != 0 ? workers : std::max(1U, std::thread::hardware_concurrency())}
        )
    );
    cuddl::detail::fastx_load_pool loader(file_paths, depth);
    for (size_t genome = 0; genome < paths.size(); ++genome) {
        // Extents arrive compacted, so a piece is a straight copy of bases.
        auto sequence = loader.take(genome);
        if (!sequence) {
            throw std::runtime_error(paths[genome] + ": " + sequence.error().message());
        }
        for (auto const& extent : (*sequence)->extents) {
            auto const* const bases = extent.begin;
            auto const size = static_cast<size_t>(extent.end - extent.begin);
            size_t start = 0;
            while (start < size) {
                refill();
                auto const count = std::min(size - start, piece_limit);
                append(genome, bases + start, count);
                if (start + count == size) break;
                // The next piece repeats the k - 1 bases it shares with this one, so no window
                // is lost at the boundary and none is counted twice.
                start += count - (k - 1U);
            }
        }
    }
    flush();
    return batches;
}

}  // namespace resident_sequence
