#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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
template <typename Consume>
size_t
for_each_batch(std::vector<std::string> const& paths, uint32_t k, size_t cap, Consume&& consume) {
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
    std::vector<char> piece;
    size_t piece_limit = 0;
    auto refill = [&] {
        // No useful piece fits the rest of this batch; seal it first.
        if (cap - current.bases.size() < k) {
            flush();
        }
        size_t const room = cap - current.bases.size();
        piece_limit = chunk_limit < room ? chunk_limit : room;
    };
    auto append = [&](size_t genome, std::vector<char>& staged) {
        if (staged.size() < k) {
            return;
        }
        if (staged.size() > cap - current.bases.size()) {
            flush();
        }
        current.chunks.push_back({genome, current.bases.size(), staged.size()});
        current.bases.insert(current.bases.end(), staged.begin(), staged.end());
    };
    for (size_t genome = 0; genome < paths.size(); ++genome) {
        auto loaded = cuddl::detail::load_fastx_sequence_file(paths[genome]);
        if (!loaded) {
            throw std::runtime_error(paths[genome] + ": " + loaded.error().message());
        }
        for (auto const& extent : (*loaded)->extents) {
            piece.clear();
            refill();
            for (auto cursor = extent.begin; cursor != extent.end; ++cursor) {
                char const base = *cursor;
                if (base == '\n' || base == '\r' || base == ' ' || base == '\t') {
                    continue;
                }
                piece.push_back(base);
                if (piece.size() == piece_limit) {
                    append(genome, piece);
                    piece.erase(piece.begin(), piece.end() - (k - 1U));
                    refill();
                }
            }
            append(genome, piece);
        }
    }
    flush();
    return batches;
}

}  // namespace resident_sequence
