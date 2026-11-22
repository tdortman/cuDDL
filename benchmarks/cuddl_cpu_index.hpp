// Inverted index for the CPU thresholded all-to-all (--index), the host counterpart of the GPU's
// reference index. A cell (bucket, score & 0x7fff) lists the IDs of the references that hold that
// score in that bucket, ascending; empty buckets are not indexed. Two rows that agree in E nonzero
// buckets share E cells, so a pair whose cell count reaches the threshold is a candidate and a pair
// that stays below it cannot reach E >= threshold. Masking bit 15 can only add matches.
//
// Posting lists of at least n / bitmap_divisor entries are also stored as n-bit bitmaps, and a
// query adds those into bit-sliced (vertical) counters, so a dense cluster costs n / 64 word
// additions per list instead of one random increment per posting. A divisor of 16 was fastest on
// the GH200 among 0, 12, 16, 24 and 32; longer cutoffs leave more random increments and shorter
// ones spend more on bitmap adds than on the lists they replace.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cpu_index {

constexpr uint32_t kBuckets = 2048U;
constexpr uint32_t kKeyMask = 0x7fffU;
/// Buckets one build task reads from every row.
constexpr uint32_t kBuildBlock = 16U;
/// Bit-sliced counter planes; counts reach at most kBuckets, which needs 12 bits.
constexpr uint32_t kPlanes = 12U;

struct Index {
    uint32_t references = 0;
    /// Score keys per bucket: the largest key present plus one.
    uint32_t keys = 0;
    uint32_t words = 0;
    /// Cell (bucket * keys + key) holds postings[offsets[cell] .. offsets[cell + 1]).
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> postings;
    /// Lists of at least `cutoff` postings (0: none) also have a bitmap; `long_cells` holds their
    /// cells in ascending order and bitmap s belongs to long_cells[s].
    uint32_t cutoff = 0;
    std::vector<uint32_t> long_cells;
    std::vector<uint64_t> bitmaps;
    uint64_t nonempty_cells = 0;
    uint64_t bitmap_cells = 0;

    [[nodiscard]] uint64_t bytes() const noexcept {
        return offsets.size() * sizeof(uint32_t) + postings.size() * sizeof(uint32_t) +
               long_cells.size() * sizeof(uint32_t) + bitmaps.size() * sizeof(uint64_t);
    }
};

/// Builds the index of the n row-major rows; lists of at least n / bitmap_divisor postings also get
/// a bitmap (0 builds none). Fails when the postings do not fit 32-bit offsets.
inline bool build(uint16_t const* rows, uint32_t n, uint32_t bitmap_divisor, Index& ix) {
    ix = Index{};
    ix.references = n;
    ix.words = (n + 63U) / 64U;
    if (static_cast<uint64_t>(n) * kBuckets >= (uint64_t{1} << 32U)) {
        return false;
    }
    uint32_t max_key = 0;
    _Pragma("omp parallel for schedule(static) reduction(max : max_key)")
    for (uint64_t k = 0; k < static_cast<uint64_t>(n) * kBuckets; ++k) {
        max_key = std::max<uint32_t>(max_key, rows[k] & kKeyMask);
    }
    ix.keys = max_key + 1U;
    size_t const cells = static_cast<size_t>(kBuckets) * ix.keys;
    ix.offsets.assign(cells + 1U, 0U);
    constexpr uint32_t blocks = kBuckets / kBuildBlock;
    _Pragma("omp parallel for schedule(dynamic, 1)")
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        for (uint32_t r = 0; r < n; ++r) {
            uint16_t const* const row = rows + static_cast<size_t>(r) * kBuckets;
            for (uint32_t bb = 0; bb < kBuildBlock; ++bb) {
                uint32_t const bucket = blk * kBuildBlock + bb;
                if (row[bucket] != 0U) {
                    ++ix.offsets
                          [static_cast<size_t>(bucket) * ix.keys + (row[bucket] & kKeyMask) + 1U];
                }
            }
        }
    }
    for (size_t c = 0; c < cells; ++c) {
        ix.nonempty_cells += ix.offsets[c + 1U] != 0U;
        ix.offsets[c + 1U] += ix.offsets[c];
    }
    ix.postings.resize(ix.offsets[cells]);
    std::vector<uint32_t> cursor(ix.offsets.begin(), ix.offsets.end() - 1);
    _Pragma("omp parallel for schedule(dynamic, 1)")
    for (uint32_t blk = 0; blk < blocks; ++blk) {
        for (uint32_t r = 0; r < n; ++r) {
            uint16_t const* const row = rows + static_cast<size_t>(r) * kBuckets;
            for (uint32_t bb = 0; bb < kBuildBlock; ++bb) {
                uint32_t const bucket = blk * kBuildBlock + bb;
                if (row[bucket] != 0U) {
                    size_t const cell =
                        static_cast<size_t>(bucket) * ix.keys + (row[bucket] & kKeyMask);
                    ix.postings[cursor[cell]++] = r;
                }
            }
        }
    }
    cursor = std::vector<uint32_t>();
    if (bitmap_divisor > 0U) {
        ix.cutoff = std::max(n / bitmap_divisor, 1U);
        for (size_t c = 0; c < cells; ++c) {
            if (ix.offsets[c + 1U] - ix.offsets[c] >= ix.cutoff) {
                ix.long_cells.push_back(static_cast<uint32_t>(c));
            }
        }
        ix.bitmap_cells = ix.long_cells.size();
        ix.bitmaps.assign(ix.long_cells.size() * ix.words, 0U);
        _Pragma("omp parallel for schedule(dynamic, 1)")
        for (size_t s = 0; s < ix.long_cells.size(); ++s) {
            uint64_t* const bits = &ix.bitmaps[s * ix.words];
            for (uint32_t p = ix.offsets[ix.long_cells[s]]; p < ix.offsets[ix.long_cells[s] + 1U];
                 ++p) {
                bits[ix.postings[p] >> 6U] |= uint64_t{1} << (ix.postings[p] & 63U);
            }
        }
    }
    return true;
}

/// Work counters of one thread's lookups.
struct Work {
    uint64_t candidates = 0;
    uint64_t list_postings = 0;
    uint64_t bitmap_lists = 0;
};

/// Per-thread scratch for lookups.
class Lookup {
   public:
    /// Appends to `out` every reference above row `i` that shares at least `min_equal` cells with
    /// the row `query`, in ascending ID order.
    void candidates(
        Index const& ix,
        uint16_t const* query,
        uint32_t i,
        uint32_t min_equal,
        std::vector<uint32_t>& out,
        Work& work
    ) {
        uint32_t const n = ix.references;
        uint32_t const first = i + 1U;
        uint32_t const first_word = first >> 6U;
        count_.resize(n);
        planes_.resize(static_cast<size_t>(ix.words) * kPlanes);
        std::memset(count_.data() + first, 0, (n - first) * sizeof(uint16_t));
        for (uint32_t p = 0; p < kPlanes; ++p) {
            std::memset(
                planes_.data() + static_cast<size_t>(p) * ix.words + first_word,
                0,
                (ix.words - first_word) * sizeof(uint64_t)
            );
        }
        bool any_bitmap = false;
        for (uint32_t bucket = 0; bucket < kBuckets; ++bucket) {
            uint32_t const key = query[bucket] & kKeyMask;
            if (query[bucket] == 0U || key >= ix.keys) {
                continue;
            }
            size_t const cell = static_cast<size_t>(bucket) * ix.keys + key;
            if (ix.cutoff != 0U && ix.offsets[cell + 1U] - ix.offsets[cell] >= ix.cutoff) {
                auto const slot = static_cast<size_t>(
                    std::lower_bound(ix.long_cells.begin(), ix.long_cells.end(), cell) -
                    ix.long_cells.begin()
                );
                add_bitmap(&ix.bitmaps[slot * ix.words], first_word, ix.words);
                any_bitmap = true;
                ++work.bitmap_lists;
                continue;
            }
            uint32_t const* p = ix.postings.data() + ix.offsets[cell];
            uint32_t const* const end = ix.postings.data() + ix.offsets[cell + 1U];
            p = std::upper_bound(p, end, i);
            work.list_postings += static_cast<uint64_t>(end - p);
            for (; p != end; ++p) {
                ++count_[*p];
            }
        }
        if (any_bitmap) {
            fold_planes(first, first_word, ix.words);
        }
        size_t const before = out.size();
        for (uint32_t j = first; j < n; ++j) {
            if (count_[j] >= min_equal) {
                out.push_back(j);
            }
        }
        work.candidates += out.size() - before;
    }

   private:
    static constexpr uint32_t kBlock = 8U;

    // Adds a bitmap to the vertical counters: ripple-carry through the planes, kBlock words at a
    // time without branches until no word of the block carries.
    void add_bitmap(uint64_t const* bits, uint32_t first_word, uint32_t words) noexcept {
        uint32_t w = first_word;
        for (; w + kBlock <= words; w += kBlock) {
            uint64_t carry[kBlock];
            std::memcpy(carry, bits + w, sizeof(carry));
            for (uint32_t p = 0; p < kPlanes; ++p) {
                uint64_t any = 0;
                for (uint32_t k = 0; k < kBlock; ++k) {
                    any |= carry[k];
                }
                if (any == 0U) {
                    break;
                }
                uint64_t* const plane = &planes_[static_cast<size_t>(p) * words + w];
                for (uint32_t k = 0; k < kBlock; ++k) {
                    uint64_t const before = plane[k];
                    plane[k] = before ^ carry[k];
                    carry[k] &= before;
                }
            }
        }
        for (; w < words; ++w) {
            uint64_t carry = bits[w];
            for (uint32_t p = 0; carry != 0U && p < kPlanes; ++p) {
                uint64_t& plane = planes_[static_cast<size_t>(p) * words + w];
                uint64_t const before = plane;
                plane = before ^ carry;
                carry &= before;
            }
        }
    }

    // Adds the vertical counts of references from `first` on into the scalar counts.
    void fold_planes(uint32_t first, uint32_t first_word, uint32_t words) noexcept {
        for (uint32_t w = first_word; w < words; ++w) {
            uint64_t any = 0;
            for (uint32_t p = 0; p < kPlanes; ++p) {
                any |= planes_[static_cast<size_t>(p) * words + w];
            }
            if (w == first_word) {
                any &= ~uint64_t{0} << (first & 63U);
            }
            while (any != 0U) {
                uint32_t const bit = static_cast<uint32_t>(std::countr_zero(any));
                any &= any - 1U;
                uint32_t value = 0;
                for (uint32_t p = 0; p < kPlanes; ++p) {
                    value |= static_cast<uint32_t>(
                                 (planes_[static_cast<size_t>(p) * words + w] >> bit) & 1U
                             )
                             << p;
                }
                count_[w * 64U + bit] = static_cast<uint16_t>(count_[w * 64U + bit] + value);
            }
        }
    }

    std::vector<uint16_t> count_;
    /// planes_[p * words + word] is bit p of the 64 counters of `word`.
    std::vector<uint64_t> planes_;
};

}  // namespace cpu_index
