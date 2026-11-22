// Near-duplicate deltas for the CPU all-to-all comparator (--deltas), the host port of
// include/cuddl/detail/reference_deltas.cuh. Rows are grouped into bases and children: a child
// differs from a higher-ID base in at most kChangeLimit buckets, and bases are never children. The
// bit-sliced comparison then runs against bases only; each child's (lower, equal, higher, empty)
// counts for a group of queries follow from the base's counts and the child's changed buckets.
//
// A delta unit emits the pairs of one query against bases and children whose IDs are scattered over
// the whole reference range. BlockedStripe stores a stripe of results so that the 16 queries of a
// block share one 64-byte line per reference column; emitting one reference for consecutive
// queries then writes one line instead of one line per query.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

namespace cpu_deltas {

constexpr uint32_t kBuckets = 2048U;
/// Most buckets in which a child may differ from its base.
constexpr uint32_t kChangeLimit = 160U;
/// Exact bucket signatures that propose base candidates, buckets per signature, and highest-ID
/// rows kept from each matching signature.
constexpr uint32_t kSignatures = 8U;
constexpr uint32_t kSignatureBuckets = 8U;
constexpr uint32_t kSignatureCandidates = 4U;
constexpr uint32_t kCandidates = kSignatures * kSignatureCandidates;
/// Queries one correction processes together, one 16-bit lane each.
constexpr uint32_t kGroup = 32U;

/// One changed bucket: `offset` indexes the bucket's row in the bucket-major query scores, and
/// `low`/`width` are min(base, child) and |base - child|.
struct Change {
    uint32_t offset;
    uint16_t low;
    uint16_t width;
};

/// A child's changes are stored as four consecutive runs: increases from an empty base bucket,
/// decreases to an empty child bucket, other increases, other decreases.
struct Child {
    uint32_t id;
    uint32_t begin;
    uint16_t runs[4];
};

struct Model {
    /// Base IDs in ascending order.
    std::vector<uint32_t> bases;
    /// children[child_begin[k] .. child_begin[k + 1]) are the children of bases[k], by ascending
    /// ID.
    std::vector<uint32_t> child_begin;
    std::vector<Child> children;
    std::vector<Change> changes;
};

/// Scores of kGroup queries, bucket-major: lane k of bucket b is query k's score.
struct alignas(64) Lanes {
    uint16_t v[kGroup];
};

/// Change in a child's counts relative to its base, for each query lane.
struct alignas(64) Correction {
    int16_t lower[kGroup];
    int16_t higher[kGroup];
    int16_t empty[kGroup];
};

inline uint32_t differing_buckets(uint16_t const* a, uint16_t const* b) noexcept {
    uint32_t differ = 0;
    for (uint32_t k = 0; k < kBuckets; ++k) {
        differ += a[k] != b[k];
    }
    return differ;
}

/// Writes the distinct candidate bases of a row in descending ID order.
inline uint32_t sorted_candidates(uint32_t const* candidates, uint32_t* out) noexcept {
    uint32_t count = 0;
    for (uint32_t k = 0; k < kCandidates; ++k) {
        if (candidates[k] != ~0U) {
            out[count++] = candidates[k];
        }
    }
    std::sort(out, out + count, std::greater<>{});
    return static_cast<uint32_t>(std::unique(out, out + count) - out);
}

/// Groups the n row-major rows. Candidates come from 8 signatures of 8 evenly spaced buckets: rows
/// whose signature scores are equal propose each other, 4 highest IDs above the row per signature.
/// A row targets its highest-ID candidate within kChangeLimit changed buckets; targeted rows are
/// bases, other targeting rows become children and then move to the closest candidate base.
inline Model build_model(uint16_t const* rows, uint32_t n) {
    auto const row = [&](uint32_t r) {
        return rows + static_cast<size_t>(r) * kBuckets;
    };
    constexpr uint32_t spacing = kBuckets / (kSignatures * kSignatureBuckets);

    std::vector<uint32_t> candidates(static_cast<size_t>(n) * kCandidates, ~0U);
    _Pragma("omp parallel for schedule(dynamic, 1)")
    for (uint32_t signature = 0; signature < kSignatures; ++signature) {
        // Rows enter in descending ID order and the sort is stable, so each run of equal keys
        // lists its highest IDs first.
        std::vector<std::pair<uint64_t, uint32_t>> keyed(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t const r = n - 1U - i;
            uint64_t key = signature + 1U;
            for (uint32_t j = 0; j < kSignatureBuckets; ++j) {
                key = (key ^ row(r)[(signature * kSignatureBuckets + j) * spacing]) *
                      0x9e3779b97f4a7c15ULL;
                key ^= key >> 29U;
            }
            keyed[i] = {key, r};
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
        });
        uint32_t start = 0;
        for (uint32_t p = 0; p < n; ++p) {
            if (p == 0U || keyed[p].first != keyed[p - 1U].first) {
                start = p;
            }
            uint32_t* const out = &candidates
                                      [static_cast<size_t>(keyed[p].second) * kCandidates +
                                       signature * kSignatureCandidates];
            for (uint32_t j = 0; j < kSignatureCandidates; ++j) {
                out[j] = start + j < p ? keyed[start + j].second : ~0U;
            }
        }
    }

    std::vector<uint32_t> target(n);
    std::vector<uint32_t> changed(n, 0U);
    _Pragma("omp parallel for schedule(dynamic, 64)")
    for (uint32_t r = 0; r < n; ++r) {
        target[r] = r;
        uint32_t list[kCandidates];
        uint32_t const count =
            sorted_candidates(&candidates[static_cast<size_t>(r) * kCandidates], list);
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t const differ = differing_buckets(row(r), row(list[k]));
            if (differ <= kChangeLimit) {
                target[r] = list[k];
                changed[r] = differ;
                break;
            }
        }
    }
    std::vector<uint8_t> is_base(n, 0U);
    for (uint32_t r = 0; r < n; ++r) {
        is_base[target[r]] = 1U;
    }
    // Retarget each child to the candidate base with the fewest changes; the base set is fixed.
    _Pragma("omp parallel for schedule(dynamic, 64)")
    for (uint32_t r = 0; r < n; ++r) {
        if (is_base[r] != 0U || changed[r] == 0U) {
            continue;
        }
        uint32_t list[kCandidates];
        uint32_t const count =
            sorted_candidates(&candidates[static_cast<size_t>(r) * kCandidates], list);
        for (uint32_t k = 0; k < count; ++k) {
            if (is_base[list[k]] == 0U || list[k] == target[r]) {
                continue;
            }
            uint32_t const differ = differing_buckets(row(r), row(list[k]));
            if (differ < changed[r]) {
                target[r] = list[k];
                changed[r] = differ;
                if (differ == 0U) {
                    break;
                }
            }
        }
    }

    Model m;
    std::vector<uint32_t> base_index(n, 0U);
    for (uint32_t r = 0; r < n; ++r) {
        if (is_base[r] != 0U) {
            base_index[r] = static_cast<uint32_t>(m.bases.size());
            m.bases.push_back(r);
        }
    }
    m.child_begin.assign(m.bases.size() + 1U, 0U);
    for (uint32_t r = 0; r < n; ++r) {
        if (is_base[r] == 0U) {
            ++m.child_begin[base_index[target[r]] + 1U];
        }
    }
    for (size_t k = 0; k < m.bases.size(); ++k) {
        m.child_begin[k + 1U] += m.child_begin[k];
    }
    m.children.resize(m.child_begin.back());
    std::vector<uint32_t> cursor(m.child_begin.begin(), m.child_begin.end() - 1);
    for (uint32_t r = 0; r < n; ++r) {
        if (is_base[r] == 0U) {
            m.children[cursor[base_index[target[r]]]++] = Child{r, 0U, {0, 0, 0, 0}};
        }
    }

    auto const kind_of = [](uint16_t before, uint16_t after) noexcept {
        return before == 0U ? 0U : after == 0U ? 1U : after > before ? 2U : 3U;
    };
    _Pragma("omp parallel for schedule(dynamic, 64)")
    for (size_t c = 0; c < m.children.size(); ++c) {
        Child& child = m.children[c];
        uint16_t const* const before = row(target[child.id]);
        uint16_t const* const after = row(child.id);
        for (uint32_t b = 0; b < kBuckets; ++b) {
            if (before[b] != after[b]) {
                ++child.runs[kind_of(before[b], after[b])];
            }
        }
    }
    uint32_t total = 0;
    for (Child& child : m.children) {
        child.begin = total;
        total += child.runs[0] + child.runs[1] + child.runs[2] + child.runs[3];
    }
    m.changes.resize(total);
    _Pragma("omp parallel for schedule(dynamic, 64)")
    for (size_t c = 0; c < m.children.size(); ++c) {
        Child const& child = m.children[c];
        uint16_t const* const before = row(target[child.id]);
        uint16_t const* const after = row(child.id);
        uint32_t next[4] = {
            child.begin,
            child.begin + child.runs[0],
            child.begin + child.runs[0] + child.runs[1],
            child.begin + child.runs[0] + child.runs[1] + child.runs[2],
        };
        for (uint32_t b = 0; b < kBuckets; ++b) {
            if (before[b] != after[b]) {
                uint16_t const low = std::min(before[b], after[b]);
                m.changes[next[kind_of(before[b], after[b])]++] = Change{
                    b * kGroup, low, static_cast<uint16_t>(std::max(before[b], after[b]) - low)
                };
            }
        }
    }
    return m;
}

/// Stores the scores of `count` (at most kGroup) rows starting at row `first` bucket-major; unused
/// lanes are zero.
inline void
transpose_group(uint16_t const* rows, uint32_t first, uint32_t count, Lanes* out) noexcept {
    constexpr uint32_t block = 64U;
    for (uint32_t b0 = 0; b0 < kBuckets; b0 += block) {
        for (uint32_t k = 0; k < kGroup; ++k) {
            uint16_t const* const src =
                k < count ? rows + static_cast<size_t>(first + k) * kBuckets : nullptr;
            for (uint32_t b = b0; b < b0 + block; ++b) {
                out[b].v[k] = src != nullptr ? src[b] : uint16_t{0};
            }
        }
    }
}

namespace detail {

// One native vector per step: zmm with AVX-512BW, ymm with AVX2, a 128-bit vector otherwise (NEON;
// wider generic vectors fall back to scalar code there).
#if defined(__AVX512BW__)
constexpr uint32_t kVectorBytes = 64U;
#elif defined(__AVX2__)
constexpr uint32_t kVectorBytes = 32U;
#else
constexpr uint32_t kVectorBytes = 16U;
#endif
constexpr uint32_t kVectorLanes = kVectorBytes / sizeof(uint16_t);
static_assert(kGroup % kVectorLanes == 0U);

using Vec = uint16_t __attribute__((vector_size(kVectorBytes)));
using SVec = int16_t __attribute__((vector_size(kVectorBytes)));

/// Applies a run of changes to the vector of query lanes starting at lane `part`. For a bucket with
/// base score b, child score a and query score v, an increase (b < a) adds [b <= v < a] to the
/// lower count and removes [b < v <= a] from the higher count; a decrease mirrors both. With
/// `Zero`, one of a and b is 0: an empty query bucket moves between the empty count and the lower
/// count instead of the equal count.
template <bool Increase, bool Zero>
inline void apply_run(
    Change const* it,
    Change const* end,
    uint16_t const* scores,
    SVec& lower,
    SVec& higher,
    SVec& empty
) noexcept {
    for (; it != end; ++it) {
        Vec v;
        std::memcpy(&v, scores + it->offset, sizeof(v));
        Vec const x = v - Vec{} - it->low;
        Vec const width = Vec{} + it->width;
        SVec const in_low = x < width;
        SVec const in_high = x - Vec{} - uint16_t{1} < width;
        if constexpr (Increase) {
            lower -= in_low;
            higher += in_high;
        } else {
            lower += in_low;
            higher -= in_high;
        }
        if constexpr (Zero) {
            SVec const query_empty = v == Vec{};
            if constexpr (Increase) {
                empty += query_empty;
            } else {
                empty -= query_empty;
            }
        }
    }
}

}  // namespace detail

/// Change of the counts of `child` against each query lane of `scores` relative to its base.
inline void
correct(Child const& child, Change const* changes, Lanes const* scores, Correction& out) noexcept {
    using detail::SVec;
    for (uint32_t part = 0; part < kGroup; part += detail::kVectorLanes) {
        SVec lower{}, higher{}, empty{};
        uint16_t const* const lanes = &scores->v[part];
        Change const* it = changes + child.begin;
        detail::apply_run<true, true>(it, it + child.runs[0], lanes, lower, higher, empty);
        it += child.runs[0];
        detail::apply_run<false, true>(it, it + child.runs[1], lanes, lower, higher, empty);
        it += child.runs[1];
        detail::apply_run<true, false>(it, it + child.runs[2], lanes, lower, higher, empty);
        it += child.runs[2];
        detail::apply_run<false, false>(it, it + child.runs[3], lanes, lower, higher, empty);
        std::memcpy(&out.lower[part], &lower, sizeof(lower));
        std::memcpy(&out.higher[part], &higher, sizeof(higher));
        std::memcpy(&out.empty[part], &empty, sizeof(empty));
    }
}

/// Layout of a stripe of upper-triangle results, query rows [first_row, first_row + rows), in
/// blocks of kBlockRows consecutive rows. Block k holds, for each column from its first row + 1 to
/// n - 1, one line of kBlockRows floats, one per row of the block; cells at or left of the diagonal
/// and lanes past the stripe's last row are padding that is never written.
struct BlockedStripe {
    static constexpr uint32_t kBlockRows = 16U;

    uint32_t first_row = 0;
    uint32_t rows = 0;
    uint32_t n = 0;
    /// Start of every block in floats, plus the total; every block is a multiple of one line.
    std::vector<uint64_t> offset;

    void init(uint32_t first, uint32_t row_count, uint32_t total) {
        first_row = first;
        rows = row_count;
        n = total;
        uint32_t const count = (rows + kBlockRows - 1U) / kBlockRows;
        offset.assign(count + 1U, 0U);
        for (uint32_t k = 0; k < count; ++k) {
            offset[k + 1U] = offset[k] + uint64_t{kBlockRows} * width(k);
        }
    }

    [[nodiscard]] uint32_t blocks() const noexcept {
        return static_cast<uint32_t>(offset.size()) - 1U;
    }

    /// Columns of block k.
    [[nodiscard]] uint32_t width(uint32_t k) const noexcept {
        return n - 1U - (first_row + k * kBlockRows);
    }

    [[nodiscard]] uint64_t floats() const noexcept {
        return offset.back();
    }

    /// Position of the pair (row, col), col > row, in a buffer of floats().
    [[nodiscard]] uint64_t index(uint32_t row, uint32_t col) const noexcept {
        uint32_t const rel = row - first_row;
        uint32_t const k = rel / kBlockRows;
        return offset[k] + uint64_t{col - (first_row + k * kBlockRows) - 1U} * kBlockRows +
               (rel % kBlockRows);
    }

    /// Floats a buffer needs for any stripe of at most `max_rows` rows starting at a multiple of
    /// `step` rows, plus alignment slack for 64-byte lines.
    [[nodiscard]] static uint64_t max_floats(uint32_t total, uint32_t max_rows, uint32_t step) {
        BlockedStripe stripe;
        uint64_t most = 0;
        for (uint32_t first = 0; first < total; first += step) {
            stripe.init(first, std::min(max_rows, total - first), total);
            most = std::max(most, stripe.floats());
        }
        return most + 16U;
    }

    /// Mask of the lanes of block k that hold rows of the stripe.
    [[nodiscard]] uint32_t row_lanes(uint32_t k) const noexcept {
        return (uint32_t{1} << std::min(kBlockRows, rows - k * kBlockRows)) - 1U;
    }

    static uint32_t nonzero_lanes(float const* line) noexcept {
        uint32_t mask = 0;
        for (uint32_t j = 0; j < kBlockRows; ++j) {
            mask |= static_cast<uint32_t>(line[j] != 0.0f) << j;
        }
        return mask;
    }

    /// Nonzero lanes of the line d columns into block k that hold results.
    [[nodiscard]] static uint32_t
    valid_lanes(float const* line, uint32_t d, uint32_t rows_mask) noexcept {
        uint32_t mask = nonzero_lanes(line) & rows_mask;
        if (d < kBlockRows - 1U) {
            mask &= (uint32_t{2} << d) - 1U;
        }
        return mask;
    }

    /// Writes the encoding of every row of block k. Row j of the block starts at dst[j]; its bytes
    /// need not be aligned. With `Edges`, a row is its nonzero results by ascending column as
    /// (row, column, value) records of 12 bytes. Otherwise it is a bitmap of (len + 63) / 64 words
    /// (bit c - row - 1 for column c) followed by the nonzero values by ascending column.
    template <bool Edges>
    void encode(float const* data, uint32_t k, char* const* dst) const noexcept {
        float const* const base = data + offset[k];
        uint32_t const cols = width(k);
        uint32_t const rows_mask = row_lanes(k);
        uint32_t const first = first_row + k * kBlockRows;
        char* values[kBlockRows];
        for (uint32_t j = 0; j < kBlockRows && (rows_mask >> j) != 0U; ++j) {
            if constexpr (Edges) {
                values[j] = dst[j];
            } else {
                uint64_t const len = n - 1U - (first + j);
                uint64_t const words = (len + 63U) / 64U;
                std::memset(dst[j], 0, words * 8U);
                values[j] = dst[j] + words * 8U;
            }
        }
        for (uint32_t d = 0; d < cols; ++d) {
            float const* const line = base + uint64_t{d} * kBlockRows;
            uint32_t mask = valid_lanes(line, d, rows_mask);
            while (mask != 0U) {
                uint32_t const j = static_cast<uint32_t>(std::countr_zero(mask));
                mask &= mask - 1U;
                if constexpr (Edges) {
                    uint32_t const row = first + j;
                    uint32_t const col = first + 1U + d;
                    std::memcpy(values[j], &row, 4);
                    std::memcpy(values[j] + 4, &col, 4);
                    std::memcpy(values[j] + 8, &line[j], 4);
                    values[j] += 12;
                } else {
                    // Column first + 1 + d and row first + j give bit d - j.
                    uint32_t const bit = d - j;
                    char* const word = dst[j] + (bit >> 6U) * 8U;
                    uint64_t bits;
                    std::memcpy(&bits, word, 8);
                    bits |= uint64_t{1} << (bit & 63U);
                    std::memcpy(word, &bits, 8);
                    std::memcpy(values[j], &line[j], 4);
                    values[j] += 4;
                }
            }
        }
    }
};

}  // namespace cpu_deltas
