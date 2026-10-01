# Near-duplicate references in exhaustive search

RefSeq holds many strains of the same species, and their sketches often differ in only a few dozen of their 2,048 buckets. An exhaustive search still compares every query with every row in full. For two rows that differ in 60 buckets, the second comparison mostly repeats the first.

`reference_database::build_async` therefore stores each row in one of two roles:

- A **base** row is compared with queries in full.
- A **child** row points at one base and lists the buckets in which it differs from that base. A query's counts against a child follow from its counts against the base and from those buckets alone.

The code lives in `include/cuddl/detail/reference_deltas.cuh`. Search results do not change. Every count is exact, and `ExhaustiveSearchesOfNearDuplicateRowsMatchScalarOracle` checks batch and all-to-all results against a scalar oracle.

## How rows are grouped

Grouping runs on the GPU at the end of `build_async`, in `build_reference_deltas`:

1. Pick 64 evenly spaced buckets and split them into 8 signatures of 8 buckets each (`delta_signatures`, `delta_signature_buckets`).
2. For each signature, hash every row's scores in those buckets and sort the rows by hash. Rows with equal hashes are candidates for each other.
3. From each signature, keep the 4 highest-ID candidates whose ID is above the row's own (`delta_signature_candidates`). That gives each row up to 32 candidates.
4. One warp per row compares the row with its candidates from the highest ID down. The first candidate that differs in at most `delta_change_limit` buckets becomes the row's target.
5. Every row that some row targets becomes a base. A row targeted by nobody becomes a child of its own target. A row without a target is a base.
6. Revisit each child's candidates and choose the existing base with the fewest changed buckets. Keep the current target on a tie. This pass leaves the base set unchanged.

Steps 5 and 6 preserve two properties that matter at search time.

The grouping is flat. A base is never a child, so correcting a child reads a base result that the base refinement has already written. Corrections never wait on other corrections.

A child's base always has a higher ID than the child. All-to-all searches fill only pairs with `query_id < reference_id`. If a pair `(query, child)` is in a tile, the pair `(query, base)` is in the same tile, because `base > child > query`.

Hashing exact scores misses rows that differ inside every signature, so some near duplicates stay bases. The closest-base pass takes about 0.16 ms of GPU time for 43,897 rows on an RTX 5070 Ti.

## How a search uses the groups

`launch_batch_exhaustive` runs two kernels per query tile:

1. `refine_batch_bitmap_kernel` with `refine_candidates::reference_mask` compares the queries with the base rows only. The mask is one bit per reference, shared by every query.
2. `apply_reference_deltas_kernel` writes every child's result. Each block decodes a group of query rows into 64 KiB of shared memory, stored bucket-major so that the lanes of a warp read one bucket for several queries at once. At 2,048 buckets a group holds 16 queries. Each warp then takes one child at a time and loads the next child's changes while it works on the current one.

For a changed bucket where the base score is `b`, the child score is `a`, and the query score is `v`, only the query's classification against that bucket can change:

- The lower count changes by `[v < a] - [v < b]`.
- The higher count changes by `[v > a] - [v > b]`.

If the score increases (`b < a`), the lower count gains one when `b <= v < a`, and the higher count loses one when `b < v <= a`. A decrease mirrors the two tests. Each changed bucket is stored as `low = min(a, b)` and `width = |a - b|`, so both tests are one subtraction and one unsigned comparison. Increases and decreases are stored in separate runs, so the loop carries no sign.

The equal count follows from the total, because lower, equal, higher, and both-empty buckets always add up to the bucket count. Buckets where `low` is zero need one more term. When the query bucket is empty and a reference bucket becomes empty or stops being empty, the bucket moves between both-empty and lower, not between equal and lower. These buckets are stored first and carry that correction.

## Which searches use it

Only exhaustive batch searches and exhaustive all-to-all searches without match counts use the groups. These searches fall back to comparing every row:

- indexed searches
- searches with a minimum-match threshold
- searches that request `result_match_counts`
- single-query `search_async`
- devices whose opt-in shared memory per block is smaller than `delta_correction_shared_bytes` (96,256 bytes)

Indexed searches compare only the pairs that pass the match filter. A correction loads a whole query group and one child's changes even when only one pair in that group passed. In a prototype with a limit of 128 buckets, corrections made the dense indexed refinement step slower: 7.22 ms became 7.64 ms for a 4,096-query batch, and 50.8 ms became 59.3 ms for all-to-all.

## What it costs

`build_async` waits for its stream once, to read how many children and changed buckets it must allocate. The groups are not part of the database file. `reference_database_file::upload` calls `build_async`, so every upload groups the rows again.

On the frozen 43,897-genome RefSeq corpus at 2,048 buckets, the grouping model gives:

| Item                  | Value                      |
| --------------------- | -------------------------- |
| Bases                 | 22,647 rows (51.59%)       |
| Children              | 21,250 rows                |
| Changed buckets       | 1,268,461 (59.7 per child) |
| Delta storage         | 10.01 MiB                  |
| Row bit-plane storage | 171.5 MiB                  |

Each child takes 16 bytes and each changed bucket takes 8 bytes. The base mask takes one bit per row.

## Why the limit is 160 buckets

A higher `delta_change_limit` turns more rows into children, but each child then costs more to correct. Base refinement dominates the run time, so the higher limits measured faster. These timings used a 43,029-reference corpus with 4,096 queries on an RTX 5070 Ti, without the closest-base pass. They are not comparable to timings on the 43,897-reference corpus above.

| `delta_change_limit` | Exhaustive batch | All-to-all |
| -------------------- | ---------------- | ---------- |
| No grouping          | 35.56 ms         | 197.43 ms  |
| 64                   | 34.09 ms         | 187.23 ms  |
| 128                  | 31.64 ms         | 174.77 ms  |
| 160                  | 30.83 ms         | 171.17 ms  |

Shared memory caps the limit. Each of the 24 warps in a correction block buffers one child's changes, `delta_change_limit * 8` bytes, next to the 64 KiB of staged queries. Each lane also holds the next child's changes in registers, so the limit must be a multiple of 32. 160 is the largest multiple of 32 that fits in the 99 KiB a block can request on `sm_120`.
