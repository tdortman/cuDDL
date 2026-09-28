# Build and save a reference database

`<cuddl/reference_database_file.cuh>` builds one GPU sketch per genome file and saves the sketches as a binary database. You can load that database later on any host, without the original genomes. The file API needs a POSIX host, because it publishes finished files with an atomic rename.

```cpp
#include <cuddl/reference_database_file.cuh>

int main() {
    cuda::stream stream{cuda::devices[0]};
    std::vector<std::filesystem::path> genomes{
        "reference-a.fna", "reference-b.fasta", "reference-c.fastq"
    };

    auto file = CUDDL_UNWRAP(
        (cuddl::reference_database_file::build<25, 2048>(genomes, stream))
    );
    CUDDL_UNWRAP(file.save("references.cuddl"));

    // In a later process, without the genome files:
    auto saved = CUDDL_UNWRAP(cuddl::reference_database_file::load("references.cuddl"));
    auto database = CUDDL_UNWRAP((saved.upload<25, 2048>(stream)));
    // saved.names()[hit.reference_id] is the path a search hit came from.
}
```

Compile against the `cuddl_dep` Meson dependency. `CUDDL_UNWRAP` throws on failure, which suits applications. Library code can inspect the returned `cuddl::Result` instead.

## Input and reference IDs

`build` takes an ordered list of FASTA or FASTQ paths, plain or gzip-compressed, BGZF included. It detects compression from the file's first bytes, not its extension. Corrupt or truncated gzip returns an error.

All records in one file form one genome. No k-mer spans two records or an ambiguous base. FASTQ quality lines are validated and then ignored, the same way `parse_fasta_file` treats them.

The position of a path in the list is its zero-based reference ID. `names()` keeps every path string as given, duplicates included. A missing file or malformed FASTQ returns an error. These cases still succeed:

- An empty file, or a genome with no valid k-mers, keeps its ID and gets an empty sketch.
- An empty path list builds an empty database.

The optional third argument to `build` is a `path_build_options`. `parser_workers` sets how many genomes load at once. It defaults to the machine's thread count, capped by the number of inputs. Zero is an error. To load one genome at a time and keep host memory low, set `parser_workers = 1`, or pass `--workers 1` to the CLI. The saved bytes are the same for every worker count and every decompression backend.

## Choose a decompression backend

`path_build_options::decompression` picks one of four backends:

- `automatic` picks for you. Without nvCOMP it uses `cpu`. With nvCOMP it uses `coherent` on a coherent-memory GPU with more than one loader, `cpu` on a coherent-memory GPU with one loader, and `gpu` everywhere else.
- `cpu` inflates and parses every file on host workers. Sketching still runs on the GPU.
- `gpu` inflates eligible gzip files with nvCOMP. The CPU still reads the files and hands anything nvCOMP can't take to the host loader.
- `coherent` inflates on CPU workers and copies the raw FASTA to the GPU, which strips headers and whitespace before sketching. It needs a GPU that reads pageable host memory, such as GH200, GB300, or GB10. cuDDL checks for that capability, not for a device name.

`reference_database_file::build`, `build_sketch_store`, and `query_sketch_batch::sketch` all take the same options:

```cpp
auto file = CUDDL_UNWRAP((cuddl::reference_database_file::build<25, 2048>(
    genomes, stream,
    {.parser_workers = 72, .decompression = cuddl::decompression_backend::cpu}
)));
```

`cuddl-build-reference-db`, `cuddl-reference-index`, and the reference-build benchmark take the same choice as `--decompression automatic|cpu|gpu|coherent`:

```sh
cuddl-build-reference-db genomes/ --workers 4 --decompression gpu
```

### Build with or without nvCOMP

Meson defaults to `-Dnvcomp=auto`, which enables GPU decompression when it finds the nvCOMP library and headers. Pass `-Dnvcomp=enabled` to require nvCOMP, or `-Dnvcomp=disabled` to build without it. CUDA is required either way. Without nvCOMP, explicit `gpu` and `coherent` requests return an error, and `automatic` falls back to `cpu`.

Consumers of `cuddl_dep` get the right macro and link flags. If you include the headers directly, they default to CPU decompression. To turn on nvCOMP, compile every translation unit with `-DCUDDL_HAS_NVCOMP=1`, add nvCOMP's include and library paths, and link `-lnvcomp`. The Nix development shell ships nvCOMP.

## How the GPU path loads files

The `gpu` backend sends a file to the host loader when:

- the file is FASTQ, BGZF, or has more than one gzip member
- a gzip member signature appears inside the compressed data, since the file may then hold extra members even if its first and last trailers match
- the inflated length or CRC32 disagrees with the gzip trailer

Host workers load the rejected files together, one row group at a time, and each file keeps its original reference ID. Host parsing runs between GPU batch submissions, so it overlaps GPU work that is already queued.

Six GPU lanes overlap file reads, inflation, and sketching. The builder splits free device memory between compressed and inflated buffers based on the input file sizes. After nvCOMP inflates a batch, CUB counts the sequence bytes each file keeps and compacts them into the buffer that held the compressed input. The memory budget covers both buffers.

Host workers compact and pack FASTA in cache-sized blocks. Neighbouring blocks overlap by `k - 1` bases, so k-mers that cross a line break survive. Separate records and runs of ambiguous bases never join.

Line scanning, FASTA compaction, base packing, and the gzip signature scan have AVX-512, AVX2, and AArch64 NEON versions, with portable fallbacks. On x86 the loader picks the widest version the CPU supports. AVX-512 byte compaction also needs VBMI2.

## How the coherent path loads files

On the `coherent` backend, CPU workers inflate into reusable pageable buffers and the GPU reads them:

- A mapping thread feeds the inflation workers through a bounded queue, at most eight files per worker ahead of the consumer.
- Six GPU batches copy and normalise FASTA asynchronously. Each finished load keeps its reference ID, so database row order does not depend on which file finishes first.
- CUDA 13.3 and newer copy with batched runtime copies and a compute-overlap hint. Older toolkits use CUB batched copies.
- Normalisation asks for streaming L2 eviction on the raw FASTA, within the device's access-policy window.
- FASTQ and files larger than one GPU batch go to the CPU parser. If normalisation runs out of header slots, the CPU parses the bytes it already inflated instead of reading the file again.
- A separate thread unmaps finished files from its own queue, so unmapping never blocks loading. A plain FASTA mapping stays alive until the GPU has read it.

On an error, cuDDL drains queued copies and sketch work before it frees any source buffer.

## Decode gzip faster on the host

cuDDL patches libdeflate for DNA, which is full of short matches:

- A short match copies at most two machine words when it fits.
- An 11-bit table decodes a short length and distance pair in one lookup.
- A literal and the short match after it share one table entry when their Huffman codes fit together.

Every other token, including long and overlapping matches, goes through the unchanged decoder with its offset and bounds checks. The patches accept any DEFLATE stream.

## Control how parsed bytes reach the GPU

`path_build_options::transfer` takes a `transfer_mode`. It applies only to bytes the CPU parsed. nvCOMP and GPU-normalised output already live on the device.

- `automatic` reads host buffers in place on coherent-memory GPUs and uses page-locked buffers elsewhere.
- `pinned` decompresses into page-locked buffers that the copy engine reads directly.
- `staged` decompresses into a heap buffer and copies it to the device.
- `in_place` lets the kernels read the heap buffer directly. Only coherent-memory GPUs support it.

The transfer mode never changes the decompression backend. Loaders reuse buffers in release order, and a loader waits until the GPU has read a buffer before it writes new bytes into it.

## Save and load a database

`build` returns a host-owned `reference_database_file` once GPU construction finishes. It holds each reference's winner scores, its label, and the build metadata, and it can outlive the build stream.

`save` writes to a temporary file next to the destination, then renames it over the destination:

- It preallocates the final size first, and falls back to checked writes where the filesystem can't preallocate.
- A failed allocation, write, or close leaves the old database untouched.
- The rename needs room for both files on disk.

The rename makes a new file appear all at once. It does not protect against power loss.

`load` needs no GPU. It checks the format version, the build metadata, every byte extent, and the checksum. Unknown formats, truncated or corrupt files, and trailing bytes return errors. The stored paths are labels only, and `load` never opens them.

`upload<K, BucketCount, Layout>` checks that the file matches the requested template arguments, then copies the scores and labels to the GPU. It returns once the copy finishes. Its stream must outlive the device database. It does not build an index.

## Understand databases and indexes

The database owns the reference rows. A `reference_index<K, BucketCount, Layout>` owns only acceleration data and neither copies nor owns the database. To use an index, pass a pointer to it to a search call. To search without one, pass `nullptr` or leave the argument out. Both paths apply the same threshold and return exact counts in reference order.

A database row stores one 16-bit winner score per bucket, so rows take `reference_count * bucket_count * 2` bytes.

On the GPU, each group of 32 buckets is stored as 16 bit-planes, one per score bit. One comparison then covers 32 buckets in a handful of logic operations. `copy_scores_async` decodes the planes back to row-major scores. Index construction decodes them too, into a temporary buffer, so it needs extra device memory. Posting offsets are 32-bit, and index construction rejects databases with more postings than that.

## Persist and reuse an index

Include `<cuddl/reference_index_file.cuh>`. With `database` from the first example:

```cpp
auto index = CUDDL_UNWRAP((cuddl::reference_index<25, 2048>::build_async(
    database, stream, cuddl::index_storage::sparse
)));
CUDDL_UNWRAP(cuddl::reference_index_file::save(index, database, "references.index", stream));
auto loaded = CUDDL_UNWRAP(cuddl::reference_index_file::load(
    "references.index", database, stream
));
```

`build_async` and `load` return the same move-only index type. `load` reads dense or sparse storage from the file header, so the file name doesn't matter. Before the index is usable, `load` checks:

- the version, extents, and CRC32
- that posting lists are sorted and agree with the database scores
- a seeded xxHash64 fingerprint of the database metadata, labels, and scores

Any mismatch returns an error. `load` uploads the validated arrays as they are and does not rebuild them.

An index belongs to the database it was built or loaded for. Moving that database keeps the link. Passing an index built for another database, or a moved-from index, returns an error. In a new process, load and upload the database first, then load its index. After that you can close or delete both files. The index's allocation stream must outlive the index, and outstanding GPU work must finish before you destroy the index or the database.

Index files cover databases saved by `reference_database_file`, with every bucket indexed and a 15-bit or 16-bit key mask. Indexes over fewer buckets stay in memory only. `save` publishes index files the same way as database files, through a temporary file and a rename.

The version-2 index file starts with `CUDDLIX\0`. A 32-bit version, a 32-bit storage kind (0 for dense, 1 for sparse), the 64-bit database fingerprint, and a 64-bit posting count follow. Then come dense offsets, posting IDs, and sparse keys, and a CRC32 at the end. Integers are little-endian.

```sh
cuddl-reference-index build references.cuddl -o references.index --format sparse
cuddl-reference-index search references.cuddl --index references.index --query query.fna
# The same search without the index:
cuddl-reference-index search references.cuddl --query query.fna
```

A single-query indexed search looks up the query's posting range in every bucket and prefix-sums the range lengths. It then spreads all postings evenly over the grid, so one long posting list can't hold up a single warp while the rest wait. Every search converts query rows to bit-planes in workspace you provide. For a single query, size it with `single_query_workspace_bytes()`. Batch search reuses its workspace across tiles.

## Read batch results

Batch and all-to-all searches process queries in tiles and pass each tile to your callback as a `cuddl::batch_result_tile`. The next tile reuses the same storage, so read each tile inside the callback. The tile points at device memory. `cuddl::download(tile, stream)` copies its passing results to pinned host memory. On the host copy, `tile().for_each_passing` visits them in query-major order, and `passing()` returns them as a `std::vector`:

```cpp
uint64_t hits = 0;
CUDDL_UNWRAP(database.search_batch_async(
    queries, compatibility, 0U, workspace, results,
    [&](cuddl::batch_result_tile const& tile) {
        auto host = CUDDL_UNWRAP(cuddl::download(tile, stream));
        host.tile().for_each_passing([&](cuddl::batch_search_result const& r) {
            hits += r.counts.equal >= 100U;
        });
    },
    {}, {.minimum_matches = 5U}, stream, &index
));
```

If you visit each result once, call `cuddl::for_each_passing(tile, stream, f)` instead of `download`. It copies the tile in chunks through two pinned buffers and runs `f` on one chunk while the next one copies, so it never holds the whole tile on the host. On RefSeq bacteria with 4,096 queries, visiting all 82 million exhaustive results this way took about 80 ms including the search, against about 97 ms through `download`.

To look up one pair, call `find(query_id, reference_id)` on `host.tile()` or on the device tile. It returns an empty optional if the pair did not pass. `passed`, `match_count`, and `count` answer the related questions. The tile's methods also work in device code.

Each result takes 8 bytes as a `cuddl::packed_pairwise_counts`. Its position implies the pair, and both-empty equals the bucket count minus the other three counts. Size result storage as `maximum_pair_count` elements of `database_type::batch_result_type`. The accessors unpack each result into a full `batch_search_result`.

On the device, every pair has a fixed slot. A tile holds one row of `reference_count` slots per query, or the upper triangle for all-to-all. A threshold search writes only the passing slots and marks them in a pass bitmap. `download` gathers the passing results on the device first, so only they cross the bus.

## Version-2 database file layout

All integers are unsigned and little-endian. Fields follow each other with no padding:

| Field                                                                    | Encoding                                                  |
| ------------------------------------------------------------------------ | --------------------------------------------------------- |
| Magic                                                                    | 8 bytes: `CUDDLDB` followed by a zero byte                |
| Version                                                                  | `uint32_t`, value 2                                       |
| K-mer length, bucket count, indexed bucket count, score encoder identity | Four `uint32_t` values                                    |
| Exponent bits, mantissa bits                                             | Two `uint16_t` values                                     |
| Hash identity, hash seed                                                 | `uint32_t`, `uint64_t`                                    |
| Canonicalization policy, blacklist identity, blacklist version           | `uint32_t`, `uint64_t`, `uint32_t`                        |
| Key mask, reference count                                                | `uint16_t`, `uint32_t`                                    |
| Labels in reference-ID order                                             | Each: `uint32_t` byte length followed by the path's bytes |
| Winner scores in reference-ID order                                      | `reference_count * bucket_count` `uint16_t` values        |
| Checksum                                                                 | `uint32_t` CRC-32 of every preceding byte                 |

Version 2 records the current hash, canonicalization, and score-encoder identities, with every bucket indexed and no blacklist. The key mask is `0x7fff` or `0xffff`, and the exponent and mantissa widths add up to 16.

To run the round-trip and malformed-input tests:

```sh
nix develop -c meson compile -C build test-cuddl
nix develop -c build/tests/test-cuddl --gtest_filter='ReferenceDatabaseFileTest.*'
```

## Measure build throughput

`cuddl-reference-build-benchmark` times FASTX parsing, GPU construction, and saving as CPU wall time. CUDA context creation falls outside the timing, and one warm-up run leaves the inputs in the page cache. The benchmark loads and validates the saved file after timing stops.

```sh
nix develop -c meson compile -C build cuddl-reference-build-benchmark
nix develop -c build/benchmarks/cuddl-reference-build-benchmark \
  --reference data/genomes/ecoli_k12_mg1655.fna \
  --database build/reference-benchmark.cuddl --copies 32 --workers 8 --samples 5
```

`--copies` repeats the inputs to make a larger synthetic collection. Its throughput does not predict a real RefSeq build, so measure the real collection on the target machine. Compare `--workers 1` with `--workers 8` to see how loading scales.

`--parse-only` times the serial CPU parser that the tests use as an oracle. The GPU builder never runs that parser. `--decompression` and `--transfer` take the same values as the build options. The JSON output records the inputs, repeat count, thread settings, samples, and median seconds.
