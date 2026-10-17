Build and save a reference database

`<cuddl/reference_database_file.cuh>` builds one GPU sketch per genome file and saves the sketches as a binary database. You can load that database later on any host, without the original genomes. The file API needs a POSIX host, because it publishes finished files with an atomic rename.

All integers in the binary formats below are unsigned and little-endian. Fields have no padding.

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

Loaders don't read their own files. Reader threads read whole files ahead of them, as many as the loaders but at least 64, because a parallel file system gives one node its bandwidth only with many reads in flight: on MOGON's Lustre, 16 threads read 2.9 GB/s of RefSeq gzip and 64 threads 6.7 GB/s. Loaders take files in the order they finish reading, so a slow read holds up only itself. The `gpu` backend reads its nvCOMP batches with the same depth. On an A100 node, 4,000 genomes that no node had cached took 10.3–11.6 s to sketch with 16 `cpu` loaders reading their own files, and 4.0–4.5 s with readers; `gpu` went from 9.8–10.5 s to 3.2–3.6 s and `hybrid` from 14.1 s to 2.1–2.2 s.

Before it reads anything, the `gpu` backend probes every file's gzip header and trailer. A probe is two small reads, so it waits on latency and moves almost no data, and 256 probes run at once. On an A100 node, 20,000 uncached genomes built in 13.9–15.6 s with 16 loaders and 64 probes in flight, and in 9.6–10.5 s with 256. Raising the whole-file read depth from 64 to 128 for the nvCOMP batches alone changed nothing (14.2–14.6 s).

### Reconstruct a RabbitTClust-style bacterial dataset

`scripts/reconstruct_refseq211.py` takes a directory, downloads missing NCBI current
and historical bacterial assembly tables, and writes `genomes.urls` alongside them.
It reuses existing tables without downloading them again. It approximates membership
at a cutoff, then applies the [RabbitTClust v2.0.0 row filter](https://raw.githubusercontent.com/RabbitBio/RabbitTClust/v2.0.0/benchmark/download/download_refseq.py)
to the old metadata columns. This can include Scaffold and Chromosome assemblies
with `genome_rep=Full`; it excludes rows containing `contig` anywhere in those columns.

Run from the repository root:

```bash
uv run scripts/reconstruct_refseq211.py data/refseq211-candidate
```

The script creates the directory if needed. Its three files are
`assembly_summary.txt`, `assembly_summary_historical.txt`, and `genomes.urls`.
Failed downloads do not leave partially written metadata files at those paths;
failed reconstruction preserves any existing URL list.

The default cutoff is `2022-03-07`, as specified in the
[release 211 notes](https://ftp.ncbi.nlm.nih.gov/refseq/release/release-notes/archive/RefSeq-release211.txt).
Use `--cutoff YYYY-MM-DD` for another date. Release dates are inclusive; assemblies
removed on or before the cutoff are excluded. Missing dates or FTP paths are
reported on stderr and excluded. Keep the two input tables with the URL list so
the selection can be reproduced after NCBI updates its tables.

This is not an exact reconstruction of the paper's 113,674 genomes. NCBI defines
[`seq_rel_date`](https://ftp.ncbi.nlm.nih.gov/genomes/README_assembly_summary.txt)
as the INSDC sequence-release date, not the date an assembly entered RefSeq.
Taxonomy, metadata and RefSeq inclusion may have changed since the cutoff. A
matching count does not prove identical membership. The script reports the count
difference but does not tune the cutoff or add genomes to force a match.

Download the candidate files separately; the script does not check every URL
or fetch genomes:

```bash
aria2c -c -j 8 -x 1 -s 1 \
  -d data/refseq211-candidate/genomes \
  -i data/refseq211-candidate/genomes.urls
uv run scripts/check_refseq211_reconstruction.py
```

The check exercises the CLI with isolated assembly tables, covering date boundaries,
historical records, the row filter, both header styles, reuse of existing metadata
and preservation of the URL list on malformed input.

To benchmark while the download runs, fix a genome set first.
`scripts/snapshot_genomes.py DIR COUNT OUT` links the first `COUNT` completed files
of `DIR`, in name order, into a new directory `OUT`. Files that still have an aria2
`.aria2` control file are skipped, and an existing `OUT` is an error, so the same
snapshot gives the same database later.

## Blacklist construction

BBTools DDL blacklists are FASTA sequence files, often gzip-compressed. Headers such as
`>kmer_HEX raw=N g=N ...` describe how a list was generated; construction ignores those
annotations. Each record contributes all valid k-mer windows. This also accepts BBTools'
`*_fused.fa.gz` lists, whose longer records compact multiple k-mers. Windows never cross
record boundaries or ambiguous bases. cuDDL uses the same A=0, C=1, T=2, G=3 encoding and
canonical maximum of the two strands. The older BBTools MinHash `.sketch` blacklists are
not DDL FASTA and are not accepted.

The loader also accepts one exact DNA k-mer per text line, case-insensitively, with blank
lines and CRLF allowed. Wrong lengths and non-ACGT bases in this form are errors. FASTA
lengths need not equal k, so choose a list generated for the intended k; a FASTA header
is not a reliable declaration of k. Reordering, duplicate entries, and reverse-complement
spelling do not change blacklist identity.

Upload once and reuse the lookup across builds or query batches:

```cpp
auto source = CUDDL_UNWRAP(cuddl::kmer_blacklist::load("blacklist.fa.gz", 25));
cuddl::device_blacklist filter(std::move(source), 2048, stream);
auto file = CUDDL_UNWRAP((cuddl::reference_database_file::build<25, 2048>(
    genomes, stream, {.blacklist = std::cref(filter)}
)));
CUDDL_UNWRAP(file.save("references.cuddl"));
```

The device owner keeps the normalized host list and a bucket-partitioned sorted device
lookup. Upload synchronizes its stream. Keep the owner and allocation stream alive until
all consuming kernels finish. `build`, `build_from_sequences`, `build_sketch_store`, and
the corresponding query builders accept an optional reference through their options.
Use `std::nullopt` for no blacklist. Owning `sketch` instances take an optional
`std::shared_ptr<device_blacklist const>` at construction and
retain that policy across additions and clears.

A loaded database exposes its embedded list through `file.blacklist()`. Construct a device
lookup from that list before sketching raw queries. The search CLI does this automatically,
without reopening the input blacklist. Search-only database uploads and all-to-all searches
do not allocate blacklist lookups. Query compatibility is instance metadata: pass
`queries.compatibility()` with `queries.scores()`. Incompatible query blacklists are rejected.
When adopting raw registers through `from_store`, explicitly supply the construction
blacklist after the stream argument, or `{}` for unfiltered registers. Raw register spans
cannot establish their own provenance. `sketch::assign_async` likewise requires the source
blacklist after its stream argument and rejects a different policy.

Filtering precedes register updates, so excluding a winner lets the next eligible k-mer
win. Both packed and raw-sequence construction skip membership checks when a score cannot
beat the CTA-local floor. Filtered raw-sequence construction computes that floor after 16
windows per bucket per CTA, and only when another tile remains. Short sequences avoid the
reduction. Unfiltered raw-sequence construction omits both the lookup and the floor.

### Compare cuDDL with and without the BBTools blacklist

The `blacklist` suite runs the same GPU sequence-construction benchmark with filtering
enabled and disabled. It uses the production launch policy and checks each result against
packed-k-mer construction with the same policy before timing. The default inputs are
E. coli K-12, WBcel235, and human chromosome 14. Repeat `--fastx` and `--label` to choose
other inputs. Both variants run in forward and reverse order.

```sh
uv run scripts/benchmark_cuddl_efficiency.py blacklist --samples 300 \
  --blacklist subprojects/bbmap/resources/refseqGenomeDDLBlacklist_k25e5b65536_fused.fa.gz \
  --output results/blacklist-comparison
uv run scripts/plot_blacklist_comparison.py results/blacklist-comparison
```

The runner preserves NVBench JSON, CSV, and logs plus a comparison manifest. The plotting
script writes PNG and PDF figures showing GPU construction time and the paired
blacklist/no-blacklist time ratio. Bars average the two run-order means; circle and square
markers show the forward and reverse means separately. Each mean contains the requested
number of NVBench samples. The plot rejects missing pairs, skipped states, mismatched
configurations, and mixed GPUs. Parsing, blacklist upload, and serialization are outside
these resident GPU timings. This measures filtering cost, not search accuracy.

On an RTX 5070 Ti with the bundled 5,944-entry blacklist, k=25, 2,048 buckets,
and the production launch policy, 200 samples per timing gave:

| Input               | No blacklist [ms] | BBTools blacklist [ms] | Time ratio |
| ------------------- | ----------------- | ---------------------- | ---------- |
| WBcel235            |            0.3983 |                 0.4229 |       1.06 |
| Human chromosome 14 |            0.3948 |                 0.4273 |       1.08 |

Both paths encode four ASCII bases at a time with CUDA bytewise comparisons and
integer dot products. They skip score calculations for hashes that cannot improve a
register. These ratios do not measure whole-pipeline overhead.

### Measure lookup and pruning

The encoder skips groups of eight windows when their shared span contains an ambiguous
base.

Single-sequence construction with or without a blacklist keeps each CTA's accepted
winners in shared memory. After tiles 4, 16, 32, and subsequent multiples of 16, CTAs
merge their winners through the output registers and take the smallest register as a
floor. The floor becomes a conservative upper bound on the hash magnitude. Comparing
the upper 32 bits rejects windows before score calculation or a
shared-memory load. Ties proceed to scoring. With a blacklist, a k-mer is looked up only
if its score also exceeds its bucket's current value. Only accepted keys reach the
registers, so pruning cannot discard a winner.

The lookup rejects most absent keys before it searches. A key whose hash is below the
smallest blacklisted hash in its bucket cannot be listed. A 2^19-bit presence map, indexed
by a second hash of the key, rejects most of the rest with one load. The remaining keys
go through a binary search of their bucket's sorted keys.

The efficiency benchmark checks register equality before timing sequence construction.
`Blacklist` loads a list once, outside the timed region. `BlocksPerSM=0` uses the
production launch policy. The `Floor` axis only changes fixed-grid launches with
`BlocksPerSM` above zero.

`blacklist_lookup` compares the bucketed lookup with a binary search of the whole sorted
list. It uses 1,651 and 23,020 keys, 0% or 50% hits, and 1,048,576 probes. On an RTX
5070 Ti, 30-sample GPU means were:

|   Keys | Hits | Bucketed [µs] | Whole-list search [µs] |
| ------ | ---- | ------------- | ---------------------- |
|  1,651 |   0% |          22.6 |                   28.7 |
|  1,651 |  50% |          26.6 |                   28.7 |
| 23,020 |   0% |          31.5 |                   73.1 |
| 23,020 |  50% |          48.0 |                   72.8 |

These measurements exclude parsing, upload, and database serialization.

```sh
./build/benchmarks/cuddl-efficiency-benchmark -b blacklist_lookup --no-batch \
  --stopping-criterion sample-count --min-samples 30 --target-samples 30
./build/benchmarks/cuddl-efficiency-benchmark -b sequence_construction \
  -a Path=data/genomes/WBcel235.fna -a Floor=1 \
  -a Blacklist=subprojects/bbmap/resources/refseqGenomeDDLBlacklist_k25e5b65536_fused.fa.gz \
  --no-batch --stopping-criterion sample-count --min-samples 30 --target-samples 30
```

## Choose a decompression backend

`path_build_options::decompression` picks one of five backends:

- `automatic` picks for you. Without nvCOMP it uses `cpu`. With nvCOMP, a coherent-memory GPU gets `hybrid` with more than one loader and `cpu` with one. A discrete GPU gets `cpu` with 48 or more loaders and `gpu` with fewer. The fastest backend depends on the GPU and the cores per GPU, so measure on your system. On an A100, 10,000 genomes took 2.2, 1.5, 1.3 and 1.5 s with `gpu` at 16, 32, 64 and 112 loaders, and 3.5, 1.75, 1.1 and 1.1 s with `cpu`. On an RTX 5070 Ti with a 12-core Ryzen 9 5900X, 4,000 genomes in the page cache took 1.9 s with `gpu` at 4 to 16 loaders, 2.7 s with `cpu` and 2.8 s with `hybrid` at 16.
- `cpu` inflates and parses every file on host workers. Sketching still runs on the GPU. Host workers read plain files into the reusable buffers that also hold inflated files, so the transfer mode applies to both kinds alike. When the bases are copied to the GPU rather than read in place, the workers pack them into two bits each first, so a quarter of the bytes cross the bus. Unpacked copies held an A100 at 2.0–2.2 s for those 10,000 genomes however many loaders ran. A worker inflates and packs a gzip file in its own reusable heap buffer and copies only the packed words into page-locked memory. The driver pins one allocation at a time: with 64 loaders on an A100, a new process that pinned whole inflated genomes built those genomes in 2.3 s, against 1.7 s with only the words pinned. For the same reason the buffers for packed words are pinned all at once before the loaders start, 2 MB each, enough for a bacterial genome. With 64 loaders on a Xeon w9-3595X, 20,000 genomes took 3.3 s when each loader pinned its own first buffer and 3.0 s with the reservation.
- `gpu` inflates eligible gzip files with nvCOMP and copies plain FASTA files to the GPU unchanged. The CPU still reads the files and hands anything the GPU can't take to the host loader.
- `gpu_cores` runs the `gpu` pipeline but makes nvCOMP inflate on the SMs even where the GPU has a hardware decompression engine. It exists to measure what the engine adds; on GPUs without one it matches `gpu`.
- `hybrid` inflates on CPU workers and copies the raw FASTA to the GPU, which strips headers and whitespace before sketching. It takes FASTA compaction, about a quarter of the host work, off the CPU, but on a discrete GPU it moves four times the bytes `cpu` does, since `cpu` packs the bases first. On an A40 node with 8 ranks of 16 loaders and 100,000 genomes in the page cache, before `cpu` packed its bases, `hybrid` sketched in 13.4 s, `cpu` in 15.1–15.6 s, and `gpu` in 16–25 s.

`reference_database_file::build`, `build_sketch_store`, and `query_sketch_batch::sketch` all take the same options:

```cpp
auto file = CUDDL_UNWRAP((cuddl::reference_database_file::build<25, 2048>(
    genomes, stream,
    {.parser_workers = 72, .decompression = cuddl::decompression_backend::cpu}
)));
```

`cuddl-build-reference-db`, `cuddl-reference-index`, and the reference-build benchmark take the same choice as `--decompression automatic|cpu|gpu|gpu-cores|hybrid`:

```sh
cuddl-build-reference-db genomes/ --workers 4 --decompression gpu
```

### Build with or without nvCOMP

Meson defaults to `-Dnvcomp=auto`, which enables GPU decompression when it finds the nvCOMP library and headers. Pass `-Dnvcomp=enabled` to require nvCOMP, or `-Dnvcomp=disabled` to build without it. CUDA 13.0 or newer is required either way. Without nvCOMP, explicit `gpu` and `hybrid` requests return an error, and `automatic` falls back to `cpu`.

Consumers of `cuddl_dep` get the right macro and link flags. If you include the headers directly, they default to CPU decompression. To turn on nvCOMP, compile every translation unit with `-DCUDDL_HAS_NVCOMP=1`, add nvCOMP's include and library paths, and link `-lnvcomp`. The Nix development shell ships nvCOMP.

## How the GPU path loads files

The `gpu` backend takes gzip files, BGZF files and plain FASTA files. A plain FASTA file is uncompressed and starts with `>`. BGZF, the blocked gzip that `bgzip` writes, records each block's size in its header, so the probe walks the blocks with one read each and the GPU inflates every block on its own, into the genome's slot. The backend sends a file to the host loader when:

- the file is FASTQ, or gzip with more than one member that is not BGZF
- an uncompressed file starts with anything but `>`, such as a blank line
- a gzip member signature appears inside the compressed data of a file that is not BGZF, since the file may then hold extra members even if its first and last trailers match
- the inflated length or CRC32 disagrees with the gzip trailer; for BGZF, with the sum of the blocks' lengths and the combination of their CRC32s

`reference_build_statistics::device_files` counts the files that took the GPU path, which `cuddl-build-reference-db` prints.

Host workers load the rejected files together, one row group at a time, and each file keeps its original reference ID. Host parsing runs between GPU batch submissions, so it overlaps GPU work that is already queued.

Six GPU lanes overlap file reads, inflation, and sketching. Each lane holds one batch, split between compressed and inflated buffers based on the input file sizes. A lane takes a 64th of the input but between 1 and 4 GiB. The compressed side is page-locked host memory, which takes about a second per 2.7 GB to pin: lanes sized from a B300's free memory pinned 48 GB in 17 s and split 10,000 genomes into only three batches. Only the first lane allocates before the device starts; each other lane allocates when it takes its first batch, so its pinning overlaps the batches already running. Plain files skip nvCOMP. Each batch puts them first, so one copy moves all of them into the inflated buffer. After nvCOMP inflates a batch, CUB counts the sequence bytes each file keeps and compacts them into the buffer that held the compressed input. The memory budget covers both buffers.

nvCOMP takes candidates from the smallest up, and the host loaders take the rest from the largest down. Before it fills a lane, the device checks that, at the rate it has inflated so far, it would finish what it holds plus the new batch before the loaders, at their rate, finish everything else. Without that check, nvCOMP filled every free lane and took 70% of the files even with 112 loaders on an A100, which held 10,000 genomes at 1.8–1.9 s from 32 loaders up, while `cpu` took 1.3 s. With the check and 1 GiB lanes, `gpu` took 2.2 s with 16 loaders, 1.6 s with 32 and 1.44–1.5 s with 48 or more.

On GPUs with a hardware decompression engine (B200, B300, GB200, GB300), the engine takes gzip files that inflate to at most its limit (4 MiB on a B300) from buffers allocated for it. cuDDL allocates the compressed, inflated and table buffers from such a pool and calls the driver's `cuMemBatchDecompressAsync` directly for those files, with each file's DEFLATE stream found by skipping its gzip header in the host copy; nvCOMP inflates the larger files on the SMs. nvCOMP's own engine path builds the request on the device and synchronises the stream to read it back, which stalled the thread that drives all six lanes for up to 117 ms per batch. Batches for the engine alternate with batches of larger files in proportion to their gzip bytes, so both run at once in different lanes. Mixing the two kinds within one batch was slower: nvCOMP inflates one file per thread block, and a batch with few large files leaves most SMs idle. On a B300, 2,000 genomes within the limit inflated at 57.6 GB/s of gzip on the engine against 7.3 GB/s on the SMs.

End to end, the engine changes little, because the largest genomes bound the build. `--decompression gpu-cores` runs the same pipeline with every file on the SMs. Building the 10,000-genome snapshot on one B300 from node-local NVMe took a median 4.48 s with `gpu` and 4.84 s with `gpu-cores` with 16 loaders, but 5.37 against 4.72 s with 4 loaders, where the engine pool's slower allocation weighs more. In those builds an SM inflation kernel ran for 16.5 of the 18.4 s after the first one started, while the engine was busy for 0.52 s: only 45% of the files and 29% of the gzip bytes (on the 10,000-genome snapshot) fit the engine, and every SM batch lasts as long as its largest genome. `scripts/plot_decompression_engine.py` plots the `RESULT` lines of such a run per backend and loader count.

The SM batches are what larger lanes speed up. nvCOMP gives each file one thread block, so a batch of a few hundred large genomes leaves most of a B300's SMs idle. All 115,223 genomes (142 GB of gzip), in the page cache with 16 loaders, took 24.7–24.9 s with `gpu` and 22.5–22.8 s with `gpu-cores` with fixed 1 GiB lanes, 19.2–20.1 s with 2 GiB and 16.7–17.6 s with lanes sized from the input (4 GiB here). 8 GiB lanes were no faster (18.1 s). The 10,000-genome snapshot stays at 1 GiB lanes and took 4.5–5.3 s either way. On the CPU, `cpu` and `hybrid` took 45–50 s for the whole collection with 16 loaders and 168–194 s with 4.

Recompressing the input as BGZF lets every block fit the engine, but with 4 GiB lanes it no longer pays. On the same 115,223 genomes with 16 loaders, BGZF written by `bgzip` (zlib) took 14.7–17.8 s with `gpu` and 18.9–20.5 s with `gpu-cores`, against 15.3–18.4 s and 15.8–17.9 s for the gzip files as distributed. With 1 GiB lanes, BGZF took 14.4 s with `gpu` against 20.5 s for the gzip files: BGZF filled the SMs then because every block is a member of its own. BGZF compressed with libdeflate built in 16.7 s with `gpu`. The CPU paths are slower on BGZF: `cpu` took 54.0 s on zlib BGZF and 82.3 s on libdeflate BGZF against 47.1 s on gzip.

`cuddl-inflate-benchmark` times inflation alone: nvCOMP on the SMs against the engine, over the same members resident in device memory, the members whose output fits the engine's limit. Pass the input directories as `-a "input=[DIR,...]"`; `scripts/plot_inflate_engine.py` plots its `--csv` output. On one B300 with the 10,000-genome snapshot, the engine inflated 211 GB/s of the gzip files of at most 4 MiB against 31 GB/s on the SMs, 169 against 27 GB/s of libdeflate-compressed BGZF and 190 against 37 GB/s of zlib-compressed BGZF: 5.1–6.8 times faster. In BGZF every genome fits, as 660,411 blocks; of the gzip files as distributed, 4,535 of 10,000 fit, 29% of the bytes.

Host workers compact and pack FASTA in cache-sized blocks. Neighbouring blocks overlap by `k - 1` bases, so k-mers that cross a line break survive. Separate records and runs of ambiguous bases never join.

Line scanning, FASTA compaction, base packing, and the gzip signature scan have AVX-512, AVX2, and AArch64 NEON versions, with portable fallbacks. On x86 the loader picks the widest version the CPU supports. AVX-512 byte compaction also needs VBMI2.

## How the hybrid path loads files

On the `hybrid` backend, CPU workers inflate into reusable host buffers and the GPU copies them:

- A mapping thread feeds the inflation workers through a bounded queue, at most eight files per worker ahead of the consumer.
- Six GPU batches copy and normalise FASTA asynchronously. Each finished load keeps its reference ID, so database row order does not depend on which file finishes first.
- Batched runtime copies move the FASTA with a compute-overlap hint.
- Normalisation asks for streaming L2 eviction on the raw FASTA, within the device's access-policy window.
- FASTQ and files larger than one GPU batch go to the CPU parser. If normalisation runs out of header slots, the CPU parses the bytes it already inflated instead of reading the file again.
- A separate thread unmaps finished files from its own queue, so unmapping never blocks loading. A plain FASTA mapping stays alive until the GPU has read it.

On an error, cuDDL drains queued copies and sketch work before it frees any source buffer.

## Decode gzip faster on the host

cuDDL patches libdeflate for DNA, which is full of short matches:

- A short match copies at most two machine words when it fits.
- An 11-bit table decodes a short length and distance pair in one lookup.
- A literal and the short match after it share one table entry when their Huffman codes fit together. The decoder writes that literal without a branch, because whether an entry carries one is close to random in DNA. On RefSeq genomes this alone raised single-thread inflation from 1.49 to 1.66 GB/s on a desktop CPU and from 1.40 to 1.79 GB/s on a Xeon w9-3595X.

Every other token, including long and overlapping matches, goes through the unchanged decoder with its offset and bounds checks. The patches accept any DEFLATE stream.

## Control how parsed bytes reach the GPU

`path_build_options::transfer` takes a `transfer_mode`. It applies only to bytes the CPU parsed. nvCOMP and GPU-normalised output already live on the device.

- `automatic` reads host buffers in place on coherent-memory GPUs and uses page-locked buffers elsewhere.
- `pinned` decompresses into page-locked buffers that the copy engine reads directly.
- `staged` decompresses into a heap buffer and copies it to the device.
- `in_place` lets the kernels read the heap buffer directly. Only coherent-memory GPUs support it.

The transfer mode never changes the decompression backend. Loaders reuse buffers in release order. A buffer returns once the GPU has copied it; when kernels read buffers in place, a loader waits for those kernels before it writes new bytes into a buffer.

Page-locked buffers come from pools that only their device can access. CCCL's default pinned pool maps every allocation into every visible GPU: with 8 ranks per A40 node, each seeing all 8 GPUs, a loader buffer allocation took 27 ms instead of 1.3 ms, and loaders spent most of their time waiting on the driver. Buffers the device writes, such as downloaded results, sit on the device's host NUMA node: with 8 ranks, downloads took 0.39 s per sample there against 1.3 s from the default pool. Buffers that loader threads fill use any node, since the loaders run on every node: device-node loader buffers made the 100,000-genome sketch slower (18.6 against 15.1 s with `cpu`).

Ranks of the end-to-end pipeline that saw only their own GPU sketched 100,000 genomes in 11.8–12.3 s against 13.4 s with all GPUs visible; `scripts/own-gpu.sh` sets `CUDA_VISIBLE_DEVICES` to the OpenMPI node-local rank. Do not use it for runs with fewer ranks than GPUs: rank placement then ignores the PCIe topology, and two ranks downloading through one PCIe switch took 1.5 s per sample against 0.81 s on separate switches.

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

A database row stores one 16-bit winner score per bucket, so rows take `reference_count * bucket_count * 2` bytes. `build_async` also groups near-duplicate rows so exhaustive searches can skip about half of the full comparisons. On RefSeq bacteria this takes about 7% more device memory than the rows. See [near-duplicate references in exhaustive search](near-duplicate-references.md).

On the GPU, each group of 32 buckets is stored as 16 bit-planes, one per score bit. One comparison then covers 32 buckets in a handful of logic operations. `copy_scores_async` decodes the planes back to row-major scores. Index construction decodes them too, into a temporary buffer, so it needs extra device memory. Posting offsets are 32-bit, and index construction rejects databases with more postings than that.

## Persist and reuse an index

Include `<cuddl/reference_index_file.cuh>`. With `database` from the first example:

```cpp
auto index = CUDDL_UNWRAP((cuddl::reference_index<25, 2048>::build_async(database, stream)));
CUDDL_UNWRAP(cuddl::reference_index_file::save(index, database, "references.index", stream));
auto loaded = CUDDL_UNWRAP(cuddl::reference_index_file::load(
    "references.index", database, stream
));
```

`build_async` defaults to `index_storage::automatic`, which compares the bytes of dense offsets against sparse keys and their lookup directory and builds the smaller one. Pass `index_storage::dense` or `index_storage::sparse` to choose. `build_async` and `load` return the same move-only index type. `load` reads dense or sparse storage from the file header, so the file name doesn't matter. Before the index is usable, `load` checks:

- the version, extents, and CRC32
- that posting lists are sorted and agree with the database scores
- a seeded xxHash64 fingerprint of the database metadata, labels, and scores

Any mismatch returns an error. `load` uploads the validated arrays as they are and does not rebuild them.

An index belongs to the database it was built or loaded for. Moving that database keeps the link. Passing an index built for another database, or a moved-from index, returns an error. In a new process, load and upload the database first, then load its index. After that you can close or delete both files. The index's allocation stream must outlive the index, and outstanding GPU work must finish before you destroy the index or the database.

Index files cover databases saved by `reference_database_file`, with every bucket indexed and a 15-bit or 16-bit key mask. Indexes over fewer buckets stay in memory only. `save` publishes index files the same way as database files, through a temporary file and a rename.

The index fingerprint covers database metadata, labels, and scores.

```text
byte           0               1               2               3
       ┌───────────────┬───────────────┬───────────────┬───────────────┐
    0  │      'C'      │      'U'      │      'D'      │      'D'      │
       ├───────────────┼───────────────┼───────────────┼───────────────┤
    4  │      'L'      │      'I'      │      'X'      │      00       │
       ├───────────────┴───────────────┴───────────────┴───────────────┤
    8  │                          version = 1                          │
       ├───────────────────────────────────────────────────────────────┤
   12  │              storage kind: 0 = dense, 1 = sparse              │
       ├───────────────────────────────────────────────────────────────┤
   16  │                  database fingerprint [31:0]                  │
       ├───────────────────────────────────────────────────────────────┤
   20  │                 database fingerprint [63:32]                  │
       ├───────────────────────────────────────────────────────────────┤
   24  │                    posting count P [31:0]                     │
       ├───────────────────────────────────────────────────────────────┤
   28  │                    posting count P [63:32]                    │
       ├───────────────────────────────────────────────────────────────┤
   32  │                     pair-work sum [31:0]                      │
       ├───────────────────────────────────────────────────────────────┤
   36  │                     pair-work sum [63:32]                     │
       └───────────────────────────────────────────────────────────────┘
```

`B` is the database's indexed bucket count, `R` is its reference count, and `C = B * (key_mask + 1)`.

```text
Dense (kind = 0)
             ┌───────────────────────────────────────────────────┐
          40 │          offsets: C + 1 uint32_t values           │
             ├───────────────────────────────────────────────────┤
  40+4*(C+1) │          posting IDs: P uint32_t values           │
             ├───────────────────────────────────────────────────┤
40+4*(C+1+P) │      CRC32 of every preceding byte: uint32_t      │
             └───────────────────────────────────────────────────┘

Sparse (kind = 1, P = B * R)
             ┌───────────────────────────────────────────────────┐
          40 │          posting IDs: P uint32_t values           │
             ├───────────────────────────────────────────────────┤
      40+4*P │              keys: P uint16_t values              │
             ├───────────────────────────────────────────────────┤
      40+6*P │      CRC32 of every preceding byte: uint32_t      │
             └───────────────────────────────────────────────────┘
```

In dense storage, cell `bucket * (key_mask + 1) + key` uses the half-open posting range `[offsets[cell], offsets[cell + 1])`. Empty scores have no postings. In sparse storage, each bucket has `R` entries sorted by key, then reference ID, including empty scores. The loader builds the lookup directory and posting bitmaps.

```sh
cuddl-reference-index build references.cuddl -o references.index
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

To download tile after tile, create one `cuddl::host_batch_results host(stream, pool)` and call `cuddl::download(tile, stream, host)` for each tile. Its pinned buffers grow only when a tile needs more. With several GPUs, put each GPU's pool on its host NUMA node. `replicated_database` below does this for you. On an 8-GPU A40 node, downloads into the default pool, which lives on NUMA node 0, ran 21 times slower with 8 GPUs because every copy crossed the socket interconnect.

If you visit each result once, call `cuddl::for_each_passing(tile, stream, f)` instead of `download`. It copies the tile in chunks through two pinned buffers and runs `f` on one chunk while the next one copies, so it never holds the whole tile on the host. On RefSeq bacteria with 4,096 queries, visiting all 82 million exhaustive results this way took about 80 ms including the search, against about 97 ms through `download`.

To look up one pair, call `find(query_id, reference_id)` on `host.tile()` or on the device tile. It returns an empty optional if the pair did not pass. `passed`, `match_count`, and `count` answer the related questions. The tile's methods also work in device code.

Each result takes 8 bytes as a `cuddl::packed_pairwise_counts`. Its position implies the pair, and both-empty equals the bucket count minus the other three counts. Size result storage as `maximum_pair_count` elements of `database_type::batch_result_type`. The accessors unpack each result into a full `batch_search_result`.

On the device, every pair has a fixed slot. A tile holds one row of `reference_count` slots per query, or the upper triangle for all-to-all. A threshold search writes only the passing slots and marks them in a pass bitmap. `download` gathers the passing results on the device first, so only they cross the bus.

## Search all-to-all on several GPUs

Include `<cuddl/replicated_database.cuh>`. A `cuddl::replicated_database<K, BucketCount, Layout>` keeps one replica of a database on each device you name. It searches all-to-all with all of them at once:

```cpp
std::vector<cuda::device_ref> devices{cuda::devices[0], cuda::devices[2]};
auto group = CUDDL_UNWRAP(
    (cuddl::replicated_database<K, BucketCount>::upload(file, devices))
);
CUDDL_UNWRAP(group.load_index("references.index"));  // optional
std::atomic<uint64_t> pairs = 0;
CUDDL_UNWRAP(group.search_all_to_all(
    [&](cuddl::replica_tile const& tile) {
        pairs += CUDDL_UNWRAP(tile.download())->passing().size();
    },
    {.minimum_matches = 20}
));
```

`adopt` takes databases you already built, one per device, instead of a file. The devices may differ in architecture, memory size and speed. Each replica sizes its own storage, and replicas claim query tiles from a shared queue until it runs dry, largest tiles first. A faster GPU therefore claims more tiles, and nothing assumes a node layout. Each replica gets its own stream, and its downloads land in pinned memory on its GPU's host NUMA node.

Each replica calls your callback from its own thread, so the callback must be thread-safe. `tile.replica` gives the replica's position in the device list. Pass `{.in_query_order = true}` as the last argument to get tiles one at a time in query order instead. Replicas then wait for their turn, which costs some overlap, but output written from the callback no longer depends on the devices. If the callback throws, the other replicas stop and the exception reaches the caller.

To split one search across processes, such as MPI ranks, give each its own group and pass `{.shard = {.index = rank, .count = ranks}}`. The parts cover every pair exactly once. They take tiles in turn, reversing direction each round (0, 1, 2, 2, 1, 0, ...), so the early, larger tiles of the triangle spread over all parts. This fixed split suits processes of equal speed. A single-database `search_all_to_all_async` takes the same `all_to_all_shard`, or an `all_to_all_tile_queue` for your own scheduling.

When the processes differ in speed, for example different GPU models or ranks with different numbers of GPUs, also pass a `next_tile` function that every process shares. It returns the next unclaimed tile number of the whole search, counting from 0. Each process then claims tiles one at a time, largest first, until none are left, so a faster process does more of the search, as a faster GPU does within one group. `shard.count` must still give the number of processes, because it sets the tile size. `<cuddl/distributed.cuh>` provides `cuddl::shared_tile_counter` for MPI ranks (see below). `in_query_order` cannot be combined with `next_tile`, because another process may hold the next tile.

Every public function that takes a stream runs on that stream's device, whatever device the calling thread has selected.

```sh
cuddl-reference-index search references.cuddl --all-to-all --minimum-matches 0 --devices 0 1 2 3 -o pairs.bin
```

`--devices` takes GPU ordinals and writes tiles in query order, so its output matches a single-GPU run byte for byte.

### Sketch across nodes

Include `<cuddl/distributed.cuh>` and link `cuddl_distributed_dep` to split sketching across the ranks of an MPI communicator. The header needs MPI; `cuddl.cuh` does not include it. Every rank makes the same three calls:

```cpp
auto slice = CUDDL_UNWRAP(cuddl::slice_genomes("genomes/", MPI_COMM_WORLD));
auto mine = CUDDL_UNWRAP(
    (cuddl::reference_database_file::build<K, BucketCount>(slice.mine(), stream))
);
auto file = CUDDL_UNWRAP(cuddl::gather_sketches(mine, slice, MPI_COMM_WORLD));
```

`slice_genomes` takes a directory or a text file with one path per line. Rank 0 lists it with `cuddl::genome_paths`, which never looks up the files it lists. Every rank then reads the sizes of an equal share of the files, using several threads, because each lookup on a parallel file system waits for a metadata round trip. Each rank gets a contiguous run of genomes holding about the same number of bytes. On 100,000 RefSeq genomes on Lustre, rank 0 alone took about 12 s to list and size the files when they were cached, and about 65 s when they were not.

`gather_sketches` sends every rank's rows to every rank over MPI. It checks that all ranks sketched with the same k, buckets, layout and blacklist. The result is the file a single-process build of `slice.paths` gives, with the same reference IDs and labels, so you can `save` it or upload it.

To search the file across the ranks, upload it to a `replicated_database` on each rank and call `cuddl::search_all_to_all` with the communicator on every rank. The ranks share tiles through a counter the call creates, and every pair reaches `on_tile` on exactly one rank. Every GPU holds every row, so the collection must fit in each GPU's memory: `BucketCount` × 2 bytes per genome (about 4 GB per million genomes with 2,048 buckets), plus the index and search storage:

```cpp
auto group = CUDDL_UNWRAP((cuddl::replicated_database<K, BucketCount>::upload(file, devices)));
CUDDL_UNWRAP(cuddl::search_all_to_all(group, on_tile, MPI_COMM_WORLD));
// With an index: group.build_index(), then
CUDDL_UNWRAP(cuddl::search_all_to_all(group, on_tile, {.minimum_matches = 20}, MPI_COMM_WORLD));
```

Code that runs many short searches can create one `shared_tile_counter` and pass it as `replica_search::next_tile`, calling `reset()` between searches, to avoid creating the MPI window each time.

Rank 0 holds the count in an MPI window, and each claim is one remote fetch-and-add, which takes about 20 µs across nodes while a tile takes milliseconds. Open MPI completes a claim from another node only while rank 0 is inside an MPI call, even with InfiniBand hardware atomics: while rank 0 waited on its GPU, claims from the other node took 70–130 ms, and at 12 GPUs on two A100 nodes, six ranks claimed almost no tiles. Rank 0 therefore runs a thread that polls MPI without sleeping (it yields the core to other threads) while the counter exists; with a 20 µs sleep between polls, claims took 10–21 ms. Initialise MPI with `MPI_Init_thread` and `MPI_THREAD_MULTIPLE`. A counter serves one search; call `counter.reset()` on every rank before the next. Making, resetting and destroying the counter are collective. A failed claim aborts the communicator, because a lost tile would silently drop pairs.

If one rank fails inside `slice_genomes` or `gather_sketches`, it returns its error and every other rank returns an error naming it, so no rank waits forever in a collective call. A rank that fails between the calls, for example while sketching, must still make the next call or abort the communicator.

### Benchmark across GPUs and nodes

`cuddl-multi-gpu-all-to-all-benchmark` runs one MPI rank per GPU. The first rank of each node reads the database file into MPI shared memory, and every rank of the node uploads its rows from there. With 100,000 genomes (410 MB of rows) on an 8-GPU A40 node, setup then took 650–715 ms per rank (load 382 ms, upload 94–162 ms, build and index about 170 ms). With rank 0 broadcasting the rows, the broadcast alone took 620–780 ms; with every rank reading the file itself, loads took 170–630 ms, because 8 reads and 16 copies of 410 MB compete for host memory bandwidth. One launch sweeps rank counts by splitting the world communicator. For each count it first downloads every result, checks the pair total, an order-independent fingerprint against the first count, and a spread of pairs against a host oracle. NVBench then times the search, with results kept on the GPU (`device`) or downloaded per tile (`host`). Each sample starts after a barrier and ends when the slowest rank is done.

Each rank holds a one-device `replicated_database` and searches its shard. Ranks on one node take GPUs farthest-first by NVML topology, so runs with fewer ranks than GPUs use separate PCIe switches. CPU binding is left to the launcher.

```sh
mpirun -n 8 build/benchmarks/cuddl-multi-gpu-all-to-all-benchmark references.cuddl \
  --ranks 1 2 4 8 --samples 5 -o all-to-all.json
```

For each rank count, the report also holds results from untimed runs:

- `rank_timeline`: when each of a rank's tiles finished on its GPU and how long the rank then waited at the closing barrier. `load_balance` gives the slowest and mean rank time.
- `phases`: each rank's search time with downloads, the part spent inside `download()` after its tile's search finished, and its wait.
- `communication`: the time and bandwidth of an MPI broadcast of the database over the ranks of that count, and MPI barrier latency.

`setup` records every rank's load, upload, build and index times. The phase figure of `scripts/plot_multi_gpu_scaling.py` shows them per rank, and below them each rank's search with results kept on the GPU next to the same search with every result copied to the host. The copies overlap the search of later tiles, so the difference is what downloading costs: with 8 GPUs on one node, 190 against 389 ms.

`--weak` runs weak scaling: P ranks search n·√(P/P_max) references, evenly spaced over the file, so every GPU gets about the same number of pairs. Spacing keeps the share of near-duplicate references close to the whole file's. That share sets the cost per pair: on the 100,000-genome RefSeq snapshot, the first 25,000 genomes searched at 6.67 Gpairs/s on an RTX 5070 Ti and every fourth genome at 4.64 Gpairs/s, because 58% and 37% of their rows were near-duplicates. `--thresholds 0 5 20 ... --write-dir DIR` sweeps the minimum index matches at `--sweep-ranks` ranks. Threshold 0 reports every pair without the index. Each threshold records the pairs emitted, the bytes written to `DIR` as raw 24-byte records, and the search, download and write times of the slowest rank. Point `DIR` at node-local storage.

`scripts/plot_multi_gpu_scaling.py` takes one or more reports and draws:

- speedup and efficiency (strong and weak scaling);
- per-rank Gantt charts and max/mean rank time;
- per-rank phases and communication costs;
- the threshold sweep;
- per-GPU throughput, with `--roofline-pairs-per-second` as an optional measured ALU ceiling (see below).

Each run is compared with the 1-GPU run of its own report. When reports repeat a point, the first one wins.

### Run the whole pipeline on several nodes

`cuddl-distributed-all-to-all-benchmark` goes from compressed FASTA files to filtered results with one rank per GPU, using `<cuddl/distributed.cuh>`. Ranks list and slice the genomes with `slice_genomes`, sketch their slices, gather all rows with `gather_sketches`, and upload the result to their GPU. Each rank then builds an index for a threshold search, searches with `search_all_to_all`, and writes the tiles it claimed to `rank-<r>.bin` in `--output-dir`. Barriers separate the phases (`list`, `sketch`, `allgather`, `build`, `index`, `search`), and every rank records its time per phase. The reference IDs match `cuddl-build-reference-db` on the same directory.

`--records similarity` (the default) writes each pair's weighted k-mer identity, `equal / max(equal + min(lower, higher), 6)` (`cuddl::weighted_kmer_identity`), as a 4-byte float. An exhaustive search computes it on the GPU and writes each tile as its first query ID and query count (two `uint32`) followed by one float per pair, query-major, each query against every later reference; with 100,000 genomes that is 20 GB instead of 120 GB. A threshold search writes 12-byte records of query ID, reference ID and float. `--records counts` writes the 24-byte records of `cuddl-reference-index search --all-to-all`, with the raw bucket counts. On 3,000 genomes the two agree exactly, and every-pair download plus write took 31 against 173 ms.

```sh
mpirun -n 16 build/benchmarks/cuddl-distributed-all-to-all-benchmark genomes/ \
  --minimum-matches 20 --output-dir /localscratch/out -o end-to-end.json
```

`scripts/plot_distributed_end_to_end.py` draws per-rank phases and wall time per phase. `scripts/compare_cpu_gpu.py` lists these reports next to the SimdSketch runs of `scripts/cpu-baseline.sh` on the same genomes. `scripts/simdsketch-nodes.sh` runs SimdSketch across the first N nodes of a job, split the way cuDDL splits the pipeline: every node sketches a contiguous slice, the sketch files are exchanged through Lustre, and every node compares an interleaved share of the rows (`simdsketch-benchmark compare --shard I --shards N`) and writes it to node-local scratch. `scripts/mgpu-eval.sh` and `scripts/sketch-scaling.sh` run it at every node count after cuDDL, with as many threads per node as the node's cuDDL ranks had cores and with the page cache their runs warmed. Both tools sketch with k=25 and 2048 buckets. SimdSketch writes one 4-byte similarity per pair and has no threshold.

Both tools inflate gzip on the CPU unless cuDDL uses nvCOMP, and inflating dominates sketching. On one A40 node (8 GPUs, 128 EPYC 7713 cores) with 100,000 genomes in the page cache, SimdSketch sketched in 27.2 s. cuDDL, with 8 ranks of 16 loaders each seeing only its own GPU, sketched in 11.8–12.3 s with `--decompression hybrid` and 14.5–15.2 s with `cpu`. On the `cpu` path cuDDL spends 66% of its host time in libdeflate and 27% compacting FASTA, while the GPU sketch kernels take 0.4 s of a 6.6 s single-rank build of 12,500 genomes; `hybrid` moves the compaction to the GPU. nvCOMP (`gpu`) inflates one gzip file per thread block and reached about 2.6 GB/s of gzip per A40, no faster than the 16 cores each rank has.

`scripts/sketch-scaling.sh` runs the end-to-end pipeline with one GPU per node: strong scaling with 100,000 genomes and weak scaling with 25,000 genomes per node, 4 runs per configuration. Weak runs below 100,000 genomes read `genomes-<N>k-spread`, genomes evenly spaced over `genomes-100k` (`scripts/snapshot_genomes.py --spread`). The first 25,000 accessions average 1.14 MB of gzip per genome, the whole set 1.24 MB. `scripts/plot_sketch_scaling.py` plots its reports as phase wall times per node count, strong scaling above weak scaling, and as sketch efficiency in compressed input bytes per second per node. Given the SimdSketch reports too, it draws a hatched SimdSketch bar (sketch, gather, compare and write) next to each cuDDL bar.

`scripts/all_to_all_roofline.py` reads an Nsight Compute export of one exhaustive search. It prints instructions and bytes per pair, and the issue, ALU, FMA and DRAM ceilings for a GPU.

The search's work is integer ALU instructions, about 80% `LOP3`, which issues at half a warp instruction per SM sub-partition per clock on both the A40 and the RTX 5070 Ti. `alu-power-ceiling` measures how many ALU warp instructions per second a GPU sustains with that mix under its power limit: it saturates the ALU pipes on every SM for `--seconds`, leaves out `--warmup-seconds`, and samples clock, power and clock-limit reasons through NVML. Dividing its rate by the search's ALU warp instructions per pair gives a pairs/s ceiling that needs no assumed clock. On an A40 at its 300 W limit it drew 253–260 W, never hit the power limit, held 1,740 MHz and sustained 2.84 × 10¹¹ ALU warp instructions/s. With the 59.9 ALU instructions per pair Nsight Compute counted on an RTX 5070 Ti for the 100,000-genome snapshot, that is 4.75 Gpairs/s per A40. The search itself reaches the power limit and drops to about 1,660 MHz, so the power drawn by its other work (shared-memory, cache and DRAM traffic) keeps it below that ceiling. The instruction count depends on the data, because near-duplicate references are cheaper, and comes from Blackwell machine code, which may differ from Ampere's.

Meson builds the target only when it finds MPI; the Nix shell provides it. Two options help on clusters whose compute nodes have no internet access or whose CUDA toolkit ships only stub driver libraries:

- `-Drapids_cmake_source=DIR` points NVBench at a pre-fetched rapids-cmake checkout. NVBench loads rapids-cmake before CMake's `project()`, so `CPM_SOURCE_CACHE` does not cover it.
- `-Ddriver_library_dir=DIR` names the directory holding the real `libcuda.so` and `libnvidia-ml.so`. Without it, CMake can pick the toolkit's stubs, and the stub directory lands in the binaries' RPATH, where it hides the driver at run time.

## Database file layout

```text
byte           0               1               2               3
       ┌───────────────┬───────────────┬───────────────┬───────────────┐
    0  │      'C'      │      'U'      │      'D'      │      'D'      │
       ├───────────────┼───────────────┼───────────────┼───────────────┤
    4  │      'L'      │      'D'      │      'B'      │      00       │
       ├───────────────┴───────────────┼───────────────┼───────────────┤
    8  │          version = 1          │ k-mer length  │ exponent bits │
       ├───────────────────────────────┼───────────────┴───────────────┤
   12  │        bucket count B         │           key mask            │
       ├───────────────────────────────┴───────────────────────────────┤
   16  │                       reference count R                       │
       ├───────────────────────────────────────────────────────────────┤
   20  │                       blacklist count N                       │
       ├───────────────────────────────────────────────────────────────┤
   24  │                   blacklist identity [31:0]                   │
       ├───────────────────────────────────────────────────────────────┤
   28  │                  blacklist identity [63:32]                   │
       └───────────────────────────────────────────────────────────────┘
```

```text
             ┌───────────────────────────────────────────────────┐
          32 │        blacklist: N uint64_t packed k-mers        │
             ├───────────────────────────────────────────────────┤
      32+8*N │      label 0: uint32_t L[0], then L[0] bytes      │
             ├───────────────────────────────────────────────────┤
             │                         ⋮                         │
             ├───────────────────────────────────────────────────┤
             │   label R-1: uint32_t L[R-1], then L[R-1] bytes   │
             ├───────────────────────────────────────────────────┤
           S │           scores: R * B uint16_t values           │
             ├───────────────────────────────────────────────────┤
     S+2*R*B │      CRC32 of every preceding byte: uint32_t      │
             └───────────────────────────────────────────────────┘

S = 32 + 8*N + sum(4 + L[i], i = 0 .. R-1)
```

Labels have no terminating zero byte. Scores use row-major order, with `[reference_id][bucket]` at byte `S + 2 * (reference_id * B + bucket)`.

`kmer_length` ranges from 1 through 31. `exponent_bits` ranges from 1 through 15. `bucket_count` is 2,048, 4,096, or 8,192. `key_mask` is `0x7fff` or `0xffff`.

The loader requires sorted, unique, canonical blacklist keys and verifies `blacklist_identity`. Empty blacklists have identity zero.

To run the round-trip and malformed-input tests:

```sh
nix develop -c meson compile -C build test-cuddl
nix develop -c build/tests/test-cuddl --gtest_filter='ReferenceDatabaseFileTest.*'
nix develop -c meson compile -C build test-blacklist
nix develop -c build/tests/test-blacklist
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

### What bounds a build

The limit differs by machine, so profile before tuning:

- **GH200, local NVMe:** the disk. It reads 5.1 GiB/s, and a cold build runs at about 4.9 GB/s of gzip. cuFile in compatible mode read no faster than `pread`.
- **Desktop with an RTX 5070 Ti and 16 cores:** nvCOMP, which is busy for 97% of the loop. On the host, `perf` puts 49% of the time in the kernel copy from the page cache into pinned lanes and 10% in the gzip signature scan. Reader counts, lane counts and sizes, and `O_DIRECT` (1.9 s to 3.2 s) gained nothing.
- **monster3, CPU only:** inflation, with `deflate_decompress_bmi2` at 65% of the profile.
- **MOGON A100 nodes, Lustre:** the latency of opening and probing many files. Deeper probing took 20,000 uncached genomes from 13.8–18.6 s to 9.7–11.5 s with `gpu` and 16 loaders, and from 14.1–15.4 s to 9.8–10.6 s with 64. `cpu` stayed within noise. cuFile read small files as fast as POSIX with half the CPU time, but no faster.
- **B300:** the SM batches of large gzip files, as described under the decompression engine.

