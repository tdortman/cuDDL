#!/bin/bash
# Ablation study on one GH200 (72 Grace cores, local NVMe) over a whole genome collection: builds
# the ablation variants, the database the search suite reads, and a 1,000-genome hashed sample
# for the single-threaded micro suite, then runs the micro, build, search and end-to-end suites
# of scripts/ablation.py into results/ablation-gh200.jsonl. MPI must be on PATH. Run from the
# project root; the meson options configure both builds:
#   scripts/ablation_gh200.sh GENOMES [MESON OPTIONS...]
source .envrc
set -euo pipefail
export PS4="+ [\$(date +%T)] "
set -x
genomes=$(realpath "$1")
shift
scratch=build/ablation/scratch
out=results/ablation-gh200.jsonl
label="GH200"
workers=64
mkdir -p $scratch/out results

scripts/ablation_builds.sh "$(nproc)" "$@"
[ -f $scratch/db.cuddl ] ||
  build/ablation/main-build/examples/cuddl-build-reference-db "$genomes" -o $scratch/db.cuddl \
    --workers $workers
[ -d $scratch/genomes-1k ] ||
  uv run --offline scripts/snapshot_genomes.py "$genomes" 1000 $scratch/genomes-1k

abl() { uv run --offline scripts/ablation.py "$@" --build-dir build/ablation/main-build -o $out; }
# One rank on one GPU: the tile queue, the progress thread, the registration cache and the
# per-rank GPU and pinned-pool choices have nothing to act on, so those ablations are left out.
abl micro $scratch/genomes-1k --stock-build build/ablation/stock-build --label "$label"
abl build "$genomes" --workers $workers --scratch $scratch --label "$label"
abl search $scratch/db.cuddl --write-dir $scratch/out --launcher "mpirun -n 1" --reps 3 \
  --label "$label" \
  --only "Near-duplicate deltas" --only "Bit-sliced scores" --only "LOP3 plane kernel" \
  --only "Inverted index" --only "Posting bitmaps" --only "Index fallback to planes" \
  --only "Low-plane match filter" --only "NUMA-local download pools"
abl e2e "$genomes" --output-dir $scratch/out --launcher "mpirun -n 1" --own-gpu "" \
  --workers $workers --stock-build build/ablation/stock-build --label "$label" \
  --only "GPU decompression (nvCOMP)" --only "Index filtering on the GPU" \
  --only "GPU/host co-scheduling" --only "Read depth (gpu)" --only "Read depth (cpu)" \
  --only "Probe depth" --only "Input-sized lanes" --only "Pre-pinned packed buffers" \
  --only "SIMD FASTX parsing" --only "Two-bit base packing" \
  --only "Branchless k-mer encoding (gpu)" --only "Branchless k-mer encoding (cpu)" \
  --only "NUMA-local download pools" --only "Near-duplicate deltas" --only "Bit-sliced scores" \
  --only "LOP3 plane kernel" --only "Inverted index" --only "Posting bitmaps" \
  --only "Index fallback to planes" --only "Low-plane match filter" \
  --only "Similarity records on the GPU" --only "Parallel result streams" \
  --only "Direct result writes" --only "DNA-tuned libdeflate"
