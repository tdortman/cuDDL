#!/bin/bash
# SimdSketch all-to-all on one node: sketch every genome, then compare every pair, with the
# settings the micro-comparison uses (k=25, 2048 buckets of 8 bits). SimdSketch has no
# thresholded output; compare writes one 32-bit similarity per pair. Prints one JSON line per
# genome count. scripts/simdsketch-nodes.sh runs the same comparison across several nodes.
[ -n "${SLURM_JOB_ID:-}" ] && source .envrc
set -euo pipefail
SS=${SS:-build/benchmarks/simdsketch-benchmark}
if [ -n "${SLURM_JOB_ID:-}" ] && [ ! -x "$SS" ]; then
  # Built on the CPU node itself: SimdSketch picks its SIMD paths for the compiling CPU. The
  # crates are vendored, so cargo needs no network. The node has no clang, so link with gcc.
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc ninja -C build benchmarks/simdsketch-benchmark
fi
GENOMES=${GENOMES:-data/refseq211/mgpu/genomes-30k}
SIZES=${SIZES:-1000 3000 10000 20000 30000}
THREADS=${THREADS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
S=${SCRATCH:-/localscratch/${SLURM_JOB_ID:-local}}/cpu-baseline
mkdir -p "$S"
now() { date +%s.%N; }
seconds() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }
for n in $SIZES; do
  list="$S/genomes-$n.txt"
  ls "$GENOMES" | sort | awk -v n="$n" -v dir="$GENOMES" 'NR <= n { print dir "/" $0 }' > "$list"
  [ "$(wc -l < "$list")" -eq "$n" ] || { echo "Error: fewer than $n genomes in $GENOMES"; exit 1; }
  input_bytes=$(xargs -a "$list" stat -L -c %s | awk '{ s += $1 } END { print s }')
  t0=$(now)
  "$SS" -j "$THREADS" sketch -k 25 -s 2048 -b 8 --output "$S/sketch.sk" "$list" > /dev/null
  t1=$(now)
  "$SS" -j "$THREADS" compare --references "$S/sketch.sk" --output "$S/pairs.f32" > /dev/null
  t2=$(now)
  printf '{"tool": "simdsketch", "genomes": %d, "threads": %d, "input_bytes": %d, ' \
    "$n" "$THREADS" "$input_bytes"
  printf '"sketch_s": %s, "compare_s": %s, "compare_bytes": %d, "host": "%s"}\n' \
    "$(seconds "$t0" "$t1")" "$(seconds "$t1" "$t2")" "$(stat -c %s "$S/pairs.f32")" "$(hostname -s)"
  rm -f "$S/sketch.sk" "$S/pairs.f32"
done
