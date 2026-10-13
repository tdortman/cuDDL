#!/bin/bash
# Strong (100k genomes) and weak (25k genomes per node) scaling of the end-to-end pipeline, one
# A40 per node. Weak runs below 100k read genomes-<N>k-spread, genomes evenly spaced over
# genomes-100k, so every size has the same mix of genomes. Each configuration runs 4 times:
# run r0 reads part of its input from Lustre for the first time, r1-r3 find it in the page cache.
# After each of r1-r3, SimdSketch runs on the same nodes and cores (SIMDSKETCH=0 skips it): the
# nodes are shared with other jobs, and alternating the tools exposes both to the same load.
# With 64 cores per GPU, the cpu backend sketched 100k genomes in 21.7 s against 29.5-30.3 s for
# hybrid, whose GPU-side FASTA normalisation becomes the bottleneck. MINIMUM (default 20) sets
# --minimum-matches; 0 reports and writes every pair, as SimdSketch does.
source .envrc
set -euo pipefail
export PS4="+ [\$(date +%T)] "
set -x
mgpu=data/refseq211/mgpu
S=/localscratch/$SLURM_JOB_ID
nodes=$SLURM_JOB_NUM_NODES
ninja -C build build.ninja
meson compile -C build -j "$SLURM_CPUS_PER_TASK" cuddl-distributed-all-to-all-benchmark
srun -N "$nodes" --ntasks="$nodes" mkdir -p "$S"
run() {
  for rep in 0 1 2 3; do
    mpirun -n "$1" --map-by ppr:1:node --bind-to none -x LD_LIBRARY_PATH -x PATH \
      ./build/benchmarks/cuddl-distributed-all-to-all-benchmark "$mgpu/genomes-$4" \
      --minimum-matches "${MINIMUM:-20}" --decompression cpu --output-dir "$S" \
      -o "results/mogon-a40-sketch-$3-$2-${1}x1-m${MINIMUM:-20}-r$rep.json" 2>&1 | grep -aE 'genomes=|Error' | cut -c1-400
    if [ "${SIMDSKETCH:-1}" = 1 ] && [ "$rep" -gt 0 ]; then
      scripts/simdsketch-nodes.sh "$mgpu/genomes-$4" "$1" "$SLURM_CPUS_PER_TASK" \
        > "results/mogon-a40-sketch-$3-$2-${1}x1-simdsketch-r$rep.json"
    fi
  done
}
# STRONG=0 runs only the weak configurations.
for n in $(seq 1 "$nodes"); do
  if [ "${STRONG:-1}" = 1 ]; then run "$n" 100k strong 100k; fi
  if [ "$n" -lt 4 ]; then run "$n" "$((25 * n))k" weak "$((25 * n))k-spread"; fi
done
