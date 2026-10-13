#!/bin/bash
# SimdSketch all-to-all on the first NODES nodes of this allocation, split the way cuDDL's
# distributed pipeline splits it: every node sketches a contiguous slice of the genomes with
# THREADS threads, the sketch parts are exchanged through Lustre, and every node compares the
# genomes r with r % NODES == node against all later genomes and writes their similarities to
# node-local scratch. SimdSketch has no thresholded output, so every pair is written. Each step
# reports its slowest node; the compare step reports that node's comparing (SimdSketch's own
# timer around the similarity loop) and the rest of its run, mostly writing the similarities.
# Prints one JSON line.
# Usage: scripts/simdsketch-nodes.sh GENOMES NODES THREADS
set -euo pipefail
shopt -s inherit_errexit
genomes=$1 nodes=$2 threads=$3
SS=build/benchmarks/simdsketch-benchmark
# SimdSketch picks its SIMD paths for the compiling CPU. The crates are vendored, so cargo needs
# no network. The nodes have no clang, so link with gcc.
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc ninja -C build benchmarks/simdsketch-benchmark > /dev/null
shared=build/scratch/simdsketch-$SLURM_JOB_ID
local=/localscratch/$SLURM_JOB_ID/simdsketch
rm -rf "$shared"
mkdir -p "$shared"
ls "$genomes" | sort | awk -v dir="$genomes" '{ print dir "/" $0 }' > "$shared/all.txt"
count=$(wc -l < "$shared/all.txt")
input_bytes=$(xargs -a "$shared/all.txt" stat -L -c %s | awk '{ s += $1 } END { print s }')
# Runs a command and prints its seconds.
timed() {
  local t
  t=$(date +%s.%N)
  "$@" > /dev/null
  awk -v a="$t" -v b="$(date +%s.%N)" 'BEGIN { print b - a }'
}
export -f timed
export SS shared local nodes threads count
# Every step runs on the same nodes: srun -N alone may pick other nodes of the job per step.
hosts=$(scontrol show hostnames | head -n "$nodes" | paste -sd,)
# Runs a shell command once per node with the node's cores. Every node prints one number per
# name in NAMES; prints `"NAME": seconds, ` for each, from the node with the largest sum.
# srun checks -c against the job's SLURM_CPUS_PER_TASK; the step's tasks get the node's cores.
step() {
  local seconds
  seconds=$(SLURM_CPUS_PER_TASK=$threads srun -N "$nodes" -w "$hosts" --ntasks="$nodes" --ntasks-per-node=1 \
    -c "$threads" bash -euo pipefail -c "$2")
  [ "$(wc -l <<< "$seconds")" -eq "$nodes" ]
  awk -v names="$1" '{ s = 0; for (i = 1; i <= NF; i++) s += $i; if (NR == 1 || s > best) { best = s; line = $0 } }
    END { n = split(names, k, " "); split(line, v, " "); for (i = 1; i <= n; i++) printf "\"%s\": %.3f, ", k[i], v[i] }' \
    <<< "$seconds"
}
sketch=$(step sketch_s 'mkdir -p $local
  awk -v i=$SLURM_PROCID -v n=$nodes -v c=$count "NR > int(i * c / n) && NR <= int((i + 1) * c / n)" \
    $shared/all.txt > $local/part.txt
  # Untimed: the node reads its slice once, so the sketch reads from the page cache like the
  # cuDDL runs it is compared with.
  xargs -a $local/part.txt -P 16 -n 64 cat > /dev/null
  timed $SS -j $threads sketch -k 25 -s 2048 -b 8 --output $shared/part-$SLURM_PROCID.sk $local/part.txt')
gather=$(step gather_s 'timed cp $shared/part-*.sk $local/')
compare=$(step "compare_s write_s" 'total=$(timed env CUDDL_RESIDENT_TIMINGS=$local/resident.json $SS -j $threads compare \
    --references $(for j in $(seq 0 $((nodes - 1))); do echo $local/part-$j.sk; done) \
    --shard $SLURM_PROCID --shards $nodes --output $local/pairs.f32)
  grep -oE "\"resident_ms\":[0-9.eE+-]+" $local/resident.json | cut -d: -f2 |
    awk -v t=$total "{ printf \"%f %f\n\", \$1 / 1000, t - \$1 / 1000 }"
  rm -f $local/pairs.f32 $local/part* $local/resident.json')
rm -rf "$shared"
wall=$(echo "$sketch$gather$compare" | grep -oE '[0-9]+\.[0-9]+' | awk '{ s += $1 } END { printf "%.3f", s }')
printf '{"tool": "simdsketch", "genomes": %d, "nodes": %d, "threads": %d, "input_bytes": %d, %s%s%s"compare_bytes": %d, "wall_s": %s}\n' \
  "$count" "$nodes" "$threads" "$input_bytes" "$sketch" "$gather" "$compare" "$((count * (count - 1) / 2 * 4))" "$wall"
