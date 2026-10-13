#!/bin/bash
# Multi-GPU evaluation inside a Slurm job with one rank per usable GPU: strong scaling with a
# threshold sweep, weak scaling, then the end-to-end pipeline (cold, then warm at every node
# count), then SimdSketch at every node count. SIZE picks data/refseq211/mgpu/genomes-SIZE and
# builds its database if missing. STRONG=0, WEAK=0, E2E=0 and SIMDSKETCH=0 skip parts. Writes
# results/mogon-<gpu>-TAG-*.json.
source .envrc
set -euo pipefail
export PS4="+ [\$(date +%T)] "
set -x
SIZE=${SIZE:-30k}
TAG=${TAG:-eval}
S=/localscratch/$SLURM_JOB_ID
mgpu=data/refseq211/mgpu
rpn=$SLURM_NTASKS_PER_NODE
nodes=$SLURM_JOB_NUM_NODES
# One rank per usable GPU of this job; on nodes shared with other jobs that is fewer than the node
# has, and a GPU in MIG mode without instances is allocated but unusable.
job_gpus=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -cv Enabled)
if [ "$rpn" -ne "$job_gpus" ]; then
  echo "Error: run one rank per usable GPU (--ntasks-per-node=$job_gpus)"
  exit 1
fi
# Regenerate first: meson compile looks up targets before it notices a changed meson.build.
ninja -C build build.ninja
meson compile -C build -j "$SLURM_CPUS_ON_NODE" cuddl-multi-gpu-all-to-all-benchmark cuddl-distributed-all-to-all-benchmark cuddl-build-reference-db
if [ ! -e "$mgpu/refseq211-$SIZE.cuddl" ]; then
  ./build/examples/cuddl-build-reference-db "$mgpu/genomes-$SIZE" -o "$S/db.cuddl" --workers "$SLURM_CPUS_ON_NODE"
  cp "$S/db.cuddl" "$mgpu/refseq211-$SIZE.cuddl"
fi
# UCX's registration cache took 31 s for the first 410 MB MPI_Bcast across two nodes and 0.8 s
# without it; later broadcasts were also faster without it (165 against 220 ms).
export UCX_RCACHE_ENABLE=n
# Only active InfiniBand ports: A100 nodes also have two Ethernet ports that are down.
UCX_NET_DEVICES=$(for d in $(ibstat -l); do
  port=$(ibstat "$d" 1)
  if [[ $port == *"State: Active"* && $port == *"Link layer: InfiniBand"* ]]; then echo "$d:1"; fi
done | paste -sd,)
export UCX_NET_DEVICES
run() { mpirun -n "$1" --bind-to none -x LD_LIBRARY_PATH -x PATH -x UCX_RCACHE_ENABLE -x UCX_NET_DEVICES "${@:2}"; }
# End-to-end ranks see only their own GPU: with all 8 visible the 100k sketch took 13.4 s instead
# of 11.8-12.3 s. The search benchmark keeps every GPU visible: it runs 2 or 4 ranks per node and
# picks GPUs on separate PCIe switches, which downloads need (808 against 1526 ms for 2 ranks).
own_gpu=scripts/own-gpu.sh
filter() { grep -aE '^rank 0|verification|load_balance|threshold|genomes=|Error' | cut -c1-400; }
bench=./build/benchmarks/cuddl-multi-gpu-all-to-all-benchmark
e2e=./build/benchmarks/cuddl-distributed-all-to-all-benchmark
# Result names carry the GPU model: a40, a100, ...
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | grep -oiE 'a[0-9]+' | tr A-Z a-z)
out=results/mogon-$gpu-$TAG-$SIZE-${nodes}x$rpn
# The sweep writes every rank's results to node-local NVMe.
srun -N "$nodes" --ntasks="$nodes" bash -c 'echo "$(hostname -s) GPUs $CUDA_VISIBLE_DEVICES"'
srun -N "$nodes" --ntasks="$nodes" mkdir -p "$S/sweep" "$S/e2e"
# The first rank of every node reads the database from node-local storage.
srun -N "$nodes" --ntasks="$nodes" cp "$mgpu/refseq211-$SIZE.cuddl" "$S/db.cuddl"
if [ "${STRONG:-1}" = 1 ]; then
  run "$SLURM_NTASKS" $bench "$S/db.cuddl" --samples 10 --warmups 2 \
    --thresholds 0 5 10 20 50 100 200 500 --write-dir "$S/sweep" -o "$out-strong.json" 2>&1 | filter
fi
if [ "${WEAK:-1}" = 1 ]; then
  run "$SLURM_NTASKS" $bench "$S/db.cuddl" --weak --samples 10 --warmups 2 -o "$out-weak.json" 2>&1 | filter
fi
[ "${E2E:-1}" = 1 ] || exit 0
# With 16 cores per A40, hybrid decompression sketched 100k genomes fastest (12 s; cpu 15 s,
# gpu 16-25 s); automatic picks gpu on discrete GPUs.
e2e="$own_gpu $e2e --decompression hybrid"
# The first read of the genomes comes from Lustre; later runs find them in the page cache.
run "$SLURM_NTASKS" $e2e "$mgpu/genomes-$SIZE" --minimum-matches 20 --output-dir "$S/e2e" \
  -o "results/mogon-$gpu-$TAG-e2e-$SIZE-${nodes}x$rpn-m20-cold.json" 2>&1 | filter
# The cold run read only each node's share; every node reads all genomes once so that runs on
# fewer nodes find their larger shares cached too.
srun -N "$nodes" --ntasks="$nodes" bash -c \
  "find -L '$mgpu/genomes-$SIZE' -type f -print0 | xargs -0 -P 16 -n 64 cat > /dev/null"
for n in $(seq 1 "$nodes"); do
  for m in 20 0; do
    run $((n * rpn)) $e2e "$mgpu/genomes-$SIZE" --minimum-matches $m --output-dir "$S/e2e" \
      -o "results/mogon-$gpu-$TAG-e2e-$SIZE-${n}x$rpn-m$m-warm.json" 2>&1 | filter
  done
done
# SimdSketch on 1..N nodes, with the cores the cuDDL ranks had and the page cache they warmed.
if [ "${SIMDSKETCH:-1}" = 1 ]; then
  for n in $(seq 1 "$nodes"); do
    scripts/simdsketch-nodes.sh "$mgpu/genomes-$SIZE" "$n" $((rpn * SLURM_CPUS_PER_TASK)) \
      > "results/mogon-$gpu-$TAG-simdsketch-$SIZE-${n}x$rpn.json"
  done
fi
