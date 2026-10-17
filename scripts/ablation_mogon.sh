#!/bin/bash
# Ablation study on whole MOGON GPU nodes. SUITES (default: all) picks the steps, so the study can
# also run as several jobs within ki-gpu-devel's two hours:
#   compile  builds the ablation variants through scripts/ablation_builds.sh;
#   micro, build, search, e2e  run that suite of scripts/ablation.py.
# Each job writes results/ablation-mogon-<job>.jsonl. Submit from the project root.
source .envrc
set -euo pipefail
export PS4="+ [\$(date +%T)] "
set -x
mgpu=data/refseq211/mgpu
S=/localscratch/$SLURM_JOB_ID
nodes=$SLURM_JOB_NUM_NODES
ranks=$SLURM_NTASKS
out=results/ablation-mogon-$SLURM_JOB_ID.jsonl
# The variants are built for this node's GPU, and the results carry its model.
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d .)
label="MOGON $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | grep -oiE 'a[0-9]+')"
suites=" ${SUITES:-compile micro build search e2e} "
want() { [[ $suites == *" $1 "* ]]; }
if want compile; then
  scripts/ablation_builds.sh "$SLURM_CPUS_ON_NODE" -Dwrap_mode=default -Dnvcomp=enabled \
    -Drapids_cmake_source=/home/tdortman/cpm-deps/rapids-cmake-src -Ddriver_library_dir=/usr/lib64 \
    "-Dcuda_args=-gencode=arch=compute_$cc,code=sm_$cc"
fi

[ -d $mgpu/genomes-1k-sample ] ||
  uv run --offline scripts/snapshot_genomes.py $mgpu/genomes-100k 1000 $mgpu/genomes-1k-sample --sample
srun -N "$nodes" --ntasks="$nodes" mkdir -p "$S/out" "$S/sweep"
srun -N "$nodes" --ntasks="$nodes" cp "$mgpu/refseq211-100k.cuddl" "$S/db.cuddl"
export UCX_RCACHE_ENABLE=n
UCX_NET_DEVICES=$(for d in $(ibstat -l); do
  port=$(ibstat "$d" 1)
  if [[ $port == *"State: Active"* && $port == *"Link layer: InfiniBand"* ]]; then echo "$d:1"; fi
done | paste -sd,)
export UCX_NET_DEVICES
launcher="mpirun -n $ranks --bind-to none -x LD_LIBRARY_PATH -x PATH -x UCX_RCACHE_ENABLE -x UCX_NET_DEVICES"
abl() { uv run --offline scripts/ablation.py "$@" --build-dir build/ablation/main-build -o $out; }

if want micro; then
  abl micro $mgpu/genomes-1k-sample --stock-build build/ablation/stock-build --label "$label"
fi
if want build; then
  abl build $mgpu/genomes-25k-sample --workers "$SLURM_CPUS_PER_TASK" --scratch "$S" --label "$label"
  abl build $mgpu/genomes-25k-sample --workers "$SLURM_CPUS_PER_TASK" --scratch "$S" --uncached \
    --reps 2 --label "$label, Lustre" --only "Read depth (gpu)" --only "Read depth (cpu)" \
    --only "Probe depth" --only "GPU/host co-scheduling" --only "GPU decompression (nvCOMP)"
fi
if want search; then
  abl search "$S/db.cuddl" --write-dir "$S/sweep" --launcher "$launcher" --ranks "$ranks" --reps 2 \
    --label "$label"
fi
if want e2e; then
  abl e2e $mgpu/genomes-100k --output-dir "$S/out" --launcher "$launcher" \
    --workers "$SLURM_CPUS_PER_TASK" --decompression hybrid --stock-build build/ablation/stock-build \
    --reps 2 --label "$label"
fi
