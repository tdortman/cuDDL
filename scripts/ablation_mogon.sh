#!/bin/bash
# Ablation study on whole MOGON GPU nodes. SUITES (default: all) picks the steps, so the study can
# also run as several jobs within ki-gpu-devel's two hours:
#   compile  builds the ablation variants through scripts/ablation_builds.sh;
#   micro, build, search, e2e  run that suite of scripts/ablation.py.
# REPS overrides the repetition count for all measurement suites.
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
  uv run --offline scripts/snapshot_genomes.py $mgpu/genomes-100k 1000 $mgpu/genomes-1k-sample
if [ ! -e $mgpu/refseq211-100k.cuddl ]; then
  build/ablation/main-build/examples/cuddl-build-reference-db $mgpu/genomes-100k -o "$S/db.cuddl" \
    --workers "$SLURM_CPUS_ON_NODE"
  cp "$S/db.cuddl" $mgpu/refseq211-100k.cuddl
fi
per_node() {
  srun -N "$nodes" --ntasks="$nodes" -c "${SLURM_CPUS_PER_TASK:-1}" \
    -o "logs/%x-%j-%s.log" -e "logs/%x-%j-%s.log" "$@"
}
if want e2e || want search; then
  per_node bash -c '
    set -euo pipefail
    usable=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader |
      awk '"'"'$0 != "Enabled" { ++n } END { print n+0 }'"'"')
    if (( usable < $1 )); then
      echo "$(hostname): $usable usable GPUs, need $1" >&2
      exit 1
    fi
    mkdir -p "$2/out" "$2/sweep"
    cp "$3" "$2/db.cuddl"
  ' bash "$SLURM_NTASKS_PER_NODE" "$S" "$mgpu/refseq211-100k.cuddl"
fi
export UCX_RCACHE_ENABLE=n
UCX_NET_DEVICES=$(for d in $(ibstat -l); do
  port=$(ibstat "$d" 1)
  if [[ $port == *"State: Active"* && $port == *"Link layer: InfiniBand"* ]]; then echo "$d:1"; fi
done | paste -sd,)
export UCX_NET_DEVICES
launcher="mpirun -n $ranks --bind-to none -x LD_LIBRARY_PATH -x PATH -x UCX_RCACHE_ENABLE -x UCX_NET_DEVICES"
abl() { uv run --offline scripts/ablation.py "$@" --build-dir build/ablation/main-build -o $out; }
# The micro and build suites run one process. It sees only the first GPU CUDA can use: every
# visible GPU slows sketching, and a GPU in MIG mode without instances is unusable. awk reads
# every line: exiting early would end nvidia-smi with SIGPIPE, which pipefail turns into a failure.
one_gpu=$(nvidia-smi --query-gpu=uuid,mig.mode.current --format=csv,noheader |
  awk -F', ' '$2 != "Enabled" && !found++ { print $1 }')

if want micro; then
  CUDA_VISIBLE_DEVICES=$one_gpu abl micro $mgpu/genomes-1k-sample \
    --stock-build build/ablation/stock-build --reps "${REPS:-3}" --label "$label"
fi
# A build is one process, so it gets the whole node.
if want build; then
  CUDA_VISIBLE_DEVICES=$one_gpu abl build $mgpu/genomes-100k --workers "$SLURM_CPUS_ON_NODE" \
    --scratch "$S" --reps "${REPS:-3}" --label "$label"
  # The same builds reading every genome from Lustre.
  CUDA_VISIBLE_DEVICES=$one_gpu abl build $mgpu/genomes-100k --workers "$SLURM_CPUS_ON_NODE" \
    --scratch "$S" --label "$label, Lustre" --uncached --reps "${REPS:-2}" \
    --only "GPU decompression (nvCOMP)" --only "GPU/host co-scheduling" \
    --only "Read depth (gpu)" --only "Read depth (cpu)" --only "Probe depth"
fi
if want search; then
  abl search "$S/db.cuddl" --write-dir "$S/sweep" --launcher "$launcher" --ranks "$ranks" \
    --reps "${REPS:-2}" --label "$label"
fi
# The ranks of a node share all of its cores: one per GPU need not divide them, so each rank's
# loader count rounds up and no core stays idle.
if want e2e; then
  per_node uv run --offline scripts/warm_page_cache.py "$mgpu/genomes-100k" --threads 16
  rank_workers=$(((SLURM_CPUS_ON_NODE + SLURM_NTASKS_PER_NODE - 1) / SLURM_NTASKS_PER_NODE))
  abl e2e $mgpu/genomes-100k --output-dir "$S/out" --launcher "$launcher" \
    --workers "$rank_workers" --stock-build build/ablation/stock-build --reps "${REPS:-2}" \
    --label "$label"
fi
