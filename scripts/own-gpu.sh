#!/bin/bash
# Runs one MPI rank with only its node-local GPU visible: the local rank's entry among the job's
# GPUs that CUDA can use. A GPU in MIG mode without instances (one per A100 node on MOGON) is
# skipped. With every GPU visible, page-locked allocations are mapped into all of them, which
# slowed sketching on 8-GPU nodes.
# UUIDs, because nvidia-smi's indices follow PCI order and CUDA's need not.
mapfile -t gpus < <(nvidia-smi --query-gpu=uuid,mig.mode.current --format=csv,noheader |
  awk -F', ' '$2 != "Enabled" { print $1 }')
export CUDA_VISIBLE_DEVICES=${gpus[$OMPI_COMM_WORLD_LOCAL_RANK]}
exec "$@"
