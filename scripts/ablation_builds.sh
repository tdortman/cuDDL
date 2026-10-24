#!/bin/bash
# Configures and builds the two ablation variants under build/ablation, both with the
# CUDDL_ABLATE switches compiled in: main-build from this tree, and stock-build from a copy of it
# whose libdeflate has the DNA patch reverted. Run from the project root:
#   scripts/ablation_builds.sh JOBS [MESON OPTIONS...]
set -euo pipefail
jobs=$1
shift
root=$PWD
targets=(cuddl-build-reference-db cuddl-multi-gpu-all-to-all-benchmark
  cuddl-distributed-all-to-all-benchmark cuddl-ablation-micro-benchmark)

# The copy shares every file but libdeflate's through hard links.
stock=build/ablation/stock-src
if [ ! -d $stock ]; then
  mkdir -p $stock
  for entry in meson.build meson.options include benchmarks examples tests scripts subprojects; do
    cp -al "$entry" "$stock/$entry"
  done
  rm -rf $stock/subprojects/libdeflate
  cp -a subprojects/libdeflate $stock/subprojects/libdeflate
  git -C $stock/subprojects/libdeflate checkout -- lib
fi
for entry in meson.build meson.options include benchmarks examples; do
  rsync -a --delete "$entry" "$stock/"
done
# A build left by a job on another GPU model takes this run's options, its -gencode among them.
setup() { # BUILD_DIR SOURCE_DIR
  if [ -d "$1" ]; then
    meson configure "$1" -Dablations=true "${options[@]}"
  else
    (cd "$2" && meson setup "$root/$1" -Dablations=true "${options[@]}")
  fi
}
options=("$@")
setup build/ablation/main-build .
setup build/ablation/stock-build $stock
for b in build/ablation/main-build build/ablation/stock-build; do
  ninja -C $b build.ninja
  meson compile -C $b -j "$jobs" "${targets[@]}"
done
