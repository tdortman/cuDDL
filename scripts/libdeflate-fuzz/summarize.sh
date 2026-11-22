#!/bin/bash
# Summarise one campaign directory: summarize.sh <work-dir> <run-name> <binary> <sanitizers> <seed-dir-name>
# Prints one JSON object. Minimises the corpus into <work-dir>/corpus/<run-name> (kept) and reads coverage from the merge.
set -euo pipefail
W=$(realpath "$1"); R=$2; BIN=$3; SAN=$4; export TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
D=$W/run/$R
mkdir -p "$W/corpus/$R"; rm -rf "$W/corpus/$R"/*
merge=$("$W/build/$BIN" -merge=1 -max_len=65536 "$W/corpus/$R" "$D/corpus" 2>&1 | grep -E "^#[0-9]+\s+DONE|MERGE-OUTER: .*(new|corpus)" | tail -2)
execs=$(cat "$D"/logs/*/fuzz-*.log | awk '/^Done [0-9]+ runs in/{s+=$2} END{print s+0}')
secs=$(cat "$D"/logs/*/fuzz-*.log | awk '/^Done [0-9]+ runs in/{if($5>m)m=$5} END{print m+0}')
cov=$(echo "$merge" | grep -oE "cov: [0-9]+ ft: [0-9]+" | tail -1)
crashes=$( (ls "$D"/crash-* "$D"/oom-* "$D"/timeout-* "$D"/leak-* 2>/dev/null || true) | wc -l)
n=$(ls "$W/corpus/$R" | wc -l)
printf '{"harness":"%s","sanitizers":"%s","seconds":%s,"execs":%s,"execs_per_sec":%s,"crashes":%s,"differential_mismatches":%s,"corpus_size":%s,"cov":"%s"}\n' \
  "$R" "$SAN" "$secs" "$execs" "$(( secs>0 ? execs/secs : 0 ))" "$crashes" \
  "$( (grep -l 'DIFFERENTIAL MISMATCH' "$D"/logs/*/*.log 2>/dev/null || true) | wc -l)" "$n" "$cov"
