#!/bin/bash
# Launch both campaigns detached. Usage: run.sh <work-dir>   (needs build.sh and gen_seeds.py output)
# Campaign 1: 3 harnesses x (4 bmi2 + 4 generic) workers for C1_SECS.  Campaign 2: 12 + 12 workers for C2_SECS.
set -euo pipefail
W=$(realpath "$1"); C1_SECS=${C1_SECS:-4500}; C2_SECS=${C2_SECS:-8100}; export TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
export ASAN_OPTIONS=abort_on_error=0:detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
launch() { # name binary seeds secs workers
  local d=$W/run/$1; mkdir -p "$d/corpus" "$d/logs"
  cp -n "$W/seeds/$3"/* "$d/corpus/" 2>/dev/null || cp -n "$W"/seeds/$3/* "$d/corpus/"
  for variant in bmi2 generic; do
    mkdir -p "$d/logs/$variant"
    (cd "$d/logs/$variant"
     if [ $variant = generic ]; then export LIBDEFLATE_DISABLE_CPU_FEATURES=bmi2; fi
     exec setsid nohup "$W/build/$2" -max_len=65536 -timeout=60 -rss_limit_mb=4096 -artifact_prefix="$d/" \
       -max_total_time="$4" -jobs="$5" -workers="$5" "$d/corpus" > main.log 2>&1 &)
  done
}
for t in deflate_decompress:raw gzip_decompress:gzip zlib_decompress:zlib; do
  launch "c1_${t%%:*}" "${t%%:*}" "${t##*:}" "$C1_SECS" 4
done
launch c2_diff diff_fuzz diff "$C2_SECS" 12
