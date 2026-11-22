#!/bin/bash
# Build the libFuzzer harnesses. Usage: build.sh <work-dir>
# <work-dir>/patched and <work-dir>/upstream hold libdeflate trees; binaries go to <work-dir>/build.
set -euo pipefail
W=$(realpath "$1"); HERE=$(cd "$(dirname "$0")" && pwd); export TMPDIR=$W/tmp; mkdir -p "$TMPDIR"
SAN=address,undefined
COMMON="-g -DLIBDEFLATE_ENABLE_ASSERTIONS=1 -Wall"
B=$W/build; mkdir -p "$B"

# Campaign 1: libdeflate's own harnesses against the patched sources, same flags as fuzz.sh.
for t in deflate_decompress gzip_decompress zlib_decompress; do
  clang $COMMON -O1 -fsanitize=fuzzer,$SAN -I"$W/patched" \
    "$W"/patched/lib/*.c "$W"/patched/lib/*/*.c \
    "$W/patched/scripts/libFuzzer/$t/fuzz.c" -o "$B/$t"
done

# Campaign 2: patched library with coverage instrumentation, upstream library without.
mkdir -p "$B/obj"
(cd "$B/obj" && rm -f *.o &&
 clang $COMMON -O2 -fsanitize=fuzzer-no-link,$SAN -I"$W/patched" -c "$W"/patched/lib/*.c "$W"/patched/lib/*/*.c)
ld -r "$B"/obj/*.o -o "$B/patched.o"
mkdir -p "$B/uobj"
(cd "$B/uobj" && rm -f *.o &&
 clang $COMMON -O2 -fsanitize=$SAN -I"$W/upstream" -c "$W"/upstream/lib/*.c "$W"/upstream/lib/*/*.c)
ld -r "$B"/uobj/*.o -o "$B/upstream.o"
nm --defined-only -g "$B/upstream.o" | awk '{print $3 " up_" $3}' > "$B/rename.txt"
objcopy --redefine-syms="$B/rename.txt" "$B/upstream.o" "$B/upstream_renamed.o"
clang $COMMON -O2 -fsanitize=fuzzer,$SAN -I"$W/patched" "$HERE/diff_fuzz.c" \
  "$B/patched.o" "$B/upstream_renamed.o" -o "$B/diff_fuzz"
