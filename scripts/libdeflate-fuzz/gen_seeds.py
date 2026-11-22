#!/usr/bin/env python3
"""Seed corpora for the libdeflate fuzz campaigns.

usage: gen_seeds.py <libdeflate.so> <genome-dir> <out-dir> [n-genomes]
Writes <out-dir>/{raw,gzip,zlib,diff}: raw DEFLATE, gzip, zlib seeds for libdeflate's
own harnesses, and 'diff' seeds (1 mode byte + raw or gzip payload) for diff_fuzz.
"""
import ctypes, glob, gzip, os, random, struct, sys, zlib

so, gdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
ngen = int(sys.argv[4]) if len(sys.argv) > 4 else 300
rng = random.Random(1)
L = ctypes.CDLL(so)
L.libdeflate_alloc_compressor.restype = ctypes.c_void_p
L.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
for f in ("deflate", "gzip", "zlib"):
    fn = getattr(L, f"libdeflate_{f}_compress")
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    getattr(L, f"libdeflate_{f}_compress_bound").restype = ctypes.c_size_t
    getattr(L, f"libdeflate_{f}_compress_bound").argtypes = [ctypes.c_void_p, ctypes.c_size_t]

MAXP = 60000  # payload cap; max_len is 65536 incl. mode byte / wrappers
seen = set()
counts = {"raw": 0, "gzip": 0, "zlib": 0, "diff": 0}


def emit(kind, data, diffmode=None):
    if len(data) > MAXP or (kind, data) in seen:
        return
    seen.add((kind, data))
    d = os.path.join(out, kind)
    os.makedirs(d, exist_ok=True)
    name = f"{kind}-{counts[kind]:06d}"
    with open(os.path.join(d, name), "wb") as fh:
        fh.write(data)
    counts[kind] += 1


def ld_compress(fmt, level, data):
    c = L.libdeflate_alloc_compressor(level)
    cap = getattr(L, f"libdeflate_{fmt}_compress_bound")(c, len(data))
    buf = ctypes.create_string_buffer(cap)
    n = getattr(L, f"libdeflate_{fmt}_compress")(c, data, len(data), buf, cap)
    L.libdeflate_free_compressor.argtypes = [ctypes.c_void_p]
    L.libdeflate_free_compressor(c)
    return buf.raw[:n] if n else None


def zraw(level, strat, data):
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strat)
    return c.compress(data) + c.flush()


def add_variants(data):
    """Compress `data` many ways; emit raw/gzip/zlib/diff seeds."""
    raws = []
    for lv, st in [(1, 0), (6, 0), (9, 0), (6, zlib.Z_FIXED), (6, zlib.Z_HUFFMAN_ONLY),
                   (6, zlib.Z_RLE), (6, zlib.Z_FILTERED)]:
        raws.append(zraw(lv, st, data))
    for lv in (1, 3, 6, 9, 12):
        raws.append(ld_compress("deflate", lv, data))
    raws.append(ld_compress("deflate", 0, data[:4000]))  # stored blocks
    for r in raws:
        if r is None:
            continue
        emit("raw", r)
        emit("diff", bytes([rng.choice((3, 4, 5, 6, 7, 8, 8, 8, 2, 1, 0))]) + r)
    for lv in (1, 6, 9, 12):
        g = ld_compress("gzip", lv, data)
        if g:
            emit("gzip", g)
            emit("diff", bytes([rng.choice((3, 4, 5, 6, 7, 8))]) + g)
    for lv in (1, 6, 9):
        emit("zlib", zlib.compress(data, lv))
    emit("zlib", ld_compress("zlib", 6, data) or b"")


def gz_payload(path):
    b = open(path, "rb").read()
    flg = b[3]
    p = 10
    if flg & 4:
        p += 2 + struct.unpack("<H", b[p:p + 2])[0]
    if flg & 8:
        p = b.index(0, p) + 1
    if flg & 16:
        p = b.index(0, p) + 1
    if flg & 2:
        p += 2
    return b[:p], b[p:-8]


files = sorted(glob.glob(os.path.join(gdir, "*.fna.gz")))
rng.shuffle(files)
for path in files[:ngen]:
    # Real RefSeq data: raw DEFLATE prefix (header/trailer stripped) and the gzip file prefix.
    hdr, raw = gz_payload(path)
    for cut in (4096, 32768):
        emit("raw", raw[:cut])
        emit("diff", bytes([8]) + raw[:cut])
    emit("gzip", open(path, "rb").read(32768))
    emit("diff", bytes([8]) + open(path, "rb").read(32768))
    # Recompressed windows of the genome text: with line breaks (as stored) and pure ACGT.
    with gzip.open(path, "rb") as fh:
        text = fh.read(2_000_000)
    for size in (rng.choice((300, 2000, 9000)), rng.choice((20000, 60000, 90000))):
        off = rng.randrange(0, max(1, len(text) - size))
        add_variants(text[off:off + size])
    add_variants(text[:600])  # header line + start of sequence

# Synthetic: DNA with injected short repeats (match-dense), highly repetitive, random bytes, text, binary.
for _ in range(60):
    n = rng.choice((200, 3000, 30000, 80000))
    s = bytearray(rng.choice(b"ACGT") for _ in range(n))
    for _ in range(n // rng.choice((4, 8, 20, 60))):
        ln = rng.choice((3, 4, 5, 6, 8, 12, 20, 32, 40, 258))
        a = rng.randrange(0, max(1, n - ln))
        b_ = rng.randrange(0, max(1, n - ln))
        s[b_:b_ + ln] = s[a:a + ln]
    add_variants(bytes(s))
for data in (b"", b"A", b"A" * 70000, bytes(range(256)) * 200, os.urandom(5000),
             (b"ACGT" * 50 + b"\n") * 200):
    add_variants(data)
# Non-DNA text and binary seeds: libdeflate's own docs (<work-dir>/patched, beside <work-dir>/seeds).
for path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(out)), "patched", "*.md")) + ["/usr/bin/clang"]:
    add_variants(open(path, "rb").read(60000))
# Edge streams: empty final stored block, empty fixed block, final empty dynamic-less blocks.
for h in (b"\x01\x00\x00\xff\xff", b"\x03\x00", b"\x00\x00\x00\xff\xff\x03\x00", b"\x03", b""):
    emit("raw", h)
    emit("diff", b"\x03" + h)
print(counts)
