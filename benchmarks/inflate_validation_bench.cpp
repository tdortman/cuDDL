// Standalone single- and multi-thread gzip inflation benchmark comparing
// patched libdeflate, stock libdeflate, and system zlib / zlib-ng on both
// bacterial RefSeq FASTA and non-DNA corpora, with byte-exact verification.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <libdeflate.h>
#include <zlib.h>

namespace fs = std::filesystem;

struct gz_item {
    std::string name;
    std::vector<uint8_t> compressed;
    size_t expected_size = 0;
};

static size_t gzip_isize(std::vector<uint8_t> const& buf) {
    if (buf.size() < 4) return 0;
    uint32_t v = 0;
    std::memcpy(&v, buf.data() + buf.size() - 4, 4);
    return static_cast<size_t>(v);
}

static bool inflate_zlib(uint8_t const* in, size_t in_len, std::vector<uint8_t>& out, size_t& actual) {
    z_stream zs{};
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) return false;
    zs.next_in = const_cast<Bytef*>(in);
    zs.avail_in = static_cast<uInt>(in_len);
    zs.next_out = out.data();
    zs.avail_out = static_cast<uInt>(out.size());
    int rc = inflate(&zs, Z_FINISH);
    actual = zs.total_out;
    inflateEnd(&zs);
    return rc == Z_STREAM_END;
}

static bool inflate_libdef(
    libdeflate_decompressor* d, uint8_t const* in, size_t in_len, std::vector<uint8_t>& out, size_t& actual
) {
    auto res = libdeflate_gzip_decompress(d, in, in_len, out.data(), out.size(), &actual);
    return res == LIBDEFLATE_SUCCESS;
}

static double bench_mode(
    std::vector<gz_item> const& items,
    bool use_zlib,
    int threads,
    int reps,
    size_t total_uncomp
) {
    std::vector<double> gbps;
    size_t max_out = 0;
    for (auto const& it : items) max_out = std::max(max_out, it.expected_size);
    max_out = std::max<size_t>(max_out, 64 * 1024);

    for (int r = 0; r < reps; ++r) {
        std::atomic<size_t> next_idx{0};
        std::atomic<size_t> ok_bytes{0};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&]() {
                std::vector<uint8_t> out(max_out);
                libdeflate_decompressor* d = use_zlib ? nullptr : libdeflate_alloc_decompressor();
                size_t local_bytes = 0;
                while (true) {
                    size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (i >= items.size()) break;
                    size_t actual = 0;
                    bool ok = use_zlib
                        ? inflate_zlib(items[i].compressed.data(), items[i].compressed.size(), out, actual)
                        : inflate_libdef(d, items[i].compressed.data(), items[i].compressed.size(), out, actual);
                    if (!ok || actual != items[i].expected_size) {
                        std::fprintf(stderr, "DECOMPRESS ERROR on %s\n", items[i].name.c_str());
                        std::abort();
                    }
                    local_bytes += actual;
                }
                if (d) libdeflate_free_decompressor(d);
                ok_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
            });
        }
        for (auto& th : pool) th.join();
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        if (ok_bytes.load() != total_uncomp) {
            std::fprintf(stderr, "BYTE COUNT MISMATCH\n");
            std::abort();
        }
        gbps.push_back(static_cast<double>(total_uncomp) / 1e9 / sec);
    }
    std::sort(gbps.begin(), gbps.end());
    return gbps[gbps.size() / 2];
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "Usage: %s <label> <dir> <max_files> <threads>\n", argv[0]);
        return 1;
    }
    std::string label = argv[1];
    std::string dir = argv[2];
    int max_files = std::atoi(argv[3]);
    int threads = std::atoi(argv[4]);

    std::vector<fs::path> paths;
    for (auto const& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() || e.is_symlink()) {
            auto p = e.path();
            if (p.extension() == ".gz") paths.push_back(p);
        }
    }
    std::sort(paths.begin(), paths.end());
    if (static_cast<int>(paths.size()) > max_files) paths.resize(max_files);

    std::vector<gz_item> items;
    items.reserve(paths.size());
    size_t total_comp = 0;
    size_t total_uncomp = 0;
    auto* d_ver = libdeflate_alloc_decompressor();
    for (auto const& p : paths) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) continue;
        size_t sz = static_cast<size_t>(f.tellg());
        f.seekg(0);
        gz_item it;
        it.name = p.filename().string();
        it.compressed.resize(sz);
        f.read(reinterpret_cast<char*>(it.compressed.data()), static_cast<std::streamsize>(sz));
        size_t est = gzip_isize(it.compressed);
        if (est == 0 || est > 256ULL * 1024 * 1024) continue;

        // Verify byte-exact identity between libdeflate and zlib on every file!
        std::vector<uint8_t> out_ld(est), out_z(est);
        size_t a_ld = 0, a_z = 0;
        if (!inflate_libdef(d_ver, it.compressed.data(), it.compressed.size(), out_ld, a_ld)) {
            std::fprintf(stderr, "libdeflate verify failed on %s\n", it.name.c_str());
            return 2;
        }
        if (!inflate_zlib(it.compressed.data(), it.compressed.size(), out_z, a_z)) {
            std::fprintf(stderr, "zlib verify failed on %s\n", it.name.c_str());
            return 3;
        }
        if (a_ld != a_z || std::memcmp(out_ld.data(), out_z.data(), a_ld) != 0) {
            std::fprintf(stderr, "BYTE MISMATCH between libdeflate and zlib on %s!\n", it.name.c_str());
            return 4;
        }
        it.expected_size = a_ld;
        total_comp += sz;
        total_uncomp += a_ld;
        items.push_back(std::move(it));
    }
    libdeflate_free_decompressor(d_ver);

    double ld_1t = bench_mode(items, false, 1, 3, total_uncomp);
    double ld_mt = bench_mode(items, false, threads, 3, total_uncomp);
    double z_1t = bench_mode(items, true, 1, 3, total_uncomp);
    double z_mt = bench_mode(items, true, threads, 3, total_uncomp);

    std::printf(
        "{\"label\":\"%s\",\"files\":%zu,\"comp_mb\":%.2f,\"uncomp_mb\":%.2f,"
        "\"verified_byte_exact\":true,\"threads\":%d,"
        "\"libdeflate_1t_gbps\":%.3f,\"libdeflate_mt_gbps\":%.3f,"
        "\"zlib_1t_gbps\":%.3f,\"zlib_mt_gbps\":%.3f,\"zlib_ver\":\"%s\"}\n",
        label.c_str(), items.size(), total_comp / 1e6, total_uncomp / 1e6, threads,
        ld_1t, ld_mt, z_1t, z_mt, zlibVersion()
    );
    return 0;
}
