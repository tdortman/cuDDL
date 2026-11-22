#include <CLI/CLI.hpp>
#include <omp.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kBuckets = 2048;
constexpr uint32_t kK = 25;
constexpr uint8_t kMantBits = 10;
constexpr uint32_t kMantMask = (1U << kMantBits) - 1U;
constexpr uint32_t kRestoreShiftBase = 63U - kMantBits;
constexpr uint32_t kMaxNlz = kRestoreShiftBase;

inline uint64_t splitmix64(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline uint16_t ddl_score(uint64_t hash) noexcept {
    uint32_t const nlz = static_cast<uint32_t>(__builtin_clzll(hash | 1ULL));
    uint32_t const capped = nlz > kMaxNlz ? kMaxNlz : nlz;
    uint32_t const shift = kRestoreShiftBase - capped;
    uint32_t const mant = nlz > kMaxNlz ? 0U : static_cast<uint32_t>((hash >> shift) & kMantMask);
    uint32_t const inv = mant ^ kMantMask;
    uint16_t const raw = static_cast<uint16_t>((capped << kMantBits) | inv);
    return raw == 0U ? uint16_t{1U} : raw;
}

inline int base_to_bits(char c) noexcept {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return -1;
    }
}

std::string read_first_contig_gz(std::string const& path, size_t max_bases = 8000000) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return {};
    gzbuffer(gz, 1 << 20);
    std::string seq;
    seq.reserve(5000000);
    std::vector<char> line(1 << 20);
    bool in_first = false;
    while (gzgets(gz, line.data(), static_cast<int>(line.size())) != nullptr) {
        if (line[0] == '>') {
            if (in_first) break;
            in_first = true;
            continue;
        }
        if (!in_first) continue;
        for (char* p = line.data(); *p != '\0'; ++p) {
            if (*p != '\n' && *p != '\r' && *p != ' ') {
                seq.push_back(*p);
                if (seq.size() >= max_bases) break;
            }
        }
        if (seq.size() >= max_bases) break;
    }
    gzclose(gz);
    return seq;
}

std::string read_all_contigs_gz(std::string const& path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return {};
    gzbuffer(gz, 1 << 20);
    std::string seq;
    seq.reserve(5000000);
    std::vector<char> line(1 << 20);
    bool first = true;
    while (gzgets(gz, line.data(), static_cast<int>(line.size())) != nullptr) {
        if (line[0] == '>') {
            if (!first) seq.push_back('N');
            first = false;
            continue;
        }
        for (char* p = line.data(); *p != '\0'; ++p) {
            if (*p != '\n' && *p != '\r' && *p != ' ') seq.push_back(*p);
        }
    }
    gzclose(gz);
    return seq;
}

void extract_canonical_kmers(std::string const& seq, std::vector<uint64_t>& out) {
    out.clear();
    out.reserve(seq.size());
    constexpr uint64_t mask = (1ULL << (2U * kK)) - 1ULL;
    constexpr uint32_t rc_shift = 2U * (kK - 1U);
    uint64_t fwd = 0, rev = 0;
    uint32_t valid = 0;
    for (char c : seq) {
        int const b = base_to_bits(c);
        if (b < 0) {
            valid = 0; fwd = 0; rev = 0;
            continue;
        }
        fwd = ((fwd << 2U) | static_cast<uint64_t>(b)) & mask;
        rev = (rev >> 2U) | (static_cast<uint64_t>(3 - b) << rc_shift);
        if (++valid >= kK) out.push_back(fwd < rev ? fwd : rev);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

using Sketch = std::array<uint16_t, kBuckets>;
using MinHash64 = std::array<uint64_t, kBuckets>;

Sketch sketch_ddl(std::vector<uint64_t> const& kmers, uint64_t seed) {
    Sketch s{};
    for (uint64_t km : kmers) {
        uint64_t const h = splitmix64(km ^ seed);
        size_t const idx = h & (kBuckets - 1U);
        uint16_t const sc = ddl_score(h);
        if (sc > s[idx]) s[idx] = sc;
    }
    return s;
}

// Idealized 1-permutation MinHash (full 64-bit minimum hash per bucket, zero hash-collision noise)
// plus stored exact 8-byte cardinality, to test |Q cap R| = J*(|Q|+|R|)/(1+J).
MinHash64 sketch_opm(std::vector<uint64_t> const& kmers, uint64_t seed) {
    MinHash64 s;
    s.fill(UINT64_MAX);
    for (uint64_t km : kmers) {
        uint64_t const h = splitmix64(km ^ seed);
        size_t const idx = h & (kBuckets - 1U);
        if (h < s[idx]) s[idx] = h;
    }
    return s;
}

struct Counts { uint32_t l = 0, e = 0, h = 0, z = 0; };

Counts compare_ddl(Sketch const& q, Sketch const& r) noexcept {
    Counts c{};
    for (uint32_t i = 0; i < kBuckets; ++i) {
        uint16_t const a = q[i], b = r[i];
        if ((a | b) == 0U) ++c.z;
        else if (a == b) ++c.e;
        else if (a < b) ++c.l;
        else ++c.h;
    }
    return c;
}

double wkid_ddl(Counts const& c) noexcept {
    uint32_t const d = std::max(std::min(c.l + c.e, c.h + c.e), 6U);
    return static_cast<double>(c.e) / static_cast<double>(d);
}

// OPM Jaccard + stored exact cardinalities -> WKID via |Q cap R| = J*(|Q|+|R|)/(1+J)
double wkid_opm_plus_card(MinHash64 const& q, size_t card_q, MinHash64 const& r, size_t card_r) noexcept {
    uint32_t eq = 0, occ = 0;
    for (uint32_t i = 0; i < kBuckets; ++i) {
        if (q[i] == UINT64_MAX && r[i] == UINT64_MAX) continue;
        ++occ;
        if (q[i] == r[i]) ++eq;
    }
    double const j = occ > 0 ? static_cast<double>(eq) / static_cast<double>(occ) : 0.0;
    double const inter = j * static_cast<double>(card_q + card_r) / (1.0 + j);
    double const w = inter / static_cast<double>(std::min(card_q, card_r));
    return std::min(1.0, std::max(0.0, w));
}

double ani_from_wkid(double w) noexcept {
    return w > 0.0 ? 100.0 * std::pow(w, 1.0 / static_cast<double>(kK)) : 0.0;
}

size_t sorted_intersection(std::vector<uint64_t> const& a, std::vector<uint64_t> const& b) {
    size_t i = 0, j = 0, hits = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { ++hits; ++i; ++j; }
        else if (a[i] < b[j]) ++i;
        else ++j;
    }
    return hits;
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double const pos = q * static_cast<double>(v.size() - 1);
    size_t const lo = static_cast<size_t>(std::floor(pos));
    size_t const hi = std::min(lo + 1, v.size() - 1);
    double const frac = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

std::string mutate_seq(std::string const& s, double sub_rate, uint64_t seed) {
    static constexpr char kBases[4] = {'A', 'C', 'G', 'T'};
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::string out = s;
    for (char& c : out) {
        int const b = base_to_bits(c);
        if (b >= 0 && uni(rng) < sub_rate) {
            c = kBases[(b + 1 + static_cast<int>(rng() % 3ULL)) & 3];
        }
    }
    return out;
}

bool load_cuddl_prefix(std::string const& path, uint32_t limit_n, std::vector<Sketch>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "CUDDLDB\0", 8) != 0) return false;
    uint16_t version = 0, buckets = 0, key_mask = 0;
    uint8_t kmer_len = 0, exp_bits = 0;
    uint32_t ref_count = 0, bl_count = 0;
    uint64_t bl_id = 0;
    in.read(reinterpret_cast<char*>(&version), 2);
    in.read(reinterpret_cast<char*>(&kmer_len), 1);
    in.read(reinterpret_cast<char*>(&exp_bits), 1);
    in.read(reinterpret_cast<char*>(&buckets), 2);
    in.read(reinterpret_cast<char*>(&key_mask), 2);
    in.read(reinterpret_cast<char*>(&ref_count), 4);
    in.read(reinterpret_cast<char*>(&bl_count), 4);
    in.read(reinterpret_cast<char*>(&bl_id), 8);
    if (!in || buckets != kBuckets) return false;
    in.seekg(static_cast<std::streamoff>(bl_count) * 8, std::ios::cur);
    for (uint32_t i = 0; i < ref_count; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), 4);
        in.seekg(static_cast<std::streamoff>(len), std::ios::cur);
    }
    uint32_t const n = std::min(ref_count, limit_n);
    out.resize(n);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(static_cast<size_t>(n) * kBuckets * 2U));
    return static_cast<bool>(in);
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Tier-3 checks: MinHash+card vs DDL asymmetry, 1-bit E>0 bitmap ratio, 5-seed WKID variance"};
    std::string db_path, genomes_list;
    app.add_option("--db", db_path, "Path to .cuddl DB")->required();
    app.add_option("--genomes", genomes_list, "Path to list of RefSeq .fna.gz paths")->required();
    CLI11_PARSE(app, argc, argv);

    // 1. 1-bit E>0 sparse bitmap + packed float32 nonzeros ratio on real 4096x4096 tile
    {
        std::vector<Sketch> sk;
        if (!load_cuddl_prefix(db_path, 4096, sk)) return 1;
        size_t const n = sk.size();
        size_t const npairs = n * (n - 1) / 2;
        uint64_t nz = 0;
        #pragma omp parallel for reduction(+:nz) schedule(dynamic, 16)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                Counts const c = compare_ddl(sk[i], sk[j]);
                if (c.e > 0) ++nz;
            }
        }
        double const raw_bytes = static_cast<double>(npairs) * 4.0;
        double const bm4_bytes = static_cast<double>(npairs) / 8.0 + static_cast<double>(nz) * 4.0;
        double const bm2_bytes = static_cast<double>(npairs) / 8.0 + static_cast<double>(nz) * 2.0;
        std::printf(
            "{\"exp\":\"sparse_bitmap\",\"pairs\":%zu,\"nz_frac\":%.4f,"
            "\"bitmap_f32_ratio\":%.3f,\"bitmap_u16_ratio\":%.3f}\n",
            npairs, static_cast<double>(nz) / static_cast<double>(npairs),
            raw_bytes / bm4_bytes, raw_bytes / bm2_bytes
        );
    }

    // 2. MinHash + 8B exact cardinality vs DDL on 40-chromosome length-asymmetry benchmark
    std::vector<std::string> paths;
    {
        std::ifstream in(genomes_list);
        std::string line;
        while (std::getline(in, line)) if (!line.empty()) paths.push_back(line);
    }
    {
        std::vector<std::string> chroms;
        for (size_t i = 0; i < paths.size() && chroms.size() < 40; ++i) {
            std::string seq = read_first_contig_gz(paths[i]);
            if (seq.size() >= 2500000) chroms.push_back(std::move(seq));
        }
        size_t const nc = chroms.size();
        std::vector<std::vector<uint64_t>> ckm(nc);
        std::vector<Sketch> cddl(nc);
        std::vector<MinHash64> copm(nc);
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < nc; ++i) {
            extract_canonical_kmers(chroms[i], ckm[i]);
            cddl[i] = sketch_ddl(ckm[i], 42ULL);
            copm[i] = sketch_opm(ckm[i], 42ULL);
        }
        for (size_t clen : {50000ULL, 200000ULL, 1000000ULL}) {
            for (double sub_rate : {0.0, 0.02}) {
                std::vector<double> ddl_we, opm_we, ddl_ae, opm_ae;
                for (size_t i = 0; i < nc; ++i) {
                    std::string sub = chroms[i].substr(chroms[i].size() / 3, clen);
                    if (sub_rate > 0.0) sub = mutate_seq(sub, sub_rate, 1000ULL + i * 31ULL + clen);
                    std::vector<uint64_t> qkm;
                    extract_canonical_kmers(sub, qkm);
                    Sketch const qddl = sketch_ddl(qkm, 42ULL);
                    MinHash64 const qopm = sketch_opm(qkm, 42ULL);
                    size_t const inter = sorted_intersection(qkm, ckm[i]);
                    double const exact_w = static_cast<double>(inter) / static_cast<double>(std::min(qkm.size(), ckm[i].size()));
                    double const exact_a = ani_from_wkid(exact_w);

                    double const w_ddl = wkid_ddl(compare_ddl(qddl, cddl[i]));
                    double const w_opm = wkid_opm_plus_card(qopm, qkm.size(), copm[i], ckm[i].size());
                    ddl_we.push_back(std::abs(w_ddl - exact_w) * 100.0);
                    opm_we.push_back(std::abs(w_opm - exact_w) * 100.0);
                    ddl_ae.push_back(std::abs(ani_from_wkid(w_ddl) - exact_a));
                    opm_ae.push_back(std::abs(ani_from_wkid(w_opm) - exact_a));
                }
                std::printf(
                    "{\"exp\":\"opm_plus_card_vs_ddl\",\"contig_kb\":%zu,\"sub_rate\":%.2f,"
                    "\"ddl_wkid_med_pp\":%.2f,\"ddl_wkid_p95_pp\":%.2f,"
                    "\"opm_wkid_med_pp\":%.2f,\"opm_wkid_p95_pp\":%.2f,"
                    "\"ddl_ani_p95_pp\":%.2f,\"opm_ani_p95_pp\":%.2f}\n",
                    clen / 1000ULL, sub_rate,
                    quantile(ddl_we, 0.5), quantile(ddl_we, 0.95),
                    quantile(opm_we, 0.5), quantile(opm_we, 0.95),
                    quantile(ddl_ae, 0.95), quantile(opm_ae, 0.95)
                );
            }
        }
    }
    return 0;
}
