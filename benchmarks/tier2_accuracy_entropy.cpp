#include <CLI/CLI.hpp>
#include <omp.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kBuckets = 2048;
constexpr uint32_t kK = 25;
constexpr uint8_t kExpBits = 6;
constexpr uint8_t kMantBits = 10;
constexpr uint32_t kMantMask = (1U << kMantBits) - 1U;
constexpr uint32_t kRestoreShiftBase = 63U - kMantBits;
constexpr uint32_t kMaxNlz = kRestoreShiftBase;  // min(53, 63) = 53

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
            valid = 0;
            fwd = 0;
            rev = 0;
            continue;
        }
        fwd = ((fwd << 2U) | static_cast<uint64_t>(b)) & mask;
        rev = (rev >> 2U) | (static_cast<uint64_t>(3 - b) << rc_shift);
        if (++valid >= kK) {
            out.push_back(fwd < rev ? fwd : rev);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

using Sketch = std::array<uint16_t, kBuckets>;

Sketch sketch_kmers(std::vector<uint64_t> const& kmers, uint64_t seed) {
    Sketch s{};
    for (uint64_t km : kmers) {
        uint64_t const h = splitmix64(km ^ seed);
        size_t const idx = h & (kBuckets - 1U);
        uint16_t const sc = ddl_score(h);
        if (sc > s[idx]) s[idx] = sc;
    }
    return s;
}

struct Counts {
    uint32_t l = 0, e = 0, h = 0, z = 0;
};

Counts compare_sketches(Sketch const& q, Sketch const& r) noexcept {
    Counts c{};
    for (uint32_t i = 0; i < kBuckets; ++i) {
        uint16_t const a = q[i], b = r[i];
        if ((a | b) == 0U) {
            ++c.z;
        } else if (a == b) {
            ++c.e;
        } else if (a < b) {
            ++c.l;
        } else {
            ++c.h;
        }
    }
    return c;
}

double wkid_from_counts(Counts const& c) noexcept {
    uint32_t const denom = std::min(c.l + c.e, c.h + c.e);
    return denom > 0U ? static_cast<double>(c.e) / static_cast<double>(denom) : 0.0;
}

double jaccard_from_counts(Counts const& c) noexcept {
    uint32_t const denom = c.l + c.e + c.h;
    return denom > 0U ? static_cast<double>(c.e) / static_cast<double>(denom) : 0.0;
}

double ani_from_wkid(double wkid) noexcept {
    if (wkid <= 0.0) return 0.0;
    return 100.0 * std::pow(wkid, 1.0 / static_cast<double>(kK));
}

size_t sorted_intersection(std::vector<uint64_t> const& a, std::vector<uint64_t> const& b) {
    size_t i = 0, j = 0, hits = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            ++hits; ++i; ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
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

size_t zlib_size(void const* data, size_t len, int level = 6) {
    uLongf dest_len = compressBound(static_cast<uLong>(len));
    std::vector<Bytef> dst(dest_len);
    int rc = compress2(dst.data(), &dest_len, static_cast<Bytef const*>(data), static_cast<uLong>(len), level);
    return rc == Z_OK ? static_cast<size_t>(dest_len) : len;
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

std::string mutate_seq(std::string const& s, double sub_rate, uint64_t seed) {
    static constexpr char kBases[4] = {'A', 'C', 'G', 'T'};
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::string out = s;
    for (char& c : out) {
        int const b = base_to_bits(c);
        if (b >= 0 && uni(rng) < sub_rate) {
            int const nb = (b + 1 + static_cast<int>(rng() % 3ULL)) & 3;
            c = kBases[nb];
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Tier-2 experiments: residual lossless coding, contig asymmetry, 5-seed variance"};
    std::string db_path;
    std::string genomes_list;
    app.add_option("--db", db_path, "Path to 100k .cuddl DB for residual-coding test")->required();
    app.add_option("--genomes", genomes_list, "Path to list of RefSeq .fna.gz paths")->required();
    CLI11_PARSE(app, argc, argv);

    // =========================================================================
    // 1. EXP T2-C: Residual-coded lossless counts on a real 4096x4096 RefSeq tile
    // =========================================================================
    {
        uint32_t const tile_n = 4096;
        std::vector<Sketch> sk;
        if (!load_cuddl_prefix(db_path, tile_n, sk)) {
            std::fprintf(stderr, "Failed loading %s\n", db_path.c_str());
            return 1;
        }
        size_t const npairs = static_cast<size_t>(tile_n) * (tile_n - 1U) / 2U;
        std::vector<uint32_t> rank_u32(npairs);
        std::vector<uint16_t> e_plane(npairs);
        std::vector<uint16_t> l_plane(npairs);
        std::vector<uint16_t> l_resid_zz(npairs);
        std::vector<uint8_t> e_lo(npairs), e_hi(npairs), lr_lo(npairs), lr_hi(npairs);
        // Also measure empirical Shannon entropy of E and (L - round((B-E)/2)).
        std::vector<uint64_t> hist_e(kBuckets + 1, 0);
        std::vector<uint64_t> hist_r(2U * kBuckets + 1, 0);

        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t i = 0; i < tile_n; ++i) {
            size_t const row_base = static_cast<size_t>(i) * (2ULL * tile_n - 1ULL - i) / 2ULL;
            for (uint32_t j = i + 1U; j < tile_n; ++j) {
                Counts const c = compare_sketches(sk[i], sk[j]);
                size_t const idx = row_base + (j - i - 1U);
                // Combinatorial rank over (L, E) with L + E <= B (bijective for full sketches).
                uint32_t const s = c.l + c.e;
                rank_u32[idx] = s * (s + 1U) / 2U + c.e;
                e_plane[idx] = static_cast<uint16_t>(c.e);
                l_plane[idx] = static_cast<uint16_t>(c.l);
                int32_t const mean_l = static_cast<int32_t>((kBuckets - c.z - c.e) / 2U);
                int32_t const diff = static_cast<int32_t>(c.l) - mean_l;
                uint16_t const zz = static_cast<uint16_t>((diff << 1) ^ (diff >> 31));
                l_resid_zz[idx] = zz;
                e_lo[idx] = static_cast<uint8_t>(c.e & 0xFFU);
                e_hi[idx] = static_cast<uint8_t>(c.e >> 8U);
                lr_lo[idx] = static_cast<uint8_t>(zz & 0xFFU);
                lr_hi[idx] = static_cast<uint8_t>(zz >> 8U);
            }
        }
        for (size_t idx = 0; idx < npairs; ++idx) {
            ++hist_e[e_plane[idx]];
            uint32_t const zz = l_resid_zz[idx];
            if (zz < hist_r.size()) ++hist_r[zz];
        }
        auto shannon_bits = [&](std::vector<uint64_t> const& h) {
            double H = 0.0;
            for (uint64_t cnt : h) {
                if (cnt > 0) {
                    double const p = static_cast<double>(cnt) / static_cast<double>(npairs);
                    H -= p * std::log2(p);
                }
            }
            return H;
        };
        double const h_e = shannon_bits(hist_e);
        double const h_r = shannon_bits(hist_r);
        size_t const raw_bytes = npairs * 4ULL;
        size_t const z_rank = zlib_size(rank_u32.data(), raw_bytes);
        size_t const z_planes = zlib_size(e_plane.data(), npairs * 2ULL) +
                                zlib_size(l_plane.data(), npairs * 2ULL);
        size_t const z_resid = zlib_size(e_lo.data(), npairs) + zlib_size(e_hi.data(), npairs) +
                               zlib_size(lr_lo.data(), npairs) + zlib_size(lr_hi.data(), npairs);
        std::printf(
            "{\"exp\":\"T2-C_residual_lossless\",\"pairs\":%zu,\"raw_bytes\":%zu,"
            "\"H_E_bits\":%.3f,\"H_Lresid_bits\":%.3f,\"H_sum_bits\":%.3f,"
            "\"theoretical_ratio_vs_4B\":%.3f,"
            "\"zlib_rank_u32_ratio\":%.3f,\"zlib_raw_planes_ratio\":%.3f,"
            "\"zlib_bytesplit_resid_ratio\":%.3f}\n",
            npairs, raw_bytes, h_e, h_r, h_e + h_r, 32.0 / (h_e + h_r),
            static_cast<double>(raw_bytes) / static_cast<double>(z_rank),
            static_cast<double>(raw_bytes) / static_cast<double>(z_planes),
            static_cast<double>(raw_bytes) / static_cast<double>(z_resid)
        );
        std::fflush(stdout);
    }

    // Load genome paths
    std::vector<std::string> paths;
    {
        std::ifstream in(genomes_list);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) paths.push_back(line);
        }
    }

    // =========================================================================
    // 2. EXP T2-D1: Length-asymmetric plasmid/contig vs chromosome accuracy
    // =========================================================================
    {
        std::vector<std::string> chroms;
        for (size_t i = 0; i < paths.size() && chroms.size() < 40; ++i) {
            std::string seq = read_first_contig_gz(paths[i]);
            if (seq.size() >= 2500000) {
                chroms.push_back(std::move(seq));
            }
        }
        size_t const nc = chroms.size();
        std::vector<std::vector<uint64_t>> chrom_kmers(nc);
        std::vector<Sketch> chrom_sk(nc);
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < nc; ++i) {
            extract_canonical_kmers(chroms[i], chrom_kmers[i]);
            chrom_sk[i] = sketch_kmers(chrom_kmers[i], 42ULL);
        }

        double mean_chrom_mb = 0.0;
        for (auto const& c : chroms) mean_chrom_mb += static_cast<double>(c.size()) / 1e6;
        mean_chrom_mb /= static_cast<double>(nc);

        for (size_t clen : {50000ULL, 200000ULL, 1000000ULL}) {
            for (double sub_rate : {0.0, 0.02}) {
                std::vector<double> wkid_errs, ani_errs, jac_errs, wkid_biases, eh_counts;
                for (size_t i = 0; i < nc; ++i) {
                    size_t const start = chroms[i].size() / 3;
                    std::string sub = chroms[i].substr(start, clen);
                    if (sub_rate > 0.0) sub = mutate_seq(sub, sub_rate, 1000ULL + i * 31ULL + clen);
                    std::vector<uint64_t> qkm;
                    extract_canonical_kmers(sub, qkm);
                    Sketch const qsk = sketch_kmers(qkm, 42ULL);

                    size_t const inter = sorted_intersection(qkm, chrom_kmers[i]);
                    double const exact_wkid = static_cast<double>(inter) / static_cast<double>(std::min(qkm.size(), chrom_kmers[i].size()));
                    double const exact_jac = static_cast<double>(inter) / static_cast<double>(qkm.size() + chrom_kmers[i].size() - inter);
                    Counts const c = compare_sketches(qsk, chrom_sk[i]);
                    double const est_wkid = wkid_from_counts(c);
                    double const est_jac = jaccard_from_counts(c);

                    wkid_errs.push_back(std::abs(est_wkid - exact_wkid));
                    wkid_biases.push_back(est_wkid - exact_wkid);
                    ani_errs.push_back(std::abs(ani_from_wkid(est_wkid) - ani_from_wkid(exact_wkid)));
                    jac_errs.push_back(std::abs(est_jac - exact_jac));
                    eh_counts.push_back(static_cast<double>(c.e + c.h));
                }
                double mean_eh = 0.0, mean_bias = 0.0;
                for (double v : eh_counts) mean_eh += v;
                for (double v : wkid_biases) mean_bias += v;
                mean_eh /= static_cast<double>(nc);
                mean_bias /= static_cast<double>(nc);
                std::printf(
                    "{\"exp\":\"T2-D1_length_asymmetry\",\"chroms\":%zu,\"mean_chrom_mb\":%.2f,"
                    "\"contig_kb\":%zu,\"sub_rate\":%.2f,\"mean_ratio\":%.1f,"
                    "\"mean_EH_buckets\":%.1f,\"wkid_med_err\":%.4f,\"wkid_p95_err\":%.4f,"
                    "\"wkid_mean_bias\":%.4f,\"ani_med_err_pp\":%.3f,\"ani_p95_err_pp\":%.3f,"
                    "\"jac_med_err\":%.4f,\"jac_p95_err\":%.4f}\n",
                    nc, mean_chrom_mb, clen / 1000ULL, sub_rate,
                    (mean_chrom_mb * 1e6) / static_cast<double>(clen),
                    mean_eh, quantile(wkid_errs, 0.5), quantile(wkid_errs, 0.95),
                    mean_bias, quantile(ani_errs, 0.5), quantile(ani_errs, 0.95),
                    quantile(jac_errs, 0.5), quantile(jac_errs, 0.95)
                );
                std::fflush(stdout);
            }
        }
    }

    // =========================================================================
    // 3. EXP T2-D2: 5-seed hash variance on 100 real RefSeq genomes
    // =========================================================================
    {
        uint32_t const ng = 100;
        std::vector<std::vector<uint64_t>> kmers(ng);
        #pragma omp parallel for schedule(dynamic, 1)
        for (uint32_t i = 0; i < ng; ++i) {
            std::string seq = read_all_contigs_gz(paths[i]);
            extract_canonical_kmers(seq, kmers[i]);
        }
        // Compute exact WKID on all related pairs (exact WKID >= 0.05).
        struct PairTruth { uint32_t i, j; double exact_wkid, exact_ani; };
        std::vector<PairTruth> related;
        #pragma omp parallel
        {
            std::vector<PairTruth> local;
            #pragma omp for schedule(dynamic, 4)
            for (uint32_t i = 0; i < ng; ++i) {
                for (uint32_t j = i + 1U; j < ng; ++j) {
                    size_t const m = std::min(kmers[i].size(), kmers[j].size());
                    if (m == 0) continue;
                    size_t const inter = sorted_intersection(kmers[i], kmers[j]);
                    double const w = static_cast<double>(inter) / static_cast<double>(m);
                    if (w >= 0.05) {
                        local.push_back({i, j, w, ani_from_wkid(w)});
                    }
                }
            }
            #pragma omp critical
            related.insert(related.end(), local.begin(), local.end());
        }

        std::array<uint64_t, 5> const seeds = {42ULL, 1001ULL, 2002ULL, 3003ULL, 4004ULL};
        for (uint64_t seed : seeds) {
            std::vector<Sketch> sk(ng);
            #pragma omp parallel for schedule(static)
            for (uint32_t i = 0; i < ng; ++i) {
                sk[i] = sketch_kmers(kmers[i], seed);
            }
            std::vector<double> ani_errs, wkid_errs;
            ani_errs.reserve(related.size());
            wkid_errs.reserve(related.size());
            for (auto const& pt : related) {
                Counts const c = compare_sketches(sk[pt.i], sk[pt.j]);
                double const ew = wkid_from_counts(c);
                wkid_errs.push_back(std::abs(ew - pt.exact_wkid));
                ani_errs.push_back(std::abs(ani_from_wkid(ew) - pt.exact_ani));
            }
            std::printf(
                "{\"exp\":\"T2-D2_seed_variance\",\"seed\":%llu,\"related_pairs\":%zu,"
                "\"ani_med_err_pp\":%.3f,\"ani_p95_err_pp\":%.3f,"
                "\"wkid_med_err\":%.4f,\"wkid_p95_err\":%.4f}\n",
                static_cast<unsigned long long>(seed), related.size(),
                quantile(ani_errs, 0.5), quantile(ani_errs, 0.95),
                quantile(wkid_errs, 0.5), quantile(wkid_errs, 0.95)
            );
            std::fflush(stdout);
        }
    }
    return 0;
}
