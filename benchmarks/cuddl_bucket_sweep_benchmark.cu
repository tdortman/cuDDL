#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>
#include <cuddl/cuddl.cuh>
#include <cuddl/replicated_database.cuh>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

template <size_t BucketCount>
void verify_small(uint32_t n_refs, bool sparse) {
    std::mt19937_64 rng(1234ULL + BucketCount + (sparse ? 7U : 0U));
    std::vector<uint16_t> rows(static_cast<size_t>(n_refs) * BucketCount, 0U);
    for (size_t i = 0; i < rows.size(); ++i) {
        uint64_t const r = rng();
        if (sparse && (r & 3U) == 0U) continue;  // 25% empty buckets to exercise non-full path
        uint16_t const exp = static_cast<uint16_t>(9U + (r & 7U));
        uint16_t const mant = static_cast<uint16_t>((r >> 3U) & 0x3FFU);
        rows[i] = static_cast<uint16_t>((exp << 10U) | mant);
    }
    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto const compat = cuddl::score_compatibility::current<25, BucketCount>();
    auto db = CUDDL_UNWRAP((cuddl::reference_database<25, BucketCount>::build_async(
        {d_rows.data(), d_rows.size()}, compat, stream
    )));
    stream.sync();
    std::vector<cuddl::reference_database<25, BucketCount>> dbs;
    dbs.push_back(std::move(db));
    auto group = CUDDL_UNWRAP((cuddl::replicated_database<25, BucketCount>::adopt(std::move(dbs))));

    uint64_t checked = 0;
    uint64_t mismatches = 0;
    CUDDL_UNWRAP(group.search_all_to_all([&](cuddl::replica_tile const& rtile) {
        auto host_tile = CUDDL_UNWRAP(rtile.download());
        host_tile->tile().for_each_passing([&](cuddl::batch_search_result const& res) {
            uint16_t const* q = rows.data() + static_cast<size_t>(res.query_id) * BucketCount;
            uint16_t const* r = rows.data() + static_cast<size_t>(res.reference_id) * BucketCount;
            uint32_t l = 0, e = 0, h = 0, z = 0;
            for (size_t b = 0; b < BucketCount; ++b) {
                if ((q[b] | r[b]) == 0U) ++z;
                else if (q[b] == r[b]) ++e;
                else if (q[b] < r[b]) ++l;
                else ++h;
            }
            if (res.counts.lower != l || res.counts.equal != e ||
                res.counts.higher != h || res.counts.both_empty != z) {
                if (mismatches < 5) {
                    std::fprintf(
                        stderr,
                        "MISMATCH B=%zu sparse=%d (%u,%u): got=(%u,%u,%u,%u) ref=(%u,%u,%u,%u)\n",
                        BucketCount, sparse ? 1 : 0, res.query_id, res.reference_id,
                        res.counts.lower, res.counts.equal, res.counts.higher, res.counts.both_empty,
                        l, e, h, z
                    );
                }
                ++mismatches;
            }
            ++checked;
        });
    }));
    uint64_t const expected = static_cast<uint64_t>(n_refs) * (n_refs - 1U) / 2ULL;
    if (checked != expected || mismatches > 0) {
        std::fprintf(stderr, "VERIFY FAILED B=%zu sparse=%d checked=%llu expected=%llu mismatches=%llu\n",
                     BucketCount, sparse ? 1 : 0, static_cast<unsigned long long>(checked),
                     static_cast<unsigned long long>(expected), static_cast<unsigned long long>(mismatches));
        std::exit(2);
    }
    std::fprintf(stderr, "VERIFY OK B=%zu sparse=%d pairs=%llu\n",
                 BucketCount, sparse ? 1 : 0, static_cast<unsigned long long>(checked));
}

template <size_t BucketCount>
void run_bucket_sweep(uint32_t n_refs, int reps) {
    verify_small<BucketCount>(64, false);
    verify_small<BucketCount>(64, true);

    std::mt19937_64 rng(42ULL + BucketCount);
    std::vector<uint16_t> rows(static_cast<size_t>(n_refs) * BucketCount);
    for (size_t i = 0; i < rows.size(); ++i) {
        uint64_t const r = rng();
        uint16_t const exp = static_cast<uint16_t>(9U + (r & 7U));
        uint16_t const mant = static_cast<uint16_t>((r >> 3U) & 0x3FFU);
        rows[i] = static_cast<uint16_t>((exp << 10U) | mant);
    }

    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto const compat = cuddl::score_compatibility::current<25, BucketCount>();
    auto db = CUDDL_UNWRAP((cuddl::reference_database<25, BucketCount>::build_async(
        {d_rows.data(), d_rows.size()}, compat, stream
    )));
    stream.sync();

    std::vector<cuddl::reference_database<25, BucketCount>> dbs;
    dbs.push_back(std::move(db));
    auto group = CUDDL_UNWRAP((cuddl::replicated_database<25, BucketCount>::adopt(std::move(dbs))));

    uint64_t const total_pairs = static_cast<uint64_t>(n_refs) * (n_refs - 1U) / 2ULL;

    // Warmup
    CUDDL_UNWRAP(group.search_all_to_all([](cuddl::replica_tile const&) {}));

    for (int rep = 0; rep < reps; ++rep) {
        auto const t0 = std::chrono::steady_clock::now();
        CUDDL_UNWRAP(group.search_all_to_all([](cuddl::replica_tile const&) {}));
        auto const t1 = std::chrono::steady_clock::now();
        double const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double const gpairs_s = (static_cast<double>(total_pairs) / 1e9) / (ms / 1e3);
        double const wkid4_gb_s = gpairs_s * 4.0;
        std::printf(
            "{\"buckets\":%zu,\"n\":%u,\"pairs\":%llu,\"rep\":%d,\"ms\":%.2f,\"gpairs_s\":%.3f,\"wkid4_gb_s\":%.2f}\n",
            BucketCount, n_refs, static_cast<unsigned long long>(total_pairs), rep, ms, gpairs_s, wkid4_gb_s
        );
        std::fflush(stdout);
    }
}

}  // namespace

int main() {
    uint32_t const n_refs = 100000;
    run_bucket_sweep<2048>(n_refs, 3);
    run_bucket_sweep<4096>(n_refs, 3);
    run_bucket_sweep<8192>(n_refs, 3);
    return 0;
}
