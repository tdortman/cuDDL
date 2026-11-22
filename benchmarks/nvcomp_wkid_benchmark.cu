// Benchmark GPU compression codecs on real bacterial RefSeq pairwise WKID tiles (2^25 pairs = 134 MB)
// and 4-byte lossless (E, H) all-metrics tiles:
//   - cuDDL 1-bit bitmap + packed non-zero float32 (sparse_similarity)
//   - cuDDL 1-bit bitmap + packed 4-byte (E, H) (all-metrics lossless)
//   - nvCOMP Bitcomp, Cascaded, ANS, zstd, LZ4 (default batched lossless)
//   - Thresholded m_min=20 formats: 12 B edge list vs. 8 B CSR vs. 1-bit bitmap + float32

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>
#include <cub/device/device_select.cuh>
#include <cuddl/cuddl.cuh>
#include <cuddl/reference_database_file.cuh>

#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/zstd.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t k_buckets = 2048U;

__global__ void build_wkid_and_eh_kernel(
    cuddl::packed_pairwise_counts const* in,
    size_t n,
    float* wkid_out,
    uint32_t* eh_out,
    uint32_t* wkid_bm,
    uint32_t* m20_bm,
    uint64_t* nz_cnt,
    uint64_t* m20_cnt
) {
    auto const tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const lane = threadIdx.x & 31U;
    bool nz = false;
    bool pass20 = false;
    if (tid < n) {
        auto const c = in[tid].unpack(k_buckets);
        uint32_t const e = c.equal;
        uint32_t const l = c.lower;
        uint32_t const h = c.higher;
        uint32_t const denom = max(e + min(l, h), 6U);
        float const w = e > 0U ? static_cast<float>(e) / static_cast<float>(denom) : 0.0f;
        wkid_out[tid] = w;
        // Pack (E: 12b, H: 12b, Z_or_L_flag: 8b) — for full sketches Z=0, (E, H) recovers L=B-E-H,
        // giving exact WKID, ANI, containment, and completeness in 4 bytes/pair.
        uint32_t const z = static_cast<uint32_t>(k_buckets) - e - l - h;
        uint32_t const eh_packed = (e == 0U && z == 0U) ? 0U : ((e & 0xFFFU) | ((h & 0xFFFU) << 12U) | ((z & 0xFFU) << 24U));
        eh_out[tid] = eh_packed;
        nz = (e > 0U);
        pass20 = (e >= 20U);
    }
    uint32_t const bm = __ballot_sync(0xffffffffU, nz);
    uint32_t const bm20 = __ballot_sync(0xffffffffU, pass20);
    if (lane == 0U && (tid / 32U) < (n + 31U) / 32U) {
        wkid_bm[tid / 32U] = bm;
        m20_bm[tid / 32U] = bm20;
        atomicAdd(reinterpret_cast<unsigned long long*>(nz_cnt), static_cast<unsigned long long>(__popc(bm)));
        atomicAdd(reinterpret_cast<unsigned long long*>(m20_cnt), static_cast<unsigned long long>(__popc(bm20)));
    }
}

struct nonzero_f32 {
    __device__ bool operator()(float v) const noexcept { return v > 0.0f; }
};

template <typename ManagerT>
void bench_nvcomp(
    char const* stream_name,
    char const* codec_name,
    ManagerT& mgr,
    uint8_t const* d_in,
    size_t in_bytes,
    cuda::stream_ref stream
) {
    auto cfg = mgr.configure_compression(in_bytes);
    auto d_out = cuda::make_device_buffer<uint8_t>(stream, stream.device(), cfg.max_compressed_buffer_size, cuda::no_init);
    // Warmup
    mgr.compress(d_in, d_out.data(), cfg);
    stream.sync();
    size_t const comp_bytes = mgr.get_compressed_output_size(d_out.data());

    // Timed 5 reps
    cudaEvent_t ev0, ev1;
    CUDDL_CUDA_CALL(cudaEventCreate(&ev0));
    CUDDL_CUDA_CALL(cudaEventCreate(&ev1));
    std::vector<double> ms_list;
    for (int rep = 0; rep < 5; ++rep) {
        CUDDL_CUDA_CALL(cudaEventRecord(ev0, stream.get()));
        mgr.compress(d_in, d_out.data(), cfg);
        CUDDL_CUDA_CALL(cudaEventRecord(ev1, stream.get()));
        CUDDL_CUDA_CALL(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDDL_CUDA_CALL(cudaEventElapsedTime(&ms, ev0, ev1));
        ms_list.push_back(static_cast<double>(ms));
    }
    std::sort(ms_list.begin(), ms_list.end());
    double const med_ms = ms_list[2];
    double const ratio = static_cast<double>(in_bytes) / static_cast<double>(comp_bytes);
    double const gb_s = (static_cast<double>(in_bytes) / 1e9) / (med_ms / 1e3);
    std::printf(
        "{\"stream\":\"%s\",\"codec\":\"%s\",\"in_bytes\":%zu,\"out_bytes\":%zu,\"ratio\":%.3f,\"b_per_pair\":%.3f,\"ms\":%.3f,\"gb_s\":%.2f}\n",
        stream_name, codec_name, in_bytes, comp_bytes, ratio,
        4.0 / ratio, med_ms, gb_s
    );
    std::fflush(stdout);
    CUDDL_CUDA_CALL(cudaEventDestroy(ev0));
    CUDDL_CUDA_CALL(cudaEventDestroy(ev1));
}

}  // namespace

int main(int argc, char** argv) {
    std::string db_path = "refseq-full.cuddl";
    uint32_t max_genomes = 8192U;
    if (argc >= 2) db_path = argv[1];
    if (argc >= 3) max_genomes = static_cast<uint32_t>(std::stoul(argv[2]));

    auto file = CUDDL_UNWRAP(cuddl::reference_database_file::load(db_path));
    uint32_t const total_refs = file.metadata().reference_count;
    uint32_t const n = std::min(total_refs, max_genomes);
    std::fprintf(stderr, "Loaded %s: total=%u, using n=%u\n", db_path.c_str(), total_refs, n);

    // Sample evenly across total_refs to preserve corpus-wide species density!
    std::vector<uint16_t> rows(static_cast<size_t>(n) * k_buckets);
    auto const& all_rows = file.rows();
    for (uint32_t i = 0; i < n; ++i) {
        size_t const src_idx = (static_cast<size_t>(i) * total_refs) / n;
        std::copy_n(all_rows.data() + src_idx * k_buckets, k_buckets, rows.data() + static_cast<size_t>(i) * k_buckets);
    }

    cuda::stream stream{cuda::devices[0]};
    auto d_rows = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto d_planes = cuda::make_device_buffer<uint32_t>(stream, stream.device(), static_cast<size_t>(n) * (k_buckets / 2U), cuda::no_init);
    auto const plane_blocks = cuddl::detail::warp_grid_blocks(static_cast<size_t>(n) * (k_buckets / 32U));
    cuddl::detail::build_score_planes_kernel<k_buckets><<<plane_blocks, cuddl::detail::block_size, 0, stream.get()>>>(
        d_rows.data(), n, d_planes.data()
    );

    uint64_t const pairs = static_cast<uint64_t>(n) * (n - 1U) / 2ULL;
    auto d_counts = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(stream, stream.device(), pairs, cuda::no_init);
    auto d_ctr = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    CUDDL_CUDA_CALL(cudaMemsetAsync(d_ctr.data(), 0, sizeof(uint32_t), stream.get()));
    auto const sm_count = static_cast<uint32_t>(stream.device().attribute(cuda::device_attributes::multiprocessor_count));
    constexpr uint32_t group = cuddl::detail::all_pairs_query_group<k_buckets>;
    constexpr size_t smem = cuddl::detail::all_pairs_dynamic_bytes<k_buckets>;
    auto* k = cuddl::detail::refine_all_pairs_kernel<k_buckets, cuddl::packed_pairwise_counts, true, group, true, false>;
    CUDDL_CUDA_CALL(cudaFuncSetAttribute(reinterpret_cast<void const*>(k), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    k<<<sm_count, cuddl::detail::all_pairs_block_size, smem, stream.get()>>>(
        d_planes.data(), 0U, n, d_planes.data(), n, d_counts.data(), nullptr, d_ctr.data(), nullptr
    );

    auto d_wkid = cuda::make_device_buffer<float>(stream, stream.device(), pairs, cuda::no_init);
    auto d_eh = cuda::make_device_buffer<uint32_t>(stream, stream.device(), pairs, cuda::no_init);
    size_t const bm_words = (pairs + 31U) / 32U;
    auto d_bm = cuda::make_device_buffer<uint32_t>(stream, stream.device(), bm_words, cuda::no_init);
    auto d_bm20 = cuda::make_device_buffer<uint32_t>(stream, stream.device(), bm_words, cuda::no_init);
    auto d_nz = cuda::make_device_buffer<uint64_t>(stream, stream.device(), 1U, cuda::no_init);
    auto d_m20 = cuda::make_device_buffer<uint64_t>(stream, stream.device(), 1U, cuda::no_init);
    CUDDL_CUDA_CALL(cudaMemsetAsync(d_nz.data(), 0, sizeof(uint64_t), stream.get()));
    CUDDL_CUDA_CALL(cudaMemsetAsync(d_m20.data(), 0, sizeof(uint64_t), stream.get()));

    build_wkid_and_eh_kernel<<<(pairs + 255U) / 256U, 256U, 0, stream.get()>>>(
        d_counts.data(), pairs, d_wkid.data(), d_eh.data(), d_bm.data(), d_bm20.data(), d_nz.data(), d_m20.data()
    );
    uint64_t h_nz = 0, h_m20 = 0;
    CUDDL_CUDA_CALL(cudaMemcpyAsync(&h_nz, d_nz.data(), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.get()));
    CUDDL_CUDA_CALL(cudaMemcpyAsync(&h_m20, d_m20.data(), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();

    double const nz_frac = static_cast<double>(h_nz) / static_cast<double>(pairs);
    double const m20_frac = static_cast<double>(h_m20) / static_cast<double>(pairs);
    std::fprintf(stderr, "pairs=%llu, nz=%llu (%.2f%%), m20=%llu (%.2f%%)\n",
                 static_cast<unsigned long long>(pairs),
                 static_cast<unsigned long long>(h_nz), nz_frac * 100.0,
                 static_cast<unsigned long long>(h_m20), m20_frac * 100.0);

    // Benchmark cuDDL sparse_similarity compaction (bitmap + cub::DeviceSelect::If)
    auto d_packed = cuda::make_device_buffer<float>(stream, stream.device(), pairs, cuda::no_init);
    auto d_num = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    size_t temp_bytes = 0;
    cub::DeviceSelect::If(nullptr, temp_bytes, d_wkid.data(), d_packed.data(), d_num.data(), static_cast<int64_t>(pairs), nonzero_f32{}, stream.get());
    auto d_temp = cuda::make_device_buffer<uint8_t>(stream, stream.device(), temp_bytes, cuda::no_init);

    auto run_sparse_compact = [&]() {
        build_wkid_and_eh_kernel<<<(pairs + 255U) / 256U, 256U, 0, stream.get()>>>(
            d_counts.data(), pairs, d_wkid.data(), d_eh.data(), d_bm.data(), d_bm20.data(), d_nz.data(), d_m20.data()
        );
        cub::DeviceSelect::If(d_temp.data(), temp_bytes, d_wkid.data(), d_packed.data(), d_num.data(), static_cast<int64_t>(pairs), nonzero_f32{}, stream.get());
    };
    run_sparse_compact();
    stream.sync();
    cudaEvent_t ev0, ev1;
    CUDDL_CUDA_CALL(cudaEventCreate(&ev0));
    CUDDL_CUDA_CALL(cudaEventCreate(&ev1));
    std::vector<double> ms_sparse;
    for (int rep = 0; rep < 5; ++rep) {
        CUDDL_CUDA_CALL(cudaEventRecord(ev0, stream.get()));
        cub::DeviceSelect::If(d_temp.data(), temp_bytes, d_wkid.data(), d_packed.data(), d_num.data(), static_cast<int64_t>(pairs), nonzero_f32{}, stream.get());
        CUDDL_CUDA_CALL(cudaEventRecord(ev1, stream.get()));
        CUDDL_CUDA_CALL(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDDL_CUDA_CALL(cudaEventElapsedTime(&ms, ev0, ev1));
        ms_sparse.push_back(static_cast<double>(ms));
    }
    std::sort(ms_sparse.begin(), ms_sparse.end());
    size_t const in_bytes = pairs * sizeof(float);
    size_t const sparse_out = 16U + bm_words * sizeof(uint32_t) + h_nz * sizeof(float);
    double const sp_ratio = static_cast<double>(in_bytes) / static_cast<double>(sparse_out);
    std::printf(
        "{\"stream\":\"wkid_f32\",\"codec\":\"cuddl_sparse_bitmap\",\"in_bytes\":%zu,\"out_bytes\":%zu,\"ratio\":%.3f,\"b_per_pair\":%.3f,\"ms\":%.3f,\"gb_s\":%.2f,\"nz_pct\":%.2f,\"m20_pct\":%.2f}\n",
        in_bytes, sparse_out, sp_ratio, 4.0 / sp_ratio, ms_sparse[2],
        (static_cast<double>(in_bytes) / 1e9) / (ms_sparse[2] / 1e3), nz_frac * 100.0, m20_frac * 100.0
    );

    // Also report m20 formats (analytical byte counts on the exact measured m20 density)
    size_t const m20_12b = h_m20 * 12U;
    size_t const m20_csr8 = (n + 1U) * 8U + h_m20 * 8U;
    size_t const m20_bm = 16U + bm_words * 4U + h_m20 * 4U;
    std::printf(
        "{\"stream\":\"m20_formats\",\"pairs\":%llu,\"m20\":%llu,\"edge12_b_pair\":%.3f,\"csr8_b_pair\":%.3f,\"bitmap_b_pair\":%.3f}\n",
        static_cast<unsigned long long>(pairs), static_cast<unsigned long long>(h_m20),
        static_cast<double>(m20_12b) / pairs,
        static_cast<double>(m20_csr8) / pairs,
        static_cast<double>(m20_bm) / pairs
    );

    // Benchmark nvCOMP codecs on wkid_f32 AND on lossless 4-byte (E, H) uint32!
    constexpr size_t chunk = 1U << 16U;  // 64 KiB chunks
    {
        nvcompBatchedBitcompCompressOpts_t opts = nvcompBatchedBitcompCompressDefaultOpts;
        opts.data_type = NVCOMP_TYPE_UINT;
        nvcomp::BitcompManager m_bc(chunk, opts, nvcompBatchedBitcompDecompressDefaultOpts, stream.get());
        bench_nvcomp("wkid_f32", "nvcomp_bitcomp", m_bc, reinterpret_cast<uint8_t const*>(d_wkid.data()), in_bytes, stream);
        bench_nvcomp("eh_u32", "nvcomp_bitcomp", m_bc, reinterpret_cast<uint8_t const*>(d_eh.data()), in_bytes, stream);
    }
    {
        nvcomp::CascadedManager m_cas(chunk, nvcompBatchedCascadedCompressDefaultOpts, nvcompBatchedCascadedDecompressDefaultOpts, stream.get());
        bench_nvcomp("wkid_f32", "nvcomp_cascaded", m_cas, reinterpret_cast<uint8_t const*>(d_wkid.data()), in_bytes, stream);
        bench_nvcomp("eh_u32", "nvcomp_cascaded", m_cas, reinterpret_cast<uint8_t const*>(d_eh.data()), in_bytes, stream);
    }
    {
        nvcomp::ANSManager m_ans(chunk, nvcompBatchedANSCompressDefaultOpts, nvcompBatchedANSDecompressDefaultOpts, stream.get());
        bench_nvcomp("wkid_f32", "nvcomp_ans", m_ans, reinterpret_cast<uint8_t const*>(d_wkid.data()), in_bytes, stream);
        bench_nvcomp("eh_u32", "nvcomp_ans", m_ans, reinterpret_cast<uint8_t const*>(d_eh.data()), in_bytes, stream);
    }
    {
        nvcomp::ZstdManager m_zstd(chunk, nvcompBatchedZstdCompressDefaultOpts, nvcompBatchedZstdDecompressDefaultOpts, stream.get());
        bench_nvcomp("wkid_f32", "nvcomp_zstd", m_zstd, reinterpret_cast<uint8_t const*>(d_wkid.data()), in_bytes, stream);
        bench_nvcomp("eh_u32", "nvcomp_zstd", m_zstd, reinterpret_cast<uint8_t const*>(d_eh.data()), in_bytes, stream);
    }
    {
        nvcomp::LZ4Manager m_lz4(chunk, nvcompBatchedLZ4CompressDefaultOpts, nvcompBatchedLZ4DecompressDefaultOpts, stream.get());
        bench_nvcomp("wkid_f32", "nvcomp_lz4", m_lz4, reinterpret_cast<uint8_t const*>(d_wkid.data()), in_bytes, stream);
        bench_nvcomp("eh_u32", "nvcomp_lz4", m_lz4, reinterpret_cast<uint8_t const*>(d_eh.data()), in_bytes, stream);
    }
    CUDDL_CUDA_CALL(cudaEventDestroy(ev0));
    CUDDL_CUDA_CALL(cudaEventDestroy(ev1));
    return 0;
}
