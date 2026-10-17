// Inflation alone, on the SMs (nvCOMP's CUDA backend) against the hardware decompression engine
// (cuMemBatchDecompressAsync), over the same DEFLATE members resident in device memory.
//
//   cuddl-inflate-benchmark -a "input=[DIR,...]" [--json out.json]
//
// Each input is a directory of gzip files: single-member gzip as NCBI distributes it, or BGZF,
// whose every block is a member. Both backends inflate the members whose output fits the engine's
// limit (4 MiB on a B300), so they do identical work; nvbench reports the inflated bytes per
// second as global memory writes. Members above the limit, which only the SMs can inflate, are
// counted on stderr and left out.
#include <cuddl/detail/device_gzip.cuh>
#include <nvbench/nvbench.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gz = cuddl::detail::device_gzip;
namespace fs = std::filesystem;

/// Members per launch, as the build pipeline batches them.
constexpr size_t batch_members = 65536;

/// One directory's engine-sized members, uploaded once and inflated in place on every run.
struct corpus {
    cuda::device_buffer<char> compressed, inflated;
    cuda::device_buffer<void const*> sources;
    cuda::device_buffer<void*> outputs;
    cuda::device_buffer<size_t> source_bytes, output_bytes, actual;
    cuda::device_buffer<uint32_t> engine_actual;
    cuda::device_buffer<nvcompStatus_t> statuses;
    cuda::device_buffer<char> temp;
    std::vector<CUmemDecompressParams> params;
    std::vector<size_t> expected;
    size_t temp_bytes = 0, inflated_bytes = 0;
};

std::unique_ptr<corpus> load(std::string const& dir, cuda::stream_ref stream, size_t limit) {
    std::vector<fs::path> files;
    for (auto const& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) files.push_back(entry.path());
    }
    std::ranges::sort(files);
    std::vector<char> host;
    // Per member: deflate offset into host, deflate bytes, inflated bytes.
    std::vector<size_t> at, source_bytes, output_bytes;
    size_t skipped = 0, skipped_bytes = 0, kept_bytes = 0;
    for (auto const& path : files) {
        auto const probe = cuddl::detail::probe_gzip_file(path);
        if (!probe.device || probe.plain) continue;
        std::ifstream in(path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        auto const base = host.size();
        host.insert(host.end(), bytes.begin(), bytes.end());
        auto const add = [&](size_t deflate_at, size_t deflate_bytes, size_t inflated) {
            if (inflated > limit) {
                ++skipped;
                skipped_bytes += inflated;
                return;
            }
            at.push_back(base + deflate_at);
            source_bytes.push_back(deflate_bytes);
            output_bytes.push_back(inflated);
            kept_bytes += inflated;
        };
        if (probe.bgzf) {
            (void)cuddl::detail::walk_bgzf(
                reinterpret_cast<unsigned char const*>(bytes.data()),
                bytes.size(),
                [&](uint64_t deflate_at, uint64_t deflate_bytes, uint32_t inflated) {
                    add(deflate_at, deflate_bytes, inflated);
                }
            );
        } else if (auto const offset = gz::deflate_offset(bytes.data(), bytes.size()); offset) {
            add(offset, bytes.size() - offset - 8, probe.isize);
        }
    }
    if (at.empty()) throw std::runtime_error(dir + ": no gzip member fits the engine");
    std::cerr << dir << ": " << at.size() << " members, " << kept_bytes << " inflated bytes; "
              << skipped << " members (" << skipped_bytes << " bytes) above the " << limit
              << "-byte engine limit left out\n";

    auto const device = cuda::devices[0];
    auto const pool = gz::hardware_inflate_pool(device);
    auto data = std::make_unique<corpus>(corpus{
        .compressed = cuda::make_buffer<char>(stream, pool, host.size(), cuda::no_init),
        .inflated = cuda::make_buffer<char>(stream, pool, kept_bytes, cuda::no_init),
        .sources = cuda::make_device_buffer<void const*>(stream, device, at.size(), cuda::no_init),
        .outputs = cuda::make_device_buffer<void*>(stream, device, at.size(), cuda::no_init),
        .source_bytes = cuda::make_device_buffer<size_t>(stream, device, source_bytes),
        .output_bytes = cuda::make_device_buffer<size_t>(stream, device, output_bytes),
        .actual = cuda::make_device_buffer<size_t>(stream, device, at.size(), cuda::no_init),
        // The engine writes each member's inflated size, so these live in its pool too.
        .engine_actual = cuda::make_buffer<uint32_t>(stream, pool, at.size(), cuda::no_init),
        .statuses =
            cuda::make_device_buffer<nvcompStatus_t>(stream, device, at.size(), cuda::no_init),
        .temp = cuda::make_device_buffer<char>(stream, device, 0, cuda::no_init),
        .params = {},
        .expected = output_bytes,
        .temp_bytes = 0,
        .inflated_bytes = kept_bytes,
    });
    CUDDL_CUDA_CALL(cudaMemcpyAsync(
        data->compressed.data(), host.data(), host.size(), cudaMemcpyHostToDevice, stream.get()
    ));
    std::vector<void const*> sources(at.size());
    std::vector<void*> outputs(at.size());
    size_t out = 0, batch_total = 0;
    for (size_t m = 0; m < at.size(); ++m) {
        sources[m] = data->compressed.data() + at[m];
        outputs[m] = data->inflated.data() + out;
        out += output_bytes[m];
        CUmemDecompressParams params{};
        params.src = sources[m];
        params.srcNumBytes = source_bytes[m];
        params.dst = outputs[m];
        params.dstNumBytes = output_bytes[m];
        params.dstActBytes = reinterpret_cast<cuuint32_t*>(data->engine_actual.data() + m);
        params.algo = CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE;
        data->params.push_back(params);
        if (m % batch_members == 0) batch_total = 0;
        batch_total += output_bytes[m];
        if (m % batch_members == batch_members - 1 || m + 1 == at.size()) {
            size_t temp = 0;
            auto const members = m % batch_members + 1;
            if (nvcompBatchedDeflateDecompressGetTempSizeAsync(
                    members, limit, nvcompBatchedDeflateDecompressDefaultOpts, &temp, batch_total
                ) != nvcompSuccess) {
                throw std::runtime_error("nvCOMP temp size query failed");
            }
            data->temp_bytes = std::max(data->temp_bytes, temp);
        }
    }
    CUDDL_CUDA_CALL(cudaMemcpyAsync(
        data->sources.data(),
        sources.data(),
        sources.size() * sizeof(void const*),
        cudaMemcpyHostToDevice,
        stream.get()
    ));
    CUDDL_CUDA_CALL(cudaMemcpyAsync(
        data->outputs.data(),
        outputs.data(),
        outputs.size() * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream.get()
    ));
    data->temp = cuda::make_device_buffer<char>(stream, device, data->temp_bytes, cuda::no_init);
    stream.sync();
    return data;
}

void inflate(corpus& data, bool engine, cuda::stream_ref stream) {
    static auto const batch_decompress = gz::batch_decompress_entry();
    auto const count = data.params.size();
    for (size_t first = 0; first < count; first += batch_members) {
        auto const members = std::min(batch_members, count - first);
        if (engine) {
            size_t error_index = 0;
            if (auto const status = batch_decompress(
                    data.params.data() + first, members, 0, &error_index, stream.get()
                );
                status != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "cuMemBatchDecompressAsync failed: " + std::to_string(status) + " at member " +
                    std::to_string(first + error_index)
                );
            }
            continue;
        }
        auto options = nvcompBatchedDeflateDecompressDefaultOpts;
        options.backend = NVCOMP_DECOMPRESS_BACKEND_CUDA;
        if (nvcompBatchedDeflateDecompressAsync(
                data.sources.data() + first,
                data.source_bytes.data() + first,
                data.output_bytes.data() + first,
                data.actual.data() + first,
                members,
                data.temp.data(),
                data.temp_bytes,
                data.outputs.data() + first,
                options,
                data.statuses.data() + first,
                stream.get()
            ) != nvcompSuccess) {
            throw std::runtime_error("nvCOMP DEFLATE launch failed");
        }
    }
}

/// Inflates once and checks that every member came out at its expected size.
void verify(corpus& data, bool engine, cuda::stream_ref stream) {
    auto const count = data.params.size();
    std::vector<size_t> sizes(count);
    CUDDL_CUDA_CALL(cudaMemsetAsync(data.actual.data(), 0, count * sizeof(size_t), stream.get()));
    CUDDL_CUDA_CALL(
        cudaMemsetAsync(data.engine_actual.data(), 0, count * sizeof(uint32_t), stream.get())
    );
    inflate(data, engine, stream);
    if (engine) {
        std::vector<uint32_t> engine_sizes(count);
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            engine_sizes.data(),
            data.engine_actual.data(),
            count * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));
        stream.sync();
        std::ranges::copy(engine_sizes, sizes.begin());
    } else {
        std::vector<nvcompStatus_t> statuses(count);
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            statuses.data(),
            data.statuses.data(),
            count * sizeof(nvcompStatus_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));
        CUDDL_CUDA_CALL(cudaMemcpyAsync(
            sizes.data(),
            data.actual.data(),
            count * sizeof(size_t),
            cudaMemcpyDeviceToHost,
            stream.get()
        ));
        stream.sync();
        if (std::ranges::any_of(statuses, [](auto s) { return s != nvcompSuccess; })) {
            throw std::runtime_error("nvCOMP reported a failed member");
        }
    }
    if (sizes != data.expected) {
        throw std::runtime_error("an inflated size differs from its trailer");
    }
}

void inflate_benchmark(nvbench::state& state) {
    auto const& dir = state.get_string("input");
    bool const engine = state.get_string("backend") == "engine";
    if (dir.empty()) {
        state.skip("pass the inputs as -a \"input=[DIR,...]\"");
        return;
    }
    static auto const limit = gz::hardware_inflate_limit(cuda::devices[0].get());
    if (limit == 0 || gz::batch_decompress_entry() == nullptr) {
        state.skip("this device has no hardware decompression engine");
        return;
    }
    auto const stream = cuda::stream_ref{state.get_cuda_stream().get_stream()};
    // One upload per directory, shared by both backends, on a stream that outlives them all.
    // Left to process exit: freed at static destruction, they would outlive the CUDA runtime.
    static auto& uploads = *new cuda::stream{cuda::devices[0]};
    static auto& corpora = *new std::map<std::string, std::unique_ptr<corpus>>;
    auto& data = corpora[dir];
    if (!data) data = load(dir, uploads, limit);
    verify(*data, engine, stream);
    state.add_element_count(data->params.size(), "Members");
    state.add_global_memory_writes<char>(data->inflated_bytes, "Inflated");
    // Synchronous: the driver can block the host while queueing many engine batches, which
    // nvbench's blocking kernel would read as a deadlock.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        inflate(*data, engine, cuda::stream_ref{launch.get_stream().get_stream()});
    });
}
}  // namespace

NVBENCH_BENCH(inflate_benchmark)
    .set_name("inflate")
    .add_string_axis("input", {""})
    .add_string_axis("backend", {"sm", "engine"});
