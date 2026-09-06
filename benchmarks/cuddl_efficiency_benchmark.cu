#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuddl/cuddl.cuh>
#include <cuddl/fastx.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <typename T>
std::vector<T> download(T const* data, size_t size, cudaStream_t stream) {
    std::vector<T> result(size);
    check(cudaMemcpyAsync(result.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
}

// The variant parameter only changes the local winner floor. Grid and assignment match production.
template <size_t B, uint32_t Rounds>
void add(uint64_t const* input, size_t n, uint32_t* output, uint32_t blocks, cudaStream_t stream) {
    if (n == 0U) return;
    cuddl::detail::add_shared_kernel<B, cuddl::default_register_layout, Rounds>
        <<<blocks, cuddl::detail::shared_construction_block_size, 0, stream>>>(
            input, n, output, output[B], (reinterpret_cast<uintptr_t>(input) & 31U) == 0U
        );
    check(cudaGetLastError());
}

// Share parsed inputs across bucket-count instantiations.
std::vector<uint64_t> const& genome_kmers(std::string const& path) {
    static std::map<std::string, std::vector<uint64_t>> genomes;
    auto [entry, inserted] = genomes.try_emplace(path);
    if (inserted) {
        auto parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(path, 25U));
        entry->second = std::move(parsed.kmers);
    }
    if (entry->second.empty()) throw std::runtime_error("genome contains no valid 25-mers: " + path);
    return entry->second;
}

template <size_t B>
void construction(nvbench::state& state) {
    auto const stream = cuda::stream_ref{state.get_cuda_stream()};
    auto n = static_cast<size_t>(state.get_int64("Items"));
    auto const distribution = state.get_string("Input");
    if (distribution != "random" && distribution != "duplicates" && distribution != "repeated" &&
        distribution != "ecoli" && distribution != "worm" && distribution != "chr14" &&
        !distribution.starts_with("fasta="))
        throw std::runtime_error("unknown construction input");
    auto const rounds = state.get_int64("FloorRounds");
    auto const offset = static_cast<size_t>(state.get_int64("Misaligned"));
    auto const start_percent = state.get_int64("StartPercent");
    if (start_percent < 0 || start_percent > 100)
        throw std::runtime_error("StartPercent must be between 0 and 100");
    std::vector<uint64_t> host;
    if (distribution == "ecoli" || distribution == "worm" || distribution == "chr14" ||
        distribution.starts_with("fasta=")) {
        auto const path = distribution.starts_with("fasta=") ? distribution.substr(6)
                        : distribution == "ecoli" ? "data/genomes/ecoli_k12_mg1655.fna"
                        : distribution == "worm" ? "data/genomes/WBcel235.fna"
                                                 : "data/genomes/chr14.fna";
        auto const& genome = genome_kmers(path);
        if (n == 0U) n = genome.size();  // Items=0 retains the full-genome benchmark.
        if (n > genome.size()) throw std::runtime_error("Items exceeds genome size");
        auto const begin = (genome.size() - n) * static_cast<size_t>(start_percent) / 100U;
        host.resize(n + offset);
        std::copy_n(genome.begin() + begin, n, host.begin() + offset);
    } else {
        if (start_percent != 0) throw std::runtime_error("StartPercent requires a genome");
        host.resize(n + offset);
        std::mt19937_64 rng(4242);
        for (auto& x : host) x = distribution == "repeated" ? 123U : rng() & ((1ULL << 50U) - 1U);
        if (distribution == "duplicates") {
            for (size_t i = offset; i < host.size(); ++i)
                host[i] = host[offset + (i - offset) % 4096U];
        }
    }
    auto input = cuda::make_device_buffer<uint64_t>(stream, stream.device(), host);
    auto reference =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), B + 1U, uint32_t{0});
    auto output = cuda::make_device_buffer<uint32_t>(stream, stream.device(), B + 1U, uint32_t{0});
    auto const sms = stream.device().attribute(cuda::device_attributes::multiprocessor_count);
    auto const blocks = static_cast<uint32_t>(std::min<size_t>(2U * sms, (n + 3071U) / 3072U));
    auto launch = [&](cudaStream_t execution_stream) {
        auto const* data = input.data() + offset;
        switch (rounds) {
            case 0:
                return add<B, 0>(data, n, output.data(), blocks, execution_stream);
            case 1:
                return add<B, 1>(data, n, output.data(), blocks, execution_stream);
            case 2:
                return add<B, 2>(data, n, output.data(), blocks, execution_stream);
            case 4:
                return add<B, 4>(data, n, output.data(), blocks, execution_stream);
            case 8:
                return add<B, 8>(data, n, output.data(), blocks, execution_stream);
            case 12:
                return add<B, 12>(data, n, output.data(), blocks, execution_stream);
            case 16:
                return add<B, 16>(data, n, output.data(), blocks, execution_stream);
            case 24:
                return add<B, 24>(data, n, output.data(), blocks, execution_stream);
            case 32:
                return add<B, 32>(data, n, output.data(), blocks, execution_stream);
            case 48:
                return add<B, 48>(data, n, output.data(), blocks, execution_stream);
            case 64:
                return add<B, 64>(data, n, output.data(), blocks, execution_stream);
            case 96:
                return add<B, 96>(data, n, output.data(), blocks, execution_stream);
            case 128:
                return add<B, 128>(data, n, output.data(), blocks, execution_stream);
            default:
                throw std::runtime_error("unsupported FloorRounds sweep value");
        }
    };
    add<B, 0>(input.data() + offset, n, reference.data(), blocks, stream.get());
    launch(stream.get());
    auto const expected = download(reference.data(), B + 1U, stream.get());
    if (download(output.data(), B + 1U, stream.get()) != expected)
        throw std::runtime_error("construction register/count/saturation mismatch");
    // Append the same stream: verify the merge and tie counts on populated sketches too.
    add<B, 0>(input.data() + offset, n, reference.data(), blocks, stream.get());
    launch(stream.get());
    if (download(output.data(), B + 1U, stream.get()) !=
        download(reference.data(), B + 1U, stream.get()))
        throw std::runtime_error("construction append mismatch");
    state.add_element_count(n, "Kmers");
    state.add_global_memory_reads<uint64_t>(n);
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& execution, auto& timer) {
        // Resident add timing, matching cuddl-hll-comparison. Clear is explicitly excluded.
        check(
            cudaMemsetAsync(output.data(), 0, (B + 1U) * sizeof(uint32_t), execution.get_stream())
        );
        timer.start();
        launch(execution.get_stream());
        timer.stop();
    });
}
void construction_efficiency(nvbench::state& state) {
    switch (state.get_int64("Buckets")) {
        case 2048:
            return construction<2048>(state);
        case 4096:
            return construction<4096>(state);
        case 8192:
            return construction<8192>(state);
        default:
            throw std::runtime_error("Buckets must be 2048, 4096, or 8192");
    }
}

}  // namespace

NVBENCH_BENCH(construction_efficiency)
    .add_int64_axis("Buckets", {2048, 4096, 8192})
    .add_int64_axis("Items", {1048576, 16777216, 67108864})
    .add_string_axis("Input", {"random", "duplicates", "repeated"})
    .add_int64_axis("FloorRounds", {0, 8, 32})
    .add_int64_axis("Misaligned", {0})
    .add_int64_axis("StartPercent", {0})
    .set_min_samples(30);
