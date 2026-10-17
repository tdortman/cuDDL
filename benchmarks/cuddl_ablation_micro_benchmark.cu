// Stage micro-benchmarks for the ablation study (scripts/ablation.py): each state times one
// stage that an optimisation targets, alone, so the study can set what the optimisation gains
// there next to what it gains end to end. Inputs are the genome files in the directory named by
// CUDDL_ABLATION_GENOMES, read once into memory; states without it are skipped.
//
// - inflate: libdeflate gzip inflation of every file on one thread (DNA-tuned libdeflate).
// - load: inflation, line scanning and FASTA compaction of every file on one thread, the CPU
//   loader's work per genome (SIMD parsing, libdeflate).
// - pack: load plus packing the bases into two-bit words (SIMD packing).
// - pinned: allocating, filling and freeing one loader buffer from the device's pinned pool with
//   a stream synchronisation in between, as loaders do per genome (device-only pinned pools).
//
// Every state times CPU wall time. Run with CUDDL_ABLATE (ablation builds) to time a stage
// without its optimisation.

#include <libdeflate.h>

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuddl/detail/fastx_sequence_file.hpp>
#include <cuddl/detail/pinned_pool.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct genome {
    std::string path;
    std::vector<char> compressed;
    size_t inflated = 0;
};

/// The genomes of CUDDL_ABLATION_GENOMES, read once; empty without it.
std::vector<genome> const& genomes() {
    static auto const loaded = [] {
        std::vector<genome> result;
        char const* const directory = std::getenv("CUDDL_ABLATION_GENOMES");
        if (directory == nullptr || *directory == '\0') return result;
        for (auto const& entry : std::filesystem::directory_iterator(directory)) {
            auto const path = std::filesystem::canonical(entry.path());
            if (!std::filesystem::is_regular_file(path)) continue;
            std::ifstream file(path, std::ios::binary);
            genome g{path.string(), {std::istreambuf_iterator<char>(file), {}}, 0};
            // A gzip member's last four bytes hold its inflated size modulo 2^32.
            if (g.compressed.size() >= 18) {
                uint32_t isize = 0;
                std::memcpy(&isize, g.compressed.data() + g.compressed.size() - 4, 4);
                g.inflated = isize;
            }
            result.push_back(std::move(g));
        }
        std::ranges::sort(result, {}, &genome::path);
        return result;
    }();
    return loaded;
}

bool skip_without_genomes(nvbench::state& state) {
    if (!genomes().empty()) return false;
    state.skip("CUDDL_ABLATION_GENOMES names no genome directory");
    return true;
}

size_t compressed_bytes() {
    size_t bytes = 0;
    for (auto const& g : genomes()) bytes += g.compressed.size();
    return bytes;
}

void inflate(nvbench::state& state) {
    if (skip_without_genomes(state)) return;
    size_t largest = 0;
    for (auto const& g : genomes()) largest = std::max(largest, g.inflated);
    std::vector<char> out(largest);
    std::unique_ptr<libdeflate_decompressor, decltype(&libdeflate_free_decompressor)> decompressor(
        libdeflate_alloc_decompressor(), &libdeflate_free_decompressor
    );
    state.add_global_memory_reads(compressed_bytes(), "compressed");
    state.exec(
        nvbench::exec_tag::timer | nvbench::exec_tag::no_batch, [&](nvbench::launch&, auto& timer) {
            timer.start();
            for (auto const& g : genomes()) {
                size_t written = 0;
                auto const status = libdeflate_gzip_decompress(
                    decompressor.get(),
                    g.compressed.data(),
                    g.compressed.size(),
                    out.data(),
                    out.size(),
                    &written
                );
                if (status != LIBDEFLATE_SUCCESS)
                    throw std::runtime_error("cannot inflate " + g.path);
            }
            timer.stop();
        }
    );
}

/// Loads every genome on one thread, packing for k-mers of @p pack_k when nonzero. The files
/// come from the page cache: reading them once beforehand leaves them there.
void load_genomes(nvbench::state& state, uint32_t pack_k) {
    if (skip_without_genomes(state)) return;
    state.add_global_memory_reads(compressed_bytes(), "compressed");
    state.exec(
        nvbench::exec_tag::timer | nvbench::exec_tag::no_batch, [&](nvbench::launch&, auto& timer) {
            timer.start();
            for (auto const& g : genomes()) {
                auto loaded =
                    cuddl::detail::load_fastx_sequence_file(g.path, {}, false, false, pack_k);
                if (!loaded) throw std::runtime_error(loaded.error().message());
                if (pack_k != 0) cuddl::detail::pack_fastx_sequence_file(**loaded, pack_k);
            }
            timer.stop();
        }
    );
}

void load(nvbench::state& state) {
    load_genomes(state, 0);
}

void pack(nvbench::state& state) {
    load_genomes(state, 25);
}

void pinned(nvbench::state& state) {
    auto const bytes = static_cast<size_t>(state.get_int64("MiB")) << 20;
    cuda::stream stream{cuda::devices[0]};
    auto pool =
        cuddl::detail::device_pinned_pool(cuda::devices[0], cuddl::detail::pinned_placement::host);
    constexpr int buffers = 64;
    state.exec(
        nvbench::exec_tag::timer | nvbench::exec_tag::no_batch, [&](nvbench::launch&, auto& timer) {
            timer.start();
            for (int i = 0; i < buffers; ++i) {
                cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible> buffer(
                    stream, pool, bytes, cuda::no_init
                );
                std::memset(buffer.data(), i, bytes);
                buffer.destroy(stream);
                stream.sync();
            }
            timer.stop();
        }
    );
    state.add_element_count(buffers, "buffers");
}

}  // namespace

NVBENCH_BENCH(inflate).set_is_cpu_only(true);
NVBENCH_BENCH(load).set_is_cpu_only(true);
NVBENCH_BENCH(pack).set_is_cpu_only(true);
NVBENCH_BENCH(pinned).set_is_cpu_only(true).add_int64_axis("MiB", {8});
