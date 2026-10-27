#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuddl/cuddl.cuh>
#include <cuddl/detail/fastx_sequence_file.hpp>
#include <cuddl/fastx.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
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

void sequence_construction(nvbench::state& state) {
    constexpr size_t buckets = 2048;
    auto const stream = cuda::stream_ref{state.get_cuda_stream()};
    auto const path = state.get_string("Path");
    auto const requested_k = state.get_int64("K");
    auto const blocks_per_sm = state.get_int64("BlocksPerSM");
    auto const blacklist_path = state.get_string("Blacklist");
    std::optional<cuddl::device_blacklist> blacklist;
    auto const offset = state.get_int64("Offset");
    if (requested_k < 1 || requested_k > 31 || blocks_per_sm < 0 || blocks_per_sm > 32 ||
        offset < 0 || offset > 7) {
        throw std::runtime_error("K must be 1..31, BlocksPerSM 0..32, and Offset 0..7");
    }
    auto const k = static_cast<uint32_t>(requested_k);
    if (!blacklist_path.empty()) {
        blacklist.emplace(
            CUDDL_UNWRAP(cuddl::kmer_blacklist::load(blacklist_path, k)), buckets, stream
        );
    }
    auto const filter = blacklist ? blacklist->view() : cuda::std::nullopt;
    auto const parsed = CUDDL_UNWRAP(cuddl::parse_fasta_file(path, k));
    auto const file = CUDDL_UNWRAP(cuddl::detail::load_fastx_sequence_file(path));
    std::vector<char> bases;
    for (auto const& extent : file->extents) {
        // An invalid separator keeps windows from crossing record boundaries.
        if (!bases.empty()) bases.push_back('N');
        for (auto cursor = extent.begin; cursor != extent.end; ++cursor) {
            if (!cuddl::detail::fastx_is_sequence_whitespace(*cursor)) bases.push_back(*cursor);
        }
    }
    auto const sequence_size = bases.size();
    bases.insert(bases.begin(), static_cast<size_t>(offset), 'N');
    auto input = cuda::make_device_buffer<char>(stream, stream.device(), bases);
    auto const sequence = cuddl::device_span<char const>{input.data() + offset, sequence_size};
    auto packed = cuda::make_device_buffer<uint64_t>(stream, stream.device(), parsed.kmers);
    auto reference =
        cuda::make_device_buffer<uint32_t>(stream, stream.device(), buckets, uint32_t{0});
    auto output = cuda::make_device_buffer<uint32_t>(stream, stream.device(), buckets, uint32_t{0});
    auto const sms = stream.device().attribute(cuda::device_attributes::multiprocessor_count);
    auto launch = [&](cuda::stream_ref s) {
        auto run = [&]<bool Filter>() {
            if (blocks_per_sm == 0) {
                CUDDL_UNWRAP((cuddl::detail::launch_sequence_add<buckets>(
                    sequence, k, {output.data(), buckets}, s, filter
                )));
            } else if (sequence.size() >= k) {
                auto const windows = sequence.size() - k + 1;
                if (windows > UINT32_MAX) throw std::runtime_error("too many sequence windows");
                auto const blocks = std::min<size_t>(sms * blocks_per_sm, (windows + 2047) / 2048);
                cuddl::detail::
                    add_sequence_single_kernel<buckets, cuddl::default_register_layout, Filter>
                    <<<blocks, 256, 0, s.get()>>>(
                        sequence.data(),
                        static_cast<uint32_t>(windows),
                        k,
                        output.data(),
                        filter.value_or(cuddl::detail::blacklist_view{})
                    );
                check(cudaGetLastError());
            }
        };
        if (filter) {
            run.template operator()<true>();
        } else {
            run.template operator()<false>();
        }
    };
    for (int append = 0; append < 2; ++append) {
        CUDDL_UNWRAP(
            cuddl::detail::launch_construction<buckets>(
                packed, {reference.data(), buckets}, stream, filter
            )
        );
        launch(stream);
        if (download(output.data(), buckets, stream.get()) !=
            download(reference.data(), buckets, stream.get())) {
            throw std::runtime_error("sequence register mismatch");
        }
    }
    state.add_element_count(parsed.kmers.size(), "Kmers");
    state.add_global_memory_reads<char>(sequence.size());
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& execution, auto& timer) {
        auto const s = cuda::stream_ref{execution.get_stream()};
        cuda::fill_bytes(s, output, 0);
        timer.start();
        launch(s);
        timer.stop();
    });
}

template <bool Bucketed>
__global__ void lookup_blacklist(
    uint64_t const* input,
    size_t count,
    uint64_t const* sorted,
    size_t size,
    cuddl::detail::blacklist_view lookup,
    uint8_t* output
) {
    for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count;
         i += size_t{gridDim.x} * blockDim.x) {
        auto const key = input[i];
        if constexpr (Bucketed) {
            output[i] = lookup.contains(
                key,
                cuddl::detail::hash_kmer(key),
                cuddl::detail::bucket_of<2048>(cuddl::detail::hash_kmer(key))
            );
        } else {
            output[i] = cuda::std::binary_search(sorted, sorted + size, key);
        }
    }
}

void blacklist_lookup(nvbench::state& state) {
    auto const stream = cuda::stream_ref{state.get_cuda_stream()};
    auto const count = static_cast<size_t>(state.get_int64("Entries"));
    auto const hit_percent = static_cast<uint64_t>(state.get_int64("HitPercent"));
    auto const bucketed = state.get_int64("Bucketed") != 0;
    std::mt19937_64 rng(9231);
    auto next = [&] {
        return cuddl::kmer_blacklist::canonical(rng() & ((uint64_t{1} << 50) - 1), 25);
    };
    std::vector<uint64_t> keys(count);
    for (auto& key : keys) {
        key = next();
    }
    cuddl::kmer_blacklist source(25, std::move(keys));
    cuddl::device_blacklist lookup(source, 2048, stream);
    auto sorted = cuda::make_device_buffer<uint64_t>(stream, stream.device(), source.keys());
    std::vector<uint64_t> queries(1 << 20);
    std::vector<uint8_t> expected(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] =
            rng() % 100 < hit_percent ? source.keys()[rng() % source.keys().size()] : next();
        expected[i] = std::binary_search(source.keys().begin(), source.keys().end(), queries[i]);
    }
    auto input = cuda::make_device_buffer<uint64_t>(stream, stream.device(), queries);
    auto output =
        cuda::make_device_buffer<uint8_t>(stream, stream.device(), queries.size(), cuda::no_init);
    auto run = [&](cudaStream_t s) {
        auto launch = [&]<bool Bucketed>() {
            lookup_blacklist<Bucketed><<<256, 256, 0, s>>>(
                input.data(),
                input.size(),
                sorted.data(),
                sorted.size(),
                *lookup.view(),
                output.data()
            );
        };
        if (bucketed) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
        check(cudaGetLastError());
    };
    run(stream.get());
    if (download(output.data(), output.size(), stream.get()) != expected) {
        throw std::runtime_error("blacklist membership mismatch");
    }
    state.add_element_count(queries.size());
    state.exec([&](nvbench::launch& execution) { run(execution.get_stream()); });
}

}  // namespace

NVBENCH_BENCH(blacklist_lookup)
    .add_int64_axis("Entries", {1651, 23020})
    .add_int64_axis("HitPercent", {0, 50})
    .add_int64_axis("Bucketed", {0, 1})
    .set_min_samples(30);

NVBENCH_BENCH(sequence_construction)
    .add_string_axis(
        "Path",
        {"data/genomes/ecoli_k12_mg1655.fna", "data/genomes/WBcel235.fna", "data/genomes/chr14.fna"}
    )
    .add_int64_axis("K", {25})
    .add_int64_axis("Offset", {0})
    .add_string_axis("Blacklist", {""})
    .add_int64_axis("BlocksPerSM", {0})  // Zero measures the production launch policy.
    .set_min_samples(30);
