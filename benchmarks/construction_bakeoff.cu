#include <cuddl/cuda_error.hpp>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t bucket_count = 2048;
constexpr std::uint32_t block_size = 256;
constexpr std::uint64_t seed = 42;

constexpr std::uint32_t kmer_length = 25;
constexpr std::uint64_t kmer_mask = (1ULL << (2U * kmer_length)) - 1U;


__host__ __device__ constexpr std::uint64_t splitmix64(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ std::uint16_t score(std::uint64_t const hash) {
    auto const exponent = static_cast<std::uint16_t>(min(__clzll(hash | 1ULL), 62));
    auto const mantissa = static_cast<std::uint16_t>((hash >> 43) & 0x03ffULL);
    return static_cast<std::uint16_t>(1U + (exponent << 10U) + (mantissa ^ 0x03ffU));
}

constexpr std::uint64_t reverse_complement(std::uint64_t value) {
    std::uint64_t result = 0;
    for (std::uint32_t base = 0; base < kmer_length; ++base) {
        result = (result << 2U) | ((value & 3U) ^ 2U);
        value >>= 2U;
    }
    return result;
}

__device__ void atomic_max_u16(std::uint16_t* const address, std::uint16_t const value) {
    auto const raw_address = reinterpret_cast<std::uintptr_t>(address);
    auto* const word = reinterpret_cast<unsigned int*>(raw_address & ~std::uintptr_t{3});
    auto const shift = static_cast<unsigned int>((raw_address & 2U) * 8U);
    auto observed = *word;
    while (value > ((observed >> shift) & 0xffffU)) {
        auto const replacement =
            (observed & ~(0xffffU << shift)) | (static_cast<unsigned int>(value) << shift);
        auto const previous = atomicCAS(word, observed, replacement);
        if (previous == observed) {
            break;
        }
        observed = previous;
    }
}

__global__ void global_atomic_kernel(
    std::uint64_t const* const inputs,
    std::uint64_t const item_count,
    std::uint16_t* const sketch
) {
    auto const stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < item_count;
         index += stride) {
        auto const hash = splitmix64(inputs[index] ^ seed);
        auto const bucket = hash & (bucket_count - 1U);
        atomic_max_u16(sketch + bucket, score(hash));
    }
}

__global__ void cta_local_kernel(
    std::uint64_t const* const inputs,
    std::uint64_t const item_count,
    std::uint16_t* const partial_sketches
) {
    __shared__ std::uint16_t sketch[bucket_count];
    for (auto bucket = threadIdx.x; bucket < bucket_count; bucket += blockDim.x) {
        sketch[bucket] = 0;
    }
    __syncthreads();

    auto const chunk = (item_count + gridDim.x - 1U) / gridDim.x;
    auto const begin = static_cast<std::uint64_t>(blockIdx.x) * chunk;
    auto const end = min(begin + chunk, item_count);
    for (auto index = begin + threadIdx.x; index < end; index += blockDim.x) {
        auto const hash = splitmix64(inputs[index] ^ seed);
        atomic_max_u16(sketch + (hash & (bucket_count - 1U)), score(hash));
    }
    __syncthreads();

    auto* const output = partial_sketches + static_cast<std::uint64_t>(blockIdx.x) * bucket_count;
    for (auto bucket = threadIdx.x; bucket < bucket_count; bucket += blockDim.x) {
        output[bucket] = sketch[bucket];
    }
}

__global__ void merge_kernel(
    std::uint16_t const* const partial_sketches,
    std::uint32_t const partial_count,
    std::uint16_t* const sketch
) {
    for (auto bucket = blockIdx.x * blockDim.x + threadIdx.x;
         bucket < bucket_count;
         bucket += blockDim.x * gridDim.x) {
        std::uint16_t maximum = 0;
        for (std::uint32_t partial = 0; partial < partial_count; ++partial) {
            maximum = max(maximum, partial_sketches[static_cast<std::uint64_t>(partial) * bucket_count + bucket]);
        }
        sketch[bucket] = maximum;
    }
}

class event_timer {
  public:
    event_timer() {
        CUDDL_CUDA_CALL(cudaEventCreate(&start_));
        CUDDL_CUDA_CALL(cudaEventCreate(&stop_));
    }

    ~event_timer() {
        CUDDL_CUDA_ABORT(cudaEventDestroy(start_));
        CUDDL_CUDA_ABORT(cudaEventDestroy(stop_));
    }

    void start() { CUDDL_CUDA_CALL(cudaEventRecord(start_)); }

    double stop() {
        CUDDL_CUDA_CALL(cudaEventRecord(stop_));
        CUDDL_CUDA_CALL(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDDL_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

  private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

std::uint64_t parse_items(int argc, char** argv) {
    std::uint64_t items = 1ULL << 24;
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view{argv[index]} == "--items") {
            auto const* begin = argv[index + 1];
            auto const* end = begin + std::char_traits<char>::length(begin);
            if (auto const [ptr, error] = std::from_chars(begin, end, items);
                error != std::errc{} || ptr != end) {
                throw std::runtime_error("invalid --items value");
            }
        }
    }
    return items;
}

bool has_flag(int argc, char** argv, std::string_view flag) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view{argv[index]} == flag) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto const item_count = parse_items(argc, argv);
        auto const check_only = has_flag(argc, argv, "--check");
        auto const profile_global = has_flag(argc, argv, "--profile-global");
        auto const profile_cta = has_flag(argc, argv, "--profile-cta");
        auto const profile_merge = has_flag(argc, argv, "--profile-merge");
        auto const profile_mode_count =
            static_cast<int>(profile_global) + static_cast<int>(profile_cta) +
            static_cast<int>(profile_merge);
        if (profile_mode_count > 1) {
            throw std::runtime_error("select only one --profile-* mode");
        }
        int device = 0;
        CUDDL_CUDA_CALL(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDDL_CUDA_CALL(cudaGetDeviceProperties(&properties, device));

        auto const block_count = std::min<std::uint32_t>(properties.multiProcessorCount * 4U, 512U);
        auto const partial_bytes =
            static_cast<std::uint64_t>(block_count) * bucket_count * sizeof(std::uint16_t);
        auto const device_bytes = item_count * sizeof(std::uint64_t) +
                                  2U * bucket_count * sizeof(std::uint16_t) + partial_bytes;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDDL_CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
        auto const memory_cap = free_bytes * 4U / 5U;
        if (device_bytes > memory_cap) {
            std::printf(
                "status=skipped reason=memory_cap items=%llu required_bytes=%llu "
                "free_bytes=%llu cap_bytes=%llu\n",
                static_cast<unsigned long long>(item_count),
                static_cast<unsigned long long>(device_bytes),
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(memory_cap));
            return EXIT_SUCCESS;
        }

        std::vector<std::uint64_t> inputs(item_count);
        for (std::uint64_t index = 0; index < item_count; ++index) {
            auto const forward = splitmix64(index ^ seed) & kmer_mask;
            inputs[index] = std::max(forward, reverse_complement(forward));
        }

        std::uint64_t* device_inputs = nullptr;
        std::uint16_t* global_sketch = nullptr;
        std::uint16_t* local_sketch = nullptr;
        std::uint16_t* partial_sketches = nullptr;
        CUDDL_CUDA_CALL(cudaMalloc(&device_inputs, item_count * sizeof(*device_inputs)));
        CUDDL_CUDA_CALL(cudaMalloc(&global_sketch, bucket_count * sizeof(*global_sketch)));
        CUDDL_CUDA_CALL(cudaMalloc(&local_sketch, bucket_count * sizeof(*local_sketch)));
        CUDDL_CUDA_CALL(cudaMalloc(&partial_sketches, partial_bytes));
        CUDDL_CUDA_CALL(cudaMemcpy(device_inputs,
                                   inputs.data(),
                                   item_count * sizeof(*device_inputs),
                                   cudaMemcpyHostToDevice));

        auto run_global = [&] {
            CUDDL_CUDA_CALL(cudaMemset(global_sketch, 0, bucket_count * sizeof(*global_sketch)));
            global_atomic_kernel<<<block_count, block_size>>>(device_inputs, item_count, global_sketch);
            CUDDL_CUDA_CALL(cudaGetLastError());
        };
        auto run_local_build = [&] {
            cta_local_kernel<<<block_count, block_size>>>(device_inputs, item_count, partial_sketches);
            CUDDL_CUDA_CALL(cudaGetLastError());
        };
        auto run_local_merge = [&] {
            merge_kernel<<<(bucket_count + block_size - 1U) / block_size, block_size>>>(
                partial_sketches, block_count, local_sketch);
            CUDDL_CUDA_CALL(cudaGetLastError());
        };
        auto run_local = [&] {
            run_local_build();
            run_local_merge();
        };

        if (profile_mode_count == 1) {
            constexpr int profile_repetitions = 20;
            if (profile_merge) {
                run_local_build();
                CUDDL_CUDA_CALL(cudaDeviceSynchronize());
            }
            CUDDL_CUDA_CALL(cudaProfilerStart());
            for (int repetition = 0; repetition < profile_repetitions; ++repetition) {
                if (profile_global) {
                    CUDDL_CUDA_CALL(cudaMemset(global_sketch, 0, bucket_count * sizeof(*global_sketch)));
                    global_atomic_kernel<<<block_count, block_size>>>(
                        device_inputs, item_count, global_sketch);
                    CUDDL_CUDA_CALL(cudaGetLastError());
                } else if (profile_cta) {
                    run_local_build();
                } else {
                    run_local_merge();
                }
            }
            CUDDL_CUDA_CALL(cudaDeviceSynchronize());
            CUDDL_CUDA_CALL(cudaProfilerStop());
            std::printf("profile=%s items=%llu blocks=%u repetitions=%d\n",
                        profile_global ? "global" : profile_cta ? "cta" : "merge",
                        static_cast<unsigned long long>(item_count),
                        block_count,
                        profile_repetitions);
            CUDDL_CUDA_CALL(cudaFree(partial_sketches));
            CUDDL_CUDA_CALL(cudaFree(local_sketch));
            CUDDL_CUDA_CALL(cudaFree(global_sketch));
            CUDDL_CUDA_CALL(cudaFree(device_inputs));
            return EXIT_SUCCESS;
        }

        run_global();
        run_local();
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());

        std::array<std::uint16_t, bucket_count> global_host{};
        std::array<std::uint16_t, bucket_count> local_host{};
        CUDDL_CUDA_CALL(cudaMemcpy(global_host.data(),
                             global_sketch,
                             sizeof(global_host),
                             cudaMemcpyDeviceToHost));
        CUDDL_CUDA_CALL(cudaMemcpy(local_host.data(), local_sketch, sizeof(local_host), cudaMemcpyDeviceToHost));
        auto const equivalent = global_host == local_host;

        if (check_only) {
            std::printf("equivalent=%s items=%llu blocks=%u\n",
                        equivalent ? "true" : "false",
                        static_cast<unsigned long long>(item_count),
                        block_count);
        } else {
            constexpr int repetitions = 9;
            event_timer timer;
            std::vector<double> global_times;
            std::vector<double> local_build_times;
            std::vector<double> local_merge_times;
            global_times.reserve(repetitions);
            local_build_times.reserve(repetitions);
            local_merge_times.reserve(repetitions);
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                timer.start();
                run_global();
                global_times.push_back(timer.stop());
                timer.start();
                run_local_build();
                local_build_times.push_back(timer.stop());
                timer.start();
                run_local_merge();
                local_merge_times.push_back(timer.stop());
            }
            auto const raw_global_times = global_times;
            auto const raw_local_build_times = local_build_times;
            auto const raw_local_merge_times = local_merge_times;
            std::ranges::sort(global_times);
            std::ranges::sort(local_build_times);
            std::ranges::sort(local_merge_times);
            auto const global_ms = global_times[repetitions / 2];
            auto const local_build_ms = local_build_times[repetitions / 2];
            auto const local_merge_ms = local_merge_times[repetitions / 2];
            auto const local_ms = local_build_ms + local_merge_ms;
            auto const global_gitems = static_cast<double>(item_count) / global_ms / 1.0e6;
            auto const local_gitems = static_cast<double>(item_count) / local_ms / 1.0e6;
            std::printf(
                "metadata,device=%s,compute_capability=%d.%d,seed=%llu,k=%u,buckets=%u,"
                "free_bytes=%llu,total_bytes=%llu,cap_bytes=%llu,repetitions=%d\n",
                properties.name,
                properties.major,
                properties.minor,
                static_cast<unsigned long long>(seed),
                kmer_length,
                bucket_count,
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(memory_cap),
                repetitions);
            std::printf("sample,global_ms,cta_build_ms,cta_merge_ms\n");
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                std::printf("sample,%0.6f,%0.6f,%0.6f\n",
                            raw_global_times[repetition],
                            raw_local_build_times[repetition],
                            raw_local_merge_times[repetition]);
            }
            std::printf(
                "architecture,items,blocks,build_ms,merge_ms,total_ms,gitems_per_second,"
                "partial_bytes,equivalent\n");
            std::printf("global_atomic,%llu,%u,%.6f,0,%.6f,%.6f,0,%s\n",
                        static_cast<unsigned long long>(item_count),
                        block_count,
                        global_ms,
                        global_ms,
                        global_gitems,
                        equivalent ? "true" : "false");
            std::printf("cta_local_merge,%llu,%u,%.6f,%.6f,%.6f,%.6f,%llu,%s\n",
                        static_cast<unsigned long long>(item_count),
                        block_count,
                        local_build_ms,
                        local_merge_ms,
                        local_ms,
                        local_gitems,
                        static_cast<unsigned long long>(partial_bytes),
                        equivalent ? "true" : "false");
        }

        CUDDL_CUDA_CALL(cudaFree(partial_sketches));
        CUDDL_CUDA_CALL(cudaFree(local_sketch));
        CUDDL_CUDA_CALL(cudaFree(global_sketch));
        CUDDL_CUDA_CALL(cudaFree(device_inputs));
        return equivalent ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "construction-bakeoff: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
