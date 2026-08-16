#include <cuddl/cuda_error.hpp>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <charconv>
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

struct counts {
    std::uint32_t lower;
    std::uint32_t equal;
    std::uint32_t higher;

    __host__ __device__ counts& operator+=(counts const other) {
        lower += other.lower;
        equal += other.equal;
        higher += other.higher;
        return *this;
    }

    friend __host__ __device__ counts operator+(counts left, counts const right) {
        return left += right;
    }

    friend bool operator==(counts const&, counts const&) = default;
};

__host__ __device__ constexpr std::uint64_t splitmix64(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__global__ void block_per_pair_kernel(
    std::uint16_t const* const queries,
    std::uint16_t const* const references,
    std::uint64_t const pair_count,
    std::uint32_t const reference_count,
    counts* const output
) {
    using block_reduce = cub::BlockReduce<counts, block_size>;
    __shared__ typename block_reduce::TempStorage storage;

    for (auto pair = static_cast<std::uint64_t>(blockIdx.x); pair < pair_count; pair += gridDim.x) {
        auto const query = pair / reference_count;
        auto const reference = pair % reference_count;
        auto const* const left = queries + query * bucket_count;
        auto const* const right = references + reference * bucket_count;
        counts local{};
        for (auto bucket = threadIdx.x; bucket < bucket_count; bucket += blockDim.x) {
            auto const lhs = left[bucket];
            auto const rhs = right[bucket];
            if ((lhs | rhs) != 0U) {
                local.lower += lhs < rhs;
                local.equal += lhs == rhs;
                local.higher += lhs > rhs;
            }
        }
        auto const total = block_reduce(storage).Reduce(local, [] __device__(counts a, counts b) {
            return a + b;
        });
        if (threadIdx.x == 0) {
            output[pair] = total;
        }
        __syncthreads();
    }
}

__global__ void warp_per_pair_kernel(
    std::uint16_t const* const queries,
    std::uint16_t const* const references,
    std::uint64_t const pair_count,
    std::uint32_t const reference_count,
    counts* const output
) {
    constexpr std::uint32_t warp_size = 32;
    auto const warp =
        (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    auto const warp_stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x / warp_size;
    auto const lane = threadIdx.x % warp_size;
    for (auto pair = warp; pair < pair_count; pair += warp_stride) {
        auto const query = pair / reference_count;
        auto const reference = pair % reference_count;
        auto const* const left = queries + query * bucket_count;
        auto const* const right = references + reference * bucket_count;
        counts local{};
        for (auto bucket = lane; bucket < bucket_count; bucket += warp_size) {
            auto const lhs = left[bucket];
            auto const rhs = right[bucket];
            if ((lhs | rhs) != 0U) {
                local.lower += lhs < rhs;
                local.equal += lhs == rhs;
                local.higher += lhs > rhs;
            }
        }
        for (auto offset = warp_size / 2; offset > 0; offset /= 2) {
            local.lower += __shfl_down_sync(0xffffffffU, local.lower, offset);
            local.equal += __shfl_down_sync(0xffffffffU, local.equal, offset);
            local.higher += __shfl_down_sync(0xffffffffU, local.higher, offset);
        }
        if (lane == 0) {
            output[pair] = local;
        }
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

    void start() {
        CUDDL_CUDA_CALL(cudaEventRecord(start_));
    }

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

std::uint32_t parse_count(int argc, char** argv, std::string_view option, std::uint32_t fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string_view{argv[index]} == option) {
            auto const* begin = argv[index + 1];
            auto const* end = begin + std::char_traits<char>::length(begin);
            if (auto const [ptr, error] = std::from_chars(begin, end, fallback);
                error != std::errc{} || ptr != end || fallback == 0) {
                throw std::runtime_error("invalid count option");
            }
        }
    }
    return fallback;
}

bool has_flag(int argc, char** argv, std::string_view flag) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view{argv[index]} == flag) {
            return true;
        }
    }
    return false;
}

counts compare_cpu(std::uint16_t const* left, std::uint16_t const* right) {
    counts result{};
    for (std::uint32_t bucket = 0; bucket < bucket_count; ++bucket) {
        auto const lhs = left[bucket];
        auto const rhs = right[bucket];
        if ((lhs | rhs) != 0U) {
            result.lower += lhs < rhs;
            result.equal += lhs == rhs;
            result.higher += lhs > rhs;
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto const query_count = parse_count(argc, argv, "--queries", 256);
        auto const reference_count = parse_count(argc, argv, "--references", 4096);
        auto const check_only = has_flag(argc, argv, "--check");
        auto const profile_block = has_flag(argc, argv, "--profile-block");
        auto const profile_warp = has_flag(argc, argv, "--profile-warp");
        if (static_cast<int>(profile_block) + static_cast<int>(profile_warp) > 1) {
            throw std::runtime_error("select at most one profile mode");
        }
        auto const pair_count = static_cast<std::uint64_t>(query_count) * reference_count;
        auto const query_values = static_cast<std::uint64_t>(query_count) * bucket_count;
        auto const reference_values = static_cast<std::uint64_t>(reference_count) * bucket_count;
        auto const device_bytes = (query_values + reference_values) * sizeof(std::uint16_t) +
                                  2U * pair_count * sizeof(counts);

        int device = 0;
        CUDDL_CUDA_CALL(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDDL_CUDA_CALL(cudaGetDeviceProperties(&properties, device));
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        CUDDL_CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
        auto const memory_cap = free_bytes * 4U / 5U;
        if (device_bytes > memory_cap) {
            std::printf(
                "status=skipped reason=memory_cap queries=%u references=%u "
                "required_bytes=%llu free_bytes=%llu cap_bytes=%llu\n",
                query_count,
                reference_count,
                static_cast<unsigned long long>(device_bytes),
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(memory_cap)
            );
            return EXIT_SUCCESS;
        }

        std::vector<std::uint16_t> queries(query_values);
        std::vector<std::uint16_t> references(reference_values);
        for (std::uint64_t index = 0; index < query_values; ++index) {
            queries[index] = static_cast<std::uint16_t>(1U + splitmix64(index ^ seed) % 65535U);
            if ((index & 15U) == 0) {
                queries[index] = 0;
            }
        }
        for (std::uint64_t index = 0; index < reference_values; ++index) {
            references[index] =
                static_cast<std::uint16_t>(1U + splitmix64(index ^ (seed + 1U)) % 65535U);
            if ((index & 31U) == 1U) {
                references[index] = 0;
            } else if ((index & 7U) == 0 && index < query_values) {
                references[index] = queries[index];
            }
        }

        std::uint16_t* device_queries = nullptr;
        std::uint16_t* device_references = nullptr;
        counts* block_output = nullptr;
        counts* warp_output = nullptr;
        CUDDL_CUDA_CALL(cudaMalloc(&device_queries, query_values * sizeof(std::uint16_t)));
        CUDDL_CUDA_CALL(cudaMalloc(&device_references, reference_values * sizeof(std::uint16_t)));
        CUDDL_CUDA_CALL(cudaMalloc(&block_output, pair_count * sizeof(counts)));
        CUDDL_CUDA_CALL(cudaMalloc(&warp_output, pair_count * sizeof(counts)));
        CUDDL_CUDA_CALL(cudaMemcpy(
            device_queries,
            queries.data(),
            query_values * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice
        ));
        CUDDL_CUDA_CALL(cudaMemcpy(
            device_references,
            references.data(),
            reference_values * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice
        ));

        auto const block_grid = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(pair_count, properties.multiProcessorCount * 8ULL)
        );
        auto const warp_grid = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            (pair_count + block_size / 32U - 1U) / (block_size / 32U),
            properties.multiProcessorCount * 8ULL
        ));
        auto run_block = [&] {
            block_per_pair_kernel<<<block_grid, block_size>>>(
                device_queries, device_references, pair_count, reference_count, block_output
            );
            CUDDL_CUDA_CALL(cudaGetLastError());
        };
        auto run_warp = [&] {
            warp_per_pair_kernel<<<warp_grid, block_size>>>(
                device_queries, device_references, pair_count, reference_count, warp_output
            );
            CUDDL_CUDA_CALL(cudaGetLastError());
        };

        run_block();
        run_warp();
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());

        if (profile_block || profile_warp) {
            constexpr int profile_repetitions = 20;
            for (int repetition = 0; repetition < profile_repetitions; ++repetition) {
                CUDDL_CUDA_CALL(cudaProfilerStart());
                profile_block ? run_block() : run_warp();
                CUDDL_CUDA_CALL(cudaDeviceSynchronize());
                CUDDL_CUDA_CALL(cudaProfilerStop());
            }
            std::printf(
                "profile=%s queries=%u references=%u pairs=%llu repetitions=%d "
                "device=%s compute_capability=%d.%d seed=%llu buckets=%u "
                "free_bytes=%llu total_bytes=%llu cap_bytes=%llu\n",
                profile_block ? "block" : "warp",
                query_count,
                reference_count,
                static_cast<unsigned long long>(pair_count),
                profile_repetitions,
                properties.name,
                properties.major,
                properties.minor,
                static_cast<unsigned long long>(seed),
                bucket_count,
                static_cast<unsigned long long>(free_bytes),
                static_cast<unsigned long long>(total_bytes),
                static_cast<unsigned long long>(memory_cap)
            );
        } else {
            std::vector<counts> block_host(pair_count);
            std::vector<counts> warp_host(pair_count);
            CUDDL_CUDA_CALL(cudaMemcpy(
                block_host.data(), block_output, pair_count * sizeof(counts), cudaMemcpyDeviceToHost
            ));
            CUDDL_CUDA_CALL(cudaMemcpy(
                warp_host.data(), warp_output, pair_count * sizeof(counts), cudaMemcpyDeviceToHost
            ));
            bool equivalent = block_host == warp_host;
            if (check_only) {
                for (std::uint32_t query = 0; query < query_count; ++query) {
                    for (std::uint32_t reference = 0; reference < reference_count; ++reference) {
                        auto const pair =
                            static_cast<std::uint64_t>(query) * reference_count + reference;
                        auto const expected = compare_cpu(
                            queries.data() + query * bucket_count,
                            references.data() + reference * bucket_count
                        );
                        equivalent &= block_host[pair] == expected;
                        equivalent &= warp_host[pair] == expected;
                    }
                }
                std::printf(
                    "equivalent=%s queries=%u references=%u pairs=%llu\n",
                    equivalent ? "true" : "false",
                    query_count,
                    reference_count,
                    static_cast<unsigned long long>(pair_count)
                );
            } else {
                constexpr int repetitions = 9;
                event_timer timer;
                std::vector<double> block_times;
                std::vector<double> warp_times;
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    timer.start();
                    run_block();
                    block_times.push_back(timer.stop());
                    timer.start();
                    run_warp();
                    warp_times.push_back(timer.stop());
                }
                auto const raw_block_times = block_times;
                auto const raw_warp_times = warp_times;
                std::ranges::sort(block_times);
                std::ranges::sort(warp_times);
                auto const block_ms = block_times[repetitions / 2];
                auto const warp_ms = warp_times[repetitions / 2];
                std::printf(
                    "metadata,device=%s,compute_capability=%d.%d,seed=%llu,buckets=%u,"
                    "free_bytes=%llu,total_bytes=%llu,cap_bytes=%llu,repetitions=%d\n",
                    properties.name,
                    properties.major,
                    properties.minor,
                    static_cast<unsigned long long>(seed),
                    bucket_count,
                    static_cast<unsigned long long>(free_bytes),
                    static_cast<unsigned long long>(total_bytes),
                    static_cast<unsigned long long>(memory_cap),
                    repetitions
                );
                std::printf("sample,block_ms,warp_ms\n");
                for (int repetition = 0; repetition < repetitions; ++repetition) {
                    std::printf(
                        "sample,%.6f,%.6f\n",
                        raw_block_times[repetition],
                        raw_warp_times[repetition]
                    );
                }
                std::printf(
                    "mapping,queries,references,pairs,median_ms,gregisters_per_second,"
                    "equivalent\n"
                );
                std::printf(
                    "block_per_pair,%u,%u,%llu,%.6f,%.6f,%s\n",
                    query_count,
                    reference_count,
                    static_cast<unsigned long long>(pair_count),
                    block_ms,
                    static_cast<double>(pair_count) * bucket_count / block_ms / 1.0e6,
                    equivalent ? "true" : "false"
                );
                std::printf(
                    "warp_per_pair,%u,%u,%llu,%.6f,%.6f,%s\n",
                    query_count,
                    reference_count,
                    static_cast<unsigned long long>(pair_count),
                    warp_ms,
                    static_cast<double>(pair_count) * bucket_count / warp_ms / 1.0e6,
                    equivalent ? "true" : "false"
                );
            }
            if (!equivalent) {
                return EXIT_FAILURE;
            }
        }

        CUDDL_CUDA_CALL(cudaFree(warp_output));
        CUDDL_CUDA_CALL(cudaFree(block_output));
        CUDDL_CUDA_CALL(cudaFree(device_references));
        CUDDL_CUDA_CALL(cudaFree(device_queries));
        return EXIT_SUCCESS;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "comparison-bakeoff: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
