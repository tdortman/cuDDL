#include <cuddl/cuda_error.hpp>

#include <cuda_runtime.h>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t block_size = 256;
constexpr uint64_t seed = 42;

__host__ __device__ constexpr uint64_t splitmix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__host__ __device__ uint16_t score(uint64_t const hash) {
#ifdef __CUDA_ARCH__
    auto const exponent = static_cast<uint16_t>(min(__clzll(hash | 1ULL), 62));
#else
    auto const exponent =
        static_cast<uint16_t>(std::min(static_cast<int>(__builtin_clzll(hash | 1ULL)), 62));
#endif
    auto const mantissa = static_cast<uint16_t>((hash >> 43) & 0x03ffULL);
    return static_cast<uint16_t>(1U + (exponent << 10U) + (mantissa ^ 0x03ffU));
}

__host__ __device__ constexpr uint32_t pack(uint16_t winner, uint16_t count) {
    return static_cast<uint32_t>(winner) << 16U | count;
}
__host__ __device__ constexpr uint16_t winner(uint32_t state) {
    return state >> 16U;
}
__host__ __device__ constexpr uint16_t count(uint32_t state) {
    return state;
}
__host__ __device__ constexpr uint16_t saturated_add(uint16_t left, uint16_t right) {
    auto const sum = static_cast<uint32_t>(left) + right;
    return sum < std::numeric_limits<uint16_t>::max() ? static_cast<uint16_t>(sum)
                                                      : std::numeric_limits<uint16_t>::max();
}

__device__ bool update(uint32_t* address, uint16_t incoming) {
    auto observed = *address;
    while (incoming >= winner(observed)) {
        uint32_t replacement;
        if (incoming > winner(observed)) {
            replacement = pack(incoming, 1);
        } else if (count(observed) == std::numeric_limits<uint16_t>::max()) {
            return false;
        } else {
            replacement = pack(incoming, static_cast<uint16_t>(count(observed) + 1U));
        }
        auto const previous = atomicCAS(address, observed, replacement);
        if (previous == observed) return true;
        observed = previous;
    }
    return false;
}

template <uint32_t BucketCount>
__global__ void global_kernel(uint64_t const* inputs, uint64_t item_count, uint32_t* sketch) {
    auto const stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < item_count;
         index += stride) {
        auto const hash = splitmix64(inputs[index] ^ seed);
        update(sketch + (hash & (BucketCount - 1U)), score(hash));
    }
}

template <uint32_t BucketCount>
__global__ void tier_floor_kernel(
    uint64_t const* inputs,
    uint64_t item_count,
    uint32_t* partials,
    uint64_t* counters,
    uint32_t rescan_cadence
) {
    constexpr auto local_buckets = BucketCount < 8192 ? BucketCount : 8192;
    __shared__ uint32_t sketch[local_buckets];
    __shared__ uint16_t floor;
    __shared__ uint32_t successes;
    __shared__ uint32_t rescan_generation;
    __shared__ uint16_t warp_min[32];
    auto const tile_count = BucketCount / local_buckets;
    auto const tile = blockIdx.x / (gridDim.x / tile_count);
    auto const cta = blockIdx.x % (gridDim.x / tile_count);
    if (threadIdx.x == 0) {
        floor = 0;
        successes = 0;
        rescan_generation = 0;
    }
    for (auto bucket = threadIdx.x; bucket < local_buckets; bucket += blockDim.x)
        sketch[bucket] = 0;
    __syncthreads();
    auto const cta_count = gridDim.x / tile_count;
    auto const begin = (static_cast<uint64_t>(cta) * item_count) / cta_count;
    auto const end = (static_cast<uint64_t>(cta + 1U) * item_count) / cta_count;
    for (auto base = begin; base < end; base += blockDim.x) {
        auto const index = base + threadIdx.x;
        if (index < end) {
            auto const hash = splitmix64(inputs[index] ^ seed);
            auto const bucket = hash & (BucketCount - 1U);
            auto const incoming = score(hash);
            if (bucket / local_buckets == tile) {
                if (incoming >= floor) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(counters), 1ULL);
                    if (update(sketch + (bucket % local_buckets), incoming)) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 1), 1ULL);
                        atomicAdd(&successes, 1U);
                    }
                } else {
                    atomicAdd(reinterpret_cast<unsigned long long*>(counters + 2), 1ULL);
                }
            }
        }
        __syncthreads();
        auto const generation = rescan_generation;
        if (rescan_cadence != 0 && threadIdx.x == 0 && successes >= rescan_cadence) {
            successes = 0;
            ++rescan_generation;
        }
        __syncthreads();
        if (rescan_generation != generation) {
            auto local = std::numeric_limits<uint16_t>::max();
            for (auto bucket = threadIdx.x; bucket < local_buckets; bucket += blockDim.x) {
                local = min(local, winner(sketch[bucket]));
            }
            for (auto offset = 16; offset != 0; offset /= 2) {
                local = min(local, __shfl_down_sync(0xffffffffU, local, offset));
            }
            if ((threadIdx.x & 31U) == 0) warp_min[threadIdx.x / 32U] = local;
            __syncthreads();
            if (threadIdx.x == 0) {
                auto minimum = std::numeric_limits<uint16_t>::max();
                for (auto warp = 0U; warp < blockDim.x / 32U; ++warp)
                    minimum = min(minimum, warp_min[warp]);
                floor = minimum;
                atomicAdd(reinterpret_cast<unsigned long long*>(counters + 3), 1ULL);
            }
        }
        __syncthreads();
    }

    auto* output = partials + static_cast<uint64_t>(cta) * BucketCount + tile * local_buckets;
    for (auto bucket = threadIdx.x; bucket < local_buckets; bucket += blockDim.x)
        output[bucket] = sketch[bucket];
}

template <uint32_t BucketCount>
__global__ void merge_kernel(uint32_t const* partials, uint32_t partial_count, uint32_t* sketch) {
    for (auto bucket = blockIdx.x * blockDim.x + threadIdx.x; bucket < BucketCount;
         bucket += blockDim.x * gridDim.x) {
        uint16_t best = 0;
        uint16_t total = 0;
        for (uint32_t partial = 0; partial < partial_count; ++partial) {
            auto const state = partials[static_cast<uint64_t>(partial) * BucketCount + bucket];
            if (winner(state) > best) {
                best = winner(state);
                total = count(state);
            } else if (winner(state) == best && best != 0)
                total = saturated_add(total, count(state));
        }
        sketch[bucket] = pack(best, total);
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

template <uint32_t BucketCount>
bool run(
    uint64_t const* inputs,
    std::vector<uint64_t> const& host_inputs,
    uint32_t ctas,
    uint32_t rescan_cadence,
    bool check
) {
    auto const item_count = host_inputs.size();
    auto const partial_bytes = static_cast<size_t>(ctas) * BucketCount * sizeof(uint32_t);
    uint32_t* global = nullptr;
    uint32_t* local = nullptr;
    uint32_t* partials = nullptr;
    uint64_t* counters = nullptr;
    CUDDL_CUDA_CALL(cudaMalloc(&global, BucketCount * sizeof(uint32_t)));
    CUDDL_CUDA_CALL(cudaMalloc(&local, BucketCount * sizeof(uint32_t)));
    CUDDL_CUDA_CALL(cudaMalloc(&partials, partial_bytes));
    CUDDL_CUDA_CALL(cudaMalloc(&counters, 4 * sizeof(uint64_t)));
    auto global_launch = [&] {
        CUDDL_CUDA_CALL(cudaMemset(global, 0, BucketCount * sizeof(uint32_t)));
        global_kernel<BucketCount><<<ctas, block_size>>>(inputs, item_count, global);
        CUDDL_CUDA_CALL(cudaGetLastError());
    };
    auto local_launch = [&] {
        CUDDL_CUDA_CALL(cudaMemset(counters, 0, 4 * sizeof(uint64_t)));
        constexpr auto tile_count = BucketCount / (BucketCount < 8192 ? BucketCount : 8192);
        tier_floor_kernel<BucketCount><<<ctas * tile_count, block_size>>>(
            inputs, item_count, partials, counters, rescan_cadence
        );
        CUDDL_CUDA_CALL(cudaGetLastError());
        merge_kernel<BucketCount>
            <<<(BucketCount + block_size - 1U) / block_size, block_size>>>(partials, ctas, local);
        CUDDL_CUDA_CALL(cudaGetLastError());
    };
    global_launch();
    local_launch();
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    std::vector<uint32_t> cpu(BucketCount), global_host(BucketCount), local_host(BucketCount);
    for (auto input : host_inputs) {
        auto const hash = splitmix64(input ^ seed);
        auto& state = cpu[hash & (BucketCount - 1U)];
        auto const incoming = score(hash);
        if (incoming > winner(state))
            state = pack(incoming, 1);
        else if (incoming == winner(state))
            state = pack(incoming, saturated_add(count(state), 1));
    }
    CUDDL_CUDA_CALL(cudaMemcpy(
        global_host.data(), global, BucketCount * sizeof(uint32_t), cudaMemcpyDeviceToHost
    ));
    CUDDL_CUDA_CALL(
        cudaMemcpy(local_host.data(), local, BucketCount * sizeof(uint32_t), cudaMemcpyDeviceToHost)
    );
    std::array<uint64_t, 4> metrics{};
    CUDDL_CUDA_CALL(cudaMemcpy(metrics.data(), counters, sizeof(metrics), cudaMemcpyDeviceToHost));
    auto const global_equivalent = cpu == global_host;
    auto const local_equivalent = cpu == local_host;
    auto const equivalent = global_equivalent;
    if (check) {
        std::printf(
            "check,buckets=%u,equivalent=%s,global_equivalent=%s,local_equivalent=%s,attempted=%"
            "llu,successful=%llu,rejected=%llu,rescans=%llu\n",
            BucketCount,
            equivalent ? "true" : "false",
            global_equivalent ? "true" : "false",
            local_equivalent ? "true" : "false",
            static_cast<unsigned long long>(metrics[0]),
            static_cast<unsigned long long>(metrics[1]),
            static_cast<unsigned long long>(metrics[2]),
            static_cast<unsigned long long>(metrics[3])
        );
    } else {
        constexpr int repetitions = 9;
        event_timer timer;
        std::vector<double> global_times;
        std::vector<double> local_times;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            timer.start();
            global_launch();
            global_times.push_back(timer.stop());
            timer.start();
            local_launch();
            local_times.push_back(timer.stop());
        }
        std::ranges::sort(global_times);
        std::ranges::sort(local_times);
        auto const global_ms = global_times[repetitions / 2];
        auto const local_ms = local_times[repetitions / 2];
        auto const rejected = metrics[2];
        auto const considered = metrics[0] + rejected;
        auto const rejection_rate =
            considered == 0 ? 0.0 : static_cast<double>(rejected) / considered;
        std::printf(
            "result,buckets=%u,ctas=%u,local_buckets=%u,cadence=%u,global_ms=%.6f,tier_floor_ms=%."
            "6f,global_speedup=%.6f,equivalent=%s,attempted=%llu,successful=%llu,rejected=%llu,"
            "rejection_rate=%.6f,rescans=%llu\n",
            BucketCount,
            ctas,
            BucketCount < 8192 ? BucketCount : 8192,
            rescan_cadence,
            global_ms,
            local_ms,
            local_ms / global_ms,
            equivalent ? "true" : "false",
            static_cast<unsigned long long>(metrics[0]),
            static_cast<unsigned long long>(metrics[1]),
            static_cast<unsigned long long>(rejected),
            rejection_rate,
            static_cast<unsigned long long>(metrics[3])
        );
    }
    CUDDL_CUDA_CALL(cudaFree(counters));
    CUDDL_CUDA_CALL(cudaFree(partials));
    CUDDL_CUDA_CALL(cudaFree(local));
    CUDDL_CUDA_CALL(cudaFree(global));
    return equivalent;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"Tier-floor construction prototype"};
        uint64_t item_count = 1ULL << 24;
        uint32_t rescan_cadence = 128;
        bool check = false;
        app.add_option("--items", item_count)->check(CLI::PositiveNumber);
        app.add_option("--rescan-cadence", rescan_cadence)
            ->check(CLI::IsMember({0, 32, 128, 512, 2048}));
        app.add_flag("--check", check);
        CLI11_PARSE(app, argc, argv);
        int device = 0;
        cudaDeviceProp properties{};
        CUDDL_CUDA_CALL(cudaGetDevice(&device));
        CUDDL_CUDA_CALL(cudaGetDeviceProperties(&properties, device));
        auto const ctas = static_cast<uint32_t>(properties.multiProcessorCount * 2);
        std::vector<uint64_t> host_inputs(item_count);
        for (uint64_t i = 0; i < item_count; ++i) host_inputs[i] = splitmix64(i);
        uint64_t* inputs = nullptr;
        CUDDL_CUDA_CALL(cudaMalloc(&inputs, item_count * sizeof(uint64_t)));
        CUDDL_CUDA_CALL(cudaMemcpy(
            inputs, host_inputs.data(), item_count * sizeof(uint64_t), cudaMemcpyHostToDevice
        ));
        auto equivalent = true;
        equivalent &= run<2048>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<4096>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<8192>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<16384>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<32768>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<65536>(inputs, host_inputs, ctas, rescan_cadence, check);
        equivalent &= run<131072>(inputs, host_inputs, ctas, rescan_cadence, check);
        CUDDL_CUDA_CALL(cudaFree(inputs));
        return equivalent ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& error) {
        std::fprintf(stderr, "tier-floor-construction: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
