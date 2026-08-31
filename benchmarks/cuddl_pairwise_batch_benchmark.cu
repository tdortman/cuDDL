#include <cuddl/cuddl.cuh>

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t bucket_count = 2048;
std::vector<nvbench::int64_t> const batch_sizes{1, 8, 32, 128, 512, 2048, 8192};

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

class batch_fixture {
   public:
    explicit batch_fixture(size_t pairs) : pairs_(pairs), register_count_(pairs * bucket_count) {
        check(cudaMallocHost(&host_left_, register_count_ * sizeof(uint32_t)));
        check(cudaMallocHost(&host_right_, register_count_ * sizeof(uint32_t)));
        check(cudaMallocHost(&host_outputs_, pairs_ * sizeof(cuddl::pairwise_summary)));
        check(cudaMalloc(&device_left_, register_count_ * sizeof(uint32_t)));
        check(cudaMalloc(&device_right_, register_count_ * sizeof(uint32_t)));
        check(cudaMalloc(&device_outputs_, pairs_ * sizeof(cuddl::pairwise_summary)));

        for (size_t index = 0; index < register_count_; ++index) {
            auto const left_hash = cuddl::detail::splitmix64(0x1234'5678'9abc'def0ULL + index);
            auto const right_hash = cuddl::detail::splitmix64(0xfedc'ba98'7654'3210ULL + index);
            auto const left_score = (left_hash & 7U) == 0U
                                        ? uint16_t{0}
                                        : static_cast<uint16_t>((left_hash >> 48U) | 1U);
            auto const independent_right = (right_hash & 7U) == 0U
                                               ? uint16_t{0}
                                               : static_cast<uint16_t>((right_hash >> 48U) | 1U);
            auto const right_score = index % 4U == 0U ? left_score : independent_right;
            host_left_[index] = cuddl::detail::pack(left_score, left_score == 0U ? 0U : 1U);
            host_right_[index] = cuddl::detail::pack(right_score, right_score == 0U ? 0U : 1U);
        }
        upload(cudaStream_t{nullptr});
        check(cudaDeviceSynchronize());
    }

    ~batch_fixture() {
        cudaFree(device_outputs_);
        cudaFree(device_right_);
        cudaFree(device_left_);
        cudaFreeHost(host_outputs_);
        cudaFreeHost(host_right_);
        cudaFreeHost(host_left_);
    }

    batch_fixture(batch_fixture const&) = delete;
    batch_fixture& operator=(batch_fixture const&) = delete;

    void upload(cudaStream_t stream) const {
        check(cudaMemcpyAsync(
            device_left_,
            host_left_,
            register_count_ * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        ));
        check(cudaMemcpyAsync(
            device_right_,
            host_right_,
            register_count_ * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        ));
    }

    void compare(cudaStream_t stream) const {
        auto result = cuddl::compare_batch_async<bucket_count>(
            {device_left_, register_count_},
            {device_right_, register_count_},
            {device_outputs_, pairs_},
            cuda::stream_ref{stream}
        );
        if (!result.has_value()) {
            throw std::runtime_error(result.error().message());
        }
    }

    void download(cudaStream_t stream) const {
        check(cudaMemcpyAsync(
            host_outputs_,
            device_outputs_,
            pairs_ * sizeof(cuddl::pairwise_summary),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }

    void verify() const {
        std::array<size_t, 3> const selected{0U, pairs_ / 2U, pairs_ - 1U};
        for (auto const pair : selected) {
            cuddl::pairwise_counts expected{};
            auto const offset = pair * bucket_count;
            for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
                auto const left = cuddl::detail::winner(host_left_[offset + bucket]);
                auto const right = cuddl::detail::winner(host_right_[offset + bucket]);
                if (left == 0U && right == 0U) {
                    ++expected.both_empty;
                } else if (left < right) {
                    ++expected.lower;
                } else if (left > right) {
                    ++expected.higher;
                } else {
                    ++expected.equal;
                }
            }
            if (!(host_outputs_[pair].counts == expected)) {
                throw std::runtime_error("batched comparison verification failed");
            }
        }
    }

    [[nodiscard]] size_t pairs() const noexcept {
        return pairs_;
    }
    [[nodiscard]] size_t register_count() const noexcept {
        return register_count_;
    }

   private:
    size_t pairs_;
    size_t register_count_;
    uint32_t* host_left_{};
    uint32_t* host_right_{};
    cuddl::pairwise_summary* host_outputs_{};
    uint32_t* device_left_{};
    uint32_t* device_right_{};
    cuddl::pairwise_summary* device_outputs_{};
};

void profile_kernel(nvbench::state& state) {
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Batch")));
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_writes<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.compare(launch.get_stream());
    });
    fixture.download(cudaStream_t{nullptr});
    check(cudaDeviceSynchronize());
    fixture.verify();
}

void profile_h2d(nvbench::state& state) {
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Batch")));
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_writes<uint32_t>(2U * fixture.register_count());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.upload(launch.get_stream());
    });
}

void profile_d2h(nvbench::state& state) {
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Batch")));
    fixture.compare(cudaStream_t{nullptr});
    check(cudaDeviceSynchronize());
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.download(launch.get_stream());
    });
    fixture.verify();
}

void profile_end_to_end(nvbench::state& state) {
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Batch")));
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_writes<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_reads<cuddl::pairwise_summary>(fixture.pairs());
    state.add_global_memory_writes<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.upload(launch.get_stream());
        fixture.compare(launch.get_stream());
        fixture.download(launch.get_stream());
    });
    fixture.verify();
}

NVBENCH_BENCH(profile_kernel).set_name("batch_compare_kernel").add_int64_axis("Batch", batch_sizes);
NVBENCH_BENCH(profile_h2d).set_name("batch_compare_h2d").add_int64_axis("Batch", batch_sizes);
NVBENCH_BENCH(profile_d2h).set_name("batch_compare_d2h").add_int64_axis("Batch", batch_sizes);
NVBENCH_BENCH(profile_end_to_end)
    .set_name("batch_compare_end_to_end")
    .add_int64_axis("Batch", batch_sizes);

}  // namespace
