#include <cuddl/cuddl.cuh>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "result_json.hpp"

namespace {

constexpr size_t bucket_count = 2048;
std::vector<nvbench::int64_t> const sketch_pair_counts{
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072
};

void add_system_metadata(nvbench::state& state) {
    auto const system = benchmark_system();
    for (auto const& [column, field] : std::array{
             std::pair{"System OS", "os"},
             std::pair{"System Kernel", "kernel"},
             std::pair{"System Architecture", "architecture"},
             std::pair{"System CPU", "cpu"},
             std::pair{"System Logical CPU Count", "logical_cpu_count"},
             std::pair{"System RAM Bytes", "ram_bytes"},
             std::pair{"Compute Capability", "compute_capability"},
             std::pair{"SM Count", "sm_count"},
             std::pair{"GPU RAM Bytes", "gpu_ram_bytes"},
             std::pair{"CUDA Runtime Version", "cuda_runtime_version"},
             std::pair{"CUDA Driver Version", "cuda_driver_version"},
             std::pair{"CUDA Compile Version", "cuda_compile_version"},
         }) {
        auto& summary = state.add_summary(column);
        summary.set_string("name", column);
        auto const& value = system.at(field);
        summary.set_string("value", value.is_string() ? value.get<std::string>() : value.dump());
    }
}

class batch_fixture {
   public:
    explicit batch_fixture(size_t pairs, cuda::stream_ref setup_stream)
        : pairs_(pairs),
          register_count_(pairs * bucket_count),
          host_left_(
              cuda::make_pinned_buffer<uint32_t>(setup_stream, register_count_, cuda::no_init)
          ),
          host_right_(
              cuda::make_pinned_buffer<uint32_t>(setup_stream, register_count_, cuda::no_init)
          ),
          host_outputs_(
              cuda::make_pinned_buffer<cuddl::pairwise_summary>(setup_stream, pairs_, cuda::no_init)
          ),
          device_left_(
              cuda::make_device_buffer<uint32_t>(
                  setup_stream,
                  setup_stream.device(),
                  register_count_,
                  cuda::no_init
              )
          ),
          device_right_(
              cuda::make_device_buffer<uint32_t>(
                  setup_stream,
                  setup_stream.device(),
                  register_count_,
                  cuda::no_init
              )
          ),
          device_outputs_(
              cuda::make_device_buffer<cuddl::pairwise_summary>(
                  setup_stream,
                  setup_stream.device(),
                  pairs_,
                  cuda::no_init
              )
          ) {
        setup_stream.sync();
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
            host_left_.data()[index] = cuddl::detail::pack(left_score, left_score == 0U ? 0U : 1U);
            host_right_.data()[index] =
                cuddl::detail::pack(right_score, right_score == 0U ? 0U : 1U);
        }
        upload(setup_stream);
        setup_stream.sync();
    }

    batch_fixture(batch_fixture const&) = delete;
    batch_fixture& operator=(batch_fixture const&) = delete;

    void upload(cuda::stream_ref stream) const {
        cuda::copy_bytes(stream, host_left_, device_left_);
        cuda::copy_bytes(stream, host_right_, device_right_);
    }

    void compare(cuda::stream_ref stream) const {
        auto result = cuddl::compare_batch_async<bucket_count>(
            {device_left_.data(), register_count_},
            {device_right_.data(), register_count_},
            {device_outputs_.data(), pairs_},
            cuda::stream_ref{stream}
        );
        if (!result.has_value()) {
            throw std::runtime_error(result.error().message());
        }
    }

    void download(cuda::stream_ref stream) const {
        cuda::copy_bytes(stream, device_outputs_, host_outputs_);
    }

    void verify() const {
        std::array<size_t, 3> const selected{0U, pairs_ / 2U, pairs_ - 1U};
        for (auto const pair : selected) {
            cuddl::pairwise_counts expected{};
            auto const offset = pair * bucket_count;
            for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
                auto const left = cuddl::detail::winner(host_left_.data()[offset + bucket]);
                auto const right = cuddl::detail::winner(host_right_.data()[offset + bucket]);
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
            if (!(host_outputs_.data()[pair].counts == expected)) {
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
    cuda::host_buffer<uint32_t> host_left_;
    cuda::host_buffer<uint32_t> host_right_;
    mutable cuda::host_buffer<cuddl::pairwise_summary> host_outputs_;
    mutable cuda::device_buffer<uint32_t> device_left_;
    mutable cuda::device_buffer<uint32_t> device_right_;
    mutable cuda::device_buffer<cuddl::pairwise_summary> device_outputs_;
};

void profile_kernel(nvbench::state& state) {
    auto const setup_stream = cuda::stream_ref{state.get_cuda_stream()};
    add_system_metadata(state);
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Sketch Pairs")), setup_stream);
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_writes<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.compare(cuda::stream_ref{launch.get_stream()});
    });
    fixture.download(setup_stream);
    setup_stream.sync();
    fixture.verify();
}

void profile_h2d(nvbench::state& state) {
    auto const setup_stream = cuda::stream_ref{state.get_cuda_stream()};
    add_system_metadata(state);
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Sketch Pairs")), setup_stream);
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_writes<uint32_t>(2U * fixture.register_count());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.upload(cuda::stream_ref{launch.get_stream()});
    });
}

void profile_d2h(nvbench::state& state) {
    auto const setup_stream = cuda::stream_ref{state.get_cuda_stream()};
    add_system_metadata(state);
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Sketch Pairs")), setup_stream);
    fixture.compare(setup_stream);
    setup_stream.sync();
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.download(cuda::stream_ref{launch.get_stream()});
    });
    fixture.verify();
}

void profile_end_to_end(nvbench::state& state) {
    auto const setup_stream = cuda::stream_ref{state.get_cuda_stream()};
    add_system_metadata(state);
    batch_fixture fixture(static_cast<size_t>(state.get_int64("Sketch Pairs")), setup_stream);
    state.add_element_count(fixture.pairs(), "Pairs");
    state.add_global_memory_reads<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_writes<uint32_t>(2U * fixture.register_count());
    state.add_global_memory_reads<cuddl::pairwise_summary>(fixture.pairs());
    state.add_global_memory_writes<cuddl::pairwise_summary>(fixture.pairs());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        fixture.upload(cuda::stream_ref{launch.get_stream()});
        fixture.compare(cuda::stream_ref{launch.get_stream()});
        fixture.download(cuda::stream_ref{launch.get_stream()});
    });
    fixture.verify();
}

NVBENCH_BENCH(profile_kernel)
    .set_name("batch_compare_kernel")
    .add_int64_axis("Sketch Pairs", sketch_pair_counts);
NVBENCH_BENCH(profile_h2d)
    .set_name("batch_compare_h2d")
    .add_int64_axis("Sketch Pairs", sketch_pair_counts);
NVBENCH_BENCH(profile_d2h)
    .set_name("batch_compare_d2h")
    .add_int64_axis("Sketch Pairs", sketch_pair_counts);
NVBENCH_BENCH(profile_end_to_end)
    .set_name("batch_compare_end_to_end")
    .add_int64_axis("Sketch Pairs", sketch_pair_counts);

}  // namespace
