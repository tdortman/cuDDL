#pragma once

#include <cuda/algorithm>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cuddl/detail/construction.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/hybrid_cardinality.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl::detail {

/**
 * @brief Non-owning implementation view of a DDL sketch.
 *
 * Provides allocation-free, stream-ordered operations on an external contiguous allocation. The
 * allocation must contain `BucketCount` packed `uint32_t` registers followed by one aligned
 * `uint32_t` saturation flag. Pass by value into device or host code.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class sketch_view {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using register_type = uint32_t;
    using layout_type = Layout;

    /// @brief Constructs a reference over @p registers and the sketch's saturation flag.
    __host__ __device__ constexpr sketch_view(
        device_span<register_type> registers,
        uint32_t& saturation
    ) noexcept
        : registers_(registers), saturation_(saturation) {}

    /// @brief Packed registers.
    [[nodiscard]] __host__ __device__ constexpr device_span<register_type> data() const noexcept {
        return registers_;
    }

    /// @brief Sketch-level saturation flag.
    [[nodiscard]] __host__ __device__ constexpr uint32_t& saturation_flag() const noexcept {
        return saturation_;
    }

    /// @brief Number of registers in the sketch.
    [[nodiscard]] static constexpr size_t bucket_count() noexcept {
        return BucketCount;
    }

    /// @brief Compile-time k-mer length.
    [[nodiscard]] static constexpr uint32_t kmer_length() noexcept {
        return K;
    }

    /// @brief Resets every register and the saturation flag to zero as one logical operation.
    ///
    /// The backing allocation is contractually `BucketCount` registers followed by the aligned
    /// saturation flag, so one memset covers both.
    [[nodiscard]] Result<void> clear_async(cuda::stream_ref stream) const noexcept {
        return cuda_try([&] {
            cuda::fill_bytes(stream, cuda::std::span{registers_.data(), registers_.size() + 1U}, 0);
        });
    }

    /// @brief Constructs the sketch from packed k-mers in @p input.
    ///
    /// The input must remain valid until @p stream completes.
    [[nodiscard]] Result<void>
    add_async(device_span<uint64_t const> input, cuda::stream_ref stream) const noexcept {
        return detail::launch_construction<BucketCount, Layout>(
            input, registers_, saturation_, stream
        );
    }

    /// @brief Computes the raw pairwise summary into caller-owned device storage.
    ///
    /// @p output must point at device memory valid until @p stream completes.
    template <bool IncludeCardinality = false>
    [[nodiscard]] Result<void> summary_async(
        sketch_view other,
        pairwise_summary& output,
        cuda::stream_ref stream
    ) const noexcept {
        detail::summary_kernel<BucketCount, IncludeCardinality, Layout>
            <<<1, detail::block_size, 0, stream.get()>>>(
                registers_.data(), other.registers_.data(), output
            );
        return cuda_try(cudaGetLastError());
    }

    /// @brief Computes this sketch's cardinality reduction on the GPU.
    ///
    /// @p empty_out receives the empty-register count and @p estimate_out the cardinality estimate.
    [[nodiscard]] Result<void> cardinality_async(
        uint64_t* empty_out,
        double* estimate_out,
        cuda::stream_ref stream
    ) const noexcept {
        detail::cardinality_kernel<BucketCount, Layout><<<1, detail::block_size, 0, stream.get()>>>(
            registers_.data(), empty_out, estimate_out
        );
        return cuda_try(cudaGetLastError());
    }

    /// @brief Computes BBTools and paper-style HybridDDL estimates in one GPU register scan.
    [[nodiscard]] Result<void> hybrid_cardinality_async(
        hybrid_cardinality_estimates* output,
        cuda::stream_ref stream
    ) const noexcept {
        detail::hybrid_cardinality_kernel<BucketCount, Layout>
            <<<1, detail::block_size, 0, stream.get()>>>(registers_.data(), output);
        return cuda_try(cudaGetLastError());
    }

    /// @brief Extracts per-register winner counts and the saturation flag to device outputs.
    ///
    /// @p counts_out must hold `BucketCount` `uint16_t` entries and @p saturation_out one
    /// `uint32_t`, all valid until @p stream completes.
    [[nodiscard]] Result<void> winner_counts_async(
        uint16_t* counts_out,
        uint32_t* saturation_out,
        cuda::stream_ref stream
    ) const noexcept {
        detail::winner_counts_kernel<BucketCount><<<1, detail::block_size, 0, stream.get()>>>(
            registers_.data(), &saturation_, counts_out, saturation_out
        );
        return cuda_try(cudaGetLastError());
    }

   private:
    device_span<register_type> registers_;
    uint32_t& saturation_;
};

}  // namespace cuddl::detail
