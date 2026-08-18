#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <algorithm>
#include <cmath>
#include <optional>

#include <cuddl/detail/construction.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/error.hpp>
#include <cuddl/pairwise_counts.cuh>
#include <cuddl/hybrid_cardinality.cuh>

namespace cuddl {

/**
 * @brief Non-owning, trivially copyable device reference to a DDL sketch.
 *
 * Provides allocation-free, stream-ordered operations on an external contiguous allocation. The
 * allocation must contain `BucketCount` packed `uint32_t` registers followed by one aligned
 * `uint32_t` saturation flag. Pass by value into device or host code.
 */
template <uint32_t K, size_t BucketCount>
class sketch_ref {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using register_type = uint32_t;

    /// @brief Constructs a reference over @p registers and the sketch's saturation flag.
    __host__
        __device__ constexpr sketch_ref(register_type* registers, uint32_t* saturation) noexcept
        : registers_(registers), saturation_(saturation) {}

    /// @brief Pointer to the first packed register.
    [[nodiscard]] __host__ __device__ constexpr register_type* data() const noexcept {
        return registers_;
    }

    /// @brief Pointer to the sketch-level saturation flag.
    [[nodiscard]] __host__ __device__ constexpr uint32_t* saturation_flag() const noexcept {
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
    [[nodiscard]] Result<void> clear_async(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        auto const registers_bytes = BucketCount * sizeof(register_type);
        if (auto const result =
                cuda_try(cudaMemsetAsync(registers_, 0, registers_bytes, stream.get()));
            !result) {
            return result;
        }
        return cuda_try(cudaMemsetAsync(saturation_, 0, sizeof(uint32_t), stream.get()));
    }

    /// @brief Constructs the sketch from packed k-mers in `[first, last)`.
    ///
    /// The input range must remain valid until @p stream completes.
    [[nodiscard]] Result<void> add_async(
        uint64_t const* first,
        uint64_t const* last,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        auto const count = static_cast<size_t>(last - first);
        return detail::launch_construction<BucketCount>(
            first, count, registers_, saturation_, stream.get()
        );
    }

    /// @brief Computes the raw pairwise summary into caller-owned device storage.
    ///
    /// @p output must point at device memory valid until @p stream completes.
    template <uint32_t Mask = summary_mask::pairwise>
    [[nodiscard]] Result<void> summary_async(
        sketch_ref other,
        pairwise_summary* output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        detail::summary_kernel<BucketCount, Mask>
            <<<1, detail::block_size, 0, stream.get()>>>(registers_, other.registers_, output);
        return cuda_try(cudaGetLastError());
    }

    /// @brief Computes raw pairwise counts into caller-owned device storage.
    [[nodiscard]] Result<void> compare_async(
        sketch_ref other,
        pairwise_summary* output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return summary_async<summary_mask::pairwise>(other, output, stream);
    }

    /// @brief Computes this sketch's cardinality reduction on the GPU.
    ///
    /// @p empty_out receives the empty-register count and @p estimate_out the cardinality estimate.
    [[nodiscard]] Result<void> cardinality_async(
        uint64_t* empty_out,
        double* estimate_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        detail::cardinality_kernel<BucketCount>
            <<<1, 64, 0, stream.get()>>>(registers_, empty_out, estimate_out);
        return cuda_try(cudaGetLastError());
    }

    /// @brief Computes BBTools and paper-style HybridDDL estimates in one GPU register scan.
    [[nodiscard]] Result<void> hybrid_cardinality_async(
        hybrid_cardinality_estimates* output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        detail::hybrid_cardinality_kernel<BucketCount>
            <<<1, detail::block_size, 0, stream.get()>>>(registers_, output);
        return cuda_try(cudaGetLastError());
    }


    /// @brief Extracts per-register winner counts and the saturation flag to device outputs.
    ///
    /// @p counts_out must hold `BucketCount` `uint16_t` entries and @p saturation_out one
    /// `uint32_t`, all valid until @p stream completes.
    [[nodiscard]] Result<void> winner_counts_async(
        uint16_t* counts_out,
        uint32_t* saturation_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        detail::winner_counts_kernel<BucketCount><<<1, detail::block_size, 0, stream.get()>>>(
            registers_, saturation_, counts_out, saturation_out
        );
        return cuda_try(cudaGetLastError());
    }

    /// @brief Weighted k-mer identity from a raw pair summary.
    ///
    /// Pads non-zero divisors below six to suppress similarity from sparse random collisions.
    /// Returns `std::nullopt` when the divisor is zero.
    [[nodiscard]] std::optional<double> wkid(pairwise_summary const& summary) const noexcept {
        auto const equal = summary.counts.equal;
        auto const divisor = equal + std::min(summary.counts.lower, summary.counts.higher);
        if (divisor == 0U) {
            return std::nullopt;
        }
        return static_cast<double>(equal) / static_cast<double>(std::max(divisor, 6U));
    }

    /// @brief Average nucleotide identity estimate from a raw pair summary.
    [[nodiscard]] std::optional<double> ani(pairwise_summary const& summary) const noexcept {
        auto const identity = wkid(summary);
        if (!identity) {
            return std::nullopt;
        }
        return std::pow(*identity, 1.0 / static_cast<double>(K));
    }

    /// @brief Fraction of `this`'s content shared with the other sketch.
    ///
    /// `equal / (equal + higher)`; `std::nullopt` when the divisor is zero.
    [[nodiscard]] std::optional<double> containment(
        pairwise_summary const& summary
    ) const noexcept {
        auto const divisor = summary.counts.equal + summary.counts.higher;
        if (divisor == 0U) {
            return std::nullopt;
        }
        return static_cast<double>(summary.counts.equal) / static_cast<double>(divisor);
    }

    /// @brief Fraction of the other sketch's content present in `this`, clamped to `[0, 1]`.
    ///
    /// `(equal + higher) / (equal + lower)`; `std::nullopt` when the divisor is zero.
    [[nodiscard]] std::optional<double> completeness(
        pairwise_summary const& summary
    ) const noexcept {
        auto const divisor = summary.counts.equal + summary.counts.lower;
        if (divisor == 0U) {
            return std::nullopt;
        }
        auto const value = static_cast<double>(summary.counts.equal + summary.counts.higher) /
                           static_cast<double>(divisor);
        return std::clamp(value, 0.0, 1.0);
    }

   private:
    register_type* registers_ = nullptr;
    uint32_t* saturation_ = nullptr;
};

}  // namespace cuddl
