#pragma once

#include <cuda/algorithm>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <cuddl/detail/construction.cuh>
#include <cuddl/detail/kernels.cuh>
#include <cuddl/detail/sequence_encode.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/error.hpp>
#include <cuddl/hybrid_cardinality.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl::detail {

/**
 * @brief Non-owning implementation view of a DDL sketch.
 *
 * Provides allocation-free, stream-ordered operations on an external allocation of
 * `BucketCount` `uint32_t` registers, each holding its bucket's winning score. Pass by value into
 * device or host code.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class sketch_view {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount >= (size_t{1} << 11) && BucketCount <= (size_t{1} << 17));
    static_assert((BucketCount & (BucketCount - 1)) == 0);

   public:
    using register_type = uint32_t;
    using layout_type = Layout;

    /// @brief Constructs a reference over @p registers.
    __host__ __device__ constexpr explicit sketch_view(
        device_span<register_type> registers
    ) noexcept
        : registers_(registers) {}

    /// @brief Device registers.
    [[nodiscard]] __host__ __device__ constexpr device_span<register_type> data() const noexcept {
        return registers_;
    }

    /// @brief Number of registers in the sketch.
    [[nodiscard]] static constexpr size_t bucket_count() noexcept {
        return BucketCount;
    }

    /// @brief Compile-time k-mer length.
    [[nodiscard]] static constexpr uint32_t kmer_length() noexcept {
        return K;
    }

    /// @brief Resets every register to zero.
    [[nodiscard]] Result<void> clear_async(cuda::stream_ref stream) const noexcept {
        return cuda_try([&] {
            cuda::fill_bytes(stream, cuda::std::span{registers_.data(), registers_.size()}, 0);
        });
    }

    /// @brief Accumulates packed device k-mers into the existing sketch without clearing it.
    ///
    /// The input must remain valid until @p stream completes.
    [[nodiscard]] Result<void>
    add_async(device_span<uint64_t const> input, cuda::stream_ref stream) const noexcept {
        return detail::launch_construction<BucketCount, Layout>(input, registers_, stream);
    }

    /// @brief Accumulates device-resident raw ASCII bases without clearing.
    ///
    /// Expects raw contiguous ASCII (no FASTA/newline stripping). Canonicalises
    /// case-insensitively with the packed path; ambiguity breaks windows and emits no k-mer
    /// across it. Accumulates without reset; short input below K is a no-op. Windows per call
    /// are capped at UINT32_MAX, so size must not exceed UINT32_MAX + K - 1. No hidden carry:
    /// callers supply the K-1 overlap within one record and no overlap between distinct
    /// records. No host copies. The input must remain valid until @p stream completes.
    [[nodiscard]] Result<void>
    add_sequence_async(device_span<char const> sequence, cuda::stream_ref stream) const noexcept {
        return detail::launch_sequence_add<BucketCount, Layout>(sequence, K, registers_, stream);
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

   private:
    device_span<register_type> registers_;
};

}  // namespace cuddl::detail
