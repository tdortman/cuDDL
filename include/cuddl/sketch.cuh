#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <cuddl/detail/sketch_view.cuh>
#include <cuddl/error.hpp>
#include <cuddl/hybrid_cardinality.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl {
namespace detail {

/// @brief RAII device buffer owned by a @ref cuddl::sketch for temporary host-visible results.
template <typename T>
struct device_buffer {
    T* pointer = nullptr;

    device_buffer() = default;

    explicit device_buffer(size_t count) : pointer(nullptr) {
        CUDDL_CUDA_ABORT(cudaMalloc(&pointer, count * sizeof(T)));
    }

    device_buffer(device_buffer const&) = delete;
    device_buffer& operator=(device_buffer const&) = delete;

    device_buffer(device_buffer&& other) noexcept : pointer(other.pointer) {
        other.pointer = nullptr;
    }
    device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
            destroy();
            pointer = other.pointer;
            other.pointer = nullptr;
        }
        return *this;
    }

    ~device_buffer() {
        destroy();
    }

    void destroy() noexcept {
        if (pointer != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(pointer));
            pointer = nullptr;
        }
    }
};

/// @brief Device-visible scratch for the owning cardinality wrapper.
struct cardinality_scratch {
    uint64_t empty;
    double estimate;
};

/// @brief Persistent zero-copy scratch used by owning synchronous result wrappers.
///
/// The scratch is allocated as mapped pinned host memory. Kernels write through the mapped
/// device pointer, so synchronous `summary()`, `cardinality()`, and `hybrid_cardinality()` calls
/// only launch and synchronise without a device-to-host copy.
struct sketch_scratch {
    cardinality_scratch cardinality;
    hybrid_cardinality_estimates hybrid;
    pairwise_summary summary;
};

}  // namespace detail

/**
 * @brief Move-only owning DDL sketch backed by one contiguous device allocation.
 *
 * The device allocation holds `BucketCount` packed `uint32_t` registers and one aligned
 * `uint32_t` saturation flag. A separate mapped pinned-host scratch area receives small
 * synchronous results without a device-to-host copy.
 */
template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
class sketch {
   public:
    using layout_type = Layout;
    using register_type = uint32_t;

    /// @brief Allocates the register array plus saturation flag and clears it (aborts on failure).
    explicit sketch(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        CUDDL_CUDA_ABORT(cudaMalloc(&storage_, allocation_words_ * sizeof(register_type)));
        CUDDL_CUDA_ABORT(
            cudaHostAlloc(&mapped_scratch_, sizeof(detail::sketch_scratch), cudaHostAllocMapped)
        );
        CUDDL_CUDA_ABORT(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&mapped_scratch_device_), mapped_scratch_, 0U
        ));
        CUDDL_UNWRAP(view().clear_async(stream));
        CUDDL_CUDA_ABORT(cudaStreamSynchronize(stream.get()));
    }

    sketch(sketch const&) = delete;
    sketch& operator=(sketch const&) = delete;

    sketch(sketch&& other) noexcept
        : storage_(other.storage_),
          mapped_scratch_(other.mapped_scratch_),
          mapped_scratch_device_(other.mapped_scratch_device_) {
        other.storage_ = nullptr;
        other.mapped_scratch_ = nullptr;
        other.mapped_scratch_device_ = nullptr;
    }
    sketch& operator=(sketch&& other) noexcept {
        if (this != &other) {
            if (storage_ != nullptr) {
                CUDDL_CUDA_ABORT(cudaFree(storage_));
            }
            if (mapped_scratch_ != nullptr) {
                CUDDL_CUDA_ABORT(cudaFreeHost(mapped_scratch_));
            }
            storage_ = other.storage_;
            mapped_scratch_ = other.mapped_scratch_;
            mapped_scratch_device_ = other.mapped_scratch_device_;
            other.storage_ = nullptr;
            other.mapped_scratch_ = nullptr;
            other.mapped_scratch_device_ = nullptr;
        }
        return *this;
    }

    ~sketch() {
        if (storage_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(storage_));
        }
        if (mapped_scratch_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFreeHost(mapped_scratch_));
        }
    }

    /// @brief Packed device registers.
    [[nodiscard]] device_span<register_type> data() noexcept {
        return {storage_, BucketCount};
    }

    /// @brief Packed device registers.
    [[nodiscard]] device_span<register_type const> data() const noexcept {
        return {storage_, BucketCount};
    }

    /// @brief Number of registers in the sketch.
    [[nodiscard]] static constexpr size_t bucket_count() noexcept {
        return BucketCount;
    }

    /// @brief Compile-time k-mer length.
    [[nodiscard]] static constexpr uint32_t kmer_length() noexcept {
        return K;
    }

    [[nodiscard]] Result<void> clear_async(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return view().clear_async(stream);
    }

    [[nodiscard]] Result<void> clear(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        if (auto const result = clear_async(stream); !result) {
            return result;
        }
        return cuda_try(cudaStreamSynchronize(stream.get()));
    }

    [[nodiscard]] Result<void> add_async(
        device_span<uint64_t const> input,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return view().add_async(input, stream);
    }

    [[nodiscard]] Result<void> add(
        device_span<uint64_t const> input,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        if (auto const result = add_async(input, stream); !result) {
            return result;
        }
        return cuda_try(cudaStreamSynchronize(stream.get()));
    }

    [[nodiscard]] Result<void> add_async(
        thrust::device_vector<uint64_t> const& input,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return add_async({thrust::raw_pointer_cast(input.data()), input.size()}, stream);
    }

    [[nodiscard]] Result<void> add(
        thrust::device_vector<uint64_t> const& input,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return add({thrust::raw_pointer_cast(input.data()), input.size()}, stream);
    }

    /// @brief Computes a fused pairwise summary into caller-owned device storage.
    template <bool IncludeCardinality = false>
    [[nodiscard]] Result<void> summary_async(
        sketch const& other,
        pairwise_summary& output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return view().template summary_async<IncludeCardinality>(other.view(), output, stream);
    }

    /// @brief Computes raw pairwise counts into caller-owned device storage.
    [[nodiscard]] Result<void> compare_async(
        sketch const& other,
        pairwise_summary& output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return summary_async(other, output, stream);
    }

    /// @brief Host-side fused pairwise summary (uses mapped scratch, synchronises).
    template <bool IncludeCardinality = false>
    [[nodiscard]] Result<pairwise_summary> summary(
        sketch const& other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = &mapped_scratch_device_->summary;
        if (auto const result =
                view().template summary_async<IncludeCardinality>(other.view(), *output, stream);
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        return mapped_scratch_->summary;
    }

    /// @brief Raw pairwise counts against @p other (host result).
    [[nodiscard]] Result<pairwise_summary> compare(
        sketch const& other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return summary(other, stream);
    }

    /// @brief Weighted k-mer identity from a raw pair summary.
    ///
    /// Pads non-zero divisors below six to suppress similarity from sparse random collisions.
    /// Returns `std::nullopt` when the divisor is zero.
    [[nodiscard]] static std::optional<double> wkid(pairwise_summary const& summary) noexcept {
        auto const equal = summary.counts.equal;
        auto const divisor = equal + std::min(summary.counts.lower, summary.counts.higher);
        if (divisor == 0U) {
            return std::nullopt;
        }
        return static_cast<double>(equal) / static_cast<double>(std::max(divisor, 6U));
    }

    /// @brief Average nucleotide identity estimate from a raw pair summary.
    [[nodiscard]] static std::optional<double> ani(pairwise_summary const& summary) noexcept {
        auto const identity = wkid(summary);
        if (!identity) {
            return std::nullopt;
        }
        return std::pow(*identity, 1.0 / static_cast<double>(K));
    }

    /// @brief Fraction of the left sketch's content shared with the right sketch.
    ///
    /// `equal / (equal + higher)`; `std::nullopt` when the divisor is zero.
    [[nodiscard]] static std::optional<double> containment(
        pairwise_summary const& summary
    ) noexcept {
        auto const divisor = summary.counts.equal + summary.counts.higher;
        if (divisor == 0U) {
            return std::nullopt;
        }
        return static_cast<double>(summary.counts.equal) / static_cast<double>(divisor);
    }

    /// @brief Relative cardinality of the left sketch to the right sketch, clamped to `[0, 1]`.
    ///
    /// BBTools completeness: `(equal + higher) / (equal + lower)`; `std::nullopt` when the
    /// divisor is zero. This is a size-ratio estimate, not reverse containment.
    [[nodiscard]] static std::optional<double> completeness(
        pairwise_summary const& summary
    ) noexcept {
        auto const divisor = summary.counts.equal + summary.counts.lower;
        if (divisor == 0U) {
            return std::nullopt;
        }
        auto const value = static_cast<double>(summary.counts.equal + summary.counts.higher) /
                           static_cast<double>(divisor);
        return std::clamp(value, 0.0, 1.0);
    }

    /// @brief Computes this sketch's hybridDDL cardinality reduction on the GPU.
    [[nodiscard]] Result<void> cardinality_async(
        uint64_t* empty_out,
        double* estimate_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return view().cardinality_async(empty_out, estimate_out, stream);
    }

    /// @brief Host-side cardinality estimate for this sketch.
    [[nodiscard]] Result<double> cardinality(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = &mapped_scratch_device_->cardinality;
        if (auto const result = view().cardinality_async(&output->empty, &output->estimate, stream);
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        return mapped_scratch_->cardinality.estimate;
    }

    /// @brief Experimental BBTools and paper-style HybridDDL estimates for comparison.
    [[nodiscard]] Result<hybrid_cardinality_estimates> hybrid_cardinality(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = &mapped_scratch_device_->hybrid;
        if (auto const result = view().hybrid_cardinality_async(output, stream); !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        return mapped_scratch_->hybrid;
    }

    /// @brief Watches the winner-count extraction on the GPU (caller-owned outputs).
    [[nodiscard]] Result<void> winner_counts_async(
        uint16_t* counts_out,
        uint32_t* saturation_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return view().winner_counts_async(counts_out, saturation_out, stream);
    }

    /// @brief Extracts per-register winner counts and saturation flag (host result).
    [[nodiscard]] Result<std::pair<std::vector<uint16_t>, bool>> winner_counts(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        detail::device_buffer<uint16_t> counts(BucketCount);
        detail::device_buffer<uint32_t> saturation(1);
        if (auto const result =
                view().winner_counts_async(counts.pointer, saturation.pointer, stream);
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        std::vector<uint16_t> host_counts(BucketCount);
        uint32_t host_saturation = 0;
        if (auto const result = cuda_try(cudaMemcpy(
                host_counts.data(),
                counts.pointer,
                BucketCount * sizeof(uint16_t),
                cudaMemcpyDeviceToHost
            ));
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaMemcpy(
                &host_saturation, saturation.pointer, sizeof(uint32_t), cudaMemcpyDeviceToHost
            ));
            !result) {
            return Err(result.error());
        }
        return std::pair{std::move(host_counts), host_saturation != 0U};
    }

   private:
    using view_type = detail::sketch_view<K, BucketCount, Layout>;

    [[nodiscard]] view_type view() const noexcept {
        return view_type({storage_, BucketCount}, storage_[BucketCount]);
    }

    /// @brief One pad register after the saturation flag keeps the next allocation aligned.
    static constexpr size_t padded_prefix_words_ = BucketCount + 2U;
    static constexpr size_t allocation_words_ = padded_prefix_words_;

    register_type* storage_ = nullptr;
    detail::sketch_scratch* mapped_scratch_ = nullptr;
    detail::sketch_scratch* mapped_scratch_device_ = nullptr;
};

}  // namespace cuddl
