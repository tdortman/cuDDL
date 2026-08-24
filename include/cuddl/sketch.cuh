#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <thrust/device_vector.h>
#include <thrust/memory.h>
#include <memory>
#include <utility>
#include <vector>

#include <cuddl/error.hpp>
#include <cuddl/hybrid_cardinality.cuh>
#include <cuddl/pairwise_counts.cuh>
#include <cuddl/sketch_ref.cuh>

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

/// @brief Device scratch for the owning cardinality wrapper.
struct cardinality_scratch {
    uint64_t empty;
    double estimate;
};

/// @brief Persistent device scratch used by owning synchronous result wrappers.
///
/// Keeping this in the same allocation as the registers avoids a `cudaMalloc`/`cudaFree` pair on
/// every synchronous `summary()`, `cardinality()`, or `hybrid_cardinality()` call. Those calls
/// synchronise before returning, so reuse across calls is stream-safe.
struct sketch_scratch {
    cardinality_scratch cardinality;
    hybrid_cardinality_estimates hybrid;
    pairwise_summary summary;
};

}  // namespace detail

/**
 * @brief Move-only owning DDL sketch backed by one contiguous device allocation.
 *
 * The allocation holds `BucketCount` packed `uint32_t` registers, one aligned `uint32_t`
 * saturation flag, one alignment pad register, and a small persistent result scratch area.
 * The owning wrapper delegates all stream-ordered work to its @ref sketch_ref and uses the
 * scratch area for synchronous result copies instead of allocating temporary device buffers per
 * call.
 */
template <uint32_t K, size_t BucketCount>
class sketch {
   public:
    using ref_type = sketch_ref<K, BucketCount>;
    using register_type = typename ref_type::register_type;

    /// @brief Allocates the register array plus saturation flag and clears it (aborts on failure).
    explicit sketch(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        CUDDL_CUDA_ABORT(cudaMalloc(&storage_, allocation_words_ * sizeof(register_type)));
        CUDDL_UNWRAP(ref().clear_async(stream));
        CUDDL_CUDA_ABORT(cudaStreamSynchronize(stream.get()));
    }

    sketch(sketch const&) = delete;
    sketch& operator=(sketch const&) = delete;

    sketch(sketch&& other) noexcept : storage_(other.storage_) {
        other.storage_ = nullptr;
    }
    sketch& operator=(sketch&& other) noexcept {
        if (this != &other) {
            if (storage_ != nullptr) {
                CUDDL_CUDA_ABORT(cudaFree(storage_));
            }
            storage_ = other.storage_;
            other.storage_ = nullptr;
        }
        return *this;
    }

    ~sketch() {
        if (storage_ != nullptr) {
            CUDDL_CUDA_ABORT(cudaFree(storage_));
        }
    }

    /// @brief Non-owning reference over this sketch's registers and saturation flag.
    [[nodiscard]] ref_type ref() const noexcept {
        return ref_type({storage_, BucketCount}, storage_[BucketCount]);
    }

    [[nodiscard]] Result<void> clear_async(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().clear_async(stream);
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
        return ref().add_async(input, stream);
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
        ref_type other,
        pairwise_summary& output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().template summary_async<IncludeCardinality>(other, output, stream);
    }

    /// @brief Host-side fused pairwise summary (uses persistent scratch, synchronises).
    template <bool IncludeCardinality = false>
    [[nodiscard]] Result<pairwise_summary> summary(
        ref_type other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = &scratch()->summary;
        if (auto const result =
                ref().template summary_async<IncludeCardinality>(other, *output, stream);
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        pairwise_summary host{};
        if (auto const result =
                cuda_try(cudaMemcpy(&host, output, sizeof(host), cudaMemcpyDeviceToHost));
            !result) {
            return Err(result.error());
        }
        return host;
    }

    /// @brief Raw pairwise counts against @p other (host result).
    [[nodiscard]] Result<pairwise_summary> compare(
        ref_type other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return summary(other, stream);
    }

    /// @brief Computes this sketch's hybridDDL cardinality reduction on the GPU.
    [[nodiscard]] Result<void> cardinality_async(
        uint64_t* empty_out,
        double* estimate_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().cardinality_async(empty_out, estimate_out, stream);
    }

    /// @brief Host-side cardinality estimate for this sketch.
    [[nodiscard]] Result<double> cardinality(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = scratch();
        if (auto const result = ref().cardinality_async(
                &output->cardinality.empty, &output->cardinality.estimate, stream
            );
            !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        double host = 0.0;
        if (auto const result = cuda_try(cudaMemcpy(
                &host, &output->cardinality.estimate, sizeof(host), cudaMemcpyDeviceToHost
            ));
            !result) {
            return Err(result.error());
        }
        return host;
    }

    /// @brief Experimental BBTools and paper-style HybridDDL estimates for comparison.
    [[nodiscard]] Result<hybrid_cardinality_estimates> hybrid_cardinality(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        auto* const output = scratch();
        if (auto const result = ref().hybrid_cardinality_async(&output->hybrid, stream); !result) {
            return Err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Err(result.error());
        }
        hybrid_cardinality_estimates host{};
        if (auto const result =
                cuda_try(cudaMemcpy(&host, &output->hybrid, sizeof(host), cudaMemcpyDeviceToHost));
            !result) {
            return Err(result.error());
        }
        return host;
    }

    /// @brief Watches the winner-count extraction on the GPU (caller-owned outputs).
    [[nodiscard]] Result<void> winner_counts_async(
        uint16_t* counts_out,
        uint32_t* saturation_out,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().winner_counts_async(counts_out, saturation_out, stream);
    }

    /// @brief Extracts per-register winner counts and saturation flag (host result).
    [[nodiscard]] Result<std::pair<std::vector<uint16_t>, bool>> winner_counts(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        detail::device_buffer<uint16_t> counts(BucketCount);
        detail::device_buffer<uint32_t> saturation(1);
        if (auto const result =
                ref().winner_counts_async(counts.pointer, saturation.pointer, stream);
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
    /// @brief One pad register after the saturation flag keeps the scratch area 8-byte aligned.
    static constexpr size_t padded_prefix_words_ = BucketCount + 2U;
    static constexpr size_t scratch_words_ =
        (sizeof(detail::sketch_scratch) + sizeof(register_type) - 1U) / sizeof(register_type);
    static constexpr size_t allocation_words_ = padded_prefix_words_ + scratch_words_;

    static_assert(
        (padded_prefix_words_ * sizeof(register_type)) % alignof(detail::sketch_scratch) == 0U
    );

    [[nodiscard]] detail::sketch_scratch* scratch() const noexcept {
        return reinterpret_cast<detail::sketch_scratch*>(storage_ + padded_prefix_words_);
    }

    register_type* storage_ = nullptr;
};

}  // namespace cuddl
