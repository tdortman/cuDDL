#pragma once

#include <cuda_runtime.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

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
        cuda_abort_on_error(cudaMalloc(&pointer, count * sizeof(T)));
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

}  // namespace detail

/**
 * @brief Move-only owning DDL sketch backed by one contiguous device allocation.
 *
 * The allocation holds `BucketCount` packed `uint32_t` registers followed by one aligned
 * `uint32_t` saturation flag. The owning wrapper delegates all stream-ordered work to its
 * @ref sketch_ref and owns temporary output storage, host copies, and synchronisation.
 */
template <uint32_t K, size_t BucketCount>
class sketch {
   public:
    using ref_type = sketch_ref<K, BucketCount>;
    using register_type = typename ref_type::register_type;

    /// @brief Allocates the register array plus saturation flag and clears it (aborts on failure).
    explicit sketch(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}) {
        cuda_abort_on_error(cudaMalloc(&storage_, (BucketCount + 1) * sizeof(register_type)));
        require_void(ref().clear_async(stream));
        cuda_abort_on_error(cudaStreamSynchronize(stream.get()));
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
        return ref_type(storage_, storage_ + BucketCount);
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
        uint64_t const* first,
        uint64_t const* last,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().add_async(first, last, stream);
    }

    [[nodiscard]] Result<void> add(
        uint64_t const* first,
        uint64_t const* last,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        if (auto const result = add_async(first, last, stream); !result) {
            return result;
        }
        return cuda_try(cudaStreamSynchronize(stream.get()));
    }

    /// @brief Computes a fused pairwise summary into caller-owned device storage.
    template <uint32_t Mask = summary_mask::pairwise>
    [[nodiscard]] Result<void> summary_async(
        ref_type other,
        pairwise_summary* output,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const noexcept {
        return ref().template summary_async<Mask>(other, output, stream);
    }

    /// @brief Host-side fused pairwise summary (allocates temporary device output, synchronises).
    template <uint32_t Mask = summary_mask::pairwise>
    [[nodiscard]] Result<pairwise_summary> summary(
        ref_type other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        detail::device_buffer<pairwise_summary> output(1);
        if (auto const result = ref().template summary_async<Mask>(other, output.pointer, stream);
            !result) {
            return Result<pairwise_summary>::err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Result<pairwise_summary>::err(result.error());
        }
        pairwise_summary host{};
        if (auto const result =
                cuda_try(cudaMemcpy(&host, output.pointer, sizeof(host), cudaMemcpyDeviceToHost));
            !result) {
            return Result<pairwise_summary>::err(result.error());
        }
        return Result<pairwise_summary>::ok(host);
    }

    /// @brief Raw pairwise counts against @p other (host result).
    [[nodiscard]] Result<pairwise_summary> compare(
        ref_type other,
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        return summary<summary_mask::pairwise>(other, stream);
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
        detail::device_buffer<uint64_t> empty(1);
        detail::device_buffer<double> estimate(1);
        if (auto const result = ref().cardinality_async(empty.pointer, estimate.pointer, stream);
            !result) {
            return Result<double>::err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Result<double>::err(result.error());
        }
        double host = 0.0;
        if (auto const result =
                cuda_try(cudaMemcpy(&host, estimate.pointer, sizeof(host), cudaMemcpyDeviceToHost));
            !result) {
            return Result<double>::err(result.error());
        }
        return Result<double>::ok(host);
    }

    /// @brief Experimental BBTools and paper-style HybridDDL estimates for comparison.
    [[nodiscard]] Result<hybrid_cardinality_estimates> hybrid_cardinality(
        cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{nullptr}}
    ) const {
        detail::device_buffer<hybrid_cardinality_estimates> output(1);
        if (auto const result = ref().hybrid_cardinality_async(output.pointer, stream); !result) {
            return Result<hybrid_cardinality_estimates>::err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Result<hybrid_cardinality_estimates>::err(result.error());
        }
        hybrid_cardinality_estimates host{};
        if (auto const result =
                cuda_try(cudaMemcpy(&host, output.pointer, sizeof(host), cudaMemcpyDeviceToHost));
            !result) {
            return Result<hybrid_cardinality_estimates>::err(result.error());
        }
        return Result<hybrid_cardinality_estimates>::ok(host);
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
            return Result<std::pair<std::vector<uint16_t>, bool>>::err(result.error());
        }
        if (auto const result = cuda_try(cudaStreamSynchronize(stream.get())); !result) {
            return Result<std::pair<std::vector<uint16_t>, bool>>::err(result.error());
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
            return Result<std::pair<std::vector<uint16_t>, bool>>::err(result.error());
        }
        if (auto const result = cuda_try(cudaMemcpy(
                &host_saturation, saturation.pointer, sizeof(uint32_t), cudaMemcpyDeviceToHost
            ));
            !result) {
            return Result<std::pair<std::vector<uint16_t>, bool>>::err(result.error());
        }
        return Result<std::pair<std::vector<uint16_t>, bool>>::ok(
            {std::move(host_counts), host_saturation != 0U}
        );
    }

   private:
    register_type* storage_ = nullptr;
};

}  // namespace cuddl
