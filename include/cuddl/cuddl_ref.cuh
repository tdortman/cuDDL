#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

namespace cuddl::experimental {

struct pairwise_counts {
    uint32_t lower{};
    uint32_t equal{};
    uint32_t higher{};

    __host__ __device__ pairwise_counts& operator+=(pairwise_counts const other) noexcept {
        lower += other.lower;
        equal += other.equal;
        higher += other.higher;
        return *this;
    }

    friend __host__ __device__ pairwise_counts
    operator+(pairwise_counts left, pairwise_counts const right) noexcept {
        return left += right;
    }

    friend bool operator==(pairwise_counts const&, pairwise_counts const&) = default;
};

template <uint32_t K, size_t BucketCount>
class sketch_ref {
    static_assert(K >= 1 && K <= 31);
    static_assert(BucketCount != 0 && (BucketCount & (BucketCount - 1)) == 0);

   public:
    using register_type = uint16_t;

    __host__ __device__ constexpr explicit sketch_ref(register_type* registers) noexcept
        : registers_{registers} {}

    [[nodiscard]] __host__ __device__ constexpr register_type* data() const noexcept {
        return registers_;
    }

    [[nodiscard]] static constexpr size_t bucket_count() noexcept {
        return BucketCount;
    }

   private:
    register_type* registers_;
};

}  // namespace cuddl::experimental
