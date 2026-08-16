#include <cuddl/cuddl.cuh>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <type_traits>

namespace {

TEST(SketchTest, ConstructsAndComparesPackedKmers) {
    static_assert(std::is_trivially_copyable_v<cuddl::sketch_ref<25, 2048>>);

    cuddl::sketch<25, 2048> left;
    cuddl::sketch<25, 2048> right;

    uint64_t* items = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&items, sizeof(*items)));
    uint64_t const packed_kmer = 0x123456789ULL;
    ASSERT_EQ(
        cudaSuccess, cudaMemcpy(items, &packed_kmer, sizeof(packed_kmer), cudaMemcpyHostToDevice)
    );

    left.add(items, items + 1);
    right.add(items, items + 1);

    auto const result = left.compare(right.ref());
    EXPECT_EQ(result.lower, 0U);
    EXPECT_EQ(result.equal, 1U);
    EXPECT_EQ(result.higher, 0U);

    EXPECT_EQ(cudaSuccess, cudaFree(items));
}

}  // namespace
