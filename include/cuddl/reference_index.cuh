#pragma once

#include <cuddl/reference_database.cuh>

namespace cuddl {

/// @brief Move-only acceleration data for an immutable reference database.
/// Build and load return the same type; pass its address to a database search.
/// The allocation stream must outlive the index.
template <uint32_t K, size_t BucketCount, typename Layout>
class reference_index {
    using database_type = reference_database<K, BucketCount, Layout>;
    friend class reference_index_file;
    friend class reference_database<K, BucketCount, Layout>;

    using score_type = typename database_type::score_type;

   public:
    reference_index(reference_index const&) = delete;

    reference_index& operator=(reference_index const&) = delete;

    __host__ ~reference_index() {}

    reference_index(reference_index&& other) noexcept
        : identity_(std::move(other.identity_)),
          index_offsets_(std::move(other.index_offsets_)),
          index_postings_(std::move(other.index_postings_)),
          index_keys_(std::move(other.index_keys_)),
          key_directory_(std::move(other.key_directory_)),
          index_posting_capacity_(std::exchange(other.index_posting_capacity_, 0)),
          pair_fraction_(std::exchange(other.pair_fraction_, 0.0)),
          indexed_(std::exchange(other.indexed_, false)) {}

    /// @brief Move-assigns the index, leaving the source empty.
    reference_index& operator=(reference_index&& other) noexcept {
        if (this != &other) {
            identity_ = std::move(other.identity_);
            index_offsets_ = std::move(other.index_offsets_);
            index_postings_ = std::move(other.index_postings_);
            index_keys_ = std::move(other.index_keys_);
            key_directory_ = std::move(other.key_directory_);
            index_posting_capacity_ = std::exchange(other.index_posting_capacity_, 0);
            pair_fraction_ = std::exchange(other.pair_fraction_, 0.0);
            indexed_ = std::exchange(other.indexed_, false);
        }
        return *this;
    }

    /// @brief Dense index when offsets cover every cell, sparse when only keys are stored.
    [[nodiscard]] index_storage storage() const noexcept {
        return index_offsets_.empty() ? index_storage::sparse : index_storage::dense;
    }

    /// @brief Build acceleration data without taking ownership of or copying reference rows.
    [[nodiscard]] static Result<reference_index> build_async(
        database_type const& database,
        cuda::stream_ref stream,
        index_storage storage = index_storage::dense
    ) {
        auto index = CUDDL_TRY(build_index<score_type>(database, stream, storage));
        CUDDL_TRY(index.measure_pair_fraction(database, stream));
        return Result<reference_index>::ok(std::move(index));
    }

    /// @brief Fraction of all (query, reference, bucket) cells an index lookup visits for
    /// queries resembling the references; searches fall back to comparing bit-planes when the
    /// index would visit too many.
    [[nodiscard]] double pair_fraction() const noexcept {
        return pair_fraction_;
    }

    /// @brief Bytes the saved index file needs for offsets, postings, and keys.
    [[nodiscard]] size_t persistent_index_bytes() const noexcept {
        return index_offsets_.size() * sizeof(uint32_t) +
               index_postings_.size() * sizeof(uint32_t) + index_keys_.size() * sizeof(uint16_t);
    }

   private:
    /// @brief Measures @ref pair_fraction; synchronizes @p stream.
    [[nodiscard]] Result<void>
    measure_pair_fraction(database_type const& database, cuda::stream_ref stream) {
        auto const reference_count = database.reference_count();
        auto const compatibility = database.metadata().compatibility;
        pair_fraction_ = 0.0;
        if (reference_count == 0U || compatibility.indexed_bucket_count == 0U) {
            return Ok();
        }
        auto work = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<unsigned long long>(stream, stream.device(), 1U, 0ULL)
        );
        detail::index_pair_work_kernel<<<
            detail::
                warp_grid_blocks(static_cast<size_t>(compatibility.indexed_bucket_count) * 1024U),
            detail::block_size,
            0,
            stream.get()>>>(
            index_offsets_.empty() ? nullptr : index_offsets_.data(),
            index_keys_.empty() ? nullptr : index_keys_.data(),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(compatibility.key_mask) + 1U,
            work.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        unsigned long long host = 0U;
        CUDDL_CUDA_TRY(cuda::copy_bytes(stream, work, cuda::std::span{&host, size_t{1}}));
        CUDDL_CUDA_TRY(stream.sync());
        pair_fraction_ =
            static_cast<double>(host) / (static_cast<double>(reference_count) * reference_count *
                                         compatibility.indexed_bucket_count);
        return Ok();
    }

    reference_index(database_type const& database, cuda::stream_ref stream)
        : identity_(database.identity_),
          index_offsets_(stream, cuda::device_default_memory_pool(stream.device())),
          index_postings_(stream, cuda::device_default_memory_pool(stream.device())),
          index_keys_(stream, cuda::device_default_memory_pool(stream.device())),
          key_directory_(stream, cuda::device_default_memory_pool(stream.device())) {}

    template <typename Row>
    [[nodiscard]] static Result<reference_index>
    build_index(database_type const& source, cuda::stream_ref stream, index_storage storage) {
        auto compatibility = source.metadata().compatibility;
        // The database keeps only bit-planes; the index passes read decoded score rows, freed
        // in stream order once the build's kernels have run.
        auto decoded = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<Row>(
                stream,
                stream.device(),
                static_cast<size_t>(source.reference_count()) * BucketCount,
                cuda::no_init
            )
        );
        CUDDL_TRY(source.copy_scores_async({decoded.data(), decoded.size()}, stream));
        auto rows = device_span<Row const>{decoded.data(), decoded.size()};
        if (storage != index_storage::dense && storage != index_storage::sparse) {
            return Err(Error::invalid_argument("unsupported index storage"));
        }
        auto const validated = database_type::validate_rows(rows, compatibility);
        if (!validated) {
            return Err(validated.error());
        }
        auto const posting_capacity = detail::indexed_posting_count(*validated, compatibility);
        if (posting_capacity > std::numeric_limits<uint32_t>::max()) {
            return Err(Error::resource("index postings exceed 32-bit offsets"));
        }
        if (posting_capacity > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return Err(Error::resource("index posting allocation overflows"));
        }
        auto const cell_count = detail::indexed_cell_count(compatibility);
        auto const offset_count = cell_count + 1U;
        if (offset_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            return Err(Error::resource("dense index offset allocation overflows"));
        }

        auto index = CUDDL_CUDA_TRY(reference_index(source, stream));
        if (storage == index_storage::sparse) {
            CUDDL_TRY(
                index.template build_sparse_index<Row>(rows, compatibility, *validated, stream)
            );
            index.indexed_ = true;
            return Result<reference_index>::ok(std::move(index));
        }
        index.index_offsets_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), static_cast<size_t>(offset_count), cuda::no_init
            )
        );
        if (rows.empty()) {
            CUDDL_CUDA_TRY(cuda::fill_bytes(stream, index.index_offsets_, 0));
            index.indexed_ = true;
            return Result<reference_index>::ok(std::move(index));
        }
        index.index_postings_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), static_cast<size_t>(posting_capacity), cuda::no_init
            )
        );
        index.index_posting_capacity_ = static_cast<size_t>(posting_capacity);

        auto const indexed_row_count = static_cast<size_t>(posting_capacity);
        auto const reference_count = static_cast<uint32_t>(rows.size() / BucketCount);

        // Bucket-major transpose of the indexed buckets: the per-bucket count and scatter
        // passes read contiguous references from it instead of striding across rows, keeping
        // each bucket's dense key range L2-resident for its atomics.
        auto transposed = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<Row>(stream, stream.device(), indexed_row_count, cuda::no_init)
        );
        constexpr uint32_t transpose_tile = 32U;
        dim3 const transpose_grid(
            (compatibility.indexed_bucket_count + transpose_tile - 1U) / transpose_tile,
            (reference_count + transpose_tile - 1U) / transpose_tile
        );
        detail::transpose_indexed_scores_kernel<<<
            transpose_grid,
            dim3(transpose_tile, transpose_tile),
            0,
            stream.get()>>>(
            rows.data(),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(BucketCount),
            transposed.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());

        // A small wave-blocked grid sweeps the buckets in rounds so the in-flight buckets'
        // dense key slices and cursor ranges stay L2-resident; one bucket per resident wave
        // would leave the SMs idle, and one block per bucket thrashes the cache (the profiled
        // failure mode of the first bucket-major version).
        constexpr uint32_t build_wave_blocks = 64;
        constexpr uint32_t build_bucket_block_size = 1024;
        auto const bucket_blocks =
            std::min<uint32_t>(compatibility.indexed_bucket_count, build_wave_blocks);
        // Both bucket kernels keep a quarter-key table in dynamic shared memory (64 KiB for
        // 16-bit keys, 32 KiB for 15-bit keys, over the default 48 KiB cap). The required size
        // depends on this build's key width, so the limit is (re)configured before every launch.
        auto const key_count = static_cast<uint32_t>(compatibility.key_mask) + 1U;
        auto const bucket_smem_bytes = static_cast<size_t>(key_count / 4U) * sizeof(uint32_t);
        CUDDL_CUDA_TRY(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(detail::count_index_cells_bucket_kernel<Row>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bucket_smem_bytes)
        ));
        CUDDL_CUDA_TRY(cudaFuncSetAttribute(
            reinterpret_cast<void const*>(detail::scatter_index_postings_bucket_kernel<Row>),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bucket_smem_bytes)
        ));

        // The count flush writes every cell of every bucket slice exactly once (plain stores
        // from the owning block), so the offsets array needs no separate zeroing before the
        // exclusive scan.
        detail::count_index_cells_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed.data(),
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            index.index_offsets_.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        // CUB allocates and releases scan scratch on this stream through its pooled resource.
        CUDDL_CUDA_TRY(
            cub::DeviceScan::ExclusiveSum(
                index.index_offsets_.data(), static_cast<int64_t>(cell_count + 1U), stream
            )
        );

        detail::scatter_index_postings_bucket_kernel<<<
            bucket_blocks,
            build_bucket_block_size,
            bucket_smem_bytes,
            stream.get()>>>(
            transposed.data(),
            compatibility.indexed_bucket_count,
            reference_count,
            compatibility.key_mask,
            index.index_offsets_.data(),
            index.index_postings_.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        // Postings ascend by reference ID within each cell, so a search can restrict a cell to a
        // reference range with a binary search.
        auto sorted = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), static_cast<size_t>(posting_capacity), cuda::no_init
            )
        );
        CUDDL_CUDA_TRY(
            cub::DeviceSegmentedSort::SortKeys(
                index.index_postings_.data(),
                sorted.data(),
                static_cast<int64_t>(posting_capacity),
                static_cast<int64_t>(cell_count),
                index.index_offsets_.data(),
                index.index_offsets_.data() + 1,
                stream
            )
        );
        index.index_postings_ = std::move(sorted);
        index.indexed_ = true;
        return Result<reference_index>::ok(std::move(index));
    }

    template <typename Row>
    [[nodiscard]] Result<void> build_sparse_index(
        device_span<Row const> rows,
        score_compatibility const& compatibility,
        uint32_t reference_count,
        cuda::stream_ref stream
    ) {
        auto const size =
            static_cast<size_t>(detail::indexed_posting_count(reference_count, compatibility));
        index_posting_capacity_ = size;
        if (size == 0U) {
            return Ok();
        }
        auto const device = stream.device();
        index_keys_ =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint16_t>(stream, device, size, cuda::no_init));
        index_postings_ =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, size, cuda::no_init));
        auto keys =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint16_t>(stream, device, size, cuda::no_init));
        auto ids =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, device, size, cuda::no_init));
        auto transposed =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<Row>(stream, device, size, cuda::no_init));
        detail::transpose_indexed_scores_kernel<<<
            dim3((compatibility.indexed_bucket_count + 31U) / 32U, (reference_count + 31U) / 32U),
            dim3(32U, 32U),
            0,
            stream.get()>>>(
            rows.data(),
            reference_count,
            compatibility.indexed_bucket_count,
            static_cast<uint32_t>(BucketCount),
            transposed.data()
        );
        CUDDL_CUDA_TRY(cudaGetLastError());
        auto const mask = compatibility.key_mask;
        // Zero denotes an empty row. Folded 15-bit keys are shifted by one so that
        // a nonempty score masked to zero remains distinguishable from an empty row.
        CUDDL_CUDA_TRY(
            cub::DeviceTransform::Transform(
                transposed.data(), keys.data(), size, detail::sparse_index_key{mask}, stream
            )
        );
        CUDDL_CUDA_TRY(
            cub::DeviceTransform::Transform(
                cuda::make_counting_iterator(uint32_t{0}),
                ids.data(),
                size,
                detail::sparse_reference_id{reference_count},
                stream
            )
        );
        auto const segment_offsets = cuda::make_transform_iterator(
            cuda::make_counting_iterator(uint32_t{0}),
            detail::sparse_segment_offset{reference_count}
        );
        // Stable, so reference IDs, which arrive ascending per bucket, stay ascending per key.
        CUDDL_CUDA_TRY(
            cub::DeviceSegmentedSort::StableSortPairs(
                keys.data(),
                index_keys_.data(),
                ids.data(),
                index_postings_.data(),
                static_cast<int64_t>(size),
                static_cast<int64_t>(compatibility.indexed_bucket_count),
                segment_offsets,
                segment_offsets + 1,
                stream
            )
        );
        return build_key_directory(reference_count, compatibility.indexed_bucket_count, stream);
    }

    /// @brief Derives the sparse key directory that narrows batch posting-range lookups.
    [[nodiscard]] Result<void> build_key_directory(
        uint32_t reference_count,
        uint32_t indexed_bucket_count,
        cuda::stream_ref stream
    ) {
        auto const size = static_cast<size_t>(detail::sparse_directory_entries(reference_count)) *
                          indexed_bucket_count;
        if (index_keys_.empty() || size == 0U) {
            return Ok();
        }
        key_directory_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), size, cuda::no_init)
        );
        auto const blocks = static_cast<uint32_t>(
            std::min<size_t>((size + detail::block_size - 1U) / detail::block_size, 65535U)
        );
        detail::build_sparse_directory_kernel<<<blocks, detail::block_size, 0, stream.get()>>>(
            index_keys_.data(), reference_count, indexed_bucket_count, key_directory_.data()
        );
        return cuda_try(cudaGetLastError());
    }

    std::shared_ptr<char const> identity_;
    cuda::device_buffer<uint32_t> index_offsets_;
    cuda::device_buffer<uint32_t> index_postings_;
    cuda::device_buffer<uint16_t> index_keys_;
    // Radix directory over index_keys_ for batch range lookups; derived, not persisted.
    cuda::device_buffer<uint32_t> key_directory_;
    size_t index_posting_capacity_{};
    double pair_fraction_{};
    bool indexed_{};
};

}  // namespace cuddl
