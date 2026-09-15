#pragma once

#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>
#include <zlib.h>

#include <cuddl/detail/database_staging.cuh>
#include <cuddl/detail/sequence_encode.cuh>
#include <cuddl/fastx.hpp>
#include <cuddl/reference_database.cuh>
#include <cuddl/sketch.cuh>

namespace cuddl::detail {

// The file format uses little-endian integers, never native struct representations.
template <typename T>
inline T database_file_little_endian(T value) {
    if constexpr (std::endian::native == std::endian::little) {
        return value;
    } else {
        T result = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            result = static_cast<T>((result << 8U) | (value & 0xffU));
            value >>= 8U;
        }
        return result;
    }
}

struct database_file_reader {
    std::ifstream input;
    uint64_t remaining{};
    uLong checksum = crc32(0, nullptr, 0);

    Result<void> bytes(void* destination, size_t size) {
        if (size > remaining ||
            size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            return Err(Error::invalid_argument("truncated reference database file"));
        }
        if (size != 0) {
            input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
            if (!input) return Err(Error::resource("cannot read reference database file"));
            checksum = crc32_z(checksum, static_cast<Bytef const*>(destination), size);
        }
        remaining -= size;
        return Ok();
    }

    template <typename T>
    Result<void> value(T& item) {
        CUDDL_TRY(bytes(&item, sizeof(item)));
        item = database_file_little_endian(item);
        return Ok();
    }

    Result<void> words(std::vector<uint32_t>& items) {
        CUDDL_TRY(bytes(items.data(), items.size() * sizeof(uint32_t)));
        if constexpr (std::endian::native != std::endian::little) {
            for (auto& item : items) {
                item = database_file_little_endian(item);
            }
        }
        return Ok();
    }
};

struct database_file_writer {
    std::string temporary;
    FILE* output{};
    uLong checksum = crc32(0, nullptr, 0);

    ~database_file_writer() {
        if (output) std::fclose(output);
        if (!temporary.empty()) {
            ::unlink(temporary.c_str());
        }
    }

    Result<void> bytes(void const* source, size_t size) {
        if (size != 0) {
            if (std::fwrite(source, 1, size, output) != size) {
                return Err(Error::resource("cannot write reference database file"));
            }
            checksum = crc32_z(checksum, static_cast<Bytef const*>(source), size);
        }
        return Ok();
    }

    template <typename T>
    Result<void> value(T item) {
        item = database_file_little_endian(item);
        return bytes(&item, sizeof(item));
    }

    Result<void> words(std::vector<uint32_t> const& items) {
        if constexpr (std::endian::native == std::endian::little) {
            return bytes(items.data(), items.size() * sizeof(uint32_t));
        } else {
            for (auto item : items) {
                CUDDL_TRY(value(item));
            }
            return Ok();
        }
    }
};

template <typename IO>
Result<void> database_file_metadata(IO& io, reference_database_metadata& metadata) {
    auto& c = metadata.compatibility;
    CUDDL_TRY(io.value(c.kmer_length));
    CUDDL_TRY(io.value(c.bucket_count));
    CUDDL_TRY(io.value(c.indexed_bucket_count));
    CUDDL_TRY(io.value(c.score_encoder_identity));
    CUDDL_TRY(io.value(c.exponent_bits));
    CUDDL_TRY(io.value(c.mantissa_bits));
    CUDDL_TRY(io.value(c.hash_identity));
    CUDDL_TRY(io.value(c.hash_seed));
    CUDDL_TRY(io.value(c.canonicalisation_policy));
    CUDDL_TRY(io.value(c.blacklist_identity));
    CUDDL_TRY(io.value(c.blacklist_version));
    CUDDL_TRY(io.value(c.key_mask));
    CUDDL_TRY(io.value(metadata.reference_count));
    return Ok();
}

// Page-locked sequence storage handed to loader threads. A lease returns to the pool when the
// parsed file holding it dies, which is after the caller enqueued every copy that reads it.
// Releasing records that point on the stream; the next loader to take the slot waits for it
// there, so a buffer is never rewritten under a DMA and the build never has to drain.
class pinned_sequence_pool {
   public:
    /// @p limit bounds one buffer; larger genomes stay on the loader's own growing buffer.
    pinned_sequence_pool(cuda::stream_ref stream, size_t buffers, size_t limit)
        : stream_(stream), limit_(limit) {
        slots_.reserve(buffers);
        in_use_.reserve(buffers);
        consumed_.reserve(buffers);
    }

    /// @brief Number of buffers the pool may grow to.
    void set_capacity(size_t buffers) noexcept {
        capacity_ = std::max<size_t>(1, buffers);
    }

    /// @brief Buffers allocated so far. Read once the loaders have stopped.
    [[nodiscard]] size_t buffers() const noexcept {
        return slots_.size();
    }

    /// @brief Returns page-locked bytes, or a null target when the pool cannot serve.
    [[nodiscard]] decompression_target acquire(size_t bytes) {
        if (bytes == 0 || bytes > limit_) return {};
        size_t index = 0;
        {
            std::lock_guard lock(mutex_);
            index = claim(bytes);
            if (index == slots_.size()) return {};
        }
        // Wait on the loader thread rather than the build loop: the slot is already reserved,
        // so this blocks only the genome that needs it. A failing wait throws out of the loader,
        // which reports it through the pool's error slot like any other load failure.
        consumed_[index].sync();
        return lease(index);
    }

   private:
    /// @brief Reserves a free slot holding at least @p bytes, or returns slots_.size().
    [[nodiscard]] size_t claim(size_t bytes) {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (in_use_[i]) continue;
            if (slots_[i]->buffer.size() < bytes) {
                // Grow a free buffer instead of falling back. Genome sizes vary within a
                // corpus, and a buffer sized for the first small genome would otherwise
                // never serve the larger ones. Rounding up to a power of two bounds how many
                // times any buffer is reallocated, and page-locked allocation is not cheap.
                auto grown = size_t{1} << 16;
                while (grown < bytes) grown *= 2;
                slots_[i] = std::make_unique<slot>(
                    cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>(
                        stream_, cuda::pinned_default_memory_pool(), grown, cuda::no_init
                    )
                );
            }
            in_use_[i] = true;
            return i;
        }
        if (slots_.size() >= capacity_) return slots_.size();
        slots_.push_back(
            std::make_unique<slot>(
                cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible>(
                    stream_, cuda::pinned_default_memory_pool(), bytes, cuda::no_init
                )
            )
        );
        in_use_.push_back(true);
        consumed_.emplace_back(stream_);
        return slots_.size() - 1;
    }

   private:
    struct slot {
        cuda::buffer<char, cuda::mr::host_accessible, cuda::mr::device_accessible> buffer;
    };

    [[nodiscard]] decompression_target lease(size_t index) {
        auto* pool = this;
        auto* owner = slots_[index].get();
        return {
            owner->buffer.data(),
            owner->buffer.size(),
            std::shared_ptr<void>(owner, [pool, index](void*) {
                std::lock_guard lock(pool->mutex_);
                // Record before publishing the slot: a loader that takes it must see the
                // event of every copy the previous lease fed.
                pool->consumed_[index].record(pool->stream_);
                pool->in_use_[index] = false;
            })
        };
    }

    cuda::stream_ref stream_;
    size_t limit_;
    size_t capacity_{1};
    std::vector<std::unique_ptr<slot>> slots_;
    std::vector<bool> in_use_;
    std::vector<cuda::event> consumed_;
    std::mutex mutex_;
};

/// @brief `decompression_source` trampoline for `pinned_sequence_pool`.
[[nodiscard]] inline decompression_target acquire_pinned_target(void* context, size_t bytes) {
    return static_cast<pinned_sequence_pool*>(context)->acquire(bytes);
}

}  // namespace cuddl::detail

namespace cuddl {

/// @brief One record of a genome the caller already holds: bases only.
///
/// No header, no line breaks, no FASTQ qualities: the shape a parsed record has, which is what
/// the encoder applies its window rules to. k-mers never cross two records.
struct sequence_record {
    std::string_view bases;
};

/// @brief One genome as its records, in order. No records keeps the genome's ID with a zero row.
struct sequence_genome {
    std::span<sequence_record const> records;
    std::string_view name;  // copied into the labels; may be empty
};

/// @brief Knobs for a build that stages bases the caller already holds.
struct reference_staging_options {
    reference_build_statistics* statistics = nullptr;
    size_t staging_bytes = 0;  // 0 sizes the arena from free device memory
};

/**
 * @brief Host-owned reference sketches and labels, ready for binary storage or GPU upload.
 *
 * One input file is one genome, including all of its FASTA/FASTQ records. Input order defines
 * stable reference IDs; labels preserve the supplied paths. Files store packed winner/count
 * rows, saturation flags and compatibility metadata. GPU indexes are rebuilt on upload.
 * Build and upload are synchronous. The supplied stream must outlive the uploaded database.
 */
class reference_database_file {
   public:
    [[nodiscard]] reference_database_metadata metadata() const noexcept {
        return metadata_;
    }
    [[nodiscard]] std::span<std::string const> names() const noexcept {
        return names_;
    }
    /// @brief Packed winner/count registers in reference-ID order, one bucket row per genome.
    [[nodiscard]] std::span<uint32_t const> rows() const noexcept {
        return rows_;
    }
    /// @brief Saturation flags in reference-ID order.
    [[nodiscard]] std::span<uint32_t const> saturation() const noexcept {
        return saturation_;
    }

    /**
     * @brief Builds one sketch per plain or gzip/BGZF FASTA/FASTQ file on the GPU.
     *
     * K-mers never cross record boundaries or ambiguous bases. FASTQ qualities are validated
     * and ignored. Loader threads compact and stage records into a device arena sized from free
     * device memory, and one batch kernel encodes every record in it, so the device is fed in
     * large batches rather than slice by slice. Register rows stay on the device for a group of
     * genomes and return to the host once, on the single drain at the end of the build.
     * @p staging_bytes caps the arena explicitly; zero sizes it from free device memory.
     * Zero workers selects up to eight. Use one worker to load only one genome at a time.
     * Empty genomes retain their IDs and have zero rows.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    /// @p pinned transfers decompressed bytes straight from page-locked memory. Disabling it
    /// restores the heap buffer plus a staging copy, which some hosts do better; the
    /// `direct_bytes` and `staged_bytes` counters say which path a build actually took.
    [[nodiscard]] static Result<reference_database_file> build(
        std::span<std::filesystem::path const> paths,
        cuda::stream_ref stream,
        unsigned parser_workers = 0,
        reference_build_statistics* statistics = nullptr,
        bool pinned = default_pinned_transfer,
        size_t staging_bytes = 0
    ) try {
        // NVCC 13.3 crashes on CUDDL_TRY directly inside a try block. Keep its GNU statement
        // expressions in a separate lambda scope, outside the exception-catching function.
        return [&]() -> Result<reference_database_file> {
            using database_type = reference_database<K, BucketCount, Layout>;
            if (paths.size() > std::numeric_limits<uint32_t>::max() / BucketCount ||
                paths.size() >
                    std::numeric_limits<size_t>::max() / (BucketCount * sizeof(uint32_t))) {
                return Err(Error::resource("reference collection exceeds database index capacity"));
            }
            reference_database_file result;
            result.metadata_ = {
                score_compatibility::current<K, BucketCount, Layout>(),
                static_cast<uint32_t>(paths.size())
            };
            result.rows_.resize(paths.size() * BucketCount);
            result.saturation_.resize(paths.size());
            result.names_.reserve(paths.size());
            // No inputs means no staging budget to resolve and nothing to encode.
            if (paths.empty()) return result;
            // Larger genomes keep the loader's own buffer rather than pinning that much RAM
            // per loader for the whole run.
            constexpr size_t pinned_limit = size_t{32} << 20;
            auto const hardware_threads = std::max(1U, std::thread::hardware_concurrency());
            auto const workers = std::max(
                size_t{1},
                std::min(
                    paths.size(),
                    size_t{std::min(parser_workers ? parser_workers : 8U, hardware_threads)}
                )
            );
            // Page-locked sequence buffers, one per loader, so the transfer engine reads the
            // decompressed genome in place instead of restaging it on the consumer thread.
            detail::pinned_sequence_pool pinned_buffers(stream, workers, pinned_limit);
            // One buffer per in-flight file, plus headroom: a worker asks for its next file
            // while every loaded file still holds a lease, so an exact match would refuse.
            pinned_buffers.set_capacity(workers + 2);
            auto* pinned_pool = pinned ? &pinned_buffers : nullptr;
            detail::decompression_source const load_source =
                pinned_pool != nullptr
                    ? detail::decompression_source{detail::acquire_pinned_target, pinned_pool}
                    : detail::decompression_source{};
            std::optional<detail::fastx_load_pool> loader;
            if (workers > 1) {
                // Loaders run a few files ahead of the consumer, which takes results in order.
                // A window of one worker-worth leaves the consumer waiting on a straggler while
                // every other loader sits idle.
                loader.emplace(paths, workers, load_source, workers * 4);
            }
            // Exact staged bytes the collection can fill: the decompressed size of every input.
            // A bound from compressed sizes alone overestimates a corpus several times over, and
            // an arena past what the corpus can hold is memory the device never needs.
            uint64_t staged_ceiling = 0;
            for (auto const& path : paths) {
                auto const decompressed = detail::gzip_decompressed_size(path.string());
                if (decompressed != 0) {
                    staged_ceiling += decompressed;
                    continue;
                }
                std::error_code error;
                auto const size = std::filesystem::file_size(path, error);
                if (!error) staged_ceiling += size;
            }
            auto const plan = CUDDL_TRY((detail::plan_staging<K, BucketCount>(
                paths.size(), staged_ceiling, staging_bytes, stream
            )));
            detail::database_stager<K, BucketCount, Layout> stager(
                stream, paths.size(), plan, statistics
            );
            size_t base = 0;
            while (base < paths.size()) {
                auto const count = std::min(plan.group, paths.size() - base);
                CUDDL_TRY(stager.begin_group(count));
                for (size_t id = base; id < base + count; ++id) {
                    auto sequence = CUDDL_TRY(
                        workers == 1
                            ? detail::load_fastx_sequence_file(paths[id].string(), load_source)
                            : loader->take(id)
                    );
                    // Read the loaded file's parts before the move below: argument order is
                    // unspecified, so a moved-from Result must not be dereferenced.
                    auto const* const pinned_base = sequence->decompressed_target;
                    auto const pinned_size = sequence->decompressed_size;
                    auto const& extents = sequence->extents;
                    CUDDL_TRY(stager.add_genome(
                        id - base, extents, pinned_base, pinned_size, std::move(sequence)
                    ));
                    result.names_.push_back(paths[id].string());
                }
                CUDDL_TRY(stager.end_group(base, count));
                base += count;
            }
            CUDDL_TRY(stager.finish(result.rows_, result.saturation_));
            // Loaders still hold leases; join them before reading pool state.
            loader.reset();
            if (statistics != nullptr) {
                statistics->pinned_buffers = pinned_buffers.buffers();
            }
            // Instantiate the same constraints as the destination GPU database.
            static_assert(sizeof(database_type) > 0);
            return result;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /**
     * @brief Builds one sketch per genome from bases the caller already holds.
     *
     * Each genome supplies its records as spans of consecutive bases with the line breaks
     * already removed, the shape a parsed FASTX record has. The caller keeps every byte alive
     * until this returns, and the build is synchronous like @ref build. Labels come from the
     * genomes' names, and reference IDs follow the order of @p genomes.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static Result<reference_database_file> build_from_sequences(
        std::span<sequence_genome const> genomes,
        cuda::stream_ref stream,
        reference_staging_options options = {}
    ) try {
        return [&]() -> Result<reference_database_file> {
            using database_type = reference_database<K, BucketCount, Layout>;
            if (genomes.size() > std::numeric_limits<uint32_t>::max() / BucketCount ||
                genomes.size() >
                    std::numeric_limits<size_t>::max() / (BucketCount * sizeof(uint32_t))) {
                return Err(Error::resource("reference collection exceeds database index capacity"));
            }
            reference_database_file result;
            result.metadata_ = {
                score_compatibility::current<K, BucketCount, Layout>(),
                static_cast<uint32_t>(genomes.size())
            };
            result.rows_.resize(genomes.size() * BucketCount);
            result.saturation_.resize(genomes.size());
            result.names_.reserve(genomes.size());
            if (genomes.empty()) return result;
            // The caller knows exactly what the corpus holds, so the arena ceiling is the sum of
            // the bases rather than an estimate read out of file headers.
            uint64_t staged_ceiling = 0;
            for (auto const& genome : genomes) {
                for (auto const& record : genome.records) staged_ceiling += record.bases.size();
            }
            auto const plan = CUDDL_TRY((detail::plan_staging<K, BucketCount>(
                genomes.size(), staged_ceiling, options.staging_bytes, stream
            )));
            detail::database_stager<K, BucketCount, Layout> stager(
                stream, genomes.size(), plan, options.statistics
            );
            std::vector<detail::fastx_sequence_extent> records;
            size_t base = 0;
            while (base < genomes.size()) {
                auto const count = std::min(plan.group, genomes.size() - base);
                CUDDL_TRY(stager.begin_group(count));
                for (size_t index = base; index < base + count; ++index) {
                    auto const& genome = genomes[index];
                    records.clear();
                    records.reserve(genome.records.size());
                    for (auto const& record : genome.records) {
                        if (record.bases.empty()) continue;
                        records.push_back(
                            {record.bases.data(), record.bases.data() + record.bases.size()}
                        );
                    }
                    CUDDL_TRY(stager.add_genome(index - base, records));
                    result.names_.push_back(std::string(genome.name));
                }
                CUDDL_TRY(stager.end_group(base, count));
                base += count;
            }
            CUDDL_TRY(stager.finish(result.rows_, result.saturation_));
            static_assert(sizeof(database_type) > 0);
            return result;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Uploads packed rows and builds a sparse index by default; waits for completion.
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] Result<reference_database<K, BucketCount, Layout>>
    upload(cuda::stream_ref stream, index_storage storage = index_storage::sparse) const {
        CUDDL_TRY((detail::validate_indexed_score_compatibility<K, BucketCount, Layout>(
            metadata_.compatibility
        )));
        auto rows =
            CUDDL_CUDA_TRY(cuda::make_device_buffer<uint32_t>(stream, stream.device(), rows_));
        auto saturation = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), saturation_)
        );
        auto database = CUDDL_TRY((reference_database<K, BucketCount, Layout>::build_indexed_async(
            {rows.data(), rows.size()},
            {saturation.data(), saturation.size()},
            metadata_.compatibility,
            stream,
            storage
        )));
        CUDDL_CUDA_TRY(stream.sync());
        return std::move(database);
    }

    /**
     * @brief Writes a versioned, little-endian, CRC32-checked binary file.
     *
     * Writes to an exclusive temporary file in the destination directory, closes it, then
     * atomically replaces the destination. Failure leaves an existing destination intact.
     * This guarantees atomic publication, not durability across power loss.
     */
    [[nodiscard]] Result<void> save(std::filesystem::path const& path) const try {
        return [&]() -> Result<void> {
            detail::database_file_writer writer;
            writer.temporary = path.string() + ".tmp.XXXXXX";
            auto fd = ::mkstemp(writer.temporary.data());
            if (fd < 0) {
                writer.temporary.clear();
                return Err(Error::resource("cannot create database file beside: " + path.string()));
            }
            writer.output = ::fdopen(fd, "wb");
            if (!writer.output) {
                ::close(fd);
                return Err(Error::resource("cannot open temporary database file"));
            }
            CUDDL_TRY(writer.bytes("CUDDLDB\0", 8));
            CUDDL_TRY(writer.value(uint32_t{1}));
            auto metadata = metadata_;
            CUDDL_TRY(detail::database_file_metadata(writer, metadata));
            for (auto const& name : names_) {
                if (name.size() > std::numeric_limits<uint32_t>::max()) {
                    return Err(Error::resource("reference label exceeds binary format capacity"));
                }
                CUDDL_TRY(writer.value(static_cast<uint32_t>(name.size())));
                CUDDL_TRY(writer.bytes(name.data(), name.size()));
            }
            CUDDL_TRY(writer.words(rows_));
            CUDDL_TRY(writer.words(saturation_));
            CUDDL_TRY(writer.value(static_cast<uint32_t>(writer.checksum)));
            auto closed = std::fclose(std::exchange(writer.output, nullptr));
            if (closed != 0) return Err(Error::resource("cannot close database file"));
            std::filesystem::rename(writer.temporary, path);
            writer.temporary.clear();
            return Ok();
        }();
    } catch (std::filesystem::filesystem_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Reads and validates a binary database without requiring a GPU.
    [[nodiscard]] static Result<reference_database_file> load(
        std::filesystem::path const& path
    ) try {
        return [&]() -> Result<reference_database_file> {
            detail::database_file_reader reader{
                std::ifstream(path, std::ios::binary | std::ios::ate)
            };
            auto size = reader.input.tellg();
            if (size < 0) {
                return Err(Error::resource("cannot open database file: " + path.string()));
            }
            reader.remaining = static_cast<uint64_t>(size);
            reader.input.seekg(0);
            char magic[8];
            CUDDL_TRY(reader.bytes(magic, sizeof(magic)));
            if (std::string_view(magic, sizeof(magic)) != std::string_view("CUDDLDB\0", 8)) {
                return Err(Error::invalid_argument("not a cuDDL reference database file"));
            }
            uint32_t version{};
            CUDDL_TRY(reader.value(version));
            if (version != 1) {
                return Err(Error::invalid_argument("unsupported database file version"));
            }
            reference_database_file result;
            CUDDL_TRY(detail::database_file_metadata(reader, result.metadata_));
            auto const& c = result.metadata_.compatibility;
            if (c.kmer_length < 1 || c.kmer_length > 31 || c.bucket_count < 2048 ||
                c.bucket_count > 131072 || !std::has_single_bit(c.bucket_count) ||
                c.indexed_bucket_count != c.bucket_count || c.key_mask != 0xffffU ||
                c.exponent_bits == 0 || c.mantissa_bits == 0 ||
                c.exponent_bits + c.mantissa_bits != 16 || c.score_encoder_identity != 1 ||
                c.hash_identity != 1 || c.hash_seed != detail::seed ||
                c.canonicalisation_policy != 1 || c.blacklist_identity != 0 ||
                c.blacklist_version != 0) {
                return Err(Error::invalid_argument("unsupported database construction metadata"));
            }
            uint64_t const count = result.metadata_.reference_count;
            uint64_t const words = count * c.bucket_count;
            uint64_t const payload_bytes = (words + count) * sizeof(uint32_t);
            if (words > std::numeric_limits<uint32_t>::max() ||
                words > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
                payload_bytes + count * sizeof(uint32_t) + sizeof(uint32_t) > reader.remaining) {
                return Err(Error::invalid_argument("invalid reference database extents"));
            }
            result.names_.reserve(count);
            for (uint64_t id = 0; id < count; ++id) {
                uint32_t length{};
                CUDDL_TRY(reader.value(length));
                auto const required =
                    payload_bytes + (count - id - 1) * sizeof(uint32_t) + sizeof(uint32_t);
                if (required > reader.remaining || length > reader.remaining - required) {
                    return Err(Error::invalid_argument("invalid reference label extent"));
                }
                std::string name(length, '\0');
                CUDDL_TRY(reader.bytes(name.data(), name.size()));
                result.names_.push_back(std::move(name));
            }
            if (reader.remaining != payload_bytes + sizeof(uint32_t)) {
                return Err(Error::invalid_argument("unexpected reference database payload size"));
            }
            result.rows_.resize(words);
            result.saturation_.resize(count);
            CUDDL_TRY(reader.words(result.rows_));
            CUDDL_TRY(reader.words(result.saturation_));
            auto const expected_checksum = static_cast<uint32_t>(reader.checksum);
            uint32_t checksum{};
            CUDDL_TRY(reader.value(checksum));
            if (checksum != expected_checksum) {
                return Err(Error::invalid_argument("database checksum mismatch"));
            }
            for (auto state : result.saturation_) {
                if (state > 1) {
                    return Err(Error::invalid_argument("invalid database saturation flag"));
                }
            }
            for (auto row : result.rows_) {
                if ((detail::winner(row) == 0) != (detail::count(row) == 0)) {
                    return Err(Error::invalid_argument("invalid packed database register"));
                }
            }
            return result;
        }();
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

   private:
    reference_database_file() = default;
    reference_database_metadata metadata_{};
    std::vector<std::string> names_;
    std::vector<uint32_t> rows_;
    std::vector<uint32_t> saturation_;
};

}  // namespace cuddl
