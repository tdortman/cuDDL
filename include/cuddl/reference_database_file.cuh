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

}  // namespace cuddl::detail

namespace cuddl {

/// @brief One record of a genome the caller already holds: bases only.
using sequence_record = detail::sequence_record;

/// @brief One genome as its records, in order. No records keeps the genome's ID with a zero row.
using sequence_genome = detail::sequence_genome;

/// @brief Loaders a path build runs unless the caller asks for another count.
inline constexpr unsigned default_parser_workers = 8;

/// @brief Knobs for a build the caller feeds with FASTA/FASTQ paths.
struct path_build_options {
    reference_build_statistics* statistics = nullptr;
    /// Loaders to run. The build runs at most one per input, at most the hardware, and never
    /// none, so a larger count is only a ceiling. One loader loads one genome at a time.
    unsigned parser_workers = default_parser_workers;
    /// Arena bytes. Unset sizes the arena from free device memory and what the inputs can fill,
    /// which is the only case a build cannot know in advance.
    std::optional<size_t> staging_bytes;
    /// Transfers decompressed bytes straight from page-locked memory. On a coherent CPU/GPU
    /// system, Grace Hopper and Grace Blackwell among them, the device reads host memory anyway
    /// and page-locking only costs host writes, so the default turns it off there and the heap
    /// buffer plus a staging copy wins; the `direct_bytes` and `staged_bytes` counters say which
    /// path a build actually took.
    bool pinned = default_pinned_transfer;
};

/// @brief Knobs for a build the caller feeds with bases it already holds.
struct sequence_build_options {
    reference_build_statistics* statistics = nullptr;
    /// Arena bytes. Unset sizes the arena from free device memory and what the bases can fill.
    std::optional<size_t> staging_bytes;
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
     * and ignored. Loader threads compact and stage records into a device arena, and one batch
     * kernel encodes every record in it, so the device is fed in large batches rather than slice
     * by slice. Register rows stay on the device for a group of genomes and return to the host
     * once per group. Empty genomes retain their IDs and have zero rows.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static Result<reference_database_file> build(
        std::span<std::filesystem::path const> paths,
        cuda::stream_ref stream,
        path_build_options options = {}
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
            auto store = CUDDL_TRY((detail::stage_paths<K, BucketCount, Layout>(
                paths,
                stream,
                options.staging_bytes,
                options.parser_workers,
                options.pinned,
                options.statistics
            )));
            detail::unpack_store<BucketCount>(store, result.rows_, result.saturation_);
            for (auto const& path : paths) result.names_.push_back(path.string());
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
        sequence_build_options options = {}
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
            auto store = CUDDL_TRY((detail::stage_sequences<K, BucketCount, Layout>(
                genomes, stream, options.staging_bytes, options.statistics
            )));
            detail::unpack_store<BucketCount>(store, result.rows_, result.saturation_);
            for (auto const& genome : genomes) result.names_.push_back(std::string(genome.name));
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
     * @brief Adopts a device store as the host rows this form saves.
     *
     * A store holds `BucketCount` packed registers followed by one saturation word per reference,
     * the layout a single sketch allocation has, so a reference's registers and its flag arrive in
     * one copy. A staged build and the streamed tile builder both write that layout; a device
     * database that keeps registers and flags in separate buffers has to hand over a store.
     *
     * @p names labels one reference each in store order, or is empty for a database without
     * labels. Registers and flags are validated as @ref load validates them.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static Result<reference_database_file> from_store(
        device_span<uint32_t const> store,
        std::span<std::string const> names,
        cuda::stream_ref stream
    ) try {
        return [&]() -> Result<reference_database_file> {
            if (store.size() % (BucketCount + 1) != 0) {
                return Err(Error::invalid_argument("a store must contain whole stored sketches"));
            }
            size_t const count = store.size() / (BucketCount + 1);
            if (!names.empty() && names.size() != count) {
                return Err(Error::invalid_argument("labels must be empty or name every reference"));
            }
            if (count != 0 && store.data() == nullptr) {
                return Err(Error::invalid_argument("a nonempty store must not be null"));
            }
            reference_database_file result;
            result.metadata_ = {
                score_compatibility::current<K, BucketCount, Layout>(), static_cast<uint32_t>(count)
            };
            result.rows_.resize(count * BucketCount);
            result.saturation_.resize(count);
            if (count == 0) return result;
            std::vector<uint32_t> host_store(store.size());
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream, store, cuda::std::span{host_store.data(), host_store.size()}
                )
            );
            CUDDL_CUDA_TRY(stream.sync());
            detail::unpack_store<BucketCount>(host_store, result.rows_, result.saturation_);
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
            result.names_.assign(names.begin(), names.end());
            return result;
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
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
