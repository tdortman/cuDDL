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

#include <cuddl/detail/fastx_device_builder.cuh>
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

// Fixed worker pool loading FASTX files ahead of the GPU loop. A pthread spawn and join
// per genome costs tens of microseconds and dominates at high file counts. Workers pull
// ids in order while the main thread takes results in order, so at most depth loads are
// in flight and output order never depends on completion order. Loader exceptions are
// rethrown by take, matching std::async propagation into the build error handlers.
class fastx_load_pool {
   public:
    fastx_load_pool(std::span<std::filesystem::path const> paths, size_t depth)
        : paths_(paths), depth_(std::max(size_t{1}, depth)), results_(paths.size()),
          errors_(paths.size()) {
        workers_.reserve(depth_);
        for (size_t i = 0; i < depth_; ++i) workers_.emplace_back([this] { work(); });
    }
    ~fastx_load_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        assign_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    fastx_load_pool(fastx_load_pool const&) = delete;
    fastx_load_pool& operator=(fastx_load_pool const&) = delete;

    [[nodiscard]] Result<std::unique_ptr<fastx_sequence_file>> take(size_t id) {
        std::unique_lock lock(mutex_);
        filled_.wait(lock, [&] { return results_[id].has_value() || errors_[id] != nullptr; });
        ++taken_;
        assign_.notify_all();
        lock.unlock();
        if (errors_[id] != nullptr) std::rethrow_exception(errors_[id]);
        return std::move(*results_[id]);
    }

   private:
    void work() {
        while (true) {
            size_t id;
            {
                std::unique_lock lock(mutex_);
                assign_.wait(lock, [&] {
                    return stop_ || next_ >= paths_.size() || next_ - taken_ < depth_;
                });
                if (stop_ || next_ >= paths_.size()) return;
                if (next_ - taken_ >= depth_) continue;
                id = next_++;
            }
            try {
                auto loaded = load_fastx_sequence_file(paths_[id].string());
                std::lock_guard lock(mutex_);
                results_[id] = std::move(loaded);
            } catch (...) {
                std::lock_guard lock(mutex_);
                errors_[id] = std::current_exception();
            }
            filled_.notify_all();
        }
    }
    std::span<std::filesystem::path const> paths_;
    size_t depth_;
    std::vector<std::optional<Result<std::unique_ptr<fastx_sequence_file>>>> results_;
    std::vector<std::exception_ptr> errors_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable assign_, filled_;
    size_t next_ = 0, taken_ = 0;
    bool stop_ = false;
};

}  // namespace cuddl::detail

namespace cuddl {

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

    /**
     * @brief Builds one sketch per plain or gzip/BGZF FASTA/FASTQ file on the GPU.
     *
     * K-mers never cross record boundaries or ambiguous bases. FASTQ qualities are validated
     * and ignored. A bounded queue loads sequences concurrently; GPU tiles encode bases
     * in shared memory and update sketches without materializing a k-mer array. Zero workers
     * selects up to eight. Use one worker to load only one genome at a time. Empty genomes retain
     * their IDs and have zero rows.
     */
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static Result<reference_database_file> build(
        std::span<std::filesystem::path const> paths,
        cuda::stream_ref stream,
        unsigned parser_workers = 0
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
            auto const hardware_threads = std::max(1U, std::thread::hardware_concurrency());
            auto const workers = std::max(
                size_t{1},
                std::min(
                    paths.size(),
                    size_t{std::min(parser_workers ? parser_workers : 8U, hardware_threads)}
                )
            );
            std::optional<detail::fastx_load_pool> loader;
            if (workers > 1) loader.emplace(paths, workers);
            auto registers = CUDDL_CUDA_TRY(
                cuda::make_device_buffer<uint32_t>(
                    stream, stream.device(), BucketCount + 1, cuda::no_init
                )
            );
            detail::fastx_device_builder packer(stream);
            CUDDL_TRY(packer.prepare(stream));
            std::string short_records;
            for (size_t id = 0; id < paths.size(); ++id) {
                auto sequence = CUDDL_TRY(
                    workers == 1 ? detail::load_fastx_sequence_file(paths[id].string())
                                 : loader->take(id)
                );
                CUDDL_CUDA_TRY(cuda::fill_bytes(stream, registers, 0));
                short_records.clear();
                auto add_sequence = [&](std::string_view record) -> Result<void> {
                    packer.reset();
                    for (size_t offset = 0; offset < record.size(); offset += packer.capacity) {
                        CUDDL_TRY((packer.add<BucketCount, Layout>(
                            record.substr(offset, packer.capacity),
                            K,
                            registers.data(),
                            registers.data()[BucketCount],
                            stream
                        )));
                    }
                    return Ok();
                };
                for (auto const& extent : sequence->extents) {
                    std::string_view const record{extent.begin, extent.end};
                    if (record.size() >= packer.capacity ||
                        short_records.size() + record.size() + 1 > packer.capacity) {
                        CUDDL_TRY(add_sequence(short_records));
                        short_records.clear();
                    }
                    if (record.size() >= packer.capacity) {
                        CUDDL_TRY(add_sequence(record));
                    } else {
                        // Batch short contigs/reads to avoid a GPU launch per record.
                        short_records.reserve(packer.capacity);
                        short_records.append(record);
                        short_records.push_back('N');
                    }
                }
                CUDDL_TRY(add_sequence(short_records));
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        device_span<uint32_t const>{registers.data(), BucketCount},
                        cuda::std::span{result.rows_.data() + id * BucketCount, BucketCount}
                    )
                );
                CUDDL_CUDA_TRY(
                    cuda::copy_bytes(
                        stream,
                        device_span<uint32_t const>{registers.data() + BucketCount, 1},
                        cuda::std::span{result.saturation_.data() + id, size_t{1}}
                    )
                );
                CUDDL_CUDA_TRY(stream.sync());
                result.names_.push_back(paths[id].string());
                sequence.reset();
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
