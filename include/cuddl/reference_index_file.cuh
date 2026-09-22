#pragma once

#include <cuddl/detail/hash.cuh>
#include <cuddl/reference_database_file.cuh>
#include <cuddl/reference_index.cuh>

namespace cuddl::detail {

struct reference_index_digest {
    uint64_t digest = 0;

    Result<void> bytes(void const* data, size_t size) {
        digest = xxhash64(static_cast<uint8_t const*>(data), size, digest);
        return Ok();
    }

    template <typename T>
    Result<void> value(T item) {
        item = database_file_little_endian(item);
        return bytes(&item, sizeof(item));
    }

    Result<void> words(std::span<uint32_t const> items) {
        if constexpr (std::endian::native == std::endian::little) {
            return bytes(items.data(), items.size_bytes());
        } else {
            std::vector<uint32_t> little_endian(items.begin(), items.end());
            for (auto& item : little_endian) {
                item = database_file_little_endian(item);
            }
            return bytes(little_endian.data(), items.size_bytes());
        }
    }
};

inline Result<uint64_t> reference_database_digest(reference_database_file const& db) {
    reference_index_digest hash;
    CUDDL_TRY(hash.bytes("CUDDLDB-XXH64-v1", sizeof("CUDDLDB-XXH64-v1") - 1));
    auto metadata = db.metadata();
    CUDDL_TRY(database_file_metadata(hash, metadata));
    CUDDL_TRY(hash.value(static_cast<uint64_t>(db.names().size())));
    for (auto const& name : db.names()) {
        CUDDL_TRY(hash.value(static_cast<uint64_t>(name.size())));
        CUDDL_TRY(hash.bytes(name.data(), name.size()));
    }
    CUDDL_TRY(hash.words(db.rows()));
    CUDDL_TRY(hash.words(db.saturation()));
    return hash.digest;
}

}  // namespace cuddl::detail

namespace cuddl {

/// @brief Index-file persistence and the validation boundary for reference_index.
class reference_index_file {
   public:
    /// @brief Atomically save an already built or loaded index.
    template <uint32_t K, size_t BucketCount, typename Layout>
    [[nodiscard]] static Result<void> save(
        reference_index<K, BucketCount, Layout> const& index,
        reference_database<K, BucketCount, Layout> const& source,
        std::filesystem::path const& path,
        cuda::stream_ref stream
    ) try {
        return [&]() -> Result<void> {
            CUDDL_TRY(source.view(&index));
            auto database = CUDDL_TRY(snapshot(source, stream));
            auto storage = index.storage();
            auto offsets = CUDDL_TRY(
                download(index.index_offsets_.data(), index.index_offsets_.size(), stream)
            );
            auto posting_count =
                storage == index_storage::dense ? offsets.back() : index.index_posting_capacity_;
            auto postings =
                CUDDL_TRY(download(index.index_postings_.data(), posting_count, stream));
            auto keys =
                CUDDL_TRY(download(index.index_keys_.data(), index.index_keys_.size(), stream));
            auto digest = CUDDL_TRY(detail::reference_database_digest(database));
            detail::database_file_writer writer;
            writer.temporary = path.string() + ".tmp.XXXXXX";
            auto fd = ::mkstemp(writer.temporary.data());
            if (fd < 0) {
                return Err(Error::resource("cannot create index file beside: " + path.string()));
            }
            writer.output = ::fdopen(fd, "wb");
            if (!writer.output) {
                ::close(fd);
                return Err(Error::resource("cannot open temporary index file"));
            }
            CUDDL_TRY(writer.bytes("CUDDLIX\0", 8));
            CUDDL_TRY(writer.value(uint32_t{1}));
            CUDDL_TRY(writer.value(storage == index_storage::dense ? uint32_t{0} : uint32_t{1}));
            CUDDL_TRY(writer.value(digest));
            CUDDL_TRY(writer.value(static_cast<uint64_t>(postings.size())));
            CUDDL_TRY(writer.words(offsets));
            CUDDL_TRY(writer.words(postings));
            if constexpr (std::endian::native == std::endian::little) {
                CUDDL_TRY(writer.bytes(keys.data(), keys.size() * sizeof(uint16_t)));
            } else {
                for (auto key : keys) {
                    CUDDL_TRY(writer.value(key));
                }
            }
            CUDDL_TRY(writer.value(static_cast<uint32_t>(writer.checksum)));
            if (std::fclose(std::exchange(writer.output, nullptr)) != 0) {
                return Err(Error::resource("cannot close index file"));
            }
            std::filesystem::rename(writer.temporary, path);
            writer.temporary.clear();
            return Ok();
        }();
    } catch (cuda::cuda_error const& error) {
        return Err(Error::cuda(static_cast<cudaError_t>(error.status())));
    } catch (std::system_error const& error) {
        return Err(Error::resource(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    /// @brief Validate the file and its database binding once, then restore a ready-to-search
    /// index.
    template <uint32_t K, size_t BucketCount, typename Layout = default_register_layout>
    [[nodiscard]] static Result<reference_index<K, BucketCount, Layout>> load(
        std::filesystem::path const& path,
        reference_database<K, BucketCount, Layout> const& database,
        cuda::stream_ref stream
    ) {
        auto source = CUDDL_TRY(snapshot(database, stream));
        auto decoded = CUDDL_TRY(read(path, std::move(source)));
        auto index = CUDDL_CUDA_TRY((reference_index<K, BucketCount, Layout>(database, stream)));
        index.index_offsets_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(stream, stream.device(), decoded.offsets_)
        );
        index.index_keys_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint16_t>(stream, stream.device(), decoded.keys_)
        );
        index.index_posting_capacity_ = static_cast<size_t>(detail::indexed_posting_count(
            database.reference_count(), database.metadata().compatibility
        ));
        index.index_postings_ = CUDDL_CUDA_TRY(
            cuda::make_device_buffer<uint32_t>(
                stream, stream.device(), index.index_posting_capacity_, cuda::no_init
            )
        );
        if (!decoded.postings_.empty()) {
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream,
                    cuda::std::span{decoded.postings_.data(), decoded.postings_.size()},
                    cuda::std::span{index.index_postings_.data(), decoded.postings_.size()}
                )
            );
        }
        index.indexed_ = true;
        CUDDL_CUDA_TRY(stream.sync());
        return index;
    }

   private:
    template <uint32_t K, size_t BucketCount, typename Layout>
    static Result<reference_database_file>
    snapshot(reference_database<K, BucketCount, Layout> const& source, cuda::stream_ref stream) {
        if (!source.preserves_multiplicity()) {
            return Err(Error::invalid_argument("index files require packed reference rows"));
        }
        CUDDL_TRY((detail::validate_non_indexed_score_compatibility<K, BucketCount, Layout>(
            source.metadata().compatibility
        )));
        reference_database_file database;
        database.metadata_ = source.metadata();
        database.names_.assign(source.names().begin(), source.names().end());
        if (database.names_.size() != source.reference_count()) {
            return Err(Error::invalid_argument("index files require named reference rows"));
        }
        database.rows_ =
            CUDDL_TRY(download(source.packed_data().data(), source.packed_data().size(), stream));
        database.saturation_ = CUDDL_TRY(
            download(source.saturation_states().data(), source.saturation_states().size(), stream)
        );
        return database;
    }

    reference_index_file() = delete;

    struct decoded_index {
        reference_database_file database_;
        index_storage storage_;
        std::vector<uint32_t> offsets_{};
        std::vector<uint32_t> postings_{};
        std::vector<uint16_t> keys_{};
    };

    [[nodiscard]] static Result<decoded_index>
    read(std::filesystem::path const& path, reference_database_file database) try {
        return [&]() -> Result<decoded_index> {
            detail::database_file_reader reader{
                std::ifstream(path, std::ios::binary | std::ios::ate)
            };
            auto size = reader.input.tellg();
            if (size < 0) {
                return Err(Error::resource("cannot open index file: " + path.string()));
            }
            reader.remaining = static_cast<uint64_t>(size);
            reader.input.seekg(0);
            char magic[8];
            CUDDL_TRY(reader.bytes(magic, sizeof(magic)));
            if (std::string_view(magic, 8) != std::string_view("CUDDLIX\0", 8)) {
                return Err(Error::invalid_argument("not a cuDDL reference index file"));
            }
            uint32_t version{}, kind{};
            CUDDL_TRY(reader.value(version));
            CUDDL_TRY(reader.value(kind));
            if (version != 1 || kind > 1) {
                return Err(Error::invalid_argument("unsupported reference index format"));
            }
            uint64_t digest{};
            CUDDL_TRY(reader.value(digest));
            auto const source = database.metadata();
            if (database.rows().size() != static_cast<uint64_t>(source.reference_count) *
                                              source.compatibility.bucket_count ||
                database.saturation().size() != source.reference_count) {
                return Err(
                    Error::invalid_argument("reference database has no complete stored rows")
                );
            }
            if (digest != CUDDL_TRY(detail::reference_database_digest(database))) {
                return Err(
                    Error::invalid_argument("reference index belongs to a different database")
                );
            }
            uint64_t count{};
            CUDDL_TRY(reader.value(count));
            auto const metadata = database.metadata();
            auto const capacity =
                detail::indexed_posting_count(metadata.reference_count, metadata.compatibility);
            auto const offset_count =
                kind == 0 ? detail::indexed_cell_count(metadata.compatibility) + 1 : 0;
            auto const key_count = kind == 1 ? capacity : 0;
            if (capacity > std::numeric_limits<uint32_t>::max() || count > capacity ||
                (kind == 1 && count != capacity) ||
                offset_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
                capacity > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
                reader.remaining != (offset_count + count) * sizeof(uint32_t) +
                                        key_count * sizeof(uint16_t) + sizeof(uint32_t)) {
                return Err(Error::invalid_argument("invalid reference index extents"));
            }
            decoded_index result{
                std::move(database), kind == 0 ? index_storage::dense : index_storage::sparse
            };
            result.offsets_.resize(static_cast<size_t>(offset_count));
            result.postings_.resize(static_cast<size_t>(count));
            result.keys_.resize(static_cast<size_t>(key_count));
            CUDDL_TRY(reader.words(result.offsets_));
            CUDDL_TRY(reader.words(result.postings_));
            CUDDL_TRY(reader.bytes(result.keys_.data(), result.keys_.size() * sizeof(uint16_t)));
            if constexpr (std::endian::native != std::endian::little) {
                for (auto& key : result.keys_) {
                    key = detail::database_file_little_endian(key);
                }
            }
            auto expected = static_cast<uint32_t>(reader.checksum);
            uint32_t checksum{};
            CUDDL_TRY(reader.value(checksum));
            if (checksum != expected) {
                return Err(Error::invalid_argument("reference index checksum mismatch"));
            }
            CUDDL_TRY(validate_payload(result));
            return result;
        }();
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    template <typename T>
    static Result<std::vector<T>> download(T const* data, size_t size, cuda::stream_ref stream) {
        std::vector<T> host(size);
        if (size != 0) {
            CUDDL_CUDA_TRY(
                cuda::copy_bytes(
                    stream, cuda::std::span{data, size}, cuda::std::span{host.data(), host.size()}
                )
            );
            CUDDL_CUDA_TRY(stream.sync());
        }
        return host;
    }

    static Result<void> validate_payload(decoded_index const& index) {
        auto const& database_ = index.database_;
        auto const& offsets_ = index.offsets_;
        auto const& postings_ = index.postings_;
        auto const& keys_ = index.keys_;
        auto storage_ = index.storage_;
        auto const m = index.database_.metadata();
        auto const& c = m.compatibility;
        std::vector<uint32_t> seen(m.reference_count, std::numeric_limits<uint32_t>::max());
        auto rows = database_.rows();
        if (storage_ == index_storage::sparse) {
            for (uint32_t bucket = 0; bucket < c.indexed_bucket_count; ++bucket) {
                auto start = static_cast<size_t>(bucket) * m.reference_count;
                for (uint32_t i = 0; i < m.reference_count; ++i) {
                    auto pos = start + i;
                    auto id = postings_[pos];
                    if (id >= m.reference_count || seen[id] == bucket ||
                        (i != 0 && keys_[pos] < keys_[pos - 1])) {
                        return Err(Error::invalid_argument("invalid sparse index postings"));
                    }
                    seen[id] = bucket;
                    auto score =
                        detail::winner(rows[static_cast<size_t>(id) * c.bucket_count + bucket]);
                    auto key =
                        score == 0 || c.key_mask == 0xffffU ? score : (score & c.key_mask) + 1U;
                    if (keys_[pos] != key) {
                        return Err(
                            Error::invalid_argument("sparse index key does not match reference")
                        );
                    }
                }
            }
        } else {
            if (offsets_.front() != 0 || offsets_.back() != postings_.size()) {
                return Err(Error::invalid_argument("invalid dense index endpoints"));
            }
            size_t expected = 0;
            for (auto row : rows) {
                expected += detail::winner(row) != 0;
            }
            if (expected != postings_.size()) {
                return Err(Error::invalid_argument("dense index omits reference scores"));
            }
            auto key_count = static_cast<size_t>(c.key_mask) + 1;
            for (uint32_t bucket = 0; bucket < c.indexed_bucket_count; ++bucket) {
                for (size_t key = 0; key < key_count; ++key) {
                    auto cell = static_cast<size_t>(bucket) * key_count + key;
                    auto begin = offsets_[cell], end = offsets_[cell + 1];
                    if (begin > end || end > postings_.size()) {
                        return Err(Error::invalid_argument("invalid dense index offsets"));
                    }
                    for (auto pos = begin; pos < end; ++pos) {
                        auto id = postings_[pos];
                        if (id >= m.reference_count || seen[id] == bucket) {
                            return Err(Error::invalid_argument("invalid dense index postings"));
                        }
                        seen[id] = bucket;
                        auto score =
                            detail::winner(rows[static_cast<size_t>(id) * c.bucket_count + bucket]);
                        if (score == 0 || (score & c.key_mask) != key) {
                            return Err(
                                Error::invalid_argument("dense index key does not match reference")
                            );
                        }
                    }
                }
            }
        }
        return Ok();
    }
};

}  // namespace cuddl
