#pragma once

#include <cuda/buffer>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/stream>

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuddl/detail/dna.hpp>
#include <cuddl/detail/hash.cuh>
#include <cuddl/error.hpp>
#include <cuddl/fastx.hpp>

namespace cuddl {

namespace detail {
/// @brief Exact membership within the candidate's destination bucket.
struct blacklist_view {
    uint64_t const* keys = nullptr;
    uint32_t const* offsets = nullptr;

    __device__ bool contains(uint64_t key, size_t bucket) const noexcept {
        auto const first = offsets[bucket];
        auto const last = offsets[bucket + 1];
        return first != last && cuda::std::binary_search(keys + first, keys + last, key);
    }
};
}  // namespace detail

/// @brief Immutable canonical k-mers excluded from construction, independent of sketch size.
class kmer_blacklist {
   public:
    kmer_blacklist() = default;

    /// @brief Validates packed canonical words and normalizes their order and duplicates.
    kmer_blacklist(uint32_t k, std::vector<uint64_t> keys) : k_(k), keys_(std::move(keys)) {
        if (k < 1 || k > 31 || keys_.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("invalid blacklist k or entry count");
        }

        for (auto key : keys_) {
            if ((key >> (2 * k)) != 0 || canonical(key, k) != key) {
                throw std::invalid_argument("blacklist words must be canonical packed k-mers");
            }
        }

        std::sort(keys_.begin(), keys_.end());
        keys_.erase(std::unique(keys_.begin(), keys_.end()), keys_.end());

        if (keys_.empty()) {
            k_ = 0;
            return;
        }
        // Explicit little-endian words make identity independent of host byte order.
        std::vector<uint64_t> words;
        words.reserve(keys_.size() + 3);
        words.insert(words.end(), {uint64_t{1}, k, keys_.size()});
        words.insert(words.end(), keys_.begin(), keys_.end());

        if constexpr (std::endian::native != std::endian::little) {
            for (auto& word : words) {
                word = cuda::std::byteswap(word);
            }
        }

        identity_ = detail::xxhash64(
            reinterpret_cast<uint8_t const*>(words.data()),
            words.size() * sizeof(uint64_t),
            0x435544444c424c31ULL  // CUDDLBL1
        );

        if (identity_ == 0) identity_ = 1;
    }

    /// @brief Reads BBTools DDL FASTA/FASTQ, including gzip and fused records, or exact DNA lines.
    /// FASTX contributes every valid k-mer window; ambiguity and record boundaries reset it.
    [[nodiscard]] static Result<kmer_blacklist>
    load(std::filesystem::path const& path, uint32_t k) try {
        if (k < 1 || k > 31) return Err(Error::invalid_argument("invalid blacklist k"));

        std::ifstream input(path);
        if (!input) return Err(Error::resource("cannot open blacklist: " + path.string()));

        input >> std::ws;
        auto const first = input.peek();
        if (first == '>' || first == '@' || first == 0x1f) {
            auto parsed = parse_fasta_file(path.string(), k, 1);
            if (!parsed) return Err(parsed.error());
            return kmer_blacklist(k, std::move(parsed->kmers));
        }
        input.clear();
        input.seekg(0);
        std::vector<uint64_t> keys;
        std::string line;
        size_t number = 0;

        while (std::getline(input, line)) {
            ++number;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.size() != k) {
                return Err(
                    Error::invalid_argument(
                        "invalid blacklist length at line " + std::to_string(number)
                    )
                );
            }
            uint64_t word = 0;
            for (char base : line) {
                auto const code = detail::encode_base(base);
                if (code == 0xffU) {
                    return Err(
                        Error::invalid_argument(
                            "invalid blacklist base at line " + std::to_string(number)
                        )
                    );
                }
                word = (word << 2) | code;
            }
            keys.push_back(canonical(word, k));
        }
        if (!input.eof()) return Err(Error::resource("cannot read blacklist: " + path.string()));
        return kmer_blacklist(k, std::move(keys));
    } catch (std::invalid_argument const& error) {
        return Err(Error::invalid_argument(error.what()));
    } catch (std::bad_alloc const& error) {
        return Err(Error::resource(error.what()));
    }

    [[nodiscard]] uint32_t kmer_length() const noexcept {
        return k_;
    }
    [[nodiscard]] uint64_t identity() const noexcept {
        return identity_;
    }
    [[nodiscard]] uint32_t version() const noexcept {
        return keys_.empty() ? 0U : 1U;
    }
    [[nodiscard]] std::vector<uint64_t> const& keys() const noexcept {
        return keys_;
    }

    static uint64_t canonical(uint64_t word, uint32_t k) noexcept {
        return std::max(word, detail::reverse_complement(word, k));
    }

   private:
    uint32_t k_ = 0;
    uint64_t identity_ = 0;
    std::vector<uint64_t> keys_;
};

/// @brief Reusable, move-only device lookup. Construction synchronizes the upload stream.
/// Keep this owner and its allocation stream alive until every consuming kernel completes.
class device_blacklist {
   public:
    device_blacklist(kmer_blacklist source, size_t buckets, cuda::stream_ref stream)
        : source_(std::move(source)), buckets_(buckets), device_(stream.device().get()) {
        if (buckets < 2048 || buckets > 8192 || !std::has_single_bit(buckets)) {
            throw std::invalid_argument("invalid blacklist bucket count");
        }

        if (source_.keys().empty()) return;

        std::vector<uint32_t> offsets(buckets + 1, 0);

        for (auto key : source_.keys()) {
            ++offsets[(detail::hash_kmer(key) & (buckets - 1)) + 1];
        }

        for (size_t b = 1; b <= buckets; ++b) {
            offsets[b] += offsets[b - 1];
        }

        auto cursor = offsets;
        std::vector<uint64_t> keys(source_.keys().size());
        for (auto key : source_.keys()) {
            keys[cursor[detail::hash_kmer(key) & (buckets - 1)]++] = key;
        }

        keys_.emplace(cuda::make_device_buffer<uint64_t>(stream, stream.device(), keys));
        offsets_.emplace(cuda::make_device_buffer<uint32_t>(stream, stream.device(), offsets));
        stream.sync();
    }
    device_blacklist(device_blacklist const&) = delete;
    device_blacklist& operator=(device_blacklist const&) = delete;
    device_blacklist(device_blacklist&&) noexcept = default;
    device_blacklist& operator=(device_blacklist&&) noexcept = default;

    [[nodiscard]] kmer_blacklist const& source() const noexcept {
        return source_;
    }
    [[nodiscard]] cuda::std::optional<detail::blacklist_view> view() const noexcept {
        if (!keys_) return cuda::std::nullopt;
        return detail::blacklist_view{keys_->data(), offsets_->data()};
    }
    [[nodiscard]] Result<void> validate(uint32_t k, size_t buckets, cuda::stream_ref stream) const {
        if ((!source_.keys().empty() && source_.kmer_length() != k) || buckets != buckets_ ||
            stream.device().get() != device_) {
            return Err(
                Error::invalid_argument("blacklist construction configuration or device mismatch")
            );
        }
        return Ok();
    }

   private:
    kmer_blacklist source_;
    size_t buckets_;
    int device_;
    std::optional<cuda::device_buffer<uint64_t>> keys_;
    std::optional<cuda::device_buffer<uint32_t>> offsets_;
};

}  // namespace cuddl
