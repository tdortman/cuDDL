#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuddl/error.hpp>

namespace cuddl::a48 {

/// @brief Per-database construction metadata decoded from the A48 file-level header.
///
/// These mirror the BBTools DDL TSV header fields that define how the score rows were produced
/// (`#k`, `#seed`, `#exponent`, `#blacklist`, `#date`). They are used both to decode the relative
/// score rows and to assert provenance before any result is accepted.
struct database_metadata {
    /// K-mer length (`#k`). Present when the file was written with an explicit k.
    uint32_t kmer_length{};
    bool has_kmer_length{};
    /// Hash seed (`#seed`). Present when the file was written with a nonzero seed.
    uint64_t seed{};
    bool has_seed{};
    /// Exponent bit width (`#exponent`) of each 16-bit bucket value. The mantissa gets
    /// `16 - exponent_bits` bits. Defaults to 6 (the BBTools DDL default) when absent.
    uint32_t exponent_bits{6U};
    bool has_exponent{};
    /// Blacklist filenames recorded in the header (`#blacklist`), if any.
    std::string blacklist;
    /// Number of decoded records.
    size_t record_count{};
};

/// @brief Metadata attached to one decoded A48 record.
struct record_metadata {
    /// First-observed sequence ID (`#id`); -1 when omitted.
    int64_t id{-1};
    /// Taxonomy ID (`#tid`); -1 when unknown.
    int64_t tax_id{-1};
    /// Organism/record name (`#name`).
    std::string name;
    /// Source filename (`#file`).
    std::string filename;
    /// Number of bases (`#bases`); -1 when omitted.
    int64_t bases{-1};
    /// Number of contigs (`#contigs`); -1 when omitted.
    int32_t contigs{-1};
    /// GC fraction (`#gc`); negative when omitted.
    float gc{-1.0f};
    /// Origin (`#origin`).
    std::string origin;
    /// Lineage (`#lineage`).
    std::string lineage;
    /// Bucket count recorded by `#len`; 0 when omitted.
    uint32_t len{};
    /// Global NLZ offset (`#offset`); -1 when omitted (legacy absolute-encoded rows).
    int32_t offset{-1};
};

/// @brief One decoded DDL score row plus its stable record identity and construction metadata.
struct record {
    /// Stable record identity: the record's ordinal in the file, matching the BBTools DDLIndex
    /// clade IDs (the index stores the record's position, not its `#id`).
    uint32_t ordinal{};
    record_metadata metadata;
    /// Absolute 16-bit bucket maxima (`maxArray`) reconstructed from the A48-encoded values.
    std::vector<uint16_t> scores;
};

/// @brief Fully decoded A48 reference database.
struct database {
    database_metadata metadata;
    std::vector<record> records;
};

/// @brief Decodes one A48 (base-64, ASCII-offset-48) token to its integer value.
///
/// Returns @c std::nullopt when a character is outside the A48 alphabet (ASCII 48..111), i.e. the
/// token is malformed.
[[nodiscard]] inline std::optional<uint64_t> decode_a48_token(std::string_view token) {
    if (token.empty()) {
        return uint64_t{0};
    }
    uint64_t value = 0;
    for (char const character : token) {
        auto const code = static_cast<uint8_t>(character);
        if (code < 48U || code > 111U) {
            return std::nullopt;
        }
        value = (value << 6U) | static_cast<uint64_t>(code - 48U);
    }
    return value;
}

/// @brief Encodes an integer as an A48 (base-64, ASCII-offset-48) token.
[[nodiscard]] inline std::string encode_a48_token(uint64_t value) {
    if (value == 0U) {
        return std::string(1, '0');
    }
    std::string reversed;
    while (value != 0U) {
        reversed.push_back(static_cast<char>(48U + (value & 0x3fU)));
        value >>= 6U;
    }
    return {reversed.rbegin(), reversed.rend()};
}

/// @brief Reconstructs one absolute 16-bit bucket maxima from an A48-decoded value.
///
/// Matches BBTools' `DynamicDemiLog.fromArray`: when @p offset is >= 0 the stored value is
/// relative to the record's global NLZ and must be promoted back into the full 16-bit space; when
/// @p offset is -1 the stored value is already an absolute score. @p exponent_bits selects the
/// split between the exponent (NLZ tier) and mantissa fields.
[[nodiscard]] inline uint16_t
promote_score(uint16_t loaded, int32_t offset, uint32_t exponent_bits) {
    if (loaded == 0U || offset < 0) {
        return loaded;
    }
    auto const mantissa_bits = 16U - exponent_bits;
    auto const mask = (1U << mantissa_bits) - 1U;
    auto const rel_nlz = static_cast<uint32_t>(loaded >> mantissa_bits);
    auto const abs_nlz = rel_nlz + static_cast<uint32_t>(offset);
    return static_cast<uint16_t>((abs_nlz << mantissa_bits) | (loaded & mask));
}

/// @brief Splits @p text on @p separator.
///
/// With @p keep_trailing_empty the split is exactly BBTools' `LineParser1` term layout: a line
/// ending in a separator produces one final empty field, which `parseLongA48` decodes as zero.
/// Header parsing drops that field instead (a trailing empty value is not a distinct header).
[[nodiscard]] inline std::vector<std::string_view>
split_fields(std::string_view text, char separator, bool keep_trailing_empty = false) {
    std::vector<std::string_view> fields;
    size_t begin = 0;
    while (true) {
        auto const end = text.find(separator, begin);
        if (end == std::string_view::npos) {
            fields.push_back(text.substr(begin));
            break;
        }
        fields.push_back(text.substr(begin, end - begin));
        begin = end + 1U;
    }
    if (!keep_trailing_empty && !fields.empty() && fields.back().empty()) {
        fields.pop_back();
    }
    return fields;
}

namespace detail {

/// @brief Parses a header field with @p parser, converting any thrown exception into an error.
template <typename T, typename Parser>
[[nodiscard]] Result<T> parse_field(Parser&& parser, std::string_view value, char const* field) {
    try {
        return Result<T>::ok(static_cast<T>(parser(std::string(value))));
    } catch (std::exception const&) {
        return Err(Error::invalid_argument(std::string("malformed A48 ") + field + " value"));
    }
}

}  // namespace detail

/// @brief True when @p key is a record-level header (part of a record block) rather than a
/// file-level header that precedes every record.
[[nodiscard]] inline bool is_record_header(std::string_view key) {
    return key == "id" || key == "tid" || key == "name" || key == "file" || key == "bases" ||
           key == "contigs" || key == "gc" || key == "origin" || key == "lineage" || key == "len" ||
           key == "offset";
}

/// @brief Parses a record header block (one "#key\tvalue" per line) into @p record_metadata.
[[nodiscard]] inline Result<record_metadata> parse_record_headers(std::string_view block) {
    record_metadata meta;
    for (auto const line : split_fields(block, '\n')) {
        if (line.empty() || line.front() != '#') {
            continue;
        }
        auto const fields = split_fields(line, '\t');
        if (fields.empty()) {
            continue;
        }
        auto key = fields.front();
        key.remove_prefix(1);  // drop '#'
        if (fields.size() < 2U) {
            continue;
        }
        auto const value = fields[1];
        if (key == "id") {
            auto const parsed = detail::parse_field<int64_t>(
                [](std::string const& s) { return std::stoll(s); }, value, "id"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.id = *parsed;
        } else if (key == "tid") {
            auto const parsed = detail::parse_field<int64_t>(
                [](std::string const& s) { return std::stoll(s); }, value, "tid"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.tax_id = *parsed;
        } else if (key == "name") {
            meta.name = std::string(value);
        } else if (key == "file") {
            meta.filename = std::string(value);
        } else if (key == "bases") {
            auto const parsed = detail::parse_field<int64_t>(
                [](std::string const& s) { return std::stoll(s); }, value, "bases"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.bases = *parsed;
        } else if (key == "contigs") {
            auto const parsed = detail::parse_field<int32_t>(
                [](std::string const& s) { return std::stol(s); }, value, "contigs"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.contigs = *parsed;
        } else if (key == "gc") {
            auto const parsed = detail::parse_field<float>(
                [](std::string const& s) { return std::stof(s); }, value, "gc"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.gc = *parsed;
        } else if (key == "origin") {
            meta.origin = std::string(value);
        } else if (key == "lineage") {
            meta.lineage = std::string(value);
        } else if (key == "len") {
            auto const parsed = detail::parse_field<uint32_t>(
                [](std::string const& s) { return std::stoul(s); }, value, "len"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.len = *parsed;
        } else if (key == "offset") {
            auto const parsed = detail::parse_field<int32_t>(
                [](std::string const& s) { return std::stol(s); }, value, "offset"
            );
            if (!parsed) {
                return Err(parsed.error());
            }
            meta.offset = *parsed;
        }
    }
    return Result<record_metadata>::ok(std::move(meta));
}

/// @brief Decodes one A48 data row into absolute 16-bit bucket scores.
[[nodiscard]] inline Result<std::vector<uint16_t>>
decode_a48_row(std::string_view row, record_metadata const& metadata, uint32_t exponent_bits) {
    auto const fields = split_fields(row, '\t', true);
    std::vector<uint16_t> scores;
    scores.reserve(fields.size());
    for (auto const field : fields) {
        auto const loaded = decode_a48_token(field);
        if (!loaded) {
            return Err(
                Error::invalid_argument(
                    "A48 data row contains a character outside the A48 alphabet"
                )
            );
        }
        scores.push_back(
            promote_score(static_cast<uint16_t>(*loaded), metadata.offset, exponent_bits)
        );
    }
    return Result<std::vector<uint16_t>>::ok(std::move(scores));
}

/// @brief Decodes an uncompressed A48-encoded DDL TSV into score rows and metadata.
///
/// The decoder is intentionally strict about the structural invariants that would otherwise yield
/// silent garbage: A48 alphabet violations, an omitted `#len` disagreeing with the data row, a
/// bucket count that changes between records, an absent `#exponent` on a relative-encoded file, and
/// records without any bucket values are all reported as malformed input. It does not, however,
/// depend on the large official RefSeq asset, so a small deterministic fixture can exercise the
/// code paths in unit tests.
[[nodiscard]] inline Result<database> decode_a48_tsv(std::string_view input) {
    database result;
    uint32_t exponent_bits = 6U;
    bool has_exponent = false;
    size_t expected_bucket_count = 0;
    bool bucket_count_set = false;

    std::string header_block;
    size_t ordinal = 0;
    size_t begin = 0;
    while (begin <= input.size()) {
        auto const end = input.find('\n', begin);
        auto line =
            input.substr(begin, end == std::string_view::npos ? input.size() - begin : end - begin);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (line.empty()) {
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1U;
            continue;
        }

        if (line.front() == '#') {
            auto const fields = split_fields(line, '\t');
            if (fields.empty()) {
                break;
            }
            auto key = fields.front();
            key.remove_prefix(1);  // drop '#'
            auto const value = fields.size() > 1U ? fields[1] : std::string_view{};

            if (key == "k") {
                auto const parsed = detail::parse_field<uint32_t>(
                    [](std::string const& s) { return std::stoul(s); }, value, "k"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                result.metadata.kmer_length = *parsed;
                result.metadata.has_kmer_length = true;
            } else if (key == "seed") {
                auto const parsed = detail::parse_field<uint64_t>(
                    [](std::string const& s) { return std::stoull(s); }, value, "seed"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                result.metadata.seed = *parsed;
                result.metadata.has_seed = true;
            } else if (key == "exponent") {
                auto const parsed = detail::parse_field<uint32_t>(
                    [](std::string const& s) { return std::stoul(s); }, value, "exponent"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                if (*parsed < 1U || *parsed > 15U) {
                    return Err(Error::invalid_argument("A48 exponent must be in [1, 15]"));
                }
                exponent_bits = *parsed;
                result.metadata.exponent_bits = *parsed;
                result.metadata.has_exponent = true;
                has_exponent = true;
            } else if (key == "blacklist") {
                result.metadata.blacklist = std::string(value);
            }

            if (is_record_header(key)) {
                if (!header_block.empty()) {
                    header_block.push_back('\n');
                }
                header_block.append(line);
            }

            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1U;
            continue;
        }

        // A data row terminates the currently accumulated record header block.
        if (header_block.empty()) {
            return Err(Error::invalid_argument("A48 data row appears before any record header"));
        }
        auto meta = CUDDL_TRY(parse_record_headers(header_block));
        header_block.clear();

        if (meta.offset >= 0 && !has_exponent) {
            return Err(
                Error::invalid_argument("relative-encoded A48 record requires a #exponent header")
            );
        }
        auto scores = CUDDL_TRY(decode_a48_row(line, meta, exponent_bits));
        if (scores.empty()) {
            return Err(Error::invalid_argument("A48 record has no bucket values"));
        }
        if (!bucket_count_set) {
            expected_bucket_count = scores.size();
            bucket_count_set = true;
        } else if (scores.size() != expected_bucket_count) {
            return Err(Error::invalid_argument("A48 data row width changed between records"));
        }
        if (meta.len != 0U && meta.len != scores.size()) {
            return Err(Error::invalid_argument("#len disagrees with the A48 data row width"));
        }

        record decoded;
        decoded.ordinal = static_cast<uint32_t>(ordinal);
        decoded.metadata = meta;
        decoded.scores = std::move(scores);
        result.records.push_back(std::move(decoded));
        ++ordinal;

        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1U;
    }

    if (!header_block.empty()) {
        return Err(Error::invalid_argument("A48 file ends inside a record header block"));
    }
    if (result.records.empty()) {
        return Err(Error::invalid_argument("A48 file contains no records"));
    }

    result.metadata.record_count = result.records.size();
    return Result<database>::ok(std::move(result));
}


/// @brief Multithreaded variant of @ref decode_a48_tsv.
///
/// One pass discovers file-level metadata, record header blocks, and data-row spans. The first
/// record is decoded synchronously to establish the bucket width; the remaining independent
/// records are decoded by a fixed pool and written directly to their ordinal slots, so order and
/// strict validation are identical to the single-threaded decoder.
[[nodiscard]] inline Result<database>
decode_a48_tsv_parallel(std::string_view input, uint32_t requested_threads = 0) {
    struct record_job {
        std::string header;
        std::string_view data;
        uint32_t ordinal{};
    };

    database result;
    std::vector<record_job> jobs;
    uint32_t exponent_bits = 6U;
    bool has_exponent = false;
    size_t expected_bucket_count = 0;
    bool bucket_count_set = false;

    std::string header_block;
    size_t ordinal = 0;
    size_t begin = 0;
    while (begin <= input.size()) {
        auto const end = input.find('\n', begin);
        auto line =
            input.substr(begin, end == std::string_view::npos ? input.size() - begin : end - begin);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1U;
            continue;
        }

        if (line.front() == '#') {
            auto const fields = split_fields(line, '\t');
            if (fields.empty()) {
                break;
            }
            auto key = fields.front();
            key.remove_prefix(1);
            auto const value = fields.size() > 1U ? fields[1] : std::string_view{};
            if (key == "k") {
                auto const parsed = detail::parse_field<uint32_t>(
                    [](std::string const& s) { return std::stoul(s); }, value, "k"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                result.metadata.kmer_length = *parsed;
                result.metadata.has_kmer_length = true;
            } else if (key == "seed") {
                auto const parsed = detail::parse_field<uint64_t>(
                    [](std::string const& s) { return std::stoull(s); }, value, "seed"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                result.metadata.seed = *parsed;
                result.metadata.has_seed = true;
            } else if (key == "exponent") {
                auto const parsed = detail::parse_field<uint32_t>(
                    [](std::string const& s) { return std::stoul(s); }, value, "exponent"
                );
                if (!parsed) {
                    return Err(parsed.error());
                }
                if (*parsed < 1U || *parsed > 15U) {
                    return Err(Error::invalid_argument("A48 exponent must be in [1, 15]"));
                }
                exponent_bits = *parsed;
                result.metadata.exponent_bits = *parsed;
                result.metadata.has_exponent = true;
                has_exponent = true;
            } else if (key == "blacklist") {
                result.metadata.blacklist = std::string(value);
            }

            if (is_record_header(key)) {
                if (!header_block.empty()) {
                    header_block.push_back('\n');
                }
                header_block.append(line);
            }
            if (end == std::string_view::npos) {
                break;
            }
            begin = end + 1U;
            continue;
        }

        if (header_block.empty()) {
            return Err(Error::invalid_argument("A48 data row appears before any record header"));
        }
        auto const record_ordinal = static_cast<uint32_t>(ordinal);
        if (record_ordinal == 0U) {
            auto meta = CUDDL_TRY(parse_record_headers(header_block));
            header_block.clear();
            if (meta.offset >= 0 && !has_exponent) {
                return Err(
                    Error::invalid_argument("relative-encoded A48 record requires a #exponent header")
                );
            }
            auto scores = CUDDL_TRY(decode_a48_row(line, meta, exponent_bits));
            if (scores.empty()) {
                return Err(Error::invalid_argument("A48 record has no bucket values"));
            }
            expected_bucket_count = scores.size();
            bucket_count_set = true;
            if (meta.len != 0U && meta.len != scores.size()) {
                return Err(Error::invalid_argument("#len disagrees with the A48 data row width"));
            }
            record decoded;
            decoded.ordinal = record_ordinal;
            decoded.metadata = std::move(meta);
            decoded.scores = std::move(scores);
            result.records.push_back(std::move(decoded));
        } else {
            jobs.push_back(
                record_job{std::move(header_block), line, static_cast<uint32_t>(ordinal)}
            );
            header_block.clear();
        }
        ++ordinal;

        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1U;
    }

    if (!header_block.empty()) {
        return Err(Error::invalid_argument("A48 file ends inside a record header block"));
    }
    if (result.records.empty()) {
        return Err(Error::invalid_argument("A48 file contains no records"));
    }

    result.records.resize(result.records.size() + jobs.size());
    auto const requested = requested_threads == 0U ? std::thread::hardware_concurrency()
                                                   : requested_threads;
    auto const worker_count = static_cast<size_t>(std::max(1U, requested));
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(worker_count);
    workers.reserve(worker_count);

    for (size_t worker = 0; worker < worker_count; ++worker) {
        auto const chunk_begin = jobs.size() * worker / worker_count;
        auto const chunk_end = jobs.size() * (worker + 1U) / worker_count;
        if (chunk_begin == chunk_end) {
            continue;
        }
        workers.emplace_back([&, chunk_begin, chunk_end, worker]() {
            try {
                for (size_t i = chunk_begin; i < chunk_end; ++i) {
                    auto const& job = jobs[i];
                    auto meta = parse_record_headers(job.header);
                    if (!meta) {
                        throw std::runtime_error(meta.error().message());
                    }
                    if (meta->offset >= 0 && !has_exponent) {
                        throw std::runtime_error(
                            "relative-encoded A48 record requires a #exponent header"
                        );
                    }
                    auto scores = decode_a48_row(job.data, *meta, exponent_bits);
                    if (!scores) {
                        throw std::runtime_error(scores.error().message());
                    }
                    if (scores->empty()) {
                        throw std::runtime_error("A48 record has no bucket values");
                    }
                    if (!bucket_count_set || scores->size() != expected_bucket_count) {
                        throw std::runtime_error("A48 data row width changed between records");
                    }
                    if (meta->len != 0U && meta->len != scores->size()) {
                        throw std::runtime_error("#len disagrees with the A48 data row width");
                    }
                    record decoded;
                    decoded.ordinal = job.ordinal;
                    decoded.metadata = std::move(*meta);
                    decoded.scores = std::move(*scores);
                    result.records[job.ordinal] = std::move(decoded);
                }
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (auto const& error : errors) {
        if (error) {
            try {
                std::rethrow_exception(error);
            } catch (std::exception const& exception) {
                return Err(Error::invalid_argument(exception.what()));
            }
        }
    }

    result.metadata.record_count = result.records.size();
    return Result<database>::ok(std::move(result));
}

/// @brief Exact bucket-classification counts for one decoded row pair.
struct row_summary {
    uint32_t lower{};
    uint32_t equal{};
    uint32_t higher{};
    uint32_t both_empty{};

    friend bool operator==(row_summary const&, row_summary const&) = default;
};

/// @brief Independent host oracle for one query over a decoded reference database.
///
/// One fused row-major pass over the flat score rows reproduces, byte for byte over the decoded
/// rows, both quantities the BBTools parity check needs:
///
/// - @c match_counts[r] is the DDLIndex match count: every bucket where the query and reference
///   hold the same nonzero score increments the count, so a reference visited from several buckets
///   accumulates all cross-bucket hits. This is exactly the counting rule of BBTools'
///   `DDLIndexBase.accumulateCounts`, shared bit-identically by the matrix, 32-bit CSR, and CSR2
///   storage backends.
/// - @c summaries[id] holds the exhaustive lower/equal/higher/both-empty classification of every
///   retained candidate (a reference whose match count meets @p minimum_matches). The
///   classification matches `Vector.compareDDL` over BBTools' absolute `maxArray`: empty-empty
///   pairs are counted separately from equal, and only the stored 16-bit scores are compared.
///
/// @param rows       Flat row-major score rows for @p reference_count references of
///                   @p bucket_count buckets each.
/// @param query      One complete decoded query row of @p bucket_count scores.
/// @param bucket_count   Width of every row.
/// @param minimum_matches Threshold for retaining per-candidate summaries.
struct oracle_result {
    std::vector<uint32_t> match_counts;
    std::vector<row_summary> summaries;
};

[[nodiscard]] inline oracle_result exhaustive_oracle(
    std::vector<uint16_t> const& rows,
    std::vector<uint16_t> const& query,
    size_t bucket_count,
    uint32_t minimum_matches
) {
    auto const reference_count = rows.size() / bucket_count;
    oracle_result result;
    result.match_counts.assign(reference_count, 0U);
    result.summaries.assign(reference_count, row_summary{});
    for (size_t reference = 0; reference < reference_count; ++reference) {
        auto summary = row_summary{};
        auto const* reference_row = rows.data() + reference * bucket_count;
        auto const* query_row = query.data();
        for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
            auto const query_score = query_row[bucket];
            auto const reference_score = reference_row[bucket];
            if (query_score == reference_score) {
                if (query_score == 0U) {
                    ++summary.both_empty;
                } else {
                    ++summary.equal;
                    ++result.match_counts[reference];
                }
            } else if (query_score < reference_score) {
                ++summary.lower;
            } else {
                ++summary.higher;
            }
        }
        if (result.match_counts[reference] >= minimum_matches) {
            result.summaries[reference] = summary;
        }
    }
    return result;
}

/// @brief Convenience overload over decoded records instead of one flat score array.
[[nodiscard]] inline oracle_result exhaustive_oracle(
    std::vector<std::vector<uint16_t>> const& rows,
    std::vector<uint16_t> const& query,
    uint32_t minimum_matches
) {
    std::vector<uint16_t> flat;
    auto const bucket_count = rows.empty() ? size_t{0} : rows.front().size();
    flat.reserve(rows.size() * bucket_count);
    for (auto const& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return exhaustive_oracle(flat, query, bucket_count, minimum_matches);
}

}  // namespace cuddl::a48
