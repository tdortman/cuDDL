#pragma once

#include <zlib.h>
#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuddl/error.hpp>
#include <cusbf/detail/fastx_buffer_reader.hpp>
#include <cusbf/detail/fastx_file_buffer.hpp>
#include <cusbf/detail/fastx_sequence_scan.hpp>

namespace cuddl::detail {

// Collects every '>' offset in data[begin, end). Phase one of the parallel FASTA scan:
// pure search, no header interpretation.
inline void gather_header_candidates(
    std::string_view data,
    size_t begin,
    size_t end,
    std::vector<size_t>& out
) {
    auto const segment = data.substr(begin, end - begin);
    size_t pos = 0;
    while (true) {
        auto const found = segment.find('>', pos);
        if (found == std::string_view::npos) return;
        out.push_back(begin + found);
        pos = found + 1;
    }
}

// Serial header walk over ascending '>' candidates. Produces exactly
// cusbf::detail::fastx_fasta_extents(data): candidates before the cursor are bytes a
// serial search jumps over with its line-end skip, and the predecessor rule is local.
inline std::vector<cusbf::detail::fastx_sequence_extent>
walk_header_candidates(std::string_view data, std::vector<size_t> const& candidates) {
    std::vector<cusbf::detail::fastx_sequence_extent> extents;
    size_t sequence = std::string_view::npos;
    size_t cursor = 0;
    for (auto const header : candidates) {
        if (header < cursor) continue;
        if (header != 0 && data[header - 1] != '\n' && data[header - 1] != '\r') {
            cursor = header + 1;
            continue;
        }
        if (sequence != std::string_view::npos && header > sequence) {
            extents.push_back({data.data() + sequence, data.data() + header});
        }
        auto const end = cusbf::detail::fastx_line_end(data, header);
        sequence = end < data.size() ? end + 1 : data.size();
        cursor = sequence;
    }
    if (sequence != std::string_view::npos && sequence < data.size()) {
        extents.push_back({data.data() + sequence, data.data() + data.size()});
    }
    return extents;
}

// Parallel equivalent of cusbf::detail::fastx_fasta_extents for large inputs. Shards the
// '>' search (the full-buffer memchr pass) across threads, then walks the ordered
// candidates once. Small inputs use the serial scan directly.
inline std::vector<cusbf::detail::fastx_sequence_extent> parallel_fastx_fasta_extents(
    std::string_view data
) {
    auto const hardware = std::max(1U, std::thread::hardware_concurrency());
    auto const shards =
        std::min<size_t>(std::min<unsigned>(hardware, 8U), data.size() / (size_t{64} << 20) + 1);
    if (shards <= 1) return cusbf::detail::fastx_fasta_extents(data);
    std::vector<std::vector<size_t>> gathered(shards);
    std::vector<std::thread> workers;
    workers.reserve(shards - 1);
    auto const span = (data.size() + shards - 1) / shards;
    for (size_t shard = 1; shard < shards; ++shard) {
        workers.emplace_back([&, shard] {
            auto const begin = std::min(shard * span, data.size());
            gather_header_candidates(
                data, begin, std::min(begin + span, data.size()), gathered[shard]
            );
        });
    }
    gather_header_candidates(data, 0, std::min(span, data.size()), gathered[0]);
    for (auto& worker : workers) worker.join();
    std::vector<size_t> candidates;
    for (auto const& shard : gathered)
        candidates.insert(candidates.end(), shard.begin(), shard.end());
    return walk_header_candidates(data, candidates);
}

// Heap ownership keeps extent addresses stable, including short decompressed strings.
struct fastx_sequence_file {
    std::unique_ptr<cusbf::detail::FastxFileBuffer> file;
    std::string decompressed;
    std::string sequence;
    std::vector<cusbf::detail::fastx_sequence_extent> extents;
};

inline Result<std::unique_ptr<fastx_sequence_file>> load_fastx_sequence_file(
    std::string const& path
) {
    auto file = cusbf::detail::FastxFileBuffer::load(path);
    if (!file) {
        return Err(Error::invalid_argument("cannot open FASTX file: " + path));
    }
    auto result = std::make_unique<fastx_sequence_file>();
    result->file = std::move(*file);
    auto data = result->file->data();
    auto& decompressed = result->decompressed;
    if (data.size() >= 2 && static_cast<unsigned char>(data[0]) == 0x1f &&
        static_cast<unsigned char>(data[1]) == 0x8b) {
        // cuSBF's GzIstream treats decompression errors as EOF. Check zlib errors here so
        // truncated or corrupt genomes cannot silently produce a partial database.
        std::unique_ptr<gzFile_s, decltype(&gzclose)> input{gzopen(path.c_str(), "rb"), &gzclose};
        if (!input) return Err(Error::resource("cannot open gzip FASTX file: " + path));
        char chunk[65536];
        while (true) {
            auto const size = gzread(input.get(), chunk, sizeof(chunk));
            int status{};
            auto const message = gzerror(input.get(), &status);
            if (size < 0 || (status != Z_OK && status != Z_STREAM_END)) {
                return Err(Error::invalid_argument("gzip FASTX error: " + std::string(message)));
            }
            if (size == 0) break;
            decompressed.append(chunk, static_cast<size_t>(size));
        }
        data = decompressed;
    }

    auto& sequence = result->sequence;
    auto& extents = result->extents;
    auto const first = data.find_first_not_of("\r\n");
    if (first != std::string_view::npos && data[first] == '@') {
        cusbf::detail::FastxBufferReader reader{data, path};
        cusbf::detail::FastxRecord record;
        std::string_view external;
        std::vector<std::pair<size_t, size_t>> records;
        while (true) {
            auto const begin = sequence.size();
            auto next = reader.appendNextRecord(record, sequence, external);
            if (!next) return Err(Error::invalid_argument(next.error().message()));
            if (!*next) break;
            if (sequence.size() > begin) records.emplace_back(begin, sequence.size());
        }
        if (sequence.empty()) return result;
        for (auto const& [begin, end] : records) {
            extents.push_back({sequence.data() + begin, sequence.data() + end});
        }
    } else {
        extents = parallel_fastx_fasta_extents(data);
    }
    if (extents.empty()) {
        if (data.size() > 0) {
            return Err(Error::invalid_argument("FASTX parse error near: " + path));
        }
        return result;
    }

    return result;
}

}  // namespace cuddl::detail
