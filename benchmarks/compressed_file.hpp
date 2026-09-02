#pragma once

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cuddl_bench {

struct bgzf_block {
    uint64_t offset{};
    uint32_t size{};
};

[[nodiscard]] inline std::vector<bgzf_block>
find_bgzf_blocks(std::string const& path, uint64_t file_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    std::vector<bgzf_block> blocks;
    uint64_t offset = 0;
    while (offset < file_size) {
        file.seekg(static_cast<std::streamoff>(offset));
        std::array<char, 12> header{};
        file.read(header.data(), header.size());
        if (!file || static_cast<uint8_t>(header[0]) != 0x1fU ||
            static_cast<uint8_t>(header[1]) != 0x8bU) {
            return {};
        }
        auto const flags = static_cast<uint8_t>(header[3]);
        if ((flags & 0x04U) == 0U) {
            return {};
        }
        auto const extra_size = static_cast<uint16_t>(
            static_cast<uint8_t>(header[10]) | (static_cast<uint8_t>(header[11]) << 8U)
        );
        std::string extra(extra_size, '\0');
        file.read(extra.data(), extra.size());
        if (!file) {
            return {};
        }
        bool found = false;
        uint32_t block_size = 0;
        size_t cursor = 0;
        while (cursor + 4U <= extra.size()) {
            auto const sub_size = static_cast<uint16_t>(
                static_cast<uint8_t>(extra[cursor + 2U]) |
                (static_cast<uint8_t>(extra[cursor + 3U]) << 8U)
            );
            if (cursor + 4U + sub_size > extra.size()) {
                return {};
            }
            if (extra[cursor] == 'B' && extra[cursor + 1U] == 'C' && sub_size >= 2U) {
                block_size = static_cast<uint16_t>(
                    static_cast<uint8_t>(extra[cursor + 4U]) |
                    (static_cast<uint8_t>(extra[cursor + 5U]) << 8U)
                );
                found = true;
                break;
            }
            cursor += 4U + sub_size;
        }
        if (!found || block_size == 0U) {
            return {};
        }
        auto const total = static_cast<uint64_t>(block_size) + 1U;
        if (total > file_size - offset) {
            return {};
        }
        blocks.push_back(bgzf_block{offset, static_cast<uint32_t>(total)});
        offset += total;
    }
    return offset == file_size ? blocks : std::vector<bgzf_block>{};
}

[[nodiscard]] inline std::string inflate_bgzf_block(std::ifstream& file, bgzf_block const& block) {
    std::string compressed(block.size, '\0');
    file.seekg(static_cast<std::streamoff>(block.offset));
    file.read(compressed.data(), compressed.size());
    if (!file) {
        throw std::runtime_error("cannot read compressed BGZF block");
    }

    z_stream stream{};
    if (inflateInit2(&stream, 15 + 16) != Z_OK) {
        throw std::runtime_error("cannot initialise gzip inflater");
    }
    stream.next_in = reinterpret_cast<Bytef*>(compressed.data());
    stream.avail_in = compressed.size();

    std::string output;
    std::vector<char> buffer(1U << 20);
    while (true) {
        stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
        stream.avail_out = buffer.size();
        auto const status = inflate(&stream, Z_NO_FLUSH);
        auto const produced = buffer.size() - stream.avail_out;
        if (produced != 0U) {
            output.append(buffer.data(), produced);
        }
        if (status == Z_STREAM_END) {
            break;
        }
        if (status != Z_OK || (stream.avail_in == 0U && produced == 0U)) {
            inflateEnd(&stream);
            throw std::runtime_error("BGZF block inflate failed");
        }
    }
    inflateEnd(&stream);
    return output;
}

/// Reads an entire plain, gzip, or BGZF file. BGZF members are inflated in parallel.
[[nodiscard]] inline std::string read_file_any(std::string const& path) {
    auto const compressed_size = std::filesystem::file_size(path);
    auto const blocks = find_bgzf_blocks(path, compressed_size);
    if (!blocks.empty()) {
        auto const requested = std::thread::hardware_concurrency();
        auto const worker_count =
            std::min<size_t>(requested == 0U ? 1U : requested, std::max<size_t>(1U, blocks.size()));
        std::vector<std::string> outputs(worker_count);
        std::vector<std::exception_ptr> errors(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            auto const begin = blocks.size() * worker / worker_count;
            auto const end = blocks.size() * (worker + 1U) / worker_count;
            if (begin == end) {
                continue;
            }
            workers.emplace_back([&, worker, begin, end]() {
                try {
                    std::ifstream file(path, std::ios::binary);
                    for (size_t i = begin; i < end; ++i) {
                        outputs[worker].append(inflate_bgzf_block(file, blocks[i]));
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
                std::rethrow_exception(error);
            }
        }
        size_t total_size = 0;
        for (auto const& output : outputs) {
            total_size += output.size();
        }
        std::string contents;
        contents.reserve(total_size);
        for (auto& output : outputs) {
            contents.append(output);
        }
        return contents;
    }

    gzFile handle = gzopen(path.c_str(), "rb");
    if (handle == nullptr) {
        throw std::runtime_error("cannot open asset (plain or .gz expected): " + path);
    }
    gzbuffer(handle, 1U << 20);
    std::string contents;
    contents.reserve(static_cast<size_t>(compressed_size) * 2U + (1U << 20));
    std::vector<char> buffer(1U << 20);
    while (true) {
        auto const count = gzread(handle, buffer.data(), static_cast<unsigned>(buffer.size()));
        if (count <= 0) {
            break;
        }
        contents.append(buffer.data(), static_cast<size_t>(count));
    }
    if (gzclose(handle) != Z_OK) {
        throw std::runtime_error("cannot finish reading compressed asset: " + path);
    }
    return contents;
}

}  // namespace cuddl_bench
