#pragma once

#include <libdeflate.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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

/// @brief Decompresses every gzip member of @p compressed, growing the output as needed.
[[nodiscard]] inline std::string gunzip(std::string_view compressed, std::string const& origin) {
    std::unique_ptr<libdeflate_decompressor, decltype(&libdeflate_free_decompressor)> decoder{
        libdeflate_alloc_decompressor(), &libdeflate_free_decompressor
    };
    if (!decoder) throw std::runtime_error("cannot allocate gzip decompressor");
    std::string output(std::max<size_t>(compressed.size() * 4U, size_t{1} << 16U), '\0');
    size_t consumed = 0;
    size_t produced = 0;
    while (consumed < compressed.size()) {
        size_t member_in = 0;
        size_t member_out = 0;
        auto const status = libdeflate_gzip_decompress_ex(
            decoder.get(),
            compressed.data() + consumed,
            compressed.size() - consumed,
            output.data() + produced,
            output.size() - produced,
            &member_in,
            &member_out
        );
        if (status == LIBDEFLATE_INSUFFICIENT_SPACE) {
            output.resize(output.size() * 2U);
            continue;
        }
        if (status != LIBDEFLATE_SUCCESS) {
            throw std::runtime_error("invalid gzip data: " + origin);
        }
        consumed += member_in;
        produced += member_out;
    }
    output.resize(produced);
    return output;
}

[[nodiscard]] inline std::string inflate_bgzf_block(std::ifstream& file, bgzf_block const& block) {
    std::string compressed(block.size, '\0');
    file.seekg(static_cast<std::streamoff>(block.offset));
    file.read(compressed.data(), compressed.size());
    if (!file) {
        throw std::runtime_error("cannot read compressed BGZF block");
    }
    return gunzip(compressed, "BGZF block");
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

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open asset (plain or .gz expected): " + path);
    }
    std::string contents{std::istreambuf_iterator<char>{file}, {}};
    auto const gzip = contents.size() >= 2U && static_cast<uint8_t>(contents[0]) == 0x1fU &&
                      static_cast<uint8_t>(contents[1]) == 0x8bU;
    return gzip ? gunzip(contents, path) : contents;
}

}  // namespace cuddl_bench
