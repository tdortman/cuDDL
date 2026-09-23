#pragma once

#include <cuddl/reference_database_file.cuh>

#include <filesystem>
#include <vector>

struct reference_index_command {
    cuddl::reference_database_file const& database;
    std::filesystem::path index_path;
    std::vector<std::filesystem::path> const& queries;
    std::filesystem::path output;
    bool all_to_all;
    uint32_t minimum_matches;
    unsigned workers;
    bool build;
    cuddl::index_storage storage;
};

/// @brief Runs @p command for one sketch configuration. Meson generates the instantiations from
/// reference_index_dispatch.cu.in, one unit per bucket count, k-mer range, and exponent width
/// (5 or 6 bits), to bound compiler memory.
template <size_t Buckets, uint32_t K, uint32_t ExponentBits>
void dispatch_reference_index(reference_index_command const& command);
