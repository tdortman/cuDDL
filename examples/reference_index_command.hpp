#pragma once

#include <cuddl/reference_database_file.cuh>
#include <filesystem>
#include <vector>

struct reference_index_command {
    cuddl::reference_database_file const& database;
    std::filesystem::path index_path;
    std::vector<std::filesystem::path> const& queries;
    uint32_t minimum_matches;
    unsigned workers;
    bool build;
    cuddl::index_storage storage;
};

void dispatch_reference_index_2048(reference_index_command const& command);
void dispatch_reference_index_4096(reference_index_command const& command);
void dispatch_reference_index_8192(reference_index_command const& command);
void dispatch_reference_index_16384(reference_index_command const& command);
void dispatch_reference_index_32768(reference_index_command const& command);
void dispatch_reference_index_65536(reference_index_command const& command);
void dispatch_reference_index_131072(reference_index_command const& command);
