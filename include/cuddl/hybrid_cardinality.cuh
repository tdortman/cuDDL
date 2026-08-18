#pragma once

namespace cuddl {

/// @brief Experimental DDL cardinality estimates computed from one register scan.
struct hybrid_cardinality_estimates {
    double bbtools;
    double paper;
    double lc;
    double dlc;
    double mean_m_raw;
};

}  // namespace cuddl
