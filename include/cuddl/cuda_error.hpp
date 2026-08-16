#pragma once

#include <cuddl/error.hpp>

#include <stdexcept>

/// @brief Checks a CUDA runtime call and throws @c std::runtime_error on failure.
///
/// Use in non-@ref cuddl::Result functions (benchmarks, RAII helpers). Prefer @ref CUDDL_CUDA_TRY
/// inside functions that return @ref cuddl::Result.
#define CUDDL_CUDA_CALL(err)                                                      \
    do {                                                                          \
        if (cudaError_t _cuddl_err = (err); _cuddl_err != cudaSuccess) {          \
            throw std::runtime_error(::cuddl::Error::cuda(_cuddl_err).message()); \
        }                                                                         \
    } while (0)
