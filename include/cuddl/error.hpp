#pragma once

#include <cuda_runtime.h>
#include <cuda/std/expected>
#include <cuda/stream>

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace cuddl {

/// @brief File, line, and column for a C++ call site or FASTX input position.
struct SourceLocation {
    std::string file;
    uint32_t line{};
    uint32_t column{};

    [[nodiscard]] static SourceLocation from(std::source_location location) {
        return SourceLocation{
            location.file_name(),
            static_cast<uint32_t>(location.line()),
            static_cast<uint32_t>(location.column()),
        };
    }

    [[nodiscard]] std::string to_string() const {
        return std::format("{}:{}:{}", file, line, column);
    }
};

struct CudaError {
    cudaError_t code{};
    SourceLocation location;
    std::string message;
};

struct InvalidArgumentError {
    std::string message;
};

struct ResourceError {
    std::string message;
};

/// @brief Error category for @ref Result failures.
enum class ErrorCategory : uint8_t {
    cuda,
    invalid_argument,
    resource,
};

/// @brief Error payload carried in @ref Result on failure.
struct Error {
    std::variant<CudaError, InvalidArgumentError, ResourceError> kind;

    [[nodiscard]] ErrorCategory category() const noexcept {
        return static_cast<ErrorCategory>(kind.index());
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return std::visit(
            [](const auto& error) -> const std::string& { return error.message; }, kind
        );
    }

    [[nodiscard]] const CudaError* as_cuda() const noexcept {
        return std::get_if<CudaError>(&kind);
    }

    [[nodiscard]] const InvalidArgumentError* as_invalid_argument() const noexcept {
        return std::get_if<InvalidArgumentError>(&kind);
    }

    [[nodiscard]] const ResourceError* as_resource() const noexcept {
        return std::get_if<ResourceError>(&kind);
    }

    [[nodiscard]] static Error
    cuda(cudaError_t code, std::source_location location = std::source_location::current()) {
        const SourceLocation site = SourceLocation::from(location);
        return Error{CudaError{
            code,
            site,
            std::format("{}: {}", site.to_string(), cudaGetErrorString(code)),
        }};
    }

    [[nodiscard]] static Error invalid_argument(std::string message) {
        return Error{InvalidArgumentError{std::move(message)}};
    }

    [[nodiscard]] static Error resource(std::string message) {
        return Error{ResourceError{std::move(message)}};
    }
};
static_assert(static_cast<size_t>(ErrorCategory::cuda) == 0);
static_assert(static_cast<size_t>(ErrorCategory::invalid_argument) == 1);
static_assert(static_cast<size_t>(ErrorCategory::resource) == 2);

/**
 * @brief Fallible API result: @c cuda::std::expected<T, Error> with cuDDL factories.
 *
 * Layout-compatible with the base type. Inherits construction, @c operator*, @c operator->,
 * @c value, @c error, and conversion from @c cuda::std::unexpected<Error>.
 */
template <typename T>
struct [[nodiscard]] Result : cuda::std::expected<T, Error> {
    using cuda::std::expected<T, Error>::expected;

    [[nodiscard]] static Result ok(T value) {
        return Result(std::move(value));
    }

    [[nodiscard]] static Result err(Error error) {
        return Result(cuda::std::unexpect, std::move(error));
    }
};

template <>
struct [[nodiscard]] Result<void> : cuda::std::expected<void, Error> {
    using cuda::std::expected<void, Error>::expected;

    [[nodiscard]] static Result ok() {
        return {};
    }

    [[nodiscard]] static Result err(Error error) {
        return Result(cuda::std::unexpect, std::move(error));
    }
};

/// @brief Terminates the process when @p error is not @c cudaSuccess (for RAII teardown only).
inline void cuda_abort_on_error(
    cudaError_t error,
    std::source_location location = std::source_location::current()
) {
    if (error == cudaSuccess) {
        return;
    }
    const Error err = Error::cuda(error, location);
    std::fputs(err.message().c_str(), stderr);
    std::fputc('\n', stderr);
    std::abort();
}

namespace detail {

template <typename T>
[[nodiscard]] T try_unwrap_success(Result<T>& result) {
    return std::move(*result);
}

inline void try_unwrap_success(Result<void>& result) {
    (void)result;
}

/// Copies @p error for propagation (avoids moving out of @c expected::error()).
[[nodiscard]] inline cuda::std::unexpected<Error> propagate_error(const Error& error) {
    return cuda::std::unexpected<Error>(Error{error});
}

}  // namespace detail

/// @brief Failure return; converts to any @c Result<T> via @c cuda::std::unexpected.
[[nodiscard]] inline cuda::std::unexpected<Error> Err(Error error) {
    return cuda::std::unexpected<Error>(std::move(error));
}

/// @brief Success return for @c Result<void>; same as @c return {}.
[[nodiscard]] inline Result<void> Ok() noexcept {
    return {};
}

/// @brief Checks a CUDA runtime call and returns an error on failure.
[[nodiscard]] inline Result<void>
cuda_try(cudaError_t error, std::source_location location = std::source_location::current()) {
    if (error == cudaSuccess) {
        return Ok();
    }
    return Err(Error::cuda(error, location));
}

/// @brief Converts CCCL exceptions to the same Result errors as CUDA status-returning APIs.
template <std::invocable Operation>
[[nodiscard]] auto
cuda_try(Operation&& operation, std::source_location location = std::source_location::current()) {
    using Value = std::invoke_result_t<Operation>;
    using Return = Result<std::conditional_t<std::is_same_v<Value, cudaError_t>, void, Value>>;
    try {
        if constexpr (std::is_void_v<Value>) {
            std::forward<Operation>(operation)();
            return Return::ok();
        } else if constexpr (std::is_same_v<Value, cudaError_t>) {
            return cuda_try(std::forward<Operation>(operation)(), location);
        } else {
            return Return::ok(std::forward<Operation>(operation)());
        }
    } catch (cuda::cuda_error const& error) {
        return Return::err(Error::cuda(static_cast<cudaError_t>(error.status()), location));
    } catch (std::bad_alloc const& error) {
        return Return::err(Error::resource(error.what()));
    }
}

}  // namespace cuddl

/// @brief Propagates a @ref cuddl::Result failure from the enclosing function (GNU statement
/// expression).
///
/// On failure, copies the error then returns @c cuda::std::unexpected<Error> (does not move out of
/// the source @c expected). On success, yields the value for valued results, or nothing for
/// @c Result<void>. Usable as a statement (@c CUDDL_TRY(expr);) or in initializers
/// (@c auto x = CUDDL_TRY(expr);).
#define CUDDL_TRY(expr)                                                     \
    ({                                                                      \
        auto _cuddl_result = (expr);                                        \
        if (!_cuddl_result) {                                               \
            return ::cuddl::detail::propagate_error(_cuddl_result.error()); \
        }                                                                   \
        ::cuddl::detail::try_unwrap_success(_cuddl_result);                 \
    })

/// @brief Unwraps a @ref cuddl::Result or throws @c std::runtime_error on failure (tests and apps).
#define CUDDL_UNWRAP(expr)                                             \
    ({                                                                 \
        auto _cuddl_result = (expr);                                   \
        if (!_cuddl_result) {                                          \
            throw std::runtime_error(_cuddl_result.error().message()); \
        }                                                              \
        ::cuddl::detail::try_unwrap_success(_cuddl_result);            \
    })

/// @brief Checks a CUDA call and aborts on failure (destructors and RAII only).
#define CUDDL_CUDA_ABORT(expr) ::cuddl::cuda_abort_on_error((expr))

/// @brief Propagates a CUDA error wrapped in @ref cuddl::Result<void>.
#define CUDDL_CUDA_TRY(...) CUDDL_TRY(::cuddl::cuda_try([&] { return (__VA_ARGS__); }))
