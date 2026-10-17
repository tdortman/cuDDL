#pragma once

// Stand-ins for the Python C API calls the library makes when it runs outside Python: no
// interrupt is ever pending, and warnings go to stderr.

#include <iostream>
#include <stdexcept>

#define PyExc_RuntimeWarning nullptr

inline int PyErr_CheckSignals() {
    return 0;
}

inline int PyErr_WarnEx(void const*, char const* message, int) {
    std::cerr << "warning: " << message << std::endl;
    return 0;
}

namespace py {
struct error_already_set : std::runtime_error {
    error_already_set() : std::runtime_error("interrupted") {}
};
}  // namespace py
