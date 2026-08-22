#pragma once

#include "nvbench/state.cuh"

template <typename T>
__forceinline__ void do_not_optimise(T& value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

void add_value(nvbench::state& state, char const* name, double value) {
    auto& summary = state.add_summary(name);
    summary.set_string("name", name);
    summary.set_float64("value", value);
}

void add_median_time(nvbench::state& state) {
    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median GPU Time", median);
}

void add_median_throughput(nvbench::state& state, size_t count) {
    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median Throughput", static_cast<double>(count) / median);
}
