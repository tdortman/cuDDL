#include "result_json.hpp"

#include <gtest/gtest.h>

#include <sstream>

TEST(ResultJsonTest, DetectsCpuModelAcrossProcfsFormats) {
    std::istringstream x86("processor : 0\nmodel name : AMD Ryzen 9 5900X\n");
    EXPECT_EQ(cpu_model(x86, "x86_64"), "AMD Ryzen 9 5900X");

    std::istringstream arm(
        "processor : 0\nCPU implementer : 0x41\nCPU part : 0xd4f\n"
    );
    EXPECT_EQ(cpu_model(arm, "aarch64"), "ARM implementer 0x41 part 0xd4f");

    std::istringstream unknown("processor : 0\nBogoMIPS : 2000.00\n");
    EXPECT_EQ(cpu_model(unknown, "aarch64"), "aarch64");
}
