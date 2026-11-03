#pragma once

#include <cuddl/detail/ablation.hpp>

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#if defined(__linux__)
    #include <fcntl.h>
    #include <sys/file.h>
    #include <unistd.h>
#endif

namespace cuddl::detail {

/// @brief Lock file for the PCIe root port above @p device, or empty where sysfs does not show
/// one. GPUs behind one root port share its uplink to host memory.
inline std::string uplink_lock_path(int device) {
    int domain = 0, bus = 0, slot = 0;
    if (cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&slot, cudaDevAttrPciDeviceId, device) != cudaSuccess) {
        return {};
    }
    char address[32];
    std::snprintf(address, sizeof address, "%04x:%02x:%02x.0", domain, bus, slot);
    std::error_code error;
    // /sys/devices/pci<domain>:<bus>/<root port>/.../<device>
    auto const path =
        std::filesystem::canonical(std::filesystem::path{"/sys/bus/pci/devices"} / address, error);
    if (error) return {};
    auto part = path.begin();
    for (auto const* expected : {"/", "sys", "devices"}) {
        if (part == path.end() || *part != expected) return {};
        ++part;
    }
    if (part == path.end() || ++part == path.end()) return {};
    return "/dev/shm/cuddl-uplink-" + part->string();
}

/**
 * @brief Held while a device's results cross PCIe to the host, so the GPUs behind one root port
 * download one at a time, across threads and processes alike.
 *
 * Downloading at once can cost a shared switch most of its uplink. On an 8-GPU A100 node, where
 * pairs of GPUs share four switches, one GPU downloaded 25.9 GB/s alone and two behind one switch
 * 12.0 GB/s together; taking turns, the two moved 25.4-25.8 GB/s and all eight 101.7-103.3 GB/s,
 * against 45.5-50.9 GB/s at once. Turns cost nothing on a switch that shares fairly, as on an A40
 * node whose pairs downloaded 26.4 GB/s together: the uplink caps a pair either way.
 *
 * The lock is an `flock` on a file in /dev/shm named after the root port. Without one, downloads
 * run at once: slower behind such a switch, never wrong.
 */
class uplink_lock {
   public:
    explicit uplink_lock(int device) {
#if defined(__linux__)
        if (ablated("switch-downloads")) return;
        auto const& path = lock_path(device);
        if (path.empty()) return;
        // flock needs no write access, so a file another user created serves too.
        fd_ = ::open(path.c_str(), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ == -1) return;
        while (::flock(fd_, LOCK_EX) != 0) {
            if (errno != EINTR) {
                ::close(fd_);
                fd_ = -1;
                return;
            }
        }
#else
        static_cast<void>(device);
#endif
    }

    uplink_lock(uplink_lock const&) = delete;
    uplink_lock& operator=(uplink_lock const&) = delete;

    ~uplink_lock() {
#if defined(__linux__)
        if (fd_ != -1) ::close(fd_);  // releases the lock
#endif
    }

   private:
    static std::string const& lock_path(int device) {
        static std::mutex mutex;
        static std::map<int, std::string> paths;
        std::lock_guard lock(mutex);
        auto [entry, added] = paths.try_emplace(device);
        if (added) entry->second = uplink_lock_path(device);
        return entry->second;
    }

    int fd_ = -1;
};

}  // namespace cuddl::detail
