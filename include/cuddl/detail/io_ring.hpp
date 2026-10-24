#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <memory>

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
    #include <linux/io_uring.h>
    #include <sys/mman.h>
    #include <sys/syscall.h>
    #include <unistd.h>
#endif

namespace cuddl::detail {

#if defined(__linux__) && __has_include(<linux/io_uring.h>) && defined(__NR_io_uring_setup)
/// @brief A minimal io_uring instance: queue writes, submit them, reap their completions.
///
/// Calls the kernel directly, so the build needs no liburing. One thread owns a ring.
class io_ring {
   public:
    io_ring(io_ring const&) = delete;
    io_ring& operator=(io_ring const&) = delete;
    ~io_ring() {
        if (sqes_ != nullptr) ::munmap(sqes_, sqes_bytes_);
        if (cq_ != nullptr) ::munmap(cq_, cq_bytes_);
        if (sq_ != nullptr) ::munmap(sq_, sq_bytes_);
        ::close(fd_);
    }

    /// @brief A ring for @p depth requests in flight, or null where the kernel has no io_uring
    /// or forbids it (RHEL 8's 4.18 kernel, `kernel.io_uring_disabled`, a seccomp filter).
    [[nodiscard]] static std::unique_ptr<io_ring> make(unsigned depth) {
        io_uring_params params{};
        int const fd = static_cast<int>(::syscall(__NR_io_uring_setup, depth, &params));
        if (fd < 0) return nullptr;
        std::unique_ptr<io_ring> ring(new io_ring(fd));
        if (!ring->map(params)) return nullptr;
        return ring;
    }

    /// @brief Queues a write of @p bytes from @p data at @p offset of @p fd. The caller keeps no
    /// more requests in flight than the depth it made the ring with.
    void write(int fd, void const* data, uint32_t bytes, uint64_t offset, uint64_t user_data) {
        auto const tail = std::atomic_ref(*sq_tail_).load(std::memory_order_relaxed);
        auto const index = tail & sq_mask_;
        auto& sqe = sqes_[index];
        sqe = io_uring_sqe{};
        sqe.opcode = IORING_OP_WRITE;
        sqe.fd = fd;
        sqe.addr = reinterpret_cast<uint64_t>(data);
        sqe.len = bytes;
        sqe.off = offset;
        sqe.user_data = user_data;
        sq_array_[index] = index;
        std::atomic_ref(*sq_tail_).store(tail + 1, std::memory_order_release);
        ++queued_;
    }

    /// @brief Submits the queued requests, waits until at least @p wait_for have completed, and
    /// calls @p on_complete(user_data, result) for every completion there is.
    /// @return 0, or the errno of a failed `io_uring_enter`.
    template <typename OnComplete>
    [[nodiscard]] int submit_and_wait(unsigned wait_for, OnComplete&& on_complete) {
        unsigned seen = 0;
        while (true) {
            seen += reap(on_complete);
            if (queued_ == 0 && seen >= wait_for) return 0;
            auto const waiting = seen >= wait_for ? 0U : 1U;
            auto const entered = ::syscall(
                __NR_io_uring_enter, fd_, queued_, waiting, IORING_ENTER_GETEVENTS, nullptr, 0
            );
            if (entered < 0) {
                // A full completion queue refuses submissions until completions are reaped.
                if (errno == EINTR || errno == EAGAIN || errno == EBUSY) continue;
                return errno;
            }
            queued_ -= static_cast<unsigned>(entered);
        }
    }

   private:
    explicit io_ring(int fd) : fd_(fd) {}

    [[nodiscard]] bool map(io_uring_params const& params) {
        auto const region = [&](size_t bytes, off_t offset) -> void* {
            void* const mapped = ::mmap(
                nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, offset
            );
            return mapped == MAP_FAILED ? nullptr : mapped;
        };
        sq_bytes_ = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
        cq_bytes_ = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        sqes_bytes_ = params.sq_entries * sizeof(io_uring_sqe);
        sq_ = static_cast<char*>(region(sq_bytes_, IORING_OFF_SQ_RING));
        cq_ = static_cast<char*>(region(cq_bytes_, IORING_OFF_CQ_RING));
        sqes_ = static_cast<io_uring_sqe*>(region(sqes_bytes_, IORING_OFF_SQES));
        if (sq_ == nullptr || cq_ == nullptr || sqes_ == nullptr) return false;
        sq_tail_ = reinterpret_cast<uint32_t*>(sq_ + params.sq_off.tail);
        sq_mask_ = *reinterpret_cast<uint32_t*>(sq_ + params.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<uint32_t*>(sq_ + params.sq_off.array);
        cq_head_ = reinterpret_cast<uint32_t*>(cq_ + params.cq_off.head);
        cq_tail_ = reinterpret_cast<uint32_t*>(cq_ + params.cq_off.tail);
        cq_mask_ = *reinterpret_cast<uint32_t*>(cq_ + params.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe*>(cq_ + params.cq_off.cqes);
        return true;
    }

    template <typename OnComplete>
    unsigned reap(OnComplete& on_complete) {
        auto head = std::atomic_ref(*cq_head_).load(std::memory_order_relaxed);
        auto const tail = std::atomic_ref(*cq_tail_).load(std::memory_order_acquire);
        unsigned const count = tail - head;
        for (; head != tail; ++head) {
            auto const& cqe = cqes_[head & cq_mask_];
            on_complete(cqe.user_data, cqe.res);
        }
        std::atomic_ref(*cq_head_).store(head, std::memory_order_release);
        return count;
    }

    int fd_;
    char* sq_ = nullptr;
    char* cq_ = nullptr;
    io_uring_sqe* sqes_ = nullptr;
    size_t sq_bytes_ = 0, cq_bytes_ = 0, sqes_bytes_ = 0;
    uint32_t* sq_tail_ = nullptr;
    uint32_t* sq_array_ = nullptr;
    uint32_t* cq_head_ = nullptr;
    uint32_t* cq_tail_ = nullptr;
    io_uring_cqe* cqes_ = nullptr;
    uint32_t sq_mask_ = 0, cq_mask_ = 0;
    unsigned queued_ = 0;
};
#else
/// @brief Stand-in where the build has no io_uring headers: never available.
class io_ring {
   public:
    [[nodiscard]] static std::unique_ptr<io_ring> make(unsigned) {
        return nullptr;
    }
    void write(int, void const*, uint32_t, uint64_t, uint64_t) {}
    template <typename OnComplete>
    [[nodiscard]] int submit_and_wait(unsigned, OnComplete&&) {
        return ENOSYS;
    }
};
#endif

}  // namespace cuddl::detail
