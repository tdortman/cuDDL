#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cuddl/detail/ablation.hpp>
#include <cuddl/detail/io_ring.hpp>

namespace cuddl::detail {

/// @brief Writes a file from a thread of its own, so the caller produces the next bytes while
/// earlier ones reach storage.
///
/// Bytes are copied into page-aligned memory by @ref append, or produced there in place through
/// @ref reserve, and written with `O_DIRECT` in requests of up to 8 MiB, bypassing the page
/// cache. Buffered writes stall in page-cache copies and writeback: on a Samsung 9100
/// PRO, one buffered stream plus `fsync` wrote 2.4-3.1 GB/s and `O_DIRECT` 8 MiB writes
/// 12.9-13.6 GB/s, the drive's rated 13.4 GB/s; 1 MiB writes reached only 9.1-9.7 GB/s. There
/// the copy itself, 11 GB/s on one core, held a 159 GB all-to-all output below the drive.
/// Where io_uring is available, four requests are in flight at once: on a btrfs file system
/// that wrote 2.9-4.4 GB/s against 1.9-2.6 GB/s for synchronous writes. Elsewhere one request
/// is written at a time, which on ext4 already reached the drive's rate.
///
/// File systems without `O_DIRECT` (tmpfs) and network file systems get buffered writes:
/// direct writes to Lustre wait for the server on every request (8 streams reached 28 MB/s).
class file_writer {
   public:
    static constexpr size_t block_bytes = size_t{8} << 20;
    /// Writes in flight with io_uring.
    static constexpr unsigned depth = 4;
    /// Bytes timed per route before a writer that need not be durable picks one; it leaves the
    /// page cache only after 16 such windows through it ran too slow.
    static constexpr size_t probe_window = size_t{64} << 20;

    /// @param durable Whether @ref finish flushes the file to storage before returning.
    /// @param queued_limit Bytes reserved or appended but not yet written before @ref reserve
    /// and @ref append wait; a larger reservation waits only for all but one other.
    file_writer(
        std::filesystem::path const& path,
        bool durable,
        size_t queued_limit = size_t{1} << 30
    )
        : path_(path), queued_limit_(queued_limit), durable_(durable) {
        int const flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC;
        fd_ = ablated("direct-io") ? -1 : ::open(path.c_str(), flags | O_DIRECT, 0644);
        direct_ = fd_ != -1;
        if (fd_ == -1) fd_ = ::open(path.c_str(), flags, 0644);
        if (fd_ == -1) fail("cannot write");
        struct statfs status{};
        // shortcut: NFS and Lustre only, the network file systems we run on; add others as met.
        if (direct_ && ::fstatfs(fd_, &status) == 0 &&
            (status.f_type == 0x6969 || status.f_type == 0x0BD00BD0)) {
            if (::fcntl(fd_, F_SETFL, ::fcntl(fd_, F_GETFL) & ~O_DIRECT) == -1) {
                ::close(fd_);
                fail("cannot write");
            }
            direct_ = false;
        }
        // Buffered writes through io_uring would run concurrently on kernel workers, and several
        // writers into one Lustre file were far slower than one.
        if (direct_ && !ablated("io-uring")) ring_ = io_ring::make(depth);
        // A drive slower than the page cache leaves a run that need not be durable waiting on
        // direct writes the cache could absorb: 159 GB to a PM9A3 took 108-158 s direct and
        // 60-97 s buffered. Such writers time both routes and keep the faster; see work().
        if (direct_ && !durable_ && !ablated("write-probe")) {
            buffered_fd_ = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
            if (buffered_fd_ == -1) fail("cannot write");
            // btrfs compresses page-cache writes during writeback, never direct ones; with zstd
            // a buffered 120 GB run took 58.5 s compressed and 45.8 s not. Other file systems
            // refuse or ignore the flag.
            int attributes = 0;
            if (::ioctl(fd_, FS_IOC_GETFLAGS, &attributes) == 0) {
                attributes |= FS_NOCOMP_FL;
                static_cast<void>(::ioctl(fd_, FS_IOC_SETFLAGS, &attributes));
            }
        }
        thread_ = std::thread([this] { work(); });
    }
    file_writer(file_writer const&) = delete;
    file_writer& operator=(file_writer const&) = delete;
    /// Stops writing without flushing; @ref finish reports whether every byte arrived.
    ~file_writer() {
        stop();
        if (fd_ != -1) ::close(fd_);
        if (buffered_fd_ != -1) ::close(buffered_fd_);
    }

    /// @brief Space for the next @p bytes of the file, for the caller to fill and then hand
    /// back with @ref commit, sparing @ref append's copy. Waits while the queued bytes reach the
    /// limit; throws if an earlier write failed. One reservation at a time.
    std::span<char> reserve(size_t bytes) {
        std::unique_lock lock(append_mutex_);
        auto const total = carry_bytes_ + bytes;
        current_ = take_extent((total + 4095) / 4096 * 4096);
        std::memcpy(current_->data.get(), carry_.data(), carry_bytes_);
        reserved_ = bytes;
        reservation_ = std::move(lock);
        return {current_->data.get() + carry_bytes_, bytes};
    }

    /// @brief Queues the bytes of the last @ref reserve. A direct write covers whole sectors, so
    /// the last partial sector waits for the next bytes.
    void commit() {
        queue_current(false);
        reservation_.unlock();
    }

    /// @brief Copies @p chunks, which reach the file together and in order; waits while the
    /// queued bytes reach the limit. Throws if an earlier write failed.
    void append(std::initializer_list<std::span<char const>> chunks) {
        size_t bytes = 0;
        for (auto const& chunk : chunks) bytes += chunk.size();
        auto* out = reserve(bytes).data();
        for (auto const& chunk : chunks) {
            if (!chunk.empty()) std::memcpy(out, chunk.data(), chunk.size());
            out += chunk.size();
        }
        commit();
    }

    void append(std::span<char const> bytes) {
        append({bytes});
    }

    /// @brief Writes everything appended and closes the file, flushing it to storage first if
    /// durable. Bytes left in the page cache otherwise reach storage after the caller moves on,
    /// as with any buffered write.
    void finish() {
        uint64_t size = 0;
        {
            reserve(0);
            size = offset_ + carry_bytes_;
            queue_current(true);
            reservation_.unlock();
        }
        {
            std::unique_lock lock(mutex_);
            space_.wait(lock, [&] { return failed_ != 0 || (queue_.empty() && in_flight_ == 0); });
        }
        stop();
        int error = failed_;
        if (error == 0 && direct_ && ::ftruncate(fd_, static_cast<off_t>(size)) != 0) error = errno;
        if (error == 0 && durable_ && ::fsync(fd_) != 0) error = errno;
        if (::close(fd_) != 0 && error == 0) error = errno;
        fd_ = -1;
        if (buffered_fd_ != -1 && ::close(buffered_fd_) != 0 && error == 0) error = errno;
        buffered_fd_ = -1;
        if (error != 0) fail("cannot write", error);
    }

    /// @brief Whether writes bypass the page cache.
    [[nodiscard]] bool direct() const noexcept {
        return direct_;
    }

    /// @brief Whether writes go through io_uring.
    [[nodiscard]] bool asynchronous() const noexcept {
        return ring_ != nullptr;
    }

   private:
    struct free_deleter {
        void operator()(char* data) const noexcept {
            std::free(data);
        }
    };

    /// Memory the producer fills and the writer writes from, reused once its pieces are written.
    struct extent {
        std::unique_ptr<char, free_deleter> data;
        size_t capacity = 0;
        unsigned pending = 0;  // queued pieces not yet written
    };

    /// At most block_bytes of an extent, written with one request.
    struct queued_block {
        extent* source = nullptr;
        char const* data = nullptr;
        size_t bytes = 0;
        uint64_t offset = 0;
    };

    [[noreturn]] void fail(char const* what, int error = errno) const {
        throw std::system_error(
            error, std::generic_category(), std::string(what) + " " + path_.string()
        );
    }

    /// An extent of at least @p bytes, reused if a free one is large enough. Waits while it
    /// would take the queued bytes past the limit and another extent is still being written, so
    /// even reservations past the limit fill while the previous one is written.
    extent* take_extent(size_t bytes) {
        std::unique_lock lock(mutex_);
        space_.wait(lock, [&] {
            return failed_ != 0 || extents_.size() - free_.size() < 2 ||
                   queued_ + bytes <= queued_limit_;
        });
        if (failed_ != 0) fail("cannot write", failed_);
        auto const fits =
            std::ranges::find_if(free_, [&](extent* e) { return e->capacity >= bytes; });
        extent* chosen = nullptr;
        if (fits != free_.end()) {
            chosen = *fits;
            free_.erase(fits);
        } else {
            // Too small for anything to come, so its memory goes to a larger extent.
            if (!free_.empty()) {
                auto* const smallest = free_.front();
                free_.erase(free_.begin());
                std::erase_if(extents_, [&](auto const& e) { return e.get() == smallest; });
            }
            auto fresh = std::make_unique<extent>();
            // O_DIRECT needs memory aligned to the logical sector, 4096 bytes at most.
            fresh->capacity =
                (std::max<size_t>(bytes, 1) + block_bytes - 1) / block_bytes * block_bytes;
            fresh->data.reset(static_cast<char*>(std::aligned_alloc(4096, fresh->capacity)));
            if (!fresh->data) throw std::bad_alloc();
            chosen = extents_.emplace_back(std::move(fresh)).get();
        }
        queued_ += chosen->capacity;
        return chosen;
    }

    /// Queues the reserved extent in pieces of at most block_bytes. Short of @p final, a direct
    /// write keeps back the last partial sector; @p final pads it with zeros instead.
    void queue_current(bool final) {
        auto const total = carry_bytes_ + reserved_;
        size_t written = total;
        if (direct_) written = final ? (total + 4095) / 4096 * 4096 : total / 4096 * 4096;
        char* const data = current_->data.get();
        std::memset(data + total, 0, written > total ? written - total : 0);
        carry_bytes_ = total - std::min(total, written);
        std::memcpy(carry_.data(), data + written, carry_bytes_);
        {
            std::lock_guard lock(mutex_);
            for (size_t at = 0; at < written; at += block_bytes) {
                queue_.push_back(
                    {current_, data + at, std::min(block_bytes, written - at), offset_ + at}
                );
                ++current_->pending;
            }
            if (current_->pending == 0) release(current_);
        }
        offset_ += written;
        current_ = nullptr;
        reserved_ = 0;
        ready_.notify_all();
    }

    /// Returns an extent with nothing left to write to the free list. Needs mutex_.
    void release(extent* done) {
        queued_ -= done->capacity;
        auto const at = std::ranges::lower_bound(free_, done->capacity, {}, &extent::capacity);
        free_.insert(at, done);
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    /// Writes @p piece with plain `pwrite`; returns the error that stopped it, or 0.
    static int write_all(int fd, queued_block const& piece) {
        size_t done = 0;
        while (done < piece.bytes) {
            auto const wrote = ::pwrite(
                fd, piece.data + done, piece.bytes - done, static_cast<off_t>(piece.offset + done)
            );
            if (wrote < 0) return errno;
            if (wrote == 0) return EIO;
            done += static_cast<size_t>(wrote);
        }
        return 0;
    }

    using clock = std::chrono::steady_clock;

    static double seconds_since(clock::time_point start) {
        return std::chrono::duration<double>(clock::now() - start).count();
    }

    /// Records a written piece, or why it was not written.
    void retire(queued_block const& written, int error) {
        {
            std::lock_guard lock(mutex_);
            if (error != 0 && failed_ == 0) failed_ = error;
            if (--written.source->pending == 0) release(written.source);
        }
        space_.notify_all();
    }

    /// Writes queued requests in order. A writer that need not be durable times direct writes
    /// over one window, then the page cache, and keeps the cache if it was faster. The cache
    /// absorbs bytes faster than a slow drive takes them until writeback throttles it, and the
    /// run ends with its last bytes still unwritten; a fast drive outpaces the copy into it from
    /// the start. Leaving the cache makes the remaining direct writes share the drive with its
    /// writeback, so the writer returns to direct writes for good only when 16 windows through
    /// the cache together ran below 3/4 of the direct rate. On a PM9A3, where throttled
    /// writeback matched direct writes within noise, switching back at the first slower window
    /// took a 159 GB run 154-162 s against 76-91 s buffered. On a btrfs file system with
    /// compression, buffered runs averaged 2.0 GB/s against 2.8-3.1 GB/s direct.
    void work() {
        std::vector<queued_block> slots(ring_ ? depth : 1U);
        std::vector<unsigned> idle;
        for (unsigned slot = 0; slot < slots.size(); ++slot) idle.push_back(slot);
        auto const window = std::min(probe_window, queued_limit_);
        bool buffered = false, settled = buffered_fd_ == -1, tried = false;
        uint64_t submitted = 0, window_bytes = 0;
        double window_seconds = 0, direct_rate = 0;
        while (true) {
            if (buffered) {
                queued_block next;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                    if (queue_.empty()) return;
                    next = queue_.front();
                    queue_.pop_front();
                    ++in_flight_;
                }
                auto const start = clock::now();
                int const error = write_all(buffered_fd_, next);
                window_seconds += seconds_since(start);
                window_bytes += next.bytes;
                {
                    std::lock_guard lock(mutex_);
                    --in_flight_;
                }
                retire(next, error);
                if (window_bytes >= (tried ? 16 : 1) * window) {
                    auto const pace = tried ? 0.75 * direct_rate : direct_rate;
                    buffered = static_cast<double>(window_bytes) >= pace * window_seconds;
                    settled = !buffered;
                    tried = true;
                    window_bytes = 0;
                    window_seconds = 0;
                }
                continue;
            }
            {
                std::unique_lock lock(mutex_);
                // With writes in flight, collect their completions before waiting for more.
                if (idle.size() == slots.size()) {
                    if (!settled && submitted >= window) {
                        direct_rate = static_cast<double>(window_bytes) / window_seconds;
                        buffered = true;
                        window_bytes = 0;
                        window_seconds = 0;
                        continue;
                    }
                    ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                    if (queue_.empty()) return;
                }
                for (; !idle.empty() && !queue_.empty() && (settled || submitted < window);
                     idle.pop_back()) {
                    slots[idle.back()] = queue_.front();
                    queue_.pop_front();
                    ++in_flight_;
                    submitted += slots[idle.back()].bytes;
                    if (ring_) {
                        auto const& next = slots[idle.back()];
                        ring_->write(
                            fd_,
                            next.data,
                            static_cast<uint32_t>(next.bytes),
                            next.offset,
                            idle.back()
                        );
                    }
                }
            }
            if (idle.size() == slots.size()) continue;
            auto const start = clock::now();
            if (!ring_) {
                int const error = write_all(fd_, slots[0]);
                window_seconds += seconds_since(start);
                window_bytes += slots[0].bytes;
                finish_slot(slots, idle, 0, error);
                continue;
            }
            int const error = ring_->submit_and_wait(1, [&](uint64_t slot, int result) {
                auto const expected = static_cast<int64_t>(slots[slot].bytes);
                window_bytes += slots[slot].bytes;
                // A short direct write means the device ran out of space or failed.
                finish_slot(
                    slots,
                    idle,
                    static_cast<unsigned>(slot),
                    result < 0 ? -result : (result == expected ? 0 : EIO)
                );
            });
            window_seconds += seconds_since(start);
            if (error != 0) {
                // The kernel may still read the extents in flight, so they are never freed.
                for (auto& extent : extents_) static_cast<void>(extent->data.release());
                std::lock_guard lock(mutex_);
                failed_ = error;
                space_.notify_all();
                return;
            }
        }
    }

    void finish_slot(
        std::vector<queued_block>& slots,
        std::vector<unsigned>& idle,
        unsigned slot,
        int error
    ) {
        {
            std::lock_guard lock(mutex_);
            --in_flight_;
        }
        retire(slots[slot], error);
        idle.push_back(slot);
    }

    std::filesystem::path path_;
    size_t const queued_limit_;
    bool const durable_;
    int fd_ = -1;
    /// The same file without O_DIRECT, for a writer that need not be durable.
    int buffered_fd_ = -1;
    bool direct_ = false;
    std::unique_ptr<io_ring> ring_;

    // Producer state, guarded by append_mutex_, which a reservation holds until its commit.
    std::mutex append_mutex_;
    std::unique_lock<std::mutex> reservation_;
    extent* current_ = nullptr;
    size_t reserved_ = 0;
    /// File offset of the next queued byte, a sector boundary for direct writes.
    uint64_t offset_ = 0;
    /// Bytes after offset_ that wait for the rest of their sector.
    std::array<char, 4096> carry_{};
    size_t carry_bytes_ = 0;

    std::mutex mutex_;
    std::condition_variable ready_, space_;
    std::deque<queued_block> queue_;
    std::vector<std::unique_ptr<extent>> extents_;
    /// Extents with nothing to write, by capacity.
    std::vector<extent*> free_;
    /// Capacity of the extents in use.
    size_t queued_ = 0;
    unsigned in_flight_ = 0;
    int failed_ = 0;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace cuddl::detail
