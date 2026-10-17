#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cuddl::detail {

/// @brief Fixed threads that run indexed work with the caller, for work repeated per batch.
///
/// Spawning and joining threads for every batch cost the thread submitting batches more time
/// than the reads it spread over them.
class worker_pool {
   public:
    explicit worker_pool(size_t workers) {
        auto const helpers = workers > 0 ? workers - 1 : 0;
        threads_.reserve(helpers);
        for (size_t thread = 0; thread < helpers; ++thread) {
            threads_.emplace_back([this] { serve(); });
        }
    }
    worker_pool(worker_pool const&) = delete;
    worker_pool& operator=(worker_pool const&) = delete;
    ~worker_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    /// @brief Runs @p work(index) for every index below @p count; one call at a time.
    template <typename Work>
    void for_each(size_t count, Work&& work) {
        if (count == 0) return;
        auto run = [&](size_t index) {
            work(index);
        };
        {
            std::lock_guard lock(mutex_);
            job_ = run;
            count_ = count;
            next_ = 0;
            active_ = threads_.size();
            ++generation_;
        }
        start_.notify_all();
        drain();
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return active_ == 0; });
        job_ = nullptr;
    }

   private:
    void drain() {
        for (size_t index = next_++; index < count_; index = next_++) job_(index);
    }

    void serve() {
        size_t seen = 0;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            drain();
            std::lock_guard lock(mutex_);
            if (--active_ == 0) done_.notify_one();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    std::function<void(size_t)> job_;
    std::atomic<size_t> next_{0};
    size_t count_ = 0, active_ = 0, generation_ = 0;
    bool stop_ = false;
};

}  // namespace cuddl::detail
