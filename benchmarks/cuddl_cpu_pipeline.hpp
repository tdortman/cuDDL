// Stripe pipeline for the compact (sparse and edges) output of the CPU all-to-all comparator. A
// stripe is a band of query rows. Its life is
//   compute   comparison tiles fill a scratch buffer (parallel over tiles),
//   prefix    the per-row byte offsets follow from the non-zero counts,
//   encode    parallel chunks compact the scratch buffer into a writer slot,
//   enqueue   the slot goes to the writer, in stripe order.
// One pool of threads works on whatever is claimable, so the encode chunks of stripe s overlap the
// tiles of stripe s + 1 and no thread waits at a phase barrier. Scratch buffers are reused after
// their stripe's encode; writer slots are taken in stripe order, which rules out a deadlock between
// a younger stripe holding the last slot and an older stripe waiting for one.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace cpu_pipeline {

/// Thread-seconds by activity, plus the time stripes spent waiting for a writer slot.
struct Profile {
    uint64_t stripes = 0;
    double compute_s = 0.0;
    double encode_s = 0.0;
    /// Waiting for a free scratch buffer, a free writer slot, or the last tiles of the final
    /// stripe.
    double idle_buffer_s = 0.0;
    double idle_slot_s = 0.0;
    double idle_tail_s = 0.0;
    /// Hooks that run under the pipeline lock (prepare, prefix, slot setup, finish).
    double serial_s = 0.0;
    /// Sum over stripes of the time between the end of compute and a writer slot being granted.
    double slot_wait_s = 0.0;
    uint64_t slot_waits = 0;
    double wall_s = 0.0;
    uint32_t threads = 0;
};

class StripePipeline {
   public:
    struct Hooks {
        /// Under the lock: stripe s takes `buffer`; returns its number of compute tiles.
        std::function<size_t(size_t stripe, uint32_t buffer)> prepare;
        /// Under the lock: every tile of the stripe is done.
        std::function<void(size_t stripe, uint32_t buffer)> finish_compute;
        /// Under the lock: the stripe got `slot`; returns its number of encode chunks.
        std::function<uint32_t(size_t stripe, uint32_t buffer, void* slot)> begin_encode;
        /// Without the lock, once per chunk.
        std::function<void(size_t stripe, uint32_t buffer, void* slot, uint32_t chunk)> encode;
        /// Under the lock: every chunk is done; returns the bytes to write.
        std::function<size_t(size_t stripe, uint32_t buffer, void* slot)> finish_encode;
        /// Without blocking: a free writer slot or nullptr.
        std::function<void*()> try_acquire_slot;
        /// Hands a finished slot to the writer; called in stripe order.
        std::function<void(void* slot, size_t bytes)> enqueue;
    };

    StripePipeline(size_t stripes, uint32_t buffers, Hooks hooks)
        : buffers_(buffers), hooks_(std::move(hooks)), stripes_(stripes) {
        profile_.stripes = stripes;
    }

    /// Runs the calling thread as a worker until every stripe is enqueued. Every thread of a team
    /// calls it; `compute(stripe, buffer, tile)` runs one comparison tile.
    template <class Compute>
    void work(Compute&& compute) {
        using Clock = std::chrono::steady_clock;
        double compute_s = 0.0, encode_s = 0.0;
        double idle[3] = {0.0, 0.0, 0.0};
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            Reason reason = Reason::Tail;
            Task const task = claim(reason);
            if (task.kind == Kind::Finished) {
                break;
            }
            if (task.kind == Kind::None) {
                auto const t0 = Clock::now();
                cv_.wait_for(lock, std::chrono::microseconds(200));
                idle[static_cast<size_t>(reason)] += seconds(t0, Clock::now());
                continue;
            }
            lock.unlock();
            auto const t0 = Clock::now();
            if (task.kind == Kind::Compute) {
                compute(task.stripe, task.buffer, task.index);
            } else {
                hooks_.encode(
                    task.stripe,
                    task.buffer,
                    stripes_[task.stripe].slot,
                    static_cast<uint32_t>(task.index)
                );
            }
            auto const t1 = Clock::now();
            lock.lock();
            if (task.kind == Kind::Compute) {
                compute_s += seconds(t0, t1);
                complete_tile(task.stripe);
            } else {
                encode_s += seconds(t0, t1);
                complete_chunk(task.stripe);
            }
        }
        profile_.compute_s += compute_s;
        profile_.encode_s += encode_s;
        profile_.idle_buffer_s += idle[static_cast<size_t>(Reason::Buffer)];
        profile_.idle_slot_s += idle[static_cast<size_t>(Reason::Slot)];
        profile_.idle_tail_s += idle[static_cast<size_t>(Reason::Tail)];
        cv_.notify_all();
    }

    [[nodiscard]] Profile const& profile() const noexcept {
        return profile_;
    }

   private:
    enum class Stage { Pending, Compute, AwaitSlot, Encode, Done };
    enum class Kind { None, Compute, Encode, Finished };
    enum class Reason { Buffer = 0, Slot = 1, Tail = 2 };
    using Clock = std::chrono::steady_clock;

    struct Stripe {
        Stage stage = Stage::Pending;
        uint32_t buffer = 0;
        size_t tile_count = 0;
        size_t next_tile = 0;
        size_t tiles_done = 0;
        uint32_t chunk_count = 0;
        uint32_t next_chunk = 0;
        uint32_t chunks_done = 0;
        void* slot = nullptr;
        size_t bytes = 0;
        Clock::time_point compute_end{};
    };

    struct Task {
        Kind kind = Kind::None;
        size_t stripe = 0;
        uint32_t buffer = 0;
        size_t index = 0;
    };

    static double seconds(Clock::time_point from, Clock::time_point to) noexcept {
        return std::chrono::duration<double>(to - from).count();
    }

    /// Runs a hook under the lock and books its time as serial.
    template <class F>
    auto serial(F&& f) {
        auto const t0 = Clock::now();
        if constexpr (std::is_void_v<decltype(f())>) {
            f();
            profile_.serial_s += seconds(t0, Clock::now());
        } else {
            auto result = f();
            profile_.serial_s += seconds(t0, Clock::now());
            return result;
        }
    }

    Task claim(Reason& reason) {
        grant_slots();
        for (size_t s = next_enqueue_; s < activated_; ++s) {
            Stripe& st = stripes_[s];
            if (st.stage == Stage::Encode && st.next_chunk < st.chunk_count) {
                return {Kind::Encode, s, st.buffer, st.next_chunk++};
            }
        }
        for (;;) {
            if (activated_ > 0) {
                Stripe& st = stripes_[activated_ - 1U];
                if (st.stage == Stage::Compute && st.next_tile < st.tile_count) {
                    return {Kind::Compute, activated_ - 1U, st.buffer, st.next_tile++};
                }
            }
            if (next_enqueue_ == stripes_.size()) {
                return {Kind::Finished, 0, 0, 0};
            }
            if (activated_ == stripes_.size()) {
                reason = slot_blocked_ ? Reason::Slot : Reason::Tail;
                return {};
            }
            if (activated_ >= buffers_ && stripes_[activated_ - buffers_].stage < Stage::Done) {
                reason = slot_blocked_ ? Reason::Slot : Reason::Buffer;
                return {};
            }
            activate(activated_++);
            grant_slots();
        }
    }

    void activate(size_t s) {
        Stripe& st = stripes_[s];
        st.buffer = static_cast<uint32_t>(s % buffers_);
        st.tile_count = serial([&] { return hooks_.prepare(s, st.buffer); });
        st.stage = Stage::Compute;
        if (st.tile_count == 0U) {
            finish_compute(s);
        }
    }

    void finish_compute(size_t s) {
        Stripe& st = stripes_[s];
        serial([&] { hooks_.finish_compute(s, st.buffer); });
        st.stage = Stage::AwaitSlot;
        st.compute_end = Clock::now();
        cv_.notify_all();
    }

    void complete_tile(size_t s) {
        Stripe& st = stripes_[s];
        if (++st.tiles_done == st.tile_count) {
            finish_compute(s);
        }
    }

    /// Slots go to stripes in order, each once its compute is done.
    void grant_slots() {
        slot_blocked_ = false;
        while (next_slot_ < activated_ && stripes_[next_slot_].stage == Stage::AwaitSlot) {
            Stripe& st = stripes_[next_slot_];
            void* const slot = hooks_.try_acquire_slot();
            if (slot == nullptr) {
                slot_blocked_ = true;
                return;
            }
            st.slot = slot;
            profile_.slot_wait_s += seconds(st.compute_end, Clock::now());
            profile_.slot_waits += 1;
            st.chunk_count =
                serial([&] { return hooks_.begin_encode(next_slot_, st.buffer, st.slot); });
            st.stage = Stage::Encode;
            size_t const s = next_slot_++;
            if (st.chunk_count == 0U) {
                finish_encode(s);
            }
            cv_.notify_all();
        }
    }

    void complete_chunk(size_t s) {
        Stripe& st = stripes_[s];
        if (++st.chunks_done == st.chunk_count) {
            finish_encode(s);
        }
    }

    void finish_encode(size_t s) {
        Stripe& st = stripes_[s];
        st.bytes = serial([&] { return hooks_.finish_encode(s, st.buffer, st.slot); });
        st.stage = Stage::Done;
        while (next_enqueue_ < activated_ && stripes_[next_enqueue_].stage == Stage::Done) {
            hooks_.enqueue(stripes_[next_enqueue_].slot, stripes_[next_enqueue_].bytes);
            ++next_enqueue_;
        }
        cv_.notify_all();
    }

    uint32_t const buffers_;
    Hooks hooks_;
    std::vector<Stripe> stripes_;
    std::mutex mutex_;
    std::condition_variable cv_;
    /// Stripes [0, activated_) have a buffer; [0, next_slot_) have a slot; [0, next_enqueue_) are
    /// with the writer.
    size_t activated_ = 0;
    size_t next_slot_ = 0;
    size_t next_enqueue_ = 0;
    bool slot_blocked_ = false;
    Profile profile_;
};

}  // namespace cpu_pipeline
