// Optional LD_PRELOAD diagnostic, never used for the official timing samples.
// Samples 1/64 pthread_mutex_lock calls per thread/call site. Times include
// uncontended acquisition and scheduling delay, not just time asleep in futex.
// Compile: g++ -O2 -std=c++17 -shared -fPIC profile_mutex.cpp -ldl -o profile_mutex.so
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>

namespace {
constexpr unsigned max_threads = 256, max_sites = 256;
struct Site {
    uintptr_t caller = 0;
    std::atomic<uint64_t> calls{0}, samples{0}, ns{0};
};
struct alignas(64) Thread {
    Site sites[max_sites];
};
Thread threads[max_threads];
std::atomic<unsigned> next_thread{0};
std::atomic<bool> collecting{true};
std::atomic<uint64_t> omitted_calls{0};
thread_local unsigned thread_id = max_threads;
using Lock = int (*)(pthread_mutex_t *);
Lock real_lock = nullptr;

uint64_t now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

__attribute__((constructor)) void init() {
    real_lock = reinterpret_cast<Lock>(dlsym(RTLD_NEXT, "pthread_mutex_lock"));
}

__attribute__((destructor)) void report() {
    collecting.store(false, std::memory_order_relaxed);
    const char *path = std::getenv("AMAZON_MUTEX_PROFILE");
    if (!path) return;
    FILE *out = std::fopen(path, "w");
    if (!out) return;
    std::fprintf(out, "thread,module,offset,calls,samples,sampled_acquire_ns\n");
    const unsigned count = next_thread.load() < max_threads ? next_thread.load() : max_threads;
    for (unsigned i = 0; i < count; ++i) {
        for (const Site &s : threads[i].sites) {
            const auto calls = s.calls.load(std::memory_order_relaxed);
            if (!calls) continue;
            Dl_info info{};
            dladdr(reinterpret_cast<void *>(s.caller), &info);
            const auto offset = s.caller - reinterpret_cast<uintptr_t>(info.dli_fbase);
            std::fprintf(out, "%u,%s,0x%lx,%llu,%llu,%llu\n", i,
                         info.dli_fname ? info.dli_fname : "unknown", (unsigned long)offset,
                         (unsigned long long)calls,
                         (unsigned long long)s.samples.load(std::memory_order_relaxed),
                         (unsigned long long)s.ns.load(std::memory_order_relaxed));
        }
    }
    std::fclose(out);
    std::fprintf(stderr, "[mutex_profile] omitted_calls=%llu max_sites_per_thread=%u\n",
                 (unsigned long long)omitted_calls.load(), max_sites);
}
}

extern "C" int pthread_mutex_lock(pthread_mutex_t *mutex) {
    // dlsym on this glibc does not recurse into the public mutex entry point.
    if (!real_lock) real_lock = reinterpret_cast<Lock>(dlsym(RTLD_NEXT, "pthread_mutex_lock"));
    if (!collecting.load(std::memory_order_relaxed)) return real_lock(mutex);
    if (thread_id == max_threads) thread_id = next_thread.fetch_add(1);
    if (thread_id >= max_threads) {
        omitted_calls.fetch_add(1, std::memory_order_relaxed);
        return real_lock(mutex);
    }
    const auto caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    Site *site = nullptr;
    for (auto &s : threads[thread_id].sites) {
        if (s.caller == caller || s.caller == 0) {
            s.caller = caller;
            site = &s;
            break;
        }
    }
    if (!site) {
        omitted_calls.fetch_add(1, std::memory_order_relaxed);
        return real_lock(mutex);
    }
    const uint64_t calls = site->calls.load(std::memory_order_relaxed) + 1;
    site->calls.store(calls, std::memory_order_relaxed);
    // Stagger samples between threads rather than timing the first lock on all.
    if ((calls + thread_id) % 64 != 0) return real_lock(mutex);
    const auto start = now();
    const int result = real_lock(mutex);
    const auto elapsed = now() - start;
    site->samples.store(site->samples.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    site->ns.store(site->ns.load(std::memory_order_relaxed) + elapsed, std::memory_order_relaxed);
    return result;
}
