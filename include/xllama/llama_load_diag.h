// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Runtime diagnostics for the llama.cpp model-load path.
//
// Motivation: on UWP a failed llama_model_load_from_file() used to surface as
// only "failed to load model: <path>". The pinned llama.cpp (3cb7ffb) logs the
// real reason through LLAMA_LOG_ERROR, but xllama never installed a log
// callback, so on UWP that text went to stderr and died.
//
// This header:
//   - forwards llama.cpp logs into xllama.log (WARN/ERROR always; INFO/CONT
//     only while a LoadWindow is open around the load call),
//   - captures the loader's WARN/ERROR lines of the current load so the caller
//     can put the PRECISE reason into its error string,
//   - checks the model file on disk (size, open result, GGUF magic) before the
//     load, using only I/O already proven under AppContainer in this codebase
//     (std::filesystem + _wfopen("rb"), both used by path_utils.cpp/platform.cpp),
//   - snapshots peak working set + available physical memory around the load.
//
// The global llama_log_set callback is installed EXACTLY ONCE per process
// (it is process-global); LoadWindow instances only gate/capture and never
// replace the callback. Diagnostics only: no loading semantics change.

#pragma once

#if defined(XLLAMA_USE_LLAMA)

    #include "llama.h"

    #include "xllama/platform.h" // log_output, peak_working_set_mb, avail_phys_mb
    #include "xllama/utf8_utils.h" // utf8_to_wstring (UWP wide paths)

    #include <cstddef>
    #include <cstdint>
    #include <cstdio>
    #include <filesystem>
    #include <mutex>
    #include <string>

namespace xllama {
namespace load_diag {

// ---------------------------------------------------------------------------
// Shared state. One mutex guards everything; the callback copies out under the
// lock and calls log_output WITHOUT holding it (log_output takes platform.cpp's
// own file mutex - keeping the critical section short avoids any ordering issue).
// ---------------------------------------------------------------------------
inline std::mutex& mtx() {
    static std::mutex m;
    return m;
}
inline bool& installed() {
    static bool b = false;
    return b;
}
inline int& window_depth() {
    static int d = 0;
    return d;
}
inline std::string& last_err() { // most recent single ERROR since the current window opened
    static std::string s;
    return s;
}
inline std::string& window_log() { // accumulated WARN/ERROR lines of the current load
    static std::string s;
    return s;
}
inline std::string& cont_buf() { // pending LLAMA_LOG_CONT text (progress dots)
    static std::string s;
    return s;
}

// Emit any CONT text accumulated so far as ONE line (the loader's progress
// callback emits one CONT per 1%; flushing per dot would spam xllama.log).
static void flush_cont_locked() {
    if (!cont_buf().empty()) {
        std::string line = "[llama.cpp] " + cont_buf();
        cont_buf().clear();
        mtx().unlock();
        log_output(line);
        mtx().lock();
    }
}

// Install the process-global llama.cpp -> xllama.log forwarding. Idempotent:
// llama_log_set is global, so the callback is replaced at most once per process.
inline void install_log_forwarding() noexcept {
    if (installed())
        return;
    installed() = true;
    llama_log_set([](enum ggml_log_level level, const char* text, void*) {
        if (!text)
            return;
        try {
            std::lock_guard<std::mutex> g(mtx());
            switch (level) {
                case GGML_LOG_LEVEL_CONT: // progress dots: accumulate, never per-line
                    if (window_depth() > 0)
                        cont_buf() += text;
                    return;
                case GGML_LOG_LEVEL_ERROR:
                    last_err() = text;
                    [[fallthrough]];
                case GGML_LOG_LEVEL_WARN:
                    if (window_depth() > 0)
                        window_log() += std::string(text) + "\n";
                    break;
                default: // INFO/DEBUG: gated below
                    break;
            }
            flush_cont_locked(); // keeps ordering of dots that preceded this line
            const bool forward = level >= GGML_LOG_LEVEL_WARN || window_depth() > 0;
            if (forward) {
                std::string line = "[llama.cpp] " + std::string(text);
                mtx().unlock();
                log_output(line);
                mtx().lock();
            }
        } catch (...) {
            // A log callback must never throw into llama.cpp.
        }
    }, nullptr);
}

// RAII scope around a llama_model_load_from_file() call. Marks load capture
// active (INFO/CONT forwarded, WARN/ERROR captured) and clears the per-load
// buffers on entry so a stale error from a previous load cannot leak into this
// one. Does NOT install/uninstall the global callback.
class LoadWindow {
  public:
    LoadWindow() {
        std::lock_guard<std::mutex> g(mtx());
        window_log().clear();
        last_err().clear();
        cont_buf().clear();
        ++window_depth();
    }
    ~LoadWindow() {
        std::lock_guard<std::mutex> g(mtx());
        if (window_depth() > 0)
            --window_depth();
        flush_cont_locked(); // any trailing progress dots become one final line
    }
    LoadWindow(const LoadWindow&) = delete;
    LoadWindow& operator=(const LoadWindow&) = delete;
};

// Most recent single ERROR text captured since the current window opened.
inline std::string last_error() {
    std::lock_guard<std::mutex> g(mtx());
    return last_err();
}

// All WARN/ERROR lines captured during the current load window.
inline std::string window_text() {
    std::lock_guard<std::mutex> g(mtx());
    return window_log();
}

// ---------------------------------------------------------------------------
// Pre-load file check. Read-only, 4 bytes, closed immediately. Uses only I/O
// already proven under AppContainer in this codebase: std::filesystem
// (path_utils.cpp) and _wfopen("rb") (platform.cpp / path_utils.cpp).
// ---------------------------------------------------------------------------
struct FileCheck {
    bool exists = false;
    std::uintmax_t size = 0;
    bool opened = false; // open + read of the first 4 bytes succeeded
    unsigned char magic[4] = {0, 0, 0, 0};
    bool is_gguf = false; // magic == "GGUF" (ggml GGUF_MAGIC)
    std::string detail;  // one-line summary for logs / error strings
};

inline FileCheck check_model_file(const std::string& path) {
    FileCheck fc;
    namespace fs = std::filesystem;
    std::error_code ec;
    fc.exists = fs::is_regular_file(path, ec);
    if (fc.exists)
        fc.size = fs::file_size(path, ec);

    FILE* fp = nullptr;
#ifdef XLLAMA_UWP
    fp = _wfopen(utf8_to_wstring(path).c_str(), L"rb"); // proven in platform.cpp
#else
    fp = std::fopen(path.c_str(), "rb");
#endif
    if (fp) {
        fc.opened = fread(fc.magic, 1, 4, fp) == 4;
        fclose(fp);
        fc.is_gguf = fc.opened && fc.magic[0] == 'G' && fc.magic[1] == 'G' && fc.magic[2] == 'U' &&
                    fc.magic[3] == 'F';
    }

    char buf[768];
    snprintf(buf, sizeof(buf),
             "[xllama] diag file-check %s: exists=%d size=%llu bytes (%.1f MB) open=%s magic=%02X%02X%02X%02X "
             "gguf=%s\n",
             path.c_str(), (int)fc.exists, (unsigned long long)fc.size,
             fc.size / 1048576.0, fc.opened ? "ok" : "FAILED", fc.magic[0], fc.magic[1],
             fc.magic[2], fc.magic[3], fc.is_gguf ? "yes" : "no");
    log_output(buf);
    fc.detail = std::string("exists=") + (fc.exists ? "1" : "0") + " size=" +
               std::to_string(fc.size) + " open=" + (fc.opened ? "ok" : "FAILED") +
               " gguf=" + (fc.is_gguf ? "yes" : "no");
    return fc;
}

// ---------------------------------------------------------------------------
// Parameter + memory snapshots. mmap/mlock in the pinned rev are not standalone
// fields: llama_model_loader derives use_mmap from load_mode (AUTO/MMAP/...)
// and logs a WARN when mmap is unsupported - that WARN is now forwarded, so
// logging load_mode (+ its derived mmap state) covers it without inventing
// fields.
// ---------------------------------------------------------------------------
inline void log_model_params(const char* tag, const llama_model_params& mp) {
    const bool mmap = mp.load_mode == LLAMA_LOAD_MODE_AUTO || mp.load_mode == LLAMA_LOAD_MODE_MMAP ||
                     mp.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    char buf[512];
    snprintf(buf, sizeof(buf),
             "[xllama] diag %s: mparams n_gpu_layers=%d split_mode=%d load_mode=%d (%s) mmap=%d "
             "main_gpu=%d vocab_only=%d check_tensors=%d use_extra_bufts=%d no_host=%d no_alloc=%d "
             "load_mtp=%d\n",
             tag, mp.n_gpu_layers, (int)mp.split_mode, (int)mp.load_mode,
             llama_load_mode_name(mp.load_mode), (int)mmap, mp.main_gpu, (int)mp.vocab_only,
             (int)mp.check_tensors, (int)mp.use_extra_bufts, (int)mp.no_host, (int)mp.no_alloc,
             (int)mp.load_mtp);
    log_output(buf);
}

inline void log_memory_snapshot(const char* tag) {
    char buf[160];
    snprintf(buf, sizeof(buf), "[xllama] diag %s: peak_ws=%lluMB avail_phys=%lluMB\n", tag,
             (unsigned long long)peak_working_set_mb(), (unsigned long long)avail_phys_mb());
    log_output(buf);
}

} // namespace load_diag
} // namespace xllama

#endif // XLLAMA_USE_LLAMA
