// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// OpenAI-compatible Server-Sent Events (SSE) framing for chat.completion.chunk.
// Header-only, WinRT-free and host-testable (mirrors json_utils.h), so the SSE
// byte format — role-first ordering, per-token content events, finish_reason,
// [DONE], UTF-8/escape handling and HTTP/1.1 chunked framing — can be unit
// tested without a model or the WinRT socket layer.
//
// The UWP server (uwp/api-server.cpp) owns the socket/DataWriter; this header
// only produces the exact bytes to hand to it, so the transport stays thin and
// every correctness-critical decision lives here where it can be tested.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xllama/json_utils.h"

namespace xllama {
namespace sse {

// One HTTP/1.1 chunked-transfer body chunk: <hex-length> CRLF <payload> CRLF.
// Streaming responses MUST NOT carry Content-Length; each SSE event is framed as
// its own chunk so the client sees bytes the instant they are written.
inline std::string http_chunk(const std::string& payload) {
    // Length is the byte length of the payload (UTF-8), in lowercase hex, no
    // leading zeros — exactly what an HTTP/1.1 receiver parses.
    char len[24];
    std::snprintf(len, sizeof(len), "%zx", static_cast<size_t>(payload.size()));
    std::string out;
    out.reserve(payload.size() + 8);
    out += len;
    out += "\r\n";
    out += payload;
    out += "\r\n";
    return out;
}

// Terminator of a chunked body: zero-length chunk + CRLF. Sent once, after the
// final [DONE] event.
inline std::string http_last_chunk() {
    return "0\r\n\r\n";
}

// SSE response header block. No Content-Length (chunked). CORS preserved so a
// browser EventSource / OpenAI SDK can consume it cross-origin, matching the
// preflight the rest of the API already answers.
inline std::string sse_response_headers() {
    return "HTTP/1.1 200 OK\r\n"
           "Content-Type: text/event-stream\r\n"
           "Cache-Control: no-cache\r\n"
           "Connection: keep-alive\r\n"
           "Transfer-Encoding: chunked\r\n"
           "Access-Control-Allow-Origin: *\r\n"
           "X-Accel-Buffering: no\r\n" // disable proxy buffering where present
           "\r\n";
}

// One SSE event frame: "data: <json>\n\n". OpenAI does not use the `event:`
// field for chat chunks — every data line is a full JSON object.
inline std::string sse_data(const std::string& json) {
    std::string out;
    out.reserve(json.size() + 8);
    out += "data: ";
    out += json;
    out += "\n\n";
    return out;
}

// The terminal SSE payload (not JSON): the literal OpenAI end-of-stream marker.
inline std::string sse_done_payload() {
    return "[DONE]";
}

// Build one chat.completion.chunk JSON object. Exactly one of role/content/
// finish is normally non-empty per event, matching OpenAI: empty fields are
// omitted from the delta, and finish_reason is emitted as null except on the
// final chunk. `created` is a unix timestamp; `index` is the choice index.
inline std::string build_chunk(const std::string& id, const std::string& model, long long created,
                               int index, const std::string& role_delta,
                               const std::string& content_delta, const std::string& finish_reason) {
    std::string delta;
    bool first = true;
    auto put = [&](const char* key, const std::string& val) {
        if (!first)
            delta += ",";
        first = false;
        delta += "\"";
        delta += key;
        delta += "\":\"";
        delta += json_escape(val);
        delta += "\"";
    };
    if (!role_delta.empty())
        put("role", role_delta);
    if (!content_delta.empty())
        put("content", content_delta);

    std::string out;
    out.reserve(64 + id.size() + model.size() + delta.size());
    out += "{";
    out += "\"id\":\"" + json_escape(id) + "\",";
    out += "\"object\":\"chat.completion.chunk\",";
    out += "\"created\":" + std::to_string(created) + ",";
    out += "\"model\":\"" + json_escape(model) + "\",";
    out += "\"system_fingerprint\":null,";
    out += "\"choices\":[{\"index\":" + std::to_string(index) + ",\"delta\":{" + delta +
           "},\"finish_reason\":";
    if (finish_reason.empty())
        out += "null";
    else
        out += "\"" + json_escape(finish_reason) + "\"";
    out += "}]}";
    return out;
}

// ---------------------------------------------------------------------------
// Token pieces can split UTF-8 scalars. Keep incomplete suffixes until the next
// piece, and replace invalid/truncated sequences so every JSON event is UTF-8.
class Utf8Assembler {
  public:
    std::string feed(const std::string& bytes, bool final = false) {
        pending_ += bytes;
        std::string out;
        size_t i = 0;
        while (i < pending_.size()) {
            const unsigned char c = static_cast<unsigned char>(pending_[i]);
            const size_t width = c < 0x80                 ? 1
                                 : c >= 0xc2 && c <= 0xdf ? 2
                                 : c >= 0xe0 && c <= 0xef ? 3
                                 : c >= 0xf0 && c <= 0xf4 ? 4
                                                          : 0;
            if (!width) {
                out += "\xef\xbf\xbd";
                ++i;
                continue;
            }
            bool valid = true;
            for (size_t j = 1; j < width && i + j < pending_.size(); ++j) {
                const unsigned char d = static_cast<unsigned char>(pending_[i + j]);
                if (d < 0x80 || d > 0xbf ||
                    (j == 1 && ((c == 0xe0 && d < 0xa0) || (c == 0xed && d > 0x9f) ||
                                (c == 0xf0 && d < 0x90) || (c == 0xf4 && d > 0x8f))))
                    valid = false;
            }
            if (!valid) {
                out += "\xef\xbf\xbd";
                ++i;
                continue;
            }
            if (pending_.size() - i < width) {
                if (!final)
                    break;
                out += "\xef\xbf\xbd";
                i = pending_.size();
                break;
            }
            out.append(pending_, i, width);
            i += width;
        }
        pending_.erase(0, i);
        return out;
    }

  private:
    std::string pending_;
};

// ---------------------------------------------------------------------------
// Stop-aware streaming assembler
//
// on_token fires for EVERY sampled piece BEFORE the decode loop runs its stop
// check, so a stop sequence split across pieces (e.g. Gemma's <end_of_turn>)
// would otherwise leak into the stream. The assembler holds back only the
// trailing bytes that could still be the start of a stop sequence and releases
// everything else immediately — so tokens stream out as they arrive without
// ever emitting a partial/complete stop marker.
// ---------------------------------------------------------------------------
class StopAwareAssembler {
  public:
    explicit StopAwareAssembler(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    // Feed one detokenized piece (a view — copied immediately by the caller).
    // Returns bytes safe to emit now as a content event; empty if everything so
    // far could still turn out to be a stop sequence.
    std::string feed(const std::string& piece) {
        if (stop_seen_)
            return {};
        pending_ += piece;
        // A complete stop sequence anywhere in pending means generation is over:
        // emit the text before it and drop the marker itself.
        size_t first_stop = std::string::npos;
        for (const auto& s : stops_) {
            if (s.empty())
                continue;
            size_t pos = pending_.find(s);
            if (pos < first_stop)
                first_stop = pos;
        }
        if (first_stop != std::string::npos) {
            stop_seen_ = true;
            std::string emit = pending_.substr(0, first_stop);
            pending_.clear();
            return utf8_.feed(emit);
        }
        // Hold back the longest suffix of pending that is a strict prefix of any
        // stop sequence; release the rest now.
        const size_t hold = held_back_len();
        if (pending_.size() > hold) {
            std::string emit = pending_.substr(0, pending_.size() - hold);
            pending_.erase(0, pending_.size() - hold);
            return utf8_.feed(emit);
        }
        return {};
    }

    // Generation ended. `ended_with_stop` reflects the decode loop's own verdict.
    // If a stop was seen (or reported), the held-back tail IS the marker and is
    // dropped; otherwise it was real content and must still be flushed.
    std::string flush_tail(bool ended_with_stop) {
        if (ended_with_stop || stop_seen_) {
            pending_.clear();
            return utf8_.feed({}, true);
        }
        std::string emit = pending_;
        pending_.clear();
        return utf8_.feed(emit, true);
    }

  private:
    // Longest suffix of pending_ that is a strict prefix of any stop sequence.
    size_t held_back_len() const {
        size_t best = 0;
        for (const auto& s : stops_) {
            if (s.empty())
                continue;
            for (size_t k = 1; k < s.size() && k <= pending_.size(); ++k) {
                // suffix of pending_ of length k == prefix of s of length k
                if (pending_.compare(pending_.size() - k, k, s, 0, k) == 0)
                    best = k > best ? k : best;
            }
        }
        return best;
    }

    std::vector<std::string> stops_;
    std::string pending_;
    bool stop_seen_ = false;
    Utf8Assembler utf8_;
};

// ---------------------------------------------------------------------------
// Producer/consumer streaming session (host-testable; the WinRT layer injects a
// sink that owns ONE DataWriter for the whole response).
//
// Hot-path contract: on_token() copies the string_view immediately, runs it
// through the StopAwareAssembler, enqueues one framed SSE event and RETURNS. It
// never touches the socket. A single consumer (drain) drains the queue and calls
// the sink; the sink is the only place that awaits StoreAsync/FlushAsync. So the
// decode thread never waits on LAN I/O; a full queue explicitly aborts the stream.
//
// The sink returns false on write failure (client gone): the session then sets its
// abort flag, which the caller wires to GenerateParams::abort_flag so the decode
// loop stops at its next iteration. No DataWriter is created per token; ordering is
// guaranteed by the FIFO queue; memory is bounded by max_frames.
// ---------------------------------------------------------------------------
class SseStreamSession {
  public:
    using Sink = std::function<bool(const std::string& framed_bytes)>;

    SseStreamSession(std::string id, std::string model, long long created,
                     std::vector<std::string> stops, Sink sink, size_t max_frames = 4096)
        : id_(std::move(id)), model_(std::move(model)), created_(created),
          assembler_(std::move(stops)), sink_(std::move(sink)),
          max_frames_(max_frames ? max_frames : 1) {}

    // The abort flag the caller must wire to GenerateParams::abort_flag. Set on
    // sink failure (client gone) OR on bounded-queue backpressure overflow.
    std::atomic<bool>* abort_flag() {
        return &abort_;
    }

    // Emit the role-first event (empty content, null finish). Call once before tokens.
    void emit_role_first() {
        enqueue_frame(build_event("assistant", "", ""));
    }

    // PRODUCER (runs on the decode thread). Copy the view immediately, assemble
    // (stop-aware), enqueue if there are emittable bytes, RETURN IMMEDIATELY. It
    // never touches the sink/socket. Enqueue is NON-BLOCKING: a full queue beyond
    // max_frames_ is treated as a backpressure failure (see enqueue_frame) — it
    // sets abort_flag and closes the session rather than blocking the decode thread.
    void on_token(std::string_view piece) {
        if (closed_.load() || abort_.load())
            return; // closed or already failing: stop producing
        const std::string out = assembler_.feed(std::string(piece)); // copy view immediately
        if (!out.empty())
            enqueue_frame(build_event("", out, ""));
    }

    // Finish (success path): flush the held-back tail (dropped if a stop completed
    // it), then the finish_reason chunk, [DONE], and the terminator; then close.
    void finish(bool ended_with_stop, bool ended_naturally = false) {
        const std::string tail = assembler_.flush_tail(ended_with_stop);
        if (!tail.empty())
            enqueue_frame(build_event("", tail, ""));
        enqueue_frame(build_event("", "", ended_with_stop || ended_naturally ? "stop" : "length"));
        enqueue_frame(http_chunk(sse_data(sse_done_payload())));
        enqueue_frame(http_last_chunk());
        close();
    }

    // UNCONDITIONAL shutdown. Every exit path (success, generation abort,
    // exception, socket failure, request cancellation) must reach this so a
    // consumer blocked in drain_one_blocking is always woken and never waits for a
    // finish() that some error path forgot to call. Idempotent.
    void close() {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_.store(true);
        }
        cv_.notify_all();
    }

    // CONSUMER (production writer thread). Blocks until a frame is available or the
    // session is closed. Writes via sink. On sink failure (client gone) it sets
    // abort_flag but KEEPS draining-and-discarding so the producer never blocks and
    // every queued frame is accounted for. Returns false ONLY when closed AND empty
    // (clean end of stream), which is the loop's exit condition.
    bool drain_one_blocking(std::string& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !q_.empty() || closed_.load(); });
        if (q_.empty())
            return false; // closed and drained -> stop the consumer loop
        out = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        cv_.notify_all(); // freed a slot for the producer
        if (!sink_failed_) {
            if (!sink_(out)) {
                sink_failed_.store(true);
                abort_.store(true); // tell the decode loop (via gp.abort_flag) to stop
            }
        }
        return true;
    }

  private:
    std::string build_event(const std::string& role, const std::string& content,
                            const std::string& finish) {
        return http_chunk(sse_data(build_chunk(id_, model_, created_, 0, role, content, finish)));
    }

    // NON-BLOCKING enqueue. Normal case: room available -> push + signal + return.
    // Overflow (queue at max_frames_): treat as a stream/backpressure FAILURE — set
    // abort_flag, close the session (waking any consumer), and DROP this frame.
    // We never block the decode thread and never silently succeed after dropping:
    // abort_flag is the explicit signal that the stream did not complete cleanly.
    void enqueue_frame(std::string frame) {
        if (frame.empty())
            return;
        std::lock_guard<std::mutex> lk(m_);
        if (closed_.load())
            return; // already closed: drop (failure already signalled)
        if (q_.size() >= max_frames_) {
            abort_.store(true); // backpressure failure — not a silent success
            closed_.store(true);
            cv_.notify_all();
            return;
        }
        q_.push_back(std::move(frame));
        cv_.notify_all(); // a frame is ready for the consumer
    }

    std::string id_, model_;
    long long created_;
    StopAwareAssembler assembler_;
    Sink sink_;
    size_t max_frames_;
    std::mutex m_;
    std::condition_variable cv_; // one CV; every state change signals it
    std::atomic<bool> sink_failed_{false};
    std::atomic<bool> closed_{false}; // producer side closed (finish/backpressure/close)
    std::deque<std::string> q_;
    std::atomic<bool> abort_{false};
};

// RAII shutdown guard: guarantees close() runs on EVERY exit path so a consumer
// blocked in drain_one_blocking is always woken, even if generation throws or an
// early return skips finish(). Idempotent with an explicit finish()->close().
class SseCloseGuard {
  public:
    explicit SseCloseGuard(SseStreamSession& s) : s_(s) {}
    ~SseCloseGuard() {
        s_.close();
    }
    SseCloseGuard(const SseCloseGuard&) = delete;
    SseCloseGuard& operator=(const SseCloseGuard&) = delete;

  private:
    SseStreamSession& s_;
};

} // namespace sse
} // namespace xllama
