// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Standalone host verifier for the pure SSE framing + StopAwareAssembler, the
// producer/consumer SseStreamSession (bounded queue, non-blocking enqueue, RAII
// shutdown), and the safe KV-continuation decision. Compiles WITHOUT llama/WinRT.
// Build+run: g++ -std=c++17 -pthread -I include tools/verify_sse.cpp -o verify && ./verify

#include "xllama/kv_continuation.h"
#include "xllama/sse.h"

#include <atomic>
#include <cstdio>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace xllama;

static int g_fail = 0;
#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cout << "FAIL: " << #cond << "  @line " << __LINE__ << "\n"; \
            ++g_fail;                                                         \
        }                                                                     \
    } while (0)

// ---- pure formatting -------------------------------------------------------
static void test_sse_format_exact() {
    std::string json = sse::build_chunk("chatcmpl-x1", "m", 1700000000, 0, "", "hi", "");
    std::string want =
        "{\"id\":\"chatcmpl-x1\",\"object\":\"chat.completion.chunk\","
        "\"created\":1700000000,\"model\":\"m\",\"system_fingerprint\":null,"
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}";
    CHECK(json == want);
    CHECK(sse::sse_data(json) == "data: " + want + "\n\n");
    char hdr[32];
    std::snprintf(hdr, sizeof(hdr), "%zx\r\n", sse::sse_data(json).size());
    CHECK(sse::http_chunk(sse::sse_data(json)) == std::string(hdr) + sse::sse_data(json) + "\r\n");
    CHECK(sse::http_last_chunk() == "0\r\n\r\n");
}
static void test_headers_and_escaping() {
    std::string h = sse::sse_response_headers();
    CHECK(h.find("Content-Type: text/event-stream") != std::string::npos);
    CHECK(h.find("Transfer-Encoding: chunked") != std::string::npos);
    CHECK(h.find("Access-Control-Allow-Origin: *") != std::string::npos);
    CHECK(h.find("Content-Length") == std::string::npos);
    std::string j = sse::build_chunk("id", "m", 1, 0, "", "caf\xc3\xa9 \"q\"\\x\ny", "");
    CHECK(j.find("caf\xc3\xa9") != std::string::npos); // UTF-8 passthrough
    CHECK(j.find("\\\"q\\\"") != std::string::npos);
}

// ---- StopAwareAssembler ----------------------------------------------------
static void test_stop_not_leaked() {
    sse::StopAwareAssembler a({"STOP"});
    std::string out = a.feed("Answer: 42 ");
    out += a.feed("ST");      // partial stop held back
    out += a.feed("OP tail"); // completes STOP -> drop marker, keep real tail
    CHECK(out == "Answer: 42 ");
    CHECK(a.flush_tail(true) == ""); // stop completed -> nothing to flush
}
static void test_flush_on_natural_end() {
    sse::StopAwareAssembler a({});
    CHECK(a.feed("a") == "a");
    CHECK(a.flush_tail(false) == ""); // nothing held back (no stops)
}

static void test_utf8_piece_boundaries() {
    const std::string text = "caf\xc3\xa9 \xe2\x9c\x93 \xf0\x9f\x91\x8b";
    for (size_t split = 0; split <= text.size(); ++split) {
        sse::StopAwareAssembler a({"STOP"});
        std::string out = a.feed(text.substr(0, split));
        out += a.feed(text.substr(split) + "ST");
        out += a.feed("OP ignored");
        out += a.feed("also ignored");
        out += a.flush_tail(true);
        CHECK(out == text);
    }
    sse::StopAwareAssembler a({});
    CHECK(a.feed("\xf0") == "");
    CHECK(a.feed("\x9f\x91") == "");
    CHECK(a.feed("\x8b") == "\xf0\x9f\x91\x8b");
    CHECK(a.feed("\xc3") == "");
    CHECK(a.flush_tail(false) == "\xef\xbf\xbd");
    sse::StopAwareAssembler earliest({"LATE", "STOP"});
    CHECK(earliest.feed("answerSTOPignoredLATE") == "answer");
}

static void test_terminal_http_chunks_and_eos() {
    std::vector<std::string> frames;
    sse::SseStreamSession s("id", "m", 1, {"STOP"}, [&](const std::string& f) {
        frames.push_back(f);
        return true;
    });
    s.emit_role_first();
    s.on_token("hello ST");
    s.finish(false, true); // EOS preserves a partial stop prefix and reports stop.
    std::string frame;
    while (s.drain_one_blocking(frame)) {
    }
    CHECK(frames.size() == 6);
    if (frames.size() != 6)
        return;
    CHECK(frames[0].find("\"role\":\"assistant\"") != std::string::npos);
    CHECK(frames[2].find("\"content\":\"ST\"") != std::string::npos);
    CHECK(frames[3].find("\"finish_reason\":\"stop\"") != std::string::npos);
    CHECK(frames[4] == sse::http_chunk("data: [DONE]\n\n"));
    CHECK(frames[5] == sse::http_last_chunk());
    for (size_t i = 0; i + 1 < frames.size(); ++i) {
        size_t at = frames[i].find("\r\n");
        CHECK(at != std::string::npos);
        if (at == std::string::npos)
            continue;
        const size_t n = std::stoul(frames[i].substr(0, at), nullptr, 16);
        CHECK(frames[i].size() == at + 2 + n + 2);
        CHECK(frames[i].substr(at + 2, 6) == "data: ");
    }
}

// ---- producer/consumer session --------------------------------------------
struct Collector {
    std::mutex m;
    std::vector<std::string> got;
};

static void test_session_concurrent_ordering() {
    Collector c;
    sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string& f) {
        std::lock_guard<std::mutex> g(c.m);
        c.got.push_back(f);
        return true;
    });
    std::thread cons([&] {
        std::string f;
        while (s.drain_one_blocking(f)) {
        }
    });
    s.emit_role_first();
    for (int i = 0; i < 200; ++i)
        s.on_token("tok" + std::to_string(i));
    s.finish(false); // closes the session -> consumer exits
    cons.join();
    std::string all;
    for (auto& x : c.got)
        all += x;
    CHECK(all.find("\"role\":\"assistant\"") != std::string::npos); // role first
    size_t prev = 0;
    for (int i = 0; i < 200; ++i) {
        size_t at = all.find("tok" + std::to_string(i));
        CHECK(at != std::string::npos && at > prev);
        if (at != std::string::npos)
            prev = at;
    }
    CHECK(all.find("\"finish_reason\":\"length\"") != std::string::npos);
    CHECK(all.find("data: [DONE]") != std::string::npos);
    CHECK(!c.got.empty() && c.got.back().find("0\r\n\r\n") != std::string::npos); // terminator last
}

static void test_session_stop_not_leaked_concurrent() {
    Collector c;
    sse::SseStreamSession s("id", "m", 1LL, {"STOP"}, [&](const std::string& f) {
        std::lock_guard<std::mutex> g(c.m);
        c.got.push_back(f);
        return true;
    });
    std::thread cons([&] {
        std::string f;
        while (s.drain_one_blocking(f)) {
        }
    });
    s.on_token("Answer: ");
    s.on_token("42 ST");
    s.on_token("OP tail");
    s.finish(true); // stop completed -> held-back marker dropped
    cons.join();
    std::string all;
    for (auto& x : c.got)
        all += x;
    CHECK(all.find("Answer: ") != std::string::npos);
    CHECK(all.find("STOP") == std::string::npos); // marker never emitted
}

// Bounded queue, no consumer draining fast enough -> overflow is an explicit
// backpressure FAILURE (abort_flag set), NOT a silent success and NOT a block.
static void test_backpressure_sets_abort_not_block() {
    Collector c;
    sse::SseStreamSession s(
        "id", "m", 1LL, {},
        [&](const std::string& f) {
            std::lock_guard<std::mutex> g(c.m);
            c.got.push_back(f);
            return true;
        },
        /*max_frames=*/3);
    // No consumer yet: fill past the bound. Producer must NOT block; overflow trips abort+close.
    for (int i = 0; i < 10; ++i)
        s.on_token("x" + std::to_string(i));
    CHECK(s.abort_flag()->load() == true); // backpressure -> explicit failure
    // Drain whatever survived (<= bound). No hang: queue is closed so drain returns.
    std::string f;
    int n = 0;
    while (s.drain_one_blocking(f)) {
        ++n;
        if (n > 100)
            break;
    }
    CHECK(n <= 3); // only pre-overflow frames survived
}

// Shutdown must be UNCONDITIONAL: a consumer blocked in drain is woken by close()
// even when finish() was never called (simulates an error path that forgets finish).
static void test_shutdown_wakes_consumer_without_finish() {
    Collector c;
    sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string& f) {
        std::lock_guard<std::mutex> g(c.m);
        c.got.push_back(f);
        return true;
    });
    std::atomic<bool> exited{false};
    std::thread cons([&] {
        std::string f;
        while (s.drain_one_blocking(f)) {
        }
        exited.store(true);
    });
    s.on_token("hello"); // one frame, no finish()
    s.close();           // unconditional shutdown wakes the consumer
    cons.join();         // would hang forever if close() failed to wake it
    CHECK(exited.load());
    std::string all;
    for (auto& x : c.got)
        all += x;
    CHECK(all.find("hello") != std::string::npos);
}

// RAII guard closes on scope exit -> consumer wakes even with no explicit finish.
static void test_raii_guard_wakes_consumer() {
    Collector c;
    std::atomic<bool> exited{false};
    {
        sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string& f) {
            std::lock_guard<std::mutex> g(c.m);
            c.got.push_back(f);
            return true;
        });
        std::thread cons([&] {
            std::string f;
            while (s.drain_one_blocking(f)) {
            }
            exited.store(true);
        });
        {
            sse::SseCloseGuard guard(s);
            s.on_token("hi");
        } // destructor alone must wake the consumer, before join()
        cons.join();
    }
    CHECK(exited.load());
}

// Sink failure -> abort_flag set; consumer keeps draining until closed.
static void test_sink_failure_sets_abort() {
    int writes = 0;
    sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string&) { return ++writes < 3; });
    std::atomic<bool> exited{false};
    std::thread cons([&] {
        std::string f;
        while (s.drain_one_blocking(f)) {
        }
        exited.store(true);
    });
    for (int i = 0; i < 5; ++i)
        s.on_token("x"); // a few frames; 3rd write fails -> abort set
    // give the consumer time to process and observe failure
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(s.abort_flag()->load() == true);
    s.close(); // unconditional shutdown so the consumer can exit
    cons.join();
    CHECK(exited.load());
}

// ---- KV continuation -------------------------------------------------------
static void test_kv_continuation() {
    using namespace xllama::kv;
    ConvState prev{"m", "sys", "fp1", {{"u1", "a1"}}, "", false, true};
    CHECK(decide(prev, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true}).reuse);
    CHECK(decide(prev, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true}).reset == false);
    CHECK(decide(prev, {"other", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true}).reason ==
          "model-changed");
    CHECK(decide(prev, {"m", "other", "fp1", {{"u1", "a1"}}, "u2", false, true}).reason ==
          "system-changed");
    CHECK(decide(prev, {"m", "sys", "fp2", {{"u1", "a1"}}, "u2", false, true}).reason ==
          "settings-changed");
    CHECK(decide(prev, {"m", "sys", "fp1", {{"uX", "YY"}}, "u2", false, true}).reason ==
          "history-changed");
    CHECK(decide(prev, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", true, true}).reason ==
          "prompt-trimmed");
    ConvState notprimed{"m", "sys", "fp1", {}, "", false, false};
    CHECK(decide(prev, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, false}).reason ==
          "session-changed");
    CHECK(decide(notprimed, {"m", "sys", "fp1", {}, "u2", false, true}).reason == "first-turn");
    // Multimodal: reuse only when the exact media fingerprint matches; any change resets.
    ConvState prev_img{"m", "sys", "fp1", {{"u1", "a1"}}, "", false, true, "FNVIMG1"};
    CHECK(decide(prev_img, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true, "FNVIMG1"})
              .reason == "continuation"); // same image -> reuse
    CHECK(decide(prev_img, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true, "FNVIMG2"})
              .reason == "images-changed"); // changed image -> reset
    CHECK(decide(prev_img, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true, ""}).reason ==
          "images-changed"); // image removed -> reset
}

int main() {
    test_sse_format_exact();
    test_headers_and_escaping();
    test_stop_not_leaked();
    test_flush_on_natural_end();
    test_utf8_piece_boundaries();
    test_terminal_http_chunks_and_eos();
    test_session_concurrent_ordering();
    test_session_stop_not_leaked_concurrent();
    test_backpressure_sets_abort_not_block();
    test_shutdown_wakes_consumer_without_finish();
    test_raii_guard_wakes_consumer();
    test_sink_failure_sets_abort();
    test_kv_continuation();
    std::cout << (g_fail == 0 ? "ALL PASS\n" : "FAILURES!\n");
    return g_fail ? 1 : 0;
}
