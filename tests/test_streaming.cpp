// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Host-testable coverage for the LAN SSE streaming layer and the safe KV-continuation
// decision. Both live in header-only, WinRT-free modules so they compile and run on a
// dev box (no llama / no WinRT) — the same headers api-server.cpp uses. The producer/
// consumer tests use real threads to prove concurrency + non-blocking backpressure +
// unconditional shutdown; see tools/verify_sse.cpp for the standalone runner.

#include "xllama/kv_continuation.h"
#include "xllama/sse.h"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace xllama;

namespace {
struct Collector {
    std::mutex m;
    std::vector<std::string> got;
};
std::string joinall(const std::vector<std::string>& v) {
    std::string s;
    for (auto& x : v)
        s += x;
    return s;
}
} // namespace

TEST_CASE("sse: framing + headers") {
    CHECK(sse::sse_data("{\"a\":1}") == "data: {\"a\":1}\n\n");
    CHECK(sse::http_chunk("hello") == "5\r\nhello\r\n");
    CHECK(sse::http_last_chunk() == "0\r\n\r\n");
    std::string h = sse::sse_response_headers();
    CHECK(h.find("Content-Type: text/event-stream") != std::string::npos);
    CHECK(h.find("Transfer-Encoding: chunked") != std::string::npos);
    CHECK(h.find("Access-Control-Allow-Origin: *") != std::string::npos);
    CHECK(h.find("Content-Length") == std::string::npos); // must be absent for streaming
}

TEST_CASE("sse: role-first, content, finish chunks") {
    CHECK(sse::build_chunk("id", "m", 1, 0, "assistant", "", "")
              .find("\"delta\":{\"role\":\"assistant\"}") != std::string::npos);
    CHECK(sse::build_chunk("id", "m", 1, 0, "", "Hi", "").find("\"content\":\"Hi\"") !=
          std::string::npos);
    CHECK(sse::build_chunk("id", "m", 1, 0, "", "", "stop").find("\"finish_reason\":\"stop\"") !=
          std::string::npos);
    CHECK(sse::sse_done_payload() == "[DONE]");
}

TEST_CASE("sse: StopAwareAssembler never leaks a split stop sequence") {
    sse::StopAwareAssembler a({"STOP"});
    std::string out = a.feed("Answer: 42 ");
    out += a.feed("ST");
    out += a.feed("OP tail");
    CHECK(out == "Answer: 42 ");     // STOP dropped, real text kept
    CHECK(a.flush_tail(true) == ""); // stop completed -> nothing held
}

TEST_CASE("sse: UTF-8 content passes through untouched") {
    CHECK(sse::build_chunk("id", "m", 1, 0, "", "caf\xc3\xa9 \xe2\x9c\x93", "")
              .find("caf\xc3\xa9 \xe2\x9c\x93") != std::string::npos);
}

TEST_CASE("sse: producer/consumer preserves order, loses nothing, ends with [DONE]") {
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
    s.finish(false); // closes -> consumer exits
    cons.join();
    REQUIRE(c.got.size() == 204); // one role, 200 content, finish, DONE, HTTP terminator
    CHECK(c.got[c.got.size() - 2] == sse::http_chunk(sse::sse_data("[DONE]")));
    CHECK(c.got.back() == sse::http_last_chunk());
    std::string all = joinall(c.got);
    CHECK(all.find("\"role\":\"assistant\"") != std::string::npos);
    size_t prev = 0;
    for (int i = 0; i < 200; ++i) {
        size_t at = all.find("tok" + std::to_string(i));
        REQUIRE(at != std::string::npos);
        CHECK(at > prev);
        prev = at;
    }
    CHECK(all.find("\"finish_reason\":\"length\"") != std::string::npos);
    CHECK(all.find("data: [DONE]") != std::string::npos);
}

TEST_CASE(
    "sse: bounded queue overflow is an explicit backpressure abort, not a block/silent drop") {
    Collector c;
    sse::SseStreamSession s(
        "id", "m", 1LL, {},
        [&](const std::string& f) {
            std::lock_guard<std::mutex> g(c.m);
            c.got.push_back(f);
            return true;
        },
        /*max_frames=*/3);
    // No consumer draining: filling past the bound must NOT block; it aborts + closes.
    for (int i = 0; i < 10; ++i)
        s.on_token("x" + std::to_string(i));
    CHECK(s.abort_flag()->load() == true); // explicit failure signal, not silent success
    std::string f;
    int n = 0;
    while (s.drain_one_blocking(f)) {
        if (++n > 50)
            break;
    }
    CHECK(n <= 3); // only pre-overflow frames survive
}

TEST_CASE("sse: unconditional shutdown wakes a blocked consumer even without finish()") {
    Collector c;
    std::atomic<bool> exited{false};
    sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string& f) {
        std::lock_guard<std::mutex> g(c.m);
        c.got.push_back(f);
        return true;
    });
    {
        sse::SseCloseGuard guard(s); // closes on scope exit (RAII)
        std::thread cons([&] {
            std::string f;
            while (s.drain_one_blocking(f)) {
                std::lock_guard<std::mutex> g(c.m);
                c.got.push_back(f);
            }
            exited.store(true);
        });
        s.on_token("hi"); // no finish(): only the guard's close() can wake the consumer
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        s.close(); // explicit close too (idempotent) so join is safe in-test
        cons.join();
    }
    CHECK(exited.load());
}

TEST_CASE("sse: sink failure sets abort_flag; consumer keeps draining until closed") {
    int writes = 0;
    std::atomic<bool> exited{false};
    sse::SseStreamSession s("id", "m", 1LL, {}, [&](const std::string&) { return ++writes < 3; });
    std::thread cons([&] {
        std::string f;
        while (s.drain_one_blocking(f)) {
        }
        exited.store(true);
    });
    for (int i = 0; i < 5; ++i)
        s.on_token("x"); // 3rd write fails -> abort set
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(s.abort_flag()->load() == true);
    s.close();
    cons.join();
    CHECK(exited.load());
}

TEST_CASE("kv: safe continuation decision") {
    using namespace xllama::kv;
    ConvState prev{"m", "sys", "fp1", {{"u1", "a1"}}, "", false, true};
    auto cont = decide(prev, {"m", "sys", "fp1", {{"u1", "a1"}}, "u2", false, true});
    CHECK(cont.reuse);
    CHECK_FALSE(cont.reset);
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
}
