// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Safe last-conversation KV-prefix reuse for the LAN chat API.
//
// The OpenAI contract is stateless: every request carries the FULL messages[].
// The Session, however, can keep a persistent KV cache and accept only the NEW
// turn's delta (GenerateParams::reuse_kv / reset_kv). Reusing that prefix is a
// large turn-2 prefill win — but ONLY when the incoming conversation is a strict
// continuation of exactly what was fed to the resident generator last time.
//
// This header holds the pure decision (no model, no WinRT) so it is host-testable:
// given the previous conversation state and the incoming one, it says whether KV
// can be reused and, if so, which prompt string to send (the delta) versus a full
// re-prefill. The caller renders strings with ChatFormat and passes them in.
//
// Reuse is refused — forcing reset_kv + full prefill — when ANY of these differ:
//   * model (different resident Session)
//   * system prompt
//   * sampling/settings fingerprint (temperature/top_p/seed/n_predict/etc.)
//   * the incoming history is not exactly prev.history + [prev turn]
//   * anything about the previous conversation changed (edit/regenerate/delete)
//   * the prompt was trimmed to fit n_ctx (a drop invalidates the prefix)

#pragma once

#include <string>
#include <vector>

namespace xllama {
namespace kv {

struct Turn {
    std::string user;
    std::string assistant; // what the model actually produced last turn
};

// A conversation as the API understands it, in already-rendered / canonical form.
struct ConvState {
    std::string model;
    std::string system;
    std::string params_fp;              // fingerprint of every setting that changes KV
    std::vector<Turn> history;          // completed exchanges BEFORE the final user turn
    std::string final_user;             // the turn to answer now
    bool trimmed = false;               // prompt was shortened to fit n_ctx this request
    bool primed = false;                // a prior reuse-capable turn primed the Session
    // Multimodal identity: an ordered fingerprint of every image in the conversation
    // (FNV-1a over each bitmap's bytes, joined). Empty == text-only. KV is only reused
    // when this EXACTLY matches the previous turn's — a changed/added/removed image means
    // the visual prefix differs and must be re-prefilled (never reuse mismatched media).
    std::string image_fingerprint;
};

struct Decision {
    bool reuse = false;   // -> GenerateParams::reuse_kv
    bool reset = true;    // -> GenerateParams::reset_kv (true on any non-continuation)
    std::string reason;   // human-readable, logged for measurement/observability
};

inline bool same_history(const std::vector<Turn>& a, const std::vector<Turn>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].user != b[i].user || a[i].assistant != b[i].assistant)
            return false;
    return true;
}

// Decide how to feed THIS request to the Session, given the PREVIOUS one.
// `prev` is the state as of the last completed turn on the resident Session
// (its history already includes that turn's user text AND the assistant reply
// the model produced). `cur` is the incoming request. On reuse the caller sends
// only render_delta(cur.final_user, prev_ended_with_stop); otherwise it sends a
// full render_prompt and must set reset_kv.
inline Decision decide(const ConvState& prev, const ConvState& cur) {
    Decision d;
    if (!prev.primed) {
        // Nothing primed on the resident Session yet (first turn, or after a
        // model swap that reset it): nothing to reuse. `primed` describes the
        // PRIOR session state, so the check is on prev, not cur.
        d.reuse = false;
        d.reset = true;
        d.reason = "first-turn";
        return d;
    }
    if (prev.model != cur.model) {
        d.reason = "model-changed";
        return d;
    }
    if (prev.system != cur.system) {
        d.reason = "system-changed";
        return d;
    }
    if (prev.params_fp != cur.params_fp) {
        d.reason = "settings-changed";
        return d;
    }
    if (cur.trimmed || prev.trimmed) {
        d.reason = "prompt-trimmed";
        return d;
    }
    // Multimodal: reuse only when the exact media prefix is identical. Any image in
    // either turn whose ordered fingerprint differs forces a full re-prefill.
    if (prev.image_fingerprint != cur.image_fingerprint) {
        d.reason = "images-changed";
        return d;
    }
    // Strict continuation test: incoming history must equal the previous
    // conversation exactly, and there must be exactly one NEW final user turn.
    if (!same_history(prev.history, cur.history)) {
        d.reason = "history-changed";
        return d;
    }
    // (prev.final_user + its assistant reply) is not part of either `history`
    // here — the caller folds it in before calling decide(). Reaching this point
    // with equal histories means the incoming request extends the previous one.
    d.reuse = true;
    d.reset = false;
    d.reason = "continuation";
    return d;
}

} // namespace kv
} // namespace xllama
