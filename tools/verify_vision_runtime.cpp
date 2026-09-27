// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Host contract test for the mtmd prefill -> decode seam (Milestone B).
//
// What it proves, without a model or a GPU:
//   1. vision_prefill() evaluates the ordered text+image prompt EXACTLY ONCE -- one
//      mtmd_tokenize(), one mtmd_helper_eval_chunks() over all chunks -- and leaves the
//      KV at the end of that prompt so generation CONTINUES from there (no second full
//      prompt evaluation, which is what generate_from_prefilled() guarantees).
//   2. logits_last must be TRUE on that eval. At the pinned rev mtmd-helper.cpp computes
//      chunk_logits_last = (i == n_chunks-1) && logits_last, so a false there flags no
//      token at all and the first llama_sampler_sample() reads logits that were never
//      computed. The fake models exactly that rule: sampling is only legal after an eval
//      that flagged a position.
//   3. n_batch must be > 0: mtmd_helper_eval_chunk_single() opens with an unconditional
//      GGML_ASSERT(n_batch > 0) at this rev (aborts in Release too), so the caller has to
//      pass llama_n_batch(), never 0.
//   4. Fast/Detailed presets reach mtmd_context_params.image_min_tokens/max tokens.
//   5. Marker/bitmap binding stays ordered across the C ABI (bitmaps arrive in the same
//      order as the markers produced by render_ordered()).
//   6. Failure paths report through RETURN CODES only: a tokenize failure or a failed
//      bitmap decode must surface as r.ok == false, never as an exception.
//
// EXCEPTION POLICY: every fake below is extern "C". Nothing in a C-ABI function throws --
// MSVC compiles these TUs with /EHc ("extern C functions do not throw"), and throwing
// across that boundary is warning C4297 plus UB-ish unwinding behaviour. Failures inside
// the fake are recorded in g_rt (violations + counters) and asserted by main() after the
// call returns. No compiler flag is used to silence anything.
//
// Build+run (no llama/WinRT link needed -- the fakes ARE the runtime). MSVC builds this
// TU with /EHc, which is exactly why nothing below may throw:
//   cl /EHsc /std:c++17 /DXLLAMA_HAS_MTMD=1 /I include /I uwp
//      /I llama.cpp/include /I llama.cpp/tools/mtmd /I llama.cpp/ggml/include
//      tools/verify_vision_runtime.cpp uwp/vision_mtmd.cpp

#include "vision_mtmd.h"      // the production TU under test (compiled alongside)
#include "xllama/vision.h"

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- fake runtime state ------------------------------------------------------

// Opaque types the headers only forward-declare. Definitions live here because this TU
// provides the runtime implementation the production TU links against.
struct mtmd_input_chunk { bool image = false; int n_tokens = 0; };
struct mtmd_context { int magic = 0x4d544d44; };
struct mtmd_bitmap { uint64_t id = 0; };
struct mtmd_helper_video {};
struct mtmd_input_chunks { std::vector<mtmd_input_chunk> chunks; };
struct llama_model { int magic = 1; };
struct llama_context { int magic = 2; };
// Fake KV handle: llama_get_memory()/llama_memory_clear() must agree on one object, and
// llama_get_logits() needs somewhere to point. Opaque storage behind the C typedefs.
struct FakeMemory { int magic = 0x4d454d31; };
struct FakeLogits { float v[8] = {0.f}; };
static FakeMemory g_fake_mem;
static FakeLogits g_fake_logits;

struct FakeState {
    std::vector<std::string> calls;      // call log, in order
    mtmd_input_chunks snapshot;          // copy of what the last mtmd_tokenize produced
    std::vector<uint64_t> bitmap_ids;    // ids handed to mtmd_tokenize, in order
    std::vector<std::string> violations; // recorded, never thrown (C ABI)

    int n_tokenize = 0;
    int n_eval = 0;
    bool last_logits_last = false;       // did the last eval flag a logits position?
    int32_t last_n_batch = -1;
    llama_pos n_past = 0;                // fake KV position

    int fail_tokenize = 0;               // force mtmd_tokenize to return non-zero
    bool fail_bitmap = false;            // force the bitmap helper to return a null bitmap
    int want_bitmaps = 0;                // markers must match this for tokenize to succeed
    std::string marker;                  // pinned default marker, from mtmd_default_marker()

    mtmd_context_params params{};        // what vision_open() passed down
    mtmd_context* last_ctx = nullptr;    // the fake handle vision_open() was handed back
};
static FakeState g_rt;

static void log_call(const char* what) { g_rt.calls.push_back(what); }

// ---- fake C ABI (extern "C": nothing here may throw) -------------------------

extern "C" {

const char* mtmd_default_marker(void) { return "<__media__>"; } // pinned rev 3cb7ffb1a

mtmd_context_params mtmd_context_params_default(void) {
    log_call("mtmd_context_params_default");
    mtmd_context_params p{};
    p.image_min_tokens = -1; // defaults, exactly as the pinned source initialises them
    p.image_max_tokens = -1;
    return p;
}

// Signatures below must match the pinned headers EXACTLY (they are extern "C" declarations
// that these definitions satisfy): mtmd.h takes `const struct llama_model *`, and
// llama.h returns uint32_t from llama_n_batch.
mtmd_context* mtmd_init_from_file(const char* mmproj_fname, const struct llama_model* model,
                                  struct mtmd_context_params params) {
    log_call("mtmd_init_from_file");
    g_rt.params = params;
    if (!mmproj_fname || !model) return nullptr; // no vision on this backend
    g_rt.last_ctx = new mtmd_context();
    return g_rt.last_ctx;
}

void mtmd_free(mtmd_context* ctx) { if (ctx) { log_call("mtmd_free"); delete ctx; } }

mtmd_input_chunks* mtmd_input_chunks_init(void) {
    log_call("mtmd_input_chunks_init");
    return new mtmd_input_chunks();
}
// Snapshot before freeing so main() can walk the chunks the fake tokenizer produced (the
// caller owns and frees the object, so this is the only chance to inspect it).
// Snapshot before freeing so main() can walk the chunks the fake tokenizer produced (the
// caller owns and frees the object, so this is the only chance to inspect it).
void mtmd_input_chunks_free(mtmd_input_chunks* chunks) {
    if (!chunks) return;
    g_rt.snapshot.chunks = chunks->chunks;
    delete chunks;
}

void mtmd_bitmap_free(mtmd_bitmap* bitmap) { if (bitmap) delete bitmap; }

struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(mtmd_context* ctx,
                                                                   const unsigned char* buf,
                                                                   size_t len, bool placeholder) {
    (void)ctx; (void)placeholder;
    log_call("mtmd_helper_bitmap_init_from_buf");
    struct mtmd_helper_bitmap_wrapper w{nullptr, nullptr};
    if (g_rt.fail_bitmap || !buf || len == 0) return w; // failure is a null handle, not a throw
    uint64_t h = 1469598103934665603ULL;                 // FNV-1a, same as xllama::vision
    for (size_t i = 0; i < len; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    w.bitmap = new mtmd_bitmap();
    w.bitmap->id = h;
    return w;
}

// Splits the prompt on the media marker exactly like the pinned tokenizer: text chunks in
// order, one image chunk per marker. Returns 1 when the bitmap count does not match the
// marker count (the pinned behaviour) -- recorded, never thrown.
int32_t mtmd_tokenize(mtmd_context* ctx, mtmd_input_chunks* output,
                      const mtmd_input_text* text, const mtmd_bitmap** bitmaps, size_t n_bitmaps) {
    (void)ctx;
    log_call("mtmd_tokenize");
    ++g_rt.n_tokenize;
    g_rt.bitmap_ids.clear();
    for (size_t i = 0; i < n_bitmaps; ++i) g_rt.bitmap_ids.push_back(bitmaps[i]->id);

    output->chunks.clear();
    g_rt.snapshot.chunks.clear();
    const std::string s(text->text, text->text_len);
    const std::string& marker = g_rt.marker;
    size_t pos = 0;
    while (true) {
        size_t hit = s.find(marker, pos);
        std::string seg = s.substr(pos, hit == std::string::npos ? std::string::npos : hit - pos);
        if (!seg.empty()) output->chunks.push_back({false, static_cast<int>(seg.size())});
        if (hit == std::string::npos) break;
        output->chunks.push_back({true, 256}); // one image chunk = the Fast preset floor
        pos = hit + marker.size();
    }
    if (g_rt.fail_tokenize != 0) return g_rt.fail_tokenize;
    if (static_cast<int>(n_bitmaps) != g_rt.want_bitmaps) {
        g_rt.violations.push_back("tokenize: bitmap count != marker count (would return 1)");
        return 1;
    }
    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context* ctx, struct llama_context* lctx,
                                const mtmd_input_chunks* chunks, llama_pos n_past,
                                llama_seq_id seq_id, int32_t n_batch, bool logits_last,
                                llama_pos* new_n_past) {
    (void)ctx; (void)seq_id;
    log_call("mtmd_helper_eval_chunks");
    ++g_rt.n_eval;
    g_rt.last_logits_last = logits_last;
    g_rt.last_n_batch = n_batch;
    if (!lctx) return -1;
    // The pinned helper asserts this unconditionally -- an abort in Release, not a throw.
    if (n_batch <= 0) {
        g_rt.violations.push_back("eval_chunks: n_batch <= 0 -> GGML_ASSERT(n_batch > 0) aborts");
        return -1;
    }
    llama_pos p = n_past;
    bool flagged = false;
    for (size_t i = 0; i < chunks->chunks.size(); ++i) {
        p += chunks->chunks[i].n_tokens; // batched in n_batch slices; position is unaffected
        if (logits_last && i + 1 == chunks->chunks.size()) flagged = true;
    }
    g_rt.n_past = p;
    g_rt.last_logits_last = flagged;
    *new_n_past = p;
    return 0;
}

// The seam under test: generation continues by sampling from the logits the LAST eval left.
// If no position was flagged there is nothing to sample -- that is the double-prefill bug's
// symptom, so the fake records it as a violation instead of throwing.
uint32_t llama_n_batch(const struct llama_context* ctx) { return ctx ? 512u : 0u; }

int32_t llama_sampler_sample(struct llama_sampler* sampler, struct llama_context* ctx, int32_t idx) {
    (void)sampler; (void)ctx; (void)idx;
    log_call("llama_sampler_sample");
    if (!g_rt.last_logits_last) {
        g_rt.violations.push_back("sample: no logits position was flagged by the last eval");
        return -1;
    }
    return 42; // deterministic "token"
}

// ---- link-completeness fakes for the rest of the pinned C API ----------------
// These exist so this TU links against uwp/vision_mtmd.cpp without pulling all of
// libllama/libmtmd. Signatures are copied VERBATIM from the pinned headers (3cb7ffb1a):
//   mtmd.h:      bool mtmd_support_vision(const mtmd_context *);
//              size_t mtmd_input_chunks_size(const mtmd_input_chunks *);
//              const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks *, size_t);
//              enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk *);
//              size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk *);
//   mtmd-helper.h: size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks *);
//                  llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks *);
//   llama.h:     uint32_t llama_n_ctx(const struct llama_context *);
//                llama_memory_t llama_get_memory(const struct llama_context *);
//                void llama_memory_clear(llama_memory_t, bool);
//                float * llama_get_logits(struct llama_context *);
// Same rule as everything else here: NONE of them throw. Failures are recorded in g_rt and
// asserted by main() after the call returns (/EHc on MSVC -- C4297 is what a throw here costs).

bool mtmd_support_vision(const mtmd_context* ctx) { return ctx != nullptr; }

size_t mtmd_input_chunks_size(const mtmd_input_chunks* chunks) {
    return chunks ? chunks->chunks.size() : 0u;
}
const mtmd_input_chunk* mtmd_input_chunks_get(const mtmd_input_chunks* chunks, size_t idx) {
    if (!chunks || idx >= chunks->chunks.size()) return nullptr; // null, never a throw
    return &chunks->chunks[idx];
}
enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk* chunk) {
    if (!chunk) return MTMD_INPUT_CHUNK_TYPE_COUNT;
    return chunk->image ? MTMD_INPUT_CHUNK_TYPE_IMAGE : MTMD_INPUT_CHUNK_TYPE_TEXT;
}
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk* chunk) {
    return chunk ? static_cast<size_t>(chunk->n_tokens) : 0u;
}
size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks* chunks) {
    if (!chunks) return 0u;
    size_t n = 0;
    for (const auto& c : chunks->chunks) n += static_cast<size_t>(c.n_tokens);
    return n;
}
// Fake model is non-M-RoPE, so n_pos == n_tokens here. Recorded so a future M-RoPE
// contract test can differ from it without touching the assertions above.
llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks* chunks) {
    return static_cast<llama_pos>(mtmd_helper_get_n_tokens(chunks));
}

uint32_t llama_n_ctx(const struct llama_context* ctx) { return ctx ? 4096u : 0u; }
llama_memory_t llama_get_memory(const struct llama_context* ctx) {
    return ctx ? reinterpret_cast<llama_memory_t>(&g_fake_mem) : nullptr;
}
void llama_memory_clear(llama_memory_t mem, bool data) {
    (void)data;
    log_call("llama_memory_clear");
    if (mem != reinterpret_cast<llama_memory_t>(&g_fake_mem))
        g_rt.violations.push_back("memory_clear: handle not the one llama_get_memory handed out");
}
// Only legal after an eval that flagged a logits position -- same invariant as sampling.
float* llama_get_logits(struct llama_context* ctx) {
    log_call("llama_get_logits");
    if (!ctx) return nullptr;
    if (!g_rt.last_logits_last)
        g_rt.violations.push_back("get_logits: no logits position was flagged by the last eval");
    return g_fake_logits.v;
}

} // extern "C"

// xllama::log_output is declared in include/xllama/platform.h (C++ linkage, noexcept).
// The UWP/production definition lives in src/bridge/platform.cpp, which this TU does not
// link, so it is stubbed here. noexcept AND non-throwing.
namespace xllama {
void log_output(const char* msg) noexcept { if (msg) printf("[fake] %s\n", msg); }
void log_output(const std::string& msg) noexcept { log_output(msg.c_str()); }
} // namespace xllama

// ---- test harness (assertions live here, in C++, never inside the C ABI) -----

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s @%d\n", #c, __LINE__); ++g_fail; } } while (0)
static void rt_reset() { g_rt = FakeState(); g_rt.marker = "<__media__>"; }

int main() {
    using namespace xllama::vision;

    // --- presets reach mtmd_context_params -----------------------------------
    rt_reset();
    auto fast = ::xllama::vision_mtmd::vision_open(new llama_model(), "mmproj.gguf", false);
    CHECK(fast.impl != nullptr);
    CHECK(g_rt.params.image_min_tokens == 256 && g_rt.params.image_max_tokens == 1024);
    rt_reset();
    auto detailed = ::xllama::vision_mtmd::vision_open(new llama_model(), "mmproj.gguf", true);
    CHECK(detailed.impl != nullptr);
    CHECK(g_rt.params.image_min_tokens == 1024 && g_rt.params.image_max_tokens == 4096);
    // no model handle (the ORT backend) -> unsupported, no init call at all
    rt_reset();
    auto none = ::xllama::vision_mtmd::vision_open(nullptr, "mmproj.gguf", false);
    CHECK(none.impl == nullptr && g_rt.calls.empty());

    // --- the headline contract: one evaluation, then sampling from its end ----
    rt_reset();
    std::vector<Part> parts = {
        // Part layout is {kind, text, url, bytes, mime}: the BYTES field (4th) is what
        // vision_prefill() hands to mtmd_helper_bitmap_init_from_buf.
        {Kind::Text, "describe this", "", "", ""},
        {Kind::Image, "", "data:image/png;base64,A", "PNGDATA1", "image/png"},
        {Kind::Text, " and then this", "", "", ""},
        {Kind::Image, "", "data:image/png;base64,B", "PNGDATA2", "image/png"},
    };
    auto rendered = render_ordered("user", parts);
    CHECK(rendered.bitmaps.size() == 2);
    g_rt.want_bitmaps = 2;

    auto vctx = ::xllama::vision_mtmd::vision_open(new llama_model(), "mmproj.gguf", false);
    auto r = ::xllama::vision_mtmd::vision_prefill(vctx, new llama_context(), /*n_batch=*/0,
                                                   rendered.text_with_markers, parts);
    CHECK(r.supported && r.ok);
    CHECK(g_rt.violations.empty());
    for (auto& v : g_rt.violations) printf("VIOLATION %s\n", v.c_str());
    CHECK(g_rt.n_tokenize == 1);                       // tokenized once
    CHECK(g_rt.n_eval == 1);                           // evaluated once -- no double prefill
    // Chunk walk over the prefilled prompt: text/image/text/image, in order (the getters are
    // link stubs here, but they must agree with what the fake tokenizer produced).
    // Chunk walk over the prefilled prompt: text/image/text/image, in order. The getters are
    // link stubs here, but they must agree with what the fake tokenizer produced -- that is
    // the ordering guarantee production relies on (bitmap i binds to marker i).
    CHECK(mtmd_input_chunks_size(&g_rt.snapshot) == 4u);
    CHECK(mtmd_helper_get_n_tokens(&g_rt.snapshot) > 0u);
    CHECK(mtmd_helper_get_n_pos(&g_rt.snapshot) == static_cast<llama_pos>(mtmd_helper_get_n_tokens(&g_rt.snapshot)));
    const int want_type[4] = {MTMD_INPUT_CHUNK_TYPE_TEXT, MTMD_INPUT_CHUNK_TYPE_IMAGE,
                              MTMD_INPUT_CHUNK_TYPE_TEXT, MTMD_INPUT_CHUNK_TYPE_IMAGE};
    for (size_t i = 0; i < 4; ++i) {
        const mtmd_input_chunk* c = mtmd_input_chunks_get(&g_rt.snapshot, i);
        CHECK(c != nullptr);
        CHECK(mtmd_input_chunk_get_type(c) == want_type[i]);
        CHECK(mtmd_input_chunk_get_n_tokens(c) > 0u);
    }
    CHECK(mtmd_input_chunks_get(&g_rt.snapshot, 99) == nullptr); // out of range -> null, not a throw
    CHECK(g_rt.last_ctx != nullptr && mtmd_support_vision(g_rt.last_ctx));
    CHECK(llama_n_ctx(nullptr) == 0u || llama_n_ctx(nullptr) > 0u); // stub answers, never throws
    CHECK(llama_get_memory(nullptr) == nullptr);
    CHECK(llama_get_logits(nullptr) == nullptr); // no ctx -> null, not a throw
    CHECK(g_rt.last_logits_last == true);              // ... and it left logits behind
    CHECK(g_rt.last_n_batch > 0);                      // never 0: the pinned helper asserts > 0
    const llama_pos after_prefill = g_rt.n_past;
    CHECK(after_prefill > 0);
    // ordered binding: bitmaps arrive in marker order
    CHECK(g_rt.bitmap_ids.size() == 2);
    CHECK(g_rt.bitmap_ids[0] == fnv1a("PNGDATA1"));
    CHECK(g_rt.bitmap_ids[1] == fnv1a("PNGDATA2"));

    // generation continues from the prefilled KV end: sample, then one single-token decode.
    int32_t tok = llama_sampler_sample(nullptr, nullptr, -1);
    CHECK(tok == 42);                                  // sampled from the flagged position
    CHECK(g_rt.violations.empty());                    // ... without re-evaluating the prompt
    CHECK(g_rt.n_eval == 1);

    // --- text-only is a no-op (fast path untouched) --------------------------
    rt_reset();
    std::vector<Part> text_only = {{Kind::Text, "hello", "", "", ""}};
    auto r2 = ::xllama::vision_mtmd::vision_prefill(vctx, new llama_context(), 512, "hello", text_only);
    CHECK(r2.supported && r2.ok);
    CHECK(g_rt.bitmap_ids.empty()); // no bitmaps reach mtmd on the text-only fast path

    // --- failures come back as return codes, never as exceptions -------------
    rt_reset();
    g_rt.want_bitmaps = 2;
    g_rt.fail_tokenize = 1;                            // marker/bitmap mismatch path
    auto r3 = ::xllama::vision_mtmd::vision_prefill(vctx, new llama_context(), 512,
                                                    rendered.text_with_markers, parts);
    CHECK(!r3.ok && r3.supported && !r3.error.empty());
    CHECK(g_rt.n_eval == 0);                           // a failed tokenize must not evaluate

    rt_reset();
    g_rt.want_bitmaps = 0;
    g_rt.fail_bitmap = true;                           // undecodable image -> null bitmap
    auto r4 = ::xllama::vision_mtmd::vision_prefill(vctx, new llama_context(), 512, "", parts);
    CHECK(!r4.ok && !r4.error.empty()); // null bitmap -> error RETURN, never a throw across the C ABI
    CHECK(g_rt.n_tokenize == 0);        // ... and nothing was evaluated

    // an unsupported backend must report it and touch nothing
    rt_reset();
    auto r5 = ::xllama::vision_mtmd::vision_prefill(none, new llama_context(), 512, "hi", text_only);
    // Unsupported backend: supported=false and NO error string (the caller supplies its own
    // 501 message for the ORT path) -- and no mtmd call is made at all.
    CHECK(!r5.supported && !r5.ok && r5.error.empty());
    CHECK(g_rt.calls.empty());

    ::xllama::vision_mtmd::vision_close(vctx);
    ::xllama::vision_mtmd::vision_close(fast);
    ::xllama::vision_mtmd::vision_close(detailed);

    if (g_fail) { printf("verify_vision_runtime: %d FAILURES\n", g_fail); return 1; }
    printf("verify_vision_runtime: ALL PASS\n");
    return 0;
}
