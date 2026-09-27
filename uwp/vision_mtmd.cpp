// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// mtmd vision prefill implementation. See vision_mtmd.h.

#include "vision_mtmd.h"
#include "xllama/platform.h"
#include "xllama/vision.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <memory>
#include <utility>

#if defined(XLLAMA_UWP) && defined(XLLAMA_USE_LLAMA) && !defined(XLLAMA_HAS_MTMD)
    #error "llama-enabled UWP builds require mtmd"
#endif

#ifdef XLLAMA_HAS_MTMD
    #include "llama.h"
    #include "mtmd-helper.h"
    #include "mtmd.h"
#endif

namespace xllama::vision_mtmd {

VisionCtx::~VisionCtx() {
    vision_close(*this);
}
VisionCtx::VisionCtx(VisionCtx&& other) noexcept : impl(std::exchange(other.impl, nullptr)) {}
VisionCtx& VisionCtx::operator=(VisionCtx&& other) noexcept {
    if (this != &other) {
        vision_close(*this);
        impl = std::exchange(other.impl, nullptr);
    }
    return *this;
}

#ifdef XLLAMA_HAS_MTMD
using MtmdPtr = std::unique_ptr<mtmd_context, decltype(&mtmd_free)>;
using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using ChunksPtr = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

const char* vision_marker() {
    return mtmd_default_marker();
}

VisionCtx vision_open(void* model_ptr, const std::string& mmproj_path, bool detailed) {
    VisionCtx vc;
    if (!model_ptr || mmproj_path.empty())
        return vc;
    mtmd_context_params params = mtmd_context_params_default();
    const auto bounds = ::xllama::vision::preset_bounds(
        detailed ? ::xllama::vision::PresetKind::Detailed : ::xllama::vision::PresetKind::Fast);
    params.image_min_tokens = bounds.image_min_tokens;
    params.image_max_tokens = bounds.image_max_tokens;
    params.use_gpu = false; // Xbox uses the resident CPU ggml backend.
    MtmdPtr ctx(mtmd_init_from_file(mmproj_path.c_str(),
                                    static_cast<struct llama_model*>(model_ptr), params),
                mtmd_free);
    if (ctx && mtmd_support_vision(ctx.get()))
        vc.impl = ctx.release();
    return vc;
}

void vision_close(VisionCtx& ctx) {
    if (ctx.impl)
        mtmd_free(static_cast<mtmd_context*>(std::exchange(ctx.impl, nullptr)));
}

VisionResult vision_prefill(VisionCtx& ctx, void* ctx_ptr, int n_batch,
                            const std::string& prompt_with_markers,
                            const std::vector<::xllama::vision::Part>& parts, int reserve_tokens) {
    VisionResult r;
    r.supported = true;
    if (!ctx.impl || !ctx_ptr) {
        r.error = "vision projector or shared llama context is unavailable";
        return r;
    }
    auto* mctx = static_cast<mtmd_context*>(ctx.impl);
    auto* lctx = static_cast<struct llama_context*>(ctx_ptr);
    // Also guards exceptions during decoding/tokenization. Never leave a partial
    // multimodal prefix resident after an error.
    struct PrefillGuard {
        llama_context* ctx;
        bool keep = false;
        ~PrefillGuard() {
            if (!keep)
                llama_memory_clear(llama_get_memory(ctx), true);
        }
    } guard{lctx};
    llama_memory_clear(llama_get_memory(lctx), true);

    const auto image_start = std::chrono::steady_clock::now();
    std::vector<BitmapPtr> owners;
    std::vector<const mtmd_bitmap*> bitmaps;
    owners.reserve(parts.size());
    bitmaps.reserve(parts.size());
    for (const auto& part : parts) {
        if (part.kind != ::xllama::vision::Kind::Image)
            continue;
        if (part.bytes.empty()) {
            r.error = "image payload is empty";
            return r;
        }
        // Decode compressed bytes directly with the pin's stb_image helper.
        const auto bitmap = mtmd_helper_bitmap_init_from_buf(
            mctx, reinterpret_cast<const unsigned char*>(part.bytes.data()), part.bytes.size(),
            false);
        BitmapPtr owner(bitmap.bitmap, mtmd_bitmap_free);
        if (!owner) {
            r.error = "mtmd could not decode an image bitmap";
            return r;
        }
        bitmaps.push_back(owner.get());
        owners.push_back(std::move(owner));
    }
    r.image_decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - image_start)
            .count();
    if (bitmaps.empty()) {
        r.error = "multimodal prefill requires an image";
        return r;
    }
    // The i-th ordered bitmap replaces the i-th marker. mtmd also checks marker
    // count; there is no independent flattening of text and images here.
    ChunksPtr chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    if (!chunks) {
        r.error = "mtmd could not allocate input chunks";
        return r;
    }
    mtmd_input_text input{prompt_with_markers.c_str(), prompt_with_markers.size(), true, true};
    int32_t rc = mtmd_tokenize(mctx, chunks.get(), &input, bitmaps.data(), bitmaps.size());
    r.image_preprocess_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - image_start)
            .count();
    if (rc != 0) {
        r.error = "mtmd_tokenize failed (" + std::to_string(rc) + ")";
        return r;
    }
    const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
    const size_t n_tokens = mtmd_helper_get_n_tokens(chunks.get());
    const auto n_positions = mtmd_helper_get_n_pos(chunks.get());
    const size_t n_ctx = llama_n_ctx(lctx);
    const size_t reserve = static_cast<size_t>(std::max(reserve_tokens, 1));
    if (n_tokens == 0 || n_tokens > INT_MAX || n_tokens >= n_ctx || reserve > n_ctx - n_tokens ||
        n_positions <= 0 || n_positions > INT_MAX - static_cast<int>(reserve)) {
        r.error = "multimodal prompt and reply exceed the context capacity";
        return r;
    }
    // The pin only honors logits_last for a nonempty TEXT chunk. Chat templates
    // append the assistant header after media; fail explicitly if that is absent.
    const auto* last = n_chunks ? mtmd_input_chunks_get(chunks.get(), n_chunks - 1) : nullptr;
    if (!last || mtmd_input_chunk_get_type(last) != MTMD_INPUT_CHUNK_TYPE_TEXT ||
        mtmd_input_chunk_get_n_tokens(last) == 0) {
        r.error = "multimodal prompt must end with the assistant text header";
        return r;
    }
    const int batch_limit = static_cast<int>(llama_n_batch(lctx));
    if (batch_limit <= 0) {
        r.error = "shared context has no decode batch capacity";
        return r;
    }
    n_batch = n_batch <= 0 ? batch_limit : std::min(n_batch, batch_limit);
    const auto prefill_start = std::chrono::steady_clock::now();
    llama_pos next_position = 0;
    // Exactly one ordered prefill on the shared context; seed the first sample
    // from the final prompt token's logits, without another Session::generate.
    rc = mtmd_helper_eval_chunks(mctx, lctx, chunks.get(), 0, 0, n_batch, true, &next_position);
    r.prefill.prefill_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_start)
            .count();
    if (rc != 0 || next_position != n_positions || !llama_get_logits(lctx)) {
        r.error = "mtmd prefill failed or did not produce continuation logits (" +
                  std::to_string(rc) + ")";
        return r;
    }
    r.prefill.n_tokens = static_cast<int>(n_tokens);
    r.prefill.next_position = next_position;
    guard.keep = true;
    r.ok = true;
    char log[240];
    snprintf(log, sizeof(log),
             "[xllama] vision: image_decode=%.1fms image_preprocess=%.1fms prefill=%.1fms "
             "kv_tokens=%d next_position=%d\n",
             r.image_decode_ms, r.image_preprocess_ms, r.prefill.prefill_ms, r.prefill.n_tokens,
             next_position);
    log_output(log);
    return r;
}

#else
const char* vision_marker() {
    return "<__media__>";
}
VisionCtx vision_open(void*, const std::string&, bool) {
    return VisionCtx{};
}
void vision_close(VisionCtx& ctx) {
    ctx.impl = nullptr;
}
VisionResult vision_prefill(VisionCtx&, void*, int, const std::string&,
                            const std::vector<::xllama::vision::Part>&, int) {
    VisionResult r;
    r.error = "vision not compiled (XLLAMA_HAS_MTMD off)";
    return r;
}
#endif

} // namespace xllama::vision_mtmd
