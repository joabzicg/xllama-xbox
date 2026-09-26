// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// mtmd vision prefill implementation. See vision_mtmd.h. Compiled only with mtmd linked.

#include "vision_mtmd.h"
#include "xllama/vision.h"

#ifdef XLLAMA_HAS_MTMD
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#endif

namespace xllama::vision_mtmd {

#ifdef XLLAMA_HAS_MTMD
struct VisionCtxImpl { mtmd_context* ctx = nullptr; };
static mtmd_context_params preset_params(bool detailed) {
    mtmd_context_params p = mtmd_context_params_default();
    // Fast/Detailed presets map straight onto libmtmd's dynamic-resolution bounds.
    Preset b = preset_bounds(detailed ? PresetKind::Detailed : PresetKind::Fast);
    p.image_min_tokens = b.image_min_tokens;
    p.image_max_tokens = b.image_max_tokens;
    return p;
}
VisionCtx vision_open(void* llama_model, const std::string& mmproj_path, bool detailed) {
    VisionCtx vc;
    if (llama_model == nullptr || mmproj_path.empty()) return vc; // no vision on this backend
    auto* impl = new VisionCtxImpl();
    impl->ctx = mtmd_init_from_file(mmproj_path.c_str(), (llama_model*)llama_model, preset_params(detailed));
    if (!impl->ctx) { delete impl; return vc; }
    vc.impl = impl;
    return vc;
}
void vision_close(VisionCtx& ctx) {
    if (!ctx.impl) return;
    auto* impl = (VisionCtxImpl*)ctx.impl;
    if (impl->ctx) mtmd_free(impl->ctx);
    delete impl; ctx.impl = nullptr;
}
VisionResult vision_prefill(VisionCtx& ctx, void* llama_context, int n_batch,
                            const std::string& prompt_with_markers,
                            const std::vector<::xllama::vision::Part>& parts) {
    VisionResult r;
    if (!ctx.impl || !llama_context) { r.supported = false; return r; }
    r.supported = true;
    mtmd_context* mctx = ((VisionCtxImpl*)ctx.impl)->ctx;

    // Build the bitmap list in ORDER and count markers; mtmd substitutes the i-th
    // bitmap at the i-th media marker, so order is preserved by construction. The prompt
    // already carries one default marker per image part (parse+render_ordered did that).
    std::vector<mtmd_bitmap*> bitmaps;
    auto cleanup = [&]{ for (auto* b : bitmaps) mtmd_bitmap_free(b); };
    for (const auto& p : parts) {
        if (p.kind != ::xllama::vision::Kind::Image || p.bytes.empty()) continue;
        // helper decodes compressed bytes -> RGB internally (stb_image), FNV id.
        auto w = mtmd_helper_bitmap_init_from_buf(mctx, (const unsigned char*)p.bytes.data(), p.bytes.size(), false);
        if (!w.bitmap) { cleanup(); r.error = "mtmd: could not decode an image bitmap"; return r; }
        bitmaps.push_back(w.bitmap);
    }

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text it{}; it.text = prompt_with_markers.c_str(); it.text_len = prompt_with_markers.size();
    it.add_special = true; it.parse_special = true;
    int32_t rc = mtmd_tokenize(mctx, chunks, &it, bitmaps.data(), bitmaps.size());
    if (rc != 0) { cleanup(); mtmd_input_chunks_free(chunks); r.error = "mtmd_tokenize failed (" + std::to_string(rc) + ")"; return r; }

    // Prefill text+image chunks into the SAME llama_context that holds the conversation KV.
    llama_pos new_n_past = 0;
    rc = mtmd_helper_eval_chunks(mctx, (llama_context*)llama_context, chunks, /*n_past=*/0, /*seq_id=*/0, n_batch, /*logits_last=*/false, &new_n_past);
    cleanup(); mtmd_input_chunks_free(chunks);
    if (rc != 0) { r.error = "mtmd_helper_eval_chunks failed (" + std::to_string(rc) + ")"; return r; }
    r.ok = true;
    return r;
}

#else // !XLLAMA_HAS_MTMD : vision not compiled in; text-only path is unaffected.
VisionCtx vision_open(void*, const std::string&, bool) { return VisionCtx{}; }
void      vision_close(VisionCtx&) {}
VisionResult vision_prefill(VisionCtx&, void*, int, const std::string&, const std::vector<::xllama::vision::Part>&) {
    VisionResult r; r.supported = false; r.error = "vision not compiled (XLLAMA_HAS_MTMD off)"; return r;
}
#endif

} // namespace xllama::vision_mtmd
