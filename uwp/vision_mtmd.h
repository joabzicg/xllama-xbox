// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Vision prefill via libmtmd at the PINNED llama.cpp rev (3cb7ffb). Compiled ONLY when
// XLLAMA_HAS_MTMD is defined (i.e. the build links tools/mtmd). When off, vision_encode
// returns supported=false and the caller keeps the text-only path — nothing else changes.
//
// Uses the generic libmtmd C API only (no model-specific internals): mtmd_default_marker
// for the media marker, mtmd_helper_bitmap_init_from_buf() to decode compressed bytes ->
// RGB bitmap, mtmd_tokenize with the ordered markers + bitmaps, and
// mtmd_helper_eval_chunks() to prefill text+image chunks into the SAME llama_context that
// holds the conversation KV (one context; no second session). Fast/Detailed presets map to
// mtmd_context_params.image_min_tokens/image_max_tokens.

#pragma once
#include <string>
#include <vector>

namespace xllama { namespace vision { struct Part; } }

namespace xllama::vision_mtmd {

struct VisionResult { bool supported = false; bool ok = false; std::string error; };

// Open (or reuse) an mtmd context on the given llama_model* for mmproj_path, with the
// Fast/Detailed token budget. `model_ptr`/`ctx_ptr` are void* from Session's seam
// (llama_model*, llama_context*). Returns supported=false when XLLAMA_HAS_MTMD is off.
struct VisionCtx {
    void* impl = nullptr; // mtmd_context* when built with mtmd; else null
};

VisionCtx vision_open(void* llama_model, const std::string& mmproj_path, bool detailed);
void      vision_close(VisionCtx& ctx);

// Prefill ordered parts (text + in-position media markers already embedded in each
// text part's position) into the session's KV. `parts` come straight from parse+order.
// Requires an open VisionCtx and a live llama_context. Text-only (no image parts) is a
// no-op (returns ok, nothing to prefill -> normal text generate continues).
VisionResult vision_prefill(VisionCtx& ctx, void* llama_context, int n_batch,
                            const std::string& prompt_with_markers,
                            const std::vector<::xllama::vision::Part>& parts);

} // namespace xllama::vision_mtmd
