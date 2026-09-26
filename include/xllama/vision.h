// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Vision request parsing + preset selection for the LAN multimodal endpoint.
// Header-only, WinRT-free and host-testable (mirrors sse.h / json_utils.h). It does
// NOT call libmtmd; it turns an OpenAI content-part list into (bytes, mime) pairs and
// maps a vision preset to mtmd_context_params bounds. The actual mtmd calls live in
// the UWP server (guarded by XLLAMA_UWP + LLAMA_MTMD).
//
// Decoding path per the pinned llama.cpp audit (3cb7ffb): base64/data-URL -> decoded
// COMPRESSED bytes -> mtmd_helper_bitmap_init_from_buf() (decodes to RGB internally).
// No Windows.Graphics.Imaging unless that helper is proven to fail under UWP.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {
namespace vision {

enum class Kind { Text, Image };

struct Part {
    Kind kind = Kind::Text;
    std::string text;         // for Text parts
    std::string url;          // image_url.url (data: or http(s):/file:)
    // Populated after materialize(): decoded COMPRESSED image bytes (png/jpeg/gif/webp)
    std::string bytes;        // raw compressed bytes for mtmd_helper_bitmap_init_from_buf
    std::string mime;         // detected/canonical MIME, e.g. "image/png"
};

// ---- base64 (standard alphabet, tolerant of whitespace/padding) -------------
inline bool base64_decode(const std::string& in, std::string& out) {
    static int T[256]; static bool init = false;
    if (!init) { for (int i=0;i<256;++i) T[i]=-1; const char* A="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; for (int i=0;i<64;++i) T[(unsigned char)A[i]]=i; init=true; }
    out.clear(); int val=0, bits=-8;
    for (unsigned char c : in) {
        if (c=='\n'||c=='\r'||c==' '||c=='\t') continue; // tolerate wrapping whitespace
        if (c=='=') break;
        if (T[c] < 0) return false; // invalid base64 byte
        val = (val << 6) | T[c]; bits += 6;
        if (bits >= 0) { out.push_back(char((val >> bits) & 0xFF)); bits -= 8; }
    }
    return true;
}

// Detect a canonical MIME from magic bytes (authoritative over any declared type).
inline std::string detect_mime(const std::string& b) {
    auto p=[&](std::initializer_list<int> sig, size_t off=0){ if (b.size()<off+sig.size()) return false; size_t i=0; for (int c: sig){ if ((unsigned char)b[off+i]!=(unsigned char)c) return false; ++i;} return true; };
    if (p({0x89,0x50,0x4E,0x47})) return "image/png";
    if (p({0xFF,0xD8,0xFF}))      return "image/jpeg";
    if (p({0x47,0x49,0x46,0x38})) return "image/gif";   // GIF8
    if (b.size()>=12 && b.compare(0,4,"RIFF")==0 && b.compare(8,4,"WEBP")==0) return "image/webp";
    return ""; // unsupported / unknown
}

inline bool mime_supported(const std::string& m) {
    return m=="image/png" || m=="image/jpeg" || m=="image/gif" || m=="image/webp";
}

// Parse a data: URL. Returns false if not a data URL or malformed base64. On success
// fills bytes (decoded compressed bytes) and mime (from magic bytes, authoritative).
inline bool parse_data_url(const std::string& url, std::string& bytes, std::string& mime) {
    const std::string pfx = "data:";
    if (url.compare(0, pfx.size(), pfx) != 0) return false;
    size_t comma = url.find(',');
    if (comma == std::string::npos) return false;
    std::string meta = url.substr(pfx.size(), comma - pfx.size()); // e.g. image/png;base64
    bool is_b64 = meta.find("base64") != std::string::npos;
    std::string declared = meta.substr(0, meta.find(';'));
    std::string payload = url.substr(comma + 1);
    if (is_b64) { if (!base64_decode(payload, bytes)) return false; }
    else { // percent-decoded raw
        for (size_t i=0;i<payload.size();++i){ if(payload[i]=='%'&&i+2<payload.size()){int v=std::stoi(payload.substr(i+1,2),nullptr,16);bytes.push_back((char)v);i+=2;} else bytes.push_back(payload[i]); }
    }
    mime = detect_mime(bytes);
    if (mime.empty()) mime = declared; // fall back to declared type if magic unknown
    return true;
}

inline bool is_http_url(const std::string& url) {
    return url.compare(0,7,"http://")==0 || url.compare(0,8,"https://")==0 || url.compare(0,5,"file:")==0;
}

// Vision presets -> mtmd_context_params.image_min_tokens / image_max_tokens.
// Fast = lower visual token/resolution budget (speed); Detailed = higher (OCR/detail).
struct Preset { int image_min_tokens; int image_max_tokens; };
enum class PresetKind { Fast, Detailed };
inline Preset preset_bounds(PresetKind k) {
    // Tunable defaults; real values confirmed on-console. Qwen2.5-VL supports dynamic
    // resolution so these are honoured by libmtmd (no manual image resize emulation).
    return (k == PresetKind::Fast) ? Preset{256, 1024} : Preset{1024, 4096};
}

// Classify a content-part list: does it contain any image? (text-only fast path check).
inline bool has_image(const std::vector<Part>& parts) {
    for (auto& p : parts) if (p.kind == Kind::Image) return true;
    return false;
}

// The media marker libmtmd substitutes each bitmap into. MUST equal
// mtmd_default_marker() at the pinned llama.cpp rev (3cb7ffb => "<__media__>").
// Kept as a parameter default so the pure parser is testable without linking mtmd.
inline const std::string& media_marker() {
    static const std::string m = "<__media__>";
    return m;
}

// FNV-1a 64-bit over bytes — stable image identity (matches libmtmd's own bitmap-ID
// convention, so a fingerprint is cheap and reproducible for KV-invalidation).
inline uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// An ordered message: role + parts IN THE ORIGINAL OPENAI ORDER. Rendering inserts the
// media marker in each image's exact position and returns the parallel bitmap vector in
// that same order, so "text A -> image1 -> text B -> image2" becomes
// "text A<__media__>text B<__media__>" with bitmaps [image1, image2]. Text-only messages
// produce no bitmaps and byte-identical text (fast path untouched).
struct RenderedMessage {
    std::string role;
    std::string text_with_markers;   // text parts joined with the media marker in position
    std::vector<const Part*> bitmaps; // ordered; index i <-> i-th marker occurrence
};

inline RenderedMessage render_ordered(const std::string& role, const std::vector<Part>& parts) {
    RenderedMessage r;
    r.role = role;
    for (const auto& p : parts) {
        if (p.kind == Kind::Text) {
            r.text_with_markers += p.text;
        } else { // Image -> marker in position + bind bitmap by index
            r.text_with_markers += media_marker();
            if (!p.bytes.empty()) // only decodable images become bitmaps
                r.bitmaps.push_back(&p);
        }
    }
    return r;
}

// ---- size limits, enforced BEFORE base64 allocation (Xbox OOM guard) ----------
inline constexpr size_t kMaxImageBytes = 16u * 1024u * 1024u;   // per decoded image (compressed)
inline constexpr size_t kMaxTotalImageBytes = 32u * 1024u * 1024u; // all images in one request

// True if accepting this many more decoded bytes stays within both budgets.
inline bool size_ok(size_t total_so_far, size_t incoming_decoded) {
    if (incoming_decoded == 0) return false;              // empty/undecodable -> reject
    if (incoming_decoded > kMaxImageBytes) return false;   // single-image cap
    if (total_so_far + incoming_decoded > kMaxTotalImageBytes) return false;
    return true;
}

} // namespace vision
} // namespace xllama
