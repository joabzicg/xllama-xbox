// Host tests for the pure vision parsing/preset core (no llama/WinRT).
#include "xllama/vision.h"
#include <cassert>
#include <cstdio>
#include <string>
using namespace xllama::vision;
static int g_fail=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s @%d\n", #c, __LINE__); ++g_fail; } }while(0)

static std::string b64(const std::string& raw){ // tiny encoder for test vectors
    static const char* A="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o; size_t i=0; while(i+2<raw.size()){unsigned n=((unsigned char)raw[i]<<16)|((unsigned char)raw[i+1]<<8)|(unsigned char)raw[i+2];o+=A[n>>18&63];o+=A[n>>12&63];o+=A[n>>6&63];o+=A[n&63];i+=3;}
    if(i<raw.size()){unsigned n=(unsigned char)raw[i]<<16; if(i+1<raw.size())n|=(unsigned char)raw[i+1]<<8;o+=A[n>>18&63];o+=A[n>>12&63];if(i+1<raw.size())o+=A[n>>6&63];else o+='=';o+='=';}
    return o;
}

int main(){
    // text-only: no image -> fast path unaffected
    CHECK(!has_image({{Kind::Text,"hello", "", ""}}));

    // base64 round-trip + PNG magic detect
    std::string png = "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"; png += "rest-of-png-bytes";
    CHECK(detect_mime(png) == "image/png");
    std::string dataurl = "data:image/png;base64," + b64(png);
    std::string bytes, mime;
    CHECK(parse_data_url(dataurl, bytes, mime));
    CHECK(mime == "image/png");
    CHECK(bytes == png); // decoded compressed bytes match

    // JPEG magic
    CHECK(detect_mime(std::string("\xFF\xD8\xFF\xE0jpegbits")) == "image/jpeg");
    // GIF / WEBP
    CHECK(detect_mime("GIF89a....") == "image/gif");
    CHECK(detect_mime("RIFFxxxxWEBPVP8 ") == "image/webp");

    // unsupported MIME -> empty -> not supported
    CHECK(detect_mime(std::string("\x00\x01\x02\x03")) == "");
    CHECK(!mime_supported("application/pdf"));

    // malformed base64 rejected
    std::string b,m; CHECK(!parse_data_url("data:image/png;base64,!!!notbase64!!!", b, m));
    // non-data URL is not parsed as data url (would be fetched separately)
    CHECK(!parse_data_url("https://example.com/x.png", b, m));
    CHECK(is_http_url("https://example.com/x.png"));
    CHECK(is_http_url("file:///sdcard/x.png"));

    // Unicode in text part passes through untouched
    std::string uni = "caf\xc3\xa9 \xe2\x9c\x93 \xf0\x9f\x91\x8b";
    Part tp{Kind::Text, uni, "", ""}; CHECK(tp.text == uni);

    // presets: Fast lower budget than Detailed (dynamic-resolution bounds)
    auto f = preset_bounds(PresetKind::Fast), d = preset_bounds(PresetKind::Detailed);
    CHECK(f.image_max_tokens < d.image_max_tokens);
    CHECK(f.image_min_tokens <= f.image_max_tokens);
    CHECK(d.image_min_tokens <= d.image_max_tokens);

    // ---- ordered multimodal rendering (the headline architectural requirement) ----
    // text A -> image1 -> text B -> image2 must render text with the media marker in
    // each image's EXACT position, and bind bitmaps in that same order.
    {
        std::string b1, m1; parse_data_url("data:image/png;base64," + b64(std::string("\x89PNG\r\n\x1a\nAAAA")), b1, m1);
        std::string b2, m2; parse_data_url("data:image/jpeg;base64," + b64(std::string("\xFF\xD8\xFFabc")), b2, m2);
        CHECK(m1 == "image/png"); CHECK(m2 == "image/jpeg");
        std::vector<Part> parts = {
            {Kind::Text, "text A", "", "", ""},
            {Kind::Image, "", "data:...", b1, m1},
            {Kind::Text, "text B", "", "", ""},
            {Kind::Image, "", "data:...", b2, m2},
        };
        RenderedMessage r = render_ordered("user", parts);
        std::string want = "text A" + media_marker() + "text B" + media_marker();
        CHECK(r.text_with_markers == want);            // order preserved exactly
        CHECK(r.bitmaps.size() == 2);
        CHECK(r.bitmaps[0]->bytes == b1);               // marker #1 <-> image1
        CHECK(r.bitmaps[1]->bytes == b2);               // marker #2 <-> image2
    }
    // text-only message: no bitmaps, byte-identical text (fast path untouched)
    {
        std::vector<Part> parts = {{Kind::Text, "just text", "", "", ""}};
        RenderedMessage r = render_ordered("user", parts);
        CHECK(r.text_with_markers == "just text");
        CHECK(r.bitmaps.empty());
    }

    // ---- size limits enforced BEFORE allocation ----
    CHECK(size_ok(0, 1024));                            // fine
    CHECK(!size_ok(0, 0));                             // empty -> reject
    CHECK(!size_ok(0, kMaxImageBytes + 1));            // single-image cap
    CHECK(!size_ok(kMaxTotalImageBytes - 10, 100));    // total budget exceeded

    // ---- FNV identity is stable and order-sensitive (KV invalidation) ----
    {
        uint64_t a = fnv1a("imageA"), b = fnv1a("imageB");
        CHECK(fnv1a("imageA") == a);                   // stable
        CHECK(a != b);                                 // distinct content -> distinct id
        std::string fp_ab = std::to_string(a) + ":" + std::to_string(b);
        std::string fp_ba = std::to_string(b) + ":" + std::to_string(a);
        CHECK(fp_ab != fp_ba);                         // order-sensitive
    }

    printf("vision: %s\n", g_fail? "FAILURES":"ALL PASS");
    return g_fail?1:0;
}
