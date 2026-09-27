// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Contract tests against the pinned public headers, using fake runtime calls.
// This tests the bridge and generation loop, not image quality or Xbox performance.
#include "decode_loop.h"
#include "mtmd-helper.h"
#include "vision_mtmd.h"
#include "xllama/vision.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

struct llama_context {};
struct llama_model {};
struct mtmd_context {};
struct mtmd_bitmap {
    char id;
};
struct mtmd_input_chunk {};
struct mtmd_input_chunks {};
static llama_context shared_context;
static llama_model shared_model;
static int evals, clears, bitmaps_alive, chunks_alive, contexts_alive, samples;
static int eval_error, tokenize_error, decode_error;
static bool last_is_text = true, throw_tokenize = false;
static size_t tokens = 20;
static std::string received_text, received_images;
static std::vector<llama_pos> decoded_positions;
static mtmd_context_params received_params;
static float logits[1] = {0};

namespace xllama {
void log_output(const char*) noexcept {}
void log_output(const std::string&) noexcept {}
} // namespace xllama

extern "C" {
const char* mtmd_default_marker() {
    return "<__media__>";
}
mtmd_context_params mtmd_context_params_default() {
    return {};
}
mtmd_context* mtmd_init_from_file(const char*, const llama_model* model,
                                  mtmd_context_params params) {
    assert(model == &shared_model);
    received_params = params;
    ++contexts_alive;
    return new mtmd_context;
}
bool mtmd_support_vision(const mtmd_context*) {
    return true;
}
void mtmd_free(mtmd_context* ctx) {
    --contexts_alive;
    delete ctx;
}
mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(mtmd_context*, const unsigned char* buf,
                                                            size_t len, bool placeholder) {
    assert(len && !placeholder);
    ++bitmaps_alive;
    return {new mtmd_bitmap{static_cast<char>(*buf)}, nullptr};
}
void mtmd_bitmap_free(mtmd_bitmap* bitmap) {
    --bitmaps_alive;
    delete bitmap;
}
mtmd_input_chunks* mtmd_input_chunks_init() {
    ++chunks_alive;
    return new mtmd_input_chunks;
}
void mtmd_input_chunks_free(mtmd_input_chunks* chunks) {
    --chunks_alive;
    delete chunks;
}
size_t mtmd_input_chunks_size(const mtmd_input_chunks*) {
    return 3;
}
const mtmd_input_chunk* mtmd_input_chunks_get(const mtmd_input_chunks*, size_t) {
    static mtmd_input_chunk chunk;
    return &chunk;
}
mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk*) {
    return last_is_text ? MTMD_INPUT_CHUNK_TYPE_TEXT : MTMD_INPUT_CHUNK_TYPE_IMAGE;
}
size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk*) {
    return 1;
}
size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks*) {
    return tokens;
}
llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks*) {
    return 7;
}
int32_t mtmd_tokenize(mtmd_context*, mtmd_input_chunks*, const mtmd_input_text* input,
                      const mtmd_bitmap** images, size_t count) {
    if (throw_tokenize)
        throw std::runtime_error("injected tokenize failure");
    received_text.assign(input->text, input->text_len);
    assert(input->add_special && input->parse_special);
    received_images.clear();
    for (size_t i = 0; i < count; ++i)
        received_images += images[i]->id;
    return tokenize_error;
}
int32_t mtmd_helper_eval_chunks(mtmd_context*, llama_context* ctx, const mtmd_input_chunks*,
                                llama_pos past, llama_seq_id seq, int32_t batch, bool last,
                                llama_pos* next) {
    assert(ctx == &shared_context && past == 0 && seq == 0 && batch == 8 && last);
    ++evals;
    *next = 7;
    return eval_error;
}
uint32_t llama_n_ctx(const llama_context*) {
    return 32;
}
uint32_t llama_n_batch(const llama_context*) {
    return 8;
}
llama_memory_t llama_get_memory(const llama_context* ctx) {
    assert(ctx == &shared_context);
    return reinterpret_cast<llama_memory_t>(&shared_context);
}
void llama_memory_clear(llama_memory_t, bool data) {
    assert(data);
    ++clears;
}
float* llama_get_logits(llama_context* ctx) {
    assert(ctx == &shared_context);
    return logits;
}
llama_batch llama_batch_get_one(llama_token* token, int32_t count) {
    llama_batch batch{};
    batch.token = token;
    batch.n_tokens = count;
    return batch;
}
int32_t llama_decode(llama_context* ctx, llama_batch batch) {
    assert(ctx == &shared_context && batch.n_tokens == 1 && batch.pos);
    decoded_positions.push_back(*batch.pos);
    return decode_error;
}
llama_token llama_sampler_sample(llama_sampler*, llama_context* ctx, int32_t index) {
    assert(ctx == &shared_context && index == -1 && evals == 1);
    ++samples;
    return 1;
}
bool llama_vocab_is_eog(const llama_vocab*, llama_token) {
    return false;
}
int32_t llama_token_to_piece(const llama_vocab*, llama_token, char* buffer, int32_t size, int32_t,
                             bool) {
    assert(size > 0);
    buffer[0] = 'x';
    return 1;
}
// Speculation is disabled by explicit multimodal positions; these must never run.
llama_batch llama_batch_init(int32_t, int32_t, int32_t) {
    assert(false);
    return {};
}
void llama_batch_free(llama_batch) {
    assert(false);
}
llama_pos llama_memory_seq_pos_max(llama_memory_t, llama_seq_id) {
    assert(false);
    return 0;
}
bool llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos, llama_pos) {
    assert(false);
    return false;
}
}

int main() {
    using namespace xllama;
    using namespace xllama::vision_mtmd;
    std::vector<vision::Part> images(2);
    images[0].kind = images[1].kind = vision::Kind::Image;
    images[0].bytes = "A";
    images[1].bytes = "B";
    const std::string prompt = "before<__media__>between<__media__>assistant";
    {
        auto ctx = vision_open(&shared_model, "mmproj.gguf", false);
        assert(received_params.image_min_tokens == 256 && received_params.image_max_tokens == 1024);
        auto r = vision_prefill(ctx, &shared_context, 0, prompt, images, 4);
        assert(r.ok && evals == 1 && clears == 1);
        assert(r.prefill.n_tokens == 20 && r.prefill.next_position == 7);
        assert(received_text == prompt && received_images == "AB");
        assert(bitmaps_alive == 0 && chunks_alive == 0);
        DecodeLoopParams params;
        params.ctx = &shared_context;
        params.n_predict = 3;
        params.next_position = r.prefill.next_position;
        params.prompt_lookup = true; // position seam must disable speculation
        std::vector<llama_token> history(16, 1);
        params.token_history = &history;
        std::string answer;
        auto result = decode_loop(params, answer);
        assert(answer == "xxx" && result.n_generated == 3 && samples == 3 && evals == 1);
        assert((decoded_positions == std::vector<llama_pos>{7, 8, 9}));
        decode_error = 1;
        result = decode_loop(params, answer);
        assert(result.decode_failed && result.n_generated == 0);
        decode_error = 0;
        tokens = 30;
        r = vision_prefill(ctx, &shared_context, 0, prompt, images, 4);
        assert(!r.ok && evals == 1); // capacity counts 30 KV cells, not 7 M-RoPE positions
        tokens = 20;
        eval_error = 1;
        const int clears_before = clears;
        r = vision_prefill(ctx, &shared_context, 0, prompt, images, 4);
        assert(!r.ok && clears == clears_before + 2);
        eval_error = 0;
        last_is_text = false;
        r = vision_prefill(ctx, &shared_context, 0, prompt, images, 4);
        assert(!r.ok);
        last_is_text = true;
        throw_tokenize = true;
        try {
            vision_prefill(ctx, &shared_context, 0, prompt, images, 4);
            assert(false);
        } catch (const std::runtime_error&) {
        }
        assert(bitmaps_alive == 0 && chunks_alive == 0);
    }
    assert(contexts_alive == 0);
    {
        auto detailed = vision_open(&shared_model, "mmproj.gguf", true);
        assert(received_params.image_min_tokens == 1024 &&
               received_params.image_max_tokens == 4096);
        auto moved = std::move(detailed);
        assert(!detailed.impl && moved.impl);
    }
    assert(contexts_alive == 0);
    std::cout << "vision runtime contract: PASS (fake backend, pinned headers)\n";
}
