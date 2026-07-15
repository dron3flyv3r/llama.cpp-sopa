#include "sopa-generate.h"
#include "sopa-manager.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "server-chat.h"
#include "server-common.h"
#include "speculative.h"
#include "mtmd-helper.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

// Control markers that some templates can emit as regular text before an EOG token.
// The streamer withholds matching prefixes so it can suppress a marker without
// already having sent its first token pieces to the client.
static const std::vector<std::string> STOP_STRINGS = {
    "<end_of_turn>",
    "</start_of_turn>",
    "<start_of_turn>",
    "<turn|>",
};

// Generation state shared between successive next() calls in the streaming response.
struct SopaGenCtx {
    llama_context *     ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler *    common_smpl = nullptr;
    common_speculative * spec = nullptr;
    common_params_speculative spec_params;
    llama_batch         batch_tgt = {};
    bool                batch_tgt_init = false;

    std::vector<llama_token> prompt_tokens;
    server_tokens       media_tokens;
    mtmd::batch_ptr     media_batch;
    bool                has_media = false;
    std::vector<llama_token> prompt_tgt;
    std::vector<llama_token> gen_token_ids;  // generated token IDs, for KV reuse
    llama_token last_token = -1;
    llama_token id_last = -1;
    int32_t     n_past = 0;
    int32_t     n_prefix = 0;   // tokens already in KV cache (prefix reuse)
    int32_t     n_generated = 0;
    int32_t     max_tokens  = 512;
    bool        prefill_done = false;
    bool        use_mtp = false;
    std::string pending;  // text withheld until it is known not to be a stop marker
    std::vector<std::string> stop_strings;
    std::string generated_text;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    bool inference_active = false;
    int32_t n_drafted = 0;
    int32_t n_accepted = 0;
    bool sent_meta = false;
    std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();

    ~SopaGenCtx() {
        if (inference_active) {
            g_sopa.on_inference_end();
            inference_active = false;
        }
        if (common_smpl) {
            common_sampler_free(common_smpl);
        }
        if (spec) {
            common_speculative_free(spec);
        }
        if (batch_tgt_init) {
            llama_batch_free(batch_tgt);
        }
    }
};

static std::string sse_chunk(const std::string & content) {
    json chunk = {
        {"id",      "sopa"},
        {"object",  "chat.completion.chunk"},
        {"choices", json::array({{
            {"index",         0},
            {"delta",         {{"content", content}}},
            {"finish_reason", nullptr}
        }})}
    };
    return "data: " + chunk.dump() + "\n\n";
}

static std::string sse_delta(const json & delta) {
    if (delta.empty()) {
        return "";
    }
    json chunk = {
        {"id",      "sopa"},
        {"object",  "chat.completion.chunk"},
        {"choices", json::array({{
            {"index",         0},
            {"delta",         delta},
            {"finish_reason", nullptr}
        }})}
    };
    return "data: " + chunk.dump() + "\n\n";
}

static json sopa_context_json(const SopaGenCtx & gen) {
    const double acceptance = gen.n_drafted > 0
        ? (double) gen.n_accepted / (double) gen.n_drafted
        : 0.0;
    return json{
        {"n_ctx",            llama_n_ctx(gen.ctx)},
        {"n_prompt_tokens",  (int32_t) gen.prompt_tokens.size()},
        {"n_prefix_tokens",  gen.n_prefix},
        {"n_generated",      gen.n_generated},
        {"mtp_enabled",      gen.use_mtp},
        {"mtp_drafted",      gen.n_drafted},
        {"mtp_accepted",     gen.n_accepted},
        {"mtp_acceptance",   acceptance},
    };
}

static json usage_json(const SopaGenCtx & gen) {
    const int32_t prompt_tokens = (int32_t) gen.prompt_tokens.size();
    const int32_t completion_tokens = gen.n_generated;
    return json{
        {"prompt_tokens",     prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens",      prompt_tokens + completion_tokens},
        {"prompt_tokens_details", json{{"cached_tokens", gen.n_prefix}}},
    };
}

static std::string sse_meta(const SopaGenCtx & gen) {
    json chunk = {
        {"id",           "sopa"},
        {"object",       "chat.completion.chunk"},
        {"choices",      json::array()},
        {"usage",        usage_json(gen)},
        {"sopa_context", sopa_context_json(gen)},
    };
    return "data: " + chunk.dump() + "\n\n";
}

static std::string sse_finish(const SopaGenCtx & gen, const std::string & reason) {
    json chunk = {
        {"id",           "sopa"},
        {"object",       "chat.completion.chunk"},
        {"choices", json::array({{
            {"index",         0},
            {"delta",         json::object()},
            {"finish_reason", reason}
        }})},
        {"usage",        usage_json(gen)},
        {"sopa_context", sopa_context_json(gen)},
    };
    return "data: " + chunk.dump() + "\n\n";
}

static std::string sse_done() {
    return "data: [DONE]\n\n";
}

static void add_unique_stop(std::vector<std::string> & stops, const std::string & stop) {
    if (stop.empty()) {
        return;
    }
    if (std::find(stops.begin(), stops.end(), stop) == stops.end()) {
        stops.push_back(stop);
    }
}

static size_t stop_prefix_suffix_len(const std::string & text, const std::vector<std::string> & stops) {
    size_t max_len = 0;
    for (const auto & stop : stops) {
        if (stop.size() <= 1) {
            continue;
        }
        max_len = std::max(max_len, stop.size() - 1);
    }

    const size_t upper = std::min(max_len, text.size());
    for (size_t len = upper; len > 0; --len) {
        const std::string_view suffix(text.data() + text.size() - len, len);
        for (const auto & stop : stops) {
            if (stop.size() >= len && std::string_view(stop.data(), len) == suffix) {
                return len;
            }
        }
    }
    return 0;
}

static std::string sse_finish_done(SopaGenCtx & gen, const std::string & reason) {
    return sse_finish(gen, reason) + sse_done();
}

static void append_parsed_chat_deltas(SopaGenCtx & gen, const std::string & text, bool is_partial, std::string & output);

static bool request_mtp_enabled(const json & body) {
    if (body.contains("mtp") && body["mtp"].is_boolean()) {
        return body["mtp"].get<bool>();
    }
    if (body.contains("extra_body") && body["extra_body"].is_object() &&
        body["extra_body"].contains("mtp") && body["extra_body"]["mtp"].is_boolean()) {
        return body["extra_body"]["mtp"].get<bool>();
    }
    return true;
}

static float json_float_or(const json & body, const char * key, float fallback) {
    if (!body.contains(key) || !body[key].is_number()) {
        return fallback;
    }
    return body[key].get<float>();
}

static int32_t json_i32_or(const json & body, const char * key, int32_t fallback) {
    if (!body.contains(key) || !body[key].is_number_integer()) {
        return fallback;
    }
    return body[key].get<int32_t>();
}

static uint32_t json_u32_or(const json & body, const char * key, uint32_t fallback) {
    if (!body.contains(key) || !body[key].is_number_unsigned()) {
        return fallback;
    }
    return body[key].get<uint32_t>();
}

static common_params_sampling make_sampling_params(const json & body) {
    common_params_sampling params;
    params.seed  = json_u32_or(body, "seed", LLAMA_DEFAULT_SEED);
    params.temp  = json_float_or(body, "temperature", 0.8f);
    params.min_p = json_float_or(body, "min_p", 0.05f);
    params.top_k = json_i32_or(body, "top_k", 0);
    params.top_p = json_float_or(body, "top_p", 1.0f);
    params.typ_p = 1.0f;
    // Repetition control: near-greedy sampling (low temperature) with no penalties
    // makes long thinking/agent generations degenerate into repetition loops.
    // Mild defaults apply unless the request overrides them.
    params.penalty_last_n     = json_i32_or(body, "repeat_last_n", 256);
    params.penalty_repeat     = json_float_or(body, "repeat_penalty", 1.05f);
    params.penalty_freq       = json_float_or(body, "frequency_penalty", 0.0f);
    params.penalty_present    = json_float_or(body, "presence_penalty", 0.0f);
    params.dry_multiplier     = json_float_or(body, "dry_multiplier", 0.0f);
    params.dry_base           = json_float_or(body, "dry_base", 1.75f);
    params.dry_allowed_length = json_i32_or(body, "dry_allowed_length", 2);
    params.dry_penalty_last_n = json_i32_or(body, "dry_penalty_last_n", -1);
    params.samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    return params;
}

static void finish_inference(SopaGenCtx & gen) {
    if (gen.inference_active) {
        if (!gen.use_mtp && !gen.has_media) {
            // Save prompt + generated tokens so the next request can reuse the KV cache
            // for any matching prefix, avoiding a full re-prefill of the conversation.
            std::vector<llama_token> kv_seq = gen.prompt_tokens;
            kv_seq.insert(kv_seq.end(), gen.gen_token_ids.begin(), gen.gen_token_ids.end());
            g_sopa.kv_update(std::move(kv_seq));
        } else {
            // MTP leaves the KV cache in a state that is complex to track; invalidate.
            g_sopa.kv_clear();
        }
        g_sopa.on_inference_end();
        gen.inference_active = false;
    }
}

static bool flush_pending_or_stop(SopaGenCtx & gen, std::string & output, const std::string & finish_reason) {
    size_t stop_pos = std::string::npos;
    for (const auto & stop : gen.stop_strings) {
        const size_t pos = gen.pending.find(stop);
        if (pos != std::string::npos) {
            stop_pos = std::min(stop_pos, pos);
        }
    }

    if (stop_pos != std::string::npos) {
        const std::string clean = gen.pending.substr(0, stop_pos);
        output.clear();
        append_parsed_chat_deltas(gen, clean, false, output);
        output += sse_finish_done(gen, "stop");
        gen.pending.clear();
        return false;
    }

    if (!finish_reason.empty()) {
        output.clear();
        append_parsed_chat_deltas(gen, gen.pending, false, output);
        output += sse_finish_done(gen, finish_reason);
        gen.pending.clear();
        return false;
    }

    const size_t keep = stop_prefix_suffix_len(gen.pending, gen.stop_strings);
    if (gen.pending.size() > keep) {
        const size_t emit_len = gen.pending.size() - keep;
        output.clear();
        append_parsed_chat_deltas(gen, gen.pending.substr(0, emit_len), true, output);
        gen.pending.erase(0, emit_len);
    } else {
        output.clear();
    }

    return true;
}

static bool decode_prompt_in_batches(SopaGenCtx & gen) {
    const int32_t n_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(gen.ctx));
    // Start after any prefix tokens already in the KV cache.
    int32_t pos = gen.n_prefix;

    if (gen.has_media) {
        size_t idx = 0;
        while (idx < gen.media_tokens.size()) {
            if (gen.media_tokens[idx] == LLAMA_TOKEN_NULL) {
                const auto & chunk = gen.media_tokens.find_chunk(idx);
                gen.media_batch.reset(mtmd_batch_init(g_sopa.get_mtmd_context()));
                if (!gen.media_batch || mtmd_batch_add_chunk(gen.media_batch.get(), chunk.get()) != 0 ||
                    mtmd_batch_encode(gen.media_batch.get()) != 0) {
                    return false;
                }
                float * embd = mtmd_batch_get_output_embd(gen.media_batch.get(), chunk.get());
                llama_pos new_n_past = pos;
                if (!embd || mtmd_helper_decode_image_chunk(
                        g_sopa.get_mtmd_context(), gen.ctx, chunk.get(), embd, pos, 0,
                        llama_n_batch(gen.ctx), &new_n_past, nullptr, nullptr) != 0) {
                    return false;
                }
                const size_t consumed = mtmd_input_chunk_get_n_tokens(chunk.get());
                idx += consumed;
                // A media chunk's model positions are not necessarily equal
                // to its placeholder token count.  Advance using mtmd's
                // position mapping so any following text is decoded at the
                // correct position.
                pos = new_n_past;
                continue;
            }
            size_t end = idx;
            while (end < gen.media_tokens.size() && gen.media_tokens[end] != LLAMA_TOKEN_NULL && end - idx < (size_t) n_batch) {
                ++end;
            }
            std::vector<llama_token> text_tokens;
            text_tokens.reserve(end - idx);
            for (size_t i = idx; i < end; ++i) text_tokens.push_back(gen.media_tokens[i]);
            llama_batch batch = llama_batch_get_one(text_tokens.data(), (int32_t) text_tokens.size());
            if (llama_decode(gen.ctx, batch) != 0) return false;
            pos += (int32_t) text_tokens.size();
            idx = end;
        }
        return true;
    }

    while (pos < (int32_t) gen.prompt_tokens.size()) {
        const int32_t n_tokens = std::min<int32_t>(n_batch, (int32_t) gen.prompt_tokens.size() - pos);
        // llama_batch_get_one with pos=nullptr uses the current KV-used count as the
        // starting position, so new tokens land at the correct positions after the prefix.
        llama_batch batch = llama_batch_get_one(gen.prompt_tokens.data() + pos, n_tokens);
        if (llama_decode(gen.ctx, batch) != 0) {
            LOG_ERR("sopa-generate: prefill decode failed at token %d/%zu (chunk=%d, n_batch=%d)\n",
                    pos, gen.prompt_tokens.size(), n_tokens, n_batch);
            return false;
        }
        pos += n_tokens;
    }

    return true;
}

static void append_parsed_chat_deltas(SopaGenCtx & gen, const std::string & text, bool is_partial, std::string & output) {
    if (text.empty()) {
        return;
    }

    gen.generated_text += text;
    try {
        common_chat_msg msg_prv = gen.chat_msg;
        common_chat_msg msg_new = common_chat_parse(gen.generated_text, is_partial, gen.chat_parser_params);
        if (msg_new.empty()) {
            return;
        }
        gen.chat_msg = msg_new;
        auto diffs = common_chat_msg_diff::compute_diffs(msg_prv, gen.chat_msg);
        for (const auto & diff : diffs) {
            output += sse_delta(server_chat_msg_diff_to_json_oaicompat(diff));
        }
    } catch (const std::exception & e) {
        if (!is_partial) {
            LOG_WRN("sopa-generate: final chat parse failed, falling back to raw content: %s\n", e.what());
            output += sse_chunk(text);
        }
    }
}

static bool append_token_delta(SopaGenCtx & gen, llama_token id, std::string & output) {
    if (llama_vocab_is_eog(gen.vocab, id)) {
        flush_pending_or_stop(gen, output, "stop");
        finish_inference(gen);
        return false;
    }

    char buf[512];
    int32_t n = llama_token_to_piece(gen.vocab, id, buf, sizeof(buf), 0, false);
    if (n < 0) {
        output = sse_done();
        finish_inference(gen);
        return false;
    }

    gen.n_generated++;
    gen.gen_token_ids.push_back(id);
    gen.pending.append(buf, n);

    const bool keep_going = flush_pending_or_stop(gen, output, "");
    if (!keep_going) {
        finish_inference(gen);
    }
    return keep_going;
}

static bool decode_mtp_prefill(SopaGenCtx & gen) {
    if (gen.prompt_tokens.empty()) {
        return false;
    }

    gen.prompt_tgt = gen.prompt_tokens;
    gen.n_past = (int32_t) gen.prompt_tgt.size();

    const int32_t n_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(gen.ctx));
    int32_t pos = 0;
    while (pos < (int32_t) gen.prompt_tgt.size()) {
        const int32_t n_tokens = std::min<int32_t>(n_batch, (int32_t) gen.prompt_tgt.size() - pos);
        common_batch_clear(gen.batch_tgt);
        for (int32_t i = 0; i < n_tokens; ++i) {
            common_batch_add(gen.batch_tgt, gen.prompt_tgt[pos + i], pos + i, { 0 }, true);
        }
        if (llama_decode(gen.ctx, gen.batch_tgt) != 0 ||
            !common_speculative_process(gen.spec, gen.batch_tgt)) {
            LOG_ERR("sopa-generate: MTP prefill decode failed at token %d/%zu\n",
                    pos, gen.prompt_tgt.size());
            return false;
        }
        pos += n_tokens;
    }

    common_speculative_begin(gen.spec, 0, gen.prompt_tgt);
    return true;
}

static bool mtp_next(SopaGenCtx & gen, std::string & output) {
    if (!gen.prefill_done) {
        if (!decode_mtp_prefill(gen)) {
            output = sse_done();
            finish_inference(gen);
            return false;
        }
        gen.prefill_done = true;

        if (g_sopa.interrupted()) {
            output = sse_finish_done(gen, "interrupted");
            finish_inference(gen);
            std::thread([]{ g_sopa.swap_to_small(); }).detach();
            return false;
        }
        if (gen.n_generated >= gen.max_tokens) {
            flush_pending_or_stop(gen, output, "length");
            finish_inference(gen);
            return false;
        }

        gen.id_last = common_sampler_sample(gen.common_smpl, gen.ctx, -1);
        common_sampler_accept(gen.common_smpl, gen.id_last, true);
        return append_token_delta(gen, gen.id_last, output);
    }

    if (g_sopa.interrupted()) {
        output = sse_finish_done(gen, "interrupted");
        finish_inference(gen);
        std::thread([]{ g_sopa.swap_to_small(); }).detach();
        return false;
    }

    if (gen.n_generated >= gen.max_tokens) {
        flush_pending_or_stop(gen, output, "length");
        finish_inference(gen);
        return false;
    }

    const int32_t remaining_after_target = std::max<int32_t>(0, gen.max_tokens - gen.n_generated - 1);
    llama_tokens draft;
    auto & dparams = common_speculative_get_draft_params(gen.spec, 0);
    dparams = {
        /* .drafting = */ remaining_after_target > 0,
        /* .n_max    = */ std::min<int32_t>(g_sopa.mtp_tokens(), remaining_after_target),
        /* .n_past   = */ gen.n_past,
        /* .id_last  = */ gen.id_last,
        /* .prompt   = */ &gen.prompt_tgt,
        /* .result   = */ &draft,
    };
    common_speculative_draft(gen.spec);
    gen.n_drafted += (int32_t) draft.size();

    common_batch_clear(gen.batch_tgt);
    common_batch_add(gen.batch_tgt, gen.id_last, gen.n_past, { 0 }, true);
    for (size_t i = 0; i < draft.size(); ++i) {
        common_batch_add(gen.batch_tgt, draft[i], gen.n_past + (llama_pos) i + 1, { 0 }, true);
    }

    if (llama_decode(gen.ctx, gen.batch_tgt) != 0 ||
        !common_speculative_process(gen.spec, gen.batch_tgt)) {
        LOG_ERR("sopa-generate: MTP target decode failed\n");
        output = sse_done();
        finish_inference(gen);
        return false;
    }

    llama_tokens ids = common_sampler_sample_and_accept_n(gen.common_smpl, gen.ctx, draft);
    if (ids.empty()) {
        output = sse_done();
        finish_inference(gen);
        return false;
    }

    if (!draft.empty()) {
        common_speculative_accept(gen.spec, 0, (uint16_t) (ids.size() - 1));
        gen.n_accepted += (int32_t) ids.size() - 1;
    }

    gen.prompt_tgt.push_back(gen.id_last);
    if (ids.size() > 1) {
        gen.prompt_tgt.insert(gen.prompt_tgt.end(), ids.begin(), ids.end() - 1);
    }
    gen.id_last = ids.back();
    gen.n_past = (int32_t) gen.prompt_tgt.size();

    common_context_seq_rm(gen.ctx, 0, gen.n_past, -1);
    if (llama_context * draft_ctx = g_sopa.get_draft_context()) {
        common_context_seq_rm(draft_ctx, 0, gen.n_past, -1);
    }

    output.clear();
    for (llama_token id : ids) {
        std::string token_output;
        if (!append_token_delta(gen, id, token_output)) {
            output += token_output;
            return false;
        }
        output += token_output;
        if (gen.n_generated >= gen.max_tokens) {
            std::string finish_output;
            flush_pending_or_stop(gen, finish_output, "length");
            output += finish_output;
            finish_inference(gen);
            return false;
        }
    }

    return true;
}

server_http_res_ptr handle_sopa_completions(const server_http_req & req) {
    // Guard: only active in Phase 2 (SopaManager owns model).
    if (g_sopa.server_owns_model()) {
        auto res    = std::make_unique<server_http_res>();
        res->status = 409;
        res->data   = json{{"error",
            "server owns the model via --model; remove it to enable Phase 2 generation"}}.dump();
        return res;
    }

    // Parse request body.
    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        auto res    = std::make_unique<server_http_res>();
        res->status = 400;
        res->data   = json{{"error", "invalid JSON body"}}.dump();
        return res;
    }

    if (!body.contains("messages") || !body["messages"].is_array()) {
        auto res    = std::make_unique<server_http_res>();
        res->status = 400;
        res->data   = json{{"error", "field 'messages' must be an array"}}.dump();
        return res;
    }

    // Wait for inference slot — queues concurrent requests instead of returning 503 immediately.
    if (!g_sopa.acquire_inference_slot()) {
        auto res    = std::make_unique<server_http_res>();
        res->status = 503;
        res->data   = json{{"error", "model not ready or busy (timeout) — call POST /sopa/load first"}}.dump();
        return res;
    }

    struct InferenceLease {
        bool active = true;
        ~InferenceLease() {
            if (active) g_sopa.on_inference_end();
        }
        void release() {
            if (active) {
                g_sopa.on_inference_end();
                active = false;
            }
        }
        void transfer() { active = false; }
    };
    InferenceLease inference_lease;

    llama_context * ctx   = g_sopa.get_context();
    llama_model *   model = g_sopa.get_model();
    if (!ctx || !model) {
        inference_lease.release();
        auto res    = std::make_unique<server_http_res>();
        res->status = 503;
        res->data   = json{{"error", "no model context available"}}.dump();
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_chat_params chat_params;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    std::vector<raw_buffer> media_files_for_prompt;
    try {
        auto tmpls = common_chat_templates_init(model, /* chat_template_override */ "");

        bool request_has_media = false;
        for (const auto & message : body["messages"]) {
            if (!message.contains("content") || !message["content"].is_array()) continue;
            for (const auto & part : message["content"]) {
                const std::string type = part.value("type", "");
                request_has_media = request_has_media || type == "image_url" || type == "input_audio";
            }
        }
        std::vector<raw_buffer> media_files;
        std::string media_prompt;
        if (request_has_media) {
            if (!g_sopa.get_mtmd_context()) {
                throw std::runtime_error("media input requires the small model and its multimodal projector");
            }
            json media_body = body;
            server_chat_params opt{};
            opt.use_jinja = true;
            opt.prefill_assistant = false;
            opt.reasoning_format = reasoning_format;
            opt.tmpls = common_chat_templates_init(model, "");
            opt.allow_image = true;
            opt.allow_audio = true;
            opt.enable_thinking = body.value("enable_thinking", g_sopa.enable_thinking());
            json parsed = oaicompat_chat_params_parse(media_body, opt, media_files);
            media_prompt = parsed.value("prompt", "");
            body["messages"] = media_body["messages"];
        }

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(body["messages"]);
        inputs.add_generation_prompt = body.value("add_generation_prompt", true);
        inputs.use_jinja             = true;
        inputs.enable_thinking       = body.value("enable_thinking", g_sopa.enable_thinking());
        if (body.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
        }
        inputs.reasoning_format      = reasoning_format;

        if (body.contains("tools") && body["tools"].is_array()) {
            inputs.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        }
        if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
        }
        inputs.parallel_tool_calls = body.value("parallel_tool_calls", false);
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            for (const auto & item : body["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        chat_params = common_chat_templates_apply(tmpls.get(), inputs);
        if (!media_prompt.empty()) chat_params.prompt = media_prompt;

        if (!media_files.empty()) {
            // Keep the encoded media chunks alive in the generation context below.
            body["__sopa_media_count"] = media_files.size();
            // Store temporarily as base64-free process-local data after the catch block.
            media_files_for_prompt = std::move(media_files);
        }
    } catch (const std::exception & e) {
        LOG_ERR("sopa-generate: failed to apply embedded chat template: %s\n", e.what());
        inference_lease.release();
        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        res->data   = json{{"error", std::string("failed to apply embedded chat template: ") + e.what()}}.dump();
        return res;
    }

    LOG_INF("sopa-generate: using embedded chat template format '%s'\n",
            common_chat_format_name(chat_params.format));

    std::string prompt = chat_params.prompt;

    server_tokens media_tokens;
    const bool has_media = body.value("__sopa_media_count", 0) > 0;
    if (has_media) {
        try {
            media_tokens = process_mtmd_prompt(g_sopa.get_mtmd_context(), prompt, std::move(media_files_for_prompt));
        } catch (const std::exception & e) {
            LOG_ERR("sopa-generate: failed to tokenize media prompt: %s\n", e.what());
            inference_lease.release();
            auto res    = std::make_unique<server_http_res>();
            res->status = 500;
            res->data   = json{{"error", std::string("failed to tokenize media prompt: ") + e.what()}}.dump();
            return res;
        }
    }

    // Tokenize first so we can compute the prefix length before touching the KV cache.
    int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                       nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    n_prompt = -n_prompt; // negative means the required buffer size
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                        prompt_tokens.data(), n_prompt,
                        true, true) < 0) {
        inference_lease.release();
        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        res->data   = json{{"error", "tokenization failed"}}.dump();
        return res;
    }

    // KV cache prefix reuse: find how many leading tokens of prompt_tokens are already
    // in the KV cache from the previous request. We must always decode at least the last
    // prompt token to produce fresh logits, so cap at n_prompt - 1.
    const int32_t raw_prefix = g_sopa.match_kv_prefix(prompt_tokens);
    const int32_t prefix_len = std::min(raw_prefix, n_prompt - 1);
    if (prefix_len > 0) {
        LOG_INF("sopa-generate: KV prefix reuse %d/%d tokens (%.0f%%)\n",
                prefix_len, n_prompt, 100.0f * prefix_len / n_prompt);
        // Remove any KV entries beyond the matched prefix (e.g. generated tokens from
        // the previous call that are not part of the new prompt).
        llama_memory_seq_rm(llama_get_memory(ctx), 0, prefix_len, -1);
    } else {
        llama_memory_clear(llama_get_memory(ctx), false);
    }

    common_params_sampling sampling_params = make_sampling_params(body);
    sampling_params.generation_prompt = chat_params.generation_prompt;
    sampling_params.reasoning_budget_tokens = json_i32_or(body, "thinking_budget_tokens", -1);
    if (!chat_params.thinking_start_tag.empty()) {
        sampling_params.reasoning_budget_start = common_tokenize(
            vocab, chat_params.thinking_start_tag, false, true);
    }
    if (!chat_params.thinking_end_tag.empty()) {
        sampling_params.reasoning_budget_end = common_tokenize(
            vocab, chat_params.thinking_end_tag, false, true);
        sampling_params.reasoning_budget_forced = common_tokenize(
            vocab, chat_params.thinking_end_tag, false, true);
    }
    common_sampler * smpl = common_sampler_init(model, sampling_params);

    int32_t max_tokens = body.value("max_tokens", 512);
    const bool mtp_requested = request_mtp_enabled(body);
    const int  mtp_tokens = mtp_requested ? g_sopa.mtp_tokens() : 0;
    llama_model * draft_model = mtp_tokens > 0 ? g_sopa.get_draft_model() : nullptr;
    llama_context * draft_ctx = mtp_tokens > 0 ? g_sopa.get_draft_context() : nullptr;

    // Transfer ownership of sampler and prompt tokens into the streaming state.
    auto gen = std::make_shared<SopaGenCtx>();
    gen->ctx           = ctx;
    gen->vocab         = vocab;
    gen->common_smpl   = smpl;
    gen->prompt_tokens = std::move(prompt_tokens);
    if (has_media) {
        gen->media_tokens = std::move(media_tokens);
        // Multimodal server_tokens contain LLAMA_TOKEN_NULL sentinels backed
        // by encoded media chunks. get_tokens() deliberately asserts for that
        // representation; keep only real text tokens in the flat bookkeeping
        // vector while media_tokens retains the complete prefill sequence.
        gen->prompt_tokens = gen->media_tokens.get_text_tokens();
        gen->has_media = true;
        gen->n_prefix = 0;
        llama_memory_clear(llama_get_memory(ctx), false);
    }
    gen->n_prefix      = has_media ? 0 : prefix_len;
    gen->max_tokens    = max_tokens;
    gen->stop_strings  = STOP_STRINGS;
    gen->chat_parser_params = common_chat_parser_params(chat_params);
    gen->chat_parser_params.reasoning_format = reasoning_format;
    gen->chat_parser_params.parse_tool_calls = true;
    if (!chat_params.parser.empty()) {
        gen->chat_parser_params.parser.load(chat_params.parser);
    }
    for (const auto & stop : chat_params.additional_stops) {
        add_unique_stop(gen->stop_strings, stop);
    }
    if (body.contains("stop")) {
        if (body["stop"].is_string()) {
            add_unique_stop(gen->stop_strings, body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto & stop : body["stop"]) {
                if (stop.is_string()) {
                    add_unique_stop(gen->stop_strings, stop.get<std::string>());
                }
            }
        }
    }

    if (draft_model && draft_ctx && gen->prompt_tokens.size() > 1) {
        const common_context_seq_rm_type seq_rm = common_context_can_seq_rm(ctx);
        if (seq_rm != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            llama_memory_clear(llama_get_memory(ctx), false);
            llama_memory_clear(llama_get_memory(draft_ctx), false);

            gen->use_mtp = true;
            gen->spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            gen->spec_params.draft.mparams.path = g_sopa.get_draft_model_path();
            gen->spec_params.draft.n_max = mtp_tokens;
            gen->spec_params.draft.n_min = 0;
            gen->spec_params.draft.p_min = 0.0f;
            gen->spec_params.draft.ctx_tgt = ctx;
            gen->spec_params.draft.ctx_dft = draft_ctx;

            gen->spec = common_speculative_init(gen->spec_params, 1);
            if (gen->spec) {
                gen->batch_tgt = llama_batch_init(llama_n_batch(ctx), 0, 1);
                gen->batch_tgt_init = true;
                LOG_INF("sopa-generate: upstream Gemma 4 MTP enabled with %d draft tokens\n",
                        mtp_tokens);
            } else {
                gen->use_mtp = false;
                LOG_WRN("sopa-generate: MTP disabled — failed to initialize speculative context\n");
            }
        } else {
            LOG_WRN("sopa-generate: MTP disabled — target context does not support partial KV removal\n");
        }
    } else if (mtp_requested && mtp_tokens > 0) {
        LOG_WRN("sopa-generate: MTP requested but no draft model is loaded\n");
    }

    auto res               = std::make_unique<server_http_res>();
    res->status            = 200;
    res->content_type      = "text/event-stream";

    res->next = [gen](std::string & output) mutable -> bool {
        if (!gen->sent_meta) {
            gen->sent_meta = true;
            output = sse_meta(*gen);
            return true;
        }

        if (gen->use_mtp) {
            return mtp_next(*gen, output);
        }

        llama_batch batch;

        if (!gen->prefill_done) {
            // First call: decode the prompt in logical-batch chunks. A prompt can
            // fit in n_ctx while still being larger than llama_n_batch(ctx).
            if (!decode_prompt_in_batches(*gen)) {
                output = sse_done();
                finish_inference(*gen);
                return false;
            }
            gen->prefill_done = true;
        } else {
            // Subsequent calls: decode the previously sampled token.
            batch = llama_batch_get_one(&gen->last_token, 1);
            if (llama_decode(gen->ctx, batch) != 0) {
                LOG_ERR("sopa-generate: decode failed\n");
                output = sse_done();
                finish_inference(*gen);
                return false;
            }
        }

        // Check interrupt between tokens — stop at the next token boundary.
        if (g_sopa.interrupted()) {
            output = sse_finish_done(*gen, "interrupted");
            finish_inference(*gen);
            std::thread([]{ g_sopa.swap_to_small(); }).detach();
            return false;
        }

        if (gen->n_generated >= gen->max_tokens) {
            flush_pending_or_stop(*gen, output, "length");
            finish_inference(*gen);
            return false;
        }

        // Sample next token.
        llama_token id = common_sampler_sample(gen->common_smpl, gen->ctx, -1);
        common_sampler_accept(gen->common_smpl, id, true);

        // End of generation: EOG token.
        if (llama_vocab_is_eog(gen->vocab, id)) {
            flush_pending_or_stop(*gen, output, "stop");
            finish_inference(*gen);
            return false;
        }

        // Decode token to text.
        char buf[512];
        int32_t n = llama_token_to_piece(gen->vocab, id, buf, sizeof(buf), 0, false);
        if (n < 0) {
            output = sse_done();
            finish_inference(*gen);
            return false;
        }
        std::string piece(buf, n);

        gen->last_token  = id;
        gen->n_generated++;
        gen->gen_token_ids.push_back(id);
        gen->pending += piece;

        const bool keep_going = flush_pending_or_stop(*gen, output, "");
        if (!keep_going) {
            finish_inference(*gen);
        }
        return keep_going;
    };

    // SopaGenCtx now owns the inference slot until streaming completes or the
    // context is destroyed. Before this point the local lease covers every
    // exception and early return.
    gen->inference_active = true;
    inference_lease.transfer();
    return res;
}
