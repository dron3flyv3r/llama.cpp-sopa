#include "sopa-generate.h"
#include "sopa-manager.h"
#include "chat.h"
#include "log.h"
#include "server-chat.h"

#include "llama.h"

#include <algorithm>
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
    llama_context *     ctx;
    const llama_vocab * vocab;
    llama_sampler *     smpl;

    std::vector<llama_token> prompt_tokens;
    llama_token last_token = -1;
    int32_t     n_generated = 0;
    int32_t     max_tokens  = 512;
    bool        prefill_done = false;
    std::string pending;  // text withheld until it is known not to be a stop marker
    std::vector<std::string> stop_strings;
    std::string generated_text;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    bool inference_active = true;

    ~SopaGenCtx() {
        if (inference_active) {
            g_sopa.on_inference_end();
            inference_active = false;
        }
        if (smpl) {
            llama_sampler_free(smpl);
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

static std::string sse_finish(const std::string & reason) {
    json chunk = {
        {"id",      "sopa"},
        {"object",  "chat.completion.chunk"},
        {"choices", json::array({{
            {"index",         0},
            {"delta",         json::object()},
            {"finish_reason", reason}
        }})}
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

static std::string sse_finish_done(const std::string & reason) {
    return sse_finish(reason) + sse_done();
}

static void append_parsed_chat_deltas(SopaGenCtx & gen, const std::string & text, bool is_partial, std::string & output);

static void finish_inference(SopaGenCtx & gen) {
    if (gen.inference_active) {
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
        output += sse_finish_done("stop");
        gen.pending.clear();
        return false;
    }

    if (!finish_reason.empty()) {
        output.clear();
        append_parsed_chat_deltas(gen, gen.pending, false, output);
        output += sse_finish_done(finish_reason);
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

    auto release_slot = [] {
        g_sopa.on_inference_end();
    };

    llama_context * ctx   = g_sopa.get_context();
    llama_model *   model = g_sopa.get_model();
    if (!ctx || !model) {
        release_slot();
        auto res    = std::make_unique<server_http_res>();
        res->status = 503;
        res->data   = json{{"error", "no model context available"}}.dump();
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_chat_params chat_params;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    try {
        auto tmpls = common_chat_templates_init(model, /* chat_template_override */ "");

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
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
            for (const auto & item : body["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        chat_params = common_chat_templates_apply(tmpls.get(), inputs);
    } catch (const std::exception & e) {
        LOG_ERR("sopa-generate: failed to apply embedded chat template: %s\n", e.what());
        release_slot();
        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        res->data   = json{{"error", std::string("failed to apply embedded chat template: ") + e.what()}}.dump();
        return res;
    }

    LOG_INF("sopa-generate: using embedded chat template format '%s'\n",
            common_chat_format_name(chat_params.format));

    std::string prompt = chat_params.prompt;

    // Tokenize — clear KV cache so each request starts fresh.
    llama_memory_clear(llama_get_memory(ctx), false);

    int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                       nullptr, 0, /*add_special=*/true, /*parse_special=*/true);
    n_prompt = -n_prompt; // negative means the required buffer size
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                        prompt_tokens.data(), n_prompt,
                        true, true) < 0) {
        release_slot();
        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        res->data   = json{{"error", "tokenization failed"}}.dump();
        return res;
    }

    // Build sampler chain.
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    int32_t max_tokens = body.value("max_tokens", 512);

    // Transfer ownership of sampler and prompt tokens into the streaming state.
    auto gen = std::make_shared<SopaGenCtx>();
    gen->ctx           = ctx;
    gen->vocab         = vocab;
    gen->smpl          = smpl;
    gen->prompt_tokens = std::move(prompt_tokens);
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

    auto res               = std::make_unique<server_http_res>();
    res->status            = 200;
    res->content_type      = "text/event-stream";

    res->next = [gen](std::string & output) mutable -> bool {
        llama_batch batch;

        if (!gen->prefill_done) {
            // First call: decode the full prompt.
            batch = llama_batch_get_one(gen->prompt_tokens.data(),
                                        (int32_t)gen->prompt_tokens.size());
            if (llama_decode(gen->ctx, batch) != 0) {
                LOG_ERR("sopa-generate: prefill decode failed\n");
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
            output = sse_finish_done("interrupted");
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
        llama_token id = llama_sampler_sample(gen->smpl, gen->ctx, -1);

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
        gen->pending += piece;

        const bool keep_going = flush_pending_or_stop(*gen, output, "");
        if (!keep_going) {
            finish_inference(*gen);
        }
        return keep_going;
    };

    return res;
}
