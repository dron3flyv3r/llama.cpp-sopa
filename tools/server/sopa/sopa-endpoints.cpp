#include "sopa-endpoints.h"
#include "sopa-generate.h"
#include "sopa-manager.h"

#include <nlohmann/json.hpp>

#include <optional>

using json = nlohmann::ordered_json;

static server_http_res_ptr json_ok(const json & body) {
    auto res    = std::make_unique<server_http_res>();
    res->status = 200;
    res->data   = body.dump();
    return res;
}

static server_http_res_ptr json_err(int status, const std::string & msg) {
    auto res    = std::make_unique<server_http_res>();
    res->status = status;
    res->data   = json{{"error", msg}}.dump();
    return res;
}

SopaRoutes make_sopa_routes() {
    SopaRoutes routes;

    routes.get_status = [](const server_http_req &) {
        return json_ok(g_sopa.status());
    };

    routes.post_load = [](const server_http_req & req) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            return json_err(400, "invalid JSON body");
        }

        std::string model_str = body.value("model", "");
        SopaModelType type;
        if (model_str == "small") {
            type = SopaModelType::SMALL;
        } else if (model_str == "large") {
            type = SopaModelType::LARGE;
        } else {
            return json_err(400, "field 'model' must be 'small' or 'large'");
        }

        std::optional<bool> enable_thinking_override;
        if (body.contains("enable_thinking")) {
            if (!body["enable_thinking"].is_boolean()) {
                return json_err(400, "field 'enable_thinking' must be a boolean");
            }
            enable_thinking_override = body["enable_thinking"].get<bool>();
        }

        if (!g_sopa.load(type, enable_thinking_override)) {
            // Distinguish the "server owns model" guard from a genuine load failure.
            // g_sopa.load() logs the specific reason; expose a useful HTTP status here.
            if (g_sopa.server_owns_model()) {
                return json_err(409,
                    "server owns the model via --model; POST /sopa/load is disabled in "
                    "Phase 1 (OpenAI-proxy mode). Remove --model to enable SopaManager "
                    "as the sole inference driver (Phase 2).");
            }
            if (g_sopa.get_state() == SopaState::INFERRING) {
                return json_err(409, "model is busy; cannot load while inference is active");
            }
            return json_err(500, "failed to load model — check server logs");
        }
        return json_ok(g_sopa.status());
    };

    routes.post_unload = [](const server_http_req &) {
        if (!g_sopa.unload()) {
            return json_err(409, "model is busy; cannot unload while inference is active");
        }
        return json_ok({{"status", "unloaded"}});
    };

    routes.post_interrupt = [](const server_http_req &) {
        g_sopa.interrupt();
        return json_ok({{"status", "interrupted"}});
    };

    // Phase 2: override /v1/chat/completions with SopaManager-owned generation.
    // Registered in server.cpp after the standard handler so it takes priority.
    routes.post_completions = handle_sopa_completions;

    return routes;
}
