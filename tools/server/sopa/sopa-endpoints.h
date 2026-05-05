#pragma once

#include "server-http.h"

struct SopaRoutes {
    server_http_context::handler_t get_status;
    server_http_context::handler_t post_load;
    server_http_context::handler_t post_unload;
    server_http_context::handler_t post_interrupt;
    // Phase 2: overrides /v1/chat/completions when SopaManager owns the model
    server_http_context::handler_t post_completions;
};

SopaRoutes make_sopa_routes();
