#pragma once

#include "server-http.h"

// Handles POST /v1/chat/completions when SopaManager owns the model (Phase 2).
// Streams tokens as SSE in OpenAI chat-completion-chunk format and checks
// g_sopa.interrupted() between every token so a wake-word can preempt generation.
server_http_res_ptr handle_sopa_completions(const server_http_req & req);
