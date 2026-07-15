#include "models.h"

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    // Base class already read n_embd_head_k_full = key_length and
    // n_embd_head_k_swa = key_length_swa.  Keep those — they match the actual
    // Q weight shapes in the GGUF: SWA layers use key_length_swa per head,
    // full-attention layer uses key_length.
    //
    // The GGUF has shared_kv_layers = n_layer (no K/V tensors exist).  For
    // standalone Phase 1 (K = V = Q) we need n_head_kv == n_head per layer so
    // that K = Q has the same shape as Q.
    const uint32_t n_head = hparams.n_head();
    for (uint32_t i = 0; i < hparams.n_layer; i++) {
        hparams.n_head_kv_arr[i] = n_head;
    }

    // Keep SWA enabled — the head dims differ between SWA and full-attention
    // layers and rope_freqs only exist for full-attention layers.
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    // Ignore shared_kv_layers = 4; give every layer its own KV cache.
    hparams.n_layer_kv_from_start = -1;

    hparams.f_attention_scale = 1.0f;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MTP head params
    ml.get_key(LLM_KV_MTP_BACKBONE_HIDDEN_SIZE,        hparams.mtp_backbone_hidden_size);
    ml.get_key(LLM_KV_MTP_NUM_CENTROIDS,               hparams.mtp_num_centroids,           false);
    ml.get_key(LLM_KV_MTP_USE_ORDERED_EMBEDDINGS,      hparams.mtp_use_ordered_embeddings,   false);

    uint32_t ignored = 0;
    ml.get_key(LLM_KV_MTP_CENTROID_INTERMEDIATE_TOP_K, ignored, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader & /*ml*/) {
    LLAMA_LOAD_LOCALS;

    const int64_t backbone_h  = hparams.mtp_backbone_hidden_size;
    const int64_t n_centroids = hparams.mtp_num_centroids;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          0);

    // pre_proj is only used in Phase 2 (backbone injection).
    mtp_pre_proj  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,  "weight"),
                                  {2 * backbone_h, n_embd}, TENSOR_NOT_REQUIRED);
    // post_proj: assistant hidden → backbone embedding space
    mtp_post_proj = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "weight"),
                                  {n_embd, backbone_h},     0);

    if (n_centroids > 0) {
        // centroids: cluster centers in assistant embedding space
        mtp_centroids    = create_tensor(tn(LLM_TENSOR_MTP_CENTROIDS, "weight"),
                                         {n_embd, n_centroids}, TENSOR_NOT_REQUIRED);
        // tok_ordering: for each vocab token, its centroid index
        mtp_tok_ordering = create_tensor(tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"),
                                         {n_vocab},             TENSOR_NOT_REQUIRED);
    }

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i    = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);  // 256 for SWA layers, 512 for full

        // rope_freqs only present for full-attention layers (same as gemma4.cpp).
        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i),
                                             {n_embd_head / 2}, rope_freqs_flag);
            rope_freqs_flag  = TENSOR_DUPLICATED;
        }

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd},                        0);
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_head_i * n_embd_head}, 0);
        // K/V projections absent in the GGUF (shared_kv_layers = n_layer).
        layer.wk             = create_tensor(tn(LLM_TENSOR_ATTN_K,         "weight", i), {n_embd, n_embd_head * n_head_i}, TENSOR_NOT_REQUIRED);
        layer.wv             = create_tensor(tn(LLM_TENSOR_ATTN_V,         "weight", i), {n_embd, n_embd_head * n_head_i}, TENSOR_NOT_REQUIRED);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head},                    0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head},                    TENSOR_NOT_REQUIRED);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_head_i * n_embd_head, n_embd},  0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd},                          0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd},         0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,       "weight", i), {n_embd, n_ff},   0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,         "weight", i), {n_embd, n_ff},   0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,       "weight", i), {n_ff,   n_embd},  0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd},          0);
        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE,"weight", i), {1u},               TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
    cb(inpL, "inp_scaled", -1);

    ggml_tensor * inp_pos     = build_inp_pos();
    auto        * inp_attn    = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);  // 256 (SWA) or 512 (full)
        const int64_t n_head      = hparams.n_head(il);
        const int     n_rot_l     = hparams.n_rot(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        // rope_freqs: non-null only for full-attention layers (proportional RoPE).
        ggml_tensor * freq_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors,
                             n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_pos", il);

        // K = V = Q: no separate K/V projections in GGUF.
        ggml_tensor * Kcur = Qcur;
        ggml_tensor * Vcur = Qcur;

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, nullptr,
                         Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                         hparams.f_attention_scale, il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.mtp_centroids && model.mtp_tok_ordering &&
        model.mtp_tok_ordering->type == GGML_TYPE_I32) {
        // lm head via centroid lookup:
        //   centroids {n_embd, n_centroids} × h {n_embd, n_tokens} → scores {n_centroids, n_tokens}
        ggml_tensor * centroid_scores = ggml_mul_mat(ctx0, model.mtp_centroids, cur);
        cb(centroid_scores, "centroid_scores", -1);

        // ggml_get_rows(A, B): selects rows from A.dim1 using indices in B.
        //   A {d0, n_rows}, B {n_idx} → result {d0, n_idx}
        // We need: for each vocab token v, logit[v, t] = centroid_scores[tok_ordering[v], t]
        // So transpose scores to {n_tokens, n_centroids}, then get_rows with tok_ordering {n_vocab}.
        ggml_tensor * scores_T = ggml_cont(ctx0, ggml_transpose(ctx0, centroid_scores));
        // logits_T: {n_tokens, n_vocab}
        ggml_tensor * logits_T = ggml_get_rows(ctx0, scores_T, model.mtp_tok_ordering);
        cb(logits_T, "logits_T", -1);

        // Standard convention: {n_vocab, n_tokens}
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, logits_T));
    } else {
        // Fallback: tok_ordering not i32 (GGUF stored as float; re-run converter to fix).
        // Use tok_embd as tied lm head — suboptimal but correct.
        cur = ggml_mul_mat(ctx0, model.tok_embd, cur);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
