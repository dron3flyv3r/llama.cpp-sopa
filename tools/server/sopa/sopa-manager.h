#pragma once

#include "llama.h"
#include "mtmd.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

enum class SopaModelType { NONE, SMALL, LARGE };
enum class SopaState { UNLOADED, LOADING, READY, INFERRING };

struct SopaConfig {
    std::string small_model_path;
    std::string large_model_path;
    std::string small_draft_model_path;
    std::string large_draft_model_path;
    std::string small_mmproj_path;
    bool small_mmproj_use_gpu = true;

    // When true the main llama-server process owns a model loaded via --model.
    // SopaManager::load() will refuse to run to avoid a duplicate VRAM allocation.
    // Set to false only when running in Phase 2 (no --model flag, SopaManager
    // drives all inference via /sopa/load + custom generation endpoints).
    bool server_owns_model = true;
    bool use_mmap          = true;
    bool swa_full          = false;

    // GPU offload: small fits fully on RTX 4070 Ti, large is partial
    int small_n_gpu_layers = -1;  // -1 = all layers on GPU
    int large_n_gpu_layers = 8;   // partial offload for 26B on 12 GB VRAM
    int small_draft_n_gpu_layers = -1;
    int large_draft_n_gpu_layers = -1;

    // Speculative/MTP draft depth per model. Set to 0 to disable for a size.
    int small_mtp_tokens = 2;
    int large_mtp_tokens = 2;

    // Thinking
    bool small_enable_thinking = false;
    bool large_enable_thinking = true;

    // Context + batching per model
    uint32_t small_n_ctx   = 32768;
    uint32_t large_n_ctx   = 131072;
    uint32_t small_n_batch = 2048;
    uint32_t large_n_batch = 4096;
    uint32_t n_ubatch      = 512;
    int32_t n_threads      = -1;
    int32_t n_threads_batch = -1;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ggml_type cache_type_k = GGML_TYPE_F16;
    ggml_type cache_type_v = GGML_TYPE_F16;
    bool offload_kqv       = true;
    bool op_offload        = true;

    // Idle unload timeouts
    int small_idle_timeout_s = 300;  // 5 min; <=0 disables idle unload
    int large_idle_timeout_s = 120;  // 2 min; <=0 disables idle unload
};

class SopaManager {
  public:
    SopaManager() = default;

    ~SopaManager() { stop(); }

    void init(const SopaConfig & cfg);

    // Load model of given type; unloads current model first if needed.
    // Blocks until the model is ready. Returns false on failure.
    bool load(SopaModelType type, std::optional<bool> enable_thinking_override = std::nullopt, bool multimodal = false);

    // Unload current model and free VRAM. Returns false while inference/loading is active.
    bool unload();

    // Phase 2: sets interrupt flag checked between tokens in generation loop.
    void interrupt();

    // Phase 2: unloads current model and loads small, used after interrupt.
    // Runs synchronously; call from a detached thread to avoid blocking callers.
    void swap_to_small();

    // Called by inference wrappers to drive idle timer and state transitions.
    void on_inference_start();
    void on_inference_end();

    // Waits until the model is READY and atomically claims the INFERRING slot.
    // Returns false if the model is unloaded or timeout_s elapses.
    bool acquire_inference_slot(int timeout_s = 90);

    json          status() const;
    SopaState     get_state() const;
    SopaModelType get_active_model() const;

    bool server_owns_model() const { return cfg_.server_owns_model; }

    // Raw access for inference code (phase 2)
    llama_model * get_model() const { return model_; }

    llama_context * get_context() const { return ctx_; }

    llama_model * get_draft_model() const { return draft_model_; }
    llama_context * get_draft_context() const { return draft_ctx_; }
    mtmd_context * get_mtmd_context() const { return mtmd_ctx_; }

    std::string get_draft_model_path() const;

    int mtp_tokens() const;

    bool interrupted() const { return interrupt_.load(); }

    bool enable_thinking() const;

    // KV cache prefix reuse — called from inference code (single-threaded per slot).
    // Returns how many leading tokens of `tokens` are already in the KV cache.
    int32_t match_kv_prefix(const std::vector<llama_token> & tokens) const;
    // Replaces the stored KV sequence with the full sequence from the last request.
    void kv_update(std::vector<llama_token> tokens);
    // Clears the stored KV sequence (called on model unload or MTP inference).
    void kv_clear();

  private:
    void do_unload_locked();  // caller must hold mtx_
    void load_draft_locked(SopaModelType type);
    void unload_draft_locked();
    void idle_loop();
    void stop();

    SopaConfig                            cfg_;
    mutable std::mutex                    mtx_;
    std::condition_variable               cv_ready_;
    SopaState                             state_       = SopaState::UNLOADED;
    SopaModelType                         active_type_ = SopaModelType::NONE;
    bool                                  active_enable_thinking_ = false;
    llama_model *                         model_       = nullptr;
    llama_context *                       ctx_         = nullptr;
    mtmd_context *                        mtmd_ctx_    = nullptr;
    llama_model *                         draft_model_ = nullptr;
    llama_context *                       draft_ctx_   = nullptr;
    bool                                  draft_loaded_ = false;
    bool                                  mtp_enabled_ = false;
    int                                   active_mtp_tokens_ = 0;
    std::string                           draft_model_path_;
    std::string                           draft_placement_ = "none";
    std::string                           draft_load_error_;
    std::atomic<bool>                     interrupt_{ false };
    std::atomic<bool>                     running_{ false };
    std::thread                           idle_thread_;
    std::chrono::steady_clock::time_point last_active_;

    // Full token sequence from the last completed generation (prompt + generated).
    // Used for KV cache prefix reuse on the next request. Not mutex-protected because
    // inference is serialized through acquire_inference_slot / on_inference_end.
    std::vector<llama_token>              kv_seq_;
};

// Process-wide singleton — shared between sopa-endpoints and server.cpp
extern SopaManager g_sopa;
