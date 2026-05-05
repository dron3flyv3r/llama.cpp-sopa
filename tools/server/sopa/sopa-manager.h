#pragma once

#include "llama.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>

using json = nlohmann::ordered_json;

enum class SopaModelType { NONE, SMALL, LARGE };
enum class SopaState { UNLOADED, LOADING, READY, INFERRING };

struct SopaConfig {
    std::string small_model_path;
    std::string large_model_path;

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

    // Thinking
    bool small_enable_thinking = false;
    bool large_enable_thinking = true;

    // Context + batching per model
    uint32_t small_n_ctx   = 32768;
    uint32_t large_n_ctx   = 131072;
    uint32_t small_n_batch = 2048;
    uint32_t large_n_batch = 4096;

    // Idle unload timeouts
    int small_idle_timeout_s = 300;  // 5 min
    int large_idle_timeout_s = 120;  // 2 min
};

class SopaManager {
  public:
    SopaManager() = default;

    ~SopaManager() { stop(); }

    void init(const SopaConfig & cfg);

    // Load model of given type; unloads current model first if needed.
    // Blocks until the model is ready. Returns false on failure.
    bool load(SopaModelType type, std::optional<bool> enable_thinking_override = std::nullopt);

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

    bool interrupted() const { return interrupt_.load(); }

    bool enable_thinking() const;

  private:
    void do_unload_locked();  // caller must hold mtx_
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
    std::atomic<bool>                     interrupt_{ false };
    std::atomic<bool>                     running_{ false };
    std::thread                           idle_thread_;
    std::chrono::steady_clock::time_point last_active_;
};

// Process-wide singleton — shared between sopa-endpoints and server.cpp
extern SopaManager g_sopa;
