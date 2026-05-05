#include "sopa-manager.h"
#include "log.h"

#include <chrono>
#include <cstdlib>
#include <thread>

using namespace std::chrono_literals;

SopaManager g_sopa;

static const char * model_type_str(SopaModelType t) {
    switch (t) {
        case SopaModelType::SMALL: return "small";
        case SopaModelType::LARGE: return "large";
        default:                   return "none";
    }
}

static const char * state_str(SopaState s) {
    switch (s) {
        case SopaState::UNLOADED:  return "unloaded";
        case SopaState::LOADING:   return "loading";
        case SopaState::READY:     return "ready";
        case SopaState::INFERRING: return "inferring";
        default:                   return "unknown";
    }
}

static bool default_enable_thinking(const SopaConfig & cfg, SopaModelType type) {
    return type == SopaModelType::SMALL ? cfg.small_enable_thinking : cfg.large_enable_thinking;
}

void SopaManager::init(const SopaConfig & cfg) {
    cfg_         = cfg;
    last_active_ = std::chrono::steady_clock::now();
    running_     = true;
    idle_thread_ = std::thread(&SopaManager::idle_loop, this);
    LOG_INF("sopa-manager: initialized — small='%s' large='%s'\n",
            cfg_.small_model_path.c_str(), cfg_.large_model_path.c_str());
}

bool SopaManager::load(SopaModelType type, std::optional<bool> enable_thinking_override) {
    if (cfg_.server_owns_model) {
        LOG_ERR("sopa-manager: load() rejected — server owns the model via --model. "
                "Remove --model and set server_owns_model=false to use SopaManager "
                "as the sole inference driver (Phase 2).\n");
        return false;
    }

    const bool enable_thinking = enable_thinking_override.value_or(default_enable_thinking(cfg_, type));

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (state_ == SopaState::LOADING) {
            LOG_WRN("sopa-manager: already loading a model\n");
            return false;
        }
        if (state_ == SopaState::INFERRING) {
            LOG_WRN("sopa-manager: load rejected while inference is active\n");
            return false;
        }
        // Fast path: requested model already loaded and ready — nothing to do.
        if (active_type_ == type && state_ == SopaState::READY) {
            active_enable_thinking_ = enable_thinking;
            LOG_INF("sopa-manager: %s model already loaded (enable_thinking=%s)\n",
                    model_type_str(type), enable_thinking ? "true" : "false");
            return true;
        }
        if (state_ != SopaState::UNLOADED) {
            do_unload_locked();
        }
        state_ = SopaState::LOADING;
    }

    const std::string & path = (type == SopaModelType::SMALL)
                               ? cfg_.small_model_path
                               : cfg_.large_model_path;

    if (path.empty()) {
        LOG_ERR("sopa-manager: no path configured for %s model\n", model_type_str(type));
        std::lock_guard<std::mutex> lock(mtx_);
        state_ = SopaState::UNLOADED;
        return false;
    }

    LOG_INF("sopa-manager: loading %s model from '%s' (mmap=%s, swa_full=%s, enable_thinking=%s)\n",
            model_type_str(type), path.c_str(),
            cfg_.use_mmap ? "true" : "false",
            cfg_.swa_full ? "true" : "false",
            enable_thinking ? "true" : "false");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = (type == SopaModelType::SMALL)
                                 ? cfg_.small_n_gpu_layers
                                 : cfg_.large_n_gpu_layers;
    mparams.use_mmap           = cfg_.use_mmap;

    llama_model * m = llama_model_load_from_file(path.c_str(), mparams);
    if (!m) {
        LOG_ERR("sopa-manager: failed to load model from '%s'\n", path.c_str());
        std::lock_guard<std::mutex> lock(mtx_);
        state_ = SopaState::UNLOADED;
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = (type == SopaModelType::SMALL) ? cfg_.small_n_ctx   : cfg_.large_n_ctx;
    cparams.n_batch = (type == SopaModelType::SMALL) ? cfg_.small_n_batch : cfg_.large_n_batch;
    cparams.swa_full = cfg_.swa_full;

    llama_context * c = llama_init_from_model(m, cparams);
    if (!c) {
        LOG_ERR("sopa-manager: failed to create context\n");
        llama_model_free(m);
        std::lock_guard<std::mutex> lock(mtx_);
        state_ = SopaState::UNLOADED;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        model_       = m;
        ctx_         = c;
        active_type_ = type;
        active_enable_thinking_ = enable_thinking;
        state_       = SopaState::READY;
        last_active_ = std::chrono::steady_clock::now();
    }

    LOG_INF("sopa-manager: %s model ready\n", model_type_str(type));
    return true;
}

bool SopaManager::unload() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (state_ == SopaState::INFERRING || state_ == SopaState::LOADING) {
        LOG_WRN("sopa-manager: unload rejected while state=%s\n", state_str(state_));
        return false;
    }
    do_unload_locked();
    return true;
}

void SopaManager::do_unload_locked() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    state_       = SopaState::UNLOADED;
    active_type_ = SopaModelType::NONE;
    active_enable_thinking_ = false;
    interrupt_.store(false);
    LOG_INF("sopa-manager: model unloaded\n");
    cv_ready_.notify_all();
}

void SopaManager::interrupt() {
    interrupt_.store(true);
    LOG_INF("sopa-manager: interrupt requested\n");
}

void SopaManager::swap_to_small() {
    LOG_INF("sopa-manager: hot-swap to small model\n");
    {
        std::lock_guard<std::mutex> lock(mtx_);
        do_unload_locked();
    }
    load(SopaModelType::SMALL);
}

void SopaManager::on_inference_start() {
    std::lock_guard<std::mutex> lock(mtx_);
    state_ = SopaState::INFERRING;
}

void SopaManager::on_inference_end() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        interrupt_.store(false);
        state_       = SopaState::READY;
        last_active_ = std::chrono::steady_clock::now();
    }
    cv_ready_.notify_all();
}

bool SopaManager::acquire_inference_slot(int timeout_s) {
    std::unique_lock<std::mutex> lock(mtx_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    cv_ready_.wait_until(lock, deadline, [this] {
        return state_ == SopaState::READY || state_ == SopaState::UNLOADED;
    });
    if (state_ != SopaState::READY) {
        return false;
    }
    state_ = SopaState::INFERRING;
    return true;
}

json SopaManager::status() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const uint32_t configured_n_ctx = active_type_ == SopaModelType::SMALL
                                      ? cfg_.small_n_ctx
                                      : active_type_ == SopaModelType::LARGE
                                          ? cfg_.large_n_ctx
                                          : 0;
    const uint32_t active_n_ctx = ctx_ ? llama_n_ctx(ctx_) : 0;
    return json{
        {"state",           state_str(state_)},
        {"model",           model_type_str(active_type_)},
        {"loaded",          state_ != SopaState::UNLOADED},
        {"enable_thinking", active_enable_thinking_},
        {"n_ctx",           active_n_ctx ? active_n_ctx : configured_n_ctx},
        {"active_n_ctx",    active_n_ctx},
        {"small_n_ctx",     cfg_.small_n_ctx},
        {"large_n_ctx",     cfg_.large_n_ctx},
    };
}

SopaState SopaManager::get_state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
}

SopaModelType SopaManager::get_active_model() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_type_;
}

bool SopaManager::enable_thinking() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_enable_thinking_;
}

void SopaManager::idle_loop() {
    while (running_) {
        std::this_thread::sleep_for(10s);

        std::lock_guard<std::mutex> lock(mtx_);
        if (state_ != SopaState::READY) {
            continue;
        }

        int timeout_s = (active_type_ == SopaModelType::SMALL)
                        ? cfg_.small_idle_timeout_s
                        : cfg_.large_idle_timeout_s;

        auto elapsed = std::chrono::steady_clock::now() - last_active_;
        if (elapsed >= std::chrono::seconds(timeout_s)) {
            LOG_INF("sopa-manager: idle timeout — unloading %s model\n",
                    model_type_str(active_type_));
            do_unload_locked();
        }
    }
}

void SopaManager::stop() {
    running_ = false;
    if (idle_thread_.joinable()) {
        idle_thread_.join();
    }
    std::lock_guard<std::mutex> lock(mtx_);
    do_unload_locked();
}
