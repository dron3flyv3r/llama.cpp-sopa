#include "sopa-manager.h"
#include "log.h"
#include "server-common.h"

#include "mtmd-helper.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
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

static int mtp_tokens_for_type(const SopaConfig & cfg, SopaModelType type) {
    return type == SopaModelType::SMALL ? cfg.small_mtp_tokens : cfg.large_mtp_tokens;
}

static int draft_gpu_layers_for_type(const SopaConfig & cfg, SopaModelType type) {
    return type == SopaModelType::SMALL ? cfg.small_draft_n_gpu_layers : cfg.large_draft_n_gpu_layers;
}

static const std::string & draft_path_for_type(const SopaConfig & cfg, SopaModelType type) {
    return type == SopaModelType::SMALL ? cfg.small_draft_model_path : cfg.large_draft_model_path;
}

void SopaManager::init(const SopaConfig & cfg) {
    cfg_         = cfg;
    last_active_ = std::chrono::steady_clock::now();
    running_     = true;
    idle_thread_ = std::thread(&SopaManager::idle_loop, this);
    LOG_INF("sopa-manager: initialized — small='%s' large='%s'\n",
            cfg_.small_model_path.c_str(), cfg_.large_model_path.c_str());
}

bool SopaManager::load(SopaModelType type, std::optional<bool> enable_thinking_override, bool multimodal) {
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
        const bool requested_mm = type == SopaModelType::SMALL && multimodal;
        if (active_type_ == type && state_ == SopaState::READY && (mtmd_ctx_ != nullptr) == requested_mm) {
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
    cparams.n_ctx            = (type == SopaModelType::SMALL) ? cfg_.small_n_ctx   : cfg_.large_n_ctx;
    cparams.n_batch          = (type == SopaModelType::SMALL) ? cfg_.small_n_batch : cfg_.large_n_batch;
    cparams.n_ubatch         = cfg_.n_ubatch;
    cparams.n_threads        = cfg_.n_threads;
    cparams.n_threads_batch  = cfg_.n_threads_batch;
    cparams.flash_attn_type  = cfg_.flash_attn_type;
    cparams.type_k           = cfg_.cache_type_k;
    cparams.type_v           = cfg_.cache_type_v;
    cparams.offload_kqv      = cfg_.offload_kqv;
    cparams.op_offload       = cfg_.op_offload;
    cparams.swa_full         = cfg_.swa_full;
    if (mtp_tokens_for_type(cfg_, type) > 0 && !draft_path_for_type(cfg_, type).empty()) {
        cparams.n_rs_seq = mtp_tokens_for_type(cfg_, type);
    }

    llama_context * c = llama_init_from_model(m, cparams);
    if (!c) {
        LOG_ERR("sopa-manager: failed to create context\n");
        llama_model_free(m);
        std::lock_guard<std::mutex> lock(mtx_);
        state_ = SopaState::UNLOADED;
        return false;
    }

    mtmd_context * mm = nullptr;
    if (type == SopaModelType::SMALL && multimodal && !cfg_.small_mmproj_path.empty()) {
        mtmd_context_params mmparams = mtmd_context_params_default();
        mmparams.use_gpu = cfg_.small_mmproj_use_gpu;
        mmparams.print_timings = false;
        mmparams.n_threads = 8;
        // oaicompat_chat_params_parse() replaces image/audio parts with the
        // process-wide marker returned by get_media_marker().  The mtmd
        // context must split on that exact marker (it is randomized unless
        // LLAMA_MEDIA_MARKER is set), otherwise it sees media bytes but zero
        // markers and rejects the prompt.
        mmparams.media_marker = get_media_marker();
        mm = mtmd_init_from_file(cfg_.small_mmproj_path.c_str(), m, mmparams);
        if (!mm && cfg_.small_mmproj_use_gpu) {
            LOG_WRN("sopa-manager: projector GPU load failed, retrying CPU\n");
            mmparams.use_gpu = false;
            mm = mtmd_init_from_file(cfg_.small_mmproj_path.c_str(), m, mmparams);
        }
        if (!mm) {
            llama_free(c);
            llama_model_free(m);
            std::lock_guard<std::mutex> lock(mtx_);
            state_ = SopaState::UNLOADED;
            LOG_ERR("sopa-manager: failed to load small multimodal projector\n");
            return false;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        model_       = m;
        ctx_         = c;
        mtmd_ctx_    = mm;
        active_type_ = type;
        active_enable_thinking_ = enable_thinking;
        load_draft_locked(type);
        state_       = SopaState::READY;
        last_active_ = std::chrono::steady_clock::now();
    }

    LOG_INF("sopa-manager: %s model ready (mtp=%s, draft=%s)\n",
            model_type_str(type), mtp_enabled_ ? "enabled" : "disabled",
            draft_placement_.c_str());
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
    unload_draft_locked();
    if (mtmd_ctx_) {
        mtmd_free(mtmd_ctx_);
        mtmd_ctx_ = nullptr;
    }
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
    active_mtp_tokens_ = 0;
    mtp_enabled_ = false;
    interrupt_.store(false);
    kv_seq_.clear();
    LOG_INF("sopa-manager: model unloaded\n");
    cv_ready_.notify_all();
}

int32_t SopaManager::match_kv_prefix(const std::vector<llama_token> & tokens) const {
    const int32_t n = (int32_t) std::min(kv_seq_.size(), tokens.size());
    int32_t i = 0;
    while (i < n && kv_seq_[i] == tokens[i]) {
        ++i;
    }
    return i;
}

void SopaManager::kv_update(std::vector<llama_token> tokens) {
    kv_seq_ = std::move(tokens);
}

void SopaManager::kv_clear() {
    kv_seq_.clear();
}

void SopaManager::unload_draft_locked() {
    if (draft_ctx_) {
        llama_free(draft_ctx_);
        draft_ctx_ = nullptr;
    }
    if (draft_model_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
    }
    draft_loaded_     = false;
    mtp_enabled_      = false;
    draft_model_path_.clear();
    draft_placement_  = "none";
    draft_load_error_.clear();
}

void SopaManager::load_draft_locked(SopaModelType type) {
    unload_draft_locked();

    const std::string & path = draft_path_for_type(cfg_, type);
    const int mtp_tokens = mtp_tokens_for_type(cfg_, type);
    active_mtp_tokens_ = mtp_tokens;
    draft_model_path_  = path;

    if (path.empty() || mtp_tokens <= 0) {
        return;
    }

    auto try_load = [&](int n_gpu_layers, const char * placement) -> llama_model * {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        mparams.use_mmap     = cfg_.use_mmap;

        LOG_INF("sopa-manager: loading %s draft from '%s' (placement=%s, n_gpu_layers=%d)\n",
                model_type_str(type), path.c_str(), placement, n_gpu_layers);
        llama_model * draft = nullptr;
        try {
            draft = llama_model_load_from_file(path.c_str(), mparams);
        } catch (const std::exception & e) {
            draft_load_error_ = e.what();
            LOG_WRN("sopa-manager: draft load failed (%s): %s\n", placement, e.what());
        }
        if (draft) {
            draft_placement_ = placement;
            draft_load_error_.clear();
        }
        return draft;
    };

    const int requested_gpu_layers = draft_gpu_layers_for_type(cfg_, type);
    if (requested_gpu_layers != 0) {
        draft_model_ = try_load(requested_gpu_layers, "gpu");
        if (!draft_model_) {
            LOG_WRN("sopa-manager: draft GPU load failed, retrying CPU-only\n");
            draft_model_ = try_load(0, "cpu");
        }
    } else {
        draft_model_ = try_load(0, "cpu");
    }

    if (!draft_model_) {
        if (draft_load_error_.empty()) {
            draft_load_error_ = "failed to load draft model";
        }
        draft_placement_  = "failed";
        LOG_WRN("sopa-manager: MTP disabled — failed to load draft model '%s'\n", path.c_str());
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = llama_n_ctx(ctx_);
    cparams.n_batch          = llama_n_batch(ctx_);
    cparams.n_ubatch         = cfg_.n_ubatch;
    cparams.n_threads        = cfg_.n_threads;
    cparams.n_threads_batch  = cfg_.n_threads_batch;
    cparams.flash_attn_type  = cfg_.flash_attn_type;
    cparams.type_k           = cfg_.cache_type_k;
    cparams.type_v           = cfg_.cache_type_v;
    cparams.offload_kqv      = cfg_.offload_kqv;
    cparams.op_offload       = cfg_.op_offload;
    cparams.swa_full         = cfg_.swa_full;
    cparams.ctx_type         = LLAMA_CONTEXT_TYPE_MTP;
    cparams.ctx_other        = ctx_;
    cparams.n_rs_seq         = 0;

    draft_ctx_ = llama_init_from_model(draft_model_, cparams);
    if (!draft_ctx_) {
        draft_load_error_ = "failed to create MTP draft context";
        LOG_WRN("sopa-manager: MTP disabled — %s\n", draft_load_error_.c_str());
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
        draft_placement_ = "failed";
        return;
    }

    draft_loaded_ = true;
    mtp_enabled_  = true;
    LOG_INF("sopa-manager: upstream MTP draft context ready (tokens=%d)\n", mtp_tokens);
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
        {"mtp_enabled",     mtp_enabled_},
        {"draft_loaded",    draft_loaded_},
        {"mtp_tokens",      active_mtp_tokens_},
        {"draft_model",     draft_model_path_},
        {"draft_placement", draft_placement_},
        {"draft_load_error", draft_load_error_},
        {"multimodal_loaded", mtmd_ctx_ != nullptr},
        {"modalities", mtmd_ctx_ ? json::array({"text", "image", "audio", "video"}) : json::array({"text"})},
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

std::string SopaManager::get_draft_model_path() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return draft_model_path_;
}

int SopaManager::mtp_tokens() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return mtp_enabled_ ? active_mtp_tokens_ : 0;
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
        if (timeout_s <= 0) {
            continue;
        }

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
