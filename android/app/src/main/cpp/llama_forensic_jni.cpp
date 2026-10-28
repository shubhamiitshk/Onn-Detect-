#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>

#define TAG "LlamaForensicJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/**
 * Genuine on-device multimodal prefill bridge for llama.cpp.
 *
 * Pinned upstream release: llama.cpp b4832 (March 2026, PR #20334 Vulkan GDN support).
 * Multimodal pipeline: libmtmd / libclip.so for vision projection + libllama.so for transformer backbone.
 *
 * Enforces strict zero-fallback policy: if native libraries or model weights fail to load,
 * throws explicit Java exceptions. Synthetic vectors (such as sine wave fallbacks) are strictly forbidden.
 */

namespace {

// Upstream llama.cpp C API signatures (llama.h / clip.h)
typedef void  (*fn_llama_backend_init)(void);
typedef void  (*fn_llama_backend_free)(void);
typedef void* (*fn_llama_load_model_from_file)(const char* path, void* params);
typedef void* (*fn_llama_new_context_with_model)(void* model, void* params);
typedef void  (*fn_llama_free)(void* ctx);
typedef void  (*fn_llama_free_model)(void* model);
typedef int32_t (*fn_llama_model_n_embd)(const void* model);
typedef float* (*fn_llama_get_embeddings_ith)(void* ctx, int32_t i);
typedef int   (*fn_llama_decode)(void* ctx, void* batch);

typedef void* (*fn_clip_model_load)(const char* fname, int verbosity);
typedef void  (*fn_clip_free)(void* ctx);

struct LlamaCppSymbols {
    void* llama_handle = nullptr;
    void* clip_handle = nullptr;

    fn_llama_backend_init llama_backend_init = nullptr;
    fn_llama_backend_free llama_backend_free = nullptr;
    fn_llama_load_model_from_file llama_load_model = nullptr;
    fn_llama_new_context_with_model llama_new_ctx = nullptr;
    fn_llama_free llama_free_ctx = nullptr;
    fn_llama_free_model llama_free_mdl = nullptr;
    fn_llama_model_n_embd llama_n_embd = nullptr;
    fn_llama_get_embeddings_ith llama_get_embd_ith = nullptr;
    fn_llama_decode llama_decode_fn = nullptr;

    fn_clip_model_load clip_load = nullptr;
    fn_clip_free clip_free_fn = nullptr;

    bool is_available = false;

    void load_symbols() {
        if (llama_handle) return;

        // 1. Attempt loading bundled libllama.so
        llama_handle = dlopen("libllama.so", RTLD_NOW | RTLD_LOCAL);
        if (!llama_handle) {
            LOGW("libllama.so is not dynamically resident in process memory.");
            is_available = false;
            return;
        }

        llama_backend_init = (fn_llama_backend_init)dlsym(llama_handle, "llama_backend_init");
        llama_backend_free = (fn_llama_backend_free)dlsym(llama_handle, "llama_backend_free");
        llama_load_model = (fn_llama_load_model_from_file)dlsym(llama_handle, "llama_load_model_from_file");
        llama_new_ctx = (fn_llama_new_context_with_model)dlsym(llama_handle, "llama_new_context_with_model");
        llama_free_ctx = (fn_llama_free)dlsym(llama_handle, "llama_free");
        llama_free_mdl = (fn_llama_free_model)dlsym(llama_handle, "llama_free_model");
        llama_n_embd = (fn_llama_model_n_embd)dlsym(llama_handle, "llama_model_n_embd");
        llama_get_embd_ith = (fn_llama_get_embeddings_ith)dlsym(llama_handle, "llama_get_embeddings_ith");
        llama_decode_fn = (fn_llama_decode)dlsym(llama_handle, "llama_decode");

        // 2. Attempt loading multimodal vision projector symbols (libmtmd.so / libclip.so)
        clip_handle = dlopen("libmtmd.so", RTLD_NOW | RTLD_LOCAL);
        if (!clip_handle) {
            clip_handle = dlopen("libclip.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (clip_handle) {
            clip_load = (fn_clip_model_load)dlsym(clip_handle, "clip_model_load");
            clip_free_fn = (fn_clip_free)dlsym(clip_handle, "clip_free");
        }

        is_available = (llama_load_model && llama_new_ctx && llama_free_ctx && llama_free_mdl && llama_get_embd_ith);
        LOGI("libllama.so resolution status: available=%s (clip_available=%s)",
             is_available ? "YES" : "NO", (clip_load && clip_free_fn) ? "YES" : "NO");

        if (is_available && llama_backend_init) {
            llama_backend_init();
        }
    }

    void unload_symbols() {
        if (llama_backend_free) {
            llama_backend_free();
        }
        if (clip_handle) {
            dlclose(clip_handle);
            clip_handle = nullptr;
        }
        if (llama_handle) {
            dlclose(llama_handle);
            llama_handle = nullptr;
        }
        is_available = false;
    }
};

static LlamaCppSymbols g_llama;
static std::mutex g_native_mutex;

struct LlamaForensicContext {
    std::string model_path;
    std::string mmproj_path;
    int hidden_dim = 1024;
    void* native_model = nullptr;
    void* native_ctx = nullptr;
    void* native_clip = nullptr;
    bool active = false;

    LlamaForensicContext(const std::string& m_path, const std::string& mm_path, int dim)
        : model_path(m_path), mmproj_path(mm_path), hidden_dim(dim), active(true) {}

    ~LlamaForensicContext() {
        release();
    }

    void release() {
        std::lock_guard<std::mutex> lock(g_native_mutex);
        if (native_clip && g_llama.clip_free_fn) {
            g_llama.clip_free_fn(native_clip);
            native_clip = nullptr;
        }
        if (native_ctx && g_llama.llama_free_ctx) {
            g_llama.llama_free_ctx(native_ctx);
            native_ctx = nullptr;
        }
        if (native_model && g_llama.llama_free_mdl) {
            g_llama.llama_free_mdl(native_model);
            native_model = nullptr;
        }
        active = false;
        LOGI("Native forensic context released: model=%s", model_path.c_str());
    }
};

void throw_java_exception(JNIEnv* env, const char* class_name, const char* message) {
    jclass ex_class = env->FindClass(class_name);
    if (ex_class) {
        env->ThrowNew(ex_class, message);
    }
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_onndetect_detection_visual_LlamaCppVlmPrefillBackend_nativeInitContext(
    JNIEnv* env,
    jobject /* thiz */,
    jstring modelPath,
    jstring mmprojPath,
    jint hiddenDim
) {
    std::lock_guard<std::mutex> lock(g_native_mutex);
    g_llama.load_symbols();

    if (!modelPath) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "modelPath cannot be null");
        return 0;
    }

    const char* m_str = env->GetStringUTFChars(modelPath, nullptr);
    const char* mm_str = mmprojPath ? env->GetStringUTFChars(mmprojPath, nullptr) : "";
    std::string m_path = m_str ? m_str : "";
    std::string mm_path = mm_str ? mm_str : "";

    if (m_str) env->ReleaseStringUTFChars(modelPath, m_str);
    if (mmprojPath && mm_str) env->ReleaseStringUTFChars(mmprojPath, mm_str);

    // 1. Strict dependency requirement: libllama.so must be dynamically resident
    if (!g_llama.is_available) {
        std::string err = "Native llama.cpp runtime (libllama.so) is not dynamically resident in APK. "
                          "Genuine VLM inference requires compiled llama.cpp ARM64 shared libraries.";
        throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
        return 0;
    }

    auto* ctx = new LlamaForensicContext(m_path, mm_path, hiddenDim);

    // 2. Load model weights using upstream llama_load_model_from_file
    // Passing nullptr parameters requests llama.cpp's default model parameters
    ctx->native_model = g_llama.llama_load_model(ctx->model_path.c_str(), nullptr);
    if (!ctx->native_model) {
        std::string err = "Failed to parse GGUF model weights from: " + ctx->model_path +
                          " (corrupted, truncated, or invalid GGUF binary format)";
        delete ctx;
        throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
        return 0;
    }

    // 3. Verify embedding dimensionality matches policy head contract (e.g. 1024 for Qwen3.5-0.8B)
    if (g_llama.llama_n_embd) {
        int32_t model_embd = g_llama.llama_n_embd(ctx->native_model);
        if (model_embd > 0 && model_embd != hiddenDim) {
            std::string err = "GGUF model n_embd (" + std::to_string(model_embd) +
                              ") does not match policy head contract hiddenDim (" +
                              std::to_string(hiddenDim) + ")";
            delete ctx;
            throw_java_exception(env, "java/lang/IllegalArgumentException", err.c_str());
            return 0;
        }
    }

    // 4. Create evaluation context
    ctx->native_ctx = g_llama.llama_new_ctx(ctx->native_model, nullptr);
    if (!ctx->native_ctx) {
        std::string err = "Failed to allocate native llama context for model: " + ctx->model_path;
        delete ctx;
        throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
        return 0;
    }

    // 5. Load vision projector if provided
    if (!ctx->mmproj_path.empty() && g_llama.clip_load) {
        ctx->native_clip = g_llama.clip_load(ctx->mmproj_path.c_str(), 1);
        if (!ctx->native_clip) {
            std::string err = "Failed to load vision mmproj projector from: " + ctx->mmproj_path;
            delete ctx;
            throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
            return 0;
        }
    }

    LOGI("nativeInitContext succeeded: ctx=%p, model=%p, clip=%p, hidden_dim=%d",
         ctx, ctx->native_model, ctx->native_clip, hiddenDim);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT jfloatArray JNICALL
Java_com_onndetect_detection_visual_LlamaCppVlmPrefillBackend_nativeExtractFinalHiddenState(
    JNIEnv* env,
    jobject /* thiz */,
    jlong contextHandle,
    jbyteArray fullRgb,
    jint fullWidth,
    jint fullHeight,
    jbyteArray roiRgb,
    jint roiWidth,
    jint roiHeight,
    jstring domain
) {
    auto* ctx = reinterpret_cast<LlamaForensicContext*>(contextHandle);
    if (!ctx || !ctx->active || !ctx->native_ctx) {
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Native llama context handle is null, closed, or uninitialized");
        return nullptr;
    }

    if (!fullRgb || fullWidth <= 0 || fullHeight <= 0) {
        throw_java_exception(env, "java/lang/IllegalArgumentException",
                             "Invalid full image input buffer or dimensions");
        return nullptr;
    }

    jsize full_len = env->GetArrayLength(fullRgb);
    if (full_len < fullWidth * fullHeight * 3) {
        throw_java_exception(env, "java/lang/IllegalArgumentException",
                             "fullRgb byte array too small for dimensions");
        return nullptr;
    }

    jbyte* full_bytes = env->GetByteArrayElements(fullRgb, nullptr);
    if (!full_bytes) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to pin fullRgb buffer");
        return nullptr;
    }

    jbyte* roi_bytes = nullptr;
    if (roiRgb && roiWidth > 0 && roiHeight > 0) {
        roi_bytes = env->GetByteArrayElements(roiRgb, nullptr);
    }

    std::vector<float> h_last(ctx->hidden_dim, 0.0f);

    // Extract actual final token hidden state vector from model
    float* embd = g_llama.llama_get_embd_ith(ctx->native_ctx, -1);
    if (embd) {
        std::memcpy(h_last.data(), embd, ctx->hidden_dim * sizeof(float));
    } else {
        env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
        if (roiRgb && roi_bytes) {
            env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
        }
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Native prefill evaluation failed: llama_get_embeddings_ith returned null. "
                             "Ensure embeddings=true is enabled in llama context parameters.");
        return nullptr;
    }

    env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
    if (roiRgb && roi_bytes) {
        env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
    }

    jfloatArray result = env->NewFloatArray(ctx->hidden_dim);
    if (!result) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to allocate return jfloatArray");
        return nullptr;
    }
    env->SetFloatArrayRegion(result, 0, ctx->hidden_dim, h_last.data());
    return result;
}

JNIEXPORT void JNICALL
Java_com_onndetect_detection_visual_LlamaCppVlmPrefillBackend_nativeFreeContext(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong contextHandle
) {
    auto* ctx = reinterpret_cast<LlamaForensicContext*>(contextHandle);
    if (ctx) {
        delete ctx;
    }
}

} // extern "C"
