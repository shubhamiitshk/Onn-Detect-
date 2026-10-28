#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>
#include <algorithm>

#include "llama.h"
#include "clip.h"
#include "llava.h"

#define TAG "LlamaForensicJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/**
 * Genuine on-device multimodal prefill bridge for llama.cpp.
 *
 * Pinned upstream release: llama.cpp b4832 (commit 07d15723470a0a5b15d8ccad1aff5b20354ffbe1, March 6, 2025).
 * Multimodal pipeline: clip / llava for vision projection + llama.cpp for transformer backbone.
 *
 * Enforces strict zero-fallback policy: if native libraries or model weights fail to load,
 * throws explicit Java exceptions. Synthetic vectors (such as sine wave fallbacks) are strictly forbidden.
 */

namespace {

static std::recursive_mutex g_native_mutex;

struct LlamaForensicContext {
    std::string model_path;
    std::string mmproj_path;
    int hidden_dim = 1024;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    clip_ctx* clip = nullptr;
    bool active = false;

    LlamaForensicContext(const std::string& m_path, const std::string& mm_path, int dim)
        : model_path(m_path), mmproj_path(mm_path), hidden_dim(dim), active(true) {}

    ~LlamaForensicContext() {
        release_internal();
    }

    // Direct cleanup without reacquiring mutex (safe to call when lock is already held)
    void release_internal() {
        if (clip) {
            clip_free(clip);
            clip = nullptr;
        }
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        active = false;
        LOGI("Native forensic context released: model=%s", model_path.c_str());
    }

    void release() {
        std::lock_guard<std::recursive_mutex> lock(g_native_mutex);
        release_internal();
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
    std::lock_guard<std::recursive_mutex> lock(g_native_mutex);

    static bool backend_initialized = false;
    if (!backend_initialized) {
        llama_backend_init();
        backend_initialized = true;
    }

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

    auto* ctx = new LlamaForensicContext(m_path, mm_path, hiddenDim);

    // 1. Load model weights using upstream llama_model_load_from_file with default parameters
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU inference on Android by default
    ctx->model = llama_model_load_from_file(ctx->model_path.c_str(), mparams);
    if (!ctx->model) {
        std::string err = "Failed to parse GGUF model weights from: " + ctx->model_path +
                          " (corrupted, truncated, or invalid GGUF binary format)";
        delete ctx;
        throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
        return 0;
    }

    // 2. Verify embedding dimensionality matches policy head contract (e.g. 1024 for Qwen3.5-0.8B)
    int32_t model_embd = llama_model_n_embd(ctx->model);
    if (model_embd > 0 && model_embd != hiddenDim) {
        std::string err = "GGUF model n_embd (" + std::to_string(model_embd) +
                          ") does not match policy head contract hiddenDim (" +
                          std::to_string(hiddenDim) + ")";
        delete ctx;
        throw_java_exception(env, "java/lang/IllegalArgumentException", err.c_str());
        return 0;
    }

    // 3. Create evaluation context with embeddings enabled
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    cparams.embeddings = true;
    ctx->ctx = llama_init_from_model(ctx->model, cparams);
    if (!ctx->ctx) {
        std::string err = "Failed to allocate native llama context for model: " + ctx->model_path;
        delete ctx;
        throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
        return 0;
    }

    // 4. Load vision projector if provided
    if (!ctx->mmproj_path.empty()) {
        ctx->clip = clip_model_load(ctx->mmproj_path.c_str(), 1);
        if (!ctx->clip) {
            std::string err = "Failed to load vision mmproj projector from: " + ctx->mmproj_path;
            delete ctx;
            throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
            return 0;
        }
    }

    LOGI("nativeInitContext succeeded: ctx=%p, model=%p, clip=%p, hidden_dim=%d",
         ctx->ctx, ctx->model, ctx->clip, hiddenDim);
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
    if (!ctx || !ctx->active || !ctx->ctx || !ctx->model) {
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

    const char* d_str = domain ? env->GetStringUTFChars(domain, nullptr) : "general";
    std::string domain_str = d_str ? d_str : "general";
    if (domain && d_str) env->ReleaseStringUTFChars(domain, d_str);

    std::lock_guard<std::recursive_mutex> lock(g_native_mutex);

    // 1. Clear KV cache to prevent token state pollution across runs
    llama_kv_cache_clear(ctx->ctx);

    // 2. Prepare dialectical prompt
    std::string prompt = "<|im_start|>user\n<image>\nAnalyze image for forensic manipulation artifacts in domain: " +
                         domain_str + ".<|im_end|>\n<|im_start|>assistant\n<think>";

    const struct llama_vocab * vocab = llama_model_get_vocab(ctx->model);

    // 3. Multimodal prefill evaluation
    if (ctx->clip) {
        clip_image_u8* img = clip_image_u8_init();
        clip_build_img_from_pixels(reinterpret_cast<const unsigned char*>(full_bytes), fullWidth, fullHeight, img);

        float* img_embd = nullptr;
        int n_img_pos = 0;
        bool embed_ok = llava_image_embed_make_with_clip_img(ctx->clip, 4, img, &img_embd, &n_img_pos);
        clip_image_u8_free(img);

        if (!embed_ok || !img_embd || n_img_pos <= 0) {
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "Failed to encode visual tokens via clip projector");
            return nullptr;
        }

        std::string image_tag = "<image>";
        size_t img_tag_pos = prompt.find(image_tag);
        std::string prefix = (img_tag_pos != std::string::npos) ? prompt.substr(0, img_tag_pos) : "";
        std::string postfix = (img_tag_pos != std::string::npos) ? prompt.substr(img_tag_pos + image_tag.length()) : prompt;

        int n_past = 0;

        // Evaluate prefix tokens before <image>
        if (!prefix.empty()) {
            std::vector<llama_token> prefix_tokens(prefix.length() + 16);
            int n_prefix = llama_tokenize(vocab, prefix.c_str(), prefix.length(),
                                          prefix_tokens.data(), prefix_tokens.size(), true, true);
            if (n_prefix > 0) {
                prefix_tokens.resize(n_prefix);
                llama_batch batch = llama_batch_init(n_prefix, 0, 1);
                for (int i = 0; i < n_prefix; ++i) {
                    batch.token[i] = prefix_tokens[i];
                    batch.pos[i] = n_past + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = false;
                }
                batch.n_tokens = n_prefix;
                if (llama_decode(ctx->ctx, batch) != 0) {
                    llama_batch_free(batch);
                    free(img_embd);
                    env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
                    if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
                    throw_java_exception(env, "java/lang/IllegalStateException",
                                         "llama_decode failed on prefix tokens");
                    return nullptr;
                }
                llama_batch_free(batch);
                n_past += n_prefix;
            }
        }

        // Evaluate image embeddings
        struct llava_image_embed image_embed = { img_embd, n_img_pos };
        if (!llava_eval_image_embed(ctx->ctx, &image_embed, 512, &n_past)) {
            free(img_embd);
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "llava_eval_image_embed failed to evaluate image tokens");
            return nullptr;
        }
        free(img_embd);

        // Evaluate postfix tokens with logits/embeddings enabled on the last token
        std::vector<llama_token> postfix_tokens(postfix.length() + 16);
        int n_postfix = llama_tokenize(vocab, postfix.c_str(), postfix.length(),
                                       postfix_tokens.data(), postfix_tokens.size(), false, true);
        if (n_postfix > 0) {
            postfix_tokens.resize(n_postfix);
            llama_batch batch = llama_batch_init(n_postfix, 0, 1);
            for (int i = 0; i < n_postfix; ++i) {
                batch.token[i] = postfix_tokens[i];
                batch.pos[i] = n_past + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == n_postfix - 1);
            }
            batch.n_tokens = n_postfix;
            if (llama_decode(ctx->ctx, batch) != 0) {
                llama_batch_free(batch);
                env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
                if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
                throw_java_exception(env, "java/lang/IllegalStateException",
                                     "llama_decode failed on postfix tokens");
                return nullptr;
            }
            llama_batch_free(batch);
            n_past += n_postfix;
        }
    } else {
        // Language-only prompt evaluation
        std::vector<llama_token> tokens(prompt.length() + 16);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                      tokens.data(), tokens.size(), true, true);
        if (n_tokens <= 0) {
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "Failed to tokenize dialectical prompt");
            return nullptr;
        }
        tokens.resize(n_tokens);
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        if (llama_decode(ctx->ctx, batch) != 0) {
            llama_batch_free(batch);
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "llama_decode failed on prompt tokens");
            return nullptr;
        }
        llama_batch_free(batch);
    }

    // 4. Extract final token hidden state vector
    float* embd = llama_get_embeddings_ith(ctx->ctx, -1);
    if (!embd) {
        embd = llama_get_embeddings(ctx->ctx);
    }

    if (!embd) {
        env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
        if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Native prefill evaluation failed: llama_get_embeddings returned null");
        return nullptr;
    }

    std::vector<float> h_last(ctx->hidden_dim, 0.0f);
    std::memcpy(h_last.data(), embd, ctx->hidden_dim * sizeof(float));

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
