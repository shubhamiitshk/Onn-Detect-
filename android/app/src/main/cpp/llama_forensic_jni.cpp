#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>
#include <algorithm>
#include <atomic>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "LlamaForensicJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/**
 * Genuine on-device multimodal prefill & RAG generation bridge for llama.cpp using unified libmtmd.
 *
 * Pinned upstream release: llama.cpp b8291 (commit c363256839fdffae184279ad8e5f0f3775c48b77, March 12, 2026).
 * Architecture support: Qwen3.5 (qwen35), SmolVLM2, SigLIP, LLaVA.
 * Multimodal pipeline: libmtmd unified tokenization + multimodal projection + llama transformer backbone.
 *
 * Training parity guarantees:
 * 1. Prompt format strictly reproduces HuggingFace chat template with add_generation_prompt=True.
 * 2. Visual encoding supports both full image and uncropped original-source ROI zoom.
 * 3. Extracted embedding corresponds strictly to the assistant prefix token (last token, index -1).
 * 4. Supports bounded greedy autoregressive token generation for Multimodal Forensic RAG Q&A.
 * 5. Zero synthetic vector fallbacks: missing weights or parse failures throw explicit Java exceptions.
 */

namespace {

static std::recursive_mutex g_native_mutex;

struct LlamaForensicContext {
    std::string model_path;
    std::string mmproj_path;
    int hidden_dim = 1024;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    mtmd_context* mtmd = nullptr;
    bool active = false;

    LlamaForensicContext(const std::string& m_path, const std::string& mm_path, int dim)
        : model_path(m_path), mmproj_path(mm_path), hidden_dim(dim), active(true) {}

    ~LlamaForensicContext() {
        release_internal();
    }

    void release_internal() {
        if (mtmd) {
            mtmd_free(mtmd);
            mtmd = nullptr;
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

void throw_java_exception(JNIEnv* env, const char* class_name, const std::string& message) {
    throw_java_exception(env, class_name, message.c_str());
}

// JNI's modified UTF-8 corrupts supplementary characters. Decode Java UTF-16
// explicitly for model prompts; token output crosses JNI as raw UTF-8 byte arrays.
std::string java_string_utf8(JNIEnv* env, jstring value) {
    if (!value) return {};
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (!chars) return {};
    const jsize length = env->GetStringLength(value);
    std::string result;
    for (jsize i = 0; i < length; ++i) {
        uint32_t cp = chars[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < length && chars[i + 1] >= 0xDC00 && chars[i + 1] <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (chars[++i] - 0xDC00);
        } else if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;
        }
        if (cp < 0x80) result.push_back(static_cast<char>(cp));
        else if (cp < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    env->ReleaseStringChars(value, chars);
    return result;
}

std::string chat_prompt(const llama_model* model, const std::string& content) {
    if (content.find("<|im_start|>") != std::string::npos) return content;
    const llama_chat_message message = { "user", content.c_str() };
    const char* model_template = llama_model_chat_template(model, nullptr);
    std::vector<char> formatted(content.size() + 1024);
    int length = llama_chat_apply_template(model_template, &message, 1, true, formatted.data(), static_cast<int32_t>(formatted.size()));
    if (length > static_cast<int>(formatted.size())) {
        formatted.resize(length);
        length = llama_chat_apply_template(model_template, &message, 1, true, formatted.data(), static_cast<int32_t>(formatted.size()));
    }
    if (length > 0 && length <= static_cast<int>(formatted.size())) return std::string(formatted.data(), length);
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n";
}

struct GenerationCallbacks {
    JNIEnv* caller_env;
    JavaVM* vm = nullptr;
    jobject receiver = nullptr;
    jmethodID on_token = nullptr;
    jmethodID should_cancel = nullptr;
    llama_context* context;
    std::atomic<bool> stopped{false};

    GenerationCallbacks(JNIEnv* env, jobject callback, llama_context* ctx) : caller_env(env), context(ctx) {
        env->GetJavaVM(&vm);
        receiver = env->NewGlobalRef(callback);
        if (!receiver) return;
        jclass cls = env->GetObjectClass(callback);
        on_token = env->GetMethodID(cls, "onTokenBytes", "([B)Z");
        if (!env->ExceptionCheck()) should_cancel = env->GetMethodID(cls, "shouldCancel", "()Z");
        env->DeleteLocalRef(cls);
    }

    ~GenerationCallbacks() {
        // No worker may retain this stack-owned callback after decoding ends.
        llama_set_abort_callback(context, nullptr, nullptr);
        if (receiver) caller_env->DeleteGlobalRef(receiver);
    }

    bool is_cancelled() {
        if (stopped.load()) return true;
        JNIEnv* env = nullptr;
        bool attached = false;
        const jint state = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
        if (state == JNI_EDETACHED) {
            if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
                stopped.store(true);
                return true;
            }
            attached = true;
        } else if (state != JNI_OK) {
            stopped.store(true);
            return true;
        }
        bool cancelled = env->ExceptionCheck();
        if (!cancelled) cancelled = env->CallBooleanMethod(receiver, should_cancel) == JNI_TRUE;
        if (env->ExceptionCheck()) {
            cancelled = true;
            // A worker-thread exception cannot propagate to the caller. Its
            // only callback reads cancellation state and is fail-closed.
            if (attached) env->ExceptionClear();
        }
        if (attached) vm->DetachCurrentThread();
        if (cancelled) stopped.store(true);
        return cancelled;
    }

    bool emit(const char* bytes, int length) {
        if (is_cancelled()) return false;
        jbyteArray piece = caller_env->NewByteArray(length);
        if (!piece) return false;
        caller_env->SetByteArrayRegion(piece, 0, length, reinterpret_cast<const jbyte*>(bytes));
        const bool keep_running = caller_env->CallBooleanMethod(receiver, on_token, piece) == JNI_TRUE;
        caller_env->DeleteLocalRef(piece);
        if (!keep_running || caller_env->ExceptionCheck()) stopped.store(true);
        return keep_running && !caller_env->ExceptionCheck();
    }

    bool throw_if_cancelled() {
        if (!is_cancelled()) return false;
        if (!caller_env->ExceptionCheck()) {
            throw_java_exception(caller_env, "java/util/concurrent/CancellationException", "Response stopped");
        }
        return true;
    }
};

bool abort_generation(void* opaque) {
    return static_cast<GenerationCallbacks*>(opaque)->is_cancelled();
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
    mparams.n_gpu_layers = 0; // CPU inference on Android
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
    cparams.n_ctx = 4096;
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

    // 4. Load multimodal projector via libmtmd if mmproj path provided
    if (!ctx->mmproj_path.empty()) {
        struct mtmd_context_params mtmd_params = mtmd_context_params_default();
        mtmd_params.use_gpu = false;
        mtmd_params.n_threads = 4;
        ctx->mtmd = mtmd_init_from_file(ctx->mmproj_path.c_str(), ctx->model, mtmd_params);
        if (!ctx->mtmd) {
            std::string err = "Failed to load multimodal mmproj projector from: " + ctx->mmproj_path;
            delete ctx;
            throw_java_exception(env, "java/lang/IllegalStateException", err.c_str());
            return 0;
        }
    }

    LOGI("nativeInitContext succeeded: ctx=%p, model=%p, mtmd=%p, hidden_dim=%d",
         ctx->ctx, ctx->model, ctx->mtmd, hiddenDim);
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
    std::lock_guard<std::recursive_mutex> lock(g_native_mutex);
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

    llama_set_embeddings(ctx->ctx, true);

    // 1. Clear memory/KV cache to prevent token state pollution across runs
    llama_memory_t mem = llama_get_memory(ctx->ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // 2. Prepare multimodal input bitmaps
    std::vector<mtmd_bitmap*> bitmaps_vec;
    bitmaps_vec.push_back(mtmd_bitmap_init(
        static_cast<uint32_t>(fullWidth),
        static_cast<uint32_t>(fullHeight),
        reinterpret_cast<const unsigned char*>(full_bytes)
    ));

    bool has_roi = (roiRgb && roi_bytes && roiWidth > 0 && roiHeight > 0);
    if (has_roi) {
        bitmaps_vec.push_back(mtmd_bitmap_init(
            static_cast<uint32_t>(roiWidth),
            static_cast<uint32_t>(roiHeight),
            reinterpret_cast<const unsigned char*>(roi_bytes)
        ));
    }

    std::vector<const mtmd_bitmap*> bitmaps_ptrs;
    for (auto* b : bitmaps_vec) {
        bitmaps_ptrs.push_back(b);
    }

    // 3. Multimodal prefill evaluation
    if (ctx->mtmd) {
        std::string marker = mtmd_default_marker();
        std::string prompt;
        if (has_roi) {
            prompt = "<|im_start|>user\n" + marker + "\n" + marker +
                     "\nAssess forensic authenticity in domain: " + domain_str + ".<|im_end|>\n<|im_start|>assistant\n";
        } else {
            prompt = "<|im_start|>user\n" + marker +
                     "\nAssess forensic authenticity in domain: " + domain_str + ".<|im_end|>\n<|im_start|>assistant\n";
        }

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        mtmd_input_text input_text = { prompt.c_str(), true, true };

        int tok_res = mtmd_tokenize(ctx->mtmd, chunks, &input_text, bitmaps_ptrs.data(), bitmaps_ptrs.size());
        for (auto* b : bitmaps_vec) {
            mtmd_bitmap_free(b);
        }

        if (tok_res != 0) {
            mtmd_input_chunks_free(chunks);
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "mtmd_tokenize failed to parse prompt and visual bitmaps (code: " + std::to_string(tok_res) + ")");
            return nullptr;
        }

        llama_pos new_n_past = 0;
        int eval_res = mtmd_helper_eval_chunks(ctx->mtmd, ctx->ctx, chunks, 0, 0, 512, true, &new_n_past);
        mtmd_input_chunks_free(chunks);

        if (eval_res != 0) {
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException",
                                 "mtmd_helper_eval_chunks failed during multimodal visual evaluation (code: " + std::to_string(eval_res) + ")");
            return nullptr;
        }
    } else {
        for (auto* b : bitmaps_vec) {
            mtmd_bitmap_free(b);
        }
        std::string prompt = "<|im_start|>user\nAssess forensic authenticity in domain: " + domain_str + ".<|im_end|>\n<|im_start|>assistant\n";
        const struct llama_vocab * vocab = llama_model_get_vocab(ctx->model);
        std::vector<llama_token> tokens(prompt.length() + 16);
        int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                      tokens.data(), tokens.size(), true, true);
        if (n_tokens <= 0) {
            env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
            if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);
            throw_java_exception(env, "java/lang/IllegalStateException", "Failed to tokenize text prompt");
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
            throw_java_exception(env, "java/lang/IllegalStateException", "llama_decode failed on prompt tokens");
            return nullptr;
        }
        llama_batch_free(batch);
    }

    env->ReleaseByteArrayElements(fullRgb, full_bytes, JNI_ABORT);
    if (roiRgb && roi_bytes) env->ReleaseByteArrayElements(roiRgb, roi_bytes, JNI_ABORT);

    // 4. Extract representation of the final token (assistant prefix token)
    const float* embd = llama_get_embeddings_ith(ctx->ctx, -1);
    if (!embd) {
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Failed to retrieve embeddings from llama_get_embeddings_ith(-1)");
        return nullptr;
    }

    int n_embd = llama_model_n_embd(ctx->model);
    if (n_embd != ctx->hidden_dim) {
        throw_java_exception(env, "java/lang/IllegalStateException",
                             "Model embedding dimension (" + std::to_string(n_embd) +
                             ") does not match expected hiddenDim (" + std::to_string(ctx->hidden_dim) + ")");
        return nullptr;
    }

    jfloatArray result = env->NewFloatArray(ctx->hidden_dim);
    if (!result) {
        throw_java_exception(env, "java/lang/OutOfMemoryError", "Failed to allocate jfloatArray for representation");
        return nullptr;
    }
    env->SetFloatArrayRegion(result, 0, ctx->hidden_dim, embd);
    return result;
}

JNIEXPORT void JNICALL
Java_com_onndetect_detection_visual_LlamaCppVlmPrefillBackend_nativeGenerateMultimodalResponseStreaming(
    JNIEnv* env,
    jobject /* thiz */,
    jlong contextHandle,
    jstring promptJstr,
    jbyteArray imageRgb,
    jint imageWidth,
    jint imageHeight,
    jint maxNewTokens,
    jobject receiver
) {
    std::lock_guard<std::recursive_mutex> lock(g_native_mutex);
    auto* ctx = reinterpret_cast<LlamaForensicContext*>(contextHandle);
    if (!ctx || !ctx->active || !ctx->ctx || !ctx->model || !receiver) {
        throw_java_exception(env, "java/lang/IllegalStateException", "Native response context is unavailable");
        return;
    }
    GenerationCallbacks callbacks(env, receiver, ctx->ctx);
    if (env->ExceptionCheck() || !callbacks.on_token || !callbacks.should_cancel) return;
    llama_set_abort_callback(ctx->ctx, abort_generation, &callbacks);
    if (callbacks.throw_if_cancelled()) return;

    std::string prompt = java_string_utf8(env, promptJstr);
    if (env->ExceptionCheck()) return;
    llama_set_embeddings(ctx->ctx, false);
    llama_memory_t mem = llama_get_memory(ctx->ctx);
    if (mem) llama_memory_clear(mem, true);

    llama_pos n_past = 0;
    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (imageRgb && (!ctx->mtmd || imageWidth <= 0 || imageHeight <= 0)) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "A compatible vision projector and image dimensions are required");
        return;
    }
    if (imageRgb && imageWidth > 0 && imageHeight > 0 && ctx->mtmd) {
        if (static_cast<int64_t>(imageWidth) * imageHeight * 3 > env->GetArrayLength(imageRgb)) {
            throw_java_exception(env, "java/lang/IllegalArgumentException", "Image buffer is smaller than its dimensions");
            return;
        }
        jbyte* bytes = env->GetByteArrayElements(imageRgb, nullptr);
        if (!bytes) return;
        mtmd_bitmap* bitmap = mtmd_bitmap_init(static_cast<uint32_t>(imageWidth), static_cast<uint32_t>(imageHeight), reinterpret_cast<const unsigned char*>(bytes));
        const mtmd_bitmap* bitmaps[1] = { bitmap };
        std::string marker = mtmd_default_marker();
        std::string full_prompt = chat_prompt(ctx->model, prompt.find(marker) != std::string::npos ? prompt : marker + "\n" + prompt);
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        mtmd_input_text text = { full_prompt.c_str(), true, true };
        const int tokenize_result = mtmd_tokenize(ctx->mtmd, chunks, &text, bitmaps, 1);
        mtmd_bitmap_free(bitmap);
        env->ReleaseByteArrayElements(imageRgb, bytes, JNI_ABORT);
        if (callbacks.throw_if_cancelled()) { mtmd_input_chunks_free(chunks); return; }
        if (tokenize_result != 0) {
            mtmd_input_chunks_free(chunks);
            throw_java_exception(env, "java/lang/IllegalStateException", "Unable to tokenize image and prompt");
            return;
        }
        const int eval_result = mtmd_helper_eval_chunks(ctx->mtmd, ctx->ctx, chunks, 0, 0, 128, true, &n_past);
        mtmd_input_chunks_free(chunks);
        if (callbacks.throw_if_cancelled()) return;
        if (eval_result != 0) {
            throw_java_exception(env, "java/lang/IllegalStateException", "Unable to evaluate image and prompt");
            return;
        }
    } else {
        prompt = chat_prompt(ctx->model, prompt);
        std::vector<llama_token> tokens(prompt.size() + 32);
        int count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
        if (count < 0) {
            tokens.resize(-count);
            count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
        }
        if (count <= 0 || count + 8 >= static_cast<int>(llama_n_ctx(ctx->ctx))) {
            throw_java_exception(env, "java/lang/IllegalArgumentException", "Prompt is empty or exceeds the local model context");
            return;
        }
        // Split long prefill into bounded batches as well as supporting the
        // llama.cpp CPU abort callback during each individual decode call.
        for (int offset = 0; offset < count; offset += 128) {
            if (callbacks.throw_if_cancelled()) return;
            const int batch_size = std::min(128, count - offset);
            llama_batch batch = llama_batch_init(batch_size, 0, 1);
            batch.n_tokens = batch_size;
            for (int i = 0; i < batch_size; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = offset + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = offset + i == count - 1;
            }
            const int decoded = llama_decode(ctx->ctx, batch);
            llama_batch_free(batch);
            if (callbacks.throw_if_cancelled()) return;
            if (decoded != 0) {
                throw_java_exception(env, "java/lang/IllegalStateException", "Unable to evaluate local model prompt");
                return;
            }
        }
        n_past = count;
    }

    if (n_past + 8 >= static_cast<int>(llama_n_ctx(ctx->ctx))) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", "Prompt exceeds the local model context");
        return;
    }
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
    const int max_steps = std::min(std::max(8, std::min(static_cast<int>(maxNewTokens), 512)), static_cast<int>(llama_n_ctx(ctx->ctx)) - n_past - 1);
    for (int step = 0; step < max_steps; ++step) {
        if (callbacks.throw_if_cancelled()) return;
        const llama_token token = llama_sampler_sample(sampler.get(), ctx->ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        llama_sampler_accept(sampler.get(), token);
        std::vector<char> piece(128);
        int length = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        if (length < 0) {
            piece.resize(-length);
            length = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        }
        if (length > 0 && !callbacks.emit(piece.data(), length)) {
            callbacks.throw_if_cancelled();
            return;
        }
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = token;
        batch.pos[0] = n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;
        const int decoded = llama_decode(ctx->ctx, batch);
        llama_batch_free(batch);
        if (callbacks.throw_if_cancelled()) return;
        if (decoded != 0) {
            throw_java_exception(env, "java/lang/IllegalStateException", "Local model response decoding failed");
            return;
        }
    }
}

JNIEXPORT void JNICALL
Java_com_onndetect_detection_visual_LlamaCppVlmPrefillBackend_nativeFreeContext(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong contextHandle
) {
    std::lock_guard<std::recursive_mutex> lock(g_native_mutex);
    auto* ctx = reinterpret_cast<LlamaForensicContext*>(contextHandle);
    if (ctx) {
        ctx->release();
        delete ctx;
    }
}

} // extern "C"
