cmake_minimum_required(VERSION 3.22.1)

# Pinned upstream llama.cpp release: b8291 (commit c363256839fdffae184279ad8e5f0f3775c48b77, March 12, 2026)
# Genuine on-device multimodal prefill runtime with Qwen3.5 (qwen35) architecture & libmtmd
project("llama_forensic_jni" C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enforce 16 KiB ELF alignment for Android 15+ 16 KiB page size support
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Configure llama.cpp build options
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Disable CLI tools; mtmd sources compiled directly into JNI target" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for Android" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Disable ccache" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries to link directly into JNI shared library" FORCE)

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b8291
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

set(MTMD_DIR "${llama_cpp_SOURCE_DIR}/tools/mtmd")

set(MTMD_SOURCES
    "${MTMD_DIR}/mtmd.cpp"
    "${MTMD_DIR}/mtmd-audio.cpp"
    "${MTMD_DIR}/mtmd-helper.cpp"
    "${MTMD_DIR}/clip.cpp"
    "${MTMD_DIR}/models/cogvlm.cpp"
    "${MTMD_DIR}/models/conformer.cpp"
    "${MTMD_DIR}/models/glm4v.cpp"
    "${MTMD_DIR}/models/internvl.cpp"
    "${MTMD_DIR}/models/kimik25.cpp"
    "${MTMD_DIR}/models/kimivl.cpp"
    "${MTMD_DIR}/models/llama4.cpp"
    "${MTMD_DIR}/models/llava.cpp"
    "${MTMD_DIR}/models/minicpmv.cpp"
    "${MTMD_DIR}/models/mobilenetv5.cpp"
    "${MTMD_DIR}/models/nemotron-v2-vl.cpp"
    "${MTMD_DIR}/models/paddleocr.cpp"
    "${MTMD_DIR}/models/pixtral.cpp"
    "${MTMD_DIR}/models/qwen2vl.cpp"
    "${MTMD_DIR}/models/qwen3vl.cpp"
    "${MTMD_DIR}/models/siglip.cpp"
    "${MTMD_DIR}/models/whisper-enc.cpp"
    "${MTMD_DIR}/models/youtuvl.cpp"
)

add_library(
    llama_forensic_jni
    SHARED
    llama_forensic_jni.cpp
    ${MTMD_SOURCES}
)

target_include_directories(
    llama_forensic_jni
    PRIVATE
    "${llama_cpp_SOURCE_DIR}/include"
    "${llama_cpp_SOURCE_DIR}/ggml/include"
    "${llama_cpp_SOURCE_DIR}/common"
    "${MTMD_DIR}"
    "${llama_cpp_SOURCE_DIR}/vendor"
    "${llama_cpp_SOURCE_DIR}/vendor/stb"
    "${llama_cpp_SOURCE_DIR}/vendor/miniaudio"
)

target_compile_options(llama_forensic_jni PRIVATE -Wno-cast-qual)

find_library(log-lib log)

target_link_libraries(
    llama_forensic_jni
    PRIVATE
    llama
    ggml
    ggml-base
    ggml-cpu
    ${log-lib}
)
