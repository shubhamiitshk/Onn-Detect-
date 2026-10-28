cmake_minimum_required(VERSION 3.22.1)

# Pinned upstream llama.cpp release: b4832 (commit 07d15723470a0a5b15d8ccad1aff5b20354ffbe1, March 6, 2025)
# Genuine on-device multimodal prefill runtime for ONN-DETECT Tier-2 VLM
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
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for Android" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Disable ccache" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries to link directly into JNI shared library" FORCE)

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b4832
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# Multimodal vision projector sources from llama.cpp examples/llava
set(LLAVA_DIR "${llama_cpp_SOURCE_DIR}/examples/llava")

add_library(
    llama_forensic_jni
    SHARED
    llama_forensic_jni.cpp
    "${LLAVA_DIR}/clip.cpp"
    "${LLAVA_DIR}/llava.cpp"
)

target_include_directories(
    llama_forensic_jni
    PRIVATE
    "${llama_cpp_SOURCE_DIR}/include"
    "${llama_cpp_SOURCE_DIR}/ggml/include"
    "${llama_cpp_SOURCE_DIR}/common"
    "${LLAVA_DIR}"
)

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
