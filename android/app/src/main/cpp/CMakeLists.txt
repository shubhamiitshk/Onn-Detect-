cmake_minimum_required(VERSION 3.22.1)

# Pinned upstream llama.cpp release: b4832 (commit 0b30268, March 2026, PR #20334 Vulkan GDN)
# Native JNI Multimodal Bridge for ONN-DETECT Tier-2 VLM
project("llama_forensic_jni" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
    llama_forensic_jni
    SHARED
    llama_forensic_jni.cpp
)

find_library(
    log-lib
    log
)

find_library(
    dl-lib
    dl
)

target_link_libraries(
    llama_forensic_jni
    ${log-lib}
    ${dl-lib}
)
