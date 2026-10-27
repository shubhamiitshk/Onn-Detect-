cmake_minimum_required(VERSION 3.22.1)

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

target_link_libraries(
    llama_forensic_jni
    ${log-lib}
)
