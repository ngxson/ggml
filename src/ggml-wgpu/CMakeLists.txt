message(STATUS "Using WebGPU backend")

set(GGML_WGPU_SRC
    ggml-wgpu.cpp
    wgpu-shaders.cpp)

ggml_add_backend_library(ggml-wgpu ${GGML_WGPU_SRC})

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "For now, we only support WebGPU build with emscripten")
    return()
endif()
