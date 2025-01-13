#pragma once

#define WEBGPU_CPP_IMPLEMENTATION
#include "webgpu.hpp"
#include "ggml.h"
#include <string>

struct ggml_wgpu_shader {
  const char * name;
  const char * code;
  const char * inpl; // code for inplace shader
};

const ggml_wgpu_shader * ggml_wgpu_get_shader(enum ggml_op op);
std::string ggml_wgpu_build_shader_code(bool inplace);
