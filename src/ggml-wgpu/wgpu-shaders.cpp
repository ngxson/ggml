#include <string>
#include <sstream>
#include <vector>
#include "webgpu.hpp"
#include "wgpu-shaders.h"
#include "ggml.h"

static const char * get_shader_code_header() {
    return R"(
        struct TensorParams {
            ne00 : i32,
            ne01 : i32,
            ne02 : i32,
            ne03 : i32,

            nb00 : u32,
            nb01 : u32,
            nb02 : u32,
            nb03 : u32,

            ne10 : i32,
            ne11 : i32,
            ne12 : i32,
            ne13 : i32,

            nb10 : u32,
            nb11 : u32,
            nb12 : u32,
            nb13 : u32,

            ne0 : i32,
            ne1 : i32,
            ne2 : i32,
            ne3 : i32,

            nb0 : u32,
            nb1 : u32,
            nb2 : u32,
            nb3 : u32,

            offs_src0 : u32,
            offs_src1 : u32,
            offs_dest : u32,
        }

        @group(0) @binding(0)
        var<storage,read_write> src0: array<f32>;

        @group(0) @binding(1)
        var<storage,read_write> src1: array<f32>;

        @group(0) @binding(2)
        var<storage,read_write> dest: array<f32>;

        @group(0) @binding(3)
        var<uniform> tensor_params: TensorParams;
    )";
}

const ggml_wgpu_shader * ggml_wgpu_get_shader(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE: {
            static const ggml_wgpu_shader sh{
                /* .name */ "kernel_none",
                /* .code */ R"(
                    @compute
                    @workgroup_size(1)
                    fn kernel_none(@builtin(global_invocation_id) global_id: vec3<u32>) {}
                )",
            };
            return &sh;
        }

        //////////////////////////////////////////////////////////

        case GGML_OP_DUP: {
            static const ggml_wgpu_shader sh{
                /* .name */ "kernel_dup",
                /* .code */ R"(
                    @compute
                    @workgroup_size(1)
                    fn kernel_dup(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let x = src0[global_id.x + tensor_params.offs_src0/4u];
                                dest[global_id.x + tensor_params.offs_dest/4u] = x;
                    }
                )",
            };
            return &sh;
        }
        case GGML_OP_ADD: {
            static const ggml_wgpu_shader sh{
                /* .name */ "kernel_add",
                /* .code */ R"(
                    @compute
                    @workgroup_size(1)
                    fn kernel_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let x = src0[global_id.x + tensor_params.offs_src0/4u];
                        let y = src1[global_id.x + tensor_params.offs_src1/4u];
                                dest[global_id.x + tensor_params.offs_dest/4u] = x + y;
                    }
                )",
            };
            return &sh;
        }
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
            return nullptr;
        case GGML_OP_DIV: {
            static const ggml_wgpu_shader sh{
                /* .name */ "kernel_div",
                /* .code */ R"(
                    @compute
                    @workgroup_size(1)
                    fn kernel_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let x = src0[global_id.x + tensor_params.offs_src0/4u];
                        let y = src1[global_id.x + tensor_params.offs_src1/4u];
                                dest[global_id.x + tensor_params.offs_dest/4u] = x / y;
                    }
                )",
            };
            return &sh;
        }
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_CONCAT:
        case GGML_OP_SILU_BACK:
        case GGML_OP_NORM: // normalize
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_GROUP_NORM:

        //////////////////////////////////////////////////////////

        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:

        //////////////////////////////////////////////////////////

        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_CLAMP:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_IM2COL:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_UPSCALE: // nearest interpolate
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_LEAKY_RELU:

        //////////////////////////////////////////////////////////

        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
        case GGML_OP_ADD_REL_POS:

        //////////////////////////////////////////////////////////

        case GGML_OP_UNARY:

        //////////////////////////////////////////////////////////

        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:

        //////////////////////////////////////////////////////////

        case GGML_OP_MAP_CUSTOM1_F32:
        case GGML_OP_MAP_CUSTOM2_F32:
        case GGML_OP_MAP_CUSTOM3_F32:

        //////////////////////////////////////////////////////////

        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:

        //////////////////////////////////////////////////////////

        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_COUNT:
        default:
            return nullptr;
    }
}

std::string ggml_wgpu_build_shader_code() {
    std::ostringstream ss;
    ss << get_shader_code_header();
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        const ggml_wgpu_shader * shader = ggml_wgpu_get_shader(static_cast<enum ggml_op>(i));
        if (shader == nullptr) continue;
        ss << shader->code;
    }
    return ss.str();
}
