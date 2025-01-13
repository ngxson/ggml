#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "ggml-wgpu.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <emscripten/emscripten.h>

// brew install emscripten
// emcmake cmake -B build-web -DGGML_WGPU=ON && cmake --build build-web -j --target wgpu_test
// npx http-server build-web/bin -p 9999 -c-1

// Based on:
// https://github.com/ggerganov/ggml/blob/master/examples/simple/simple-backend.cpp

const int rows_A = 4, cols_A = 2;
const int rows_B = 4, cols_B = 2;
float demo_mat_A[rows_A * cols_A] = {
    2, 8,
    -5, 1,
    4, 2,
    8, -6
};
float demo_mat_B[rows_B * cols_B] = {
    1, 6,
    3, -7,
    9, 0.1,
    -1, 5
};

/**
 * Expected result:
 * [
 *  3.00 14.00
 *  -2.00 -6.00
 *  13.00 2.10
 *  7.00 -1.00
 * ]
 */

struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;

    simple_model() {
        printf("%s: using webgpu backend\n", __func__);
        backend = ggml_backend_wgpu_init();
        //backend = ggml_backend_cpu_init();
        //ggml_backend_cpu_set_n_threads(backend, 1);
        if (!backend) {
            printf("%s: ggml_backend_wgpu_init() failed\n", __func__);
        }

        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 128,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
        ggml_set_name(a, "tensor_a");
        ggml_set_name(b, "tensor_b");

        // for (int i = 0; i < 64; i++) {
        //     auto t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
        //     ggml_format_name(t, "test_%d", i);
        // }

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(a, demo_mat_A, 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, demo_mat_B, 0, ggml_nbytes(b));
    }
};

struct ggml_cgraph * build_graph(const simple_model & model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);
    struct ggml_tensor * result = ggml_add(ctx0, model.a, model.b);
    // result = ggml_div(ctx0, result, model.b);
    // result = ggml_div(ctx0, result, model.b);
    // result = ggml_div(ctx0, result, model.b);
    // result = ggml_div(ctx0, result, model.b);
    ggml_build_forward_expand(gf, result);
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_graph_compute(model.backend, gf);
    return ggml_graph_node(gf, -1);
}

int main() {
    ggml_time_init();
    simple_model model;

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    struct ggml_tensor * result = compute(model, allocr);

    // get result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));
    printf("output (%d x %d):\n[\n", (int) result->ne[0], (int) result->ne[1]);
    for (int i = 0; i < result->ne[1] /* cols */; i++) {
        for (int j = 0; j < result->ne[0] /* rows */; j++) {
            printf(" %.2f", out_data[i * result->ne[0] + j]);
        }
        printf("\n");
    }
    printf("]\n");
    emscripten_sleep(10000);

    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
