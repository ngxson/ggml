#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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

float randomFloat(float fmin, float fmax) {
    return ((float)(rand()) / (float)(RAND_MAX)) * (fmax - fmin) + fmin;
}

const int rows_A = 512, cols_A = 256;
const int rows_B = 512, cols_B = 256;
float demo_mat_A[rows_A * cols_A];
float demo_mat_B[rows_B * cols_B];

float imat_A[rows_A * cols_A];
float imat_B[rows_B * cols_B];

int SEED = 42;
void init_demo_data() {
    srand(SEED);
    for (int i = 0; i < rows_A * cols_A; i++) {
        demo_mat_A[i] = randomFloat(-1.0f, 1.0f);
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        demo_mat_B[i] = randomFloat(-1.0f, 1.0f);
    }
    for (int i = 0; i < rows_A * cols_A; i++) {
        imat_A[i] = randomFloat(-0.5f, 0.5f);
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        imat_B[i] = randomFloat(-0.5f, 0.5f);
    }
}

void quantize(ggml_type type_out, float * src, float * dst, int nrow, int n_per_row, float * imat) {
    //size_t row_size = ggml_row_size(type_out, n_per_row);
    auto type_trait = ggml_get_type_traits_cpu(type_out);
    if (type_out == GGML_TYPE_F16) {
        ggml_quantize_chunk(type_out, src, dst, 0, nrow, n_per_row, imat);
    } else {
        type_trait->from_float(src, dst, (int64_t)nrow*n_per_row);
    }
}

static ggml_backend_t backend = NULL;

struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;

    simple_model(ggml_type ta, ggml_type tb) {
        if (!backend) {
            printf("%s: ggml_backend_wgpu_init() failed\n", __func__);
        }

        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * 128,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);

        if (ta != GGML_TYPE_F32) {
            float quant_mat_A[rows_A * cols_A];
            quantize(ta, demo_mat_A, quant_mat_A, rows_A, cols_A, imat_A);
            memcpy(demo_mat_A, quant_mat_A, sizeof(quant_mat_A));
        }

        if (tb != GGML_TYPE_F32) {
            float quant_mat_B[rows_B * cols_B];
            quantize(tb, demo_mat_B, quant_mat_B, rows_B, cols_B, imat_B);
            memcpy(demo_mat_B, quant_mat_B, sizeof(quant_mat_B));
        }

        a = ggml_new_tensor_2d(ctx, ta, cols_A, rows_A);
        b = ggml_new_tensor_2d(ctx, tb, cols_B, rows_B);
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
    struct ggml_tensor * result = ggml_mul_mat(ctx0, model.a, model.b);
    ggml_build_forward_expand(gf, result);
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);
    ggml_gallocr_alloc_graph(allocr, gf);
    ggml_backend_graph_compute(backend, gf);
    return ggml_graph_node(gf, -1);
}

int run(ggml_type ta, ggml_type tb) {
    init_demo_data();

    ggml_time_init();
    simple_model model(ta, tb);

    auto t_start = ggml_time_ms();
    const int N_RUN = 100;
    printf("run %d times, ta = %s, tb = %s\n", N_RUN, ggml_type_name(ta), ggml_type_name(tb));
    float sum = 0;
    for (int i = 0; i < N_RUN; i++) {
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        struct ggml_tensor * result = compute(model, allocr);

        // get result
        std::vector<float> out_data(ggml_nelements(result));

        // bring the data from the backend memory
        ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));
        //printf("output sum (%d x %d): \n", (int) result->ne[0], (int) result->ne[1]);
        sum = 0;
        for (int i = 0; i < result->ne[1] /* cols */; i++) {
            for (int j = 0; j < result->ne[0] /* rows */; j++) {
                //printf(" %.2f", out_data[i * result->ne[0] + j]);
                sum += out_data[i * result->ne[0] + j];
            }
            //printf("\n");
        }
        //printf("]\n");
        ggml_gallocr_free(allocr);
    }

    printf("sum all elem = %f, time elapsed: %lld ms\n", sum, ggml_time_ms() - t_start);
    // expected = -6873.335938

    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);

    emscripten_sleep(100);
    return 0;
}

int main() {
    printf("%s: using cpu backend\n", __func__);
    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    printf("sleep for 1s to settle\n");
    emscripten_sleep(1000);
    for (; SEED < 45; SEED++) {
        run(GGML_TYPE_F32, GGML_TYPE_F32);
        //run(GGML_TYPE_F16, GGML_TYPE_F32);

        //run(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
        //run(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
        //run(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);

        //run(GGML_TYPE_Q6_K, GGML_TYPE_Q8_K);
        //run(GGML_TYPE_Q5_K, GGML_TYPE_Q8_K);
        //run(GGML_TYPE_Q4_K, GGML_TYPE_Q8_K);
        run(GGML_TYPE_Q3_K, GGML_TYPE_Q8_K);
        run(GGML_TYPE_Q2_K, GGML_TYPE_Q8_K);

        // run(GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K);

        // run(GGML_TYPE_IQ3_S, GGML_TYPE_Q8_K);
        // run(GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K);

        // run(GGML_TYPE_IQ2_S, GGML_TYPE_Q8_K);
        // run(GGML_TYPE_IQ2_XS, GGML_TYPE_Q8_K);
        // run(GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K);

        printf("\n\n====================\n\n");
    }

    ggml_backend_free(backend);
}
