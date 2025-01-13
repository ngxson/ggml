#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <emscripten/emscripten.h>

#define WEBGPU_CPP_IMPLEMENTATION
#include "webgpu.hpp"
#include "wgpu-shaders.h"

#include "ggml-wgpu.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#define BUF_ALIGN 256
#define BUF_BASE  0x1000
#define UNUSED GGML_UNUSED

#define WGPU_ENABLE_LOG_TRACE

#ifdef WGPU_ENABLE_LOG_TRACE
// trace
#define LOGT(...) printf(__VA_ARGS__)
#else
#define LOGT(...) // do nothing
#endif

#define LOGD(...) printf(__VA_ARGS__)

// TODO: check secured context (https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#security_requirements)

// https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/input-geometry/playing-with-buffers.html
void wgpuPollEvents(wgpu::Device & device, bool yieldToWebBrowser) {
#if defined(WEBGPU_BACKEND_DAWN)
    GGML_UNUSED(yieldToWebBrowser);
    device.tick();
#elif defined(WEBGPU_BACKEND_WGPU)
    GGML_UNUSED(yieldToWebBrowser);
    device.poll(false);
#elif defined(__EMSCRIPTEN__)
    GGML_UNUSED(device);
    LOGT("%s [emscripten]\n", __func__);
    if (yieldToWebBrowser) {
        emscripten_sleep(1); // IMPORTANT: need linker flag -sASYNCIFY=1
    }
#endif
}

size_t wgpu_tensor_get_nbytes(const ggml_tensor * tensor) {
    return GGML_PAD(ggml_nbytes(tensor), BUF_ALIGN);
}

struct ggml_wgpu_context;
struct ggml_wgpu_buffer_context;

struct ggml_wgpu_context {
    wgpu::Instance        instance;
    wgpu::Adapter         adapter;
    wgpu::Device          device;
    //wgpu::SupportedLimits limits;
    wgpu::Queue           queue;
    wgpu::ShaderModule    shader_module;
    wgpu::BindGroupLayout bind_group_layout;
    wgpu::PipelineLayout  pipeline_layout;

    // one pipeline per kernel
    wgpu::ComputePipeline pipeline_op[GGML_OP_COUNT];
    wgpu::ComputePipeline pipeline_inpl_op[GGML_OP_COUNT];
    wgpu::ComputePipeline pipeline_unary_op[GGML_UNARY_OP_COUNT];

    ggml_backend_buffer_type_t buft = nullptr;
    bool                  buft_initialized = false;
    wgpu::Buffer          buf_dummy;
    wgpu::Buffer          buf_tensor_params;
    ggml_wgpu_tensor_params tensor_params_host;

    ggml_wgpu_context() {
        LOGT("%s [constructor]\n", __func__);
        // instance = wgpu::createInstance(&instanceDesc);
        // descriptor not implemented yet in emscripten
        instance = wgpuCreateInstance(nullptr);
        wgpu::RequestAdapterOptions reqAdaptOpts = wgpu::Default;
        adapter = instance.requestAdapter(reqAdaptOpts);
        wgpu::DeviceDescriptor deviceDesc = wgpu::Default;
        device = adapter.requestDevice(deviceDesc);
        device.setLabel("wgpu_device");
        queue = device.getQueue();

        // init bind group layout
        wgpu::BindGroupLayoutEntry bglEntries[4];
        {
            bglEntries[0].setDefault();
            bglEntries[0].binding = 0;
            bglEntries[0].visibility = wgpu::ShaderStage::Compute;
            bglEntries[0].buffer.type = wgpu::BufferBindingType::Storage;
            bglEntries[0].buffer.hasDynamicOffset = false;
            bglEntries[0].buffer.minBindingSize = 0;

            bglEntries[1].setDefault();
            bglEntries[1].binding = 1;
            bglEntries[1].visibility = wgpu::ShaderStage::Compute;
            bglEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
            bglEntries[1].buffer.hasDynamicOffset = false;
            bglEntries[1].buffer.minBindingSize = 0;

            bglEntries[2].setDefault();
            bglEntries[2].binding = 2;
            bglEntries[2].visibility = wgpu::ShaderStage::Compute;
            bglEntries[2].buffer.type = wgpu::BufferBindingType::Storage;
            bglEntries[2].buffer.hasDynamicOffset = false;
            bglEntries[2].buffer.minBindingSize = 0;

            bglEntries[3].setDefault();
            bglEntries[3].binding = 3;
            bglEntries[3].visibility = wgpu::ShaderStage::Compute;
            bglEntries[3].buffer.type = wgpu::BufferBindingType::Uniform;
            bglEntries[3].buffer.hasDynamicOffset = false;
            bglEntries[3].buffer.minBindingSize = sizeof(ggml_wgpu_tensor_params);
        }
        wgpu::BindGroupLayoutDescriptor bglDesc = wgpu::Default;
        {
            bglDesc.label = "ggml-wgpu-bind-group-layout";
            bglDesc.entryCount = 4;
            bglDesc.entries = bglEntries;
        };
        bind_group_layout = device.createBindGroupLayout(bglDesc);
        GGML_ASSERT(bind_group_layout && "cannot create BindGroupLayout");

        // load shaders
        {
            wgpu::ShaderModuleWGSLDescriptor wgslDesc = wgpu::Default;
            auto code = new std::string(ggml_wgpu_build_shader_code()); // TODO: free
            wgslDesc.code = code->c_str();
            // LOGD("%s\n", wgslDesc.code);
            wgpu::ShaderModuleDescriptor shaderModuleDescriptor;
            shaderModuleDescriptor.nextInChain = (const WGPUChainedStruct *) &wgslDesc;
            shader_module = device.createShaderModule(shaderModuleDescriptor);
            GGML_ASSERT(shader_module && "cannot create shaderModule");
        }

        // create pipeline from shader
        {
            wgpu::PipelineLayoutDescriptor plDesc = wgpu::Default;
            {
                plDesc.label = "ggml-wgpu-pipeline-layout";
                plDesc.bindGroupLayoutCount = 1;
                plDesc.bindGroupLayouts = (const WGPUBindGroupLayout *) &bind_group_layout;
            };
            pipeline_layout = device.createPipelineLayout(plDesc);
            GGML_ASSERT(pipeline_layout);

            for (int i = 0; i < GGML_OP_COUNT; i++) {
                const ggml_wgpu_shader * shader = ggml_wgpu_get_shader(static_cast<enum ggml_op>(i));
                if (shader == nullptr) {
                    pipeline_op[i] = nullptr;
                    continue;
                }
                wgpu::ComputePipelineDescriptor cpDesc = wgpu::Default;
                {
                    cpDesc.label = shader->name;
                    cpDesc.layout = pipeline_layout;
                    wgpu::ProgrammableStageDescriptor psDesc = wgpu::Default;
                    {
                        psDesc.module = shader_module;
                        psDesc.entryPoint = shader->name;
                    }
                    cpDesc.compute = psDesc;
                };
                pipeline_op[i] = device.createComputePipeline(cpDesc);
                GGML_ASSERT(pipeline_op[i] && "cannot create Pipeline");
                // create inplace version if needed
                if (shader->inpl) {
                    std::string entrypoint = std::string(shader->name) + "_inplace";
                    cpDesc.label = entrypoint.c_str();
                    cpDesc.compute.entryPoint = entrypoint.c_str();
                    pipeline_inpl_op[i] = device.createComputePipeline(cpDesc);
                    GGML_ASSERT(pipeline_inpl_op[i] && "cannot create Pipeline (inplace)");
                }
            }
        }

        // alloc buffer to store tensor params
        {
            wgpu::BufferDescriptor bufDesc = wgpu::Default;
            {
                bufDesc.label = "buffer_params";
                bufDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
                bufDesc.size = GGML_PAD(sizeof(ggml_wgpu_tensor_params), BUF_ALIGN);
                bufDesc.mappedAtCreation = false;
            };
            buf_tensor_params = device.createBuffer(bufDesc);
            GGML_ASSERT(buf_tensor_params && "cannot create buffer_params");
        }

        // alloc dumy buffer, to be used by inplace ops
        {
            wgpu::BufferDescriptor bufDesc = wgpu::Default;
            {
                bufDesc.label = "buffer_dummy";
                bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
                bufDesc.size = BUF_ALIGN;
                bufDesc.mappedAtCreation = false;
            };
            buf_dummy = device.createBuffer(bufDesc);
            GGML_ASSERT(buf_dummy && "cannot create buffer_dummy");
        }
    }

    ~ggml_wgpu_context() {
        // TODO: clean up other things if needed
        device.release();
    }
};

// we only support single device for now
static ggml_wgpu_context * ggml_wgpu_ctx_instance = nullptr;
using ggml_wgpu_buffer_type_context = ggml_wgpu_context;

int buff_id = 0;
struct ggml_wgpu_buffer_context {
    ggml_wgpu_context * ctx;
    wgpu::Buffer buffer;
    size_t       size;
    size_t       next_free_ptr = 0;
    std::string  label;

    ggml_wgpu_buffer_context(ggml_wgpu_buffer_type_context * _ctx, size_t aligned_size): size(aligned_size), ctx(_ctx) {
        label = std::string("wgpu_buf_") + std::to_string(buff_id++);
        LOGT("%s: [constructor] buf=%s size=%ld\n", __func__, label.c_str(), size);
        init_buf();
    }

    ~ggml_wgpu_buffer_context() {
        LOGT("%s: free\n", label.c_str());
        buffer.unmap();
    }

    void init_buf() {
        wgpu::BufferDescriptor bufDesc = wgpu::Default;
        {
            bufDesc.label = label.c_str();
            bufDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
            bufDesc.size = size;
            bufDesc.mappedAtCreation = false;
        };
        buffer = ctx->device.createBuffer(bufDesc);
        GGML_ASSERT(buffer && "cannot create storage_buffer");
    }
    
    void init_tensor(const ggml_tensor * tensor) {
        LOGT("%s: %s, init to offset %ld\n", label.c_str(), tensor->name, next_free_ptr);
        next_free_ptr += wgpu_tensor_get_nbytes(tensor);
    }

    void write_tensor(const ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        size_t offs_in_buf = (size_t)tensor->data - BUF_BASE;
        ctx->queue.writeBuffer(buffer, offs_in_buf, data, size);
    }

    void read_tensor(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        size_t offs_in_buf = (size_t)tensor->data - BUF_BASE;
        LOGD("%s: %s, read from offset %ld\n", label.c_str(), tensor->name, offs_in_buf);
        wgpu::BufferDescriptor descBuf = wgpu::Default;
        {
            descBuf.label = "map_read_buffer";
            descBuf.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
            descBuf.size = GGML_PAD(size, BUF_ALIGN);
            descBuf.mappedAtCreation = false;
        };
        auto tmpbuf = ctx->device.createBuffer(descBuf);

        // command encoder
        wgpu::CommandEncoderDescriptor ceDesc = wgpu::Default;
        ceDesc.label = "ggml_command_encoder_get_tensor";
        auto commandEncoder = ctx->device.createCommandEncoder(ceDesc);
        commandEncoder.copyBufferToBuffer(buffer, offs_in_buf, tmpbuf, 0, size);

        // run cmd
        auto cmdBuffer = commandEncoder.finish();
        ctx->queue.submit(1, &cmdBuffer);
        bool ready = false;
        auto ret = tmpbuf.mapAsync(wgpu::MapMode::Read, 0, size, [&ready](WGPUBufferMapAsyncStatus status) {
            printf("buffer_map status=%#.8x\n", status);
            if (status == WGPUBufferMapAsyncStatus_Success) {
                ready = true;
            }
        });
        while (!ready) {
            wgpuPollEvents(ctx->device, true);
        }

        // get output buf
        const void * buf = tmpbuf.getConstMappedRange(0, size);
        GGML_ASSERT(buf);
        memcpy(data, buf, size);
        tmpbuf.unmap();
    }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend buffer interface

static void ggml_backend_wgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    LOGT("%s\n", __func__);
    ggml_wgpu_buffer_context * buf_ctx = (ggml_wgpu_buffer_context *)buffer->context;
    delete buf_ctx;
}

static void * ggml_backend_wgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    //return (void *)buffer->context;
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)BUF_BASE;
}


static void ggml_backend_wgpu_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor * tensor) {
    ggml_wgpu_buffer_context * buf_ctx = (ggml_wgpu_buffer_context *)buffer->context;
    LOGT("%s: %s buf=%s\n", __func__, tensor->name, buf_ctx->label.c_str());
    buf_ctx->init_tensor(tensor);
}

static void ggml_backend_wgpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor * tensor,
                                                const void * data, size_t offset,
                                                size_t size) {
    ggml_wgpu_buffer_context * buf_ctx = (ggml_wgpu_buffer_context *)buffer->context;
    LOGT("%s: buf=%s tensor=%s off=%ld size=%ld\n", __func__, buf_ctx->label.c_str(), tensor->name, offset, size);
    buf_ctx->write_tensor(tensor, data, offset, size);
}

static void ggml_backend_wgpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) {
    ggml_wgpu_buffer_context * buf_ctx = (ggml_wgpu_buffer_context *)buffer->context;
    LOGT("%s: buf=%s tensor=%s off=%ld size=%ld\n", __func__, buf_ctx->label.c_str(), tensor->name, offset, size);
    buf_ctx->read_tensor(tensor, data, offset, size);
}

static void ggml_backend_wgpu_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) {
    LOGT("%s: %d\n", __func__, value);
}

static void ggml_backend_wgpu_buffer_reset(ggml_backend_buffer_t buffer) {
    LOGT("%s\n", __func__);
}

static struct ggml_backend_buffer_i ggml_backend_wgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_wgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_wgpu_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_wgpu_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_wgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_wgpu_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_wgpu_buffer_clear,
    /* .reset           = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend buffer type interface

static const char * ggml_backend_wgpu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "wgpu_buffer_type";
    UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_wgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_wgpu_buffer_type_context * buft_ctx = (ggml_wgpu_buffer_type_context *)buft->context;
    size_t padded_size = GGML_PAD(size, BUF_ALIGN);
    ggml_wgpu_buffer_context * ctx = new ggml_wgpu_buffer_context(buft_ctx, padded_size);
    LOGT("%s: size=%ld buf=%s\n", __func__, size, ctx->label.c_str());
    return ggml_backend_buffer_init(
        /* .buft      = */ buft,
        /* .interface = */ ggml_backend_wgpu_buffer_interface,
        /* .context   = */ ctx,
        /* .size      = */ padded_size
    );
}

static size_t ggml_backend_wgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return BUF_ALIGN;
}

static size_t ggml_backend_wgpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    size_t size = GGML_PAD(ggml_nbytes(tensor), BUF_ALIGN);
    // LOGT("%s: %ld\n", __func__, size);
    return size;
}

static bool ggml_backend_wgpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    LOGT("%s\n", __func__);
    UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_i ggml_backend_wgpu_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_wgpu_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_wgpu_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_wgpu_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_wgpu_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_wgpu_buffer_type_is_host,
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// backend interface

static const char * ggml_backend_wgpu_get_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "WebGPU";
}

static void ggml_backend_wgpu_free(ggml_backend_t backend) {
    ggml_wgpu_context * ctx = (ggml_wgpu_context *)backend->context;
    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_wgpu_get_default_buffer_type(ggml_backend_dev_t dev) {
    LOGT("%s\n", __func__);
    ggml_wgpu_context * ctx = (ggml_wgpu_context *)dev->context;
    if (ctx->buft == nullptr) {
        ctx->buft = new ggml_backend_buffer_type{
            /* .iface    = */ ggml_backend_wgpu_buffer_type_interface,
            /* .device   = */ dev,
            /* .context  = */ ctx,
        };
    }
    return ctx->buft;
}

static void ggml_backend_wgpu_synchronize(ggml_backend_t backend) {
    LOGT("%s\n", __func__);
    UNUSED(backend);
}

static ggml_wgpu_buffer_context * wgpu_tensor_get_buf_ctx(const ggml_tensor * t) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return (ggml_wgpu_buffer_context *)buf->context;
}

static const char * wgpu_tensor_get_buf_label(const ggml_tensor * t) {
    ggml_wgpu_buffer_context * buf_ctx = wgpu_tensor_get_buf_ctx(t);
    return buf_ctx->label.c_str();
}

static wgpu::Buffer wgpu_tensor_get_buffer(const ggml_tensor * t) {
    ggml_wgpu_buffer_context * buf_ctx = wgpu_tensor_get_buf_ctx(t);
    return buf_ctx->buffer;
}

static size_t wgpu_tensor_get_offset(const ggml_tensor * t) {
    return (size_t)t->data - BUF_BASE;
}

bool ggml_wgpu_compute_forward(ggml_wgpu_context * ctx, struct ggml_tensor * tensor) {
    LOGT("%s: %s op=%s\n", __func__, tensor->name, ggml_op_name(tensor->op));

    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    const ggml_tensor * dest = tensor;

    // inplace == same buf && same offset
    bool inplace = src0->buffer == dest->buffer && src0->data == dest->data;

    wgpu::ComputePipeline compPipeline = inplace
        ? ctx->pipeline_inpl_op[tensor->op]
        : ctx->pipeline_op[tensor->op];

    // ctx->tensor_params_host
    ctx->queue.writeBuffer(ctx->buf_tensor_params, 0, &ctx->tensor_params_host, sizeof(ggml_wgpu_tensor_params));

    // set bind group entry
    wgpu::BindGroupEntry bgEntries[4];
    {
        bgEntries[0].binding = 0;
        bgEntries[0].buffer  = wgpu_tensor_get_buffer(src0);
        bgEntries[0].offset  = wgpu_tensor_get_offset(src0);
        bgEntries[0].size    = wgpu_tensor_get_nbytes(src0);

        bgEntries[1].binding = 1;
        bgEntries[1].buffer  = wgpu_tensor_get_buffer(src1);
        bgEntries[1].offset  = wgpu_tensor_get_offset(src1);
        bgEntries[1].size    = wgpu_tensor_get_nbytes(src1);

        bgEntries[2].binding = 2;
        bgEntries[2].buffer  = !inplace ? wgpu_tensor_get_buffer(dest) : ctx->buf_dummy;
        bgEntries[2].offset  = !inplace ? wgpu_tensor_get_offset(dest) : 0;
        bgEntries[2].size    = !inplace ? wgpu_tensor_get_nbytes(dest) : BUF_ALIGN;

        bgEntries[3].binding = 3;
        bgEntries[3].buffer  = ctx->buf_tensor_params;
        bgEntries[3].offset  = 0;
        bgEntries[3].size    = sizeof(ggml_wgpu_tensor_params);

        LOGT("%s: src0=%s buf=%s off=%llu\n", __func__, src0->name, wgpu_tensor_get_buf_label(src0), bgEntries[0].offset);
        LOGT("%s: src1=%s buf=%s off=%llu\n", __func__, src1->name, wgpu_tensor_get_buf_label(src1), bgEntries[1].offset);
        LOGT("%s: dest=%s buf=%s off=%llu\n", __func__, dest->name, wgpu_tensor_get_buf_label(dest), bgEntries[2].offset);
    }
    wgpu::BindGroupDescriptor bgDesc = wgpu::Default;
    {
        bgDesc.label = "bind_group";
        bgDesc.layout = ctx->bind_group_layout;
        bgDesc.entryCount = 4;
        bgDesc.entries = bgEntries;
    };
    auto bindGroup = ctx->device.createBindGroup(bgDesc);
    GGML_ASSERT(bindGroup);

    // compute
    wgpu::CommandEncoderDescriptor ceDesc = wgpu::Default;
    ceDesc.label = "ggml_command_encoder";
    auto commandEncoder = ctx->device.createCommandEncoder(ceDesc);
    GGML_ASSERT(commandEncoder);
    auto computePassEncoder = commandEncoder.beginComputePass();
    GGML_ASSERT(computePassEncoder);
    computePassEncoder.setPipeline(compPipeline);
    computePassEncoder.setBindGroup(0, bindGroup, 0, NULL);
    computePassEncoder.dispatchWorkgroups(ggml_nelements(tensor), 1, 1); // TODO: find correct shape of workgroup
    computePassEncoder.end();

    auto cmdBuffer = commandEncoder.finish();
    GGML_ASSERT(cmdBuffer);
    ctx->queue.submit(1, &cmdBuffer);

    return true;
}

static ggml_status ggml_backend_wgpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    struct ggml_wgpu_context * ctx = (struct ggml_wgpu_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node)
            || node->op == GGML_OP_RESHAPE
            || node->op == GGML_OP_TRANSPOSE
            || node->op == GGML_OP_VIEW
            || node->op == GGML_OP_PERMUTE
            || node->op == GGML_OP_NONE
        ) {
            continue;
        }
        
        ggml_wgpu_compute_forward(ctx, node);
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_wgpu_supports_op(ggml_backend_dev_t backend, const struct ggml_tensor * op) {
    LOGT("%s\n", __func__);
    UNUSED(backend);
    const ggml_wgpu_shader * shader = ggml_wgpu_get_shader(op->op);
    return shader != nullptr;
}

static bool ggml_backend_wgpu_supports_buft(ggml_backend_dev_t backend, ggml_backend_buffer_type_t buft) {
    LOGT("%s\n", __func__);
    return buft->iface.get_name == ggml_backend_wgpu_buffer_type_name;
}

static bool ggml_backend_wgpu_offload_op(ggml_backend_dev_t backend, const ggml_tensor * op) {
    LOGT("%s\n", __func__);
    return true;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i ggml_backend_wgpu_interface = {
    /* .get_name                = */ ggml_backend_wgpu_get_name,
    /* .free                    = */ ggml_backend_wgpu_free,
    /* .set_tensor_async        = */ NULL, // ggml_backend_wgpu_set_tensor_async,
    /* .get_tensor_async        = */ NULL, // ggml_backend_wgpu_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_wgpu_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_wgpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static void ggml_backend_wgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    LOGT("%s\n", __func__);
    props->name        = "WebGPU";
    props->description = "WebGPU";
    props->type        = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->memory_free = 8192; // dummy values
    props->memory_total = 8192; // dummy values
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

GGML_API ggml_backend_t ggml_backend_wgpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    LOGT("%s\n", __func__);
    GGML_UNUSED(params);
    if (!ggml_wgpu_ctx_instance) {
        ggml_wgpu_ctx_instance = new ggml_wgpu_context;
    }
    static const char * guid_str = "__ggml_webgpu :)";
    ggml_backend_t wgpu_backend = new ggml_backend{
        /* .guid      = */ reinterpret_cast<ggml_guid_t>((void *)guid_str),
        /* .interface = */ ggml_backend_wgpu_interface,
        /* .device    = */ dev,
        /* .context   = */ ggml_wgpu_ctx_instance,
    };
    return wgpu_backend;
}

GGML_API ggml_backend_t ggml_backend_wgpu_init(void) {
    LOGT("%s\n", __func__);
    return ggml_backend_wgpu_device_init_backend(ggml_backend_wgpu_add_device(), nullptr);
}

static const struct ggml_backend_device_i ggml_backend_wgpu_device_i = {
    /* .get_name             = */ [](ggml_backend_dev_t) { return "WebGPU"; },
    /* .get_description      = */ [](ggml_backend_dev_t) { return "WebGPU"; },
    /* .get_memory           = */ [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = 8192; *total = 8192; },
    /* .get_type             = */ [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; },
    /* .get_props            = */ ggml_backend_wgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_wgpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_wgpu_get_default_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_wgpu_supports_op,
    /* .supports_buft        = */ ggml_backend_wgpu_supports_buft,
    /* .offload_op           = */ ggml_backend_wgpu_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const struct ggml_backend_reg_i ggml_backend_wgpu_reg_i = {
    /* .get_name         = */ [](ggml_backend_reg_t) { return "WebGPU"; },
    /* .get_device_count = */ [](ggml_backend_reg_t) -> size_t { return 1; },
    /* .get_device       = */ [](ggml_backend_reg_t, size_t) -> ggml_backend_dev_t { GGML_ABORT("unused"); },
    /* .get_proc_address = */ [](ggml_backend_reg_t, const char *) -> void * { return nullptr; },
};

ggml_backend_reg_t ggml_backend_wgpu_reg(void) {
    static struct ggml_backend_reg ggml_backend_wgpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_wgpu_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_wgpu_reg;
}

ggml_backend_dev_t ggml_backend_wgpu_add_device(void) {
    if (!ggml_wgpu_ctx_instance) {
        ggml_wgpu_ctx_instance = new ggml_wgpu_context;
    }
    ggml_backend_dev_t dev = new ggml_backend_device {
        /* .iface   = */ ggml_backend_wgpu_device_i,
        /* .reg     = */ ggml_backend_wgpu_reg(),
        /* .context = */ ggml_wgpu_ctx_instance,
    };
    return dev;
}
