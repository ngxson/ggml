
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

    off_src0 : u32,
    off_src1 : u32,
    off_dest : u32,
}

@group(0) @binding(0)
var<storage,read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage,read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage,read_write> dest: array<f32>;

@group(0) @binding(3)
var<uniform> tp: TensorParams;


@compute
@workgroup_size(1)
fn kernel_div_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = src0[global_id.x + tensor_params.offs_src0/4u];
    let y = src1[global_id.x + tensor_params.offs_src1/4u];
            dest[global_id.x + tensor_params.offs_dest/4u] = x / y;
}


@compute
@workgroup_size(1)
fn kernel_div_inplace_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = src0[global_id.x + tensor_params.offs_src0/4u];
    let y = src1[global_id.x + tensor_params.offs_src1/4u];
            dest[global_id.x + tensor_params.offs_dest/4u] = x / y;
}


@compute
@workgroup_size(1)
fn kernel_add_f32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = src0[global_id.x + tensor_params.offs_src0/4u];
    let y = src1[global_id.x + tensor_params.offs_src1/4u];
            dest[global_id.x + tensor_params.offs_dest/4u] = x / y;
}


