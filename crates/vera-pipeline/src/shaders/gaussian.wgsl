struct Params {
    nrows     : u32,
    ncols     : u32,
    radius    : u32,
    kernel_len: u32,
}

@group(0) @binding(0) var<storage, read>       src    : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst    : array<f32>;
@group(0) @binding(2) var<storage, read>       kern   : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

// Horizontal pass: for each (row, col) sum over kernel along col axis.
@compute @workgroup_size(16, 16, 1)
fn main_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.ncols || row >= params.nrows { return; }

    let r  = i32(params.radius);
    let nc = i32(params.ncols);
    var acc: f32 = 0.0;

    for (var k: u32 = 0u; k < params.kernel_len; k++) {
        let ic = clamp(i32(col) + i32(k) - r, 0, nc - 1);
        acc += src[row * params.ncols + u32(ic)] * kern[k];
    }

    dst[row * params.ncols + col] = acc;
}

// Vertical pass: for each (row, col) sum over kernel along row axis.
@compute @workgroup_size(16, 16, 1)
fn main_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.ncols || row >= params.nrows { return; }

    let r  = i32(params.radius);
    let nr = i32(params.nrows);
    var acc: f32 = 0.0;

    for (var k: u32 = 0u; k < params.kernel_len; k++) {
        let ir = clamp(i32(row) + i32(k) - r, 0, nr - 1);
        acc += src[u32(ir) * params.ncols + col] * kern[k];
    }

    dst[row * params.ncols + col] = acc;
}
