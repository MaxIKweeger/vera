// GPU Kron aperture flux — one workgroup per detected source.
//
// Each workgroup scans its expanded bounding box and sums flux within
// the Kron ellipse: (u/a)^2 + (v/b)^2 <= aperture^2, where u,v are
// rotated-frame coordinates around the source centroid.

struct KronParams {
    x_c       : f32,  // centroid column
    y_c       : f32,  // centroid row
    cos_theta : f32,
    sin_theta : f32,
    inv_a     : f32,  // 1 / semi-major axis
    inv_b     : f32,  // 1 / semi-minor axis
    ap_sq     : f32,  // (kron_factor * kron_radius)^2
    r0 : u32, c0 : u32, r1 : u32, c1 : u32,  // expanded bbox
}

struct Globals { ncols : u32, n_dets : u32 }

@group(0) @binding(0) var<storage, read>       residual    : array<f32>;
@group(0) @binding(1) var<storage, read>       kron_params : array<KronParams>;
@group(0) @binding(2) var<storage, read_write> kron_out    : array<f32>;
@group(0) @binding(3) var<uniform>             globals     : Globals;

var<workgroup> sh_flux : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main_kron(
    @builtin(workgroup_id)          wg  : vec3<u32>,
    @builtin(local_invocation_index) lid : u32,
) {
    let det = wg.x;
    if det >= globals.n_dets { return; }

    let kp     = kron_params[det];
    let bbox_w = kp.c1 - kp.c0;
    let bbox_h = kp.r1 - kp.r0;
    let n_pix  = bbox_w * bbox_h;

    var local_flux : f32 = 0.0;
    var i : u32 = lid;
    loop {
        if i >= n_pix { break; }
        let r  = kp.r0 + i / bbox_w;
        let c  = kp.c0 + i % bbox_w;
        let dx =  f32(c) - kp.x_c;
        let dy =  f32(r) - kp.y_c;
        let u  =  dx * kp.cos_theta + dy * kp.sin_theta;
        let v  = -dx * kp.sin_theta + dy * kp.cos_theta;
        if u * u * kp.inv_a * kp.inv_a + v * v * kp.inv_b * kp.inv_b <= kp.ap_sq {
            local_flux += residual[r * globals.ncols + c];
        }
        i += 256u;
    }

    // Parallel reduction within workgroup.
    sh_flux[lid] = local_flux;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride { sh_flux[lid] += sh_flux[lid + stride]; }
        workgroupBarrier();
    }
    if lid == 0u { kron_out[det] = sh_flux[0]; }
}
