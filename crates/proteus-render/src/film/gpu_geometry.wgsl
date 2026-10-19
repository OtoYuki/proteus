// Film frames on the GPU: the CPU rasteriser's pipeline (`rasterizer::pipeline`), pass for
// pass. Geometry is drawn into a supersampled colour + linear-depth target with the same
// orthographic camera and per-vertex (Gouraud) shading; `downsample` box-filters it onto the
// ground colour; `post` applies the same outline and SSAO kernel at output resolution.

struct Frame {
    rot: mat4x4<f32>,
    center: vec4<f32>,
    pan: vec4<f32>,
    // x: pixels per Å, y: target width, z: target height, w: unused
    scale: vec4<f32>,
    // x: depth that maps to 0 (near), y: depth that maps to 1 (far),
    // z: fog range low, w: fog range high
    depths: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> residue_colors: array<u32>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) residue: u32,
    // alpha 0: take the colour from residue_colors
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) depth: f32,
};

fn unpack(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 0xffu), f32((c >> 8u) & 0xffu), f32((c >> 16u) & 0xffu)) / 255.0;
}

// rasterizer::shader::shade_blinn_phong
fn shade(base: vec3<f32>, normal_in: vec3<f32>) -> vec3<f32> {
    let key_light = normalize(vec3<f32>(0.5, 0.8, 1.0));
    let fill_light = normalize(vec3<f32>(-0.6, -0.4, 0.5));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (dot(normal_in, normal_in) >= 1e-6) {
        n = normalize(normal_in);
    }
    if (n.z < 0.0) {
        n = -n;
    }
    let wrap = max((dot(n, key_light) + 0.3) / 1.3, 0.0);
    let intensity = 0.34 + 0.12 * (n.y * 0.5 + 0.5) + wrap * 0.62
        + max(dot(n, fill_light), 0.0) * 0.16;
    let half_vec = normalize(key_light + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 28.0) * 0.22;
    let rim = pow(1.0 - n.z, 3.0) * 0.30;
    let highlight = vec3<f32>(1.0, 0.98, 0.88);
    let v = min(base * intensity, vec3<f32>(1.0)) + highlight * spec + base * rim;
    return min(v, vec3<f32>(1.0));
}

@vertex
fn vs_geometry(v: VsIn) -> VsOut {
    let rot = mat3x3<f32>(frame.rot[0].xyz, frame.rot[1].xyz, frame.rot[2].xyz);
    let view = rot * (v.position - frame.center.xyz) + frame.pan.xyz;
    let s = frame.scale.x;
    let w = frame.scale.y;
    let h = frame.scale.z;
    // camera.project: screen x = w/2 + view.x·s, y = h/2 − view.y·s; depth = −view.z.
    let ndc_x = view.x * s * 2.0 / w;
    let ndc_y = view.y * s * 2.0 / h;
    let depth = -view.z;
    let z = (depth - frame.depths.x) / (frame.depths.y - frame.depths.x);
    var base = v.color.rgb;
    if (v.color.a < 0.5) {
        base = unpack(residue_colors[v.residue]);
    }
    let fog_range = max(frame.depths.w - frame.depths.z, 1e-3);
    let fraction = clamp((depth - frame.depths.z) / fog_range, 0.0, 1.0);
    let fog = 1.0 - 0.30 * fraction;
    var out: VsOut;
    out.clip = vec4<f32>(ndc_x, ndc_y, z, 1.0);
    out.color = shade(base, rot * v.normal) * fog;
    out.depth = depth;
    return out;
}

struct GeomOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

@fragment
fn fs_geometry(v: VsOut) -> GeomOut {
    var out: GeomOut;
    out.color = vec4<f32>(v.color, 1.0);
    out.depth = v.depth;
    return out;
}
