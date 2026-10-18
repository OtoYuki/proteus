// Full-screen passes of the GPU film renderer: downsample onto the ground, then outlines and
// SSAO, as the CPU path does (see gpu_geometry.wgsl).
struct Post {
    // x: supersampling factor, y: output width, z: output height
    params: vec4<u32>,
    ground: vec4<f32>,
};

const EMPTY: f32 = 1e30;

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var src_color: texture_2d<f32>;
@group(0) @binding(2) var src_depth: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct Pixel {
    @location(0) color: vec4<f32>,
    @location(1) depth: f32,
};

// Framebuffer::downsample_on_ground (film): the mean of k×k samples with empty ones taken as
// the ground; a pixel keeps a depth only when at least half its samples hit.
@fragment
fn fs_downsample(@builtin(position) p: vec4<f32>) -> Pixel {
    let k = post.params.x;
    let base = vec2<u32>(u32(p.x), u32(p.y)) * k;
    var sum = vec3<f32>(0.0);
    var depth = EMPTY;
    var hits = 0u;
    for (var sy = 0u; sy < k; sy++) {
        for (var sx = 0u; sx < k; sx++) {
            let at = vec2<i32>(base + vec2<u32>(sx, sy));
            let d = textureLoad(src_depth, at, 0).r;
            if (d < EMPTY * 0.5) {
                hits++;
                depth = min(depth, d);
                sum += textureLoad(src_color, at, 0).rgb;
            } else {
                sum += post.ground.rgb;
            }
        }
    }
    var out: Pixel;
    out.color = vec4<f32>(sum / f32(k * k), 1.0);
    if (hits * 2u >= k * k) {
        out.depth = depth;
    } else {
        out.depth = EMPTY;
    }
    return out;
}

fn depth_at(x: i32, y: i32) -> f32 {
    return textureLoad(src_depth, vec2<i32>(x, y), 0).r;
}

// Rasterizer::apply_post_processing: a pixel next to empty space or a depth step over 4 Å is
// an outline (× 0.4); elsewhere SSAO from 8 samples, darkening at most to 0.62.
@fragment
fn fs_post(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let x = i32(p.x);
    let y = i32(p.y);
    let w = i32(post.params.y);
    let h = i32(post.params.z);
    let c = textureLoad(src_color, vec2<i32>(x, y), 0);
    let d = depth_at(x, y);
    if (d >= EMPTY * 0.5) {
        return c;
    }
    var edge = false;
    let edges = array<vec2<i32>, 4>(vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1));
    for (var i = 0; i < 4; i++) {
        let q = vec2<i32>(x, y) + edges[i];
        if (q.x < 0 || q.y < 0 || q.x >= w || q.y >= h) {
            edge = true;
            break;
        }
        let nd = depth_at(q.x, q.y);
        if (nd >= EMPTY * 0.5 || abs(nd - d) > 4.0) {
            edge = true;
            break;
        }
    }
    if (edge) {
        return vec4<f32>(c.rgb * 0.4, 1.0);
    }
    let offsets = array<vec2<i32>, 8>(
        vec2(-2, 0), vec2(2, 0), vec2(0, -2), vec2(0, 2),
        vec2(-3, -3), vec2(3, 3), vec2(-3, 3), vec2(3, -3)
    );
    var occlusion = 0.0;
    var valid = 0;
    for (var i = 0; i < 8; i++) {
        let q = vec2<i32>(x, y) + offsets[i];
        if (q.x >= 0 && q.y >= 0 && q.x < w && q.y < h) {
            let nd = depth_at(q.x, q.y);
            if (nd < EMPTY * 0.5) {
                valid++;
                let diff = d - nd;
                if (diff > 0.05 && diff < 8.0) {
                    occlusion += min(diff / 4.0, 1.0);
                }
            }
        }
    }
    if (valid > 0) {
        let ao = clamp(1.0 - (occlusion / f32(valid)) * 0.35, 0.62, 1.0);
        return vec4<f32>(c.rgb * ao, 1.0);
    }
    return c;
}
