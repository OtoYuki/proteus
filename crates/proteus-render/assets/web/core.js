// Proteus browser viewer — pure functions: mesh decoding, colours, camera maths.
// No DOM, no WebGL, so `node --test` can check them against fixtures written by the Rust side
// (crates/proteus-render/assets/web/test). Colours reproduce rasterizer/shader.rs bit for bit:
// every arithmetic step is rounded to f32 with Math.fround, as Rust's f32 maths is, and the final
// `as u8` is a truncation clamped to 0..=255.
(function (root) {
  'use strict';

  const f = Math.fround;

  /** Rust's `x as u8` for an f32: truncate toward zero, saturate, NaN → 0. */
  function asU8(x) {
    if (!(x > 0)) return 0;
    if (x >= 255) return 255;
    return Math.trunc(x);
  }

  function clamp01(t) {
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  /** ColorRGB::lerp */
  function lerp(a, b, t) {
    t = f(clamp01(f(t)));
    return [0, 1, 2].map((k) => asU8(f(f(a[k]) + f(f(f(b[k]) - f(a[k])) * t))));
  }

  /** shader.rs plddt_to_color */
  function plddtColor(p) {
    p = f(p);
    if (p >= 90) return [0, 83, 214];
    if (p >= 70) return lerp([101, 203, 243], [0, 83, 214], f(f(p - 70) / 20));
    if (p >= 50) return lerp([255, 219, 19], [101, 203, 243], f(f(p - 50) / 20));
    return lerp([255, 125, 69], [255, 219, 19], clamp01(f(p / 50)));
  }

  /** shader.rs secondary_structure_to_color; codes from web.rs ss_code. */
  function ssColor(code) {
    return code === 0 ? [217, 70, 239] : code === 1 ? [245, 158, 11] : [6, 182, 212];
  }

  function hsvToRgb(h, s, v) {
    h = f(h);
    const c = f(f(v) * f(s));
    const x = f(c * f(1 - Math.abs(f(f(f(h / 60) % 2) - 1))));
    const m = f(f(v) - c);
    let r1, g1, b1;
    if (h < 60) [r1, g1, b1] = [c, x, 0];
    else if (h < 120) [r1, g1, b1] = [x, c, 0];
    else if (h < 180) [r1, g1, b1] = [0, c, x];
    else if (h < 240) [r1, g1, b1] = [0, x, c];
    else if (h < 300) [r1, g1, b1] = [x, 0, c];
    else [r1, g1, b1] = [c, 0, x];
    return [r1, g1, b1].map((u) => asU8(f(f(u + m) * 255)));
  }

  /** shader.rs rainbow_color */
  function rainbowColor(i, n) {
    if (n === 0) return [255, 255, 255];
    const t = clamp01(f(f(i) / f(n)));
    return hsvToRgb(f(f(1 - t) * 240), 0.85, 0.95);
  }

  function b64ToBytes(b64) {
    const bin = atob(b64.trim());
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /** Inflate gzip with the browser's (or Node's) own DecompressionStream. */
  async function gunzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  const MAGIC = 'PRMESH1\0';

  /** Decode the blob written by web.rs `encode_meshes`. Typed arrays view the buffer. */
  function parseMesh(bytes) {
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const dv = new DataView(buf);
    let magic = '';
    for (let i = 0; i < 8; i++) magic += String.fromCharCode(dv.getUint8(i));
    if (magic !== MAGIC) throw new Error('unknown mesh format ' + JSON.stringify(magic));
    const nv = dv.getUint32(8, true), nt = dv.getUint32(12, true);
    const dsNv = dv.getUint32(16, true), dsNt = dv.getUint32(20, true);
    const lo = [0, 1, 2].map((k) => dv.getFloat32(24 + 4 * k, true));
    const span = [0, 1, 2].map((k) => dv.getFloat32(36 + 4 * k, true));
    let o = 48;
    const up4 = (x) => Math.ceil(x / 4) * 4;
    function mesh(n, ntri, full) {
      const m = { n, nt: ntri };
      m.pos = new Uint16Array(buf, o, n * 3); o = up4(o + n * 6);
      m.nrm = new Int8Array(buf, o, n * 3); o = up4(o + n * 3);
      if (full) {
        m.res = new Uint32Array(buf, o, n); o += n * 4;
        m.plddt = new Float32Array(buf, o, n); o += n * 4;
        m.ss = new Uint8Array(buf, o, n); o = up4(o + n);
      }
      m.idx = new Uint32Array(buf, o, ntri * 3); o += ntri * 12;
      return m;
    }
    const ribbon = mesh(nv, nt, true);
    const disulfides = mesh(dsNv, dsNt, false);
    if (o !== buf.byteLength) throw new Error('mesh blob has ' + (buf.byteLength - o) + ' trailing bytes');
    return { lo, span, ribbon, disulfides };
  }

  /** Per-vertex RGB (u8) for a scheme, exactly as pipeline.rs picks it. */
  function vertexColors(ribbon, scheme) {
    const n = ribbon.n, out = new Uint8Array(n * 3);
    let total = 1;
    if (scheme === 'rainbow') {
      let max = 0;
      for (let i = 0; i < n; i++) if (ribbon.res[i] > max) max = ribbon.res[i];
      total = max + 1;
    }
    for (let i = 0; i < n; i++) {
      const c = scheme === 'plddt' ? plddtColor(ribbon.plddt[i])
        : scheme === 'rainbow' ? rainbowColor(ribbon.res[i], total)
        : ssColor(ribbon.ss[i]);
      out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2];
    }
    return out;
  }

  /**
   * Pixels per Å for an orthographic view, camera.rs `project`: fit the oriented half-extents
   * to 90 % of the viewport, never below the bounding-sphere fit, times zoom.
   */
  function fitScale(width, height, radius, halfExtents, zoom) {
    const sphere = Math.min(width, height) * 0.9 / (radius * 2);
    let s = sphere;
    if (halfExtents && halfExtents[0] > 0 && halfExtents[1] > 0) {
      s = Math.max(Math.min(width * 0.9 / (2 * halfExtents[0]), height * 0.9 / (2 * halfExtents[1])), sphere);
    }
    return s * zoom;
  }

  /** Row-major 3×3 product. */
  function mat3Mul(a, b) {
    const r = new Array(9).fill(0);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++) r[3 * i + j] += a[3 * i + k] * b[3 * k + j];
    return r;
  }

  /** Rotation about the view-space Y (yaw) then X (pitch) axes, row-major, as camera.rs. */
  function yawPitch(yaw, pitch) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const ry = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
    const rp = [1, 0, 0, 0, cp, -sp, 0, sp, cp];
    return mat3Mul(ry, rp);
  }

  const api = { asU8, lerp, plddtColor, ssColor, rainbowColor, b64ToBytes, gunzip, parseMesh,
    vertexColors, fitScale, mat3Mul, yawPitch, MAGIC };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ProteusCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
