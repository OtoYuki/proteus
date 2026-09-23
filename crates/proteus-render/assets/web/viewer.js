// Proteus browser viewer — WebGL2 renderer, interaction and panels.
// Geometry comes from proteus-render (the same ribbon the terminal draws); this file only draws
// it. Shading ports rasterizer/shader.rs `shade_blinn_phong` and pipeline.rs depth cueing,
// outlines and SSAO. Pure helpers live in core.js (ProteusCore).
(function () {
  'use strict';
  const C = window.ProteusCore;
  const $ = (id) => document.getElementById(id);
  const meta = JSON.parse($('proteus-meta').textContent);
  // The brand roles, read from the CSS variables the page declares (brand::css_vars).
  const css = getComputedStyle(document.documentElement);
  const role = (name) => css.getPropertyValue('--' + name).trim();

  // ---------------------------------------------------------------- panels (no WebGL needed)
  // The brand already names the product: the heading is the structure, and the line under it
  // says what kind of file it is.
  $('title').textContent = meta.caption || meta.title;
  $('caption').textContent = meta.residues + ' residues · ' +
    (meta.predicted ? 'predicted model' : 'experimental structure');
  // The tab and a saved PNG are named after the structure, not the generic page title.
  const subject = meta.caption.split(' · ')[0] || meta.title;
  document.title = subject + ' — Proteus';
  buildPanel();

  const SCHEMES = ['ss', 'plddt', 'rainbow'];
  let scheme = SCHEMES.includes(meta.scheme) ? meta.scheme : 'ss';
  showLegend();

  function fail(msg) {
    const fb = $('fallback');
    fb.hidden = false;
    fb.textContent = msg;
    $('view').style.display = 'none';
  }

  const canvas = $('view');
  const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) {
    fail('This browser has no WebGL2, so the 3D view is unavailable. The measurements do not need it.');
    return;
  }
  if (typeof DecompressionStream === 'undefined') {
    fail('This browser has no DecompressionStream (Chrome 80+, Firefox 113+, Safari 16.4+), so the 3D view is unavailable.');
    return;
  }

  C.gunzip(C.b64ToBytes($('proteus-mesh').textContent))
    .then((bytes) => start(C.parseMesh(bytes)))
    .catch((e) => fail('The 3D view could not be built: ' + e.message));

  // ---------------------------------------------------------------- shaders
  const GEOM_VS = `#version 300 es
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec3 aNrm;
  layout(location=2) in vec3 aCol;
  // The residue index as a float, exact to 2^24 (see PICK_FS for why not an integer).
  layout(location=3) in float aRes;
  uniform vec3 uLo, uSpan, uCenter;
  uniform mat3 uRot;
  uniform vec2 uScale, uPan, uOffset;
  uniform float uZ;
  out vec3 vNormal;
  out vec3 vColor;
  out float vDepth;
  flat out float vRes;
  void main() {
    vec3 world = uLo + aPos * uSpan;
    vec3 view = uRot * (world - uCenter);
    view.xy += uPan;
    vNormal = uRot * aNrm;
    vColor = aCol;
    vDepth = -view.z;          // camera.rs: smaller is closer, viewer at +Z
    vRes = aRes;
    gl_Position = vec4(view.xy * uScale + uOffset, clamp(vDepth / uZ, -1.0, 1.0), 1.0);
  }`;

  // shade_blinn_phong, per fragment: key + fill + ambient 0.30 + specular, clamped at 1.3, then
  // pipeline.rs depth cueing (foreground 100 %, background 55 %). A normal that points away
  // from the viewer is flipped toward it, which the CPU pipeline does not do (it leaves such a
  // surface at the 0.30 ambient). The rule uses the normal, not gl_FrontFacing, because the
  // ribbon's triangle winding is not consistent enough to say which side is the front.
  const GEOM_FS = `#version 300 es
  precision highp float;
  in vec3 vNormal;
  in vec3 vColor;
  in float vDepth;
  flat in float vRes;
  uniform float uFogLo, uFogHi;
  out vec4 outColor;
  void main() {
    vec3 n = length(vNormal) < 1e-3 ? vec3(0.0, 0.0, 1.0) : normalize(vNormal);
    if (n.z < 0.0) n = -n;
    vec3 key = normalize(vec3(0.5, 0.8, 1.0));
    vec3 fill = normalize(vec3(-0.6, -0.4, 0.5));
    vec3 h = normalize(key + vec3(0.0, 0.0, 1.0));
    float i = 0.30 + max(dot(n, key), 0.0) * 0.65 + max(dot(n, fill), 0.0) * 0.25
            + pow(max(dot(n, h), 0.0), 16.0) * 0.25;
    i = clamp(i, 0.0, 1.3);
    float frac = clamp((vDepth - uFogLo) / max(uFogHi - uFogLo, 1e-3), 0.0, 1.0);
    outColor = vec4(min(vColor * i, vec3(1.0)) * (1.0 - 0.45 * frac), 1.0);
  }`;

  // The residue id (+1, so 0 is background) in red and green only, computed in float (exact
  // below 2^24). Found in WebKit (Playwright 1.55's build): a UNSIGNED_INT vertex attribute,
  // uint constants of 65536 and above, and the blue and alpha bytes of the RGBA8 target all
  // came back wrong. With more than 65 535 residues a second pass (uHigh) writes the high part.
  const PICK_FS = `#version 300 es
  precision highp float;
  flat in float vRes;
  in vec3 vNormal; in vec3 vColor; in float vDepth;
  uniform bool uHigh;
  out vec4 outColor;
  void main() {
    float id = floor(vRes + 0.5) + 1.0;
    float v = uHigh ? floor(id / 65536.0) : mod(id, 65536.0);
    outColor = vec4(mod(v, 256.0), floor(v / 256.0), 0.0, 255.0) / 255.0;
  }`;

  const POST_VS = `#version 300 es
  out vec2 vUv;
  void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  }`;

  // pipeline.rs apply_post_processing: an outline where a 4-neighbour is background or more
  // than 4 Å away in depth (colour × 0.35); otherwise SSAO from 8 samples on two rings,
  // occlusion min(diff/4, 1) for 0.05 < diff < 8 Å, × 0.45, floor 0.5. Offsets are in pixels and
  // scaled with the canvas, since a browser pixel is much smaller than a terminal one.
  const POST_FS = `#version 300 es
  precision highp float;
  in vec2 vUv;
  uniform sampler2D uColor, uDepth;
  uniform vec2 uTexel;
  uniform float uZ, uStep;
  uniform bool uFx;
  uniform vec3 uBg;
  out vec4 outColor;
  float depthAt(vec2 uv) {
    float d = texture(uDepth, uv).r;
    return d >= 1.0 ? 1e9 : (d * 2.0 - 1.0) * uZ;
  }
  void main() {
    float d = depthAt(vUv);
    if (d > 1e8) { outColor = vec4(uBg, 1.0); return; }
    vec3 c = texture(uColor, vUv).rgb;
    if (!uFx) { outColor = vec4(c, 1.0); return; }
    vec2 e = uTexel * max(1.0, floor(uStep * 0.5));
    vec2 edge[4] = vec2[4](vec2(-1,0), vec2(1,0), vec2(0,-1), vec2(0,1));
    for (int k = 0; k < 4; k++) {
      float n = depthAt(vUv + edge[k] * e);
      if (n > 1e8 || abs(n - d) > 4.0) { outColor = vec4(c * 0.35, 1.0); return; }
    }
    vec2 ring[8] = vec2[8](vec2(-2,0), vec2(2,0), vec2(0,-2), vec2(0,2),
                           vec2(-3,-3), vec2(3,3), vec2(-3,3), vec2(3,-3));
    float occ = 0.0, valid = 0.0;
    for (int k = 0; k < 8; k++) {
      float n = depthAt(vUv + ring[k] * uTexel * uStep);
      if (n < 1e8) {
        valid += 1.0;
        float diff = d - n;
        if (diff > 0.05 && diff < 8.0) occ += min(diff / 4.0, 1.0);
      }
    }
    float ao = valid > 0.0 ? clamp(1.0 - occ / valid * 0.45, 0.5, 1.0) : 1.0;
    outColor = vec4(c * ao, 1.0);
  }`;

  function program(vs, fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { p, u };
  }

  // ---------------------------------------------------------------- scene
  function start(mesh) {
    const geom = program(GEOM_VS, GEOM_FS);
    const pick = program(GEOM_VS, PICK_FS);
    const post = program(POST_VS, POST_FS);

    function vao(m, withAttrs) {
      const v = gl.createVertexArray();
      gl.bindVertexArray(v);
      const buf = (loc, data, size, type, norm) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, type, norm, 0, 0);
        return b;
      };
      buf(0, m.pos, 3, gl.UNSIGNED_SHORT, true);
      buf(1, m.nrm, 3, gl.BYTE, true);
      let colorBuf = null;
      if (withAttrs) {
        colorBuf = buf(2, C.vertexColors(m, scheme), 3, gl.UNSIGNED_BYTE, true);
        const rb = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, rb);
        gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from(m.res), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 1, gl.FLOAT, false, 0, 0);
      }
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { v, colorBuf, count: m.nt * 3 };
    }
    const ribbon = vao(mesh.ribbon, true);
    const ds = mesh.disulfides.n > 0 ? vao(mesh.disulfides, false) : null;

    // Depth-cueing range: the view-space depth extent of the ribbon, like pipeline.rs, measured
    // on a subsample of vertices each frame.
    const sampleStride = Math.max(1, Math.floor(mesh.ribbon.n / 20000));
    const world = [];
    for (let i = 0; i < mesh.ribbon.n; i += sampleStride) {
      world.push([0, 1, 2].map((k) => mesh.lo[k] + mesh.ribbon.pos[3 * i + k] / 65535 * mesh.span[k]));
    }

    const cam = meta.camera;
    const base = cam.rotation.flat();
    const center = cam.center;
    const zRange = cam.radius * 1.5 + 12;
    const state = { yaw: 0, pitch: 0, zoom: 1, pan: [0, 0], fx: true, ds: true, spin: false };
    const reset = () => Object.assign(state, { yaw: 0, pitch: 0, zoom: 1, pan: [0, 0] });

    // Offscreen targets: colour + depth for the post pass, colour + depth for picking.
    let W = 0, H = 0, sceneFb, colorTex, depthTex, pickFb, pickColor, pickDepth;
    function targets() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (w === W && h === H) return;
      W = w; H = h;
      canvas.width = W; canvas.height = H;
      // Free the previous size's targets: a resize drag fires many times.
      for (const t of [colorTex, depthTex, pickColor, pickDepth]) if (t) gl.deleteTexture(t);
      for (const f of [sceneFb, pickFb]) if (f) gl.deleteFramebuffer(f);
      colorTex = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
      depthTex = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT);
      sceneFb = fbo(colorTex, depthTex);
      pickColor = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
      pickDepth = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT);
      pickFb = fbo(pickColor, pickDepth);
    }
    function tex(internal, format, type) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, W, H, 0, format, type, null);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      return t;
    }
    function fbo(color, depth) {
      const f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete');
      return f;
    }

    function rotation() { return C.mat3Mul(C.yawPitch(state.yaw, state.pitch), base); }

    // The panel covers part of the canvas: fit the structure to the rest, in device pixels.
    function visibleArea() {
      const r = $('panel').getBoundingClientRect();
      const k = W / canvas.clientWidth;
      const right = r.left > canvas.clientWidth / 2 ? (canvas.clientWidth - r.left) * k : 0;
      const bottom = right === 0 && r.top > canvas.clientHeight / 2 ? (canvas.clientHeight - r.top) * k : 0;
      return { w: Math.max(1, W - right), h: Math.max(1, H - bottom), right, bottom };
    }
    function pxPerA() {
      const v = visibleArea();
      return C.fitScale(v.w, v.h, cam.radius, cam.halfExtents, state.zoom);
    }

    function setGeomUniforms(prog, rot) {
      const s = pxPerA();
      gl.uniform3fv(prog.u.uLo, mesh.lo);
      gl.uniform3fv(prog.u.uSpan, mesh.span);
      gl.uniform3fv(prog.u.uCenter, center);
      gl.uniformMatrix3fv(prog.u.uRot, true, rot);
      gl.uniform2f(prog.u.uScale, 2 * s / W, 2 * s / H);
      gl.uniform2fv(prog.u.uPan, state.pan);
      const v = visibleArea();
      gl.uniform2f(prog.u.uOffset, -v.right / W, v.bottom / H);
      gl.uniform1f(prog.u.uZ, zRange);
    }

    function drawMeshes(prog, forPick) {
      gl.bindVertexArray(ribbon.v);
      gl.drawElements(gl.TRIANGLES, ribbon.count, gl.UNSIGNED_INT, 0);
      if (ds && state.ds && !forPick) {
        gl.bindVertexArray(ds.v);
        const dsColour = meta.palette.disulfide; // brand::structure::DISULFIDE
        gl.vertexAttrib3f(2, dsColour[0] / 255, dsColour[1] / 255, dsColour[2] / 255);
        gl.vertexAttrib1f(3, 0);
        gl.drawElements(gl.TRIANGLES, ds.count, gl.UNSIGNED_INT, 0);
      }
      gl.bindVertexArray(null);
    }

    function render() {
      targets();
      const rot = rotation();
      let lo = Infinity, hi = -Infinity;
      for (const p of world) {
        const d = -(rot[6] * (p[0] - center[0]) + rot[7] * (p[1] - center[1]) + rot[8] * (p[2] - center[2]));
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFb);
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.useProgram(geom.p);
      setGeomUniforms(geom, rot);
      gl.uniform1f(geom.u.uFogLo, lo);
      gl.uniform1f(geom.u.uFogHi, hi);
      drawMeshes(geom, false);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(post.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, colorTex); gl.uniform1i(post.u.uColor, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, depthTex); gl.uniform1i(post.u.uDepth, 1);
      gl.uniform2f(post.u.uTexel, 1 / W, 1 / H);
      gl.uniform1f(post.u.uZ, zRange);
      gl.uniform1f(post.u.uStep, Math.max(1, Math.min(W, H) / 400));
      gl.uniform1i(post.u.uFx, state.fx ? 1 : 0);
      const bg = meta.palette.ground; // brand::DARK.ground
      gl.uniform3f(post.u.uBg, bg[0] / 255, bg[1] / 255, bg[2] / 255);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function pickAt(cssX, cssY) {
      const x = Math.floor(cssX * W / canvas.clientWidth);
      const y = H - 1 - Math.floor(cssY * H / canvas.clientHeight);
      if (x < 0 || y < 0 || x >= W || y >= H) return -1;
      gl.bindFramebuffer(gl.FRAMEBUFFER, pickFb);
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(pick.p);
      setGeomUniforms(pick, rotation());
      const px = new Uint8Array(4);
      const pass = (high) => {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.uniform1i(pick.u.uHigh, high ? 1 : 0);
        drawMeshes(pick, true);
        gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        return px[0] + px[1] * 256;
      };
      let id = pass(false);
      if (id !== 0 && meta.residues >= 65535) id += pass(true) * 65536;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return id - 1;
    }

    // ---------------------------------------------------------------- frame scheduling
    let queued = false;
    function frame() {
      queued = false;
      if (state.spin) { state.yaw += 0.01; request(); }
      render();
    }
    function request() { if (!queued) { queued = true; requestAnimationFrame(frame); } }
    new ResizeObserver(() => { hideTip(); request(); }).observe(canvas);
    // Fade in once the first frame is drawn (the stylesheet skips it for reduced motion).
    requestAnimationFrame(() => requestAnimationFrame(() => canvas.classList.add('ready')));
    request();

    function recolor() {
      gl.bindBuffer(gl.ARRAY_BUFFER, ribbon.colorBuf);
      gl.bufferData(gl.ARRAY_BUFFER, C.vertexColors(mesh.ribbon, scheme), gl.STATIC_DRAW);
      showLegend();
      request();
    }

    // ---------------------------------------------------------------- input
    const pointers = new Map();
    let pinch = null;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: state.zoom };
      }
      hideTip();
    });
    canvas.addEventListener('pointermove', (e) => {
      const p = pointers.get(e.pointerId);
      if (!p) { hover(e); return; }
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (pointers.size === 2 && pinch) {
        const [a, b] = [...pointers.values()];
        state.zoom = Math.min(50, Math.max(0.1, pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d));
      } else if (p.pan) {
        const s = pxPerA() / (W / canvas.clientWidth);
        state.pan[0] += dx / s; state.pan[1] -= dy / s;
      } else {
        state.yaw += dx * 0.01; state.pitch += dy * 0.01;
      }
      request();
    });
    const up = (e) => { pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null; };
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('pointerleave', hideTip);
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      state.zoom = Math.min(50, Math.max(0.1, state.zoom * Math.exp(-e.deltaY * 0.0015)));
      request();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { reset(); request(); });
    window.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      switch (e.key) {
        case 'c': scheme = SCHEMES[(SCHEMES.indexOf(scheme) + 1) % SCHEMES.length]; recolor(); break;
        case 'o': state.fx = !state.fx; request(); break;
        case 'd': state.ds = !state.ds; request(); break;
        case ' ': state.spin = !state.spin; e.preventDefault(); request(); break;
        case 'r': reset(); request(); break;
        case 's': save(); break;
        // Rotate and zoom without a pointer, with the terminal viewer's keys.
        case 'ArrowLeft': case 'h': state.yaw -= 0.15; request(); break;
        case 'ArrowRight': case 'l': state.yaw += 0.15; request(); break;
        case 'ArrowUp': case 'k': state.pitch -= 0.15; request(); break;
        case 'ArrowDown': case 'j': state.pitch += 0.15; request(); break;
        case '+': case '=': state.zoom = Math.min(50, state.zoom * 1.15); request(); break;
        case '-': case '_': state.zoom = Math.max(0.1, state.zoom * 0.85); request(); break;
        default: return;
      }
    });

    let hoverQueued = null;
    function hover(e) {
      if (hoverQueued) { hoverQueued = e; return; }
      hoverQueued = e;
      requestAnimationFrame(() => {
        const ev = hoverQueued; hoverQueued = null;
        const r = canvas.getBoundingClientRect();
        const i = pickAt(ev.clientX - r.left, ev.clientY - r.top);
        request();
        if (i < 0 || i >= meta.residues) { hideTip(); return; }
        showTip(i, ev.clientX, ev.clientY);
      });
    }

    function save() {
      render();
      canvas.toBlob((b) => {
        const a = document.createElement('a');
        a.download = subject.replace(/[^\w.-]+/g, '_').replace(/^_+|_+$/g, '') + '.png';
        a.href = URL.createObjectURL(b);
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      });
    }
    window.ProteusViewer = { state, render, pickAt, get scheme() { return scheme; } };
  }

  // ---------------------------------------------------------------- panel helpers
  const SS8 = { H: 'α-helix', G: '3₁₀-helix', I: 'π-helix', E: 'strand', B: 'bridge', T: 'turn', S: 'bend', '-': 'coil', ' ': 'coil' };

  function showTip(i, x, y) {
    const L = meta.labels;
    const tip = $('tip');
    const value = meta.perResidue[i];
    const conf = value === undefined ? '' : meta.predicted ? 'pLDDT ' + value.toFixed(1) : 'B-factor ' + value.toFixed(1);
    tip.textContent = '';
    const head = document.createElement('b');
    head.textContent = L.name[i] + ' ' + L.chain[i] + L.number[i] + L.icode[i];
    const rest = document.createElement('span');
    rest.textContent = (SS8[meta.dssp[i]] || 'coil') + ' · ' + (meta.dssp[i] || '-') + (conf ? ' · ' + conf : '');
    tip.append(head, document.createElement('br'), rest);
    tip.hidden = false;
    tip.style.left = Math.min(x + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = Math.min(y + 14, window.innerHeight - tip.offsetHeight - 8) + 'px';
  }
  function hideTip() { $('tip').hidden = true; }

  function swatch(rgb, label) {
    const s = document.createElement('span');
    s.className = 'sw';
    const d = document.createElement('i');
    d.style.background = 'rgb(' + rgb.join(',') + ')';
    s.append(d, document.createTextNode(label));
    return s;
  }

  function showLegend() {
    const el = $('legend');
    el.textContent = '';
    const name = document.createElement('b');
    if (scheme === 'plddt') {
      name.textContent = meta.predicted ? '(plddt · AlphaFold colours)' : '(b-factor on the pLDDT scale)';
      el.append(name, swatch(C.plddtColor(95), '>90'), swatch(C.plddtColor(80), '70–90'),
        swatch(C.plddtColor(60), '50–70'), swatch(C.plddtColor(25), '<50'));
      if (!meta.predicted) {
        const w = document.createElement('span');
        w.className = 'warn';
        w.textContent = '! not a confidence';
        el.append(w);
      }
    } else if (scheme === 'rainbow') {
      name.textContent = '(sequence position)';
      el.append(name, swatch(C.rainbowColor(0, 4), 'N-terminus'), swatch(C.rainbowColor(2, 4), 'middle'),
        swatch(C.rainbowColor(4, 4), 'C-terminus'));
    } else {
      name.textContent = '(secondary structure · dssp)';
      el.append(name, swatch(C.ssColor(0), 'helix'), swatch(C.ssColor(1), 'strand'), swatch(C.ssColor(2), 'coil'));
    }
  }

  function buildPanel() {
    const panel = $('panel');
    const h = (text) => { const e = document.createElement('h2'); e.textContent = '(' + text + ')'; panel.append(e); };
    if (meta.metrics.length) {
      h('measurements');
      const dl = document.createElement('dl');
      for (const [k, v] of meta.metrics) {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd'); dd.textContent = v;
        dl.append(dt, dd);
      }
      panel.append(dl);
    }
    if (meta.rama.length) {
      h('ramachandran φ, ψ');
      panel.append(ramaPlot());
      // Each region has its own shape, so the plot reads without colour.
      const key = document.createElement('div');
      key.className = 'key';
      for (const [shape, word, r] of [['●', 'favoured', 'accent'], ['○', 'allowed', 'warm'], ['▲', 'outlier', 'bad']]) {
        const s = document.createElement('span');
        const g = document.createElement('span');
        g.style.color = role(r);
        g.textContent = shape + ' ';
        s.append(g, document.createTextNode(word));
        key.append(s);
      }
      panel.append(key);
    }
    if (meta.perResidue.length) {
      h(meta.predicted ? 'plddt along the chain' : 'b-factor along the chain');
      panel.append(residueStrip());
      if (!meta.predicted) {
        const w = document.createElement('p');
        w.className = 'note warn';
        w.textContent = '! An experimental B-factor measures motion and disorder, not confidence.';
        panel.append(w);
      }
    }
    const note = document.createElement('p');
    note.className = 'note';
    note.textContent = 'Drawn by Proteus from its own ribbon geometry and DSSP. Nothing is fetched from the network.';
    panel.append(note);
  }

  function ramaPlot() {
    const c = document.createElement('canvas');
    const px = 268, dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = px * dpr; c.height = px * dpr; c.className = 'rama';
    const g = c.getContext('2d');
    g.scale(dpr, dpr);
    g.fillStyle = role('ground'); g.fillRect(0, 0, px, px);
    g.strokeStyle = role('line'); g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, px - 1, px - 1);
    const to = (a) => (a + 180) / 360 * px;
    g.setLineDash([2, 3]);
    for (const a of [-90, 90]) {
      g.beginPath(); g.moveTo(to(a), 0); g.lineTo(to(a), px); g.stroke();
      g.beginPath(); g.moveTo(0, px - to(a)); g.lineTo(px, px - to(a)); g.stroke();
    }
    g.setLineDash([]);
    g.beginPath(); g.moveTo(to(0), 0); g.lineTo(to(0), px); g.moveTo(0, px - to(0)); g.lineTo(px, px - to(0)); g.stroke();
    // Favoured first, outliers last, so an outlier is never hidden under a favoured point.
    const colour = [role('accent'), role('warm'), role('bad')];
    const pts = [...meta.rama].sort((a, b) => a[2] - b[2]);
    for (const [phi, psi, r] of pts) {
      const x = to(phi), y = px - to(psi);
      g.fillStyle = g.strokeStyle = colour[r] || colour[2];
      g.beginPath();
      if (r === 2) { g.moveTo(x, y - 3.4); g.lineTo(x + 3, y + 2); g.lineTo(x - 3, y + 2); g.closePath(); g.fill(); }
      else if (r === 1) { g.lineWidth = 1.2; g.arc(x, y, 2.4, 0, 2 * Math.PI); g.stroke(); }
      else { g.arc(x, y, 1.9, 0, 2 * Math.PI); g.fill(); }
    }
    g.fillStyle = role('dim'); g.font = "10px 'Geist Mono', ui-monospace, monospace";
    g.fillText('φ →', px - 28, px - 6); g.fillText('ψ ↑', 6, 14);
    return c;
  }

  function residueStrip() {
    const c = document.createElement('canvas');
    const n = meta.perResidue.length, w = 268, h = 16, dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = w * dpr; c.height = h * dpr; c.className = 'strip';
    const g = c.getContext('2d');
    g.scale(dpr, dpr);
    let lo = Infinity, hi = -Infinity;
    for (const v of meta.perResidue) { if (v < lo) lo = v; if (v > hi) hi = v; }
    // B-factors: a neutral ramp over the structure's own range, from the line colour to muted.
    const hex = (s) => [1, 3, 5].map((k) => parseInt(s.slice(k, k + 2), 16));
    const [a, b] = [hex(role('line')), hex(role('muted'))];
    for (let i = 0; i < n; i++) {
      const v = meta.perResidue[i];
      const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      const rgb = meta.predicted ? C.plddtColor(v) : a.map((x, k) => Math.round(x + (b[k] - x) * t));
      g.fillStyle = 'rgb(' + rgb.join(',') + ')';
      g.fillRect(i / n * w, 0, Math.max(w / n, 1), h);
    }
    return c;
  }
})();
