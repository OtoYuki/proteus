// Proteus browser viewer — WebGL2 renderer, interaction and panels.
// Geometry comes from proteus-render (the same ribbon the terminal draws); this file only draws
// it. Shading ports rasterizer/shader.rs `shade_blinn_phong` and pipeline.rs depth cueing,
// outlines and SSAO. Pure helpers live in core.js (ProteusCore).
//
// Everything the panel reports can be pointed at: a selection (click the structure, the
// sequence, a Ramachandran point, the pLDDT strip, the PAE map or a finding) dims the rest of
// the ribbon, draws the selected residues and their 5 Å neighbourhood as sticks, and says what
// lies between them.
(function () {
  'use strict';
  const C = window.ProteusCore;
  const $ = (id) => document.getElementById(id);
  const meta = JSON.parse($('proteus-meta').textContent);
  // The brand roles, read from the CSS variables the page declares (brand::css_vars).
  const css = getComputedStyle(document.documentElement);
  const role = (name) => css.getPropertyValue('--' + name).trim();
  const rgbOf = (name) => { const s = role(name); return [1, 3, 5].map((k) => parseInt(s.slice(k, k + 2), 16)); };

  const N = meta.residues;
  const SS8 = { H: 'α-helix', G: '3₁₀-helix', I: 'π-helix', E: 'strand', B: 'bridge', T: 'turn', S: 'bend', '-': 'coil', ' ': 'coil' };
  const ligands = meta.ligands || [];
  const issues = meta.issues || { rama: [], clashes: [], hbonds: [], saltBridges: [], piStacks: [], cationPi: [] };
  // Contact kinds: the list key in meta.issues, a name, and the brand role its lines are drawn in.
  const KINDS = [
    ['clashes', 'heavy-atom overlaps', 'bad', 'overlap'],
    ['hbonds', 'hydrogen bonds', 'sea', ''],
    ['saltBridges', 'salt bridges', 'warm', ''],
    ['piStacks', 'π–π stacking', 'accent', ''],
    ['cationPi', 'cation–π', 'accent', ''],
  ];

  // ---------------------------------------------------------------- panels (no WebGL needed)
  // The brand already names the product: the heading is the structure, and the line under it
  // says what kind of file it is.
  $('title').textContent = meta.caption || meta.title;
  $('caption').textContent = N + ' residues' + (ligands.length ? ' · ' + ligands.length + ' ligand' + (ligands.length > 1 ? 's' : '') : '') +
    ' · ' + (meta.predicted ? 'predicted model' : 'experimental structure');
  // The tab and a saved PNG are named after the structure, not the generic page title.
  const subject = meta.caption.split(' · ')[0] || meta.title;
  document.title = subject + ' — Proteus';

  // With more than one chain a residue number alone is ambiguous: prefix the chain.
  const multiChain = new Set(meta.labels.chain).size > 1;
  function shortLabel(i) {
    if (i >= N) return label(i);
    return (multiChain ? meta.labels.chain[i] + ':' : '') + meta.labels.name[i] + meta.labels.number[i] + meta.labels.icode[i];
  }

  const scores = meta.scores;
  const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
  const compare = meta.compare;
  const SCHEMES = ['ss', 'plddt', 'rainbow'].concat(scores ? ['score'] : [], compare ? ['deviation'] : []);
  let scheme = SCHEMES.includes(meta.scheme) ? meta.scheme : 'ss';
  let noticeText = '';
  let surfaceKey = ''; // what the surface colours mean, while one is shown
  let noticeTimer = 0;
  let V = null; // the 3D view, once WebGL has started

  // The selection: residue ids (ligand k is N + k), shown contacts, and a hover preview.
  const sel = { set: new Set(), anchor: -1, neigh: true, contacts: [], preview: null, pae: null };
  const showKinds = new Set();
  let seqSpans = [];
  let marked = new Set();

  // PAE, 1/8 Å per byte (web.rs), decoded once the page has loaded.
  let pae = null;
  const paeInfo = meta.confidence && meta.confidence.pae;
  if (paeInfo && $('proteus-pae').textContent.trim()) {
    C.gunzip(C.b64ToBytes($('proteus-pae').textContent))
      .then((bytes) => { pae = { n: paeInfo.n, max: paeInfo.max, v: bytes }; drawPae(); updateSelectionUI(); })
      .catch(() => {});
  }
  const paeAt = (i, j) => pae.v[i * pae.n + j] / 8;

  buildPanel();
  buildSequence();
  showLegend();
  updateSelectionUI();
  // The key help wraps on narrow windows: stack the sequence track and the selection box on it.
  function layoutBottom() {
    const keys = $('keys'), seq = $('seq'), box = $('selbox');
    // offsetParent is always null for a fixed element: ask the computed style instead.
    if (getComputedStyle(keys).display === 'none') { seq.style.bottom = box.style.bottom = ''; return; }
    const top = 16 + keys.offsetHeight + 6;
    seq.style.bottom = top + 'px';
    box.style.bottom = top + seq.offsetHeight + 8 + 'px';
  }
  layoutBottom();
  window.addEventListener('resize', layoutBottom);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(layoutBottom);

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
  // 1 for a selected (or undimmed) vertex, 0 for one outside the selection.
  layout(location=4) in float aSel;
  uniform vec3 uLo, uSpan, uCenter;
  uniform mat3 uRot;
  uniform vec2 uScale, uPan, uOffset;
  uniform float uZ;
  out vec3 vNormal;
  out vec3 vColor;
  out float vDepth;
  out float vSel;
  flat out float vRes;
  void main() {
    vec3 world = uLo + aPos * uSpan;
    vec3 view = uRot * (world - uCenter);
    view.xy += uPan;
    vNormal = uRot * aNrm;
    vColor = aCol;
    vDepth = -view.z;          // camera.rs: smaller is closer, viewer at +Z
    vRes = aRes;
    vSel = aSel;
    gl_Position = vec4(view.xy * uScale + uOffset, clamp(vDepth / uZ, -1.0, 1.0), 1.0);
  }`;

  // shade_blinn_phong, per fragment: key + fill + ambient 0.30 + specular, clamped at 1.3, then
  // pipeline.rs depth cueing (foreground 100 %, background 55 %). A normal that points away
  // from the viewer is flipped toward it, which the CPU pipeline does not do (it leaves such a
  // surface at the 0.30 ambient). The rule uses the normal, not gl_FrontFacing, because the
  // ribbon's triangle winding is not consistent enough to say which side is the front.
  // Outside an active selection the colour is pulled 62 % of the way to the background.
  const GEOM_FS = `#version 300 es
  precision highp float;
  in vec3 vNormal;
  in vec3 vColor;
  in float vDepth;
  in float vSel;
  flat in float vRes;
  uniform float uFogLo, uFogHi;
  uniform bool uSelActive;
  uniform vec3 uBg;
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
    vec3 c = min(vColor * i, vec3(1.0)) * (1.0 - 0.45 * frac);
    if (uSelActive && vSel < 0.5) c = mix(c, uBg, 0.62);
    outColor = vec4(c, 1.0);
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
    const bgRgb = meta.palette.ground; // brand::DARK.ground

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
      let colorBuf = null, selBuf = null;
      if (withAttrs) {
        colorBuf = buf(2, C.vertexColors(m, scheme === 'deviation' ? 'score' : scheme, scheme === 'deviation' ? compare.colors : scores && scores.colors), 3, gl.UNSIGNED_BYTE, true);
        buf(3, Float32Array.from(m.res), 1, gl.FLOAT, false);
        selBuf = buf(4, new Float32Array(m.n).fill(1), 1, gl.FLOAT, false);
      }
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { v, colorBuf, selBuf, count: m.nt * 3 };
    }
    // A mesh built on the page (sticks, contact lines): world-space floats.
    function dynVao(m) {
      const v = gl.createVertexArray();
      gl.bindVertexArray(v);
      const bufs = [];
      const buf = (loc, data, size, type, norm) => {
        const b = gl.createBuffer();
        bufs.push(b);
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, type, norm, 0, 0);
      };
      buf(0, m.pos, 3, gl.FLOAT, false);
      buf(1, m.nrm, 3, gl.FLOAT, false);
      buf(2, m.col, 3, gl.UNSIGNED_BYTE, true);
      buf(3, m.res, 1, gl.FLOAT, false);
      const ib = gl.createBuffer();
      bufs.push(ib);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      return { v, count: m.nt * 3, free() { gl.deleteVertexArray(v); for (const b of bufs) gl.deleteBuffer(b); } };
    }
    const ribbon = vao(mesh.ribbon, true);
    const ds = mesh.disulfides.n > 0 ? vao(mesh.disulfides, false) : null;
    const refRibbon = mesh.reference.n > 0 ? vao(mesh.reference, false) : null;
    let sticks = null, lines = null, surface = null;
    // Measurements: each a list of 2–4 atom indices; `current` is the one being built.
    const measures = [];
    let current = [];
    let measuring = false;
    // Pinned residue labels.
    const pinned = new Set();

    // Depth-cueing range: the view-space depth extent of the ribbon, like pipeline.rs, measured
    // on a subsample of vertices each frame.
    const sampleStride = Math.max(1, Math.floor(mesh.ribbon.n / 20000));
    const world = [];
    for (let i = 0; i < mesh.ribbon.n; i += sampleStride) {
      world.push([0, 1, 2].map((k) => mesh.lo[k] + mesh.ribbon.pos[3 * i + k] / 65535 * mesh.span[k]));
    }
    // C-alpha of each residue: the ribbon vertex nearest the spline passes through it closely
    // enough to aim the camera; the atom table has the real one when present.
    const caOf = new Array(N);
    for (let i = 0; i < mesh.atoms.n; i++) {
      const r = mesh.atoms.res[i];
      if (r < N && meta.atomNames[i] === 'CA') caOf[r] = [mesh.atoms.pos[3 * i], mesh.atoms.pos[3 * i + 1], mesh.atoms.pos[3 * i + 2]];
    }

    const cam = meta.camera;
    const base = cam.rotation.flat();
    const center = cam.center;
    const zRange = cam.radius * 1.5 + 12;
    const state = { yaw: 0, pitch: 0, zoom: 1, pan: [0, 0], fx: true, ds: true, ref: true, spin: false, surface: 0 };
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
      gl.uniform3fv(prog.u.uCenter, center);
      gl.uniformMatrix3fv(prog.u.uRot, true, rot);
      gl.uniform2f(prog.u.uScale, 2 * s / W, 2 * s / H);
      gl.uniform2fv(prog.u.uPan, state.pan);
      const v = visibleArea();
      gl.uniform2f(prog.u.uOffset, -v.right / W, v.bottom / H);
      gl.uniform1f(prog.u.uZ, zRange);
    }

    function drawMeshes(prog, forPick) {
      // Quantised meshes are in the blob's box; page-built ones are plain world coordinates.
      const quantised = () => { gl.uniform3fv(prog.u.uLo, mesh.lo); gl.uniform3fv(prog.u.uSpan, mesh.span); };
      const worldSpace = () => { gl.uniform3f(prog.u.uLo, 0, 0, 0); gl.uniform3f(prog.u.uSpan, 1, 1, 1); };
      quantised();
      gl.bindVertexArray(ribbon.v);
      gl.drawElements(gl.TRIANGLES, ribbon.count, gl.UNSIGNED_INT, 0);
      gl.vertexAttrib1f(4, 1);
      if (ds && state.ds && !forPick) {
        gl.bindVertexArray(ds.v);
        const dsColour = meta.palette.disulfide; // brand::structure::DISULFIDE
        gl.vertexAttrib3f(2, dsColour[0] / 255, dsColour[1] / 255, dsColour[2] / 255);
        gl.vertexAttrib1f(3, 0);
        gl.drawElements(gl.TRIANGLES, ds.count, gl.UNSIGNED_INT, 0);
      }
      if (refRibbon && state.ref && !forPick) {
        gl.bindVertexArray(refRibbon.v);
        const c = compare.colour; // brand::structure::REFERENCE
        gl.vertexAttrib3f(2, c[0] / 255, c[1] / 255, c[2] / 255);
        gl.vertexAttrib1f(3, 0);
        gl.vertexAttrib1f(4, selActive ? 0 : 1);
        gl.drawElements(gl.TRIANGLES, refRibbon.count, gl.UNSIGNED_INT, 0);
        gl.vertexAttrib1f(4, 1);
      }
      worldSpace();
      if (surface && state.surface) {
        gl.bindVertexArray(surface.vao.v);
        gl.drawElements(gl.TRIANGLES, surface.vao.count, gl.UNSIGNED_INT, 0);
      }
      if (sticks) {
        gl.bindVertexArray(sticks.v);
        gl.drawElements(gl.TRIANGLES, sticks.count, gl.UNSIGNED_INT, 0);
      }
      if (lines && !forPick) {
        gl.bindVertexArray(lines.v);
        gl.drawElements(gl.TRIANGLES, lines.count, gl.UNSIGNED_INT, 0);
      }
      gl.bindVertexArray(null);
    }

    let selActive = false;
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
      gl.uniform1i(geom.u.uSelActive, selActive ? 1 : 0);
      gl.uniform3f(geom.u.uBg, bgRgb[0] / 255, bgRgb[1] / 255, bgRgb[2] / 255);
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
      gl.uniform3f(post.u.uBg, bgRgb[0] / 255, bgRgb[1] / 255, bgRgb[2] / 255);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      placeOverlay();
    }

    // World → CSS pixels, the inverse of what GEOM_VS does.
    function project(p) {
      const rot = rotation(), s = pxPerA(), v = visibleArea();
      const d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
      const x = rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2] + state.pan[0];
      const y = rot[3] * d[0] + rot[4] * d[1] + rot[5] * d[2] + state.pan[1];
      const nx = x * 2 * s / W - v.right / W, ny = y * 2 * s / H + v.bottom / H;
      return [(nx + 1) / 2 * canvas.clientWidth, (1 - ny) / 2 * canvas.clientHeight];
    }
    const atomPos = (i) => [mesh.atoms.pos[3 * i], mesh.atoms.pos[3 * i + 1], mesh.atoms.pos[3 * i + 2]];
    const atomName = (i) => shortLabel(mesh.atoms.res[i]) + ' ' + meta.atomNames[i];
    function measureText(m) {
      const v = C.measure(m.map(atomPos));
      return m.length === 2 ? v.toFixed(2) + ' Å' : v.toFixed(1) + '°';
    }
    // Labels over the canvas: measurements at their middle, pinned residues at their C-alpha.
    function placeOverlay() {
      const layer = $('overlay');
      const items = [];
      for (const m of measures.concat(current.length >= 2 ? [current] : [])) {
        const pts = m.map(atomPos);
        const mid = [0, 1, 2].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length);
        items.push(['m', mid, measureText(m)]);
      }
      for (const r of pinned) {
        const p = r < N ? caOf[r] : null;
        if (p) items.push(['l', p, shortLabel(r)]);
        else if (r >= N) for (let i = 0; i < mesh.atoms.n; i++) if (mesh.atoms.res[i] === r) { items.push(['l', atomPos(i), label(r)]); break; }
      }
      while (layer.children.length > items.length) layer.lastChild.remove();
      items.forEach(([kind, p, text], k) => {
        let el = layer.children[k];
        if (!el) { el = document.createElement('span'); layer.append(el); }
        el.className = kind === 'm' ? 'measure' : 'pin';
        el.textContent = text;
        const [x, y] = project(p);
        el.style.transform = 'translate(' + Math.round(x + 6) + 'px,' + Math.round(y - 18) + 'px)';
      });
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
      if (id !== 0 && N + ligands.length >= 65535) id += pass(true) * 65536;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return id - 1;
    }

    // ---------------------------------------------------------------- frame scheduling
    let queued = false;
    function frame() {
      queued = false;
      if (state.spin) { state.yaw += 0.01; request(); }
      render();
      if (!state.spin) saveSession();
    }
    function request() { if (!queued) { queued = true; requestAnimationFrame(frame); } }
    new ResizeObserver(() => { hideTip(); request(); }).observe(canvas);
    // Fade in once the first frame is drawn (the stylesheet skips it for reduced motion).
    requestAnimationFrame(() => requestAnimationFrame(() => canvas.classList.add('ready')));
    request();

    // Per-residue colour of the current scheme (a residue's first ribbon vertex): carbons in
    // the sticks take it.
    let resColour = [];
    function residueColours(colours) {
      resColour = new Array(N);
      for (let i = mesh.ribbon.n - 1; i >= 0; i--) {
        const r = mesh.ribbon.res[i];
        resColour[r] = [colours[3 * i], colours[3 * i + 1], colours[3 * i + 2]];
      }
    }
    residueColours(C.vertexColors(mesh.ribbon, scheme === 'deviation' ? 'score' : scheme, scheme === 'deviation' ? compare.colors : scores && scores.colors));

    function recolor() {
      const colours = C.vertexColors(mesh.ribbon, scheme === 'deviation' ? 'score' : scheme, scheme === 'deviation' ? compare.colors : scores && scores.colors);
      gl.bindBuffer(gl.ARRAY_BUFFER, ribbon.colorBuf);
      gl.bufferData(gl.ARRAY_BUFFER, colours, gl.STATIC_DRAW);
      residueColours(colours);
      rebuildSticks();
      if (state.surface === 1) buildSurface();
      showLegend();
      request();
    }

    // ---------------------------------------------------------------- selection in 3D
    const ligandCarbon = rgbOf('accent');
    function rebuildSticks() {
      if (sticks) { sticks.free(); sticks = null; }
      if (lines) { lines.free(); lines = null; }
      const shown = new Set(sel.neigh ? C.neighbours(mesh.atoms, sel.set, 5) : sel.set);
      for (let k = 0; k < ligands.length; k++) shown.add(N + k);
      for (const c of sel.contacts) { shown.add(c[0]); shown.add(c[1]); }
      for (const m of measures.concat([current])) for (const i of m) shown.add(mesh.atoms.res[i]);
      if (shown.size && mesh.atoms.n) {
        const m = C.stickMesh(mesh.atoms, meta.elements, shown,
          (r) => (r < N ? resColour[r] : ligandCarbon) || [200, 200, 200], 0.17);
        if (m.nt) sticks = dynVao(m);
      }
      const b = C.meshBuilder();
      const drawn = new Set();
      const add = (c, colour) => {
        const key = c[2].join() + c[3].join();
        if (drawn.has(key)) return;
        drawn.add(key);
        b.dashes(c[2], c[3], 0.06, 0.22, 0.16, colour, c[0]);
      };
      for (const [key, , r] of KINDS) {
        if (!showKinds.has(key)) continue;
        const colour = rgbOf(r);
        for (const c of issues[key]) add(c, colour);
      }
      for (const c of sel.contacts) add(c, rgbOf(c.role || 'text'));
      const accent = rgbOf('accent');
      for (const m of measures.concat(current.length >= 2 ? [current] : [])) {
        for (let k = 0; k + 1 < m.length; k++) b.dashes(atomPos(m[k]), atomPos(m[k + 1]), 0.05, 0.18, 0.12, accent, 0);
      }
      for (const i of current) b.sphere(atomPos(i), 0.32, accent, mesh.atoms.res[i]);
      const lm = b.build();
      if (lm.nt) lines = dynVao(lm);
    }

    function applyDimming() {
      const set = sel.preview || sel.set;
      selActive = set.size > 0;
      const f = new Float32Array(mesh.ribbon.n);
      if (selActive) for (let i = 0; i < mesh.ribbon.n; i++) f[i] = set.has(mesh.ribbon.res[i]) ? 1 : 0;
      else f.fill(1);
      gl.bindBuffer(gl.ARRAY_BUFFER, ribbon.selBuf);
      gl.bufferData(gl.ARRAY_BUFFER, f, gl.STATIC_DRAW);
    }

    // Centre the view on the selection (and zoom in to about 24 Å across when asked).
    function focus(zoomIn) {
      const pts = [];
      for (const c of sel.contacts) pts.push(c[2], c[3]);
      for (const r of sel.set) {
        if (r < N && caOf[r]) pts.push(caOf[r]);
        else if (r >= N) for (let i = 0; i < mesh.atoms.n; i++) if (mesh.atoms.res[i] === r) pts.push([mesh.atoms.pos[3 * i], mesh.atoms.pos[3 * i + 1], mesh.atoms.pos[3 * i + 2]]);
      }
      if (!pts.length) return;
      const c = [0, 1, 2].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length);
      const rot = rotation();
      const d = [c[0] - center[0], c[1] - center[1], c[2] - center[2]];
      state.pan = [-(rot[0] * d[0] + rot[1] * d[1] + rot[2] * d[2]), -(rot[3] * d[0] + rot[4] * d[1] + rot[5] * d[2])];
      if (zoomIn) {
        let extent = 0;
        for (const p of pts) extent = Math.max(extent, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
        state.zoom = Math.min(50, Math.max(state.zoom, cam.radius / Math.max(12, extent + 8)));
      }
      request();
    }

    // ---------------------------------------------------------------- surfaces
    const SURFACES = ['', 'surface', 'hydrophobicity', 'Coulombic potential'];
    let surfaceGeom = null;
    function buildSurface() {
      if (surface) { surface.vao.free(); surface = null; }
      if (!state.surface) { if (surfaceKey) { surfaceKey = ''; showLegend(); } return; }
      if (!surfaceGeom) {
        const t0 = performance.now();
        const prot = (i) => mesh.atoms.res[i] < N;
        surfaceGeom = C.gaussianSurface(mesh.atoms, meta.elements, prot, mesh.atoms.n > 20000 ? 1.5 : 1.0);
        surfaceGeom.ms = performance.now() - t0;
      }
      const g = surfaceGeom;
      const col = new Uint8Array(g.n * 3), res = new Float32Array(g.n);
      const mode = SURFACES[state.surface];
      let charges = null, grid = null;
      if (mode === 'Coulombic potential') {
        const resName = (i) => meta.labels.name[mesh.atoms.res[i]] || '';
        const names = [], rn = [];
        for (let i = 0; i < mesh.atoms.n; i++) { names.push(meta.atomNames[i]); rn.push(mesh.atoms.res[i] < N ? resName(i) : ''); }
        charges = C.formalCharges(names, rn, mesh.atoms.res);
        // Charges binned on a 10 Å grid; a charge beyond 20 Å adds under 0.3 kcal/mol/e.
        grid = new Map();
        for (const c of charges) {
          const k = [0, 1, 2].map((d) => Math.floor(mesh.atoms.pos[3 * c[0] + d] / 10)).join();
          if (!grid.has(k)) grid.set(k, []);
          grid.get(k).push(c);
        }
      }
      for (let v = 0; v < g.n; v++) {
        const a = g.atom[v], r = mesh.atoms.res[a];
        res[v] = r;
        let c;
        if (mode === 'hydrophobicity') {
          const kd = C.KYTE_DOOLITTLE[meta.labels.name[r]];
          // ChimeraX's palette: dark cyan (hydrophilic) → white → goldenrod (hydrophobic).
          c = kd === undefined ? [200, 200, 200] : kd < 0 ? C.lerp([0, 139, 139], [255, 255, 255], 1 + kd / 4.5) : C.lerp([255, 255, 255], [218, 165, 32], kd / 4.5);
        } else if (mode === 'Coulombic potential') {
          const p = [g.pos[3 * v], g.pos[3 * v + 1], g.pos[3 * v + 2]];
          const cell = p.map((x) => Math.floor(x / 10));
          const near = [];
          for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) for (let dz = -2; dz <= 2; dz++) {
            const list = grid.get([cell[0] + dx, cell[1] + dy, cell[2] + dz].join());
            if (list) for (const q of list) near.push(q);
          }
          const phi = C.coulomb(p, mesh.atoms.pos, near);
          // Red −10 → white → blue +10 kcal/mol/e, ChimeraX's default range.
          c = phi < 0 ? C.lerp([255, 255, 255], [220, 30, 30], Math.min(1, -phi / 10)) : C.lerp([255, 255, 255], [30, 60, 220], Math.min(1, phi / 10));
        } else {
          c = (r < N ? resColour[r] : ligandCarbon) || [200, 200, 200];
        }
        col[3 * v] = c[0]; col[3 * v + 1] = c[1]; col[3 * v + 2] = c[2];
      }
      surface = { vao: dynVao({ n: g.n, nt: g.nt, pos: g.pos, nrm: g.nrm, col, res, idx: g.idx }) };
      surfaceKey = mode === 'hydrophobicity' ? 'surface: cyan hydrophilic → gold hydrophobic (Kyte–Doolittle)'
        : mode === 'Coulombic potential' ? 'surface: red −10 → blue +10 kcal/mol/e (formal charges, ε = 4r)'
        : 'surface: coloured as the ribbon';
      showLegend();
    }

    // ---------------------------------------------------------------- measuring
    // The stick atom nearest a click, within 14 px on screen; else the clicked residue's C-alpha.
    function atomAt(cssX, cssY, residue) {
      let best = -1, bd = 14 * 14;
      const shown = new Set(sel.neigh ? C.neighbours(mesh.atoms, sel.set, 5) : sel.set);
      for (let k = 0; k < ligands.length; k++) shown.add(N + k);
      if (residue >= 0) shown.add(residue);
      for (let i = 0; i < mesh.atoms.n; i++) {
        if (!shown.has(mesh.atoms.res[i])) continue;
        const [x, y] = project(atomPos(i));
        const d = (x - cssX) * (x - cssX) + (y - cssY) * (y - cssY);
        if (d < bd) { bd = d; best = i; }
      }
      if (best < 0 && residue >= 0) {
        for (let i = 0; i < mesh.atoms.n; i++) if (mesh.atoms.res[i] === residue && meta.atomNames[i] === 'CA') return i;
      }
      return best;
    }
    function measureClick(cssX, cssY, residue) {
      const i = atomAt(cssX, cssY, residue);
      if (i < 0) { notice('no atom there: select a residue first so its atoms are drawn'); return; }
      if (current.length === 4) current = [];
      current.push(i);
      const kinds = ['', 'pick 1–3 more atoms', 'distance', 'angle', 'dihedral'];
      notice(current.map(atomName).join(' – ') + (current.length >= 2 ? ' = ' + measureText(current) + ' (' + kinds[current.length] + '; enter keeps it)' : ' · ' + kinds[1]));
      rebuildSticks();
      saveSession();
      request();
    }
    function finishMeasure() {
      if (current.length >= 2) measures.push(current);
      current = [];
      rebuildSticks();
      saveSession();
      request();
    }

    // ---------------------------------------------------------------- sessions
    // The view lives in the URL fragment: reload, bookmark or share the link to get it back.
    let sessionTimer = 0;
    // Written at once when the page is left, so a reload right after a change keeps it.
    function saveSessionNow() {
      clearTimeout(sessionTimer);
      const v = { y: +state.yaw.toFixed(3), p: +state.pitch.toFixed(3), z: +state.zoom.toFixed(3),
        pan: state.pan.map((x) => +x.toFixed(2)), c: scheme, fx: state.fx ? 1 : 0, ds: state.ds ? 1 : 0,
        ref: state.ref ? 1 : 0, u: state.surface, s: [...sel.set], n: sel.neigh ? 1 : 0, k: [...showKinds],
        l: [...pinned], m: measures };
      try { history.replaceState(null, '', '#v=' + C.encodeSession(v)); } catch (_) { /* file:// in some browsers */ }
    }
    function saveSession() {
      clearTimeout(sessionTimer);
      sessionTimer = setTimeout(saveSessionNow, 250);
    }
    window.addEventListener('pagehide', saveSessionNow);
    function restoreSession() {
      const m = /[#&]v=([A-Za-z0-9_-]+)/.exec(location.hash);
      const v = m && C.decodeSession(m[1]);
      if (!v) return;
      const num = (x, d) => (typeof x === 'number' && isFinite(x) ? x : d);
      state.yaw = num(v.y, 0); state.pitch = num(v.p, 0); state.zoom = Math.min(50, Math.max(0.1, num(v.z, 1)));
      if (Array.isArray(v.pan) && v.pan.length === 2) state.pan = v.pan.map((x) => num(x, 0));
      if (SCHEMES.includes(v.c)) scheme = v.c;
      state.fx = v.fx !== 0; state.ds = v.ds !== 0; state.ref = v.ref !== 0;
      state.surface = Math.max(0, Math.min(SURFACES.length - 1, num(v.u, 0) | 0));
      sel.neigh = v.n !== 0;
      const valid = (i) => Number.isInteger(i) && i >= 0 && i < N + ligands.length;
      for (const k of v.k || []) if (KINDS.some((x) => x[0] === k)) showKinds.add(k);
      for (const r of v.l || []) if (valid(r)) pinned.add(r);
      for (const mm of v.m || []) if (Array.isArray(mm) && mm.length >= 2 && mm.length <= 4 && mm.every((i) => Number.isInteger(i) && i >= 0 && i < mesh.atoms.n)) measures.push(mm);
      sel.set = new Set((v.s || []).filter(valid));
      document.querySelectorAll('#panel .findings input').forEach((cb) => { cb.checked = showKinds.has(cb.dataset.kind); });
    }

    V = {
      selectionChanged(opts) {
        rebuildSticks();
        applyDimming();
        if (opts && opts.focus) focus(opts.zoom);
        saveSession();
        request();
      },
      togglePins() {
        if (!sel.set.size) { notice('select residues to label them'); return; }
        const all = [...sel.set].every((r) => pinned.has(r));
        for (const r of sel.set) if (all) pinned.delete(r); else pinned.add(r);
        notice((all ? 'unpinned ' : 'pinned ') + sel.set.size + ' label' + (sel.set.size > 1 ? 's' : ''));
        saveSession();
        request();
      },
      link() { saveSession(); return location.href; },
      previewChanged() { applyDimming(); request(); },
      atoms: mesh.atoms,
      caOf,
    };

    // ---------------------------------------------------------------- input
    const pointers = new Map();
    let pinch = null, down = null;
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey });
      down = pointers.size === 1 && e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
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
    const up = (e) => {
      // A press that did not move is a click: pick what is under it.
      if (down && e.type === 'pointerup' && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) {
        const r = canvas.getBoundingClientRect();
        const i = pickAt(e.clientX - r.left, e.clientY - r.top);
        if (measuring) measureClick(e.clientX - r.left, e.clientY - r.top, i);
        else clickResidue(i, e);
        request();
      }
      down = null;
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = null;
    };
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
      if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      switch (e.key) {
        case 'c': scheme = SCHEMES[(SCHEMES.indexOf(scheme) + 1) % SCHEMES.length]; recolor(); break;
        case 'o': state.fx = !state.fx; request(); break;
        case 'd':
          if (!meta.disulfides) { notice('no disulfides in this structure'); break; }
          state.ds = !state.ds;
          notice('disulfides ' + (state.ds ? 'shown' : 'hidden') + ' (' + meta.disulfides + ')');
          request();
          break;
        case 'x':
          if (!refRibbon) { notice('no reference: open with --compare REF'); break; }
          state.ref = !state.ref;
          notice('reference ' + compare.name + (state.ref ? ' shown' : ' hidden'));
          request();
          break;
        case 'n':
          sel.neigh = !sel.neigh;
          notice('neighbours within 5 Å ' + (sel.neigh ? 'shown' : 'hidden'));
          select(sel.set);
          break;
        case 'f':
          if (!sel.set.size) { notice('nothing selected to focus on'); break; }
          focus(true);
          break;
        case 'm':
          measuring = !measuring;
          if (!measuring) finishMeasure();
          canvas.style.cursor = measuring ? 'crosshair' : '';
          notice(measuring ? 'measure: click 2 atoms for a distance, 3 for an angle, 4 for a dihedral; enter keeps it' :
            measures.length + ' measurement' + (measures.length === 1 ? '' : 's') + ' kept (backspace removes the last)');
          break;
        case 'Enter': if (measuring) { finishMeasure(); notice(measures.length + ' kept'); } else return; break;
        case 'Backspace':
          if (current.length) current.pop(); else if (measures.length) measures.pop(); else return;
          rebuildSticks(); saveSession(); request();
          break;
        case 'l': V.togglePins(); break;
        case 'u':
          state.surface = (state.surface + 1) % SURFACES.length;
          buildSurface();
          notice(state.surface ? surfaceGeom.nt + ' triangles in ' + Math.round(surfaceGeom.ms) + ' ms' : 'surface off');
          saveSession();
          request();
          break;
        case 'Escape':
          if (measuring && current.length) { current = []; rebuildSticks(); request(); break; }
          select(new Set());
          break;
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
        if (i < 0 || i >= N + ligands.length) { hideTip(); return; }
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
    restoreSession();
    if (scheme !== (SCHEMES.includes(meta.scheme) ? meta.scheme : 'ss')) recolor();
    rebuildSticks();
    applyDimming();
    buildSurface();
    updateSelectionUI();
    window.ProteusViewer = { state, render, pickAt, sel, select, focus, project, measures, pinned, measureClick, finishMeasure, saveSessionNow,
      get surfaceTriangles() { return surface ? surface.vao.count / 3 : 0; }, get scheme() { return scheme; },
      get sticks() { return sticks ? sticks.count / 3 : 0; }, get lines() { return lines ? lines.count / 3 : 0; } };
  }

  // ---------------------------------------------------------------- selection model
  /**
   * Replace the selection. `opts.contacts` are findings to draw (with `role` for colour),
   * `opts.focus` centres the view on the result, `opts.pae` records a PAE box for the summary.
   */
  function select(set, opts) {
    opts = opts || {};
    sel.set = new Set(set);
    sel.contacts = opts.contacts || [];
    sel.pae = opts.pae || null;
    sel.preview = null;
    if (V) V.selectionChanged(opts);
    updateSelectionUI();
  }

  function preview(set) {
    sel.preview = set;
    if (V) V.previewChanged();
    markSequence(set || sel.set);
  }

  /** A click on residue `i` in any view: replace, extend (shift: range) or toggle (ctrl). */
  function clickResidue(i, e) {
    if (i < 0) { if (!e.shiftKey && !e.ctrlKey && !e.metaKey) select(new Set()); return; }
    let next;
    if ((e.ctrlKey || e.metaKey)) {
      next = new Set(sel.set);
      if (next.has(i)) next.delete(i); else next.add(i);
    } else if (e.shiftKey && sel.anchor >= 0 && i < N && sel.anchor < N) {
      next = new Set(sel.set);
      const [a, b] = sel.anchor < i ? [sel.anchor, i] : [i, sel.anchor];
      for (let r = a; r <= b; r++) next.add(r);
    } else {
      next = sel.set.size === 1 && sel.set.has(i) ? new Set() : new Set([i]);
    }
    sel.anchor = i;
    select(next);
  }

  function label(i) {
    if (i >= N) {
      const l = ligands[i - N];
      return l ? l.name + ' ' + (l.chain ? l.chain + ':' : '') + l.number : '?';
    }
    const L = meta.labels;
    return L.name[i] + ' ' + (L.chain[i].trim() ? L.chain[i] + ':' : '') + L.number[i] + L.icode[i];
  }

  function updateSelectionUI() {
    markSequence(sel.set);
    const box = $('selbox');
    const ids = [...sel.set].sort((a, b) => a - b);
    box.hidden = ids.length === 0;
    if (!ids.length) return;
    box.textContent = '';
    const h = document.createElement('b');
    h.textContent = '(selection · ' + ids.length + ')';
    const names = document.createElement('div');
    names.className = 'names';
    names.textContent = ids.slice(0, 10).map(shortLabel).join(' · ') + (ids.length > 10 ? ' · +' + (ids.length - 10) + ' more' : '');
    box.append(h, names);
    const facts = [];
    if (ids.length === 2 && V) {
      const [a, b] = ids;
      if (a < N && b < N && V.caOf[a] && V.caOf[b]) {
        const p = V.caOf[a], q = V.caOf[b];
        facts.push('Cα–Cα ' + Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]).toFixed(1) + ' Å');
      }
      const [d, i, j] = C.closestAtoms(V.atoms, a, b);
      if (isFinite(d)) facts.push('closest atoms ' + d.toFixed(2) + ' Å (' + meta.atomNames[i] + '–' + meta.atomNames[j] + ')');
    }
    if (pae && ids.length === 2 && ids[1] < N) {
      const [a, b] = ids;
      facts.push('PAE ' + paeAt(a, b).toFixed(1) + ' / ' + paeAt(b, a).toFixed(1) + ' Å');
    }
    if (pae && sel.pae) {
      const { rows, cols } = sel.pae;
      let s = 0, n = 0;
      for (let i = rows[0]; i <= rows[1]; i++) for (let j = cols[0]; j <= cols[1]; j++) { s += (paeAt(i, j) + paeAt(j, i)) / 2; n++; }
      facts.push('mean PAE between ' + shortLabel(rows[0]) + '–' + shortLabel(rows[1]) + ' and ' + shortLabel(cols[0]) + '–' + shortLabel(cols[1]) + ': ' + (s / n).toFixed(1) + ' Å');
    }
    for (const c of sel.contacts) facts.push(c[5] + ' · ' + c[4].toFixed(2) + ' Å' + (c.kind === 'clashes' ? ' overlap' : ''));
    if (V && sel.neigh && ids.length <= 3) {
      const near = [...C.neighbours(V.atoms, sel.set, 5)].filter((i) => !sel.set.has(i)).sort((a, b) => a - b);
      if (near.length) facts.push('within 5 Å (' + near.length + '): ' + near.slice(0, 24).map(shortLabel).join(' ') + (near.length > 24 ? ' …' : ''));
    }
    if (ids.length === 1 && ids[0] < N) {
      const i = ids[0];
      const v = meta.perResidue[i];
      facts.push((meta.dssp[i] ? SS8[meta.dssp[i]] || 'coil' : 'coil') + (v !== undefined ? ' · ' + (meta.predicted ? 'pLDDT ' : 'B-factor ') + v.toFixed(1) : ''));
    }
    for (const f of facts) { const p = document.createElement('div'); p.textContent = f; box.append(p); }
    const row = document.createElement('div');
    row.className = 'actions';
    const btn = (text, fn, title) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = text; if (title) b.title = title; b.addEventListener('click', fn); row.append(b); };
    btn(sel.neigh ? '5 Å neighbours: on' : '5 Å neighbours: off', () => { sel.neigh = !sel.neigh; select(sel.set, { contacts: sel.contacts, pae: sel.pae }); }, 'n');
    btn('focus', () => { if (V) window.ProteusViewer.focus(true); }, 'f');
    btn('label', () => { if (V) V.togglePins(); }, 'l');
    const residues = ids.filter((i) => i < N).map((i) => ({ chain: meta.labels.chain[i], number: meta.labels.number[i], icode: meta.labels.icode[i] }));
    const ligs = ids.filter((i) => i >= N).map((i) => ligands[i - N]).filter(Boolean)
      .map((l) => ({ chain: l.chain, number: l.number, icode: '' }));
    const pymol = C.pymolSelection(residues.concat(ligs));
    btn('copy PyMOL', () => {
      const done = () => notice('copied: select sele, ' + pymol);
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText('select sele, ' + pymol).then(done, () => notice('select sele, ' + pymol));
      else notice('select sele, ' + pymol);
    });
    btn('clear', () => select(new Set()), 'esc');
    box.append(row);
  }

  // ---------------------------------------------------------------- panel helpers

  function showTip(i, x, y, extra) {
    const tip = $('tip');
    tip.textContent = '';
    const head = document.createElement('b');
    const rest = document.createElement('span');
    if (i >= N) {
      const l = ligands[i - N];
      head.textContent = label(i);
      rest.textContent = 'ligand · ' + l.atoms + ' heavy atoms';
    } else {
      const value = meta.perResidue[i];
      const conf = value === undefined ? '' : meta.predicted ? 'pLDDT ' + value.toFixed(1) : 'B-factor ' + value.toFixed(1);
      head.textContent = label(i);
      const dv = compare && compare.deviation[i];
      const sv = scores && scores.values[i];
      rest.textContent = (SS8[meta.dssp[i]] || 'coil') + ' · ' + (meta.dssp[i] || '-') + (conf ? ' · ' + conf : '') +
        (sv !== undefined && sv !== null ? ' · ' + scores.column + ' ' + fmt(sv) : '') +
        (dv !== undefined && dv !== null ? ' · moved ' + dv.toFixed(2) + ' Å' : '');
    }
    tip.append(head, document.createElement('br'), rest);
    if (extra) { const e = document.createElement('span'); e.textContent = extra; tip.append(document.createElement('br'), e); }
    placeTip(x, y);
  }
  function tipText(title, text, x, y) {
    const tip = $('tip');
    tip.textContent = '';
    const head = document.createElement('b');
    head.textContent = title;
    const rest = document.createElement('span');
    rest.textContent = text;
    tip.append(head, document.createElement('br'), rest);
    placeTip(x, y);
  }
  function placeTip(x, y) {
    const tip = $('tip');
    tip.hidden = false;
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - tip.offsetWidth - 8)) + 'px';
    tip.style.top = Math.max(8, Math.min(y + 14, window.innerHeight - tip.offsetHeight - 8)) + 'px';
  }
  function hideTip() { $('tip').hidden = true; }

  function swatch(rgb, text) {
    const s = document.createElement('span');
    s.className = 'sw';
    const d = document.createElement('i');
    d.style.background = 'rgb(' + rgb.join(',') + ')';
    s.append(d, document.createTextNode(text));
    return s;
  }

  // A key that changes nothing visible still answers, in the legend (aria-live) for a moment.
  function notice(text) {
    noticeText = text;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => { noticeText = ''; showLegend(); }, 2500);
    showLegend();
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
    } else if (scheme === 'score') {
      const sc = scores.scale;
      const [bad, good] = scores.higherIsWorse ? [sc.hi, sc.lo] : [sc.lo, sc.hi];
      name.textContent = '(' + scores.column + ')';
      const col = (v) => C.scoreColor(v, { lo: sc.lo, hi: sc.hi, higherIsWorse: scores.higherIsWorse }, scores.stops, scores.none);
      el.append(name, swatch(col(bad), fmt(bad) + ' damaging'));
      if (sc.diverging) el.append(swatch(col(0), '0'));
      el.append(swatch(col(good), fmt(good) + ' tolerated'), swatch(scores.none, 'no score'));
    } else if (scheme === 'deviation') {
      name.textContent = '(Cα deviation from ' + compare.name + ')';
      const col = (v) => C.scoreColor(v, { lo: 0, hi: compare.max, higherIsWorse: true }, compare.stops, compare.none);
      el.append(name, swatch(col(0), '0 Å'), swatch(col(compare.max / 2), fmt(compare.max / 2) + ' Å'),
        swatch(col(compare.max), '≥ ' + fmt(compare.max) + ' Å'), swatch(compare.none, 'unpaired'),
        swatch(compare.colour, 'reference'));
    } else if (scheme === 'rainbow') {
      name.textContent = '(sequence position)';
      el.append(name, swatch(C.rainbowColor(0, 4), 'N-terminus'), swatch(C.rainbowColor(2, 4), 'middle'),
        swatch(C.rainbowColor(4, 4), 'C-terminus'));
    } else {
      name.textContent = '(secondary structure · dssp)';
      el.append(name, swatch(C.ssColor(0), 'helix'), swatch(C.ssColor(1), 'strand'), swatch(C.ssColor(2), 'coil'));
    }
    if (surfaceKey) {
      const k = document.createElement('span');
      k.className = 'surface-key';
      k.textContent = surfaceKey;
      el.append(k);
    }
    if (noticeText) {
      const n = document.createElement('span');
      n.className = 'notice';
      n.textContent = noticeText;
      el.append(n);
    }
  }

  function buildPanel() {
    const panel = $('panel');
    const h = (text) => { const e = document.createElement('h2'); e.textContent = '(' + text + ')'; panel.append(e); return e; };
    const conf = meta.confidence;
    const rows = meta.metrics.slice();
    if (conf && (conf.ptm != null || conf.iptm != null)) {
      const parts = [];
      if (conf.ptm != null) parts.push('pTM ' + conf.ptm.toFixed(3));
      if (conf.iptm != null) parts.push('ipTM ' + conf.iptm.toFixed(3));
      if (conf.ligandIptm != null) parts.push('ligand ipTM ' + conf.ligandIptm.toFixed(3));
      rows.splice(Math.min(2, rows.length), 0, ['predicted TM-score', parts.join(' · ')]);
    }
    if (compare) {
      rows.unshift(['superposed on ' + compare.name, 'Cα RMSD ' + compare.rmsd.toFixed(2) + ' Å over ' + compare.paired +
        ' pairs (' + compare.pairing + ')' + (compare.mismatched ? ' · ' + compare.mismatched + ' differ in residue' : '')]);
    }
    if (rows.length) {
      h('measurements');
      const dl = document.createElement('dl');
      for (const [k, v] of rows) {
        const dt = document.createElement('dt'); dt.textContent = k;
        const dd = document.createElement('dd'); dd.textContent = C.keepUnits(v);
        dl.append(dt, dd);
      }
      panel.append(dl);
    }
    // Complexes: each chain's pTM and each pair's ipTM (Boltz), chains in input order.
    if (conf && conf.pairIptm && conf.pairIptm.length > 1) {
      h('chains · pTM on the diagonal, ipTM off it');
      const names = chainNames();
      const t = document.createElement('table');
      t.className = 'grid';
      const head = document.createElement('tr');
      head.append(document.createElement('th'));
      for (const n of names.slice(0, conf.pairIptm.length)) { const th = document.createElement('th'); th.textContent = n; head.append(th); }
      t.append(head);
      conf.pairIptm.forEach((row, i) => {
        const tr = document.createElement('tr');
        const th = document.createElement('th'); th.textContent = names[i] || String(i); tr.append(th);
        row.forEach((v, j) => {
          const td = document.createElement('td');
          td.textContent = v.toFixed(2);
          td.className = v >= 0.8 ? 'good' : v >= 0.6 ? '' : 'bad';
          if (i === j) td.classList.add('diag');
          tr.append(td);
        });
        t.append(tr);
      });
      panel.append(t);
      const note = document.createElement('p');
      note.className = 'note tight';
      note.textContent = 'ipTM above 0.8 is a confident interface; below 0.6 the chains\' relative placement is a guess.';
      panel.append(note);
    }
    if (meta.models && meta.models.length > 1) {
      h('models · ranked by the predictor');
      const t = document.createElement('table');
      t.className = 'grid models';
      const hr = document.createElement('tr');
      const hasLig = meta.models.some((m) => m.ligandRmsd !== null && m.ligandRmsd !== undefined);
      for (const c of ['#', 'score', 'pTM', 'ipTM', 'pLDDT', 'Cα', ...(hasLig ? ['ligand'] : [])]) { const th = document.createElement('th'); th.textContent = c; hr.append(th); }
      t.append(hr);
      const f = (v, d) => (v === null || v === undefined ? '—' : v.toFixed(d));
      for (const m of meta.models) {
        const tr = document.createElement('tr');
        if (m.shown) tr.className = 'shown';
        tr.title = m.file + (m.shown ? ' (shown)' : ' — proteus view … --model ' + m.rank);
        const cells = [String(m.rank) + (m.shown ? ' ◀' : ''), f(m.score, 3), f(m.ptm, 3), f(m.iptm, 3), f(m.plddt, 1), m.shown ? '—' : f(m.rmsd, 2)];
        if (hasLig) cells.push(m.shown ? '—' : f(m.ligandRmsd, 2));
        for (const v of cells) {
          const td = document.createElement('td'); td.textContent = v; tr.append(td);
        }
        t.append(tr);
      }
      panel.append(t);
      const note = document.createElement('p');
      note.className = 'note tight';
      note.textContent = 'RMSDs (Å) to the model shown, after superposing on the protein; the ligand column is heavy atoms. Models that agree are more likely right; open another with --model K.';
      panel.append(note);
    }
    if (paeInfo) {
      h('predicted aligned error');
      const c = document.createElement('canvas');
      c.className = 'pae';
      c.id = 'pae';
      c.setAttribute('aria-label', 'predicted aligned error heatmap');
      panel.append(c);
      const key = document.createElement('div');
      key.className = 'key gradient';
      const bar = document.createElement('i');
      bar.style.background = 'linear-gradient(90deg,' + [0, 0.25, 0.5, 0.75, 1].map((t) => 'rgb(' + C.paeColor(t * paeInfo.max, paeInfo.max).join(',') + ')').join(',') + ')';
      key.append(document.createTextNode('0'), bar, document.createTextNode(paeInfo.max.toFixed(1) + ' Å'));
      panel.append(key);
      const note = document.createElement('p');
      note.className = 'note tight';
      note.textContent = 'Expected error (Å) at the scored residue (x) when the model is aligned on residue (y); mean ' +
        paeInfo.mean.toFixed(1) + ' Å. Dark blocks are parts placed confidently relative to each other. Drag a box to select two ranges.';
      panel.append(note);
    }
    if (scores && scores.matrix) {
      h('substitutions · ' + scores.column);
      panel.append(scanMap());
      const note = document.createElement('p');
      note.className = 'note tight';
      note.textContent = 'One column per residue, one row per amino acid (A at the top). Red is the damaging end; ' +
        'dotted cells are the wild type. Positions matched by ' + scores.matchedBy + '.';
      panel.append(note);
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
    buildFindings(h);
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
    // Files: the model this page was drawn from, and its PAE in AlphaFold DB's JSON layout.
    const src = $('proteus-source');
    const files = [];
    if (src && src.textContent.trim()) files.push(['model · ' + src.dataset.name, () =>
      C.gunzip(C.b64ToBytes(src.textContent)).then((b) => download(src.dataset.name, b, 'chemical/x-pdb'))]);
    if (paeInfo) files.push(['PAE · json', () => {
      if (!pae) return;
      const rows = [];
      for (let i = 0; i < pae.n; i++) rows.push(Array.from(pae.v.subarray(i * pae.n, (i + 1) * pae.n), (x) => x / 8));
      const name = (src && src.dataset.name ? src.dataset.name.replace(/\.[^.]+$/, '') : subject) + '-predicted_aligned_error.json';
      download(name, new TextEncoder().encode(JSON.stringify([{ predicted_aligned_error: rows, max_predicted_aligned_error: pae.max }])), 'application/json');
    }]);
    files.push(['view link', () => {
      const url = V ? V.link() : location.href;
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(() => notice('link to this view copied'), () => notice(url));
      else notice(url);
    }]);
    if (files.length) {
      h('files');
      const row = document.createElement('div');
      row.className = 'files';
      for (const [text, fn] of files) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = '↓ ' + text;
        b.addEventListener('click', fn);
        row.append(b);
      }
      panel.append(row);
    }
    const note = document.createElement('p');
    note.className = 'note';
    note.textContent = 'Drawn by Proteus from its own ribbon geometry and DSSP. Nothing is fetched from the network.' +
      (conf && conf.sources && conf.sources.length ? ' Confidence read from ' + conf.sources.join(', ') + '.' : '');
    panel.append(note);
  }

  // The findings, each one a click away from being shown in 3D.
  function buildFindings(h) {
    const groups = [];
    const rama = issues.rama || [];
    if (rama.length || meta.rama.length) {
      const outliers = rama.filter((x) => x[1] === 2).length;
      groups.push({ title: 'Ramachandran: ' + outliers + ' outlier' + (outliers === 1 ? '' : 's') + ', ' + (rama.length - outliers) + ' allowed',
        items: rama.map(([r, reg]) => ({ text: shortLabel(r) + ' · ' + (reg === 2 ? 'outlier' : 'allowed'), cls: reg === 2 ? 'bad' : 'warm', go: () => select(new Set([r]), { focus: true, zoom: true }) })) });
    }
    for (const [key, name, r, unit] of KINDS) {
      const list = issues[key] || [];
      if (!list.length && key !== 'clashes') continue;
      groups.push({ title: name + ' (' + list.length + ')', kind: key, role: r,
        items: list.map((c) => ({ text: c[5] + ' · ' + c[4].toFixed(2) + ' Å' + (unit ? ' ' + unit : ''),
          go: () => { const x = c.slice(); x.role = r; x.kind = key; select(new Set([c[0], c[1]]), { contacts: [x], focus: true, zoom: true }); } })) });
    }
    if (ligands.length) {
      groups.push({ title: 'ligands (' + ligands.length + ')',
        items: ligands.map((l, k) => ({ text: l.name + ' ' + (l.chain ? l.chain + ':' : '') + l.number + ' · ' + l.atoms + ' atoms · binding site ≤ 5 Å',
          go: () => { sel.neigh = true; select(new Set([N + k]), { focus: true, zoom: true }); } })) });
    }
    if (!groups.length) return;
    h('findings');
    const wrap = document.createElement('div');
    wrap.className = 'findings';
    for (const g of groups) {
      const d = document.createElement('details');
      const s = document.createElement('summary');
      s.textContent = g.title;
      d.append(s);
      if (g.kind && g.items.length) {
        const lab = document.createElement('label');
        lab.className = 'all';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.dataset.kind = g.kind;
        cb.addEventListener('change', () => {
          if (cb.checked) showKinds.add(g.kind); else showKinds.delete(g.kind);
          if (V) V.selectionChanged();
        });
        lab.append(cb, document.createTextNode(' draw all'));
        d.append(lab);
      }
      // Built on first open: a large structure has thousands of hydrogen bonds.
      d.addEventListener('toggle', () => {
        if (!d.open || d.dataset.built) return;
        d.dataset.built = '1';
        const ul = document.createElement('ul');
        const LIMIT = 400;
        for (const it of g.items.slice(0, LIMIT)) {
          const li = document.createElement('li');
          const b = document.createElement('button');
          b.type = 'button';
          b.textContent = it.text;
          if (it.cls) b.className = it.cls;
          b.addEventListener('click', it.go);
          li.append(b);
          ul.append(li);
        }
        if (g.items.length > LIMIT) { const li = document.createElement('li'); li.className = 'more'; li.textContent = '+' + (g.items.length - LIMIT) + ' more (draw all shows them in 3D)'; ul.append(li); }
        if (!g.items.length) { const li = document.createElement('li'); li.className = 'more'; li.textContent = 'none'; ul.append(li); }
        d.append(ul);
      });
      wrap.append(d);
    }
    $('panel').append(wrap);
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
    // Hover names the residue under the pointer; a click selects it.
    const nearest = (e) => {
      const r = c.getBoundingClientRect(), k = px / r.width;
      const x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
      let best = null, bd = 7 * 7;
      for (const p of meta.rama) {
        if (p[3] < 0) continue;
        const dx = to(p[0]) - x, dy = px - to(p[1]) - y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = p; }
      }
      return best;
    };
    c.addEventListener('mousemove', (e) => {
      const p = nearest(e);
      c.style.cursor = p ? 'pointer' : 'default';
      if (!p) { hideTip(); preview(null); return; }
      showTip(p[3], e.clientX, e.clientY, 'φ ' + p[0].toFixed(0) + '° ψ ' + p[1].toFixed(0) + '° · ' + ['favoured', 'allowed', 'outlier'][p[2]]);
      preview(new Set([p[3]]));
    });
    c.addEventListener('mouseleave', () => { hideTip(); preview(null); });
    c.addEventListener('click', (e) => { const p = nearest(e); if (p) { sel.anchor = p[3]; select(new Set([p[3]]), { focus: true, zoom: true }); } });
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
    const at = (e) => {
      const r = c.getBoundingClientRect();
      return Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n)));
    };
    c.style.cursor = 'pointer';
    c.addEventListener('mousemove', (e) => { const i = at(e); showTip(i, e.clientX, e.clientY); preview(new Set([i])); });
    c.addEventListener('mouseleave', () => { hideTip(); preview(null); });
    c.addEventListener('click', (e) => { const i = at(e); clickResidue(i, e); });
    return c;
  }

  // The substitution map of a mutational scan: residues across, amino acids down.
  function scanMap() {
    const c = document.createElement('canvas');
    c.className = 'scan';
    const n = N, rows = 20, w = 268, cellH = 5, h = rows * cellH, dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = w * dpr; c.height = h * dpr;
    const off = document.createElement('canvas');
    off.width = n; off.height = rows;
    const img = new ImageData(n, rows);
    const sc = { lo: scores.scale.lo, hi: scores.scale.hi, higherIsWorse: scores.higherIsWorse };
    const seq = meta.sequence || '';
    for (let i = 0; i < n; i++) for (let k = 0; k < rows; k++) {
      const row = scores.matrix[i];
      const v = row ? row[k] : null;
      const rgb = C.scoreColor(v, sc, scores.stops, scores.none);
      const o = 4 * (k * n + i);
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
    off.getContext('2d').putImageData(img, 0, 0);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, 0, 0, w, h);
    // Wild-type cells, when a column is wide enough to mark.
    if (w / n >= 2.5) {
      g.fillStyle = role('ground');
      for (let i = 0; i < n; i++) {
        const k = scores.aa.indexOf(seq[i]);
        if (k >= 0) g.fillRect((i + 0.5) * w / n - 0.75, (k + 0.5) * cellH - 0.75, 1.5, 1.5);
      }
    }
    const at = (e) => {
      const r = c.getBoundingClientRect();
      return [Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n))),
        Math.max(0, Math.min(rows - 1, Math.floor((e.clientY - r.top) / r.height * rows)))];
    };
    c.addEventListener('mousemove', (e) => {
      const [i, k] = at(e);
      const v = scores.matrix[i] && scores.matrix[i][k];
      const name = (seq[i] || '?') + meta.labels.number[i] + meta.labels.icode[i] + scores.aa[k];
      tipText(name, (v === null || v === undefined ? 'no score' : scores.column + ' ' + fmt(v)) +
        (scores.values[i] !== null ? ' · position mean ' + fmt(scores.values[i]) : ''), e.clientX, e.clientY);
      preview(new Set([i]));
    });
    c.addEventListener('mouseleave', () => { hideTip(); preview(null); });
    c.addEventListener('click', (e) => { const [i] = at(e); clickResidue(i, e); });
    return c;
  }

  // The PAE map: N×N pixels scaled without smoothing. Hover reads a cell; a click selects its
  // two residues; a drag selects two ranges and reports the mean error between them.
  // Chain IDs in the order a predictor numbers chains: protein chains as they appear, then
  // ligand chains not already named.
  function chainNames() {
    const out = [];
    for (const c of meta.labels.chain) if (!out.includes(c)) out.push(c);
    for (const l of ligands) if (!out.includes(l.chain)) out.push(l.chain);
    return out.map((c) => c.trim() || '·');
  }

  function download(name, bytes, type) {
    const a = document.createElement('a');
    a.download = name;
    a.href = URL.createObjectURL(new Blob([bytes], { type }));
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function drawPae() {
    const c = $('pae');
    if (!c || !pae) return;
    const n = pae.n, px = 268, dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = px * dpr; c.height = px * dpr;
    const img = new ImageData(n, n);
    for (let i = 0; i < n * n; i++) {
      const rgb = C.paeColor(pae.v[i] / 8, pae.max);
      img.data[4 * i] = rgb[0]; img.data[4 * i + 1] = rgb[1]; img.data[4 * i + 2] = rgb[2]; img.data[4 * i + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = n; off.height = n;
    off.getContext('2d').putImageData(img, 0, 0);
    const g = c.getContext('2d');
    let box = null;
    const paint = () => {
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(off, 0, 0, px, px);
      // Chain boundaries.
      g.strokeStyle = role('warm'); g.lineWidth = 1;
      const chainOf = (i) => (i < N ? meta.labels.chain[i] : 'ligand ' + (i - N));
      for (let i = 1; i < n; i++) if (chainOf(i) !== chainOf(i - 1)) {
        const t = i / n * px;
        g.beginPath(); g.moveTo(t, 0); g.lineTo(t, px); g.moveTo(0, t); g.lineTo(px, t); g.stroke();
      }
      if (box) {
        const [i0, j0, i1, j1] = box;
        g.strokeStyle = role('accent'); g.lineWidth = 1.5;
        g.strokeRect(Math.min(j0, j1) / n * px, Math.min(i0, i1) / n * px, (Math.abs(j1 - j0) + 1) / n * px, (Math.abs(i1 - i0) + 1) / n * px);
      }
    };
    paint();
    const cell = (e) => {
      const r = c.getBoundingClientRect();
      const j = Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n)));
      const i = Math.max(0, Math.min(n - 1, Math.floor((e.clientY - r.top) / r.height * n)));
      return [i, j];
    };
    let drag = null;
    c.addEventListener('mousedown', (e) => { drag = cell(e); box = [drag[0], drag[1], drag[0], drag[1]]; paint(); e.preventDefault(); });
    c.addEventListener('mousemove', (e) => {
      const [i, j] = cell(e);
      if (drag) { box = [drag[0], drag[1], i, j]; paint(); }
      tipText('aligned ' + shortLabel(i) + ' · scored ' + shortLabel(j), paeAt(i, j).toFixed(1) + ' Å (reverse ' + paeAt(j, i).toFixed(1) + ' Å)', e.clientX, e.clientY);
      preview(new Set([i, j]));
    });
    c.addEventListener('mouseleave', () => { hideTip(); preview(null); });
    window.addEventListener('mouseup', (e) => {
      if (!drag) return;
      const [i, j] = cell(e);
      const [i0, j0] = drag;
      drag = null;
      if (i === i0 && j === j0) {
        box = null; paint();
        select(new Set([i, j]), { focus: true });
        return;
      }
      const rows = [Math.min(i0, i), Math.max(i0, i)], cols = [Math.min(j0, j), Math.max(j0, j)];
      const set = new Set();
      for (let r = rows[0]; r <= rows[1]; r++) set.add(r);
      for (let r = cols[0]; r <= cols[1]; r++) set.add(r);
      select(set, { pae: { rows, cols } });
    });
  }

  // ---------------------------------------------------------------- sequence track
  function buildSequence() {
    const track = $('seq');
    if (!track || !N) return;
    const letters = meta.sequence || '';
    const frag = document.createDocumentFragment();
    const ssClass = { H: 'h', G: 'h', I: 'h', E: 'e', B: 'e' };
    seqSpans = new Array(N);
    for (let i = 0; i < N; i++) {
      if (i > 0 && meta.labels.chain[i] !== meta.labels.chain[i - 1]) {
        const gap = document.createElement('span');
        gap.className = 'chain';
        gap.textContent = meta.labels.chain[i] || '·';
        frag.append(gap);
      } else if (i === 0 && meta.labels.chain[0].trim()) {
        const gap = document.createElement('span');
        gap.className = 'chain';
        gap.textContent = meta.labels.chain[0];
        frag.append(gap);
      }
      const s = document.createElement('span');
      s.className = 'r ' + (ssClass[meta.dssp[i]] || 'c');
      s.textContent = letters[i] || 'X';
      s.dataset.i = i;
      if (meta.labels.number[i] % 10 === 0) s.dataset.n = meta.labels.number[i];
      seqSpans[i] = s;
      frag.append(s);
    }
    track.append(frag);
    track.addEventListener('mouseover', (e) => {
      const t = e.target;
      if (!t.dataset || t.dataset.i === undefined) return;
      const i = +t.dataset.i;
      const r = t.getBoundingClientRect();
      showTip(i, r.left, r.top - 60);
      preview(new Set([i]));
    });
    track.addEventListener('mouseleave', () => { hideTip(); preview(null); });
    track.addEventListener('click', (e) => {
      const t = e.target;
      if (!t.dataset || t.dataset.i === undefined) return;
      clickResidue(+t.dataset.i, e);
    });
  }
  function markSequence(set) {
    for (const i of marked) if (seqSpans[i]) seqSpans[i].classList.remove('on');
    marked = new Set();
    for (const i of set) if (i < N && seqSpans[i]) { seqSpans[i].classList.add('on'); marked.add(i); }
    // Bring the first selected residue into view when the selection did not come from the track.
    const first = [...set].filter((i) => i < N).sort((a, b) => a - b)[0];
    if (first !== undefined && seqSpans[first] && !sel.preview) {
      const track = $('seq'), s = seqSpans[first];
      if (s.offsetLeft < track.scrollLeft || s.offsetLeft > track.scrollLeft + track.clientWidth - 20) track.scrollLeft = s.offsetLeft - track.clientWidth / 3;
    }
  }
})();
