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
    // brand::structure: Clay, Tide, pale cream.
    return code === 0 ? [216, 166, 100] : code === 1 ? [79, 154, 148] : [231, 233, 200];
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

  const MAGIC = 'PRMESH2\0';

  /** Decode the blob written by web.rs `encode_page`. Typed arrays view the buffer. */
  function parseMesh(bytes) {
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const dv = new DataView(buf);
    let magic = '';
    for (let i = 0; i < 8; i++) magic += String.fromCharCode(dv.getUint8(i));
    if (magic !== MAGIC) throw new Error('unknown mesh format ' + JSON.stringify(magic));
    const h = (k) => dv.getUint32(8 + 4 * k, true);
    const [nv, nt, dsNv, dsNt, nAtoms, nBonds, refNv, refNt] = [0, 1, 2, 3, 4, 5, 6, 7].map(h);
    const lo = [0, 1, 2].map((k) => dv.getFloat32(40 + 4 * k, true));
    const span = [0, 1, 2].map((k) => dv.getFloat32(52 + 4 * k, true));
    let o = 64;
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
    const atoms = { n: nAtoms, nBonds };
    const q = new Uint16Array(buf, o, nAtoms * 3); o = up4(o + nAtoms * 6);
    atoms.pos = new Float32Array(nAtoms * 3);
    for (let i = 0; i < nAtoms * 3; i++) atoms.pos[i] = lo[i % 3] + q[i] / 65535 * span[i % 3];
    atoms.el = new Uint8Array(buf, o, nAtoms); o = up4(o + nAtoms);
    atoms.res = new Uint32Array(buf, o, nAtoms); o += nAtoms * 4;
    atoms.bonds = new Uint32Array(buf, o, nBonds * 2); o += nBonds * 8;
    const reference = mesh(refNv, refNt, true);
    if (o !== buf.byteLength) throw new Error('mesh blob has ' + (buf.byteLength - o) + ' trailing bytes');
    return { lo, span, ribbon, disulfides, atoms, reference };
  }

  // ColorBrewer Greens, reversed: AlphaFold DB's PAE scale (0 Å darkest, max lightest).
  const GREENS_R = [[0, 68, 27], [0, 109, 44], [35, 139, 69], [65, 171, 93], [116, 196, 118],
    [161, 217, 155], [199, 233, 192], [229, 245, 224], [247, 252, 245]];

  /** Colour of a PAE value on AlphaFold DB's scale, 0 → max Å. */
  function paeColor(v, max) {
    const t = clamp01(v / (max > 0 ? max : 31.75)) * (GREENS_R.length - 1);
    const i = Math.min(Math.floor(t), GREENS_R.length - 2);
    const a = GREENS_R[i], b = GREENS_R[i + 1], u = t - i;
    return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * u));
  }

  // Element colours: Jmol's CPK for heteroatoms. Carbon takes its residue's colour (as ChimeraX
  // does), so the sticks read as part of the ribbon they hang off.
  const ELEMENT_RGB = { N: [48, 80, 248], O: [255, 13, 13], S: [255, 200, 50], P: [255, 128, 0],
    SE: [255, 161, 0], H: [255, 255, 255], X: [255, 20, 147] };

  /** Unit vectors perpendicular to d (length > 0), for building tubes. */
  function frame(d) {
    const a = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let u = [d[1] * a[2] - d[2] * a[1], d[2] * a[0] - d[0] * a[2], d[0] * a[1] - d[1] * a[0]];
    const lu = Math.hypot(u[0], u[1], u[2]);
    u = u.map((x) => x / lu);
    const v = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
    return [u, v];
  }

  /**
   * A growable triangle mesh in world coordinates: Float32 positions and normals, u8 colours,
   * a float residue id per vertex (for picking), u32 indices.
   */
  function meshBuilder() {
    const P = [], N = [], Cl = [], R = [], I = [];
    return {
      P, N, Cl, R, I,
      vertex(p, n, c, r) { P.push(p[0], p[1], p[2]); N.push(n[0], n[1], n[2]); Cl.push(c[0], c[1], c[2]); R.push(r); return R.length - 1; },
      /** Open cylinder from a to b. */
      tube(a, b, radius, c, r, sides) {
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const len = Math.hypot(d[0], d[1], d[2]);
        if (len < 1e-4) return;
        const dn = d.map((x) => x / len);
        const [u, v] = frame(dn);
        const base = R.length;
        for (let s = 0; s < sides; s++) {
          const t = 2 * Math.PI * s / sides, cs = Math.cos(t), sn = Math.sin(t);
          const n = [0, 1, 2].map((k) => u[k] * cs + v[k] * sn);
          this.vertex(a.map((x, k) => x + n[k] * radius), n, c, r);
          this.vertex(b.map((x, k) => x + n[k] * radius), n, c, r);
        }
        for (let s = 0; s < sides; s++) {
          const i0 = base + 2 * s, i1 = base + 2 * ((s + 1) % sides);
          I.push(i0, i0 + 1, i1, i1, i0 + 1, i1 + 1);
        }
      },
      /** Low-poly sphere (lat-long, 6 × 8). */
      sphere(p, radius, c, r) {
        const base = R.length, rings = 6, segs = 8;
        for (let i = 0; i <= rings; i++) {
          const th = Math.PI * i / rings;
          for (let j = 0; j < segs; j++) {
            const ph = 2 * Math.PI * j / segs;
            const n = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
            this.vertex(p.map((x, k) => x + n[k] * radius), n, c, r);
          }
        }
        for (let i = 0; i < rings; i++) for (let j = 0; j < segs; j++) {
          const a = base + i * segs + j, b = base + i * segs + (j + 1) % segs;
          I.push(a, a + segs, b, b, a + segs, b + segs);
        }
      },
      /** Dashed tube: dash and gap lengths in Å. */
      dashes(a, b, radius, dash, gap, c, r) {
        const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const len = Math.hypot(d[0], d[1], d[2]);
        if (len < 1e-4) return;
        for (let t = 0; t < len; t += dash + gap) {
          const e = Math.min(t + dash, len);
          this.tube(a.map((x, k) => x + d[k] * t / len), a.map((x, k) => x + d[k] * e / len), radius, c, r, 6);
        }
      },
      build() {
        return { n: R.length, nt: I.length / 3, pos: Float32Array.from(P), nrm: Float32Array.from(N),
          col: Uint8Array.from(Cl), res: Float32Array.from(R), idx: Uint32Array.from(I) };
      },
    };
  }

  /**
   * Stick geometry for the atoms whose residue is in `show` (a Set of residue ids): half-bonds
   * coloured by their atom, a sphere at every atom (a bigger one for an unbonded atom, an ion).
   * `carbon(res)` gives a carbon's colour; `elements` maps element codes to symbols.
   */
  function stickMesh(atoms, elements, show, carbon, radius) {
    const b = meshBuilder();
    const at = (i) => [atoms.pos[3 * i], atoms.pos[3 * i + 1], atoms.pos[3 * i + 2]];
    const colour = (i) => {
      const sym = elements[atoms.el[i]];
      return sym === 'C' ? carbon(atoms.res[i]) : ELEMENT_RGB[sym] || ELEMENT_RGB.X;
    };
    const bonded = new Uint8Array(atoms.n);
    for (let k = 0; k < atoms.nBonds; k++) {
      const i = atoms.bonds[2 * k], j = atoms.bonds[2 * k + 1];
      if (!show.has(atoms.res[i]) || !show.has(atoms.res[j])) continue;
      bonded[i] = bonded[j] = 1;
      const a = at(i), c = at(j), m = [0, 1, 2].map((q) => (a[q] + c[q]) / 2);
      b.tube(a, m, radius, colour(i), atoms.res[i], 8);
      b.tube(m, c, radius, colour(j), atoms.res[j], 8);
    }
    for (let i = 0; i < atoms.n; i++) {
      if (!show.has(atoms.res[i])) continue;
      b.sphere(at(i), bonded[i] ? radius : radius * 3, colour(i), atoms.res[i]);
    }
    return b.build();
  }

  /** Residue ids with any atom within `cutoff` Å of an atom of a residue in `seed`. */
  function neighbours(atoms, seed, cutoff) {
    const out = new Set(seed);
    const src = [];
    for (let i = 0; i < atoms.n; i++) if (seed.has(atoms.res[i])) src.push(i);
    if (!src.length) return out;
    const c2 = cutoff * cutoff;
    // A coarse box test first: most atoms are nowhere near the seed.
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of src) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], atoms.pos[3 * i + k]); hi[k] = Math.max(hi[k], atoms.pos[3 * i + k]);
    }
    for (let j = 0; j < atoms.n; j++) {
      if (out.has(atoms.res[j])) continue;
      const x = atoms.pos[3 * j], y = atoms.pos[3 * j + 1], z = atoms.pos[3 * j + 2];
      if (x < lo[0] - cutoff || x > hi[0] + cutoff || y < lo[1] - cutoff || y > hi[1] + cutoff ||
        z < lo[2] - cutoff || z > hi[2] + cutoff) continue;
      for (const i of src) {
        const dx = x - atoms.pos[3 * i], dy = y - atoms.pos[3 * i + 1], dz = z - atoms.pos[3 * i + 2];
        if (dx * dx + dy * dy + dz * dz <= c2) { out.add(atoms.res[j]); break; }
      }
    }
    return out;
  }

  /** Shortest heavy-atom distance between two residues, with the two atom indices. */
  function closestAtoms(atoms, ra, rb) {
    let best = [Infinity, -1, -1];
    for (let i = 0; i < atoms.n; i++) {
      if (atoms.res[i] !== ra) continue;
      for (let j = 0; j < atoms.n; j++) {
        if (atoms.res[j] !== rb) continue;
        const d = Math.hypot(atoms.pos[3 * i] - atoms.pos[3 * j], atoms.pos[3 * i + 1] - atoms.pos[3 * j + 1],
          atoms.pos[3 * i + 2] - atoms.pos[3 * j + 2]);
        if (d < best[0]) best = [d, i, j];
      }
    }
    return best;
  }

  /** PyMOL selection for residues given as {chain, number, icode} labels. */
  function pymolSelection(items) {
    const byChain = new Map();
    for (const { chain, number, icode } of items) {
      const key = chain.trim();
      if (!byChain.has(key)) byChain.set(key, []);
      byChain.get(key).push(String(number).replace('-', '\\-') + (icode || ''));
    }
    const parts = [...byChain].map(([c, rs]) => (c ? 'chain ' + c + ' and ' : '') + 'resi ' + rs.join('+'));
    return parts.length === 1 ? parts[0] : parts.map((p) => '(' + p + ')').join(' or ');
  }

  /**
   * shader.rs ScoreScale::color: `scale` {lo, hi, higherIsWorse}, `stops` from the damaging end.
   * Used for the mutational-scan map; the ribbon takes the per-residue colours Rust computed.
   */
  function scoreColor(v, scale, stops, none) {
    if (v === null || v === undefined || !isFinite(v)) return none;
    let t = f(clamp01((v - scale.lo) / (scale.hi - scale.lo)));
    if (scale.higherIsWorse) t = f(1 - t);
    const x = f(t * (stops.length - 1));
    const i = Math.min(Math.floor(x), stops.length - 2);
    return lerp(stops[i], stops[i + 1], f(x - i));
  }

  /** Per-vertex RGB (u8) for a scheme, exactly as pipeline.rs picks it. */
  function vertexColors(ribbon, scheme, residueColors) {
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
        : scheme === 'score' && residueColors ? residueColors[ribbon.res[i]] || [96, 96, 90]
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

  // A measurement line for a narrow panel: a number stays with the word before it and with its
  // unit ("coil 32 %", never "coil 32" and a lone "%"), and a " · " separator never starts a
  // line. Lines still wrap at the separators and between other words.
  function keepUnits(s) {
    const NB = '\u00a0';
    return s
      .replace(/(\S) (?=\d)/g, '$1' + NB)
      .replace(/(\d) (?=\S)/g, '$1' + NB)
      .replace(/ · /g, NB + '· ');
  }

  // ---------------------------------------------------------------- geometry of measurements
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);

  /** Distance (Å), angle at the middle point (°) or dihedral (°, IUPAC sign) of 2–4 points. */
  function measure(pts) {
    if (pts.length === 2) return norm(sub(pts[0], pts[1]));
    if (pts.length === 3) {
      const u = sub(pts[0], pts[1]), v = sub(pts[2], pts[1]);
      return Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (norm(u) * norm(v))))) * 180 / Math.PI;
    }
    if (pts.length === 4) {
      const b0 = sub(pts[0], pts[1]), b1 = sub(pts[2], pts[1]), b2 = sub(pts[3], pts[2]);
      const n1 = norm(b1), b1n = b1.map((x) => x / n1);
      const v = sub(b0, b1n.map((x) => x * dot(b0, b1n)));
      const w = sub(b2, b1n.map((x) => x * dot(b2, b1n)));
      return Math.atan2(dot(cross(b1n, v), w), dot(v, w)) * 180 / Math.PI;
    }
    return NaN;
  }

  // ---------------------------------------------------------------- surfaces
  // Kyte & Doolittle (1982) hydropathy.
  const KYTE_DOOLITTLE = { ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5, MET: 1.9, ALA: 1.8, GLY: -0.4,
    THR: -0.7, SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6, HIS: -3.2, GLU: -3.5, GLN: -3.5, ASP: -3.5,
    ASN: -3.5, LYS: -3.9, ARG: -4.5 };
  // van der Waals radii (Bondi 1964) by element code order of atoms.rs ELEMENTS.
  const VDW = { C: 1.7, N: 1.55, O: 1.52, S: 1.8, P: 1.8, SE: 1.9, H: 1.1, X: 1.8 };

  /**
   * Formal charges at pH 7: Lys NZ +1, Arg NH1/NH2 +½ each, Asp OD1/OD2 and Glu OE1/OE2 −½
   * each, the first N of a chain +1, a terminal OXT −1. Histidine neutral. `names`/`resNames`
   * per atom. Returns [atomIndex, charge] pairs.
   */
  function formalCharges(names, resNames, atomRes) {
    const q = [];
    for (let i = 0; i < names.length; i++) {
      const a = names[i], r = resNames[i];
      if (r === 'LYS' && a === 'NZ') q.push([i, 1]);
      else if (r === 'ARG' && (a === 'NH1' || a === 'NH2')) q.push([i, 0.5]);
      else if (r === 'ASP' && (a === 'OD1' || a === 'OD2')) q.push([i, -0.5]);
      else if (r === 'GLU' && (a === 'OE1' || a === 'OE2')) q.push([i, -0.5]);
      else if (a === 'OXT') q.push([i, -1]);
      else if (a === 'N' && (i === 0 || atomRes[i - 1] !== atomRes[i] - 1 && atomRes[i - 1] !== atomRes[i])) q.push([i, 1]);
    }
    return q;
  }

  /**
   * Coulombic potential (kcal/mol/e) at `p` from point charges with a distance-dependent
   * dielectric ε = 4r, as ChimeraX's `coulombic` defaults to: φ = 332 Σ qᵢ / (4 rᵢ²).
   */
  function coulomb(p, pos, charges) {
    let phi = 0;
    for (const [i, q] of charges) {
      const dx = p[0] - pos[3 * i], dy = p[1] - pos[3 * i + 1], dz = p[2] - pos[3 * i + 2];
      const r2 = Math.max(dx * dx + dy * dy + dz * dz, 1);
      phi += q / (4 * r2);
    }
    return 332 * phi;
  }

  /**
   * A molecular surface: the level set of a sum of atom Gaussians (Grant & Pickup 1995, the
   * idea behind VMD QuickSurf), meshed with naive surface nets. `which(i)` keeps atom i.
   * Returns world-space positions, normals (the density gradient), and the nearest atom of
   * each vertex (for colouring).
   */
  function gaussianSurface(atoms, elements, which, spacing) {
    const idx = [];
    for (let i = 0; i < atoms.n; i++) if (which(i)) idx.push(i);
    if (!idx.length) return { n: 0, nt: 0, pos: new Float32Array(0), nrm: new Float32Array(0), atom: new Uint32Array(0), idx: new Uint32Array(0) };
    const h = spacing || 1.0, pad = 5;
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of idx) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], atoms.pos[3 * i + k]); hi[k] = Math.max(hi[k], atoms.pos[3 * i + k]);
    }
    lo = lo.map((x) => x - pad);
    const dim = [0, 1, 2].map((k) => Math.ceil((hi[k] + pad - lo[k]) / h) + 1);
    const [nx, ny, nz] = dim;
    const rho = new Float32Array(nx * ny * nz);
    const near = new Int32Array(nx * ny * nz).fill(-1);
    const best = new Float32Array(nx * ny * nz).fill(Infinity);
    const at = (x, y, z) => x + nx * (y + ny * z);
    // Grant & Pickup (1995) Gaussian density: exp(-B (d²/r² - 1)), which is 1 exactly at the van
    // der Waals radius of a lone atom; B = 2.3 gives a smooth molecular-like surface.
    const B = 2.3;
    for (const i of idx) {
      const r = VDW[elements[atoms.el[i]]] || 1.8, cut = r * 2;
      const p = [atoms.pos[3 * i], atoms.pos[3 * i + 1], atoms.pos[3 * i + 2]];
      const g0 = p.map((x, k) => Math.max(0, Math.floor((x - cut - lo[k]) / h)));
      const g1 = p.map((x, k) => Math.min(dim[k] - 1, Math.ceil((x + cut - lo[k]) / h)));
      for (let z = g0[2]; z <= g1[2]; z++) for (let y = g0[1]; y <= g1[1]; y++) for (let x = g0[0]; x <= g1[0]; x++) {
        const dx = lo[0] + x * h - p[0], dy = lo[1] + y * h - p[1], dz = lo[2] + z * h - p[2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > cut * cut) continue;
        const v = at(x, y, z);
        rho[v] += Math.exp(-B * (d2 / (r * r) - 1));
        const e = d2 / (r * r);
        if (e < best[v]) { best[v] = e; near[v] = i; }
      }
    }
    const iso = 1.0;
    // Surface nets: one vertex per cell that the level set crosses, at the mean of its edge
    // crossings; one quad per crossing grid edge.
    const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
    const cid = (x, y, z) => x + (nx - 1) * (y + (ny - 1) * z);
    const P = [], Nm = [], A = [], I = [];
    const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    const grad = (x, y, z) => {
      const g = (a, b, c) => rho[at(Math.max(0, Math.min(nx - 1, a)), Math.max(0, Math.min(ny - 1, b)), Math.max(0, Math.min(nz - 1, c)))];
      const v = [g(x - 1, y, z) - g(x + 1, y, z), g(x, y - 1, z) - g(x, y + 1, z), g(x, y, z - 1) - g(x, y, z + 1)];
      const l = Math.hypot(v[0], v[1], v[2]) || 1;
      return v.map((c) => c / l);
    };
    for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
      const val = corners.map(([a, b, c]) => rho[at(x + a, y + b, z + c)] - iso);
      let inside = 0;
      for (const v of val) if (v > 0) inside++;
      if (inside === 0 || inside === 8) continue;
      let sx = 0, sy = 0, sz = 0, cnt = 0;
      for (const [a, b] of edges) {
        if ((val[a] > 0) === (val[b] > 0)) continue;
        const t = val[a] / (val[a] - val[b]);
        const ca = corners[a], cb = corners[b];
        sx += ca[0] + (cb[0] - ca[0]) * t; sy += ca[1] + (cb[1] - ca[1]) * t; sz += ca[2] + (cb[2] - ca[2]) * t;
        cnt++;
      }
      const fx = x + sx / cnt, fy = y + sy / cnt, fz = z + sz / cnt;
      cellVert[cid(x, y, z)] = A.length;
      P.push(lo[0] + fx * h, lo[1] + fy * h, lo[2] + fz * h);
      const gx = Math.round(fx), gy = Math.round(fy), gz = Math.round(fz);
      const n = grad(gx, gy, gz);
      Nm.push(n[0], n[1], n[2]);
      let a = near[at(gx, gy, gz)];
      if (a < 0) for (const [p, q, r] of corners) { const v = near[at(Math.min(nx - 1, x + p), Math.min(ny - 1, y + q), Math.min(nz - 1, z + r))]; if (v >= 0) { a = v; break; } }
      A.push(Math.max(a, 0));
    }
    // Quads: for each grid edge crossing the level, the four cells around it.
    for (let z = 1; z < nz - 1; z++) for (let y = 1; y < ny - 1; y++) for (let x = 1; x < nx - 1; x++) {
      const v0 = rho[at(x, y, z)] > iso;
      for (let axis = 0; axis < 3; axis++) {
        const [dx, dy, dz] = axis === 0 ? [1, 0, 0] : axis === 1 ? [0, 1, 0] : [0, 0, 1];
        if (x + dx >= nx || y + dy >= ny || z + dz >= nz) continue;
        const v1 = rho[at(x + dx, y + dy, z + dz)] > iso;
        if (v0 === v1) continue;
        // The four cells sharing this edge.
        const cells = axis === 0 ? [[x, y - 1, z - 1], [x, y, z - 1], [x, y, z], [x, y - 1, z]]
          : axis === 1 ? [[x - 1, y, z - 1], [x - 1, y, z], [x, y, z], [x, y, z - 1]]
          : [[x - 1, y - 1, z], [x, y - 1, z], [x, y, z], [x - 1, y, z]];
        const q = cells.map(([a, b, c]) => cellVert[cid(a, b, c)]);
        if (q.some((v) => v < 0)) continue;
        if (v0) I.push(q[0], q[1], q[2], q[0], q[2], q[3]);
        else I.push(q[0], q[2], q[1], q[0], q[3], q[2]);
      }
    }
    return { n: A.length, nt: I.length / 3, pos: Float32Array.from(P), nrm: Float32Array.from(Nm),
      atom: Uint32Array.from(A), idx: Uint32Array.from(I) };
  }

  // ---------------------------------------------------------------- sessions
  /** View state ↔ URL fragment (`#v=` base64url JSON), so a link reopens the same view. */
  function encodeSession(obj) {
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    let bin = '';
    for (const b of bytes) bin += String.fromCharCode(b);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function decodeSession(text) {
    try {
      const b64 = text.replace(/-/g, '+').replace(/_/g, '/');
      const bin = atob(b64 + '==='.slice((b64.length + 3) % 4));
      const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
      const v = JSON.parse(new TextDecoder().decode(bytes));
      return v && typeof v === 'object' ? v : null;
    } catch (_) {
      return null;
    }
  }

  const api = { asU8, keepUnits, lerp, plddtColor, ssColor, rainbowColor, b64ToBytes, gunzip, parseMesh,
    vertexColors, fitScale, mat3Mul, yawPitch, MAGIC, paeColor, ELEMENT_RGB, meshBuilder, stickMesh,
    neighbours, closestAtoms, pymolSelection, scoreColor, measure, gaussianSurface, KYTE_DOOLITTLE,
    formalCharges, coulomb, encodeSession, decodeSession };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ProteusCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
