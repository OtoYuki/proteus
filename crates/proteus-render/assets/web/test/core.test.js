// Parity tests for core.js against fixtures written by the Rust side (web.rs tests).
// Run: node --test crates/proteus-render/assets/web/test/core.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const C = require('../core.js');

const here = (f) => path.join(__dirname, f);
const colors = JSON.parse(fs.readFileSync(here('colors.json'), 'utf8'));

test('pLDDT colours match plddt_to_color exactly', () => {
  for (const [p, r, g, b] of colors.plddt) assert.deepEqual(C.plddtColor(p), [r, g, b], `pLDDT ${p}`);
});

test('secondary-structure colours match secondary_structure_to_color', () => {
  for (const [code, r, g, b] of colors.ss) assert.deepEqual(C.ssColor(code), [r, g, b], `code ${code}`);
});

test('score colours match ScoreScale::color exactly', () => {
  for (const [v, lo, hi, worse, r, g, b] of colors.score) {
    assert.deepEqual(C.scoreColor(v, { lo, hi, higherIsWorse: worse }, colors.scoreStops, [96, 96, 90]), [r, g, b], `${v} in ${lo}..${hi}`);
  }
  assert.deepEqual(C.scoreColor(null, { lo: 0, hi: 1 }, colors.scoreStops, [96, 96, 90]), [96, 96, 90]);
});

test('rainbow colours match rainbow_color exactly', () => {
  for (const [i, n, r, g, b] of colors.rainbow) assert.deepEqual(C.rainbowColor(i, n), [r, g, b], `${i}/${n}`);
});

test('the 1CRN blob written by Rust decodes to the same mesh', async () => {
  const want = JSON.parse(fs.readFileSync(here('1crn.mesh.json'), 'utf8'));
  const mesh = C.parseMesh(await C.gunzip(fs.readFileSync(here('1crn.mesh.gz'))));
  const r = mesh.ribbon;
  assert.equal(r.n, want.nv);
  assert.equal(r.nt, want.nt);
  assert.equal(mesh.disulfides.n, want.dsNv);
  assert.deepEqual([...r.idx.slice(0, 3)], want.firstIndex);
  assert.equal(r.res[r.n - 1], want.lastResidue);
  assert.equal(r.plddt[0], want.firstPlddt);
  for (const m of [r, mesh.disulfides]) {
    for (let i = 0; i < m.idx.length; i++) assert.ok(m.idx[i] < m.n, `index ${i} out of range`);
  }
  for (let i = 0; i < r.n; i++) assert.ok(r.ss[i] <= 2);
  assert.equal(C.vertexColors(r, 'ss').length, r.n * 3);
  // Heavy atoms and bonds ride in the same blob.
  const a = mesh.atoms;
  assert.equal(a.n, want.atoms);
  assert.equal(a.nBonds, want.bonds);
  assert.deepEqual([a.bonds[0], a.bonds[1]], want.firstBond);
  for (let k = 0; k < a.nBonds; k++) {
    const i = a.bonds[2 * k], j = a.bonds[2 * k + 1];
    const d = Math.hypot(a.pos[3 * i] - a.pos[3 * j], a.pos[3 * i + 1] - a.pos[3 * j + 1], a.pos[3 * i + 2] - a.pos[3 * j + 2]);
    assert.ok(d > 1.1 && d < 2.2, `bond ${k} is ${d} Å`);
  }
  assert.equal(mesh.reference.n, 0);
});

test('the PAE scale is AlphaFold DB\'s: dark green at 0, near white at the maximum', () => {
  assert.deepEqual(C.paeColor(0, 31.75), [0, 68, 27]);
  assert.deepEqual(C.paeColor(31.75, 31.75), [247, 252, 245]);
  assert.deepEqual(C.paeColor(99, 31.75), [247, 252, 245]);
  const mid = C.paeColor(15, 31.75);
  assert.ok(mid[1] > 68 && mid[1] < 252);
});

test('sticks, neighbours and closest atoms on crambin', async () => {
  const mesh = C.parseMesh(await C.gunzip(fs.readFileSync(here('1crn.mesh.gz'))));
  const a = mesh.atoms;
  const els = ['C', 'N', 'O', 'S', 'P', 'SE', 'H', 'X'];
  // Cys3 and Cys40 form a disulfide (0-based residues 2 and 39).
  const sticks = C.stickMesh(a, els, new Set([2, 39]), () => [200, 200, 200], 0.2);
  assert.ok(sticks.nt > 0);
  for (let i = 0; i < sticks.idx.length; i++) assert.ok(sticks.idx[i] < sticks.n);
  for (let i = 0; i < sticks.n; i++) assert.ok(sticks.res[i] === 2 || sticks.res[i] === 39);
  const near = C.neighbours(a, new Set([2]), 5);
  assert.ok(near.has(2) && near.has(39), 'the disulfide partner is within 5 Å');
  assert.ok(near.size < 20);
  const [d] = C.closestAtoms(a, 2, 39);
  assert.ok(d > 1.9 && d < 2.2, `SG–SG ${d} Å`);
});

test('PyMOL selections name chains and escape negative numbers', () => {
  assert.equal(C.pymolSelection([{ chain: 'A', number: 43, icode: '' }, { chain: 'A', number: 44, icode: 'B' }]),
    'chain A and resi 43+44B');
  assert.equal(C.pymolSelection([{ chain: 'A', number: -2, icode: '' }, { chain: 'B', number: 7, icode: '' }]),
    '(chain A and resi \\-2) or (chain B and resi 7)');
});

test('a blob with the wrong magic or trailing bytes is rejected', async () => {
  const bytes = await C.gunzip(fs.readFileSync(here('1crn.mesh.gz')));
  const bad = bytes.slice(); bad[0] = 0x58;
  assert.throws(() => C.parseMesh(bad), /unknown mesh format/);
  const long = new Uint8Array(bytes.length + 4); long.set(bytes);
  assert.throws(() => C.parseMesh(long), /trailing bytes/);
});

test('fitScale never goes below the bounding-sphere fit', () => {
  const sphere = C.fitScale(800, 600, 20, null, 1);
  assert.equal(sphere, 600 * 0.9 / 40);
  assert.ok(C.fitScale(800, 600, 20, [30, 30], 1) >= sphere);
  assert.equal(C.fitScale(800, 600, 20, [5, 5], 2), 2 * Math.min(800 * 0.9 / 10, 600 * 0.9 / 10));
});

test('yawPitch is a proper rotation', () => {
  const m = C.yawPitch(0.7, -0.4);
  const t = [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
  const id = C.mat3Mul(m, t);
  id.forEach((v, k) => assert.ok(Math.abs(v - (k % 4 === 0 ? 1 : 0)) < 1e-12));
});

test('panel lines keep numbers with their units and never start with a separator', () => {
  const NB = '\u00a0';
  assert.equal(C.keepUnits('helix 25 % · strand 43 % · coil 32 %'),
    `helix${NB}25${NB}%${NB}· strand${NB}43${NB}%${NB}· coil${NB}32${NB}%`);
  assert.equal(C.keepUnits('3733 Å² · hydrophobic burial 95 %'),
    `3733${NB}Å²${NB}· hydrophobic burial${NB}95${NB}%`);
  assert.equal(C.keepUnits('92.9 / 100'), `92.9${NB}/${NB}100`);
  assert.equal(C.keepUnits('no numbers here'), 'no numbers here');
});

test('distances, angles and dihedrals', () => {
  assert.equal(C.measure([[0, 0, 0], [3, 4, 0]]), 5);
  assert.ok(Math.abs(C.measure([[1, 0, 0], [0, 0, 0], [0, 1, 0]]) - 90) < 1e-9);
  // A trans dihedral is 180, a cis 0; sign follows IUPAC.
  assert.ok(Math.abs(Math.abs(C.measure([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, -1, 0]])) - 180) < 1e-9);
  assert.ok(Math.abs(C.measure([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]])) < 1e-9);
  assert.ok(Math.abs(C.measure([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]]) - -90) < 1e-9 ||
    Math.abs(C.measure([[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]]) - 90) < 1e-9);
});

test('a Gaussian surface around one atom is a closed shell near its van der Waals radius', () => {
  const atoms = { n: 1, pos: new Float32Array([0, 0, 0]), el: new Uint8Array([0]), res: new Uint32Array([0]) };
  const s = C.gaussianSurface(atoms, ['C'], () => true, 0.3);
  assert.ok(s.n > 100 && s.nt > 100);
  for (let i = 0; i < s.idx.length; i++) assert.ok(s.idx[i] < s.n);
  let rmin = Infinity, rmax = 0;
  for (let i = 0; i < s.n; i++) {
    const r = Math.hypot(s.pos[3 * i], s.pos[3 * i + 1], s.pos[3 * i + 2]);
    rmin = Math.min(rmin, r); rmax = Math.max(rmax, r);
  }
  assert.ok(rmin > 1.2 && rmax < 2.4, `shell radius ${rmin}–${rmax} Å for C (vdW 1.7)`);
  // Euler characteristic of a sphere: V − E + F = 2.
  const edges = new Set();
  for (let t = 0; t < s.nt; t++) for (let k = 0; k < 3; k++) {
    const a = s.idx[3 * t + k], b = s.idx[3 * t + (k + 1) % 3];
    edges.add(a < b ? a + ',' + b : b + ',' + a);
  }
  assert.equal(s.n - edges.size + s.nt, 2);
});

test('the crambin surface encloses every atom', async () => {
  const mesh = C.parseMesh(await C.gunzip(fs.readFileSync(here('1crn.mesh.gz'))));
  const s = C.gaussianSurface(mesh.atoms, ['C', 'N', 'O', 'S', 'P', 'SE', 'H', 'X'], () => true, 1.0);
  assert.ok(s.nt > 1000);
  for (let i = 0; i < s.atom.length; i++) assert.ok(s.atom[i] < mesh.atoms.n);
});

test('Coulomb potential: positive near a lysine, negative near a glutamate', () => {
  const pos = new Float32Array([0, 0, 0, 10, 0, 0]);
  const q = C.formalCharges(['NZ', 'OE1'], ['LYS', 'GLU'], [0, 1]);
  assert.deepEqual(q, [[0, 1], [1, -0.5]]);
  assert.ok(C.coulomb([1.5, 0, 0], pos, q) > 0);
  assert.ok(C.coulomb([8.5, 0, 0], pos, q) < 0);
});

test('a session survives the URL fragment', () => {
  const v = { yaw: 0.5, sel: [1, 2, 300], scheme: 'plddt', m: [[10, 11]] };
  assert.deepEqual(C.decodeSession(C.encodeSession(v)), v);
  assert.equal(C.decodeSession('not base64 json'), null);
});
