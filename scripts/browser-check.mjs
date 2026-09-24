// Open viewer pages written by `proteus view --html` in real browsers and fail on anything a
// reader would notice: a script error, no WebGL2, a blank canvas, a hover label for a residue
// that does not exist, keys that do nothing, a click that selects nothing, a finding that
// cannot be shown, or a PAE map that is not drawn.
//
//   node scripts/browser-check.mjs page.html [more.html …]
//   BROWSERS=chromium,firefox,webkit (default: all three)
//
// Needs the `playwright` package and its browsers (`npx playwright install --with-deps`).
import { chromium, firefox, webkit } from 'playwright';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const engines = { chromium, firefox, webkit };
const wanted = (process.env.BROWSERS || 'chromium,firefox,webkit').split(',');
const pages = process.argv.slice(2);
if (!pages.length) {
  console.error('usage: node scripts/browser-check.mjs page.html …');
  process.exit(2);
}

let failures = 0;
const fail = (where, what) => { failures++; console.log(`FAIL ${where}: ${what}`); };

for (const name of wanted) {
  const browser = await engines[name].launch();
  for (const file of pages) {
    const where = `${name} ${file}`;
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(pathToFileURL(resolve(file)).href);
    try {
      await page.waitForFunction(() => window.ProteusViewer || !document.getElementById('fallback').hidden, null, { timeout: 30000 });
    } catch {
      fail(where, 'the viewer never started');
    }
    await page.waitForTimeout(500);
    const r = await page.evaluate(() => {
      const V = window.ProteusViewer;
      if (!V) return { started: false };
      const c = document.getElementById('view');
      const gl = c.getContext('webgl2');
      const meta = JSON.parse(document.getElementById('proteus-meta').textContent);
      V.render();
      // Canvas coverage: pixels that differ from the ground colour.
      const px = new Uint8Array(4), g = meta.palette.ground;
      let drawn = 0, samples = 0;
      for (let y = 10; y < c.height; y += 23) for (let x = 10; x < c.width; x += 23) {
        gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        samples++;
        if (Math.abs(px[0] - g[0]) + Math.abs(px[1] - g[1]) + Math.abs(px[2] - g[2]) > 12) drawn++;
      }
      // Picking: every id must be a real residue.
      let valid = 0, invalid = 0;
      const seen = new Set();
      for (let y = 20; y < c.clientHeight; y += 20) for (let x = 20; x < c.clientWidth; x += 20) {
        const i = V.pickAt(x, y);
        if (i < 0) continue;
        // Ligand k is picked as residues + k.
        if (i < meta.residues + meta.ligands.length) { valid++; seen.add(i); } else invalid++;
      }
      return { started: true, drawn, samples, valid, invalid, distinct: seen.size, residues: meta.residues };
    }).catch((e) => ({ started: false, error: e.message.split('\n')[0] }));
    if (r.error) fail(where, 'drawing threw: ' + r.error);
    if (errors.length) fail(where, 'page errors: ' + errors.join(' | '));
    if (!r.started) {
      // No WebGL2 in this browser: the fallback must say so and still show the measurements.
      const fb = await page.evaluate(() => ({ shown: !document.getElementById('fallback').hidden, rows: document.querySelectorAll('#panel dd').length }));
      if (!r.error && fb.shown && fb.rows > 0) console.log(`ok   ${where}: no WebGL2 here; fallback shown with ${fb.rows} measurements`);
      else if (!r.error) fail(where, 'neither the viewer nor a complete fallback');
      await page.close();
      continue;
    }
    if (r.drawn < r.samples * 0.01) fail(where, `blank canvas (${r.drawn}/${r.samples} samples drawn)`);
    if (r.invalid) fail(where, `${r.invalid} picks returned a residue that does not exist`);
    if (r.valid === 0) fail(where, 'picking found no residue at all');
    // Keys: rotate with an arrow, change colours with c.
    await page.locator('#view').focus();
    const before = await page.evaluate(() => [window.ProteusViewer.state.yaw, window.ProteusViewer.scheme]);
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('c');
    const after = await page.evaluate(() => [window.ProteusViewer.state.yaw, window.ProteusViewer.scheme]);
    if (after[0] === before[0]) fail(where, 'ArrowRight did not rotate');
    if (after[1] === before[1]) fail(where, 'c did not change the colours');
    // d answers in the legend, including on a structure with no disulfides to show.
    await page.keyboard.press('d');
    const legend = await page.locator('#legend').textContent();
    if (!/disulfides/.test(legend)) fail(where, `d gave no answer (legend: "${legend}")`);
    // A click on the structure selects the residue under it, which draws sticks and fills the
    // selection box; Escape clears it.
    const hit = await page.evaluate(() => {
      const V = window.ProteusViewer, c = document.getElementById('view');
      for (let y = 40; y < c.clientHeight; y += 17) for (let x = 40; x < c.clientWidth - 320; x += 17) {
        if (V.pickAt(x, y) >= 0) return [x, y];
      }
      return null;
    });
    if (hit) {
      await page.mouse.click(hit[0], hit[1]);
      await page.waitForTimeout(100);
      const s = await page.evaluate(() => ({ n: window.ProteusViewer.sel.set.size, sticks: window.ProteusViewer.sticks,
        box: !document.getElementById('selbox').hidden }));
      if (s.n !== 1 || !s.box) fail(where, `a click selected ${s.n} residues (box shown: ${s.box})`);
      if (!s.sticks) fail(where, 'a selected residue drew no sticks');
      await page.keyboard.press('Escape');
      if (await page.evaluate(() => window.ProteusViewer.sel.set.size)) fail(where, 'Escape did not clear the selection');
    }
    // The first finding with items shows itself: its residues selected, a line drawn for a contact.
    const finding = await page.evaluate(async () => {
      for (const d of document.querySelectorAll('#panel details')) {
        d.open = true;
        await new Promise((r) => setTimeout(r, 30));
        const b = d.querySelector('li button');
        if (!b) continue;
        b.click();
        await new Promise((r) => setTimeout(r, 60));
        return { name: d.querySelector('summary').textContent, n: window.ProteusViewer.sel.set.size };
      }
      return null;
    });
    if (finding && !finding.n) fail(where, `the finding "${finding.name}" selected nothing`);
    // u builds a surface; a measurement is drawn and labelled; the view survives a reload.
    const extra = await page.evaluate(async () => {
      const V = window.ProteusViewer;
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'u' }));
      const tris = V.surfaceTriangles;
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'u' }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'u' }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'u' }));
      V.measures.push([0, 1]);
      V.finishMeasure();
      V.state.zoom = 1.7;
      V.render();
      await new Promise((r) => setTimeout(r, 400));
      return { tris, labels: document.querySelectorAll('#overlay .measure').length, hash: location.hash };
    });
    if (!extra.tris) fail(where, 'u drew no surface');
    if (!extra.labels) fail(where, 'a measurement has no label');
    if (!extra.hash.startsWith('#v=')) fail(where, 'the view is not kept in the URL');
    else {
      await page.reload();
      await page.waitForFunction(() => window.ProteusViewer, null, { timeout: 30000 });
      const back = await page.evaluate(() => ({ zoom: window.ProteusViewer.state.zoom, m: window.ProteusViewer.measures.length }));
      if (back.zoom !== 1.7 || back.m !== 1) fail(where, `the view did not come back from its URL (${JSON.stringify(back)})`);
    }
    // A page with PAE draws the map (not left blank) in the AlphaFold greens.
    const pae = await page.evaluate(() => {
      const c = document.getElementById('pae');
      if (!c) return null;
      const g = c.getContext('2d').getImageData(c.width >> 1, c.height >> 1, 1, 1).data;
      return [g[0], g[1], g[2], g[3]];
    });
    if (pae && !(pae[3] === 255 && pae[1] >= pae[0] && pae[1] >= pae[2])) fail(where, `the PAE map is not drawn (${pae})`);
    console.log(`ok   ${where}: ${r.drawn}/${r.samples} drawn, ${r.distinct} of ${r.residues} residues picked` +
      (finding ? `, "${finding.name}" shown` : '') + (pae ? ', PAE drawn' : ''));
    await page.close();
  }
  await browser.close();
}
if (failures) {
  console.log(`${failures} failure(s)`);
  process.exit(1);
}
