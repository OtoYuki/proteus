// Open viewer pages written by `proteus view --html` in real browsers and fail on anything a
// reader would notice: a script error, no WebGL2, a blank canvas, a hover label for a residue
// that does not exist, or keys that do nothing.
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
        if (i < meta.residues) { valid++; seen.add(i); } else invalid++;
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
    console.log(`ok   ${where}: ${r.drawn}/${r.samples} drawn, ${r.distinct} of ${r.residues} residues picked`);
    await page.close();
  }
  await browser.close();
}
if (failures) {
  console.log(`${failures} failure(s)`);
  process.exit(1);
}
