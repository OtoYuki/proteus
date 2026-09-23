# Our own browser viewer: WebGL2 over `proteus-render`'s geometry

2026-09-23. Replaces the vendored 3Dmol.js page behind `proteus view --html/--web` and the
daemon's `/view/{job}`.

## Why

The browser page was 3Dmol.js with a Proteus caption. Four problems, all observed rather than
supposed:

1. **It is not our picture.** The terminal draws `proteus-render`'s ribbon: carbonyl-oriented
   (Carson & Bugg), flip-corrected, blended across residues, with our DSSP, SSAO and outlines.
   The page drew 3Dmol's own cartoon, so the same structure looked different in the two places
   the product shows it.
2. **Its shading had a defect we had to patch.** 3Dmol's Lambert shader has one light and no
   ambient term; twisted strands rendered black (1PGB). We fixed it by string-patching the
   minified library at page-build time. `shade_blinn_phong` in `proteus-render` has a 0.30
   ambient term and a fill light, so the defect cannot occur in our shading.
3. **Size.** Every page carried 525 KB of 3Dmol.js plus the structure file base64-encoded
   (6VXX: 2.56 MB mmCIF → 3.4 MB).
4. **Ownership.** The one thing in the product a user sees in a browser was someone else's code.

## Facts the design rests on

Measured on this machine (i7-11800H) with `proteus-render` at `d17ae9c`:

| structure | residues | vertices | triangles | mesh build | raw buffers | gzip, 16-bit positions | + base64 |
|---|---|---|---|---|---|---|---|
| 1CRN | 46 | 1 450 | 2 896 | 9 ms | 0.07 MB | 0.02 MB | 0.03 MB |
| 4HHB | 574 | 18 280 | 36 544 | 0.12 s | 0.82 MB | 0.27 MB | 0.36 MB |
| AF spike | 1 273 | 40 714 | 81 424 | 0.31 s | 1.83 MB | 0.61 MB | 0.81 MB |
| 6VXX | 2 916 | 92 520 | 184 896 | 0.71 s | 4.16 MB | 1.37 MB | 1.83 MB |
| 1AON | 8 015 | 256 018 | 511 952 | 2.1 s | 11.5 MB | 3.80 MB | 5.07 MB |

- About 32 vertices and 64 triangles per residue.
- 16-bit positions quantised over the bounding box lose at most 0.0015 Å (1AON), which is
  invisible.
- 8-bit normals are enough for Gouraud shading.

From the sources:
- **WebGL 2.0:** 96.44 % global usage; Chrome 56, Firefox 51, Safari 15 (desktop and iOS),
  Edge 79 (caniuse.com/webgl2, 2026-09-23). It supports `UNSIGNED_INT` indices and sampling depth
  textures without extensions, which 1AON's 256 k vertices and the post-processing pass need.
- **`DecompressionStream`:** "Baseline — widely available … since May 2023" (MDN); 95.8 %; Chrome
  80, Firefox 113, Safari 16.4 (caniuse). The Compression Standard's `CompressionFormat` includes
  `"gzip"` (compression.spec.whatwg.org). So the page can carry gzip and let the browser inflate
  it, with no JavaScript inflate code.
- **Camera** (`rasterizer/camera.rs`): orthographic. `rotation_matrix()` is yaw·pitch·roll·base,
  where `base` is the principal-axis frame from `OrbitCamera::oriented`. The scale fits the
  oriented half-extents to 90 % of the viewport, never below the bounding-sphere fit. The viewer
  is at +Z. Exporting `base`, `center`, `half_extents` and `bounding_radius` lets the page open on
  exactly the terminal's view.
- **Shading** (`rasterizer/shader.rs`, `pipeline.rs`), all Gouraud:
  - Blinn–Phong with a key light (0.5, 0.8, 1), a fill light (−0.6, −0.4, 0.5), ambient 0.30,
    diffuse 0.65 / 0.25, specular (exponent 16) 0.25, intensity clamped at 1.3.
  - Linear depth cueing to 55 % at the back.
  - Post-processing:
    - outlines where a 4-neighbour depth step exceeds 4 Å (colour × 0.35)
    - SSAO from 8 depth samples at 2–3 px, occlusion ≤ 45 %, floor 0.5
- **Colours:** the AlphaFold pLDDT ramp (#FF7D45 → #FFDB13 → #65CBF3 → #0053D6 with linear
  interpolation), secondary structure (helix #D946EF, strand #F59E0B, coil #06B6D4), and a rainbow
  HSV hue from 240° to 0° at s 0.85, v 0.95.

## Design

**Geometry comes from Rust, rendering happens in the browser.** One implementation of the ribbon
(`proteus-render`) feeds both the terminal rasteriser and the page. The browser never
re-implements geometry.

**Page contents:**
- a small JSON header: title, caption, camera, residue labels, the DSSP string, metrics,
  Ramachandran points, per-residue pLDDT or B-factor, the default colour scheme
- one gzip-compressed binary blob, base64-encoded, holding the typed arrays:
  - positions as u16×3, quantised over the bounding box
  - normals as i8×3
  - residue index u32, pLDDT f32 (so the colour is computed from the same value as in Rust),
    secondary structure u8
  - triangle indices u32
  - the same for the disulfide mesh
- our viewer JavaScript and CSS, inline

No network access, no third-party code. `<script>`-context escaping is kept for every string
taken from a structure file (chain ids are free text).

**Renderer** (WebGL2, about 400 lines of our own JavaScript):
- a geometry pass into a framebuffer with colour and depth textures
- a full-screen post pass for outlines and SSAO, the same rules as the CPU pass, with pixel
  offsets scaled to the canvas resolution
- a picking pass that writes residue ids to an RGBA8 target, read back under the pointer, for
  the hover tooltip

Colours and lighting are exact ports of the Rust functions above, evaluated per fragment rather
than per vertex. **One deliberate difference:** a normal that points away from the viewer
(view-space z < 0) is flipped toward it. The CPU pipeline lights such a surface with the 0.30
ambient term alone. The first version flipped on `gl_FrontFacing` instead and rendered 1PGB at
about half the terminal's brightness: the ribbon's winding is not consistent, so "front face"
says nothing about which way the normal points.

The fit uses the part of the canvas the side panel does not cover (the bottom panel on a phone),
so the structure is never drawn under it.

**Interaction (the same keys as the terminal viewer):** drag to rotate, wheel or pinch to zoom,
right-drag to pan; `c` cycles colour schemes, `o` toggles SSAO and outlines, `d` disulfides,
`Space` spin, `r` resets; plus `s`, which saves a PNG. Pointer events cover mouse and touch.

**Panels:** a legend for the current scheme; the metrics already computed for `analyze` (Rg and
Rg/Rg₀, SASA, burial, Ramachandran favoured and outliers, overlaps, pLDDT when the file is a
prediction, fitness); a Ramachandran scatter; and a per-residue pLDDT or B-factor strip. Hovering
shows chain, residue number, insertion code, name, 8-state DSSP and pLDDT or B-factor.

**Fallback:** without WebGL2 or `DecompressionStream` the page says so and still shows the
metrics and panels.

**Code layout:**
- `proteus_render::web` builds the page from `StructureRenderData`.
- The JavaScript lives in `crates/proteus-render/assets/web/`: `core.js` holds the pure
  functions (decode, colours, camera maths) and runs under Node for tests; `viewer.js` holds the
  DOM and WebGL.
- `proteus-server` gains a dependency on `proteus-render` for `/view/{job}`.
- `proteus_core::webview`, the vendored 3Dmol.js, its licence and `scripts/vendor_3dmol.sh` are
  removed.

## Tests

- **Rust:**
  - encode, then decode with a Rust mirror of the page's decoder: positions within the
    quantisation bound, indices and attributes exact
  - the page is self-contained (no `http`, no `src=`, no `href=` outside the page)
  - escaping (a chain id and a title of `</script>…` stay inert)
  - size bounds on real structures
  - Cα-only traces still produce a page
- **Colour parity:** Rust writes a committed fixture of colour samples for each scheme; a Node
  test (`node --test`) asserts that `core.js` reproduces it exactly. CI runs Node.
- **Decode parity:** a Node test decodes a fixture blob written by the Rust encoder.
- **Browser**, by hand with Playwright before merge (a headless-browser CI job is not worth its
  weight yet):
  - the page renders with a WebGL2 context and no console errors
  - screenshots on 1PGB, the p53 model (pLDDT) and 4HHB are compared by eye against the Sixel
    renders of the same structures
  - picking returns the residue under the pointer

## Verification (2026-09-23, Chromium via Playwright, 1400×860)

- 1CRN, 1PGB, 4HHB, the p53 AlphaFold model and 6VXX all open with a WebGL2 context, no console
  errors or warnings and `gl.getError()` = 0.
- Orientation and colours match the Sixel render of the same file (1PGB compared side by side).
- Picking: a 40 px grid over each canvas returns only valid residues, and the tooltip agrees with
  the known structure (1PGB ASP36 in the helix 23–36; 1CRN ILE7 in the helix 7–19).
- `c` cycles pLDDT → rainbow → secondary structure and the legend follows; `o`, drag, wheel, `r`,
  `Space` and `s` (a PNG of the view) all work.
- With `getContext('webgl2')` forced to return null, the fallback message shows and the panel
  still lists all seven measurements.
- At 390×844 there is no horizontal scroll and the structure fits above the panel.
- `/view/{job}` on the daemon returns the page for a Cα-only simulated model (server test).

## Non-goals

Atom-level representations (sticks, spheres, surfaces), which the terminal viewer doesn't draw
either. Mol\*-style analysis, sequence views and trajectories. WebGL1.

## Known costs

- A structure file is no longer in the page, so "save the structure" from the browser is gone.
  The file is next to the page, or at `/api/v1/predictions/by-job/{id}/pdb` on the daemon.
- The mesh build moves into the CLI and daemon: 0.7 s for 6VXX and 2.1 s for 1AON, once per page.
