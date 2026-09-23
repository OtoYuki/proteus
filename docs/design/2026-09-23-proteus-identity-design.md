# Proteus identity: a s1re.sh sub-brand across terminal and browser

2026-09-23. Sub-project 1 (foundation) of the identity pass. The home screen, the terminal 3-D
viewer, the browser viewer and the brand assets and CLI output (sub-projects 2–5) build on it.

## Why

The home screen and the browser viewer worked but had no character: default ratatui cyan, a
generic dark page. They also shared no tokens, so they could only drift apart. Proteus is a
portfolio piece of s1re.sh and should be recognisable at a glance, in a screenshot, without a
caption.

## Decisions taken with the owner

- **A sub-brand of s1re.sh.** Proteus has its own mark and wordmark, built from the s1re.sh
  identity v0.1 (the "s1re.sh wordmark" canvas, 2026-09-23), and is signed "a s1re.sh project".
- **Motion: still, alive at moments.** Motion only where it means something: the launch, a
  change of state, a transition. Idle screens send nothing, and `NO_MOTION` turns motion off.
- **Scope:** the home screen, the browser viewer, the terminal 3-D viewer, brand assets and CLI
  output.

## The source identity (s1re.sh v0.1), as read

| token | hex | role in s1re.sh |
|---|---|---|
| Root | `#141C10` | default dark ground |
| Cream | `#FBFFE1` | default light ground |
| Chartreuse | `#99920B` | accent ("loud") |
| Clay | `#D8A664` | warm; the block cursor |
| Moss | `#5A6042` | quiet dark; hairlines |
| Khaki | `#D1CF8B` | paper alternative; hairlines on light |

**Type:**
- Freigeist (display; XWide 900 for the logo)
- Goga (text)
- Geist Mono (system; labels uppercase at 11–14 px, tracked 0.14–0.16 em)

**Marks:**
- M1 sunburst (16 rays, one per layout)
- M2 "s1" monogram
- M3 the metaball "Organism" (for "bio, pipelines, things that grow")
- M4 dot matrix

**Micrographics:** 4-point sparkles, `(parenthesised_labels)`, a `~ $` prompt with a Clay block
cursor.

## Measured, not assumed

Contrast is the WCAG 2.x ratio. Colour-blind separation is the smallest CIE76 ΔE between any
two colours after simulating the vision with Machado, Oliveira & Fernandes (2009) at full
severity, in linear RGB. The tests in `brand` assert the thresholds (text ≥ 4.5:1, structure colours ≥ ΔE 30 for every vision type) with the same functions that produced the numbers below; the individual numbers were recomputed independently in review, not by a test.

| on Root `#141C10` | Cream | Khaki | Clay | Chartreuse | Moss |
|---|---|---|---|---|---|
| contrast | 17.0 | 10.8 | 7.9 | 5.4 | **2.65** |

- Moss is below 3:1 on Root, so it is used for hairlines and ornaments, never for text.
- Chartreuse on Cream is 3.2:1. In the light theme the accent used for text is a derived Olive
  `#6B660A` (5.8:1).

**Secondary-structure colours.** The worst-case pair ΔE by vision:

| candidate | normal | deutan | protan | tritan |
|---|---|---|---|---|
| Clay / Chartreuse / Khaki (palette only) | 23.9 | **12.5** | 13.1 | 23.3 |
| magenta / amber / cyan (shipped until now) | 109.7 | 23.2 | 50.3 | 28.9 |
| **Clay / Tide `#4F9A94` / Pale `#E7E9C8`** | 36.3 | **32.2** | 32.4 | 35.9 |

The palette alone fails for red–green colour blindness. One cool colour is added, **Tide**
(`#4F9A94`), as a data colour. The chosen trio separates better than what shipped before, for
every vision type.

**Status colours cannot carry meaning alone.** The failure red, Ember `#E0694A`, is only ΔE 11
from Clay under deuteranopia. So every state is written as glyph + word:
- `✓ done`
- `✗ failed`
- `● running`
- `◌ queued`
- `– cancelled`

This is a rule for all surfaces.

**Unchanged by convention:** the AlphaFold pLDDT ramp (orange → yellow → cyan → blue). Readers
decode it by habit. The legend says whose convention it is.

## The system

**Palette roles** (one definition in `proteus_render::brand`, read by every surface):

| role | dark | light | use |
|---|---|---|---|
| ground | Root `#141C10` | Cream `#FBFFE1` | background |
| surface | `#1B2516` | `#F3F6D6` | panels (derived, 1.1:1 from ground) |
| line | Moss | Khaki | hairlines, borders |
| text | Cream | Root | body |
| muted | Khaki | Moss | secondary text (10.8:1 / 6.4:1) |
| dim | Sage `#9A9F80` | `#646948` | tertiary text (≥ 4.5:1 on ground and panel; `#6F7458` failed on the light panel at 4.39) |
| accent | Chartreuse | Olive `#6B660A` | the one loud colour; selection |
| warm | Clay | Clay | cursor, running, highlights (not text on light) |
| sea | Tide | `#2F7771` | data; strand |
| bad | Ember `#E0694A` | `#B4472B` | failure (always with ✗) |

**Terminal colour depth.**
- `NO_COLOR` (no-color.org) → no colour at all.
- `COLORTERM=truecolor|24bit` → exact RGB.
- a `TERM` containing `256color` → nearest xterm-256 entry (6×6×6 cube levels 0/95/135/175/215/255,
  or the 24-step grey ramp).
- otherwise → the 16 ANSI colours by role: accent yellow, warm bright yellow, sea cyan, bad
  red, dim and line bright black, text and muted the terminal's default. Data colours
  (structure, pLDDT) go by hue bins instead: helix red, strand cyan, coil white, pLDDT blue /
  cyan / yellow / red.
- `TERM=dumb` → no escape sequences at all. Over SSH, where `COLORTERM` is not passed on,
  a `TERM` naming kitty, ghostty, alacritty, foot or wezterm, or ending in `-direct` or
  containing `truecolor`, means 24-bit.

The ground is painted only in truecolor and 256 modes. In 16-colour mode the terminal's own
background is kept.

**The mark: a chain that folds into a p.**
- Twelve beads (residues) of radius 6.4–8.8 in a 100×100 box, joined by a smooth union
  (polynomial smooth-min, k = 3.2) between *consecutive* beads only. So it reads as a chain
  with waists, not a blob, and the chain folds into a lowercase **p**: a stem, then a bowl that
  closes against it.
- The launch animation starts from the same chain extended (a low sine) and folds into the p
  over about 0.9 s with a cubic ease-out. This is the myth in one second: Proteus changes shape,
  then holds still.
- Filled on dark grounds, outlined on light (the s1re.sh rule for organic shapes).
- One Clay "ligand" ring sits off the chain as the single accent.
- Rendered from one signed-distance definition to:
  - terminal braille
  - SVG, as a vector path traced by marching squares
  - the browser page

**The wordmark: dot matrix.**
- "proteus" set in a 5×9 dot-matrix face (ascender 2, x-height 5, descender 2), extending the
  s1re.sh M4 Matrix. It is the same everywhere (braille or half-blocks in a terminal, SVG circles
  on the web), with no font to license or embed.
- The lockup: mark + wordmark, with "a s1re.sh project" in small tracked mono.

**Type in the browser.**
- Geist Mono (SIL OFL) for labels and numbers.
- (Planned, not shipped: Archivo as a display face. The dot-matrix wordmark made it unnecessary.)
- Figtree (OFL) as text.

These stand in for Freigeist and Goga, whose licences for embedding in distributed software are
not established. The page stays offline, so the fonts are embedded as WOFF2 subsets. If the
Freigeist and Goga licences are confirmed, swapping them in is a one-line change per face.

**Voice.**
- Short sentences.
- Labels in tracked uppercase mono.
- `(parenthesised)` names for panels.
- `SYS/` numbering is left to s1re.sh itself.
- Every number keeps its unit and its caveat. The honesty rules of the product (for example,
  a B-factor is not a confidence) are part of the identity, not exceptions to it.

## Tests

- Contrast of every text role (text, muted, dim, accent, sea, bad) on its ground and panel: ≥ 4.5:1. The warm colour is used for ornaments and is asserted only ≥ 2:1 on the dark ground; on Cream it is 2.14:1 and is not used for text there.
- Colour-blind ΔE of the three structure colours: ≥ 30 for every vision type. The pLDDT bands
  follow the AlphaFold convention and are not tested for separation. (With the disulfide
  colour included, the closest pair under deuteranopia is helix vs disulfide at ΔE 21.2.)
- Colour depth: detection over a table of environments, 256 and 16 mappings stable.
- The mark: renders at every size down to 1×1 cells without panicking; a folded frame is
  non-empty; the fold covers more at t = 1 than at t = 0, and any t past 1 gives the folded
  mark.
- Assets are generated by the code and checked to be current, as the web fixtures are.

## Terminal surfaces, as built (sub-projects 2 and 3)

**The home screen:**
- **Launch:** the chain folds into the p over 0.9 s, then the half-block wordmark and the
  signature appear and it holds for 0.45 s. Any key skips it; `NO_MOTION` turns it off.
- **Header:** `proteus` in the accent, the version, the signature, then the tabs as
  `(jobs) (structures) (run)` over a hairline.
- **Panels:** sections are a hairline with a `(name)`, not boxes.
- **Selection:** the selected row gets the surface colour and an accent `▌`.
- **Status line:** the `~ $` prompt with the last command run and its result (`✓ done` or
  `✗ …`), ending in the Clay cursor.
- **Job states:** glyph + word (`✓ done`, `✗ failed`, `● running`, which pulses `●`/`◉` every
  0.5 s while motion is on, `◌ queued`).

**The terminal 3-D viewer:**
- **Status bar:** the structure; a legend for the current colours as swatch + word; the toggles
  as "on"/"off" words; frame rate; then the keys in the accent.
- **pLDDT legend on an experimental file:** it says it is the B-factor on the pLDDT scale, not a
  confidence.
- **Dashboard:**
  - The pLDDT strip uses the AlphaFold ramp. It used four colours of its own
    (`30,64,175` / `56,189,248` / `250,204,21` / `239,68,68`), so it disagreed with the ribbon
    beside it.
  - Without colour, the strip shows the bands as shade density (`█ ▓ ▒ ░`).
  - Ramachandran favoured and allowed had the same glyph and differed only by colour; allowed is
    now `○`.
- **`--compare`:** the target is Tide and the reference Clay (ΔE 51 under deuteranopia), instead
  of cyan and ruby.

**Found while building, and fixed:**
- **Blank braille cells:** an empty braille cell (U+2800) draws as faint dots in some fonts, so
  blank cells are spaces.
- **Header wordmark:** the braille wordmark was unreadable at header size, so the header is
  typographic and the dot-matrix wordmark appears only in the launch, as half-blocks.
- **The 3-D view's background:** the half-block compositor treats pure black as an empty pixel
  and leaves the terminal's own background there. Painting the ground would repaint every empty
  pixel on every frame over SSH, so the 3-D view keeps the terminal's background.
- **Re-cased job names:** a job's name was lower-cased in a panel title (`I6A` → `i6a`,
  another mutation). Names are data and are never re-cased; a test holds it.
- **Recording the demos:** under asciinema with piped input, bare `proteus` correctly prints
  the usage (stdin is not a terminal). Key-driven demos are therefore recorded inside tmux
  (`docs/media/record.sh`, `demos/home.keys`).

**The 3-D picture itself** (half-block, braille, Sixel, kitty) is 24-bit colour by nature and does
not follow `NO_COLOR` or the colour depth. Only the text around it (status bar, dashboard,
home screen) does.

**Measured at each depth** with `tmux capture-pane -e`:

| depth | colour codes |
|---|---|
| truecolor | 24-bit only |
| `TERM=xterm-256color` | 256-colour entries only |
| `TERM=xterm` | ANSI colours only (3, 6, 8, 11, written as `38;5;N`), ground not painted |
| `NO_COLOR` | none |

## The browser viewer, as built (sub-project 4)

**Colours and type:**
- Every colour on the page comes from `brand::css_vars`, declared on `:root` when the page is
  built. `viewer.css` names no colour of its own.
- The fonts are embedded as WOFF2 data URIs: Geist Mono and Figtree, latin subsets from Google
  Fonts, 23 KB and 20 KB. Both are SIL OFL 1.1, confirmed from the upstream repositories; the
  licences are in `assets/web/fonts/`. The 1CRN page grows from 72,484 to 156,930 bytes.

**Layout:**
- **Header:** the mark and the dot-matrix wordmark, inline SVG. There is no `xmlns`, because the
  self-containment test forbids any URL in the page. Then the structure's name as the heading,
  and "N residues · predicted model / experimental structure" under it.
- **Panel:**
  - `(named)` hairline sections.
  - Ramachandran with a shape per region (● ○ ▲) and a key.
  - For an experimental file, a note that a B-factor measures motion and disorder, not
    confidence.
- **Legend:** swatch + word. The pLDDT legend names the AlphaFold colours, or says "! not a
  confidence" on an experimental file.
- **Loading:** the canvas fades in over 0.45 s once drawn. There is no fade under
  `prefers-reduced-motion`.

**Verified in Chromium, Firefox and WebKit.** `scripts/browser-check.mjs` runs in CI on every push.
It checks WebGL2, a drawn canvas, that every picked residue exists, the keys, and the
no-WebGL2 fallback. WebKit ran in Playwright's official Ubuntu image (`v1.55.0-noble`, with
podman here), because Playwright's WebKit build needs Ubuntu's libraries. Real Safari on macOS
was not run.

WebKit found three problems that the other two engines did not show. Hover labels
returned residues that do not exist. Each was probed separately with known ids:
- an `UNSIGNED_INT` integer vertex attribute read back near 2³²;
- `uint` constants of 65536 and above in a shader gave wrong results;
- the blue and alpha bytes of the RGBA8 pick target did not hold what was written.

The pick pass now carries the residue index as a float attribute and does its arithmetic in
float, which is exact below 2²⁴. It writes the id in red and green only, with a second pass
for the high bits above 65,535 residues. Probed with ids up to 16,777,000 in Chromium and
WebKit.

**Found and fixed:**
- The panel code read the colour roles before they were declared (a temporal-dead-zone error)
  and the page never started. No automated test runs `viewer.js` (only `core.js`), so the
  browser check caught it.
- A tooltip left over from before a resize was squeezed into one column; it is now kept on one
  line and hidden on resize.

## Brand assets and CLI output, as built (sub-project 5)

**Assets:**
- The README opens with the lockup in a `<picture>`: the dark version for GitHub's dark theme,
  the light one otherwise.
- `docs/brand/proteus-social.svg` and `.png` are the 1280 × 640 link preview. It has to be set
  by hand in the repository's settings; the API has no endpoint for it.
- All SVGs are generated by `brand::assets` and checked to be current. SVG collapses runs of
  spaces, so the prompt line in the card keeps its spacing with `xml:space="preserve"`.

**CLI output:**
- Progress bars and spinners (`submit`, `screen`, `analyze`) take brand colours through
  `cli::tint`, at the detected depth: `#rrggbb`, a 256 index, or an ANSI name.
- `console` (indicatif's styling) turns colour off by itself under `NO_COLOR` and in a pipe;
  this was read in its source, 0.16.6 `unix_term.rs`.
- The `--compare` summary reads `✓` in the accent when the two structures are the same
  sequence residue for residue, and `!` in the warm colour otherwise, with target and
  reference swatches.
- The home screen's child commands print the `~ $` prompt and a `✓`/`✗` result.

The ESM-2 heat map keeps its own diverging scale: it is a data scale centred on the scan's
median, not a brand colour.

## Non-goals

A new logo for s1re.sh itself; changing the pLDDT convention; any font that cannot be embedded
under its licence.
