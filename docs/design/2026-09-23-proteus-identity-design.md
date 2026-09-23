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
severity, in linear RGB. The tests in `brand` recompute every number here.

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
| dim | Sage `#9A9F80` | `#6F7458` | tertiary text (6.4:1 / ≥ 4.5:1) |
| accent | Chartreuse | Olive `#6B660A` | the one loud colour; selection |
| warm | Clay | Clay | cursor, running, highlights (not text on light) |
| sea | Tide | `#2F7771` | data; strand |
| bad | Ember `#E0694A` | `#B4472B` | failure (always with ✗) |

**Terminal colour depth.**
- `NO_COLOR` (no-color.org) → no colour at all.
- `COLORTERM=truecolor|24bit` → exact RGB.
- a `TERM` containing `256color` → nearest xterm-256 entry (6×6×6 cube levels 0/95/135/175/215/255,
  or the 24-step grey ramp).
- otherwise → the 16 ANSI colours by role: accent yellow, sea cyan, bad red, warm and text
  default, dim bright black.

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
- Archivo (OFL) at width 125, weight 800–900, as the display face.
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

- Contrast of every text role on its ground: ≥ 4.5:1. Every accent: ≥ 3:1.
- Colour-blind ΔE of the structure colours: ≥ 30 for every vision type. Of the pLDDT bands:
  reported, not asserted (the convention is not ours).
- Colour depth: detection over a table of environments, 256 and 16 mappings stable.
- The mark: renders at every size down to 1×1 cells without panicking; a folded frame is
  non-empty; the fold is monotone in time (no frame after t = 1).
- Assets are generated by the code and checked to be current, as the web fixtures are.

## Non-goals

A new logo for s1re.sh itself; changing the pLDDT convention; any font that cannot be embedded
under its licence.
