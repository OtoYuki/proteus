# Candidate diff and film: what a change did, shown without a person at the screen

2026-10-18. New commands `proteus diff` and `proteus film`, a new module in `proteus-core`
(`variant`), a film renderer in `proteus-render` (CPU reference + GPU), and a residue-pairing
module moved from `proteus-render` into `proteus-core`.

This widens the product past the 2026-09-27 scope freeze ("breadth is frozen; the one extension
is binder-interface triage"). The owner lifted it for this feature on 2026-10-18. It is still
triage: it answers a question a design campaign asks of every candidate, and its numbers are
columns a ranking can use, not only pictures.

## Why

A design campaign ends with candidates and a parent (a scaffold, a wild type, the previous
round's best). Before a candidate earns a wet-lab slot, somebody asks of it: **what did the
change do at the sites that changed, and does the model trust it there?** Today that question
is answered by hand, one structure at a time.

What exists, checked 2026-10-18:

| tool | what it does | what it leaves to a person |
|---|---|---|
| MutAmore (Weissenow et al., 2023; PyMOL + ffmpeg) | a movie of every single-residue variant, coloured by pLDDT, with a 19 × L matrix of lDDT to wild type | the score is global (lDDT), interactions are not computed, and the paper says buried residues and internal changes "stay hidden"; rendering its 12-protein benchmark took ~40 000 s at 720p |
| `wt_vs_mut` PyMOL plugin | finds residues that differ, then "lets the user zoom in on the interactions… one at a time" | everything after finding the residues is interactive |
| CoRINs, GetContacts, Arpeggio (VIVID) | residue interaction networks; shared and unique contacts between structures | no rendering, or a per-structure web page; an Arpeggio user asked for wild-type vs mutant on one page (issue #23) and nothing shows it was built |
| PyMOL / ChimeraX / Mol\* headless | scripted stills and movies | the script, per candidate; ChimeraX offscreen is Linux-only through OSMesa |
| jvogan/proteus (an unrelated project with the same name) | agent skills that drive PyMOL/ChimeraX, PyMOL turntables | wraps other tools; no comparison to a parent |

The gap is not any one of those capabilities. It is the step between them: for each candidate,
pair it to its parent, find the changed sites, diff the non-covalent interactions, read the
model's confidence there, and render that as something you can send, in one pass, with no
person at a screen. This is inferred from the tool landscape, not from interviews with design
teams; the first people to show it to are the owner's contacts.

## What the predictors can and cannot tell us

Pak et al. (PLoS ONE 2023, e0282689) found "very weak or no correlation" between AlphaFold's
pLDDT change on a single mutation and measured ΔΔG or GFP fluorescence. McBride et al. (PRL
2023) are more optimistic about the average effect. Either way, a predicted single mutant often
barely moves. So the feature must:

- report pLDDT only when the file's B-factor column is a predicted confidence
  (`proteus_core::confidence`), and say "experimental B-factors, no confidence" otherwise;
- never present "nothing moved" as "nothing changed": a lost hydrogen bond at a site with
  pLDDT 55 is a finding even when the backbone RMSD is 0.2 Å;
- work on experimental pairs too (crystal structures of a wild type and its mutant), which is
  where the demonstrations come from.

## What `proteus diff` computes

Inputs: a parent structure and one or more candidates (PDB or mmCIF, gzipped or not), optional
`--chains` to restrict both (crystal files often hold several copies of a complex).

1. **Pairing.** Residues are paired by identity, by number for single-chain files with
   different chain ids, or by global alignment, whichever pairs the most identical amino acids
   (the logic `prepare_superposition_for_rendering` already uses, moved to
   `proteus_core::pairing` so the diff and the viewer cannot disagree). Unpaired residues are
   counted as insertions/deletions.
2. **Changed sites.** Paired residues whose amino acid differs. Each site carries chain, number
   and `WT→MUT` notation in the parent's numbering (`D39A`).
3. **Superposition.** Kabsch on paired C-alphas (`compute_kabsch_superposition`): global C-alpha
   RMSD, and per site the C-alpha RMSD of the paired residues within 10 Å of the site
   (`local_rmsd`).
4. **Confidence.** Per site, the predicted confidence in parent and candidate, when the file has
   one (see above).
5. **Interaction diff.** `compute_interaction_network` on both structures. Each interaction is
   keyed by its kind and its endpoints (paired residue, atom name; rings for π–π and cation–π),
   with the candidate mapped into parent numbering. Lost = in parent only; gained = in candidate
   only; kept = both. For every lost or gained interaction, the same atoms are looked up in the
   other structure: if both exist, their distance there is reported (`3.0 → 4.4 Å`, "stretched");
   if an atom is gone, it says which (`OD1 removed by D39A`). That separates "the side chain is
   no longer there" from "it moved", which is what a reader needs.
6. **Locality.** Each lost/gained interaction gets the distance from its nearest endpoint atom to
   the nearest changed-site atom, in the candidate. ≤ 10 Å is local, beyond is distal. Distal
   changes are counted but not narrated: two crystal structures of the same protein differ by
   noise at the cut-offs, and a film that narrates noise is worse than no film.
7. **Interface.** An interaction between two chains is flagged `interface`.

Output: a table on stdout, `--json` (one object per candidate), and `--csv` (one row per
candidate: sites, global RMSD, local lost/gained H-bonds, salt bridges, π–π, cation–π, interface
lost/gained, min site confidence, mean confidence change at sites). The CSV is the triage column
set: a campaign can sort on "local interface H-bonds lost".

## What `proteus film` renders

One MP4 (H.264, yuv420p, faststart) and one PNG per candidate, plus the diff JSON, from a fixed
shot list. Everything is a pure function of (structure, diff, frame index): no clock, no random
numbers, so a film can be regenerated from the files and the CPU frames are byte-identical
across runs.

Shot list (30 fps; durations at the default `--pace 1`):

1. **Establish** (2.5 s): the whole candidate, principal-axis view, slow yaw; sites marked.
2. **Per site group** (sites within 10 Å are one group; at most `--max-sites` groups, default 3,
   ordered by local interactions changed): fly in (1.2 s, eased) to the group's centroid, hold
   with a ±20° orbit (3 s). A near clipping slab opens in front of the site so buried sites are
   visible: MutAmore's stated limitation, removed by construction. Shown:
   - the site's side chains as sticks; the parent's side chain at the same place as dim ghost
     sticks (superposed into the candidate's frame), so a truncation (D→A) is visible as what
     is missing;
   - lost interactions as dashed lines in the theme's `bad` colour between the parent's atom
     positions; gained ones as solid lines in `accent`;
   - the rest of the protein dimmed toward the ground colour.
   - Caption (lower third): `D39A · chain D`, `−2 H-bonds  −1 salt bridge  (interface)`, and
     `pLDDT 91 → 64` when there is a predicted confidence.
3. **Summary** (2.5 s): pull back; a card with the candidate's row (sites, RMSD, counts).

Formats: `--aspect 1:1` (1080 × 1080, default; LinkedIn and Bluesky feeds), `4:5`
(1080 × 1350), `16:9` (1920 × 1080). Brand: the dark theme's ground, Geist Mono for text (OFL,
embedded), structure colours from `brand::structure`.

## Rendering: CPU reference, GPU by default

- **CPU** (`proteus_render::rasterizer`): the existing software rasterizer, extended with a
  near clipping plane, a scene-wide depth range for fog (overlays and ribbon share one fog), and
  per-residue colours for the dimmed context. Frames are rendered 3 × 3 supersampled, then
  post-processed (SSAO, outlines) as `preview_framebuffer` does. Measured at 1920 × 1080: 1PGB
  0.17 s, 6M0J 0.30 s per frame on one core; 1AON 0.50 s at 1080². Frames are independent, so
  they render in parallel across cores and are written to the encoder in order.
- **GPU** (wgpu over Vulkan/Metal, offscreen, no window, no display server): the same vertex
  shading (Gouraud Blinn–Phong, fog) in WGSL, rasterised by the GPU into a 3× supersampled
  target with a depth texture, then the same SSAO/outline kernel as a full-screen pass, then
  downsampled and read back. Text is composited on the CPU after readback, so captions are the
  same pixels on either path. `--renderer auto` (default) uses the GPU when an adapter exists
  and falls back to the CPU; `gpu` refuses to fall back; `cpu` forces the reference. The GPU
  path is checked against the CPU path by a tolerance test (it cannot be bit-identical across
  vendors), skipped when no adapter is present.
- **Encoding**: frames go as raw RGB to `ffmpeg` on a pipe. The encoder is probed once,
  `h264_nvenc` first and `libx264` second (H.264 because every feed and player takes it), with a
  one-frame test encode, because a listed NVENC encoder can still fail at runtime. ffmpeg is a runtime
  dependency of `film` only, like podman is of the container tier; without it `film` writes the
  PNG and says why there is no MP4. PNGs are written by the `png` crate (pure Rust).

## Dependencies added

| crate | why | weight |
|---|---|---|
| `png` 0.18 | the still, without ffmpeg | pure Rust; flate2 is already a dependency |
| `fontdue` 0.9 | caption text | pure Rust, no_std-capable, no system fonts |
| `wgpu` 30, `pollster`, `bytemuck` | GPU renderer | large; behind the `gpu` feature of `proteus-render` (on by default in `proteus-cli`). Vulkan and EGL are loaded at runtime, so the binary still starts on a node without a GPU and falls back to the CPU |

Geist Mono Regular (OFL 1.1) is embedded as TTF; its licence ships next to it, as for the
browser page's WOFF2.

## Non-goals

- Predicting mutant structures. `film` takes structures; `screen` or the user's own pipeline
  makes them.
- ΔΔG or any stability/affinity estimate. The film shows geometry and the model's confidence,
  never a free energy.
- Water-mediated interactions, ligands, and metal sites (the interaction network does not
  compute them).
- Molecular-dynamics morphs between parent and candidate. Interpolating coordinates between two
  structures draws a path no physics produced.

## Anti-goals

- A film that looks impressive and says something false. Every caption number comes from the
  diff JSON written next to the MP4; the test suite checks captions against it.
- Narrating noise: distal interaction changes are counted, never shown as findings.
- A dependency that stops `proteus` running offline on a cluster node.

## Demonstration data

Barnase–barstar crystal structures (Vaughan et al. 1999, "Structural response to mutation at a
protein–protein interface"): `1B27` (barstar C40A/C82A, the usual pseudo-wild type) against
`1B3S` (barnase H102A, barstar Y30F) and `1B2U` (barnase K27A, barstar D36A); `1X1U` against
`1X1W`, `1X1X`, `1X1Y` (barstar E80A, E76A, D35A). These are interface hotspot mutations with
measured effects, so the film's claims can be checked against the literature.
