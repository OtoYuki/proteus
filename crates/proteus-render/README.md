<img src="https://raw.githubusercontent.com/OtoYuki/proteus/main/docs/brand/proteus-mark.png" width="48" align="right" alt="Proteus">

# proteus-render

Software 3D rendering of protein structures for the terminal and the browser, from
[Proteus](https://github.com/OtoYuki/proteus): a cartoon-ribbon mesh (Hermite spline through
the C-alpha trace, the wide face oriented by the backbone carbonyl, parallel-transport frames
for C-alpha-only models, DSSP-driven cross-sections with Richardson arrowheads, disulfide
sticks), a z-buffer rasteriser with Blinn–Phong shading, SSAO and outlines, and four terminal
back-ends — half-block cells, Braille, DEC Sixel and the kitty graphics protocol — plus an
interactive crossterm viewer with an optional Ramachandran/pLDDT dashboard.

`web` writes the same mesh, colours and measurements into one self-contained HTML page drawn
with WebGL2, and `brand` holds the identity every surface reads: palette roles, colour-depth
detection, the mark and the dot-matrix wordmark.

<img src="https://raw.githubusercontent.com/OtoYuki/proteus/main/docs/media/view.gif" alt="the interactive terminal viewer: protein G as a clay and tide ribbon beside a live Ramachandran plot and measurements">

```rust
use proteus_render::{parse_pdb_structure, render_structure_snapshot};
use proteus_render::{rasterizer::ColorScheme, terminal::TerminalBackend};
let data = parse_pdb_structure(&std::fs::read_to_string("1crn.pdb")?)?;
let frame = render_structure_snapshot(&data, 100, 30, TerminalBackend::HalfBlock, data.default_color_scheme())?;
print!("{frame}");
```

The camera opens on the model's principal-axis frame; `default_color_scheme()` colours by
pLDDT only when the B-factor column really is a confidence.
