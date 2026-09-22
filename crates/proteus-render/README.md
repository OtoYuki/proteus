# proteus-render

Software 3D rendering of protein structures for the terminal, from
[Proteus](https://github.com/OtoYuki/proteus): a cartoon-ribbon mesh (Hermite spline through
the C-alpha trace, Bishop parallel-transport frames, DSSP-driven cross-sections with Richardson
arrowheads, disulfide sticks), a z-buffer rasteriser with Blinn–Phong shading, SSAO and
outlines, and three terminal back-ends — half-block cells, Braille, and the kitty graphics
protocol — plus an interactive crossterm viewer with an optional Ramachandran/pLDDT dashboard.

```rust
use proteus_render::{parse_pdb_structure, render_structure_snapshot};
use proteus_render::{rasterizer::ColorScheme, terminal::TerminalBackend};
let data = parse_pdb_structure(&std::fs::read_to_string("1crn.pdb")?)?;
let frame = render_structure_snapshot(&data, 100, 30, TerminalBackend::HalfBlock, data.default_color_scheme())?;
print!("{frame}");
```

The camera opens on the model's principal-axis frame; `default_color_scheme()` colours by
pLDDT only when the B-factor column really is a confidence.
