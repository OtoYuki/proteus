//! The browser viewer: one self-contained HTML page that draws `proteus-render`'s own ribbon
//! with WebGL2.
//!
//! The geometry is built here, by the same code the terminal rasteriser draws, and shipped to
//! the page as a compact binary blob; the page only renders it. Colours are computed in the
//! page with the same integer arithmetic as [`crate::rasterizer::shader`] (pinned by a shared
//! fixture), and lighting is a GLSL port of `shade_blinn_phong`. See
//! `docs/design/2026-09-23-webgl-viewer-design.md` for the measurements behind the format.
//!
//! No network access and no third-party code: the page works from a file on an air-gapped
//! machine, the case this exists for.

use crate::geometry::mesh::TriangleMesh;
use crate::rasterizer::ColorScheme;
use crate::StructureRenderData;
use proteus_core::structure::{RamachandranRegion, SecondaryStructure};
use std::io::Write;

const CORE_JS: &str = include_str!("../assets/web/core.js");
const VIEWER_JS: &str = include_str!("../assets/web/viewer.js");
const VIEWER_CSS: &str = include_str!("../assets/web/viewer.css");
/// SIL OFL 1.1 fonts, embedded so the page looks the same offline (assets/web/fonts/README.md).
const GEIST_MONO: &[u8] = include_bytes!("../assets/web/fonts/GeistMono-latin.woff2");
const FIGTREE: &[u8] = include_bytes!("../assets/web/fonts/Figtree-latin.woff2");

/// The page's head styles: the embedded fonts, the brand's colour roles, then viewer.css.
fn page_css() -> String {
    use base64::Engine;
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    format!(
        "@font-face{{font-family:'Geist Mono';font-weight:100 900;font-display:swap;src:url(data:font/woff2;base64,{})format('woff2')}}\
         @font-face{{font-family:'Figtree';font-weight:300 900;font-display:swap;src:url(data:font/woff2;base64,{})format('woff2')}}\
         :root{{{}}}{}",
        b64(GEIST_MONO),
        b64(FIGTREE),
        crate::brand::css_vars(&crate::brand::DARK),
        VIEWER_CSS
    )
}

/// The mark and the dot-matrix wordmark, inline (no `xmlns`: inline SVG in HTML needs none,
/// and the page must not name a URL).
fn brand_header() -> String {
    use crate::brand::{assets, matrix, DARK};
    let mark = assets::mark_svg(&DARK, true).replace(r#" xmlns="http://www.w3.org/2000/svg""#, "");
    let (w, h, _) = matrix::bitmap("proteus");
    let pitch = 3.0;
    format!(
        r#"<div id="brand" role="img" aria-label="Proteus">{mark}<svg class="word" viewBox="0 0 {vw} {vh}" aria-hidden="true"><g fill="currentColor">{dots}</g></svg></div>"#,
        vw = w as f32 * pitch,
        vh = h as f32 * pitch,
        dots = matrix::svg_dots("proteus", pitch, pitch * 0.38),
    )
}

/// Magic bytes and version of the mesh blob. Bump the digit when the layout changes; the page
/// refuses a blob it does not know.
pub const MESH_MAGIC: &[u8; 8] = b"PRMESH2\0";

/// Code of a secondary-structure state in the blob and in `core.js`.
pub fn ss_code(ss: SecondaryStructure) -> u8 {
    match ss {
        SecondaryStructure::Helix => 0,
        SecondaryStructure::Strand => 1,
        SecondaryStructure::Coil => 2,
    }
}

/// The page's name for a colour scheme.
pub fn scheme_name(scheme: ColorScheme) -> &'static str {
    match scheme {
        ColorScheme::Plddt => "plddt",
        ColorScheme::Rainbow => "rainbow",
        ColorScheme::Scores => "score",
        _ => "ss",
    }
}

fn pad4(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

/// Quantisation box shared by both meshes: minimum corner and extent per axis, in Å.
fn bounds(meshes: &[&TriangleMesh], points: &[nalgebra::Vector3<f32>]) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    let all = meshes
        .iter()
        .flat_map(|m| m.vertices.iter().map(|v| v.position))
        .chain(points.iter().copied());
    for p in all {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if lo[0] > hi[0] {
        return ([0.0; 3], [1.0; 3]);
    }
    let span = [0, 1, 2].map(|k| (hi[k] - lo[k]).max(1e-3));
    (lo, span)
}

fn push_mesh(buf: &mut Vec<u8>, m: &TriangleMesh, lo: [f32; 3], span: [f32; 3], full: bool) {
    for v in &m.vertices {
        for k in 0..3 {
            buf.extend(quantise(v.position[k], lo[k], span[k]).to_le_bytes());
        }
    }
    pad4(buf);
    for v in &m.vertices {
        let n = if v.normal.norm_squared() > 1e-12 {
            v.normal.normalize()
        } else {
            v.normal
        };
        for k in 0..3 {
            buf.push((n[k] * 127.0).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    pad4(buf);
    if full {
        for v in &m.vertices {
            buf.extend((v.residue_index as u32).to_le_bytes());
        }
        for v in &m.vertices {
            buf.extend(v.plddt.to_le_bytes());
        }
        for v in &m.vertices {
            buf.push(ss_code(v.secondary_structure));
        }
        pad4(buf);
    }
    for tri in &m.indices {
        for i in tri {
            buf.extend(i.to_le_bytes());
        }
    }
}

/// The geometry a page ships: the ribbon, the disulfide sticks, every heavy atom with its bonds
/// (for side-chain and ligand sticks), and optionally a superposed reference ribbon.
pub struct PageGeometry<'a> {
    pub ribbon: &'a TriangleMesh,
    pub disulfides: Option<&'a TriangleMesh>,
    pub atoms: &'a crate::atoms::AtomTable,
    pub reference: Option<&'a TriangleMesh>,
}

impl<'a> PageGeometry<'a> {
    /// The geometry of one structure, with its superposed reference when it has one.
    pub fn of(s: &'a StructureRenderData) -> Self {
        Self {
            ribbon: &s.ribbon_mesh,
            disulfides: s.disulfide_mesh.as_ref(),
            atoms: &s.atoms,
            reference: s.comparison.as_ref().map(|c| &c.reference_mesh),
        }
    }
}

/// Encode a page's geometry into its binary format, uncompressed. Little-endian throughout:
///
/// ```text
/// magic[8]  u32 nv  u32 nt  u32 ds_nv  u32 ds_nt  u32 n_atoms  u32 n_bonds  u32 ref_nv
///           u32 ref_nt  f32 lo[3]  f32 span[3]
/// ribbon:    u16 pos[3·nv] (pad4)  i8 nrm[3·nv] (pad4)  u32 res[nv]  f32 plddt[nv]
///            u8 ss[nv] (pad4)  u32 idx[3·nt]
/// disulfide: u16 pos[3·ds_nv] (pad4)  i8 nrm[3·ds_nv] (pad4)  u32 idx[3·ds_nt]
/// atoms:     u16 pos[3·n_atoms] (pad4)  u8 element[n_atoms] (pad4)  u32 res[n_atoms]
///            u32 bond[2·n_bonds]
/// reference: laid out as the ribbon (ref_nv, ref_nt)
/// ```
///
/// Positions are quantised to 16 bits over one bounding box of everything (≤ 0.0015 Å error
/// on an 8 000-residue assembly); normals to 8 bits. Element codes are [`crate::atoms::ELEMENTS`].
pub fn encode_page(g: &PageGeometry<'_>) -> Vec<u8> {
    let empty = TriangleMesh::new();
    let ds = g.disulfides.unwrap_or(&empty);
    let reference = g.reference.unwrap_or(&empty);
    let (lo, span) = bounds(&[g.ribbon, ds, reference], &g.atoms.positions);
    let a = g.atoms;
    let mut buf = Vec::with_capacity(g.ribbon.vertex_count() * 32 + a.positions.len() * 16 + 64);
    buf.extend_from_slice(MESH_MAGIC);
    for n in [
        g.ribbon.vertex_count(),
        g.ribbon.triangle_count(),
        ds.vertex_count(),
        ds.triangle_count(),
        a.positions.len(),
        a.bonds.len(),
        reference.vertex_count(),
        reference.triangle_count(),
    ] {
        buf.extend((n as u32).to_le_bytes());
    }
    for x in lo.iter().chain(span.iter()) {
        buf.extend(x.to_le_bytes());
    }
    push_mesh(&mut buf, g.ribbon, lo, span, true);
    push_mesh(&mut buf, ds, lo, span, false);
    for p in &a.positions {
        for k in 0..3 {
            buf.extend(quantise(p[k], lo[k], span[k]).to_le_bytes());
        }
    }
    pad4(&mut buf);
    buf.extend(&a.elements);
    pad4(&mut buf);
    for r in &a.residues {
        buf.extend(r.to_le_bytes());
    }
    for [x, y] in &a.bonds {
        buf.extend(x.to_le_bytes());
        buf.extend(y.to_le_bytes());
    }
    push_mesh(&mut buf, reference, lo, span, true);
    buf
}

fn quantise(x: f32, lo: f32, span: f32) -> u16 {
    ((x - lo) / span * 65535.0).round().clamp(0.0, 65535.0) as u16
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    e.write_all(bytes).expect("writing to a Vec cannot fail");
    e.finish().expect("writing to a Vec cannot fail")
}

/// JSON safe inside an inline `<script>`: `<`, `>`, `&` and the U+2028/U+2029 line separators
/// are written as `\u` escapes, which JSON.parse reads back unchanged. A structure file's chain
/// ids and names are free text; `</script>` in one must stay data.
pub fn script_safe_json(value: &serde_json::Value) -> String {
    let raw = serde_json::to_string(value).expect("a serde_json::Value always serialises");
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c => out.push(c),
        }
    }
    out
}

/// Escape text interpolated into HTML (the `<title>`).
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

fn region_code(r: RamachandranRegion) -> u8 {
    match r {
        RamachandranRegion::Favored => 0,
        RamachandranRegion::Allowed => 1,
        RamachandranRegion::Outlier => 2,
    }
}

/// Everything the page needs besides the mesh.
fn metadata(page: &WebPage<'_>) -> serde_json::Value {
    let s = page.structure;
    let cam = &s.camera;
    let base = cam.rotation_matrix();
    let rows: Vec<[f32; 3]> = (0..3)
        .map(|r| [base[(r, 0)], base[(r, 1)], base[(r, 2)]])
        .collect();
    // [φ, ψ, region, ribbon residue]. The analysis's points follow the backbone list, which is
    // the ribbon's order when every residue has a C-alpha; the index is dropped (-1) otherwise.
    let aligned = s.ramachandran_points.len() == s.num_residues;
    let rama: Vec<[f64; 4]> = s
        .ramachandran_points
        .iter()
        .enumerate()
        .filter_map(|(i, (phi, psi, reg))| {
            Some([
                (*phi)?,
                (*psi)?,
                region_code(*reg) as f64,
                if aligned { i as f64 } else { -1.0 },
            ])
        })
        .collect();
    let contact = |c: &crate::atoms::Contact| {
        let r1 = |x: f32| (x * 1000.0).round() / 1000.0;
        serde_json::json!([
            c.residues[0],
            c.residues[1],
            [r1(c.points[0].x), r1(c.points[0].y), r1(c.points[0].z)],
            [r1(c.points[1].x), r1(c.points[1].y), r1(c.points[1].z)],
            (c.value * 100.0).round() / 100.0,
            c.label
        ])
    };
    let a = &s.annotations;
    let contacts = |v: &[crate::atoms::Contact]| v.iter().map(contact).collect::<Vec<_>>();
    let confidence = s.confidence.as_ref().map(|c| {
        serde_json::json!({
            "ptm": c.ptm,
            "iptm": c.iptm,
            "ligandIptm": c.ligand_iptm,
            "score": c.confidence_score,
            "chainPtm": c.chain_ptm,
            "pairIptm": c.pair_iptm,
            "pae": c.pae.as_ref().map(|p| serde_json::json!({
                "n": p.n,
                "max": p.max,
                "mean": (p.mean() * 100.0).round() / 100.0,
            })),
            "sources": c.sources.iter()
                .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
                .collect::<Vec<_>>(),
        })
    });
    let per_residue: Vec<f64> = s.plddts.iter().map(|v| (v * 10.0).round() / 10.0).collect();
    serde_json::json!({
        "title": page.title,
        "caption": page.caption,
        "residues": s.num_residues,
        "disulfides": s.num_disulfides,
        // A comparison opens on its deviation colours, unless scores were asked for.
        "scheme": if s.comparison.is_some() && page.scheme != ColorScheme::Scores {
            "deviation"
        } else {
            scheme_name(page.scheme)
        },
        "predicted": s.is_predicted(),
        "camera": {
            "rotation": rows,
            "center": [cam.center.x, cam.center.y, cam.center.z],
            "radius": cam.bounding_radius,
            "halfExtents": cam.half_extents.map(|(x, y)| [x, y]),
        },
        "labels": {
            "chain": s.residue_labels.iter().map(|l| l.chain.as_str()).collect::<Vec<_>>(),
            "number": s.residue_labels.iter().map(|l| l.number).collect::<Vec<_>>(),
            "icode": s.residue_labels.iter().map(|l| l.insertion_code.as_deref().unwrap_or("")).collect::<Vec<_>>(),
            "name": s.residue_labels.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(),
        },
        "dssp": s.dssp,
        "palette": {
            "ground": [crate::brand::DARK.ground.r, crate::brand::DARK.ground.g, crate::brand::DARK.ground.b],
            "disulfide": [
                crate::brand::structure::DISULFIDE.r,
                crate::brand::structure::DISULFIDE.g,
                crate::brand::structure::DISULFIDE.b
            ],
        },
        "perResidue": per_residue,
        "rama": rama,
        "sequence": s.residue_labels.iter().map(|l| crate::atoms::one_letter(&l.name)).collect::<String>(),
        "elements": crate::atoms::ELEMENTS,
        "atomNames": s.atoms.names,
        "ligands": s.ligands.iter().map(|l| serde_json::json!({
            "name": l.name, "chain": l.chain, "number": l.number, "atoms": l.atom_count,
        })).collect::<Vec<_>>(),
        "issues": {
            "rama": a.rama.iter().map(|(r, reg)| [*r as u64, region_code(*reg) as u64]).collect::<Vec<_>>(),
            "clashes": contacts(&a.clashes),
            "hbonds": contacts(&a.hbonds),
            "saltBridges": contacts(&a.salt_bridges),
            "piStacks": contacts(&a.pi_stacks),
            "cationPi": contacts(&a.cation_pi),
        },
        "confidence": confidence,
        "models": s.models.iter().map(|m| serde_json::json!({
            "rank": m.rank, "file": m.file, "score": m.score, "ptm": m.ptm, "iptm": m.iptm,
            "plddt": m.plddt, "rmsd": m.rmsd_to_shown, "ligandRmsd": m.ligand_rmsd_to_shown,
            "shown": m.shown,
        })).collect::<Vec<_>>(),
        "compare": s.comparison.as_ref().map(|c| {
            let r2 = |v: f64| (v * 100.0).round() / 100.0;
            // Deviation on the score scale, 0 Å (blue) to at least 2 Å or the 98th percentile (red).
            let fit = crate::rasterizer::shader::ScoreScale::fit(&c.deviation, true);
            let scale = crate::rasterizer::shader::ScoreScale {
                lo: 0.0,
                hi: fit.hi.max(2.0),
                diverging: false,
                higher_is_worse: true,
            };
            let rgb = |x: crate::rasterizer::buffer::ColorRGB| [x.r, x.g, x.b];
            serde_json::json!({
                "name": c.name,
                "rmsd": r2(c.stats.rmsd),
                "paired": c.stats.paired,
                "referenceResidues": c.stats.reference_residues,
                "mismatched": c.stats.mismatched_names,
                "pairing": c.stats.pairing,
                "deviation": c.deviation.iter().map(|v| v.map(r2)).collect::<Vec<_>>(),
                "colors": c.deviation.iter().map(|v| rgb(scale.color(*v))).collect::<Vec<_>>(),
                "max": r2(scale.hi),
                "stops": crate::rasterizer::shader::SCORE_STOPS.map(rgb),
                "none": rgb(crate::rasterizer::shader::NO_SCORE),
                "colour": rgb(crate::brand::structure::REFERENCE),
            })
        }),
        "scores": s.scores.as_ref().map(|a| {
            let r3 = |v: f64| (v * 1000.0).round() / 1000.0;
            let sc = &a.scores;
            let scale = crate::rasterizer::shader::ScoreScale::fit(&sc.values, a.higher_is_worse);
            let rgb = |c: crate::rasterizer::buffer::ColorRGB| [c.r, c.g, c.b];
            serde_json::json!({
                "column": sc.column,
                "higherIsWorse": a.higher_is_worse,
                "scale": { "lo": r3(scale.lo), "hi": r3(scale.hi), "diverging": scale.diverging },
                "stops": crate::rasterizer::shader::SCORE_STOPS.map(rgb),
                "none": rgb(crate::rasterizer::shader::NO_SCORE),
                "colors": sc.values.iter().map(|v| rgb(scale.color(*v))).collect::<Vec<_>>(),
                "matchedBy": sc.matched_by,
                "values": sc.values.iter().map(|v| v.map(r3)).collect::<Vec<_>>(),
                "aa": proteus_core::scores::AMINO_ACIDS.iter().collect::<String>(),
                "matrix": sc.matrix.as_ref().map(|m| m.iter()
                    .map(|row| row.iter().map(|v| v.map(|x| r3(x as f64))).collect::<Vec<_>>())
                    .collect::<Vec<_>>()),
            })
        }),
        "metrics": s.metrics.as_ref().map_or_else(Vec::new, |m| proteus_core::qc::summary_rows(m, s.num_residues)),
    })
}

/// A browser page for one structure.
pub struct WebPage<'a> {
    /// The page's `<title>` and heading.
    pub title: &'a str,
    /// One line under the heading: the file or the job and its engine.
    pub caption: &'a str,
    pub structure: &'a StructureRenderData,
    /// Colour scheme the page opens with (`c` cycles through all three).
    pub scheme: ColorScheme,
    /// The structure file itself, `(file name, text)`, embedded so the page can hand it back
    /// (the "model" download); `None` leaves it out.
    pub source: Option<(&'a str, &'a str)>,
}

impl WebPage<'_> {
    /// The complete, self-contained HTML document.
    pub fn render(&self) -> String {
        let blob = encode_page(&PageGeometry::of(self.structure));
        use base64::Engine;
        let mesh_b64 = base64::engine::general_purpose::STANDARD.encode(gzip(&blob));
        let meta = script_safe_json(&metadata(self));
        // PAE in 1/8 Å steps (AlphaFold's own bins are 0.25–0.5 Å): one byte per pair.
        let pae_b64 = self
            .structure
            .confidence
            .as_ref()
            .and_then(|c| c.pae.as_ref())
            .map(|p| {
                let q: Vec<u8> = p
                    .values
                    .iter()
                    .map(|v| (v * 8.0).round().clamp(0.0, 255.0) as u8)
                    .collect();
                base64::engine::general_purpose::STANDARD.encode(gzip(&q))
            })
            .unwrap_or_default();
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="proteus {version}">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<canvas id="view" tabindex="0" aria-label="3D structure"></canvas>
<header id="head">{brand}<div id="subject"><h1 id="title"></h1><p id="caption"></p></div></header>
<aside id="panel" aria-label="structure details"></aside>
<div id="legend" aria-live="polite"></div>
<div id="overlay" aria-hidden="true"></div>
<div id="tip" role="tooltip" hidden></div>
<section id="selbox" aria-live="polite" hidden></section>
<nav id="seq" aria-label="sequence"></nav>
<footer id="keys"><span><kbd>drag</kbd> <kbd>←↑↓→</kbd> rotate</span><span><kbd>wheel</kbd> <kbd>+ −</kbd> zoom</span><span><kbd>right-drag</kbd> pan</span><span><kbd>click</kbd> select</span><span><kbd>n</kbd> neighbours</span><span><kbd>f</kbd> focus</span><span><kbd>m</kbd> measure</span><span><kbd>l</kbd> label</span><span><kbd>u</kbd> surface</span><span><kbd>esc</kbd> clear</span><span><kbd>c</kbd> colour</span><span><kbd>o</kbd> effects</span><span><kbd>d</kbd> disulfides</span><span><kbd>x</kbd> reference</span><span><kbd>space</kbd> spin</span><span><kbd>r</kbd> reset</span><span><kbd>s</kbd> save png</span><span class="sig">{signature}</span></footer>
<div id="fallback" hidden></div>
<script id="proteus-meta" type="application/json">{meta}</script>
<script id="proteus-mesh" type="application/octet-stream">{mesh}</script>
<script id="proteus-pae" type="application/octet-stream">{pae}</script>
<script id="proteus-source" type="application/octet-stream" data-name="{source_name}">{source}</script>
<script>{core}</script>
<script>{viewer}</script>
</body>
</html>
"#,
            version = env!("CARGO_PKG_VERSION"),
            title = escape_html(self.title),
            css = page_css(),
            brand = brand_header(),
            signature = crate::brand::SIGNATURE,
            meta = meta,
            mesh = mesh_b64,
            pae = pae_b64,
            source_name = escape_html(self.source.map_or("", |(n, _)| n)),
            source = self
                .source
                .map(|(_, t)| base64::engine::general_purpose::STANDARD.encode(gzip(t.as_bytes())))
                .unwrap_or_default(),
            core = CORE_JS,
            viewer = VIEWER_JS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rasterizer::shader::{plddt_to_color, rainbow_color, secondary_structure_to_color};

    const CRAMBIN: &str = include_str!("../../proteus-core/tests/data/1crn.pdb");

    fn crambin() -> StructureRenderData {
        crate::parse_pdb_structure(CRAMBIN).unwrap()
    }

    /// A Rust mirror of `core.js`'s `parseMesh`, so the format is checked from both sides.
    struct Decoded {
        pos: Vec<[f32; 3]>,
        nrm: Vec<[f32; 3]>,
        res: Vec<u32>,
        plddt: Vec<f32>,
        ss: Vec<u8>,
        idx: Vec<u32>,
        ds_nv: usize,
        ds_idx: Vec<u32>,
        atom_pos: Vec<[f32; 3]>,
        elements: Vec<u8>,
        atom_res: Vec<u32>,
        bonds: Vec<[u32; 2]>,
        ref_nv: usize,
    }

    fn decode(b: &[u8]) -> Decoded {
        assert_eq!(&b[..8], MESH_MAGIC);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let h = |k: usize| u32_at(8 + 4 * k) as usize;
        let (nv, nt, ds_nv, ds_nt) = (h(0), h(1), h(2), h(3));
        let (n_atoms, n_bonds, ref_nv, ref_nt) = (h(4), h(5), h(6), h(7));
        let lo = [f32_at(40), f32_at(44), f32_at(48)];
        let span = [f32_at(52), f32_at(56), f32_at(60)];
        let mut o = 64;
        let up4 = |o: usize| o.div_ceil(4) * 4;
        let read_mesh = |o: &mut usize, n: usize, full: bool| {
            let pos: Vec<[f32; 3]> = (0..n)
                .map(|i| {
                    [0, 1, 2].map(|k| {
                        let q = u16::from_le_bytes(
                            b[*o + (i * 3 + k) * 2..*o + (i * 3 + k) * 2 + 2]
                                .try_into()
                                .unwrap(),
                        );
                        lo[k] + q as f32 / 65535.0 * span[k]
                    })
                })
                .collect();
            *o = up4(*o + n * 6);
            let nrm: Vec<[f32; 3]> = (0..n)
                .map(|i| [0, 1, 2].map(|k| b[*o + i * 3 + k] as i8 as f32 / 127.0))
                .collect();
            *o = up4(*o + n * 3);
            let (mut res, mut plddt, mut ss) = (vec![], vec![], vec![]);
            if full {
                res = (0..n).map(|i| u32_at(*o + i * 4)).collect();
                *o += n * 4;
                plddt = (0..n).map(|i| f32_at(*o + i * 4)).collect();
                *o += n * 4;
                ss = b[*o..*o + n].to_vec();
                *o = up4(*o + n);
            }
            (pos, nrm, res, plddt, ss)
        };
        let (pos, nrm, res, plddt, ss) = read_mesh(&mut o, nv, true);
        let idx: Vec<u32> = (0..nt * 3).map(|i| u32_at(o + i * 4)).collect();
        o += nt * 12;
        let _ = read_mesh(&mut o, ds_nv, false);
        let ds_idx: Vec<u32> = (0..ds_nt * 3).map(|i| u32_at(o + i * 4)).collect();
        o += ds_nt * 12;
        let atom_pos: Vec<[f32; 3]> = (0..n_atoms)
            .map(|i| {
                [0, 1, 2].map(|k| {
                    let at = o + (i * 3 + k) * 2;
                    lo[k] + u16::from_le_bytes([b[at], b[at + 1]]) as f32 / 65535.0 * span[k]
                })
            })
            .collect();
        o = up4(o + n_atoms * 6);
        let elements = b[o..o + n_atoms].to_vec();
        o = up4(o + n_atoms);
        let atom_res: Vec<u32> = (0..n_atoms).map(|i| u32_at(o + i * 4)).collect();
        o += n_atoms * 4;
        let bonds: Vec<[u32; 2]> = (0..n_bonds)
            .map(|i| [u32_at(o + i * 8), u32_at(o + i * 8 + 4)])
            .collect();
        o += n_bonds * 8;
        let (ref_pos, _, _, _, _) = read_mesh(&mut o, ref_nv, true);
        o += ref_nt * 12;
        assert_eq!(o, b.len(), "trailing bytes in the blob");
        Decoded {
            pos,
            nrm,
            res,
            plddt,
            ss,
            idx,
            ds_nv,
            ds_idx,
            atom_pos,
            elements,
            atom_res,
            bonds,
            ref_nv: ref_pos.len(),
        }
    }

    #[test]
    fn the_mesh_round_trips_through_the_page_format() {
        let s = crambin();
        let blob = encode_page(&PageGeometry::of(&s));
        let d = decode(&blob);
        let m = &s.ribbon_mesh;
        assert_eq!(d.pos.len(), m.vertex_count());
        let (_, span) = bounds(&[m], &s.atoms.positions);
        let bound = span.iter().cloned().fold(0.0f32, f32::max) / 65535.0;
        for (v, p) in m.vertices.iter().zip(&d.pos) {
            for (a, b) in v.position.iter().zip(p) {
                assert!((a - b).abs() <= bound, "quantisation");
            }
        }
        for (v, n) in m.vertices.iter().zip(&d.nrm) {
            if v.normal.norm() > 1e-6 {
                let dot: f32 = (0..3).map(|k| v.normal.normalize()[k] * n[k]).sum();
                assert!(dot > 0.99, "normal direction lost: {dot}");
            }
        }
        assert_eq!(
            d.res,
            m.vertices
                .iter()
                .map(|v| v.residue_index as u32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            d.plddt,
            m.vertices.iter().map(|v| v.plddt).collect::<Vec<_>>()
        );
        assert_eq!(
            d.ss,
            m.vertices
                .iter()
                .map(|v| ss_code(v.secondary_structure))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            d.idx,
            m.indices.iter().flatten().copied().collect::<Vec<_>>()
        );
        // Crambin has three disulfides; their sticks travel in the same blob.
        let ds = s.disulfide_mesh.as_ref().unwrap();
        assert_eq!(d.ds_nv, ds.vertex_count());
        assert_eq!(d.ds_idx.len(), ds.triangle_count() * 3);
        // Every heavy atom and bond travels too, to the same precision.
        assert_eq!(d.atom_pos.len(), s.atoms.positions.len());
        for (a, p) in s.atoms.positions.iter().zip(&d.atom_pos) {
            for k in 0..3 {
                assert!((a[k] - p[k]).abs() <= bound, "atom quantisation");
            }
        }
        assert_eq!(d.elements, s.atoms.elements);
        assert_eq!(d.atom_res, s.atoms.residues);
        assert_eq!(d.bonds, s.atoms.bonds);
        assert_eq!(d.ref_nv, 0);
    }

    #[test]
    fn the_page_is_self_contained_and_carries_our_viewer() {
        let s = crambin();
        let html = WebPage {
            title: "1crn.pdb",
            caption: "X-ray",
            structure: &s,
            scheme: s.default_color_scheme(),
            source: Some(("1crn.pdb", CRAMBIN)),
        }
        .render();
        assert!(html.contains("id=\"proteus-mesh\""));
        assert!(html.contains("getContext('webgl2'"), "viewer.js missing");
        assert!(!html.contains("$3Dmol"), "a third-party viewer is back");
        for forbidden in ["http://", "https://", "src=", "href=", "import(", "fetch("] {
            assert!(
                !html.contains(forbidden),
                "page reaches outside itself: {forbidden}"
            );
        }
        // Measured 2026-09-24: 264 KB = fonts 58 KB, viewer code 112 KB, crambin's mesh and 327
        // atoms, and the embedded model file.
        assert!(html.len() < 290_000, "crambin page is {} bytes", html.len());
    }

    #[test]
    fn text_from_the_structure_file_stays_data() {
        let mut s = crambin();
        s.residue_labels[0].chain = "</script><script>alert(1)</script>".into();
        let html = WebPage {
            title: "</script><b>x",
            caption: "y",
            structure: &s,
            scheme: ColorScheme::SecondaryStructure,
            source: Some(("\"><b>x.pdb", "</script>")),
        }
        .render();
        // The page's own six script elements, and no more.
        assert_eq!(
            html.matches("</script>").count(),
            6,
            "a string closed a script"
        );
        assert!(!html.contains("<b>x"));
    }

    #[test]
    fn labels_dssp_and_mesh_agree_on_the_residue_count() {
        let s = crambin();
        assert_eq!(s.residue_labels.len(), s.num_residues);
        assert_eq!(s.dssp.chars().count(), s.num_residues);
        let max_idx = s.ribbon_mesh.vertices.iter().map(|v| v.residue_index).max();
        assert_eq!(max_idx, Some(s.num_residues - 1));
        assert_eq!(s.residue_labels[0].name, "THR");
    }

    /// The colour fixture `core.js` is tested against (`assets/web/test/colors.json`). The page
    /// computes colours itself, so the two implementations are pinned to one file:
    /// `UPDATE_WEB_FIXTURES=1 cargo test -p proteus-render web::` rewrites it.
    #[test]
    fn colour_fixture_matches_the_rust_shader() {
        let plddt: Vec<serde_json::Value> = (0..=1000)
            .map(|i| {
                let p = i as f32 * 0.1;
                let c = plddt_to_color(p);
                serde_json::json!([p, c.r, c.g, c.b])
            })
            .collect();
        let ss: Vec<serde_json::Value> = [
            SecondaryStructure::Helix,
            SecondaryStructure::Strand,
            SecondaryStructure::Coil,
        ]
        .iter()
        .map(|&s| {
            let c = secondary_structure_to_color(s);
            serde_json::json!([ss_code(s), c.r, c.g, c.b])
        })
        .collect();
        let rainbow: Vec<serde_json::Value> = [1usize, 2, 7, 46, 574, 8015]
            .iter()
            .flat_map(|&n| {
                (0..n.min(200)).map(move |i| {
                    let c = rainbow_color(i, n);
                    serde_json::json!([i, n, c.r, c.g, c.b])
                })
            })
            .collect();
        use crate::rasterizer::shader::{ScoreScale, SCORE_STOPS};
        let mut score = Vec::new();
        for (lo, hi, diverging) in [(-8.0, 8.0, true), (0.05, 0.99, false), (-1.0, 3.5, false)] {
            for worse in [false, true] {
                let scale = ScoreScale {
                    lo,
                    hi,
                    diverging,
                    higher_is_worse: worse,
                };
                for k in 0..=60 {
                    let v = lo - 1.0 + (hi - lo + 2.0) * k as f64 / 60.0;
                    let c = scale.color(Some(v));
                    score.push(serde_json::json!([v, lo, hi, worse, c.r, c.g, c.b]));
                }
            }
        }
        let fixture = serde_json::json!({
            "plddt": plddt, "ss": ss, "rainbow": rainbow, "score": score,
            "scoreStops": SCORE_STOPS.map(|c| [c.r, c.g, c.b]),
        });
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/web/test/colors.json");
        let text = serde_json::to_string(&fixture).unwrap() + "\n";
        if std::env::var_os("UPDATE_WEB_FIXTURES").is_some() {
            std::fs::write(path, &text).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(path).unwrap(),
            text,
            "assets/web/test/colors.json is stale; rerun with UPDATE_WEB_FIXTURES=1"
        );
    }

    /// The mesh fixture `core.js`'s decoder is tested against.
    #[test]
    fn mesh_fixture_is_current() {
        let s = crambin();
        let blob = gzip(&encode_page(&PageGeometry::of(&s)));
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/web/test/1crn.mesh.gz");
        let summary_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/web/test/1crn.mesh.json"
        );
        let m = &s.ribbon_mesh;
        let summary = serde_json::json!({
            "nv": m.vertex_count(),
            "nt": m.triangle_count(),
            "dsNv": s.disulfide_mesh.as_ref().map_or(0, |d| d.vertex_count()),
            "firstIndex": m.indices[0],
            "lastResidue": m.vertices.last().unwrap().residue_index,
            "firstPlddt": m.vertices[0].plddt,
            "atoms": s.atoms.positions.len(),
            "bonds": s.atoms.bonds.len(),
            "firstBond": s.atoms.bonds[0],
        });
        let summary_text = serde_json::to_string(&summary).unwrap() + "\n";
        if std::env::var_os("UPDATE_WEB_FIXTURES").is_some() {
            std::fs::write(path, &blob).unwrap();
            std::fs::write(summary_path, &summary_text).unwrap();
        }
        // gzip output is deterministic for a given input and level, so compare decoded content.
        let stored = std::fs::read(path).unwrap();
        let mut d = flate2::read::GzDecoder::new(&stored[..]);
        let mut raw = Vec::new();
        std::io::Read::read_to_end(&mut d, &mut raw).unwrap();
        assert_eq!(
            raw,
            encode_page(&PageGeometry::of(&s)),
            "assets/web/test/1crn.mesh.gz is stale; rerun with UPDATE_WEB_FIXTURES=1"
        );
        assert_eq!(std::fs::read_to_string(summary_path).unwrap(), summary_text);
    }
}
