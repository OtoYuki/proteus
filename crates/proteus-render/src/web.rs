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

/// Magic bytes and version of the mesh blob. Bump the digit when the layout changes; the page
/// refuses a blob it does not know.
pub const MESH_MAGIC: &[u8; 8] = b"PRMESH1\0";

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
        _ => "ss",
    }
}

fn pad4(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
}

/// Quantisation box shared by both meshes: minimum corner and extent per axis, in Å.
fn bounds(meshes: &[&TriangleMesh]) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for m in meshes {
        for v in &m.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(v.position[k]);
                hi[k] = hi[k].max(v.position[k]);
            }
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
            let q = ((v.position[k] - lo[k]) / span[k] * 65535.0)
                .round()
                .clamp(0.0, 65535.0) as u16;
            buf.extend(q.to_le_bytes());
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

/// Encode the ribbon and the (optional) disulfide mesh into the page's binary format,
/// uncompressed. Little-endian throughout:
///
/// ```text
/// magic[8]  u32 nv  u32 nt  u32 ds_nv  u32 ds_nt  f32 lo[3]  f32 span[3]
/// ribbon:    u16 pos[3·nv] (pad4)  i8 nrm[3·nv] (pad4)  u32 res[nv]  f32 plddt[nv]
///            u8 ss[nv] (pad4)  u32 idx[3·nt]
/// disulfide: u16 pos[3·ds_nv] (pad4)  i8 nrm[3·ds_nv] (pad4)  u32 idx[3·ds_nt]
/// ```
///
/// Positions are quantised to 16 bits over the shared bounding box (≤ 0.0015 Å error on an
/// 8 000-residue assembly); normals to 8 bits.
pub fn encode_meshes(ribbon: &TriangleMesh, disulfides: Option<&TriangleMesh>) -> Vec<u8> {
    let empty = TriangleMesh::new();
    let ds = disulfides.unwrap_or(&empty);
    let (lo, span) = bounds(&[ribbon, ds]);
    let mut buf = Vec::with_capacity(ribbon.vertex_count() * 32 + 64);
    buf.extend_from_slice(MESH_MAGIC);
    for n in [
        ribbon.vertex_count(),
        ribbon.triangle_count(),
        ds.vertex_count(),
        ds.triangle_count(),
    ] {
        buf.extend((n as u32).to_le_bytes());
    }
    for x in lo.iter().chain(span.iter()) {
        buf.extend(x.to_le_bytes());
    }
    push_mesh(&mut buf, ribbon, lo, span, true);
    push_mesh(&mut buf, ds, lo, span, false);
    buf
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
    let rama: Vec<[f64; 3]> = s
        .ramachandran_points
        .iter()
        .filter_map(|(phi, psi, reg)| Some([(*phi)?, (*psi)?, region_code(*reg) as f64]))
        .collect();
    let per_residue: Vec<f64> = s.plddts.iter().map(|v| (v * 10.0).round() / 10.0).collect();
    serde_json::json!({
        "title": page.title,
        "caption": page.caption,
        "residues": s.num_residues,
        "disulfides": s.num_disulfides,
        "scheme": scheme_name(page.scheme),
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
            "disulfide": [
                crate::brand::structure::DISULFIDE.r,
                crate::brand::structure::DISULFIDE.g,
                crate::brand::structure::DISULFIDE.b
            ],
        },
        "perResidue": per_residue,
        "rama": rama,
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
}

impl WebPage<'_> {
    /// The complete, self-contained HTML document.
    pub fn render(&self) -> String {
        let blob = encode_meshes(
            &self.structure.ribbon_mesh,
            self.structure.disulfide_mesh.as_ref(),
        );
        use base64::Engine;
        let mesh_b64 = base64::engine::general_purpose::STANDARD.encode(gzip(&blob));
        let meta = script_safe_json(&metadata(self));
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
<header id="head"><h1 id="title"></h1><p id="caption"></p></header>
<aside id="panel" aria-label="structure details"></aside>
<div id="legend" aria-live="polite"></div>
<div id="tip" role="tooltip" hidden></div>
<footer id="keys">drag rotate · wheel zoom · right-drag pan · <kbd>c</kbd> colour · <kbd>o</kbd> effects · <kbd>d</kbd> disulfides · <kbd>space</kbd> spin · <kbd>r</kbd> reset · <kbd>s</kbd> save PNG</footer>
<div id="fallback" hidden></div>
<script id="proteus-meta" type="application/json">{meta}</script>
<script id="proteus-mesh" type="application/octet-stream">{mesh}</script>
<script>{core}</script>
<script>{viewer}</script>
</body>
</html>
"#,
            version = env!("CARGO_PKG_VERSION"),
            title = escape_html(self.title),
            css = VIEWER_CSS,
            meta = meta,
            mesh = mesh_b64,
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
    }

    fn decode(b: &[u8]) -> Decoded {
        assert_eq!(&b[..8], MESH_MAGIC);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (nv, nt, ds_nv, ds_nt) = (
            u32_at(8) as usize,
            u32_at(12) as usize,
            u32_at(16) as usize,
            u32_at(20) as usize,
        );
        let lo = [f32_at(24), f32_at(28), f32_at(32)];
        let span = [f32_at(36), f32_at(40), f32_at(44)];
        let mut o = 48;
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
        }
    }

    #[test]
    fn the_mesh_round_trips_through_the_page_format() {
        let s = crambin();
        let blob = encode_meshes(&s.ribbon_mesh, s.disulfide_mesh.as_ref());
        let d = decode(&blob);
        let m = &s.ribbon_mesh;
        assert_eq!(d.pos.len(), m.vertex_count());
        let (_, span) = bounds(&[m]);
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
    }

    #[test]
    fn the_page_is_self_contained_and_carries_our_viewer() {
        let s = crambin();
        let html = WebPage {
            title: "1crn.pdb",
            caption: "X-ray",
            structure: &s,
            scheme: s.default_color_scheme(),
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
        assert!(html.len() < 200_000, "crambin page is {} bytes", html.len());
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
        }
        .render();
        // The page's own four script elements, and no more.
        assert_eq!(
            html.matches("</script>").count(),
            4,
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
        let fixture = serde_json::json!({ "plddt": plddt, "ss": ss, "rainbow": rainbow });
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
        let blob = gzip(&encode_meshes(&s.ribbon_mesh, s.disulfide_mesh.as_ref()));
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
            encode_meshes(&s.ribbon_mesh, s.disulfide_mesh.as_ref()),
            "assets/web/test/1crn.mesh.gz is stale; rerun with UPDATE_WEB_FIXTURES=1"
        );
        assert_eq!(std::fs::read_to_string(summary_path).unwrap(), summary_text);
    }
}
