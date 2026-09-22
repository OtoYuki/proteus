//! Self-contained browser page for one structure.
//!
//! The viewer library and the structure are both embedded in the HTML, so the page opens with
//! no network: on an HPC login node, an airgapped cluster or a laptop on a plane. That is the
//! whole reason this is not a `<script src="https://cdn...">` tag — Proteus is used exactly
//! where a CDN is not reachable, and a scientific artifact should not depend on a third
//! party's uptime to render.
//!
//! One page shape is shared by `proteus view --html/--web` and the daemon's `/view/{job}`;
//! they differ only in the caption. See `assets/README.md` for the vendoring policy and why
//! 3Dmol.js rather than Mol*.

use crate::confidence::ConfidenceSource;
use crate::io::StructureFormat;
use crate::structure::SecondaryStructure;

/// 3Dmol.js, vendored (BSD-3-Clause; see `assets/3Dmol-LICENSE.txt`).
const VIEWER_JS: &str = include_str!("../assets/3Dmol-min.js");

/// How to colour the cartoon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WebColorScheme {
    /// AlphaFold pLDDT palette over the B-factor column. `scale` multiplies the raw value, so
    /// a file written on the 0–1 scale (the ESM Atlas API) still spans the full ramp.
    Plddt { scale: f64 },
    /// Helix / sheet / coil from the file's own secondary-structure records.
    SecondaryStructure,
}

impl WebColorScheme {
    /// pLDDT only when the B-factor column really is a confidence; otherwise secondary
    /// structure, so an experimental structure is not painted "very low confidence" because
    /// its B-factors happen to be small.
    pub fn from_provenance(source: Option<ConfidenceSource>, plddt_scale: f64) -> Self {
        match source {
            Some(ConfidenceSource::Predicted) => Self::Plddt { scale: plddt_scale },
            _ => Self::SecondaryStructure,
        }
    }

    fn style_js(&self) -> String {
        match self {
            // The AlphaFold bands: <50 orange, 50-70 yellow, 70-90 cyan, >90 blue. Each colour
            // is repeated so CustomLinear steps between bands instead of blending across them.
            Self::Plddt { scale } => format!(
                "{{cartoon:{{colorscheme:{{prop:'b',gradient:new $3Dmol.Gradient.CustomLinear(\
                 0,{max},['#FF7D45','#FF7D45','#FFDB13','#FFDB13','#65CBF3','#65CBF3','#0053D6','#0053D6'])}}}}}}",
                max = 100.0 / scale.max(f64::MIN_POSITIVE)
            ),
            Self::SecondaryStructure => "{cartoon:{colorscheme:'ssJmol'}}".to_string(),
        }
    }

    /// One line under the title saying what the colours mean, so a reader never has to guess.
    fn legend(&self) -> &'static str {
        match self {
            Self::Plddt { .. } => {
                "pLDDT: &lt;50 very low · 50–70 low · 70–90 confident · &gt;90 very high"
            }
            Self::SecondaryStructure => "secondary structure: helix · sheet · coil",
        }
    }
}

/// Everything the page needs. `caption` is shown under the title (a file name, a job id and
/// its engine); `structure` is the raw PDB or mmCIF text.
pub struct WebViewPage<'a> {
    pub title: &'a str,
    pub caption: &'a str,
    pub structure: &'a str,
    pub format: StructureFormat,
    pub color: WebColorScheme,
    /// Proteus's own Kabsch–Sander assignment, as `(chain, residue number, state)`. When given,
    /// the page overrides the viewer's built-in guess so that the browser shows the same
    /// secondary structure as `proteus analyze`, the terminal viewer and the Parquet export.
    /// 3Dmol.js assigns its own when this is empty, which is a heuristic, not DSSP.
    pub secondary_structure: &'a [(String, isize, SecondaryStructure)],
}

impl WebViewPage<'_> {
    /// The complete HTML document. No network requests, no external references.
    pub fn render(&self) -> String {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(self.structure.as_bytes());
        let fmt = match self.format {
            StructureFormat::MmCif => "cif",
            StructureFormat::Pdb => "pdb",
        };
        // "chain:resi" -> 3Dmol's per-atom ss code. Keyed rather than positional so it cannot
        // silently mis-align if the viewer orders atoms differently from the backbone walk.
        let ss_map = if self.secondary_structure.is_empty() {
            String::from("null")
        } else {
            let entries: Vec<String> = self
                .secondary_structure
                .iter()
                .map(|(chain, resi, ss)| {
                    let code = match ss {
                        SecondaryStructure::Helix => 'h',
                        SecondaryStructure::Strand => 's',
                        SecondaryStructure::Coil => 'c',
                    };
                    format!("\"{}:{}\":\"{}\"", escape_js(chain), resi, code)
                })
                .collect();
            format!("{{{}}}", entries.join(","))
        };
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #0f172a; color: #e2e8f0;
    font: 13px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  #viewport {{ position: absolute; inset: 0; }}
  #caption {{ position: absolute; left: 50%; bottom: 14px; transform: translateX(-50%);
    z-index: 10; pointer-events: none; text-align: center; padding: 10px 18px;
    background: rgba(15,23,42,.82); border: 1px solid rgba(148,163,184,.25);
    border-radius: 10px; backdrop-filter: blur(10px); max-width: min(90vw, 720px); }}
  #caption b {{ display: block; color: #38bdf8; font-size: 14px; }}
  #caption span {{ color: #94a3b8; font-size: 11px; }}
  #error {{ position: absolute; inset: 0; display: none; place-content: center;
    text-align: center; padding: 2rem; color: #fca5a5; }}
</style>
<script>{viewer_js}</script>
</head>
<body>
<div id="viewport"></div>
<div id="caption"><b>{title}</b><span>{caption}</span><br><span>{legend}</span></div>
<div id="error">This structure could not be rendered. The file itself is unchanged on disk.</div>
<script>
(function () {{
  try {{
    var viewer = $3Dmol.createViewer(document.getElementById('viewport'),
      {{ backgroundColor: '#0f172a' }});
    viewer.addModel(atob("{b64}"), '{fmt}');
    // Proteus's DSSP wins over the viewer's built-in guess when we have it.
    var dssp = {ss_map};
    if (dssp) {{
      viewer.getModel().selectedAtoms({{}}).forEach(function (a) {{
        var s = dssp[a.chain + ':' + a.resi];
        if (s) {{ a.ss = s; }}
      }});
    }}
    viewer.setStyle({{}}, {style});
    viewer.zoomTo();
    viewer.render();
  }} catch (e) {{
    document.getElementById('viewport').style.display = 'none';
    var err = document.getElementById('error');
    err.style.display = 'grid';
    err.textContent = 'This structure could not be rendered: ' + e;
  }}
}})();
</script>
</body>
</html>"#,
            title = escape_html(self.title),
            caption = escape_html(self.caption),
            legend = self.color.legend(),
            viewer_js = VIEWER_JS,
            b64 = b64,
            fmt = fmt,
            ss_map = ss_map,
            style = self.color.style_js(),
        )
    }
}

/// Proteus's own DSSP assignment for a parsed structure, as `(chain, residue number, state)`
/// ready for [`WebViewPage::secondary_structure`]. Empty when the structure has no backbone
/// to assign, in which case the page leaves the viewer's own guess alone.
pub fn dssp_by_residue(pdb: &pdbtbx::PDB) -> Vec<(String, isize, SecondaryStructure)> {
    let protein = crate::io::protein_heavy_atoms(pdb);
    let backbone = crate::backbone::extract_backbone(&protein);
    if backbone.is_empty() {
        return Vec::new();
    }
    let summary = crate::structure::assign_secondary_structure(&backbone);
    backbone
        .iter()
        .zip(summary.assignment)
        .map(|(r, ss)| (r.chain_id.clone(), r.seq_num, ss))
        .collect()
}

/// Escape a string used inside a double-quoted JS string literal.
fn escape_js(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Escape text interpolated into the page. The structure itself is base64, never inlined raw.
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(color: WebColorScheme, structure: &str, format: StructureFormat) -> String {
        WebViewPage {
            title: "1crn.pdb",
            caption: "engine: esmfold-api",
            structure,
            format,
            color,
            secondary_structure: &[],
        }
        .render()
    }

    #[test]
    fn page_is_self_contained() {
        let html = page(
            WebColorScheme::SecondaryStructure,
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\n",
            StructureFormat::Pdb,
        );
        // No network of any kind: the failure mode this design exists to prevent.
        assert!(html.contains("$3Dmol"), "viewer library is not embedded");
        assert!(
            html.len() > 400_000,
            "viewer library looks absent ({} bytes)",
            html.len()
        );
        // Our own markup and script must reference nothing external. The vendored library is
        // excluded from the scan: it carries code paths for fetching remote structures that
        // this page never reaches, and matching on those would only test 3Dmol.js's source.
        let ours = html.replace(VIEWER_JS, "/* vendored viewer */");
        for forbidden in [
            "http://",
            "https://",
            "//cdn",
            "unpkg",
            "fetch(",
            "XMLHttpRequest",
            "import(",
            "src=",
            "href=",
        ] {
            assert!(
                !ours.contains(forbidden),
                "page reaches outside itself via {forbidden}"
            );
        }
    }

    #[test]
    fn plddt_scale_sets_the_gradient_domain() {
        // 0-100 file: ramp tops out at 100. 0-1 file (ESM Atlas): scale 100 => same ramp.
        let full = page(
            WebColorScheme::Plddt { scale: 1.0 },
            "ATOM\n",
            StructureFormat::Pdb,
        );
        let unit = page(
            WebColorScheme::Plddt { scale: 100.0 },
            "ATOM\n",
            StructureFormat::Pdb,
        );
        assert!(full.contains("CustomLinear(0,100,"), "{}", &full[..0]);
        assert!(unit.contains("CustomLinear(0,1,"));
        assert!(full.contains("#0053D6") && full.contains("#FF7D45"));
    }

    #[test]
    fn colour_follows_provenance() {
        assert_eq!(
            WebColorScheme::from_provenance(Some(ConfidenceSource::Predicted), 1.0),
            WebColorScheme::Plddt { scale: 1.0 }
        );
        assert_eq!(
            WebColorScheme::from_provenance(Some(ConfidenceSource::ExperimentalBFactor), 1.0),
            WebColorScheme::SecondaryStructure
        );
        assert_eq!(
            WebColorScheme::from_provenance(None, 1.0),
            WebColorScheme::SecondaryStructure
        );
    }

    #[test]
    fn mmcif_is_declared_as_cif_not_pdb() {
        // Handing mmCIF to the viewer as 'pdb' loads an empty scene; this is the bug the
        // previous Mol* page shipped with.
        let cif = page(
            WebColorScheme::SecondaryStructure,
            "data_1CRN\n",
            StructureFormat::MmCif,
        );
        assert!(cif.contains("addModel(atob(\"ZGF0YV8xQ1JOCg==\"), 'cif')"));
    }

    /// The whole point of carrying our own assignment: a reviewer can see, in the page, that
    /// the browser is told what Proteus's DSSP decided rather than guessing for itself.
    #[test]
    fn proteus_dssp_overrides_the_viewers_guess() {
        let ss = vec![
            ("A".to_string(), 1, SecondaryStructure::Helix),
            ("A".to_string(), 2, SecondaryStructure::Strand),
            ("B".to_string(), 7, SecondaryStructure::Coil),
        ];
        let html = WebViewPage {
            title: "t",
            caption: "c",
            structure: "ATOM\n",
            format: StructureFormat::Pdb,
            color: WebColorScheme::SecondaryStructure,
            secondary_structure: &ss,
        }
        .render();
        assert!(
            html.contains(r#"{"A:1":"h","A:2":"s","B:7":"c"}"#),
            "ss map missing"
        );
        assert!(html.contains("a.ss = s;"), "override is not applied");

        // With no assignment the page must not emit a map at all, so 3Dmol keeps its own.
        let none = page(
            WebColorScheme::SecondaryStructure,
            "ATOM\n",
            StructureFormat::Pdb,
        );
        assert!(none.contains("var dssp = null;"));
    }

    #[test]
    fn caption_and_title_are_escaped() {
        let html = WebViewPage {
            title: "<script>alert(1)</script>",
            caption: "a & b \"quoted\"",
            structure: "ATOM\n",
            format: StructureFormat::Pdb,
            color: WebColorScheme::SecondaryStructure,
            secondary_structure: &[],
        }
        .render();
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(html.contains("a &amp; b &quot;quoted&quot;"));
    }
}
