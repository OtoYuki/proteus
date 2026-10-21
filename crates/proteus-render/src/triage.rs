//! The campaign page: every model `analyze` measured, as one sortable, filterable table.
//!
//! A binder campaign ends with more models than anyone opens one at a time. This is the same
//! rows `analyze --export` writes, in a self-contained page: sort by any column, filter by
//! name, and copy a `proteus view` command for the row worth looking at. No network, no 3D.

use proteus_core::qc::StructureQc;

const TRIAGE_JS: &str = include_str!("../assets/web/triage.js");
const TRIAGE_CSS: &str = include_str!("../assets/web/triage.css");

/// Columns worth a first look, in order. The rest are in the page and appear with the
/// "every column" switch; `interface_*` ones are dropped when no interface was measured.
const FIRST: &[&str] = &[
    "model",
    "ipsae_min",
    "iptm",
    "ipae",
    "lis",
    "interface_sc",
    "interface_dsasa",
    "interface_hbonds",
    "plddt_mean",
    "rama_favored_pct",
    "heavy_atom_overlap_score",
    "fitness",
    "n_residues",
];

/// What the page says about itself: where the rows came from, and the cutoff to mark.
pub struct TriagePage<'a> {
    /// What was analysed, as the command named it.
    pub subject: &'a str,
    pub rows: &'a [StructureQc],
    /// A line about the ipSAE cutoff for the predictor that wrote these models, when one is
    /// known (`analyze` prints the same sentence).
    pub cutoff_note: Option<&'a str>,
    /// The ipSAE cutoff to mark values at or above, when one applies.
    pub cutoff: Option<f64>,
    /// Version of Proteus, for the footer.
    pub version: &'a str,
    /// When the page was written, as text.
    pub written: &'a str,
}

impl TriagePage<'_> {
    /// The page, as one self-contained HTML string.
    pub fn render(&self) -> String {
        let data = serde_json::to_value(self.rows).unwrap_or(serde_json::Value::Null);
        let present = |col: &str| {
            data.as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|r| !matches!(r.get(col), None | Some(serde_json::Value::Null)))
            })
        };
        // A column nobody measured is noise in a table this wide.
        let first: Vec<&str> = FIRST
            .iter()
            .copied()
            .filter(|c| *c == "model" || present(c))
            .collect();
        let all: Vec<&str> = proteus_core::qc::QC_COLUMNS
            .iter()
            .copied()
            // The sequence is a paragraph, not a cell; the file path is on the model column.
            .filter(|c| !matches!(*c, "sequence" | "file" | "dssp"))
            .collect();
        let has_interface = present("ipsae_min");
        let payload = serde_json::json!({
            "rows": data,
            "columns": first,
            "allColumns": all,
            "sort": if has_interface { "ipsae_min" } else { "fitness" },
            "cutoff": self.cutoff,
            "cutoffColumn": "ipsae_min",
        });
        let esc = crate::web::escape_html;
        let n = self.rows.len();
        let note = match self.cutoff_note {
            Some(t) => format!("<p id=\"note\">{}</p>", esc(t)),
            None => "<p id=\"note\" hidden></p>".to_string(),
        };
        let over = match self.cutoff {
            Some(c) if has_interface => {
                format!("<label><input type=\"checkbox\" id=\"over\"> ipSAE ≥ {c}</label>")
            }
            _ => String::new(),
        };
        format!(
            r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Proteus triage</title>
<style>{brand_css}{css}</style>
</head>
<body>
<header id="top">{brand}<div id="subject"><h1>{title}</h1><p>{n} model{plural} · analysed by Proteus {version}</p></div></header>
{note}
<div id="controls">
<input type="search" id="filter" placeholder="filter by name  /" aria-label="filter by model name">
{over}
<label><input type="checkbox" id="all"> every column</label>
<span class="count" id="count"></span>
</div>
<div id="wrap"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p id="empty" hidden>Nothing matches that filter.</p>
<p id="foot">Click a model to copy a command that opens it. Every number is the one
<code>proteus analyze --export</code> writes; the repository's README defines each column.
Written {written}. Nothing is fetched from the network.</p>
<div id="toast" hidden></div>
<script>window.PROTEUS_TRIAGE={payload};</script>
<script>{js}</script>
</body></html>"#,
            title = esc(self.subject),
            brand_css = crate::web::brand_css(),
            css = TRIAGE_CSS,
            brand = crate::web::brand_header_html(),
            n = n,
            plural = if n == 1 { "" } else { "s" },
            version = esc(self.version),
            written = esc(self.written),
            payload = crate::web::script_safe_json(&payload),
            js = TRIAGE_JS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proteus_core::qc::StructureQc;

    fn row(model: &str, ipsae: Option<f64>) -> StructureQc {
        let mut r = StructureQc {
            file: format!("designs/{model}.cif"),
            model: model.into(),
            n_chains: 2,
            n_residues: 120,
            sequence: "MKT".into(),
            confidence_source: "predicted".into(),
            fitness: 50.0,
            ..Default::default()
        };
        r.interface.ipsae_min = ipsae;
        r
    }

    fn page(rows: &[StructureQc], cutoff: Option<f64>) -> String {
        TriagePage {
            subject: "designs/",
            rows,
            cutoff_note: Some("Boltz-2 fit: 0.766"),
            cutoff,
            version: "0.11.0",
            written: "2026-10-21",
        }
        .render()
    }

    /// The page carries its rows, names the columns worth a first look, and sorts by the
    /// interface score when there is one.
    #[test]
    fn the_page_carries_its_rows_and_sorts_by_the_interface_score() {
        let rows = [row("a", Some(0.8)), row("b", Some(0.2))];
        let html = page(&rows, Some(0.61));
        assert!(html.contains("\"sort\":\"ipsae_min\""), "{html}");
        assert!(html.contains("designs/a.cif") && html.contains("designs/b.cif"));
        assert!(html.contains("2 models ·"));
        assert!(
            html.contains("ipSAE ≥ 0.61"),
            "the cutoff filter is offered"
        );
        assert!(
            html.contains("Boltz-2 fit: 0.766"),
            "the cutoff note is shown"
        );
        // Self-contained: no network of any kind.
        for bad in ["http://", "https://", "//cdn", "<iframe"] {
            assert!(!html.contains(bad), "{bad} in the page");
        }
    }

    /// With no interface measured, its columns are not offered and the triage score sorts.
    #[test]
    fn a_page_without_an_interface_drops_those_columns() {
        let rows = [row("a", None)];
        let html = page(&rows, None);
        assert!(html.contains("\"sort\":\"fitness\""), "{html}");
        let cols = html
            .split("\"columns\":[")
            .nth(1)
            .and_then(|s| s.split(']').next())
            .unwrap_or_default()
            .to_string();
        assert!(cols.contains("\"model\""), "{cols}");
        assert!(!cols.contains("ipsae_min"), "{cols}");
        assert!(
            !html.contains("ipSAE ≥"),
            "no cutoff filter without a cutoff"
        );
        assert!(
            html.contains("1 model ·") && !html.contains("1 models"),
            "singular"
        );
    }

    /// A model name and a path are data, not markup.
    #[test]
    fn a_hostile_name_cannot_escape_the_page() {
        let mut r = row("</script><img src=x onerror=alert(1)>", Some(0.5));
        r.file = "designs/</script>evil.cif".into();
        let html = page(&[r], Some(0.61));
        assert!(!html.contains("<img src=x"), "{html}");
        assert!(!html.contains("</script>evil"), "{html}");
        assert!(
            html.contains("\\u003c/script\\u003e"),
            "escaped in the payload"
        );
    }
}
