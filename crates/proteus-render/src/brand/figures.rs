//! The README's figures, drawn from the validation reports `make validate-binders` and
//! `make validate-nipah` write (`validate/*/last_run.md`), in the brand's light and dark
//! themes. A test keeps `docs/img/` current (run with `UPDATE_README_FIGURES=1` to rewrite).
//!
//! Colour follows emphasis, not identity: ipSAE_min is the one accent mark, everything it is
//! compared with is neutral. Where two things share a row (a score and the random baseline),
//! they differ in shape as well, filled against hollow, and a legend names both.

use super::{css, Theme, DARK, LIGHT};
use std::fmt::Write as _;

const MONO: &str = "Geist Mono, IBM Plex Mono, DejaVu Sans Mono, ui-monospace, monospace";

/// The first Markdown table after the line containing `marker`: its body rows, cells trimmed.
fn table_after(md: &str, marker: &str) -> Vec<Vec<String>> {
    let start = md
        .find(marker)
        .unwrap_or_else(|| panic!("no '{marker}' in the report"));
    md[start..]
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .skip(2)
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
}

fn num(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| panic!("not a number in the report: {s:?}"))
}

/// One target of the Overath et al. set: name, designs, binders, random AP (binder rate),
/// ipSAE_min AP.
pub struct Target {
    pub name: String,
    pub designs: usize,
    pub binders: usize,
    pub random: f64,
    pub ap: f64,
}

pub fn targets(binders_report: &str) -> Vec<Target> {
    let mut t: Vec<Target> = table_after(binders_report, "Per target, ranked by")
        .into_iter()
        .map(|r| Target {
            name: r[0].clone(),
            designs: num(&r[1]) as usize,
            binders: num(&r[2]) as usize,
            random: num(&r[3]),
            ap: num(&r[4]),
        })
        .collect();
    t.sort_by(|a, b| b.ap.total_cmp(&a.ap));
    t
}

/// (label, per-target AP, is ipSAE_min) for the scores the comparison shows, best first.
pub fn metrics(binders_report: &str) -> Vec<(String, f64, bool)> {
    const SHOWN: &[(&str, &str, &str)] = &[
        ("ipSAE_min", "proteus", "ipSAE_min"),
        ("LIS", "proteus", "LIS"),
        ("−ipAE", "proteus", "−ipAE"),
        ("pDockQ2_min", "dataset (AF3)", "pDockQ2"),
        ("ipTM", "proteus (from AF3's file)", "ipTM"),
        ("pLDDT (mean)", "proteus", "pLDDT"),
        ("Sc", "proteus", "Sc"),
        ("actifpTM", "dataset (ColabFold)", "actifpTM"),
        ("−interface ΔG", "dataset (Rosetta)", "Rosetta ΔG"),
        ("dSASA", "proteus", "dSASA"),
    ];
    let rows = table_after(binders_report, "| metric | source | AP per target");
    let mut out: Vec<(String, f64, bool)> = SHOWN
        .iter()
        .map(|(m, src, label)| {
            let r = rows
                .iter()
                .find(|r| r[0] == *m && r[1] == *src)
                .unwrap_or_else(|| panic!("no {m} ({src}) row in the report"));
            (label.to_string(), num(&r[2]), *m == "ipSAE_min")
        })
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// The random per-target AP the report states ("0.131 averaged over the 15 targets").
pub fn random_per_target(binders_report: &str) -> f64 {
    let at = binders_report
        .find(" averaged over the ")
        .expect("no random baseline in the report");
    let head = &binders_report[..at];
    num(head.rsplit(' ').next().unwrap())
}

/// (bin label, designs, binders, binder rate) of the Nipah report's `ipsae_min` bins, and the
/// overall binder rate.
pub fn nipah_bins(nipah_report: &str) -> (Vec<(String, usize, usize, f64)>, f64) {
    let bins = table_after(nipah_report, "Binder rate by")
        .into_iter()
        .map(|r| {
            (
                r[0].clone(),
                num(&r[1]) as usize,
                num(&r[2]) as usize,
                num(&r[3]),
            )
        })
        .collect::<Vec<_>>();
    let at = nipah_report.find("(prevalence ").expect("no prevalence");
    let overall = num(nipah_report[at + 12..].split(',').next().unwrap());
    (bins, overall)
}

fn frame(t: &Theme, w: f64, h: f64, label: &str, body: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {h:.0}" role="img" aria-label="{label}" font-family="{MONO}"><rect width="100%" height="100%" rx="6" fill="{}"/>{body}</svg>"#,
        css(t.ground)
    )
}

fn text(x: f64, y: f64, fill: &str, size: f64, anchor: &str, s: &str) -> String {
    format!(
        r#"<text x="{x:.1}" y="{y:.1}" fill="{fill}" font-size="{size}" text-anchor="{anchor}">{s}</text>"#
    )
}

/// Per target: the random ranking's AP (hollow) and ipSAE_min's (filled), joined.
pub fn per_target_svg(t: &Theme, rows: &[Target]) -> String {
    let (w, top, row, left, right) = (880.0, 70.0, 26.0, 210.0, 70.0);
    let h = top + row * rows.len() as f64 + 44.0;
    let x = |v: f64| left + v * (w - left - right);
    let (acc, muted, dim, line, txt) = (
        css(t.accent),
        css(t.muted),
        css(t.dim),
        css(t.line),
        css(t.text),
    );
    let ground = css(t.ground);
    let mut b = String::new();
    b += &text(
        24.0,
        30.0,
        &txt,
        15.0,
        "start",
        "Ranking each target's designs: average precision",
    );
    // Legend.
    let _ = write!(
        b,
        r#"<circle cx="{:.1}" cy="50" r="5" fill="{acc}"/>"#,
        left
    );
    b += &text(left + 12.0, 54.0, &muted, 12.0, "start", "ipSAE_min");
    let _ = write!(
        b,
        r#"<circle cx="{:.1}" cy="50" r="6" fill="none" stroke="{dim}" stroke-width="2"/>"#,
        left + 110.0
    );
    b += &text(
        left + 122.0,
        54.0,
        &muted,
        12.0,
        "start",
        "random order (the binder rate)",
    );
    // Grid and axis.
    for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let gx = x(v);
        let _ = write!(
            b,
            r#"<line x1="{gx:.1}" y1="{:.1}" x2="{gx:.1}" y2="{:.1}" stroke="{line}" stroke-width="1" opacity="0.5"/>"#,
            top - 8.0,
            top + row * rows.len() as f64
        );
        b += &text(gx, h - 18.0, &dim, 11.0, "middle", &format!("{v:.2}"));
    }
    for (i, r) in rows.iter().enumerate() {
        let cy = top + row * (i as f64 + 0.5);
        b += &text(
            24.0,
            cy + 4.0,
            &txt,
            12.5,
            "start",
            &format!("{:<14}", r.name).replace(' ', "\u{a0}"),
        );
        b += &text(
            left - 14.0,
            cy + 4.0,
            &dim,
            11.0,
            "end",
            &format!("{}/{}", r.binders, r.designs),
        );
        let (x0, x1) = (x(r.random), x(r.ap));
        let _ = write!(
            b,
            r#"<line x1="{x0:.1}" y1="{cy:.1}" x2="{x1:.1}" y2="{cy:.1}" stroke="{line}" stroke-width="2"/><circle cx="{x1:.1}" cy="{cy:.1}" r="5" fill="{acc}" stroke="{ground}" stroke-width="2"/><circle cx="{x0:.1}" cy="{cy:.1}" r="6" fill="none" stroke="{dim}" stroke-width="2"/>"#
        );
        b += &text(
            w - 24.0,
            cy + 4.0,
            &muted,
            11.5,
            "end",
            &format!("{:.2}", r.ap),
        );
    }
    b += &text(24.0, h - 18.0, &dim, 11.0, "start", "binders/designs");
    frame(
        t,
        w,
        h,
        "Average precision of ranking each target's designs by ipSAE_min, against a random order, for the 15 targets of Overath et al. 2025",
        &b,
    )
}

/// Per-target average precision of each score, ipSAE_min highlighted, with the random line.
pub fn metrics_svg(t: &Theme, rows: &[(String, f64, bool)], random: f64) -> String {
    let (w, top, row, left, right) = (880.0, 56.0, 28.0, 150.0, 70.0);
    let h = top + row * rows.len() as f64 + 44.0;
    let max = 0.6;
    let x = |v: f64| left + v / max * (w - left - right);
    let (acc, muted, dim, line, txt) = (
        css(t.accent),
        css(t.muted),
        css(t.dim),
        css(t.line),
        css(t.text),
    );
    let mut b = String::new();
    b += &text(
        24.0,
        30.0,
        &txt,
        15.0,
        "start",
        "Mean average precision per target, by score",
    );
    for v in [0.0, 0.2, 0.4, 0.6] {
        let gx = x(v);
        let _ = write!(
            b,
            r#"<line x1="{gx:.1}" y1="{:.1}" x2="{gx:.1}" y2="{:.1}" stroke="{line}" stroke-width="1" opacity="0.5"/>"#,
            top - 6.0,
            top + row * rows.len() as f64
        );
        b += &text(gx, h - 18.0, &dim, 11.0, "middle", &format!("{v:.1}"));
    }
    for (i, (label, v, hi)) in rows.iter().enumerate() {
        let y = top + row * i as f64 + 5.0;
        let fill = if *hi { &acc } else { &line };
        let weight = if *hi { &txt } else { &muted };
        b += &text(left - 12.0, y + 13.0, weight, 12.5, "end", label);
        let (x0, x1) = (x(0.0), x(*v));
        let (bh, r) = (18.0, 4.0);
        // Square at the baseline, 4 px rounded at the data end.
        let _ = write!(
            b,
            r#"<path d="M{x0:.1} {y:.1} H{:.1} Q{x1:.1} {y:.1} {x1:.1} {:.1} V{:.1} Q{x1:.1} {:.1} {:.1} {:.1} H{x0:.1} Z" fill="{fill}"/>"#,
            x1 - r,
            y + r,
            y + bh - r,
            y + bh,
            x1 - r,
            y + bh
        );
        b += &text(
            x1 + 8.0,
            y + 13.0,
            weight,
            11.5,
            "start",
            &format!("{v:.2}"),
        );
    }
    let rx = x(random);
    let _ = write!(
        b,
        r#"<line x1="{rx:.1}" y1="{:.1}" x2="{rx:.1}" y2="{:.1}" stroke="{txt}" stroke-width="1.5" stroke-dasharray="4 4"/>"#,
        top - 6.0,
        top + row * rows.len() as f64
    );
    b += &text(
        rx + 6.0,
        top - 12.0,
        &muted,
        11.0,
        "start",
        &format!("random {random:.2}"),
    );
    frame(
        t,
        w,
        h,
        "Mean per-target average precision of ten scores on the 3 669 lab-tested designs of Overath et al. 2025; ipSAE_min is highest",
        &b,
    )
}

/// Nipah: the binder rate in each `ipsae_min` bin, with the overall rate.
pub fn nipah_svg(t: &Theme, bins: &[(String, usize, usize, f64)], overall: f64) -> String {
    let (w, h, top, bottom, left, right) = (880.0, 340.0, 60.0, 70.0, 70.0, 24.0);
    let max = 0.4;
    let plot_h = h - top - bottom;
    let y = |v: f64| top + plot_h * (1.0 - v / max);
    let slot = (w - left - right) / bins.len() as f64;
    let (acc, muted, dim, line, txt) = (
        css(t.accent),
        css(t.muted),
        css(t.dim),
        css(t.line),
        css(t.text),
    );
    let mut b = String::new();
    b += &text(
        24.0,
        30.0,
        &txt,
        15.0,
        "start",
        "Nipah: share of designs that bound, by ipsae_min",
    );
    for v in [0.0, 0.1, 0.2, 0.3, 0.4] {
        let gy = y(v);
        let _ = write!(
            b,
            r#"<line x1="{left:.1}" y1="{gy:.1}" x2="{:.1}" y2="{gy:.1}" stroke="{line}" stroke-width="1" opacity="0.5"/>"#,
            w - right
        );
        b += &text(
            left - 10.0,
            gy + 4.0,
            &dim,
            11.0,
            "end",
            &format!("{:.0}%", v * 100.0),
        );
    }
    for (i, (label, designs, binders, rate)) in bins.iter().enumerate() {
        let cx = left + slot * (i as f64 + 0.5);
        let (bw, r) = (slot * 0.5, 4.0);
        let (x0, x1, y0, y1) = (cx - bw / 2.0, cx + bw / 2.0, y(0.0), y(*rate));
        let _ = write!(
            b,
            r#"<path d="M{x0:.1} {y0:.1} V{:.1} Q{x0:.1} {y1:.1} {:.1} {y1:.1} H{:.1} Q{x1:.1} {y1:.1} {x1:.1} {:.1} V{y0:.1} Z" fill="{acc}"/>"#,
            y1 + r,
            x0 + r,
            x1 - r,
            y1 + r
        );
        b += &text(
            cx,
            y1 - 8.0,
            &txt,
            12.0,
            "middle",
            &format!("{:.0}%", rate * 100.0),
        );
        b += &text(cx, h - bottom + 20.0, &muted, 12.0, "middle", label);
        b += &text(
            cx,
            h - bottom + 38.0,
            &dim,
            11.0,
            "middle",
            &format!("{binders} of {designs}"),
        );
    }
    let oy = y(overall);
    let _ = write!(
        b,
        r#"<line x1="{left:.1}" y1="{oy:.1}" x2="{:.1}" y2="{oy:.1}" stroke="{txt}" stroke-width="1.5" stroke-dasharray="4 4"/>"#,
        w - right
    );
    b += &text(
        left + 6.0,
        oy - 8.0,
        &muted,
        11.0,
        "start",
        &format!("all designs {:.1}%", overall * 100.0),
    );
    frame(
        t,
        w,
        h,
        "Binder rate of the 1 196 Nipah designs in six ipsae_min bins, rising from 3% below 0.2 to 38% above 0.8",
        &b,
    )
}

/// Every figure: (file name, SVG), light and dark, from the two reports' text.
pub fn all(binders_report: &str, nipah_report: &str) -> Vec<(String, String)> {
    let tg = targets(binders_report);
    let mt = metrics(binders_report);
    let rnd = random_per_target(binders_report);
    let (bins, overall) = nipah_bins(nipah_report);
    let mut out = Vec::new();
    for (name, t) in [("light", &LIGHT), ("dark", &DARK)] {
        out.push((format!("per-target-{name}.svg"), per_target_svg(t, &tg)));
        out.push((format!("scores-{name}.svg"), metrics_svg(t, &mt, rnd)));
        out.push((
            format!("nipah-bins-{name}.svg"),
            nipah_svg(t, &bins, overall),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(rel: &str) -> String {
        std::fs::read_to_string(format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[test]
    fn the_reports_parse_to_what_they_state() {
        let b = read("validate/binders/last_run.md");
        let t = targets(&b);
        assert_eq!(t.len(), 15);
        let mean = t.iter().map(|t| t.ap).sum::<f64>() / t.len() as f64;
        assert!((mean - 0.513).abs() < 0.001, "{mean}");
        assert_eq!(t.iter().map(|t| t.binders).sum::<usize>(), 394);
        assert_eq!(random_per_target(&b), 0.131);
        assert_eq!(metrics(&b)[0].0, "ipSAE_min");
        let (bins, overall) = nipah_bins(&read("validate/nipah/last_run.md"));
        assert_eq!(bins.iter().map(|b| b.1).sum::<usize>(), 1196);
        assert_eq!(overall, 0.093);
    }

    #[test]
    fn readme_figures_are_current() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/img");
        let update = std::env::var_os("UPDATE_README_FIGURES").is_some();
        for (name, svg) in all(
            &read("validate/binders/last_run.md"),
            &read("validate/nipah/last_run.md"),
        ) {
            assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
            let path = format!("{dir}/{name}");
            if update {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(&path, &svg).unwrap();
            } else {
                let on_disk = std::fs::read_to_string(&path)
                    .unwrap_or_else(|_| panic!("{path} missing: run with UPDATE_README_FIGURES=1"));
                assert_eq!(
                    on_disk, svg,
                    "{name} is stale: run with UPDATE_README_FIGURES=1"
                );
            }
        }
    }
}
