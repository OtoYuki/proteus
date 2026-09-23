//! The brand assets as SVG, generated from the same geometry the terminal draws, and kept
//! current in `docs/brand/` by a test (run with `UPDATE_BRAND_ASSETS=1` to rewrite them).

use super::{css, mark, matrix, Theme, DARK, LIGHT, SIGNATURE};

/// The mark alone in a 100 × 100 view box: filled chain on a dark ground, outlined on a light
/// one; the ligand ring in Clay.
pub fn mark_svg(theme: &Theme, filled: bool) -> String {
    let (lx, ly, lr) = mark::LIGAND;
    let chain = if filled {
        format!(
            r#"<path d="{}" fill="{}" fill-rule="evenodd"/>"#,
            mark::svg_path(240),
            css(theme.accent)
        )
    } else {
        format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="1.6" stroke-linejoin="round"/>"#,
            mark::svg_path(240),
            css(theme.text)
        )
    };
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Proteus">{chain}<circle cx="{lx}" cy="{ly}" r="{lr}" fill="none" stroke="{}" stroke-width="1.6"/></svg>"#,
        css(theme.warm)
    )
}

/// Mark, dot-matrix wordmark and signature, on the theme's ground.
pub fn lockup_svg(theme: &Theme) -> String {
    let dark = theme.ground == DARK.ground;
    let mark = mark_svg(theme, dark)
        .replace(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Proteus">"#, "")
        .replace("</svg>", "");
    let pitch = 7.0;
    let (ww, _, _) = matrix::bitmap("proteus");
    let word_w = ww as f32 * pitch;
    let (w, h) = (150.0 + word_w + 60.0, 150.0);
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0} {h:.0}" role="img" aria-label="Proteus, a s1re.sh project"><rect width="100%" height="100%" fill="{ground}"/><g transform="translate(25 25)">{mark}</g><g transform="translate(150 38)" fill="{text}">{dots}</g><text x="150" y="126" fill="{muted}" font-family="Geist Mono, IBM Plex Mono, ui-monospace, monospace" font-size="10.5" letter-spacing="1.6">{sig}</text></svg>"#,
        ground = css(theme.ground),
        text = css(theme.text),
        muted = css(theme.muted),
        dots = matrix::svg_dots("proteus", pitch, pitch * 0.36),
        sig = SIGNATURE.to_uppercase(),
    )
}

/// Every asset: (file name, contents).
pub fn all() -> Vec<(&'static str, String)> {
    vec![
        ("proteus-mark.svg", mark_svg(&DARK, true)),
        ("proteus-mark-outline.svg", mark_svg(&LIGHT, false)),
        ("proteus-lockup-dark.svg", lockup_svg(&DARK)),
        ("proteus-lockup-light.svg", lockup_svg(&LIGHT)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brand_assets_are_current() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/brand");
        let update = std::env::var_os("UPDATE_BRAND_ASSETS").is_some();
        for (name, svg) in all() {
            assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
            let path = format!("{dir}/{name}");
            if update {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(&path, &svg).unwrap();
            } else {
                let on_disk = std::fs::read_to_string(&path)
                    .unwrap_or_else(|_| panic!("{path} missing: run with UPDATE_BRAND_ASSETS=1"));
                assert_eq!(
                    on_disk, svg,
                    "{name} is stale: run with UPDATE_BRAND_ASSETS=1"
                );
            }
        }
    }
}
