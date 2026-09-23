//! DEC Sixel encoder.
//!
//! Sixel reaches terminals the kitty protocol does not — xterm (with `-ti vt340`), mlterm,
//! foot, contour, Windows Terminal, WezTerm, and recent VTE builds — and it is the only true
//! pixel path available on several of them. On a terminal that speaks neither, the caller falls
//! back to half-blocks or Braille.
//!
//! The format, briefly: a `\x1bPq` introducer, raster attributes, a colour palette, then bands
//! six pixels tall. Within a band each byte carries six vertical pixels as a bitmask offset by
//! `?` (0x3F), `$` returns to the start of the band to overlay the next colour, and `-` advances
//! to the next band.
//!
//! Sixel is palette-indexed, so the 24-bit framebuffer is quantised (median cut, ≤ 256 colours).
//! That loss is measured rather than assumed: [`SixelRenderer::encode_with_stats`] reports the
//! palette size and the worst per-channel error, and the tests hold it to a bound *and* decode
//! the output with libsixel's own `sixel2png` to confirm an independent implementation agrees.

use crate::rasterizer::buffer::{ColorRGB, Framebuffer};

/// Sixel palettes are conventionally capped here; every terminal that speaks Sixel at all
/// supports 256 registers.
const MAX_COLORS: usize = 256;

/// What the quantiser gave up, so a caller can report it instead of guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantisationStats {
    /// Colours in the emitted palette.
    pub palette_len: usize,
    /// Largest absolute per-channel difference between a source pixel and its palette entry.
    pub max_channel_error: u8,
}

pub struct SixelRenderer;

impl SixelRenderer {
    /// Whether the terminal is likely to render Sixel.
    ///
    /// There is no reliable environment variable for this — the authoritative check is a DA1
    /// query (`\x1b[c`) and looking for `;4;` in the reply, which needs a live terminal and a
    /// read timeout. This is the conservative name check; the caller warns rather than refuses,
    /// because a false negative here should not stop someone who knows their terminal.
    pub fn is_supported() -> bool {
        if std::env::var_os("SIXEL").is_some() {
            return true;
        }
        match std::env::var("TERM") {
            Ok(term) => {
                let t = term.to_ascii_lowercase();
                t.contains("sixel")
                    || t.contains("foot")
                    || t.contains("contour")
                    || t.contains("mlterm")
                    || t.contains("wezterm")
                    || t.starts_with("xterm-kitty")
                    || t == "xterm-256color" && std::env::var_os("WT_SESSION").is_some()
            }
            Err(_) => false,
        }
    }

    /// Encode a framebuffer as one Sixel image.
    pub fn render_snapshot(fb: &Framebuffer) -> String {
        Self::encode_with_stats(fb).0
    }

    /// As [`Self::render_snapshot`], also reporting what the palette cost.
    pub fn encode_with_stats(fb: &Framebuffer) -> (String, QuantisationStats) {
        let (palette, indices, max_channel_error) = quantise(&fb.colors);
        let mut out = String::with_capacity(fb.width * fb.height / 4 + 1024);

        // P1=0 (pixels of aspect 1:1 via the raster attribute), P2=1 (leave background
        // untouched rather than painting it), P3=0.
        out.push_str("\x1bP0;1;0q");
        // "Pan;Pad;Ph;Pv — square pixels, explicit size so the terminal reserves the right cells.
        out.push_str(&format!("\"1;1;{};{}", fb.width, fb.height));

        for (i, c) in palette.iter().enumerate() {
            // Sixel colour components are percentages, not 0-255.
            out.push_str(&format!(
                "#{};2;{};{};{}",
                i,
                (c.r as u32 * 100).div_ceil(255),
                (c.g as u32 * 100).div_ceil(255),
                (c.b as u32 * 100).div_ceil(255)
            ));
        }

        let mut band = vec![0u8; fb.width];
        for band_top in (0..fb.height).step_by(6) {
            let rows = 6.min(fb.height - band_top);

            // Which palette entries appear in this band. Emitting only those keeps the output
            // near the size of the image rather than 256 passes over every band.
            let mut present = vec![false; palette.len()];
            for y in band_top..band_top + rows {
                for x in 0..fb.width {
                    present[indices[y * fb.width + x] as usize] = true;
                }
            }

            let mut first_pass = true;
            for (colour, _) in present.iter().enumerate().filter(|(_, p)| **p) {
                band.iter_mut().for_each(|b| *b = 0);
                for (row, y) in (band_top..band_top + rows).enumerate() {
                    for x in 0..fb.width {
                        if indices[y * fb.width + x] as usize == colour {
                            band[x] |= 1 << row;
                        }
                    }
                }
                if !first_pass {
                    out.push('$'); // overlay this colour on the same band
                }
                first_pass = false;
                out.push_str(&format!("#{colour}"));
                emit_band(&band, &mut out);
            }
            out.push('-'); // next band
        }

        out.push_str("\x1b\\");
        (
            out,
            QuantisationStats {
                palette_len: palette.len(),
                max_channel_error,
            },
        )
    }
}

/// Run-length encode one band. `!<n><char>` repeats, which matters: a protein render is mostly
/// background, and without this the payload is one byte per pixel per colour pass.
fn emit_band(band: &[u8], out: &mut String) {
    let mut i = 0;
    while i < band.len() {
        let value = band[i];
        let mut run = 1;
        while i + run < band.len() && band[i + run] == value {
            run += 1;
        }
        let ch = (b'?' + value) as char;
        // Three characters is the break-even point for `!n<char>`.
        if run > 3 {
            out.push_str(&format!("!{run}{ch}"));
        } else {
            for _ in 0..run {
                out.push(ch);
            }
        }
        i += run;
    }
}

/// Median-cut quantisation to at most [`MAX_COLORS`] entries.
///
/// Returns the palette, a palette index per pixel, and the worst per-channel error, so the
/// caller can state the cost rather than hide it. An image with few enough distinct colours —
/// which a cartoon render usually is — passes through exactly, with a reported error of 0.
///
/// Above the budget, colours are weighted by how many pixels use them, and the most common
/// colour — the background, in a render — keeps a register of its own. Unweighted, the one
/// background colour counted the same as each of hundreds of rarely used shading tones, was
/// averaged into a box with dark greys, and a black background came out tinted (15,15,13) on
/// 6VXX.
fn quantise(colors: &[ColorRGB]) -> (Vec<ColorRGB>, Vec<u8>, u8) {
    use std::collections::HashMap;

    let mut distinct: Vec<(ColorRGB, u64)> = {
        let mut seen: HashMap<(u8, u8, u8), (ColorRGB, u64)> = HashMap::new();
        for c in colors {
            seen.entry((c.r, c.g, c.b)).or_insert((*c, 0)).1 += 1;
        }
        seen.into_values().collect()
    };
    distinct.sort_by_key(|(c, _)| (c.r, c.g, c.b));

    let palette = if distinct.len() <= MAX_COLORS {
        distinct.into_iter().map(|(c, _)| c).collect()
    } else {
        let dominant = distinct
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, n))| *n)
            .map(|(i, _)| i)
            .expect("more than MAX_COLORS colours");
        let (background, _) = distinct.remove(dominant);
        let mut palette = median_cut(&distinct, MAX_COLORS - 1);
        palette.push(background);
        palette
    };

    // Exact-match fast path, then nearest neighbour for anything the cut merged. Built in
    // reverse so that, should a box average coincide with the reserved colour, the first
    // register wins; either is exact.
    let exact: HashMap<(u8, u8, u8), u8> = palette
        .iter()
        .enumerate()
        .rev()
        .map(|(i, c)| ((c.r, c.g, c.b), i as u8))
        .collect();

    let mut indices = Vec::with_capacity(colors.len());
    let mut max_err = 0u8;
    let mut cache: HashMap<(u8, u8, u8), u8> = HashMap::new();
    for c in colors {
        let key = (c.r, c.g, c.b);
        let idx = if let Some(&i) = exact.get(&key) {
            i
        } else if let Some(&i) = cache.get(&key) {
            i
        } else {
            let i = nearest(&palette, *c);
            cache.insert(key, i);
            i
        };
        let p = palette[idx as usize];
        let err = (p.r.abs_diff(c.r))
            .max(p.g.abs_diff(c.g))
            .max(p.b.abs_diff(c.b));
        max_err = max_err.max(err);
        indices.push(idx);
    }
    (palette, indices, max_err)
}

fn nearest(palette: &[ColorRGB], c: ColorRGB) -> u8 {
    let mut best = 0usize;
    let mut best_d = u32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let d = (p.r as i32 - c.r as i32).pow(2) as u32
            + (p.g as i32 - c.g as i32).pow(2) as u32
            + (p.b as i32 - c.b as i32).pow(2) as u32;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as u8
}

/// Median cut over pixel-weighted colours: repeatedly split the box with the widest channel
/// at that channel's weighted median (half the box's *pixels* on each side, not half its
/// distinct colours), then take each box's pixel-weighted mean.
fn median_cut(colors: &[(ColorRGB, u64)], target: usize) -> Vec<ColorRGB> {
    let mut boxes: Vec<Vec<(ColorRGB, u64)>> = vec![colors.to_vec()];
    while boxes.len() < target {
        let Some((idx, channel)) = widest_box(&boxes) else {
            break;
        };
        let mut b = boxes.swap_remove(idx);
        b.sort_by_key(|(c, _)| match channel {
            0 => c.r,
            1 => c.g,
            _ => c.b,
        });
        let total: u64 = b.iter().map(|(_, n)| n).sum();
        let mut seen = 0u64;
        let mut mid = b.len();
        for (i, (_, n)) in b.iter().enumerate() {
            seen += n;
            if seen * 2 >= total {
                mid = i + 1;
                break;
            }
        }
        // Both halves non-empty (the box has at least two colours).
        let mid = mid.clamp(1, b.len() - 1);
        let hi = b.split_off(mid);
        boxes.push(b);
        boxes.push(hi);
    }
    boxes
        .into_iter()
        .filter(|b| !b.is_empty())
        .map(|b| {
            let n: u64 = b.iter().map(|(_, w)| w).sum();
            let (r, g, bl) = b.iter().fold((0u64, 0u64, 0u64), |acc, (c, w)| {
                (
                    acc.0 + c.r as u64 * w,
                    acc.1 + c.g as u64 * w,
                    acc.2 + c.b as u64 * w,
                )
            });
            let mean = |sum: u64| ((sum + n / 2) / n) as u8;
            ColorRGB::new(mean(r), mean(g), mean(bl))
        })
        .collect()
}

/// The box with the largest single-channel spread, and which channel that is.
fn widest_box(boxes: &[Vec<(ColorRGB, u64)>]) -> Option<(usize, u8)> {
    let mut best: Option<(usize, u8, u8)> = None;
    for (i, b) in boxes.iter().enumerate() {
        if b.len() < 2 {
            continue;
        }
        let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
        for (c, _) in b {
            for (k, v) in [c.r, c.g, c.b].into_iter().enumerate() {
                lo[k] = lo[k].min(v);
                hi[k] = hi[k].max(v);
            }
        }
        for k in 0..3 {
            let spread = hi[k] - lo[k];
            if best.is_none_or(|(_, _, s)| spread > s) {
                best = Some((i, k as u8, spread));
            }
        }
    }
    best.map(|(i, c, _)| (i, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize) -> Framebuffer {
        let mut fb = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if (x + y) % 2 == 0 {
                    ColorRGB::new(255, 0, 0)
                } else {
                    ColorRGB::new(0, 0, 255)
                };
                fb.set_pixel(x, y, c, 0.0);
            }
        }
        fb
    }

    #[test]
    fn output_is_a_well_formed_sixel_image() {
        let s = SixelRenderer::render_snapshot(&checkerboard(8, 12));
        assert!(s.starts_with("\x1bP"), "missing DCS introducer");
        assert!(s.ends_with("\x1b\\"), "missing string terminator");
        assert!(
            s.contains("\"1;1;8;12"),
            "missing raster attributes: {}",
            &s[..40]
        );
        assert!(s.contains("#0;2;"), "missing palette definition");
        // Every data byte must be in the sixel range or a control character.
        let body = &s[s.find('q').unwrap() + 1..s.len() - 2];
        for ch in body.chars() {
            assert!(
                ('?'..='~').contains(&ch) || "#;$-!\"0123456789".contains(ch),
                "byte {ch:?} is not legal inside a sixel payload"
            );
        }
    }

    #[test]
    fn a_small_palette_is_reproduced_exactly() {
        let (_, stats) = SixelRenderer::encode_with_stats(&checkerboard(16, 12));
        assert_eq!(stats.palette_len, 2, "two colours should need two entries");
        assert_eq!(
            stats.max_channel_error, 0,
            "an image within the palette budget must not lose colour"
        );
    }

    #[test]
    fn a_large_palette_is_quantised_within_a_stated_bound() {
        // A smooth gradient has far more than 256 distinct colours, so the cut must engage.
        let (w, h) = (64, 48);
        let mut fb = Framebuffer::new(w, h);
        for y in 0..h {
            for x in 0..w {
                fb.set_pixel(
                    x,
                    y,
                    ColorRGB::new((x * 4) as u8, (y * 5) as u8, ((x + y) * 2) as u8),
                    0.0,
                );
            }
        }
        let (_, stats) = SixelRenderer::encode_with_stats(&fb);
        assert!(stats.palette_len <= MAX_COLORS, "{stats:?}");
        assert!(
            stats.max_channel_error <= 32,
            "quantisation lost more than a stated eighth of the range: {stats:?}"
        );
    }

    /// The encoder is checked by **libsixel's own decoder**, not by this module's idea of the
    /// format. `sixel2png` is the reference implementation's round-trip: if it reproduces the
    /// framebuffer pixel for pixel, the bytes are right in a way our own parser could never
    /// prove.
    ///
    /// Ignored by default because it shells out; `cargo test -p proteus-render -- --ignored`
    /// and `make validate` run it where libsixel is installed.
    #[test]
    #[ignore]
    fn libsixel_decodes_our_output_back_to_the_same_pixels() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        if Command::new("sixel2png").arg("-V").output().is_err() {
            eprintln!("libsixel's sixel2png not installed; skipping");
            return;
        }

        let fb = checkerboard(24, 18);
        let (encoded, stats) = SixelRenderer::encode_with_stats(&fb);
        assert_eq!(stats.max_channel_error, 0, "fixture should be lossless");

        let dir = std::env::temp_dir().join(format!("proteus-sixel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("roundtrip.png");

        let mut child = Command::new("sixel2png")
            .arg("-o")
            .arg(&png)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sixel2png");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(encoded.as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "sixel2png rejected our output: {}",
            String::from_utf8_lossy(&out.stderr)
        );

        // Decode the PNG without pulling an image crate into the dependency graph: ask Python,
        // which the validation harness already requires.
        let script = format!(
            "from PIL import Image; im = Image.open(r'{}').convert('RGB');              print(im.width, im.height);              print(' '.join('%d,%d,%d' % im.getpixel((x, y)) for y in range(im.height)              for x in range(im.width)))",
            png.display()
        );
        let py = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../validate/.venv/bin/python");
        let Ok(decoded) = Command::new(&py).arg("-c").arg(&script).output() else {
            eprintln!("validate venv not built; skipping pixel comparison");
            return;
        };
        if !decoded.status.success() {
            eprintln!("PIL unavailable; skipping pixel comparison");
            return;
        }
        let text = String::from_utf8_lossy(&decoded.stdout);
        let mut lines = text.lines();
        let dims: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            (dims[0], dims[1]),
            (fb.width, fb.height),
            "libsixel decoded a different image size"
        );

        let pixels: Vec<ColorRGB> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|t| {
                let v: Vec<u8> = t.split(',').map(|n| n.parse().unwrap()).collect();
                ColorRGB::new(v[0], v[1], v[2])
            })
            .collect();
        assert_eq!(pixels.len(), fb.colors.len());
        for (i, (got, want)) in pixels.iter().zip(fb.colors.iter()).enumerate() {
            // Sixel colour registers are percentages, so a channel can land one step off.
            let d = got
                .r
                .abs_diff(want.r)
                .max(got.g.abs_diff(want.g))
                .max(got.b.abs_diff(want.b));
            assert!(
                d <= 3,
                "pixel {i} at ({}, {}) differs by {d}: decoded {got:?}, rendered {want:?}",
                i % fb.width,
                i / fb.width
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// On a real render the palette budget is exceeded (shading gives far more than 256
    /// tones) and the background must still come out exactly black. Before colours were
    /// weighted by pixel count, the background — one colour covering most of the frame —
    /// was averaged with dark shading tones into a tinted grey, the whole frame's backdrop.
    #[test]
    fn background_survives_quantisation_of_a_real_render() {
        let text = include_str!("../../../proteus-core/tests/data/1crn.pdb");
        let data = crate::parse_pdb_structure(text).unwrap();
        let (w, h) =
            crate::viewport_pixels(110, 30, crate::terminal::TerminalBackend::Sixel).unwrap();
        let mut fb = Framebuffer::new(w, h);
        fb.clear(ColorRGB::BLACK);
        let scheme = crate::rasterizer::ColorScheme::Rainbow;
        let mut rasterizer = crate::rasterizer::Rasterizer::new(scheme);
        rasterizer.rasterize_mesh(&data.ribbon_mesh, &data.camera, &mut fb, scheme);
        rasterizer.apply_post_processing(&mut fb);

        let distinct: std::collections::HashSet<_> =
            fb.colors.iter().map(|c| (c.r, c.g, c.b)).collect();
        assert!(
            distinct.len() > MAX_COLORS,
            "fixture has only {} colours, so the cut is not exercised",
            distinct.len()
        );

        let (palette, indices, _) = quantise(&fb.colors);
        assert!(palette.len() <= MAX_COLORS);
        assert_eq!(
            fb.colors[0],
            ColorRGB::BLACK,
            "corner pixel is not background"
        );
        for (src, &idx) in fb.colors.iter().zip(&indices) {
            if *src == ColorRGB::BLACK {
                assert_eq!(
                    palette[idx as usize],
                    ColorRGB::BLACK,
                    "background quantised to a tint"
                );
            }
        }
        // And on the wire: the register the corner uses is defined as 0% 0% 0%.
        let encoded = SixelRenderer::render_snapshot(&fb);
        let register = indices[0];
        assert!(
            encoded.contains(&format!("#{register};2;0;0;0#")),
            "background register {register} is not defined as black"
        );
    }

    #[test]
    fn run_length_encoding_shortens_flat_runs() {
        let mut fb = Framebuffer::new(64, 6);
        for x in 0..64 {
            fb.set_pixel(x, 0, ColorRGB::new(10, 20, 30), 0.0);
        }
        let s = SixelRenderer::render_snapshot(&fb);
        assert!(
            s.contains('!'),
            "a 64-pixel flat run was not run-length encoded"
        );
        assert!(
            s.len() < 64 * 3,
            "payload {} bytes for a flat 64px row is not compressed",
            s.len()
        );
    }
}
