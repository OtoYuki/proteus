//! The wordmark: "proteus" in a 5 × 9 dot-matrix face (ascender 2, x-height 5, descender 2),
//! extending the s1re.sh M4 Matrix. The same dots everywhere: half-blocks or braille in a
//! terminal, circles in SVG. No font to license or embed.

pub const GLYPH_W: usize = 5;
pub const GLYPH_H: usize = 9;
/// Columns of space between letters.
pub const GAP: usize = 1;

fn glyph(c: char) -> [&'static str; GLYPH_H] {
    match c {
        'p' => [
            ".....", ".....", "####.", "#...#", "#...#", "#...#", "####.", "#....", "#....",
        ],
        'r' => [
            ".....", ".....", "#.##.", "##..#", "#....", "#....", "#....", ".....", ".....",
        ],
        'o' => [
            ".....", ".....", ".###.", "#...#", "#...#", "#...#", ".###.", ".....", ".....",
        ],
        't' => [
            ".....", ".#...", "####.", ".#...", ".#...", ".#...", "..##.", ".....", ".....",
        ],
        'e' => [
            ".....", ".....", ".###.", "#...#", "#####", "#....", ".####", ".....", ".....",
        ],
        'u' => [
            ".....", ".....", "#...#", "#...#", "#...#", "#..##", ".##.#", ".....", ".....",
        ],
        's' => [
            ".....", ".....", ".####", "#....", ".###.", "....#", "####.", ".....", ".....",
        ],
        _ => ["....."; GLYPH_H],
    }
}

/// The word as a dot bitmap, row-major, `(width, height, dots)`.
pub fn bitmap(word: &str) -> (usize, usize, Vec<bool>) {
    let letters: Vec<char> = word.chars().collect();
    let w = if letters.is_empty() {
        0
    } else {
        letters.len() * (GLYPH_W + GAP) - GAP
    };
    let mut dots = vec![false; w * GLYPH_H];
    for (k, c) in letters.iter().enumerate() {
        let g = glyph(*c);
        for (y, row) in g.iter().enumerate() {
            for (x, ch) in row.chars().enumerate() {
                if ch == '#' {
                    dots[y * w + k * (GLYPH_W + GAP) + x] = true;
                }
            }
        }
    }
    (w, GLYPH_H, dots)
}

/// Half-block rendering: one column per dot, two dot rows per line (`▀ ▄ █`).
pub fn half_blocks(word: &str) -> Vec<String> {
    let (w, h, d) = bitmap(word);
    (0..h.div_ceil(2))
        .map(|r| {
            (0..w)
                .map(|x| {
                    let top = d[(2 * r) * w + x];
                    let bottom = 2 * r + 1 < h && d[(2 * r + 1) * w + x];
                    match (top, bottom) {
                        (true, true) => '█',
                        (true, false) => '▀',
                        (false, true) => '▄',
                        (false, false) => ' ',
                    }
                })
                .collect()
        })
        .collect()
}

/// Braille rendering: 2 × 4 dots per cell (the compact form, for a header line).
pub fn braille(word: &str) -> Vec<String> {
    let (w, h, d) = bitmap(word);
    const BITS: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    (0..h.div_ceil(4))
        .map(|r| {
            (0..w.div_ceil(2))
                .map(|c| {
                    let mut bits = 0;
                    for (dx, col) in BITS.iter().enumerate() {
                        for (dy, bit) in col.iter().enumerate() {
                            let (x, y) = (c * 2 + dx, r * 4 + dy);
                            if x < w && y < h && d[y * w + x] {
                                bits |= bit;
                            }
                        }
                    }
                    char::from_u32(0x2800 + bits).unwrap_or(' ')
                })
                .collect()
        })
        .collect()
}

/// SVG circles for each dot, `pitch` apart with radius `r`, starting at the origin.
pub fn svg_dots(word: &str, pitch: f32, r: f32) -> String {
    let (w, h, d) = bitmap(word);
    let mut s = String::new();
    for y in 0..h {
        for x in 0..w {
            if d[y * w + x] {
                s.push_str(&format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{r:.1}"/>"#,
                    x as f32 * pitch + pitch / 2.0,
                    y as f32 * pitch + pitch / 2.0
                ));
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wordmark_has_its_shape() {
        let (w, h, d) = bitmap("proteus");
        assert_eq!((w, h), (41, 9));
        // Only p descends; only t rises above the x-height.
        let row = |y: usize| (0..w).filter(|x| d[y * w + x]).count();
        assert_eq!(row(0), 0);
        assert_eq!(row(1), 1, "the t's ascender");
        assert_eq!(row(7) + row(8), 2, "the p's descender");
        let hb = half_blocks("proteus");
        assert_eq!(hb.len(), 5);
        assert!(hb.iter().all(|l| l.chars().count() == 41));
        let br = braille("proteus");
        assert_eq!(br.len(), 3);
        assert!(br.iter().all(|l| l.chars().count() == 21));
        assert!(svg_dots("proteus", 10.0, 4.0).matches("<circle").count() > 60);
        assert_eq!(bitmap("").0, 0);
    }
}
