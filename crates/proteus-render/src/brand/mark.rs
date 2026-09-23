//! The Proteus mark: a chain of twelve beads (residues) that folds into a lowercase **p**.
//!
//! One signed-distance definition, rendered three ways: terminal braille (the home screen's
//! launch, where the chain folds over about 0.9 s), an SVG path traced by marching squares
//! (brand assets, the browser page) and a boolean bitmap for tests. Coordinates are in a
//! 100 × 100 box, y down.

/// Bead centres and radii of the folded mark: the stem from the bottom up, then the bowl
/// clockwise until it closes against the stem.
pub const FOLD: [(f32, f32, f32); 12] = [
    (30.0, 92.0, 7.2),
    (30.0, 77.0, 7.6),
    (30.0, 62.0, 8.0),
    (31.0, 46.0, 8.4),
    (38.0, 31.0, 8.6),
    (53.0, 22.0, 8.8),
    (69.0, 25.0, 8.8),
    (80.0, 38.0, 8.4),
    (81.0, 54.0, 8.0),
    (71.0, 66.0, 7.6),
    (56.0, 70.0, 7.2),
    (43.0, 64.0, 6.4),
];

/// The ligand: one ring off the chain, the mark's single warm accent.
pub const LIGAND: (f32, f32, f32) = (84.0, 86.0, 4.6);

/// Smoothing between consecutive beads (polynomial smooth-min width).
const NECK: f32 = 3.2;

/// How long the fold takes on launch.
pub const FOLD_SECONDS: f32 = 0.9;

/// Bead `i` at fold progress `t` ∈ [0, 1]: from an extended chain (a low sine across the box,
/// beads at 62 % size) to the folded mark, with a cubic ease-out.
pub fn bead(i: usize, t: f32) -> (f32, f32, f32) {
    let n = FOLD.len();
    let x0 = 6.0 + 88.0 * i as f32 / (n - 1) as f32;
    let y0 = 50.0 + 8.0 * (i as f32 * 1.1 + 0.4).sin();
    let (x1, y1, r1) = FOLD[i];
    let r0 = r1 * 0.62;
    let e = ease_out_cubic(t.clamp(0.0, 1.0));
    (x0 + (x1 - x0) * e, y0 + (y1 - y0) * e, r0 + (r1 - r0) * e)
}

pub fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Signed distance to the chain at progress `t` (negative inside). Only consecutive beads are
/// smoothed together, so the mark reads as a chain with waists rather than a blob.
pub fn chain_distance(x: f32, y: f32, t: f32) -> f32 {
    let mut prev = f32::MAX;
    let mut best = f32::MAX;
    for i in 0..FOLD.len() {
        let (cx, cy, r) = bead(i, t);
        let d = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r;
        if i > 0 {
            best = best.min(smin(prev, d, NECK));
        }
        prev = d;
    }
    best
}

fn ligand_distance(x: f32, y: f32) -> f32 {
    let (cx, cy, r) = LIGAND;
    ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r
}

/// What a sample of the mark is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ink {
    None,
    Chain,
    Ligand,
}

/// Sample the mark on a `w` × `h` grid covering the 100 × 100 box (non-square grids keep the
/// aspect by letterboxing). `outline` draws a ring of the given width instead of a fill. The
/// ligand appears only once the fold has settled (t ≥ 0.85).
pub fn sample(w: usize, h: usize, t: f32, outline: Option<f32>) -> Vec<Ink> {
    let mut out = vec![Ink::None; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    let scale = 100.0 / w.min(h) as f32;
    let (ox, oy) = (
        (w as f32 * scale - 100.0) / 2.0,
        (h as f32 * scale - 100.0) / 2.0,
    );
    let inside = |d: f32| match outline {
        Some(width) => d.abs() <= width / 2.0,
        None => d <= 0.0,
    };
    for j in 0..h {
        for i in 0..w {
            let x = (i as f32 + 0.5) * scale - ox;
            let y = (j as f32 + 0.5) * scale - oy;
            out[j * w + i] = if inside(chain_distance(x, y, t)) {
                Ink::Chain
            } else if t >= 0.85 && ligand_distance(x, y).abs() <= 1.0_f32.max(scale * 0.6) {
                Ink::Ligand
            } else {
                Ink::None
            };
        }
    }
    out
}

/// One line of braille per four sample rows: each cell is 2 × 4 dots. A cell with any ligand
/// dot is reported as `Ink::Ligand` so the caller can colour it.
pub fn braille(cols: usize, rows: usize, t: f32) -> Vec<Vec<(char, Ink)>> {
    let (w, h) = (cols * 2, rows * 4);
    let s = sample(w, h, t, None);
    // Unicode braille dot bits for (dx, dy).
    const BITS: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let mut bits = 0;
                    let mut ink = Ink::None;
                    for (dx, col) in BITS.iter().enumerate() {
                        for (dy, bit) in col.iter().enumerate() {
                            match s[(r * 4 + dy) * w + c * 2 + dx] {
                                Ink::None => {}
                                Ink::Chain => {
                                    bits |= bit;
                                    if ink == Ink::None {
                                        ink = Ink::Chain;
                                    }
                                }
                                Ink::Ligand => {
                                    bits |= bit;
                                    ink = Ink::Ligand;
                                }
                            }
                        }
                    }
                    // An empty cell is a space, not U+2800: some fonts draw the blank braille
                    // pattern as faint dots.
                    let ch = if bits == 0 {
                        ' '
                    } else {
                        char::from_u32(0x2800 + bits).unwrap_or(' ')
                    };
                    (ch, ink)
                })
                .collect()
        })
        .collect()
}

/// The folded chain's outline as an SVG path (`d`), traced by marching squares on a `n` × `n`
/// grid over the 100 × 100 box and closed into loops.
pub fn svg_path(n: usize) -> String {
    let step = 100.0 / n as f32;
    let f = |i: usize, j: usize| chain_distance(i as f32 * step, j as f32 * step, 1.0);
    let field: Vec<f32> = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| (i, j)))
        .map(|(i, j)| f(i, j))
        .collect();
    let at = |i: usize, j: usize| field[j * (n + 1) + i];
    // Edge crossing, linearly interpolated.
    let cross = |(i0, j0): (usize, usize), (i1, j1): (usize, usize)| {
        let (a, b) = (at(i0, j0), at(i1, j1));
        let t = a / (a - b);
        (
            (i0 as f32 + (i1 as f32 - i0 as f32) * t) * step,
            (j0 as f32 + (j1 as f32 - j0 as f32) * t) * step,
        )
    };
    // A crossing is identified by its grid edge (corners in a fixed order), never by its float
    // coordinates: two cells sharing an edge must agree on the point exactly.
    type Key = (usize, usize, usize, usize);
    let edge_key = |a: (usize, usize), b: (usize, usize)| -> Key {
        let (p, q) = if a <= b { (a, b) } else { (b, a) };
        (p.0, p.1, q.0, q.1)
    };
    // Undirected: each crossing point joins exactly two segments on a closed contour, so the
    // walk needs no orientation table.
    let mut adj: std::collections::HashMap<Key, Vec<Key>> = Default::default();
    let mut point: std::collections::HashMap<Key, (f32, f32)> = Default::default();
    for j in 0..n {
        for i in 0..n {
            let c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let inside: Vec<bool> = c.iter().map(|&(a, b)| at(a, b) < 0.0).collect();
            // Edges whose ends differ: 0 top, 1 right, 2 bottom, 3 left.
            let crossings: Vec<usize> = (0..4)
                .filter(|&e| inside[e] != inside[(e + 1) % 4])
                .collect();
            let pairs: Vec<(usize, usize)> = match crossings.len() {
                2 => vec![(crossings[0], crossings[1])],
                4 => {
                    // Saddle: decide by the centre value which corners connect.
                    let centre = (at(c[0].0, c[0].1)
                        + at(c[1].0, c[1].1)
                        + at(c[2].0, c[2].1)
                        + at(c[3].0, c[3].1))
                        / 4.0;
                    if (centre < 0.0) == inside[0] {
                        vec![(0, 1), (2, 3)]
                    } else {
                        vec![(3, 0), (1, 2)]
                    }
                }
                _ => vec![],
            };
            for (a, b) in pairs {
                let (kp, kq) = (
                    edge_key(c[a], c[(a + 1) % 4]),
                    edge_key(c[b], c[(b + 1) % 4]),
                );
                for k in [kp, kq] {
                    point
                        .entry(k)
                        .or_insert_with(|| cross((k.0, k.1), (k.2, k.3)));
                }
                adj.entry(kp).or_default().push(kq);
                adj.entry(kq).or_default().push(kp);
            }
        }
    }
    let mut d = String::new();
    let mut keys: Vec<Key> = adj.keys().copied().collect();
    keys.sort();
    let mut seen: std::collections::HashSet<Key> = Default::default();
    for start in keys {
        if seen.contains(&start) {
            continue;
        }
        seen.insert(start);
        let p0 = point[&start];
        d.push_str(&format!("M{:.2} {:.2}", p0.0, p0.1));
        let (mut prev, mut cur) = (start, adj[&start][0]);
        while cur != start && seen.insert(cur) {
            let p = point[&cur];
            d.push_str(&format!("L{:.2} {:.2}", p.0, p.1));
            let nexts = &adj[&cur];
            let nxt = if nexts[0] == prev {
                nexts.get(1).copied().unwrap_or(prev)
            } else {
                nexts[0]
            };
            prev = cur;
            cur = nxt;
        }
        d.push('Z');
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(t: f32) -> usize {
        sample(80, 80, t, None)
            .iter()
            .filter(|s| **s == Ink::Chain)
            .count()
    }

    #[test]
    fn the_fold_starts_extended_and_settles() {
        assert_eq!(bead(0, 1.0), FOLD[0]);
        assert_eq!(bead(11, 7.0), FOLD[11], "no frame beyond the fold");
        let (x0, _, _) = bead(0, 0.0);
        let (x11, _, _) = bead(11, 0.0);
        assert!(x11 - x0 > 80.0, "the extended chain spans the box");
        assert!(count(0.0) > 0 && count(1.0) > count(0.0));
    }

    #[test]
    fn beads_keep_waists_between_them() {
        // Midway between two consecutive beads of the stem, the chain is narrower than a bead.
        let (x, y) = (30.0, 69.5);
        let width = |dx: f32| chain_distance(x + dx, y, 1.0) <= 0.0;
        let at_neck = (-100..=100).filter(|k| width(*k as f32 * 0.1)).count();
        let (cx, cy, _) = FOLD[2];
        let at_bead = (-100..=100)
            .filter(|k| chain_distance(cx + *k as f32 * 0.1, cy, 1.0) <= 0.0)
            .count();
        assert!(
            at_neck > 0 && at_neck < at_bead,
            "neck {at_neck} bead {at_bead}"
        );
    }

    #[test]
    fn it_renders_at_every_size() {
        for (c, r) in [(0, 0), (1, 1), (2, 1), (8, 4), (20, 10), (41, 7)] {
            let b = braille(c, r, 1.0);
            assert_eq!(b.len(), r);
            assert!(b.iter().all(|row| row.len() == c));
        }
        let b = braille(16, 8, 1.0);
        assert!(b.iter().flatten().any(|(_, i)| *i == Ink::Chain));
        assert!(b.iter().flatten().any(|(_, i)| *i == Ink::Ligand));
        assert!(braille(16, 8, 0.2)
            .iter()
            .flatten()
            .all(|(_, i)| *i != Ink::Ligand));
    }

    #[test]
    fn the_svg_path_is_closed_loops_inside_the_box() {
        // Several grid sizes: a tracing bug can hide at one size and show at another.
        for n in [120, 200, 240, 317] {
            check_path(&svg_path(n), 100.0 / n as f32);
        }
    }

    fn check_path(d: &str, cell: f32) {
        assert!(d.starts_with('M') && d.ends_with('Z'));
        // The p has one outer contour and one hole (the bowl), and nothing else.
        assert_eq!(d.matches('M').count(), 2, "{}", &d[..200.min(d.len())]);
        // Consecutive points are neighbours on the grid: a jump would be a chord across the mark.
        for contour in d.split('Z').filter(|c| !c.is_empty()) {
            let pts: Vec<(f32, f32)> = contour
                .split(['M', 'L'])
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let mut it = s.split(' ').map(|v| v.parse::<f32>().unwrap());
                    (it.next().unwrap(), it.next().unwrap())
                })
                .collect();
            assert!(pts.len() > 50);
            for w in pts
                .windows(2)
                .chain(std::iter::once(&[pts[pts.len() - 1], pts[0]][..]))
            {
                let step = ((w[0].0 - w[1].0).powi(2) + (w[0].1 - w[1].1).powi(2)).sqrt();
                // At most the diagonal of one grid cell.
                assert!(
                    step <= cell * std::f32::consts::SQRT_2 + 1e-3,
                    "a jump of {step} in the outline"
                );
            }
        }
        for n in d
            .split(|c: char| c.is_ascii_alphabetic())
            .flat_map(|s| s.split(' '))
            .filter(|s| !s.is_empty())
        {
            let v: f32 = n.parse().unwrap();
            assert!((0.0..=100.0).contains(&v), "{v}");
        }
    }
}

#[cfg(test)]
mod preview {
    /// `cargo test -p proteus-render --lib brand::mark::preview -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn print_the_fold() {
        for t in [0.0, 0.5, 1.0] {
            println!("t = {t}");
            for row in super::braille(20, 10, t) {
                println!("{}", row.iter().map(|(c, _)| c).collect::<String>());
            }
        }
        for l in super::super::matrix::half_blocks("proteus") {
            println!("{l}");
        }
        for l in super::super::matrix::braille("proteus") {
            println!("{l}");
        }
    }
}
