//! Icons in the wordmark's dot matrix: 4 × 4 dots each, so one braille line two cells wide in
//! a terminal, and circles in SVG like the wordmark. Drawn, not taken from a font: the same
//! dots as the rest of the identity, and nothing that a terminal's font may lack.

use super::matrix;

/// An icon of the identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    /// Jobs: two entries of a list, each a bullet and a line.
    Jobs,
    /// Structures: a chain folded into a small **p**, the Proteus mark in four rows of dots.
    Structures,
    /// Run: a play mark.
    Run,
}

impl Icon {
    pub const ALL: [Icon; 3] = [Icon::Jobs, Icon::Structures, Icon::Run];

    /// The dots, row by row (`#` a dot).
    pub fn rows(self) -> [&'static str; 4] {
        match self {
            Icon::Jobs => ["#.##", "....", "#.##", "...."],
            Icon::Structures => [".##.", "#..#", "###.", "#..."],
            Icon::Run => ["#...", "###.", "###.", "#..."],
        }
    }

    /// The dots as a bitmap, `(width, height, dots)`.
    pub fn bitmap(self) -> (usize, usize, Vec<bool>) {
        let rows = self.rows();
        let w = rows[0].len();
        let dots = rows
            .iter()
            .flat_map(|r| r.chars().map(|c| c == '#'))
            .collect();
        (w, rows.len(), dots)
    }

    /// Two braille cells.
    pub fn braille(self) -> String {
        let (w, h, d) = self.bitmap();
        matrix::braille_of(w, h, &d).concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icons_are_two_cells_of_braille_and_tell_apart() {
        let all: Vec<String> = Icon::ALL.iter().map(|i| i.braille()).collect();
        assert_eq!(all, ["⠅⠭", "⡮⠕", "⡷⠆"]);
        for (i, a) in all.iter().enumerate() {
            assert_eq!(a.chars().count(), 2);
            assert!(!all[i + 1..].contains(a));
        }
    }
}
