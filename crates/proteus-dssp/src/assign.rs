//! DSSP 2.x secondary-structure assignment: bridges/ladders, helices, turns and bends.
//!
//! The control flow mirrors DSSP-2.2.0 `structure.cpp` as ported by mdtraj, so that the
//! output can be validated residue-by-residue against `mdtraj.compute_dssp`.

use std::collections::VecDeque;

use crate::hbond::{compute_bonds, dot, norm, sub, test_bond, Bonds, V};

/// Backbone atoms of one residue. All four atoms are required; residues lacking any of
/// them must be dropped by the caller before calling [`assign`].
#[derive(Clone, Debug)]
pub struct Residue {
    pub n: V,
    pub ca: V,
    pub c: V,
    pub o: V,
    /// Proline has no amide hydrogen and never donates.
    pub is_proline: bool,
    /// True when no peptide bond connects this residue to the previous one
    /// (new chain, or a gap in the model).
    pub chain_break_before: bool,
}

/// Eight-state DSSP code.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Ss {
    /// α-helix
    H,
    /// Isolated β-bridge
    B,
    /// Extended strand in a β-ladder
    E,
    /// 3₁₀-helix
    G,
    /// π-helix
    I,
    /// Hydrogen-bonded turn
    T,
    /// Bend
    S,
    /// None of the above
    Loop,
}

/// Three-state reduction (H,G,I → Helix; E,B → Strand; rest → Coil).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Simple {
    Helix,
    Strand,
    Coil,
}

impl Ss {
    /// Single-letter DSSP code; loop is `-` (mdtraj prints a space).
    pub fn as_char(self) -> char {
        match self {
            Ss::H => 'H',
            Ss::B => 'B',
            Ss::E => 'E',
            Ss::G => 'G',
            Ss::I => 'I',
            Ss::T => 'T',
            Ss::S => 'S',
            Ss::Loop => '-',
        }
    }

    pub fn simplify(self) -> Simple {
        match self {
            Ss::H | Ss::G | Ss::I => Simple::Helix,
            Ss::E | Ss::B => Simple::Strand,
            _ => Simple::Coil,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HelixFlag {
    None,
    Start,
    End,
    StartAndEnd,
    Middle,
}

impl HelixFlag {
    fn is_start(self) -> bool {
        matches!(self, HelixFlag::Start | HelixFlag::StartAndEnd)
    }
}

/// Residues `a..=b` (either order) lie in one continuous peptide segment.
fn continuous(res: &[Residue], a: usize, b: usize) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    !res[lo + 1..=hi].iter().any(|r| r.chain_break_before)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BridgeType {
    Parallel,
    Antiparallel,
}

#[derive(Debug)]
struct Bridge {
    kind: BridgeType,
    i: Vec<usize>,
    j: VecDeque<usize>,
}

/// DSSP `MResidue::TestBridge`: is there a β-bridge between residues `i` and `j`?
fn test_bridge(res: &[Residue], bonds: &[Bonds], i: usize, j: usize) -> Option<BridgeType> {
    let n = res.len();
    if i == 0 || j == 0 || i + 1 >= n || j + 1 >= n {
        return None;
    }
    let (a, b, c) = (i - 1, i, i + 1);
    let (d, e, f) = (j - 1, j, j + 1);
    if !(continuous(res, a, c) && continuous(res, d, f)) {
        return None;
    }
    let tb = |x, y| test_bond(bonds, x, y);
    if (tb(c, e) && tb(e, a)) || (tb(f, b) && tb(b, d)) {
        Some(BridgeType::Parallel)
    } else if (tb(c, d) && tb(f, a)) || (tb(e, b) && tb(b, e)) {
        Some(BridgeType::Antiparallel)
    } else {
        None
    }
}

/// DSSP `MProtein::CalculateBetaSheets`.
fn assign_sheets(res: &[Residue], bonds: &[Bonds], ss: &mut [Ss]) {
    let n = res.len();
    if n < 5 {
        return;
    }
    let mut bridges: Vec<Bridge> = Vec::new();
    for i in 1..(n - 4) {
        for j in (i + 3)..(n - 1) {
            let Some(kind) = test_bridge(res, bonds, j, i) else {
                continue;
            };
            let mut found = false;
            for br in bridges.iter_mut() {
                if br.kind != kind || i != br.i[br.i.len() - 1] + 1 {
                    continue;
                }
                match kind {
                    BridgeType::Parallel if br.j[br.j.len() - 1] + 1 == j => {
                        br.i.push(i);
                        br.j.push_back(j);
                        found = true;
                    }
                    BridgeType::Antiparallel if br.j[0] == j + 1 => {
                        br.i.push(i);
                        br.j.push_front(j);
                        found = true;
                    }
                    _ => {}
                }
                if found {
                    break;
                }
            }
            if !found {
                bridges.push(Bridge {
                    kind,
                    i: vec![i],
                    j: VecDeque::from(vec![j]),
                });
            }
        }
    }

    // Extend ladders across β-bulges (variable names follow DSSP: <strand><begin|end><bridge>).
    let mut bi = 0;
    while bi < bridges.len() {
        let mut bj = bi + 1;
        while bj < bridges.len() {
            let ibi = bridges[bi].i[0];
            let iei = bridges[bi].i[bridges[bi].i.len() - 1];
            let jbi = bridges[bi].j[0];
            let jei = bridges[bi].j[bridges[bi].j.len() - 1];
            let ibj = bridges[bj].i[0];
            let iej = bridges[bj].i[bridges[bj].i.len() - 1];
            let jbj = bridges[bj].j[0];
            let jej = bridges[bj].j[bridges[bj].j.len() - 1];

            let same_kind = bridges[bi].kind == bridges[bj].kind;
            let cont_i = continuous(res, ibi.min(ibj), iei.max(iej));
            let cont_j = continuous(res, jbi.min(jbj), jei.max(jej));
            let overlap = iei >= ibj && ibi <= iej;
            // Signed arithmetic below reproduces DSSP's `int` comparisons exactly.
            let (iei, jbi, jei) = (iei as i64, jbi as i64, jei as i64);
            let (ibj, jbj, jej) = (ibj as i64, jbj as i64, jej as i64);
            if !same_kind || !cont_i || !cont_j || ibj - iei >= 6 || overlap {
                bj += 1;
                continue;
            }
            let bulge = match bridges[bi].kind {
                BridgeType::Parallel => {
                    jbj > jbi && ((jbj - jei < 6 && ibj - iei < 3) || (jbj - jei < 3))
                }
                BridgeType::Antiparallel => {
                    jbj < jbi && ((jbi - jej < 6 && ibj - iei < 3) || (jbi - jej < 3))
                }
            };
            if !bulge {
                bj += 1;
                continue;
            }
            let other = bridges.remove(bj);
            bridges[bi].i.extend(other.i);
            match bridges[bi].kind {
                BridgeType::Parallel => bridges[bi].j.extend(other.j),
                BridgeType::Antiparallel => {
                    for x in other.j.into_iter().rev() {
                        bridges[bi].j.push_front(x);
                    }
                }
            }
            // `bj` now points at the element that shifted into the removed slot.
        }
        bi += 1;
    }

    for br in &bridges {
        let kind = if br.i.len() > 1 { Ss::E } else { Ss::B };
        let (i0, i1) = (br.i[0], br.i[br.i.len() - 1]);
        for slot in &mut ss[i0.min(i1)..=i0.max(i1)] {
            if *slot != Ss::E {
                *slot = kind;
            }
        }
        let (j0, j1) = (br.j[0], br.j[br.j.len() - 1]);
        for slot in &mut ss[j0.min(j1)..=j0.max(j1)] {
            if *slot != Ss::E {
                *slot = kind;
            }
        }
    }
}

/// DSSP `MProtein::CalculateAlphaHelices` with `inPreferPiHelices = true` (mdtraj default).
fn assign_helices(res: &[Residue], bonds: &[Bonds], ss: &mut [Ss]) {
    let n = res.len();
    // flags[stride - 3][i]
    let mut flags = vec![vec![HelixFlag::None; n]; 3];
    for stride in 3..=5usize {
        let f = &mut flags[stride - 3];
        for i in 0..n.saturating_sub(stride) {
            if test_bond(bonds, i + stride, i) && continuous(res, i, i + stride) {
                f[i + stride] = HelixFlag::End;
                for flag in &mut f[(i + 1)..(i + stride)] {
                    if *flag == HelixFlag::None {
                        *flag = HelixFlag::Middle;
                    }
                }
                f[i] = if f[i] == HelixFlag::End {
                    HelixFlag::StartAndEnd
                } else {
                    HelixFlag::Start
                };
            }
        }
    }
    // α-helix: two consecutive 4-turns, unconditional.
    for i in 1..n.saturating_sub(4) {
        if flags[1][i].is_start() && flags[1][i - 1].is_start() {
            ss[i..=(i + 3)].fill(Ss::H);
        }
    }
    // 3₁₀-helix: only into loop / G.
    for i in 1..n.saturating_sub(3) {
        if flags[0][i].is_start() && flags[0][i - 1].is_start() {
            let empty = (i..=(i + 2)).all(|k| matches!(ss[k], Ss::Loop | Ss::G));
            if empty {
                ss[i..=(i + 2)].fill(Ss::G);
            }
        }
    }
    // π-helix: into loop / I / H (prefer-π, as mdtraj).
    for i in 1..n.saturating_sub(5) {
        if flags[2][i].is_start() && flags[2][i - 1].is_start() {
            let empty = (i..=(i + 4)).all(|k| matches!(ss[k], Ss::Loop | Ss::I | Ss::H));
            if empty {
                ss[i..=(i + 4)].fill(Ss::I);
            }
        }
    }
    // Turns and bends fill what is left.
    for i in 1..n.saturating_sub(1) {
        if ss[i] != Ss::Loop {
            continue;
        }
        let mut is_turn = false;
        'outer: for stride in 3..=5usize {
            for k in 1..stride {
                if i >= k && flags[stride - 3][i - k].is_start() {
                    is_turn = true;
                    break 'outer;
                }
            }
        }
        if is_turn {
            ss[i] = Ss::T;
        } else if is_bend(res, i) {
            ss[i] = Ss::S;
        }
    }
}

/// κ(i) = angle between CA(i−2)→CA(i) and CA(i)→CA(i+2); a bend when κ > 70°.
fn is_bend(res: &[Residue], i: usize) -> bool {
    if i < 2 || i + 2 >= res.len() || !continuous(res, i - 2, i + 2) {
        return false;
    }
    let u = sub(res[i].ca, res[i - 2].ca);
    let v = sub(res[i + 2].ca, res[i].ca);
    let denom = norm(u) * norm(v);
    if denom < 1e-9 {
        return false;
    }
    let c = (dot(u, v) / denom).clamp(-1.0, 1.0);
    c.acos().to_degrees() > 70.0
}

/// Assign eight-state DSSP secondary structure to a backbone.
pub fn assign(residues: &[Residue]) -> Vec<Ss> {
    let n = residues.len();
    let mut ss = vec![Ss::Loop; n];
    if n < 3 {
        return ss;
    }
    let bonds = compute_bonds(residues);
    assign_sheets(residues, &bonds, &mut ss);
    assign_helices(residues, &bonds, &mut ss);
    ss
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `n` residues from φ/ψ with NeRF; ω = 180°, standard bond geometry.
    fn build_chain(n: usize, phi: f64, psi: f64) -> Vec<Residue> {
        use std::f64::consts::PI;
        fn place(a: V, b: V, c: V, bond: f64, angle: f64, torsion: f64) -> V {
            let cross = |p: V, q: V| {
                [
                    p[1] * q[2] - p[2] * q[1],
                    p[2] * q[0] - p[0] * q[2],
                    p[0] * q[1] - p[1] * q[0],
                ]
            };
            let unit = |p: V| {
                let l = norm(p);
                [p[0] / l, p[1] / l, p[2] / l]
            };
            let bc = unit(sub(c, b));
            let nrm = unit(cross(sub(b, a), bc));
            let m = [bc, cross(nrm, bc), nrm];
            let (ang, tor) = (angle * PI / 180.0, torsion * PI / 180.0);
            let d2 = [
                -bond * ang.cos(),
                bond * ang.sin() * tor.cos(),
                bond * ang.sin() * tor.sin(),
            ];
            [
                c[0] + m[0][0] * d2[0] + m[1][0] * d2[1] + m[2][0] * d2[2],
                c[1] + m[0][1] * d2[0] + m[1][1] * d2[1] + m[2][1] * d2[2],
                c[2] + m[0][2] * d2[0] + m[1][2] * d2[1] + m[2][2] * d2[2],
            ]
        }
        let mut n_ = [0.0, 0.0, 0.0];
        let mut ca = [1.458, 0.0, 0.0];
        let mut c = place([0.0, 1.0, 0.0], n_, ca, 1.525, 111.2, -120.0);
        let mut out = Vec::new();
        for i in 0..n {
            let n_next = place(n_, ca, c, 1.329, 116.2, psi);
            let o = place(n_next, ca, c, 1.231, 120.5, 180.0);
            out.push(Residue {
                n: n_,
                ca,
                c,
                o,
                is_proline: false,
                chain_break_before: i == 0,
            });
            let ca_next = place(ca, c, n_next, 1.458, 121.7, 180.0);
            let c_next = place(c, n_next, ca_next, 1.525, 111.2, phi);
            n_ = n_next;
            ca = ca_next;
            c = c_next;
        }
        out
    }

    #[test]
    fn ideal_helix_is_mostly_h() {
        let ss = assign(&build_chain(20, -57.0, -47.0));
        let s: String = ss.iter().map(|x| x.as_char()).collect();
        let h = ss.iter().filter(|x| **x == Ss::H).count();
        assert!(h >= 14, "expected a long H run, got {s}");
        assert!(!s.contains('E'), "{s}");
    }

    #[test]
    fn extended_chain_alone_is_coil() {
        let ss = assign(&build_chain(12, -120.0, 130.0));
        assert!(
            ss.iter().all(|x| matches!(x, Ss::Loop | Ss::S | Ss::T)),
            "{:?}",
            ss
        );
    }

    #[test]
    fn crambin_matches_mdtraj_8_state() {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
            .unwrap();
        let expect = include_str!("../tests/data/1crn_dssp_mdtraj.txt")
            .trim()
            .to_string();
        let mut residues: Vec<Residue> = pdb
            .residues()
            .map(|r| {
                let get = |n: &str| {
                    let a = r.atoms().find(|a| a.name() == n).unwrap();
                    [a.x(), a.y(), a.z()]
                };
                Residue {
                    n: get("N"),
                    ca: get("CA"),
                    c: get("C"),
                    o: get("O"),
                    is_proline: r.name() == Some("PRO"),
                    chain_break_before: false,
                }
            })
            .collect();
        residues[0].chain_break_before = true;
        let got: String = assign(&residues).iter().map(|s| s.as_char()).collect();
        let agree = got
            .chars()
            .zip(expect.chars())
            .filter(|(a, b)| a == b)
            .count();
        eprintln!("mdtraj  {expect}\nproteus {got}\nagree {agree}/46");
        assert!(agree >= 44, "8-state agreement {agree}/46 below 44");
    }
}
