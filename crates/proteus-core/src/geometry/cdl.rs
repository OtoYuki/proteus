//! Conformation-Dependent Library v1.2 (Moriarty, Tronrud, Adams & Karplus): backbone bond and
//! angle targets that depend on the residue's own φ/ψ.
//!
//! Data and semantics come from cctbx `mmtbx/conformation_dependent_library/` (`cdl_database.py`,
//! `cdl_setup.py`, `cdl_utils.py`, `__init__.py`, `multi_residue_cdl_class.py`); see
//! `data/cdl/NOTICE` for provenance and licence. Phenix applies the CDL by default
//! (`pdb_interpretation.restraints_library.cdl = True`), as do the wwPDB validation reports.

use std::sync::OnceLock;

const GRID: usize = 36;
/// Twelve (target, esd) pairs per φ/ψ cell, in [`CdlTerm`] order.
const VALUES: usize = 24;

/// Residue class of the CDL (`cdl_setup.py`): glycine, proline, Ile/Val or any other residue,
/// each with a separate table when the next residue is a proline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CdlGroup {
    class: CdlClass,
    before_pro: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CdlClass {
    /// `NonPGIV`: not Pro, Gly, Ile or Val.
    Other = 0,
    IleVal = 1,
    Gly = 2,
    Pro = 3,
}

impl CdlGroup {
    /// `cdl_utils.get_res_type_group` for the 20 standard residues.
    pub(crate) fn classify(name: &str, next_name: &str) -> CdlGroup {
        let class = match name {
            "GLY" => CdlClass::Gly,
            "PRO" => CdlClass::Pro,
            "ILE" | "VAL" => CdlClass::IleVal,
            _ => CdlClass::Other,
        };
        CdlGroup {
            class,
            before_pro: next_name == "PRO",
        }
    }

    /// Table index in `cdl_v1_2.f32` (the `CDL_GROUPS` order of `scripts/convert_geostd.py`).
    fn index(self) -> usize {
        self.class as usize + if self.before_pro { 4 } else { 0 }
    }
}

/// The backbone restraints the CDL sets, in the column order of `cdl_database.py`. Atom names
/// are relative to the central residue `i`: `-` is residue i−1, `+` residue i+1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CdlTerm {
    /// C(i−1)–N–CA
    Cna,
    /// N–CA–CB
    Nab,
    /// N–CA–C
    Nac,
    /// CB–CA–C
    Bac,
    /// CA–C–O
    Aco,
    /// CA–C–N(i+1)
    Acn,
    /// O–C–N(i+1)
    Ocn,
    /// C(i−1)–N
    Cn,
    /// N–CA
    Na,
    /// CA–CB
    Ab,
    /// CA–C
    Ac,
    /// C–O
    Co,
    // Proline ring terms, set only for cis-proline (`cis_127` in cctbx).
    /// C(i−1)–N–CD
    Cnd,
    /// CA–N–CD
    And,
    /// N–CD–CG
    Ndg,
    /// CA–CB–CG
    Abg,
    /// CB–CG–CD
    Bgd,
    /// CB–CG
    Bg,
    /// CG–CD
    Gd,
    /// N–CD
    Nd,
}

/// Where a term's atoms live: offset from the central residue and atom name.
pub(crate) type TermAtom = (i8, &'static str);

impl CdlTerm {
    pub(crate) const TABLE: [CdlTerm; 12] = [
        CdlTerm::Cna,
        CdlTerm::Nab,
        CdlTerm::Nac,
        CdlTerm::Bac,
        CdlTerm::Aco,
        CdlTerm::Acn,
        CdlTerm::Ocn,
        CdlTerm::Cn,
        CdlTerm::Na,
        CdlTerm::Ab,
        CdlTerm::Ac,
        CdlTerm::Co,
    ];

    /// Atoms of the term (`multi_residue_cdl_class.apply_updates`).
    pub(crate) fn atoms(self) -> &'static [TermAtom] {
        match self {
            CdlTerm::Cna => &[(-1, "C"), (0, "N"), (0, "CA")],
            CdlTerm::Nab => &[(0, "N"), (0, "CA"), (0, "CB")],
            CdlTerm::Nac => &[(0, "N"), (0, "CA"), (0, "C")],
            CdlTerm::Bac => &[(0, "CB"), (0, "CA"), (0, "C")],
            CdlTerm::Aco => &[(0, "CA"), (0, "C"), (0, "O")],
            CdlTerm::Acn => &[(0, "CA"), (0, "C"), (1, "N")],
            CdlTerm::Ocn => &[(0, "O"), (0, "C"), (1, "N")],
            CdlTerm::Cn => &[(-1, "C"), (0, "N")],
            CdlTerm::Na => &[(0, "N"), (0, "CA")],
            CdlTerm::Ab => &[(0, "CA"), (0, "CB")],
            CdlTerm::Ac => &[(0, "CA"), (0, "C")],
            CdlTerm::Co => &[(0, "C"), (0, "O")],
            CdlTerm::Cnd => &[(-1, "C"), (0, "N"), (0, "CD")],
            CdlTerm::And => &[(0, "CA"), (0, "N"), (0, "CD")],
            CdlTerm::Ndg => &[(0, "N"), (0, "CD"), (0, "CG")],
            CdlTerm::Abg => &[(0, "CA"), (0, "CB"), (0, "CG")],
            CdlTerm::Bgd => &[(0, "CB"), (0, "CG"), (0, "CD")],
            CdlTerm::Bg => &[(0, "CB"), (0, "CG")],
            CdlTerm::Gd => &[(0, "CG"), (0, "CD")],
            CdlTerm::Nd => &[(0, "N"), (0, "CD")],
        }
    }
}

/// Engh & Huber (1999) targets that replace the CDL for a cis-proline
/// (`get_restraint_values`, `cis_pro_eh99 = True` by default).
pub(crate) const CIS_PRO_EH99: [(CdlTerm, f64, f64); 20] = [
    (CdlTerm::Cna, 127.0, 2.4),
    (CdlTerm::Nab, 102.6, 1.1),
    (CdlTerm::Nac, 112.1, 2.6),
    (CdlTerm::Bac, 112.0, 2.5),
    (CdlTerm::Aco, 120.2, 2.4),
    (CdlTerm::Acn, 117.1, 2.8),
    (CdlTerm::Ocn, 121.1, 1.9),
    (CdlTerm::Cn, 1.338, 0.019),
    (CdlTerm::Na, 1.468, 0.017),
    (CdlTerm::Ab, 1.533, 0.018),
    (CdlTerm::Ac, 1.524, 0.020),
    (CdlTerm::Co, 1.228, 0.020),
    (CdlTerm::Cnd, 120.6, 2.2),
    (CdlTerm::And, 111.5, 1.4),
    (CdlTerm::Ndg, 103.8, 1.2),
    (CdlTerm::Abg, 104.0, 1.9),
    (CdlTerm::Bgd, 105.4, 2.3),
    (CdlTerm::Bg, 1.506, 0.039),
    (CdlTerm::Gd, 1.512, 0.027),
    (CdlTerm::Nd, 1.474, 0.014),
];

fn table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let bytes = include_bytes!("../../data/cdl/cdl_v1_2.f32");
        assert_eq!(bytes.len(), 8 * GRID * GRID * VALUES * 4, "CDL grid size");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    })
}

/// `cdl_utils.round_to_ten`: Python's round-half-to-even on degrees/10, with 180 wrapped to −180.
fn round_to_ten(angle: f64) -> i32 {
    let t = (angle / 10.0).round_ties_even() as i32 * 10;
    if t == 180 {
        -180
    } else {
        t
    }
}

/// The twelve (target, esd) pairs for a residue class at φ/ψ, in [`CdlTerm::TABLE`] order.
/// Targets of −1 mark a term the class does not have (CB terms of glycine).
pub(crate) fn lookup(group: CdlGroup, phi: f64, psi: f64) -> [(f64, f64); 12] {
    let (p, s) = (round_to_ten(phi), round_to_ten(psi));
    let cell = ((p + 180) / 10) as usize * GRID + ((s + 180) / 10) as usize;
    let base = (group.index() * GRID * GRID + cell) * VALUES;
    let row = &table()[base..base + VALUES];
    std::array::from_fn(|k| (f64::from(row[2 * k]), f64::from(row[2 * k + 1])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_like_python() {
        assert_eq!(round_to_ten(-65.0), -60); // -6.5 → -6 (half to even)
        assert_eq!(round_to_ten(-75.0), -80);
        assert_eq!(round_to_ten(175.0), -180); // 17.5 → 18 → 180 → wrapped
        assert_eq!(round_to_ten(179.9), -180);
        assert_eq!(round_to_ten(-179.9), -180);
        assert_eq!(round_to_ten(4.99), 0);
    }

    #[test]
    fn first_cell_matches_cdl_database() {
        // cdl_database["Gly_nonxpro"][(-180, -180)] =
        //   ['B', 14, 120.86, 1.62, -1, -1, 110.42, 1.49, -1, -1, 121.63, 0.95, 114.72, 1.32,
        //    123.61, 1.16, 1.327, 0.0115, 1.4508, 0.0113, -1, -1, 1.5117, 0.0124, 1.2356, 0.0123]
        let row = lookup(CdlGroup::classify("GLY", "ALA"), -180.0, -180.0);
        assert!((row[0].0 - 120.86).abs() < 1e-4 && (row[0].1 - 1.62).abs() < 1e-5);
        assert_eq!(row[1].0, -1.0);
        assert!((row[7].0 - 1.327).abs() < 1e-6 && (row[7].1 - 0.0115).abs() < 1e-7);
        assert!((row[11].0 - 1.2356).abs() < 1e-6);
    }

    #[test]
    fn classes() {
        // CDL_GROUPS order: NonPGIV, IleVal, Gly, Pro, then the same four before a proline.
        assert_eq!(CdlGroup::classify("ALA", "PRO").index(), 4);
        assert_eq!(CdlGroup::classify("VAL", "GLY").index(), 1);
        assert_eq!(CdlGroup::classify("PRO", "PRO").index(), 7);
        assert_eq!(CdlGroup::classify("GLY", "SER").index(), 2);
    }
}
