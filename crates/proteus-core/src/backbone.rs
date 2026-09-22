//! Chain-aware backbone extraction shared by Ramachandran, DSSP and confidence analysis.

use nalgebra::Vector3;

use crate::structure::compute_dihedral;

/// Maximum C(i-1)–N(i) distance for a continuous peptide bond (DSSP convention).
pub const MAX_PEPTIDE_BOND: f64 = 2.5;
/// Maximum CA(i-1)–CA(i) distance for consecutive residues when the peptide atoms are absent
/// (C-alpha-only traces): 3.8 Å trans, 2.9 Å cis, with slack for coarse models.
pub const MAX_CA_CA: f64 = 4.2;

/// One residue's backbone atoms, in file order, with chain-break bookkeeping.
#[derive(Debug, Clone)]
pub struct BackboneResidue {
    pub chain_id: String,
    pub seq_num: isize,
    /// PDB insertion code, if any.
    pub insertion_code: Option<String>,
    /// Three-letter residue name, trimmed and upper-cased.
    pub name: String,
    pub n: Option<Vector3<f64>>,
    pub ca: Option<Vector3<f64>>,
    pub c: Option<Vector3<f64>>,
    pub o: Option<Vector3<f64>>,
    /// B-factor of the C-alpha atom (pLDDT for predicted structures).
    pub b_factor: f64,
    /// True when no peptide bond connects this residue to the previous entry: different
    /// chain, C–N distance above [`MAX_PEPTIDE_BOND`], or — when either residue lacks its C or
    /// N atom — CA–CA distance above [`MAX_CA_CA`].
    pub chain_break_before: bool,
}

impl BackboneResidue {
    pub fn is_proline(&self) -> bool {
        self.name == "PRO"
    }
}

/// Extract one entry per residue that has a C-alpha, in file order, with chain breaks marked.
pub fn extract_backbone(pdb: &pdbtbx::PDB) -> Vec<BackboneResidue> {
    let mut out: Vec<BackboneResidue> = Vec::new();
    for chain in pdb.chains() {
        for residue in chain.residues() {
            let mut r = BackboneResidue {
                chain_id: chain.id().to_string(),
                seq_num: residue.serial_number(),
                insertion_code: residue.insertion_code().map(|c| c.trim().to_string()),
                name: residue
                    .name()
                    .map(|n| n.trim().to_uppercase())
                    .unwrap_or_default(),
                n: None,
                ca: None,
                c: None,
                o: None,
                b_factor: 0.0,
                chain_break_before: false,
            };
            for atom in residue.atoms() {
                let v = Vector3::new(atom.x(), atom.y(), atom.z());
                match atom.name().trim() {
                    "N" if r.n.is_none() => r.n = Some(v),
                    "CA" if r.ca.is_none() => {
                        r.ca = Some(v);
                        r.b_factor = atom.b_factor();
                    }
                    "C" if r.c.is_none() => r.c = Some(v),
                    "O" | "O1" | "OXT" if r.o.is_none() => r.o = Some(v),
                    _ => {}
                }
            }
            if r.ca.is_none() {
                continue;
            }
            r.chain_break_before = match out.last() {
                None => true,
                Some(prev) => {
                    prev.chain_id != r.chain_id
                        || match (prev.c, r.n, prev.ca, r.ca) {
                            (Some(c), Some(n), _, _) => (c - n).norm() > MAX_PEPTIDE_BOND,
                            // C-alpha-only trace: judge continuity by CA spacing instead.
                            (_, _, Some(a), Some(b)) => (a - b).norm() > MAX_CA_CA,
                            _ => true,
                        }
                }
            };
            out.push(r);
        }
    }
    out
}

/// φ(i) = C(i−1)–N(i)–CA(i)–C(i). `None` across a chain break or with missing atoms.
pub fn phi(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64> {
    if cur.chain_break_before {
        return None;
    }
    compute_dihedral(&prev.c?, &cur.n?, &cur.ca?, &cur.c?).ok()
}

/// ψ(i) = N(i)–CA(i)–C(i)–N(i+1). `None` across a chain break or with missing atoms.
pub fn psi(cur: &BackboneResidue, next: &BackboneResidue) -> Option<f64> {
    if next.chain_break_before {
        return None;
    }
    compute_dihedral(&cur.n?, &cur.ca?, &cur.c?, &next.n?).ok()
}

/// ω(i) = CA(i−1)–C(i−1)–N(i)–CA(i); ≈180° for trans, ≈0° for cis peptides.
pub fn omega(prev: &BackboneResidue, cur: &BackboneResidue) -> Option<f64> {
    if cur.chain_break_before {
        return None;
    }
    compute_dihedral(&prev.ca?, &prev.c?, &cur.n?, &cur.ca?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crambin() -> pdbtbx::PDB {
        pdbtbx::ReadOptions::default()
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/1crn.pdb"))
            .unwrap()
            .0
    }

    #[test]
    fn extracts_46_residues_with_all_backbone_atoms() {
        let bb = extract_backbone(&crambin());
        assert_eq!(bb.len(), 46);
        assert!(bb
            .iter()
            .all(|r| r.n.is_some() && r.ca.is_some() && r.c.is_some() && r.o.is_some()));
        assert_eq!(bb[0].name, "THR");
        assert_eq!(bb[0].seq_num, 1);
        assert!(bb[0].chain_break_before);
        assert!(bb[1..].iter().all(|r| !r.chain_break_before));
    }

    #[test]
    fn phi_psi_omega_on_crambin() {
        let bb = extract_backbone(&crambin());
        let phi7 = phi(&bb[5], &bb[6]).unwrap();
        let psi7 = psi(&bb[6], &bb[7]).unwrap();
        assert!((phi7 + 63.6).abs() < 0.1, "{phi7}");
        assert!((psi7 + 42.1).abs() < 0.1, "{psi7}");
        let w = omega(&bb[5], &bb[6]).unwrap();
        assert!(w.abs() > 150.0, "trans peptide expected, got {w}");
        assert!(phi(&bb[45], &bb[0]).is_none());
    }

    #[test]
    fn two_chains_break() {
        let text =
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n\
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n\
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O\n\
TER\n\
ATOM      5  N   GLY B   1      20.000   0.000   0.000  1.00  0.00           N\n\
ATOM      6  CA  GLY B   1      21.458   0.000   0.000  1.00  0.00           C\n\
ATOM      7  C   GLY B   1      22.009   1.420   0.000  1.00  0.00           C\n\
ATOM      8  O   GLY B   1      21.251   2.390   0.000  1.00  0.00           O\n\
END\n";
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Pdb)
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(std::io::Cursor::new(text)))
            .unwrap();
        let bb = extract_backbone(&pdb);
        assert_eq!(bb.len(), 2);
        assert!(bb[1].chain_break_before);
        assert!(phi(&bb[0], &bb[1]).is_none());
    }

    /// A C-alpha-only trace (coarse-grained model, the simulated runner's output) has no
    /// peptide atoms to test, so continuity falls back to the CA–CA distance.
    #[test]
    fn ca_only_trace_breaks_on_ca_distance() {
        let mut text = String::new();
        // Residues 1–3 at 3.8 Å spacing, residue 4 after a 12 Å jump, residue 5 back at 3.8 Å.
        for (i, x) in [0.0, 3.8, 7.6, 19.6, 23.4].into_iter().enumerate() {
            text += &format!(
                "ATOM  {:5}  CA  ALA A{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C\n",
                i + 1,
                i + 1,
                x,
                0.0,
                0.0
            );
        }
        text += "END\n";
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Pdb)
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(std::io::Cursor::new(text)))
            .unwrap();
        let bb = extract_backbone(&pdb);
        let breaks: Vec<bool> = bb.iter().map(|r| r.chain_break_before).collect();
        assert_eq!(breaks, vec![true, false, false, true, false]);
        // Dihedrals still need the peptide atoms.
        assert!(phi(&bb[0], &bb[1]).is_none());
    }
}
