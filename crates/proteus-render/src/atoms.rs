//! Atom-level data for the browser viewer: every heavy atom with its bonds (for side-chain and
//! ligand sticks), the ligands themselves, and the analysis's contacts located on the ribbon.
//!
//! The ribbon only carries C-alphas, so without this the viewer can count 85 hydrogen bonds but
//! cannot show one. Residues are addressed by their ribbon index (`residue_labels` order);
//! ligand `k` is addressed as `num_residues + k`, so one picking id space covers both.

use nalgebra::Vector3;
use proteus_core::models::BiophysicalMetrics;
use proteus_core::structure::RamachandranRegion;
use std::collections::HashMap;

/// Element codes shared with `viewer.js` (`ELEMENTS`).
pub const ELEMENTS: [&str; 8] = ["C", "N", "O", "S", "P", "SE", "H", "X"];

fn element_code(symbol: &str) -> u8 {
    let s = symbol.trim().to_ascii_uppercase();
    ELEMENTS
        .iter()
        .position(|e| *e == s)
        .unwrap_or(ELEMENTS.len() - 1) as u8
}

/// Covalent radius in Å (Cordero et al. 2008), for inferring bonds from distances.
fn covalent_radius(code: u8) -> f32 {
    match ELEMENTS.get(code as usize).copied() {
        Some("C") => 0.76,
        Some("N") => 0.71,
        Some("O") => 0.66,
        Some("S") => 1.05,
        Some("P") => 1.07,
        Some("SE") => 1.20,
        Some("H") => 0.31,
        _ => 1.40,
    }
}

/// Heavy atoms of the protein residues on the ribbon and of the ligands, with bonds.
#[derive(Debug, Clone, Default)]
pub struct AtomTable {
    pub positions: Vec<Vector3<f32>>,
    pub elements: Vec<u8>,
    /// Ribbon residue index, or `num_residues + k` for ligand `k`.
    pub residues: Vec<u32>,
    pub names: Vec<String>,
    /// Pairs of atom indices.
    pub bonds: Vec<[u32; 2]>,
}

/// A non-polymer group drawn as sticks: a ligand, cofactor or ion. Waters are left out.
#[derive(Debug, Clone, PartialEq)]
pub struct Ligand {
    pub name: String,
    pub chain: String,
    pub number: isize,
    pub atom_count: usize,
}

/// One contact located on the structure: the two residues (for highlighting), the two points
/// (atoms, or ring centroids for π stacking) for the dashed line, a value and a label.
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    pub residues: [u32; 2],
    pub points: [Vector3<f32>; 2],
    /// Å: distance for bonds and stacking, overlap for clashes.
    pub value: f32,
    pub label: String,
}

/// Everything the viewer can point at, by kind.
#[derive(Debug, Clone, Default)]
pub struct Annotations {
    /// `(ribbon residue, region)` for every residue that is not Ramachandran-favoured.
    pub rama: Vec<(u32, RamachandranRegion)>,
    pub clashes: Vec<Contact>,
    pub hbonds: Vec<Contact>,
    pub salt_bridges: Vec<Contact>,
    pub pi_stacks: Vec<Contact>,
    pub cation_pi: Vec<Contact>,
}

type ResidueKey = (String, isize, Option<String>);

fn key_of(chain: &pdbtbx::Chain, residue: &pdbtbx::Residue) -> ResidueKey {
    (
        chain.id().to_string(),
        residue.serial_number(),
        residue.insertion_code().map(str::to_string),
    )
}

fn is_water(name: &str) -> bool {
    matches!(
        name.trim(),
        "HOH" | "WAT" | "DOD" | "H2O" | "SOL" | "TIP" | "TIP3"
    )
}

fn is_hydrogen(atom: &pdbtbx::Atom) -> bool {
    matches!(
        proteus_core::io::element_symbol(atom)
            .to_ascii_uppercase()
            .as_str(),
        "H" | "D"
    )
}

/// Bonds within each residue by distance, plus the peptide bond C(i)–N(i+1) and disulfides.
fn infer_bonds(t: &AtomTable, residue_spans: &[(usize, usize)]) -> Vec<[u32; 2]> {
    let mut bonds = Vec::new();
    let bonded = |a: usize, b: usize| {
        let d = (t.positions[a] - t.positions[b]).norm();
        let limit = covalent_radius(t.elements[a]) + covalent_radius(t.elements[b]) + 0.45;
        d > 0.4 && d <= limit
    };
    for &(start, end) in residue_spans {
        for a in start..end {
            for b in a + 1..end {
                if bonded(a, b) {
                    bonds.push([a as u32, b as u32]);
                }
            }
        }
    }
    // Peptide bonds between consecutive protein residues, and SG–SG disulfides.
    let find = |span: (usize, usize), name: &str| (span.0..span.1).find(|&i| t.names[i] == name);
    for w in residue_spans.windows(2) {
        if let (Some(c), Some(n)) = (find(w[0], "C"), find(w[1], "N")) {
            if (t.positions[c] - t.positions[n]).norm() <= 1.75 {
                bonds.push([c as u32, n as u32]);
            }
        }
    }
    let sulfurs: Vec<usize> = (0..t.names.len()).filter(|&i| t.names[i] == "SG").collect();
    for (k, &a) in sulfurs.iter().enumerate() {
        for &b in &sulfurs[k + 1..] {
            if (t.positions[a] - t.positions[b]).norm() <= 2.5 {
                bonds.push([a as u32, b as u32]);
            }
        }
    }
    bonds
}

/// Build the atom table and ligand list. `protein` is the normalised protein (the one the
/// ribbon was built from), `full` the file as read, for its ligands; `ribbon_ids` gives each
/// ribbon residue's chain, number and insertion code.
pub fn build_atoms(
    protein: &pdbtbx::PDB,
    full: &pdbtbx::PDB,
    ribbon_ids: &[ResidueKey],
) -> (AtomTable, Vec<Ligand>) {
    let index: HashMap<&ResidueKey, usize> =
        ribbon_ids.iter().enumerate().map(|(i, k)| (k, i)).collect();
    let n_res = ribbon_ids.len() as u32;
    let mut t = AtomTable::default();
    let mut spans = Vec::new();
    for chain in protein.chains() {
        for residue in chain.residues() {
            let Some(&r) = index.get(&key_of(chain, residue)) else {
                continue;
            };
            let start = t.positions.len();
            for atom in residue.atoms() {
                if is_hydrogen(atom) {
                    continue;
                }
                t.positions.push(Vector3::new(
                    atom.x() as f32,
                    atom.y() as f32,
                    atom.z() as f32,
                ));
                t.elements
                    .push(element_code(&proteus_core::io::element_symbol(atom)));
                t.residues.push(r as u32);
                t.names.push(atom.name().trim().to_string());
            }
            spans.push((start, t.positions.len()));
        }
    }
    let protein_spans = spans.clone();
    let mut ligands = Vec::new();
    let mut ligand_spans = Vec::new();
    for chain in full.chains() {
        for residue in chain.residues() {
            let name = residue.name().unwrap_or("").trim().to_string();
            if is_water(&name) || proteus_core::io::is_protein_residue(residue) {
                continue;
            }
            let start = t.positions.len();
            let id = n_res + ligands.len() as u32;
            // First conformer only, as for the protein.
            if let Some(conf) = residue.conformers().next() {
                for atom in conf.atoms() {
                    if is_hydrogen(atom) {
                        continue;
                    }
                    t.positions.push(Vector3::new(
                        atom.x() as f32,
                        atom.y() as f32,
                        atom.z() as f32,
                    ));
                    t.elements
                        .push(element_code(&proteus_core::io::element_symbol(atom)));
                    t.residues.push(id);
                    t.names.push(atom.name().trim().to_string());
                }
            }
            let count = t.positions.len() - start;
            if count == 0 {
                continue;
            }
            ligand_spans.push((start, t.positions.len()));
            ligands.push(Ligand {
                name,
                chain: chain.id().to_string(),
                number: residue.serial_number(),
                atom_count: count,
            });
        }
    }
    let mut bonds = infer_bonds(&t, &protein_spans);
    for &(start, end) in &ligand_spans {
        for a in start..end {
            for b in a + 1..end {
                let d = (t.positions[a] - t.positions[b]).norm();
                let limit = covalent_radius(t.elements[a]) + covalent_radius(t.elements[b]) + 0.45;
                if d > 0.4 && d <= limit {
                    bonds.push([a as u32, b as u32]);
                }
            }
        }
    }
    t.bonds = bonds;
    (t, ligands)
}

/// Locate the analysis's findings on the ribbon. `protein` is the structure the analysis ran on;
/// its residue enumeration (chains, then residues) is what the analyses' `res_idx` counts.
pub fn annotate(
    metrics: &BiophysicalMetrics,
    rama_points: &[(Option<f64>, Option<f64>, RamachandranRegion)],
    backbone_ids: &[ResidueKey],
    protein: &pdbtbx::PDB,
    atoms: &AtomTable,
    ribbon_ids: &[ResidueKey],
    ribbon_names: &[String],
) -> Annotations {
    let ribbon: HashMap<&ResidueKey, u32> = ribbon_ids
        .iter()
        .enumerate()
        .map(|(i, k)| (k, i as u32))
        .collect();
    // analysis res_idx → ribbon index.
    let global: Vec<Option<u32>> = protein
        .chains()
        .flat_map(|c| c.residues().map(move |r| key_of(c, r)))
        .map(|k| ribbon.get(&k).copied())
        .collect();
    let to_ribbon = |i: usize| global.get(i).copied().flatten();
    let atom_at: HashMap<(u32, &str), usize> = (0..atoms.names.len())
        .map(|i| ((atoms.residues[i], atoms.names[i].as_str()), i))
        .collect();
    let pos = |r: u32, name: &str| atom_at.get(&(r, name)).map(|&i| atoms.positions[i]);
    let label = |r: u32, atom: &str| -> String {
        let k = &ribbon_ids[r as usize];
        let res = &ribbon_names[r as usize];
        let chain = if k.0.trim().is_empty() {
            String::new()
        } else {
            format!("{}:", k.0)
        };
        let icode = k.2.clone().unwrap_or_default();
        if atom.is_empty() {
            format!("{chain}{res}{}{icode}", k.1)
        } else {
            format!("{chain}{res}{}{icode} {atom}", k.1)
        }
    };
    let mut out = Annotations::default();

    let rama_ribbon: HashMap<&ResidueKey, u32> = ribbon.clone();
    // A residue without both φ and ψ (a chain end, a gap) is not judged, as in the statistics.
    for (k, (phi, psi, region)) in backbone_ids.iter().zip(rama_points) {
        if phi.is_some() && psi.is_some() && *region != RamachandranRegion::Favored {
            if let Some(&r) = rama_ribbon.get(k) {
                out.rama.push((r, *region));
            }
        }
    }

    let pair = |ra: Option<u32>, aa: &str, rb: Option<u32>, ab: &str, value: f64| {
        let (ra, rb) = (ra?, rb?);
        Some(Contact {
            residues: [ra, rb],
            points: [pos(ra, aa)?, pos(rb, ab)?],
            value: value as f32,
            label: format!("{} – {}", label(ra, aa), label(rb, ab)),
        })
    };
    if let Some(s) = &metrics.steric_overlap {
        out.clashes = s
            .clashes
            .iter()
            .filter_map(|c| {
                pair(
                    to_ribbon(c.res1_idx),
                    &c.atom1_name,
                    to_ribbon(c.res2_idx),
                    &c.atom2_name,
                    c.overlap,
                )
            })
            .collect();
    }
    if let Some(net) = &metrics.interaction_network {
        out.hbonds = net
            .hbonds
            .iter()
            .filter_map(|h| {
                pair(
                    to_ribbon(h.donor_res_idx),
                    &h.donor_atom_name,
                    to_ribbon(h.acceptor_res_idx),
                    &h.acceptor_atom_name,
                    h.distance,
                )
            })
            .collect();
        out.salt_bridges = net
            .salt_bridges
            .iter()
            .filter_map(|s| {
                pair(
                    to_ribbon(s.cation_res_idx),
                    &s.cation_atom_name,
                    to_ribbon(s.anion_res_idx),
                    &s.anion_atom_name,
                    s.distance,
                )
            })
            .collect();
        let centroid = |r: u32, name: &str| -> Option<Vector3<f32>> {
            let ring: &[&str] = match name {
                "PHE" | "TYR" => &["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
                "HIS" => &["CG", "ND1", "CD2", "CE1", "NE2"],
                "TRP" => &["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
                _ => return None,
            };
            let pts: Vec<Vector3<f32>> = ring.iter().filter_map(|a| pos(r, a)).collect();
            (pts.len() == ring.len()).then(|| pts.iter().sum::<Vector3<f32>>() / pts.len() as f32)
        };
        out.pi_stacks = net
            .pi_pi_stacks
            .iter()
            .filter_map(|p| {
                let (a, b) = (to_ribbon(p.ring1_res_idx)?, to_ribbon(p.ring2_res_idx)?);
                Some(Contact {
                    residues: [a, b],
                    points: [
                        centroid(a, &p.ring1_res_name)?,
                        centroid(b, &p.ring2_res_name)?,
                    ],
                    value: p.centroid_distance as f32,
                    label: format!("{} – {}", label(a, ""), label(b, "")),
                })
            })
            .collect();
        out.cation_pi = net
            .cation_pi_interactions
            .iter()
            .filter_map(|c| {
                let (a, b) = (to_ribbon(c.cation_res_idx)?, to_ribbon(c.ring_res_idx)?);
                Some(Contact {
                    residues: [a, b],
                    points: [pos(a, &c.cation_atom_name)?, centroid(b, &c.ring_res_name)?],
                    value: c.distance_to_centroid as f32,
                    label: format!("{} – {} ring", label(a, &c.cation_atom_name), label(b, "")),
                })
            })
            .collect();
    }
    out
}

/// One-letter code of a residue name; `X` for anything non-standard.
pub fn one_letter(name: &str) -> char {
    match name.trim().to_ascii_uppercase().as_str() {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" | "MSE" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        "SEC" => 'U',
        "PYL" => 'O',
        _ => 'X',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CRAMBIN: &str = include_str!("../../proteus-core/tests/data/1crn.pdb");

    #[test]
    fn crambin_atoms_bond_like_a_protein() {
        let s = crate::parse_pdb_structure(CRAMBIN).unwrap();
        let t = &s.atoms;
        // 1CRN: 327 heavy atoms, all protein.
        assert_eq!(t.positions.len(), 327);
        assert!(s.ligands.is_empty());
        // A protein has about as many bonds as heavy atoms (a tree plus ring closures).
        let ratio = t.bonds.len() as f64 / t.positions.len() as f64;
        assert!((1.0..1.12).contains(&ratio), "bonds/atoms = {ratio}");
        // Every bond is a real covalent length.
        for [a, b] in &t.bonds {
            let d = (t.positions[*a as usize] - t.positions[*b as usize]).norm();
            assert!((1.1..2.2).contains(&d), "bond of {d} Å");
        }
        // The three disulfides are bonds too.
        let sg: Vec<usize> = (0..t.names.len()).filter(|&i| t.names[i] == "SG").collect();
        let ss = t
            .bonds
            .iter()
            .filter(|[a, b]| sg.contains(&(*a as usize)) && sg.contains(&(*b as usize)))
            .count();
        assert_eq!(ss, 3);
    }

    #[test]
    fn contacts_land_on_the_named_atoms() {
        let s = crate::parse_pdb_structure(CRAMBIN).unwrap();
        let a = &s.annotations;
        let m = s.metrics.as_ref().unwrap();
        let net = m.interaction_network.as_ref().unwrap();
        // Nothing is lost in the mapping on a clean single-chain file.
        assert_eq!(a.hbonds.len(), net.hbonds.len());
        assert_eq!(a.salt_bridges.len(), net.salt_bridges.len());
        for c in a.hbonds.iter().chain(&a.salt_bridges) {
            let d = (c.points[0] - c.points[1]).norm();
            assert!(
                (d - c.value).abs() < 0.01,
                "{}: {d} vs {}",
                c.label,
                c.value
            );
            assert!(c.residues.iter().all(|r| (*r as usize) < s.num_residues));
        }
    }

    #[test]
    fn rama_findings_agree_with_the_statistics() {
        let s = crate::parse_pdb_structure(CRAMBIN).unwrap();
        let stats = s
            .metrics
            .as_ref()
            .unwrap()
            .ramachandran_stats
            .as_ref()
            .unwrap();
        let outliers = s
            .annotations
            .rama
            .iter()
            .filter(|(_, r)| *r == RamachandranRegion::Outlier)
            .count();
        assert_eq!(outliers, stats.outlier_count);
    }

    #[test]
    fn a_ligand_is_kept_and_waters_are_not() {
        let pdb = "\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 10.00           C
ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00 10.00           C
ATOM      4  O   GLY A   1       1.251   2.390   0.000  1.00 10.00           O
ATOM      5  N   GLY A   2       3.332   1.536   0.000  1.00 10.00           N
ATOM      6  CA  GLY A   2       3.970   2.846   0.000  1.00 10.00           C
ATOM      7  C   GLY A   2       5.486   2.705   0.000  1.00 10.00           C
ATOM      8  O   GLY A   2       6.009   1.590   0.000  1.00 10.00           O
HETATM    9  C1  EOH A 101      10.000  10.000  10.000  1.00 10.00           C
HETATM   10  C2  EOH A 101      11.500  10.000  10.000  1.00 10.00           C
HETATM   11  O   EOH A 101      12.200  11.200  10.000  1.00 10.00           O
HETATM   12  O   HOH A 201      20.000  20.000  20.000  1.00 10.00           O
END
";
        let s = crate::parse_pdb_structure(pdb).unwrap();
        assert_eq!(s.ligands.len(), 1);
        assert_eq!(s.ligands[0].name, "EOH");
        assert_eq!(s.ligands[0].atom_count, 3);
        let lig: Vec<usize> = (0..s.atoms.residues.len())
            .filter(|&i| s.atoms.residues[i] == s.num_residues as u32)
            .collect();
        assert_eq!(lig.len(), 3);
        // C1–C2 and C2–O are bonded; the peptide bond joins the two glycines.
        assert_eq!(s.atoms.bonds.len(), 3 + 3 + 1 + 2);
    }
}
