use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Severe steric overlap between two non-bonded atoms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct StericClash {
    pub res1_idx: usize,
    pub res1_name: String,
    pub atom1_name: String,
    pub res2_idx: usize,
    pub res2_name: String,
    pub atom2_name: String,
    pub distance: f64,
    pub overlap: f64,
}

/// Summary of crystallographic steric clashes per the MolProbity standard.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ClashStats {
    /// Total number of severe steric overlaps (> 0.40 Å).
    pub clash_count: usize,
    /// MolProbity clashscore: number of serious overlaps per 1,000 evaluated atoms.
    pub clashscore: f64,
    /// Maximum overlap distance in angstroms.
    pub worst_overlap: f64,
    /// Total number of heavy atoms evaluated.
    pub total_atoms_evaluated: usize,
    /// Detailed list of detected steric clashes.
    pub clashes: Vec<StericClash>,
}

/// Standard Bondi (1964) / MolProbity (Lovell 2003) van der Waals radius in angstroms.
pub fn vdw_radius(element: &str, atom_name: &str) -> f64 {
    let elem = element.trim().to_uppercase();
    let name = atom_name.trim().to_uppercase();

    match elem.as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        "SE" => 1.90,
        _ => match name.chars().next() {
            Some('C') => 1.70,
            Some('N') => 1.55,
            Some('O') => 1.52,
            Some('S') => 1.80,
            Some('H') => 1.20,
            _ => 1.70,
        },
    }
}

/// Internal representation of an atom for clash detection.
struct ClashAtom {
    res_idx: usize,
    res_name: String,
    atom_name: String,
    pos: Vector3<f64>,
    radius: f64,
    is_backbone: bool,
}

/// Evaluates all-atom steric clashes in a PDB structure using an O(N) spatial grid.
/// An overlap is classified as a severe clash if:
/// `overlap = r_vdw(A) + r_vdw(B) - distance > 0.40 Å`
pub fn compute_clash_stats(pdb: &pdbtbx::PDB) -> ClashStats {
    let mut atoms: Vec<ClashAtom> = Vec::new();

    for (res_idx, residue) in pdb.residues().enumerate() {
        let res_name = residue
            .name()
            .map(|n| n.trim().to_string())
            .unwrap_or_else(|| "UNK".to_string());

        for atom in residue.atoms() {
            let atom_name = atom.name().trim().to_string();
            let elem_symbol = atom
                .element()
                .map(|e| e.symbol().to_string())
                .unwrap_or_else(|| atom_name.chars().next().unwrap_or('C').to_string());

            // Exclude explicit hydrogens if present to match standard heavy-atom clashscore
            if elem_symbol.eq_ignore_ascii_case("H") {
                continue;
            }

            let is_bb = matches!(atom_name.as_str(), "N" | "CA" | "C" | "O");
            let radius = vdw_radius(&elem_symbol, &atom_name);
            let pos = Vector3::new(atom.x(), atom.y(), atom.z());

            atoms.push(ClashAtom {
                res_idx,
                res_name: res_name.clone(),
                atom_name,
                pos,
                radius,
                is_backbone: is_bb,
            });
        }
    }

    let total_atoms = atoms.len();
    if total_atoms < 2 {
        return ClashStats {
            clash_count: 0,
            clashscore: 0.0,
            worst_overlap: 0.0,
            total_atoms_evaluated: total_atoms,
            clashes: Vec::new(),
        };
    }

    // Spatial cell-list hashing with cell_size = 4.0 Å
    let cell_size = 4.0f64;
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

    // Pre-identify covalent disulfide bonds (CYS SG - CYS SG within [1.70, 2.60] Å)
    let mut disulfide_pairs = std::collections::HashSet::new();
    let cys_sulfurs: Vec<(usize, Vector3<f64>)> = atoms
        .iter()
        .filter(|a| a.res_name == "CYS" && a.atom_name == "SG")
        .map(|a| (a.res_idx, a.pos))
        .collect();

    for i in 0..cys_sulfurs.len() {
        for j in (i + 1)..cys_sulfurs.len() {
            let d = (cys_sulfurs[i].1 - cys_sulfurs[j].1).norm();
            if (1.70..=2.60).contains(&d) {
                disulfide_pairs.insert((cys_sulfurs[i].0, cys_sulfurs[j].0));
                disulfide_pairs.insert((cys_sulfurs[j].0, cys_sulfurs[i].0));
            }
        }
    }

    for (idx, atom) in atoms.iter().enumerate() {
        let cx = (atom.pos.x / cell_size).floor() as i64;
        let cy = (atom.pos.y / cell_size).floor() as i64;
        let cz = (atom.pos.z / cell_size).floor() as i64;
        grid.entry((cx, cy, cz)).or_default().push(idx);
    }

    let mut clashes = Vec::new();
    let mut worst_overlap = 0.0f64;

    for (idx_a, atom_a) in atoms.iter().enumerate() {
        let cx = (atom_a.pos.x / cell_size).floor() as i64;
        let cy = (atom_a.pos.y / cell_size).floor() as i64;
        let cz = (atom_a.pos.z / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(neighbors) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx_b in neighbors {
                            if idx_a >= idx_b {
                                continue;
                            }
                            let atom_b = &atoms[idx_b];

                            // Exclusion 1: Same residue
                            if atom_a.res_idx == atom_b.res_idx {
                                continue;
                            }

                            // Exclusion 2: Adjacent residue backbone and Proline pyrrolidine ring linkages
                            let res_diff =
                                (atom_a.res_idx as isize - atom_b.res_idx as isize).abs();
                            if res_diff == 1 {
                                let is_bb_a = atom_a.is_backbone
                                    || (atom_a.res_name == "PRO" && atom_a.atom_name == "CD");
                                let is_bb_b = atom_b.is_backbone
                                    || (atom_b.res_name == "PRO" && atom_b.atom_name == "CD");
                                if is_bb_a && is_bb_b {
                                    continue;
                                }
                            }

                            // Exclusion 3: Disulfide bridge bonded cysteine pairs
                            if disulfide_pairs.contains(&(atom_a.res_idx, atom_b.res_idx)) {
                                continue;
                            }

                            let dist = (atom_a.pos - atom_b.pos).norm();

                            let sum_radii = atom_a.radius + atom_b.radius;
                            let overlap = sum_radii - dist;

                            // MolProbity threshold: overlap > 0.40 Å is a serious steric clash
                            if overlap > 0.40 {
                                if overlap > worst_overlap {
                                    worst_overlap = overlap;
                                }
                                clashes.push(StericClash {
                                    res1_idx: atom_a.res_idx,
                                    res1_name: atom_a.res_name.clone(),
                                    atom1_name: atom_a.atom_name.clone(),
                                    res2_idx: atom_b.res_idx,
                                    res2_name: atom_b.res_name.clone(),
                                    atom2_name: atom_b.atom_name.clone(),
                                    distance: dist,
                                    overlap,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    let clash_count = clashes.len();
    let clashscore = if total_atoms > 0 {
        (clash_count as f64 * 1000.0) / (total_atoms as f64)
    } else {
        0.0
    };

    ClashStats {
        clash_count,
        clashscore,
        worst_overlap,
        total_atoms_evaluated: total_atoms,
        clashes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdw_radii_lookup() {
        assert_eq!(vdw_radius("C", "CA"), 1.70);
        assert_eq!(vdw_radius("N", "N"), 1.55);
        assert_eq!(vdw_radius("O", "O"), 1.52);
        assert_eq!(vdw_radius("S", "SG"), 1.80);
        assert_eq!(vdw_radius("H", "H"), 1.20);
    }

    #[test]
    fn test_crambin_steric_clash_detection() {
        const CRAMBIN_PDB: &str = include_str!("../tests/data/1crn.pdb");
        let cursor = std::io::Cursor::new(CRAMBIN_PDB.as_bytes());
        let (pdb, _) = pdbtbx::open_raw(
            std::io::BufReader::new(cursor),
            pdbtbx::StrictnessLevel::Loose,
        )
        .expect("Failed to parse Crambin PDB");

        let stats = compute_clash_stats(&pdb);
        assert_eq!(stats.clash_count, 0);
        assert_eq!(stats.clashscore, 0.0);
        assert!(stats.total_atoms_evaluated > 300);
    }
}
