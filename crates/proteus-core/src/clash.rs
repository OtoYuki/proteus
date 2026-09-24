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

/// Heavy-atom steric overlap statistics.
///
/// **Not** the MolProbity clashscore: MolProbity adds hydrogens (Reduce) before counting
/// ≥ 0.4 Å overlaps of the H-inclusive van der Waals envelope; this metric uses heavy atoms
/// only and therefore under-counts on essentially every deposited structure. It is useful as a
/// relative screen for grossly overlapping predicted models, not as a MolProbity-comparable
/// number.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct StericOverlapStats {
    /// Total number of severe heavy-atom overlaps (> 0.40 Å).
    pub clash_count: usize,
    /// Overlaps per 1,000 heavy atoms evaluated (MolProbity-style normalisation, no hydrogens).
    pub heavy_atom_overlap_score: f64,
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

/// Hydrogen-bond roles of a heavy atom: (can donate, can accept). Backbone N donates (not in
/// proline), backbone O and the C-terminal OXT accept; side chains as in the usual tables
/// (hydroxyls and histidine both ways, amide N and the basic nitrogens donate, carboxylate and
/// amide O accept). Waters both. Everything else neither.
pub fn hbond_roles(res_name: &str, atom_name: &str) -> (bool, bool) {
    match (res_name, atom_name) {
        ("HOH" | "WAT" | "DOD", _) => (true, true),
        ("PRO", "N") => (false, false),
        (_, "N") => (true, false),
        (_, "O" | "OXT") => (false, true),
        ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => (true, true),
        ("HIS", "ND1" | "NE2") => (true, true),
        ("ASN", "ND2") | ("GLN", "NE2") | ("TRP", "NE1") | ("LYS", "NZ") => (true, false),
        ("ARG", "NE" | "NH1" | "NH2") => (true, false),
        ("ASN", "OD1") | ("GLN", "OE1") => (false, true),
        ("ASP", "OD1" | "OD2") | ("GLU", "OE1" | "OE2") => (false, true),
        _ => (false, false),
    }
}

/// Shortest donor–acceptor distance treated as a hydrogen bond rather than a clash, in Å.
/// Heavy-atom radii (N 1.55, O 1.52) put every ordinary N–H···O bond (2.6–3.1 Å) past the
/// 0.4 Å overlap line; MolProbity only avoids calling them clashes because it adds the
/// hydrogens. 2.4 Å is below the shortest common hydrogen bonds.
pub const HBOND_MIN_DISTANCE: f64 = 2.4;

/// Internal representation of an atom for clash detection.
struct ClashAtom {
    chain_id: String,
    res_idx: usize,
    res_seq: isize,
    res_name: String,
    atom_name: String,
    pos: Vector3<f64>,
    radius: f64,
    is_backbone: bool,
}

/// Evaluates all-atom steric clashes in a PDB structure using an O(N) spatial grid.
/// An overlap is classified as a severe clash if:
/// `overlap = r_vdw(A) + r_vdw(B) - distance > 0.40 Å`
pub fn compute_steric_overlap(pdb: &pdbtbx::PDB) -> StericOverlapStats {
    let mut atoms: Vec<ClashAtom> = Vec::new();
    let mut global_res_idx = 0;

    for chain in pdb.chains() {
        let chain_id = chain.id().to_string();
        for residue in chain.residues() {
            let res_seq = residue.serial_number();
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

                // Exclude explicit hydrogens if present to match standard heavy-atom heavy_atom_overlap_score
                if elem_symbol.eq_ignore_ascii_case("H") {
                    continue;
                }

                let is_bb = matches!(atom_name.as_str(), "N" | "CA" | "C" | "O");
                let radius = vdw_radius(&elem_symbol, &atom_name);
                let pos = Vector3::new(atom.x(), atom.y(), atom.z());

                atoms.push(ClashAtom {
                    chain_id: chain_id.clone(),
                    res_idx: global_res_idx,
                    res_seq,
                    res_name: res_name.clone(),
                    atom_name,
                    pos,
                    radius,
                    is_backbone: is_bb,
                });
            }
            global_res_idx += 1;
        }
    }

    let total_atoms = atoms.len();
    if total_atoms < 2 {
        return StericOverlapStats {
            clash_count: 0,
            heavy_atom_overlap_score: 0.0,
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

                            // Exclusion 2: Adjacent residue backbone and Proline pyrrolidine ring linkages (same chain only)
                            if atom_a.chain_id == atom_b.chain_id {
                                let res_diff =
                                    (atom_a.res_idx as isize - atom_b.res_idx as isize).abs();
                                let seq_diff = (atom_a.res_seq - atom_b.res_seq).abs();
                                if res_diff == 1 || seq_diff == 1 {
                                    let is_bb_a = atom_a.is_backbone
                                        || (atom_a.res_name == "PRO" && atom_a.atom_name == "CD");
                                    let is_bb_b = atom_b.is_backbone
                                        || (atom_b.res_name == "PRO" && atom_b.atom_name == "CD");
                                    if is_bb_a && is_bb_b {
                                        continue;
                                    }
                                }
                            }

                            // Exclusion 3: Disulfide bridge bonded cysteine pairs
                            if disulfide_pairs.contains(&(atom_a.res_idx, atom_b.res_idx)) {
                                continue;
                            }

                            let dist = (atom_a.pos - atom_b.pos).norm();

                            // Exclusion 4: a donor and an acceptor at hydrogen-bond distance.
                            if dist >= HBOND_MIN_DISTANCE {
                                let (da, aa) = hbond_roles(&atom_a.res_name, &atom_a.atom_name);
                                let (db, ab) = hbond_roles(&atom_b.res_name, &atom_b.atom_name);
                                if (da && ab) || (db && aa) {
                                    continue;
                                }
                            }

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
    let heavy_atom_overlap_score = if total_atoms > 0 {
        (clash_count as f64 * 1000.0) / (total_atoms as f64)
    } else {
        0.0
    };

    StericOverlapStats {
        clash_count,
        heavy_atom_overlap_score,
        worst_overlap,
        total_atoms_evaluated: total_atoms,
        clashes,
    }
}

#[cfg(test)]
mod hbond_tests {
    use super::*;

    fn two(res1: &str, a1: &str, res2: &str, a2: &str, el1: &str, el2: &str, d: f64) -> String {
        format!(
            "ATOM      1  {a1:<3} {res1} A   1       0.000   0.000   0.000  1.00 90.00           {el1}\n\
             ATOM      2  {a2:<3} {res2} B  20    {d:>8.3}   0.000   0.000  1.00 90.00           {el2}\nEND\n"
        )
    }

    #[test]
    fn a_hydrogen_bond_is_not_a_clash_but_two_carbonyls_are() {
        let open = |t: &str| crate::io::open_structure_bytes(t.as_bytes(), Some("x.pdb")).unwrap();
        // Backbone N–H···O at 2.6 Å: overlap 0.47 Å by radii, and a hydrogen bond.
        let hb = compute_steric_overlap(&open(&two("ALA", "N", "GLY", "O", "N", "O", 2.6)));
        assert_eq!(hb.clash_count, 0);
        // Lys NZ – Glu OE1 salt bridge at 2.5 Å.
        let sb = compute_steric_overlap(&open(&two("LYS", "NZ", "GLU", "OE1", "N", "O", 2.5)));
        assert_eq!(sb.clash_count, 0);
        // Two carbonyl oxygens cannot hydrogen-bond: 2.5 Å is a clash.
        let oo = compute_steric_overlap(&open(&two("ALA", "O", "GLY", "O", "O", "O", 2.5)));
        assert_eq!(oo.clash_count, 1);
        // Below 2.4 Å even a donor–acceptor pair is too close.
        let close = compute_steric_overlap(&open(&two("ALA", "N", "GLY", "O", "N", "O", 2.2)));
        assert_eq!(close.clash_count, 1);
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
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Pdb)
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(cursor))
            .expect("Failed to parse Crambin PDB");

        let stats = compute_steric_overlap(&pdb);
        assert_eq!(stats.clash_count, 0);
        assert_eq!(stats.heavy_atom_overlap_score, 0.0);
        assert!(stats.total_atoms_evaluated > 300);
    }
}
