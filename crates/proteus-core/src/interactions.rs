use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Classification of hydrogen bonds based on participating structural components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum HBondCategory {
    BackboneBackbone,
    BackboneSidechain,
    SidechainSidechain,
}

/// Hydrogen bond between a donor and acceptor heavy atom.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct HydrogenBond {
    pub donor_res_idx: usize,
    pub donor_res_seq: isize,
    pub donor_res_name: String,
    pub donor_atom_name: String,
    pub acceptor_res_idx: usize,
    pub acceptor_res_seq: isize,
    pub acceptor_res_name: String,
    pub acceptor_atom_name: String,
    pub distance: f64,
    pub donor_angle_deg: f64,
    pub acceptor_angle_deg: f64,
    pub category: HBondCategory,
}

/// Ionic salt bridge between basic cationic nitrogen and acidic anionic oxygen.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SaltBridge {
    pub cation_res_idx: usize,
    pub cation_res_seq: isize,
    pub cation_res_name: String,
    pub cation_atom_name: String,
    pub anion_res_idx: usize,
    pub anion_res_seq: isize,
    pub anion_res_name: String,
    pub anion_atom_name: String,
    pub distance: f64,
}

/// Aromatic ring orientation geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum PiStackingCategory {
    /// Parallel displaced or face-to-face stacking (|cos θ| ≥ 0.866).
    Parallel,
    /// Edge-to-face or T-shaped stacking (|cos θ| ≤ 0.500).
    TShaped,
}

/// Aromatic pi-pi stacking interaction between two conjugated rings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct PiStacking {
    pub ring1_res_idx: usize,
    pub ring1_res_seq: isize,
    pub ring1_res_name: String,
    pub ring2_res_idx: usize,
    pub ring2_res_seq: isize,
    pub ring2_res_name: String,
    pub centroid_distance: f64,
    pub normal_angle_deg: f64,
    pub category: PiStackingCategory,
}

/// Cation-pi interaction between basic amine/guanidinium and aromatic pi-system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CationPiInteraction {
    pub cation_res_idx: usize,
    pub cation_res_seq: isize,
    pub cation_res_name: String,
    pub cation_atom_name: String,
    pub ring_res_idx: usize,
    pub ring_res_seq: isize,
    pub ring_res_name: String,
    pub distance_to_centroid: f64,
    pub angle_to_normal_deg: f64,
}

/// Summary metrics of the all-atom non-covalent interaction network.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct InteractionSummary {
    pub total_hbonds: usize,
    pub bb_bb_hbonds: usize,
    pub bb_sc_hbonds: usize,
    pub sc_sc_hbonds: usize,
    pub total_salt_bridges: usize,
    pub total_pi_pi_stacks: usize,
    pub total_cation_pi: usize,
    /// Non-covalent tertiary contact density (interactions per 100 residues).
    pub network_density: f64,
}

/// Complete all-atom non-covalent interaction network.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct InteractionNetwork {
    pub summary: InteractionSummary,
    pub hbonds: Vec<HydrogenBond>,
    pub salt_bridges: Vec<SaltBridge>,
    pub pi_pi_stacks: Vec<PiStacking>,
    pub cation_pi_interactions: Vec<CationPiInteraction>,
}

#[derive(Clone)]
struct ExtractedAtom {
    res_idx: usize,
    res_seq: isize,
    res_name: String,
    atom_name: String,
    pos: Vector3<f64>,
    ante_pos: Option<Vector3<f64>>,
    is_backbone: bool,
}

#[derive(Clone)]
struct ExtractedAromaticRing {
    res_idx: usize,
    res_seq: isize,
    res_name: String,
    centroid: Vector3<f64>,
    normal: Vector3<f64>,
}

/// Computes the angle in degrees between vectors (A - vertex) and (B - vertex).
fn compute_angle_deg(vertex: &Vector3<f64>, a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let va = a - vertex;
    let vb = b - vertex;
    let norm_prod = va.norm() * vb.norm();
    if norm_prod <= 1e-9 {
        return 0.0;
    }
    let cos_theta = (va.dot(&vb) / norm_prod).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

/// Evaluates all-atom non-covalent interactions in a PDB structure using an O(N) spatial grid.
pub fn compute_interaction_network(pdb: &pdbtbx::PDB) -> InteractionNetwork {
    let mut donors: Vec<ExtractedAtom> = Vec::new();
    let mut acceptors: Vec<ExtractedAtom> = Vec::new();
    let mut cations: Vec<ExtractedAtom> = Vec::new();
    let mut anions: Vec<ExtractedAtom> = Vec::new();
    let mut aromatic_rings: Vec<ExtractedAromaticRing> = Vec::new();

    let mut global_res_idx = 0;
    let mut total_residues = 0;

    for chain in pdb.chains() {
        for residue in chain.residues() {
            total_residues += 1;
            let res_seq = residue.serial_number();
            let res_name = residue
                .name()
                .map(|n| n.trim().to_uppercase())
                .unwrap_or_else(|| "UNK".to_string());

            let mut atom_map: HashMap<String, Vector3<f64>> = HashMap::new();
            for atom in residue.atoms() {
                let name = atom.name().trim().to_uppercase();
                atom_map.insert(name, Vector3::new(atom.x(), atom.y(), atom.z()));
            }

            // 1. Backbone Donor & Acceptor
            if let (Some(&n_pos), Some(&ca_pos)) = (atom_map.get("N"), atom_map.get("CA")) {
                donors.push(ExtractedAtom {
                    res_idx: global_res_idx,
                    res_seq,
                    res_name: res_name.clone(),
                    atom_name: "N".to_string(),
                    pos: n_pos,
                    ante_pos: Some(ca_pos),
                    is_backbone: true,
                });
            }

            if let (Some(&o_pos), Some(&c_pos)) = (atom_map.get("O"), atom_map.get("C")) {
                acceptors.push(ExtractedAtom {
                    res_idx: global_res_idx,
                    res_seq,
                    res_name: res_name.clone(),
                    atom_name: "O".to_string(),
                    pos: o_pos,
                    ante_pos: Some(c_pos),
                    is_backbone: true,
                });
            }

            // 2. Sidechain Donors, Acceptors, Cations, Anions
            match res_name.as_str() {
                "ARG" => {
                    let cd = atom_map.get("CD").copied();
                    let cz = atom_map.get("CZ").copied();

                    for &(atom, ante) in &[("NE", cd.or(cz)), ("NH1", cz), ("NH2", cz)] {
                        if let Some(&pos) = atom_map.get(atom) {
                            let item = ExtractedAtom {
                                res_idx: global_res_idx,
                                res_seq,
                                res_name: res_name.clone(),
                                atom_name: atom.to_string(),
                                pos,
                                ante_pos: ante,
                                is_backbone: false,
                            };
                            donors.push(item.clone());
                            cations.push(item);
                        }
                    }
                }
                "LYS" => {
                    let ce = atom_map.get("CE").copied();
                    if let Some(&nz) = atom_map.get("NZ") {
                        let item = ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "NZ".to_string(),
                            pos: nz,
                            ante_pos: ce,
                            is_backbone: false,
                        };
                        donors.push(item.clone());
                        cations.push(item);
                    }
                }
                "HIS" => {
                    let cg = atom_map.get("CG").copied();
                    let ce1 = atom_map.get("CE1").copied();

                    for &(atom, ante) in &[("ND1", cg), ("NE2", ce1)] {
                        if let Some(&pos) = atom_map.get(atom) {
                            let item = ExtractedAtom {
                                res_idx: global_res_idx,
                                res_seq,
                                res_name: res_name.clone(),
                                atom_name: atom.to_string(),
                                pos,
                                ante_pos: ante,
                                is_backbone: false,
                            };
                            donors.push(item.clone());
                            acceptors.push(item);
                        }
                    }

                    // Aromatic ring: CG, ND1, CD2, CE1, NE2
                    let ring_atoms = ["CG", "ND1", "CD2", "CE1", "NE2"];
                    let coords: Vec<Vector3<f64>> = ring_atoms
                        .iter()
                        .filter_map(|&a| atom_map.get(a).copied())
                        .collect();
                    if coords.len() == 5 {
                        let centroid: Vector3<f64> =
                            coords.iter().sum::<Vector3<f64>>() / (coords.len() as f64);
                        let normal = (coords[1] - coords[0])
                            .cross(&(coords[2] - coords[0]))
                            .normalize();
                        aromatic_rings.push(ExtractedAromaticRing {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            centroid,
                            normal,
                        });
                    }
                }
                "ASP" => {
                    let cg = atom_map.get("CG").copied();
                    for &atom in &["OD1", "OD2"] {
                        if let Some(&pos) = atom_map.get(atom) {
                            let item = ExtractedAtom {
                                res_idx: global_res_idx,
                                res_seq,
                                res_name: res_name.clone(),
                                atom_name: atom.to_string(),
                                pos,
                                ante_pos: cg,
                                is_backbone: false,
                            };
                            acceptors.push(item.clone());
                            anions.push(item);
                        }
                    }
                }
                "GLU" => {
                    let cd = atom_map.get("CD").copied();
                    for &atom in &["OE1", "OE2"] {
                        if let Some(&pos) = atom_map.get(atom) {
                            let item = ExtractedAtom {
                                res_idx: global_res_idx,
                                res_seq,
                                res_name: res_name.clone(),
                                atom_name: atom.to_string(),
                                pos,
                                ante_pos: cd,
                                is_backbone: false,
                            };
                            acceptors.push(item.clone());
                            anions.push(item);
                        }
                    }
                }
                "ASN" => {
                    let cg = atom_map.get("CG").copied();
                    if let Some(&nd2) = atom_map.get("ND2") {
                        donors.push(ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "ND2".to_string(),
                            pos: nd2,
                            ante_pos: cg,
                            is_backbone: false,
                        });
                    }
                    if let Some(&od1) = atom_map.get("OD1") {
                        acceptors.push(ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "OD1".to_string(),
                            pos: od1,
                            ante_pos: cg,
                            is_backbone: false,
                        });
                    }
                }
                "GLN" => {
                    let cd = atom_map.get("CD").copied();
                    if let Some(&ne2) = atom_map.get("NE2") {
                        donors.push(ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "NE2".to_string(),
                            pos: ne2,
                            ante_pos: cd,
                            is_backbone: false,
                        });
                    }
                    if let Some(&oe1) = atom_map.get("OE1") {
                        acceptors.push(ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "OE1".to_string(),
                            pos: oe1,
                            ante_pos: cd,
                            is_backbone: false,
                        });
                    }
                }
                "SER" => {
                    let cb = atom_map.get("CB").copied();
                    if let Some(&og) = atom_map.get("OG") {
                        let item = ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "OG".to_string(),
                            pos: og,
                            ante_pos: cb,
                            is_backbone: false,
                        };
                        donors.push(item.clone());
                        acceptors.push(item);
                    }
                }
                "THR" => {
                    let cb = atom_map.get("CB").copied();
                    if let Some(&og1) = atom_map.get("OG1") {
                        let item = ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "OG1".to_string(),
                            pos: og1,
                            ante_pos: cb,
                            is_backbone: false,
                        };
                        donors.push(item.clone());
                        acceptors.push(item);
                    }
                }
                "TYR" => {
                    let cz = atom_map.get("CZ").copied();
                    if let Some(&oh) = atom_map.get("OH") {
                        let item = ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "OH".to_string(),
                            pos: oh,
                            ante_pos: cz,
                            is_backbone: false,
                        };
                        donors.push(item.clone());
                        acceptors.push(item);
                    }

                    // Aromatic ring: CG, CD1, CD2, CE1, CE2, CZ
                    let ring_atoms = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"];
                    let coords: Vec<Vector3<f64>> = ring_atoms
                        .iter()
                        .filter_map(|&a| atom_map.get(a).copied())
                        .collect();
                    if coords.len() == 6 {
                        let centroid: Vector3<f64> =
                            coords.iter().sum::<Vector3<f64>>() / (coords.len() as f64);
                        let normal = (coords[1] - coords[0])
                            .cross(&(coords[2] - coords[0]))
                            .normalize();
                        aromatic_rings.push(ExtractedAromaticRing {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            centroid,
                            normal,
                        });
                    }
                }
                "PHE" => {
                    let ring_atoms = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"];
                    let coords: Vec<Vector3<f64>> = ring_atoms
                        .iter()
                        .filter_map(|&a| atom_map.get(a).copied())
                        .collect();
                    if coords.len() == 6 {
                        let centroid: Vector3<f64> =
                            coords.iter().sum::<Vector3<f64>>() / (coords.len() as f64);
                        let normal = (coords[1] - coords[0])
                            .cross(&(coords[2] - coords[0]))
                            .normalize();
                        aromatic_rings.push(ExtractedAromaticRing {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            centroid,
                            normal,
                        });
                    }
                }
                "TRP" => {
                    let cd1 = atom_map.get("CD1").copied();
                    if let Some(&ne1) = atom_map.get("NE1") {
                        donors.push(ExtractedAtom {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            atom_name: "NE1".to_string(),
                            pos: ne1,
                            ante_pos: cd1,
                            is_backbone: false,
                        });
                    }

                    // 9-atom indole system
                    let ring_atoms = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"];
                    let coords: Vec<Vector3<f64>> = ring_atoms
                        .iter()
                        .filter_map(|&a| atom_map.get(a).copied())
                        .collect();
                    if coords.len() == 9 {
                        let centroid: Vector3<f64> =
                            coords.iter().sum::<Vector3<f64>>() / (coords.len() as f64);
                        let normal = (coords[1] - coords[0])
                            .cross(&(coords[2] - coords[0]))
                            .normalize();
                        aromatic_rings.push(ExtractedAromaticRing {
                            res_idx: global_res_idx,
                            res_seq,
                            res_name: res_name.clone(),
                            centroid,
                            normal,
                        });
                    }
                }
                _ => {}
            }

            global_res_idx += 1;
        }
    }

    // Spatial cell-list hashing (cell_size = 7.0 Å covers max cutoff of 6.5 Å)
    let cell_size = 7.0f64;

    // Grid for Acceptors
    let mut acceptor_grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (idx, acc) in acceptors.iter().enumerate() {
        let key = (
            (acc.pos.x / cell_size).floor() as i64,
            (acc.pos.y / cell_size).floor() as i64,
            (acc.pos.z / cell_size).floor() as i64,
        );
        acceptor_grid.entry(key).or_default().push(idx);
    }

    // Grid for Anions
    let mut anion_grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (idx, ani) in anions.iter().enumerate() {
        let key = (
            (ani.pos.x / cell_size).floor() as i64,
            (ani.pos.y / cell_size).floor() as i64,
            (ani.pos.z / cell_size).floor() as i64,
        );
        anion_grid.entry(key).or_default().push(idx);
    }

    // Grid for Aromatic Rings
    let mut aromatic_grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (idx, ring) in aromatic_rings.iter().enumerate() {
        let key = (
            (ring.centroid.x / cell_size).floor() as i64,
            (ring.centroid.y / cell_size).floor() as i64,
            (ring.centroid.z / cell_size).floor() as i64,
        );
        aromatic_grid.entry(key).or_default().push(idx);
    }

    // --- 1. Evaluate Hydrogen Bonds ---
    let mut hbonds: Vec<HydrogenBond> = Vec::new();
    let max_hbond_dist_sq = 3.5 * 3.5;
    let min_hbond_dist_sq = 2.4 * 2.4;

    for donor in &donors {
        let cx = (donor.pos.x / cell_size).floor() as i64;
        let cy = (donor.pos.y / cell_size).floor() as i64;
        let cz = (donor.pos.z / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(acc_indices) = acceptor_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &acc_idx in acc_indices {
                            let acc = &acceptors[acc_idx];

                            // Exclude intra-residue and adjacent peptide turns (|Δseq| < 2)
                            let seq_diff = (donor.res_seq - acc.res_seq).abs();
                            let idx_diff = (donor.res_idx as isize - acc.res_idx as isize).abs();
                            if seq_diff < 2 || idx_diff < 2 {
                                continue;
                            }

                            let dist_sq = (donor.pos - acc.pos).norm_squared();
                            if dist_sq < min_hbond_dist_sq || dist_sq > max_hbond_dist_sq {
                                continue;
                            }

                            // Geometric antecedent angles: both must be >= 90°
                            let donor_angle = match donor.ante_pos {
                                Some(ante) => compute_angle_deg(&donor.pos, &ante, &acc.pos),
                                None => 120.0,
                            };
                            let acceptor_angle = match acc.ante_pos {
                                Some(ante) => compute_angle_deg(&acc.pos, &ante, &donor.pos),
                                None => 120.0,
                            };

                            if donor_angle >= 90.0 && acceptor_angle >= 90.0 {
                                let category = match (donor.is_backbone, acc.is_backbone) {
                                    (true, true) => HBondCategory::BackboneBackbone,
                                    (false, false) => HBondCategory::SidechainSidechain,
                                    _ => HBondCategory::BackboneSidechain,
                                };

                                hbonds.push(HydrogenBond {
                                    donor_res_idx: donor.res_idx,
                                    donor_res_seq: donor.res_seq,
                                    donor_res_name: donor.res_name.clone(),
                                    donor_atom_name: donor.atom_name.clone(),
                                    acceptor_res_idx: acc.res_idx,
                                    acceptor_res_seq: acc.res_seq,
                                    acceptor_res_name: acc.res_name.clone(),
                                    acceptor_atom_name: acc.atom_name.clone(),
                                    distance: dist_sq.sqrt(),
                                    donor_angle_deg: donor_angle,
                                    acceptor_angle_deg: acceptor_angle,
                                    category,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // --- 2. Evaluate Salt Bridges ---
    let mut candidate_salt_bridges: Vec<SaltBridge> = Vec::new();
    let max_salt_dist_sq = 4.0 * 4.0;

    for cat in &cations {
        let cx = (cat.pos.x / cell_size).floor() as i64;
        let cy = (cat.pos.y / cell_size).floor() as i64;
        let cz = (cat.pos.z / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ani_indices) = anion_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &ani_idx in ani_indices {
                            let ani = &anions[ani_idx];
                            if cat.res_idx == ani.res_idx || cat.res_seq == ani.res_seq {
                                continue;
                            }

                            let dist_sq = (cat.pos - ani.pos).norm_squared();
                            if dist_sq <= max_salt_dist_sq {
                                candidate_salt_bridges.push(SaltBridge {
                                    cation_res_idx: cat.res_idx,
                                    cation_res_seq: cat.res_seq,
                                    cation_res_name: cat.res_name.clone(),
                                    cation_atom_name: cat.atom_name.clone(),
                                    anion_res_idx: ani.res_idx,
                                    anion_res_seq: ani.res_seq,
                                    anion_res_name: ani.res_name.clone(),
                                    anion_atom_name: ani.atom_name.clone(),
                                    distance: dist_sq.sqrt(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Deduplicate salt bridges to report the closest contact per interacting residue pair
    let mut salt_bridge_map: HashMap<(usize, usize), SaltBridge> = HashMap::new();
    for sb in candidate_salt_bridges {
        let key = (sb.cation_res_idx, sb.anion_res_idx);
        match salt_bridge_map.get(&key) {
            Some(existing) if existing.distance <= sb.distance => {}
            _ => {
                salt_bridge_map.insert(key, sb);
            }
        }
    }
    let mut salt_bridges: Vec<SaltBridge> = salt_bridge_map.into_values().collect();
    salt_bridges.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- 3. Evaluate Pi-Pi Stacking ---
    let mut pi_pi_stacks: Vec<PiStacking> = Vec::new();
    let max_pi_dist_sq = 6.5 * 6.5;
    let mut seen_ring_pairs: HashSet<(usize, usize)> = HashSet::new();

    for (r1_idx, r1) in aromatic_rings.iter().enumerate() {
        let cx = (r1.centroid.x / cell_size).floor() as i64;
        let cy = (r1.centroid.y / cell_size).floor() as i64;
        let cz = (r1.centroid.z / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(r2_indices) = aromatic_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &r2_idx in r2_indices {
                            if r1_idx >= r2_idx {
                                continue;
                            }
                            let r2 = &aromatic_rings[r2_idx];
                            if r1.res_idx == r2.res_idx || r1.res_seq == r2.res_seq {
                                continue;
                            }

                            let pair_key = (r1.res_idx.min(r2.res_idx), r1.res_idx.max(r2.res_idx));
                            if seen_ring_pairs.contains(&pair_key) {
                                continue;
                            }

                            let dist_sq = (r1.centroid - r2.centroid).norm_squared();
                            if dist_sq <= max_pi_dist_sq {
                                let cos_theta = r1.normal.dot(&r2.normal).abs().clamp(0.0, 1.0);
                                let angle_deg = cos_theta.acos().to_degrees();

                                // Parallel: cos_theta >= cos(30°) ~ 0.866
                                // T-shaped: cos_theta <= cos(60°) ~ 0.500
                                let category = if cos_theta >= 0.8660 {
                                    Some(PiStackingCategory::Parallel)
                                } else if cos_theta <= 0.5000 {
                                    Some(PiStackingCategory::TShaped)
                                } else {
                                    None
                                };

                                if let Some(cat) = category {
                                    seen_ring_pairs.insert(pair_key);
                                    pi_pi_stacks.push(PiStacking {
                                        ring1_res_idx: r1.res_idx,
                                        ring1_res_seq: r1.res_seq,
                                        ring1_res_name: r1.res_name.clone(),
                                        ring2_res_idx: r2.res_idx,
                                        ring2_res_seq: r2.res_seq,
                                        ring2_res_name: r2.res_name.clone(),
                                        centroid_distance: dist_sq.sqrt(),
                                        normal_angle_deg: angle_deg,
                                        category: cat,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- 4. Evaluate Cation-Pi Interactions ---
    let mut candidate_cation_pi: Vec<CationPiInteraction> = Vec::new();
    let max_cation_pi_dist_sq = 6.0 * 6.0;

    for cat in &cations {
        let cx = (cat.pos.x / cell_size).floor() as i64;
        let cy = (cat.pos.y / cell_size).floor() as i64;
        let cz = (cat.pos.z / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ring_indices) = aromatic_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &ring_idx in ring_indices {
                            let ring = &aromatic_rings[ring_idx];
                            if cat.res_idx == ring.res_idx || cat.res_seq == ring.res_seq {
                                continue;
                            }

                            let v = cat.pos - ring.centroid;
                            let dist_sq = v.norm_squared();
                            if dist_sq <= max_cation_pi_dist_sq {
                                let dist = dist_sq.sqrt();
                                if dist > 1e-6 {
                                    let cos_alpha =
                                        (ring.normal.dot(&v).abs() / dist).clamp(0.0, 1.0);
                                    let angle_deg = cos_alpha.acos().to_degrees();

                                    // Cation within cone of pi cloud (angle to normal <= 45°, cos_alpha >= cos(45°))
                                    if cos_alpha >= std::f64::consts::FRAC_1_SQRT_2 {
                                        candidate_cation_pi.push(CationPiInteraction {
                                            cation_res_idx: cat.res_idx,
                                            cation_res_seq: cat.res_seq,
                                            cation_res_name: cat.res_name.clone(),
                                            cation_atom_name: cat.atom_name.clone(),
                                            ring_res_idx: ring.res_idx,
                                            ring_res_seq: ring.res_seq,
                                            ring_res_name: ring.res_name.clone(),
                                            distance_to_centroid: dist,
                                            angle_to_normal_deg: angle_deg,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Deduplicate cation-pi to report the closest atom per cation residue and aromatic ring
    let mut cation_pi_map: HashMap<(usize, usize), CationPiInteraction> = HashMap::new();
    for cpi in candidate_cation_pi {
        let key = (cpi.cation_res_idx, cpi.ring_res_idx);
        match cation_pi_map.get(&key) {
            Some(existing) if existing.distance_to_centroid <= cpi.distance_to_centroid => {}
            _ => {
                cation_pi_map.insert(key, cpi);
            }
        }
    }
    let mut cation_pi_interactions: Vec<CationPiInteraction> =
        cation_pi_map.into_values().collect();
    cation_pi_interactions.sort_by(|a, b| {
        a.distance_to_centroid
            .partial_cmp(&b.distance_to_centroid)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- 5. Compile Summary ---
    let bb_bb_hbonds = hbonds
        .iter()
        .filter(|h| h.category == HBondCategory::BackboneBackbone)
        .count();
    let bb_sc_hbonds = hbonds
        .iter()
        .filter(|h| h.category == HBondCategory::BackboneSidechain)
        .count();
    let sc_sc_hbonds = hbonds
        .iter()
        .filter(|h| h.category == HBondCategory::SidechainSidechain)
        .count();

    let total_interactions =
        hbonds.len() + salt_bridges.len() + pi_pi_stacks.len() + cation_pi_interactions.len();
    let network_density = if total_residues > 0 {
        (total_interactions as f64) / (total_residues as f64) * 100.0
    } else {
        0.0
    };

    let summary = InteractionSummary {
        total_hbonds: hbonds.len(),
        bb_bb_hbonds,
        bb_sc_hbonds,
        sc_sc_hbonds,
        total_salt_bridges: salt_bridges.len(),
        total_pi_pi_stacks: pi_pi_stacks.len(),
        total_cation_pi: cation_pi_interactions.len(),
        network_density,
    };

    InteractionNetwork {
        summary,
        hbonds,
        salt_bridges,
        pi_pi_stacks,
        cation_pi_interactions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_angle_deg() {
        let v = Vector3::new(0.0, 0.0, 0.0);
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        let angle = compute_angle_deg(&v, &a, &b);
        assert!((angle - 90.0).abs() < 1e-6);

        let c = Vector3::new(-1.0, 0.0, 0.0);
        let angle_180 = compute_angle_deg(&v, &a, &c);
        assert!((angle_180 - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_crambin_interaction_network() {
        const CRAMBIN_PDB: &str = include_str!("../tests/data/1crn.pdb");
        let cursor = std::io::Cursor::new(CRAMBIN_PDB.as_bytes());
        let (pdb, _) = pdbtbx::open_raw(
            std::io::BufReader::new(cursor),
            pdbtbx::StrictnessLevel::Loose,
        )
        .expect("Failed to parse Crambin PDB");

        let network = compute_interaction_network(&pdb);

        // Crambin (46 residues) contains 2 major alpha-helices and a beta-sheet.
        // It must have extensive backbone hydrogen bonding (> 15 BB-BB hbonds).
        assert!(
            network.summary.total_hbonds >= 20,
            "Expected >= 20 H-bonds in Crambin, got {}",
            network.summary.total_hbonds
        );
        assert!(
            network.summary.bb_bb_hbonds >= 15,
            "Expected >= 15 backbone H-bonds in Crambin, got {}",
            network.summary.bb_bb_hbonds
        );
        assert!(
            network.summary.network_density > 40.0,
            "Expected network density > 40.0, got {}",
            network.summary.network_density
        );

        // Crambin contains Phe13, Tyr29, Tyr44. It should detect aromatic interactions.
        assert!(
            network.summary.total_pi_pi_stacks + network.summary.total_cation_pi >= 1,
            "Expected aromatic interactions in Crambin"
        );
    }

    #[test]
    fn test_synthetic_salt_bridge() {
        let pdb_str = r#"HEADER    SYNTHETIC SALT BRIDGE
ATOM      1  N   ARG A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ARG A   1       1.200   0.000   0.000  1.00 90.00           C
ATOM      3  C   ARG A   1       2.000   1.200   0.000  1.00 90.00           C
ATOM      4  O   ARG A   1       2.000   2.000   1.000  1.00 90.00           O
ATOM      5  CB  ARG A   1       1.500  -1.000   0.000  1.00 90.00           C
ATOM      6  CG  ARG A   1       2.500  -1.000   0.000  1.00 90.00           C
ATOM      7  CD  ARG A   1       3.000  -4.000   0.000  1.00 90.00           C
ATOM      8  NE  ARG A   1       4.000  -4.000   0.000  1.00 90.00           N
ATOM      9  CZ  ARG A   1       4.500  -3.000   0.000  1.00 90.00           C
ATOM     10  NH1 ARG A   1       5.000  -3.000   0.000  1.00 90.00           N
ATOM     11  N   ASP A   5       0.000   5.000   0.000  1.00 90.00           N
ATOM     12  CA  ASP A   5       1.000   5.000   0.000  1.00 90.00           C
ATOM     13  C   ASP A   5       2.000   6.000   0.000  1.00 90.00           C
ATOM     14  O   ASP A   5       2.000   7.000   0.000  1.00 90.00           O
ATOM     15  CB  ASP A   5       1.500   4.000   0.000  1.00 90.00           C
ATOM     16  CG  ASP A   5       3.000   4.000   0.000  1.00 90.00           C
ATOM     17  OD1 ASP A   5       5.000  -0.500   0.000  1.00 90.00           O
END
"#;
        // Dist between ARG NH1 (5.0, -3.0, 0.0) and ASP OD1 (5.0, -0.5, 0.0) is 2.5 Å (<= 4.0 Å)
        let cursor = std::io::Cursor::new(pdb_str.as_bytes());
        let (pdb, _) = pdbtbx::open_raw(
            std::io::BufReader::new(cursor),
            pdbtbx::StrictnessLevel::Loose,
        )
        .expect("Failed to parse synthetic PDB");

        let network = compute_interaction_network(&pdb);
        assert_eq!(network.salt_bridges.len(), 1);
        let sb = &network.salt_bridges[0];
        assert_eq!(sb.cation_res_name, "ARG");
        assert_eq!(sb.anion_res_name, "ASP");
        assert!((sb.distance - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_synthetic_pi_stacking_parallel_and_t_shaped() {
        let pdb_str = r#"HEADER    SYNTHETIC PI STACKING
ATOM      1  N   PHE A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  PHE A   1       1.000   0.000   0.000  1.00 90.00           C
ATOM      3  C   PHE A   1       2.000   0.000   0.000  1.00 90.00           C
ATOM      4  O   PHE A   1       2.000   1.000   0.000  1.00 90.00           O
ATOM      5  CB  PHE A   1       1.000  -1.000   0.000  1.00 90.00           C
ATOM      6  CG  PHE A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      7  CD1 PHE A   1       1.000   0.000   0.000  1.00 90.00           C
ATOM      8  CD2 PHE A   1       0.000   1.000   0.000  1.00 90.00           C
ATOM      9  CE1 PHE A   1       1.000   1.000   0.000  1.00 90.00           C
ATOM     10  CE2 PHE A   1       0.500   1.500   0.000  1.00 90.00           C
ATOM     11  CZ  PHE A   1       1.500   1.500   0.000  1.00 90.00           C
ATOM     12  N   PHE A   5       0.000   0.000   4.000  1.00 90.00           N
ATOM     13  CA  PHE A   5       1.000   0.000   4.000  1.00 90.00           C
ATOM     14  C   PHE A   5       2.000   0.000   4.000  1.00 90.00           C
ATOM     15  O   PHE A   5       2.000   1.000   4.000  1.00 90.00           O
ATOM     16  CB  PHE A   5       1.000  -1.000   4.000  1.00 90.00           C
ATOM     17  CG  PHE A   5       0.000   0.000   4.000  1.00 90.00           C
ATOM     18  CD1 PHE A   5       1.000   0.000   4.000  1.00 90.00           C
ATOM     19  CD2 PHE A   5       0.000   1.000   4.000  1.00 90.00           C
ATOM     20  CE1 PHE A   5       1.000   1.000   4.000  1.00 90.00           C
ATOM     21  CE2 PHE A   5       0.500   1.500   4.000  1.00 90.00           C
ATOM     22  CZ  PHE A   5       1.500   1.500   4.000  1.00 90.00           C
END
"#;
        let cursor = std::io::Cursor::new(pdb_str.as_bytes());
        let (pdb, _) = pdbtbx::open_raw(
            std::io::BufReader::new(cursor),
            pdbtbx::StrictnessLevel::Loose,
        )
        .expect("Failed to parse synthetic PDB");

        let network = compute_interaction_network(&pdb);
        assert_eq!(network.pi_pi_stacks.len(), 1);
        let stack = &network.pi_pi_stacks[0];
        assert_eq!(stack.category, PiStackingCategory::Parallel);
        assert!((stack.centroid_distance - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_synthetic_cation_pi() {
        let pdb_str = r#"HEADER    SYNTHETIC CATION PI
ATOM      1  N   PHE A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  PHE A   1       1.000   0.000   0.000  1.00 90.00           C
ATOM      3  C   PHE A   1       2.000   0.000   0.000  1.00 90.00           C
ATOM      4  O   PHE A   1       2.000   1.000   0.000  1.00 90.00           O
ATOM      5  CB  PHE A   1       1.000  -1.000   0.000  1.00 90.00           C
ATOM      6  CG  PHE A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      7  CD1 PHE A   1       1.000   0.000   0.000  1.00 90.00           C
ATOM      8  CD2 PHE A   1       0.000   1.000   0.000  1.00 90.00           C
ATOM      9  CE1 PHE A   1       1.000   1.000   0.000  1.00 90.00           C
ATOM     10  CE2 PHE A   1       0.500   1.500   0.000  1.00 90.00           C
ATOM     11  CZ  PHE A   1       1.500   1.500   0.000  1.00 90.00           C
ATOM     12  N   LYS A   5       0.000   0.000   8.000  1.00 90.00           N
ATOM     13  CA  LYS A   5       1.000   0.000   8.000  1.00 90.00           C
ATOM     14  C   LYS A   5       2.000   0.000   8.000  1.00 90.00           C
ATOM     15  O   LYS A   5       2.000   1.000   8.000  1.00 90.00           O
ATOM     16  CB  LYS A   5       1.000  -1.000   8.000  1.00 90.00           C
ATOM     17  CG  LYS A   5       1.000  -1.000   7.000  1.00 90.00           C
ATOM     18  CD  LYS A   5       1.000  -1.000   6.000  1.00 90.00           C
ATOM     19  CE  LYS A   5       1.000  -1.000   5.000  1.00 90.00           C
ATOM     20  NZ  LYS A   5       0.667   0.667   4.000  1.00 90.00           N
END
"#;
        let cursor = std::io::Cursor::new(pdb_str.as_bytes());
        let (pdb, _) = pdbtbx::open_raw(
            std::io::BufReader::new(cursor),
            pdbtbx::StrictnessLevel::Loose,
        )
        .expect("Failed to parse synthetic PDB");

        let network = compute_interaction_network(&pdb);
        assert_eq!(network.cation_pi_interactions.len(), 1);
        let cpi = &network.cation_pi_interactions[0];
        assert_eq!(cpi.cation_res_name, "LYS");
        assert_eq!(cpi.ring_res_name, "PHE");
        assert!((cpi.distance_to_centroid - 4.0).abs() < 1e-2);
        assert!(cpi.angle_to_normal_deg < 5.0);
    }
}
