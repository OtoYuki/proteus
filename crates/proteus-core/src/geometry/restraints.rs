//! Build the covalent restraints of a protein model and measure it against them.
//!
//! This reproduces what cctbx `pdb_interpretation` builds for the 20 standard amino acids
//! without hydrogens, with Phenix's default restraint library (geostd monomers, CDL v1.2,
//! EH99 cis-proline), and what `mmtbx/validation/restraints.py` then reports. Every rule below
//! names the cctbx code it comes from; `validate/geometry_ref.py` checks the result structure by
//! structure.

use std::collections::HashMap;

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

use super::cdl::{self, CdlGroup, CdlTerm};
use super::library::{COO, LINKS, RESIDUES};
use super::library_types::{AngleDef, BondDef, PlaneDef, ResidueDef};
use super::Residue;

/// `pdb_interpretation.link_distance_cutoff`: a peptide or disulfide bond longer than this is
/// treated as broken and none of its restraints are made.
pub(crate) const LINK_DISTANCE_CUTOFF: f64 = 3.0;
/// `pdb_interpretation.chir_volume_esd`.
pub(crate) const CHIRAL_VOLUME_ESD: f64 = 0.2;
/// `multi_residue_class._define_omega_a_la_duke_using_limit(limit=45)`, used by the CDL to
/// decide that the peptide before a residue is cis.
const CDL_CIS_LIMIT: f64 = 45.0;
/// Disulfide restraints (the geostd `SS` link as cctbx applies it): SG–SG and CB–SG–SG.
const SS_BOND: (f64, f64) = (2.031, 0.020);
const SS_ANGLE: (f64, f64) = (104.2, 2.1);

/// Where a restraint comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The monomer definition (possibly updated by the CDL or the C-terminal mod).
    Residue,
    /// The peptide link between consecutive residues.
    Peptide,
    /// A disulfide bridge.
    Disulfide,
}

#[derive(Clone, Debug)]
pub(crate) struct BondRestraint {
    pub atoms: [usize; 2],
    pub ideal: f64,
    pub esd: f64,
    pub origin: Origin,
}

#[derive(Clone, Debug)]
pub(crate) struct AngleRestraint {
    pub atoms: [usize; 3],
    pub ideal: f64,
    pub esd: f64,
    pub origin: Origin,
}

#[derive(Clone, Debug)]
pub(crate) struct ChiralRestraint {
    pub atoms: [usize; 4],
    pub ideal: f64,
    pub both_signs: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PlaneRestraint {
    pub atoms: Vec<usize>,
    pub esds: Vec<f64>,
    pub origin: Origin,
}

/// All restraints of one structure, over atom indices into [`super::Model::atoms`].
#[derive(Default, Debug)]
pub(crate) struct Restraints {
    pub bonds: Vec<BondRestraint>,
    pub angles: Vec<AngleRestraint>,
    pub chiralities: Vec<ChiralRestraint>,
    pub planes: Vec<PlaneRestraint>,
    /// Bond and angle lookup by atoms, for the CDL updates (orientation-free keys).
    bond_index: HashMap<(usize, usize), usize>,
    angle_index: HashMap<(usize, usize, usize), usize>,
}

fn library(name: &str) -> Option<&'static ResidueDef> {
    RESIDUES.iter().find(|r| r.name == name)
}

impl Restraints {
    fn push_bond(&mut self, r: BondRestraint) {
        let [a, b] = r.atoms;
        self.bond_index.insert((a.min(b), a.max(b)), self.bonds.len());
        self.bonds.push(r);
    }

    fn push_angle(&mut self, r: AngleRestraint) {
        let [a, b, c] = r.atoms;
        self.angle_index
            .insert((a.min(c), b, a.max(c)), self.angles.len());
        self.angles.push(r);
    }

    fn find_bond(&mut self, a: usize, b: usize) -> Option<&mut BondRestraint> {
        let k = *self.bond_index.get(&(a.min(b), a.max(b)))?;
        Some(&mut self.bonds[k])
    }

    fn find_angle(&mut self, a: usize, b: usize, c: usize) -> Option<&mut AngleRestraint> {
        let k = *self.angle_index.get(&(a.min(c), b, a.max(c)))?;
        Some(&mut self.angles[k])
    }

    fn add_bond(&mut self, res: &Residue, d: &BondDef, origin: Origin) {
        if let (Some(a), Some(b)) = (res.atom(d.a), res.atom(d.b)) {
            self.push_bond(BondRestraint {
                atoms: [a, b],
                ideal: d.ideal,
                esd: d.esd,
                origin,
            });
        }
    }

    fn add_angle(&mut self, res: &Residue, d: &AngleDef, origin: Origin) {
        if let (Some(a), Some(b), Some(c)) = (res.atom(d.a), res.atom(d.b), res.atom(d.c)) {
            self.push_angle(AngleRestraint {
                atoms: [a, b, c],
                ideal: d.ideal,
                esd: d.esd,
                origin,
            });
        }
    }

    /// `add_planarity_proxies`: absent atoms are dropped, and a plane needs four atoms left.
    fn add_plane(&mut self, atoms: impl Iterator<Item = (Option<usize>, f64)>, origin: Origin) {
        let (atoms, esds): (Vec<usize>, Vec<f64>) =
            atoms.filter_map(|(i, e)| i.map(|i| (i, e))).unzip();
        if atoms.len() >= 4 {
            self.planes.push(PlaneRestraint {
                atoms,
                esds,
                origin,
            });
        }
    }

    /// Restraints of one residue from its monomer definition, with the C-terminal carboxylate
    /// modification applied when the residue carries an OXT (`_get_mod_mod_ids`: the mod is
    /// chosen by the presence of OXT, not by the residue's position).
    fn add_residue(&mut self, res: &Residue, def: &ResidueDef) {
        let coo = res.atom("OXT").is_some();
        for d in def.bonds {
            let d = COO
                .bonds
                .iter()
                .find(|m| !m.add && coo && same_pair(&m.bond, d))
                .map_or(d, |m| &m.bond);
            self.add_bond(res, d, Origin::Residue);
        }
        for d in def.angles {
            let d = COO
                .angles
                .iter()
                .find(|m| !m.add && coo && same_angle(&m.angle, d))
                .map_or(d, |m| &m.angle);
            self.add_angle(res, d, Origin::Residue);
        }
        if coo {
            for m in COO.bonds.iter().filter(|m| m.add) {
                self.add_bond(res, &m.bond, Origin::Residue);
            }
            for m in COO.angles.iter().filter(|m| m.add) {
                self.add_angle(res, &m.angle, Origin::Residue);
            }
        }
        for c in def.chiralities {
            let atoms = c.atoms.map(|n| res.atom(n));
            if let [Some(a), Some(b), Some(cc), Some(d)] = atoms {
                self.chiralities.push(ChiralRestraint {
                    atoms: [a, b, cc, d],
                    ideal: c.ideal,
                    both_signs: c.both_signs,
                });
            }
        }
        let mod_planes: &[PlaneDef] = if coo { COO.planes } else { &[] };
        for plane in def.planes.iter().chain(mod_planes) {
            self.add_plane(plane.iter().map(|&(n, e)| (res.atom(n), e)), Origin::Residue);
        }
    }

    /// The peptide link from `a` to the following residue `b` (TRANS, or PTRANS when `b` is a
    /// proline). Returns false when the C–N bond is missing or longer than
    /// [`LINK_DISTANCE_CUTOFF`], in which case cctbx records a chain break and no link
    /// restraint is made.
    fn add_link(&mut self, model: &super::Model, a: &Residue, b: &Residue) -> bool {
        let (Some(c), Some(n)) = (a.atom("C"), b.atom("N")) else {
            return false;
        };
        if (model.atoms[c].pos - model.atoms[n].pos).norm() > LINK_DISTANCE_CUTOFF {
            return false;
        }
        let id = if b.name == "PRO" { "PTRANS" } else { "TRANS" };
        let link = LINKS.iter().find(|l| l.id == id).expect("link in library");
        let pick = |(side, name): (u8, &str)| {
            if side == 1 {
                a.atom(name)
            } else {
                b.atom(name)
            }
        };
        for d in link.bonds {
            if let (Some(i), Some(j)) = (pick(d.a), pick(d.b)) {
                self.push_bond(BondRestraint {
                    atoms: [i, j],
                    ideal: d.ideal,
                    esd: d.esd,
                    origin: Origin::Peptide,
                });
            }
        }
        for d in link.angles {
            if let [Some(i), Some(j), Some(k)] = d.atoms.map(pick) {
                self.push_angle(AngleRestraint {
                    atoms: [i, j, k],
                    ideal: d.ideal,
                    esd: d.esd,
                    origin: Origin::Peptide,
                });
            }
        }
        for plane in link.planes {
            self.add_plane(
                plane.iter().map(|&(side, n, e)| (pick((side, n)), e)),
                Origin::Peptide,
            );
        }
        true
    }

    /// Apply the CDL (or, for a cis-proline, the EH99 targets) to the central residue of a
    /// linked triple (`conformation_dependent_library.update_restraints`).
    fn apply_cdl(&mut self, model: &super::Model, prev: &Residue, cur: &Residue, next: &Residue) {
        let pos = |r: &Residue, n: &str| r.atom(n).map(|i| model.atoms[i].pos);
        let dihedral = |p: [Option<Vector3<f64>>; 4]| -> Option<f64> {
            let [a, b, c, d] = p;
            crate::structure::compute_dihedral(&a?, &b?, &c?, &d?).ok()
        };
        let omega = dihedral([
            pos(prev, "CA"),
            pos(prev, "C"),
            pos(cur, "N"),
            pos(cur, "CA"),
        ]);
        let cis = omega.is_some_and(|w| w.abs() < CDL_CIS_LIMIT);
        let targets: Vec<(CdlTerm, f64, f64)> = if cis {
            if cur.name != "PRO" {
                return;
            }
            cdl::CIS_PRO_EH99.to_vec()
        } else {
            let phi = dihedral([pos(prev, "C"), pos(cur, "N"), pos(cur, "CA"), pos(cur, "C")]);
            let psi = dihedral([pos(cur, "N"), pos(cur, "CA"), pos(cur, "C"), pos(next, "N")]);
            let (Some(phi), Some(psi)) = (phi, psi) else {
                return;
            };
            let row = cdl::lookup(CdlGroup::classify(&cur.name, &next.name), phi, psi);
            CdlTerm::TABLE
                .iter()
                .zip(row)
                .map(|(&t, (v, e))| (t, v, e))
                .collect()
        };
        for (term, ideal, esd) in targets {
            let atoms: Option<Vec<usize>> = term
                .atoms()
                .iter()
                .map(|&(offset, name)| match offset {
                    -1 => prev.atom(name),
                    0 => cur.atom(name),
                    _ => next.atom(name),
                })
                .collect();
            // `apply_updates` skips terms whose atoms (CB of glycine, a missing O) are absent.
            let Some(atoms) = atoms else { continue };
            match atoms[..] {
                [a, b] => {
                    if let Some(r) = self.find_bond(a, b) {
                        r.ideal = ideal;
                        r.esd = esd;
                    }
                }
                [a, b, c] => {
                    if let Some(r) = self.find_angle(a, b, c) {
                        r.ideal = ideal;
                        r.esd = esd;
                    }
                }
                _ => unreachable!("CDL terms are bonds or angles"),
            }
        }
    }

    /// Disulfide bridges: every pair of cysteine SG atoms closer than
    /// [`LINK_DISTANCE_CUTOFF`] (`pdb_interpretation.disulfide_distance_cutoff`).
    fn add_disulfides(&mut self, model: &super::Model) {
        let cys: Vec<&Residue> = model
            .residues
            .iter()
            .filter(|r| r.name == "CYS" && r.atom("SG").is_some())
            .collect();
        for (k, a) in cys.iter().enumerate() {
            for b in &cys[k + 1..] {
                let (sa, sb) = (a.atom("SG").unwrap(), b.atom("SG").unwrap());
                if (model.atoms[sa].pos - model.atoms[sb].pos).norm() > LINK_DISTANCE_CUTOFF {
                    continue;
                }
                self.push_bond(BondRestraint {
                    atoms: [sa, sb],
                    ideal: SS_BOND.0,
                    esd: SS_BOND.1,
                    origin: Origin::Disulfide,
                });
                for (x, y, z) in [(a.atom("CB"), sa, sb), (b.atom("CB"), sb, sa)] {
                    if let Some(cb) = x {
                        self.push_angle(AngleRestraint {
                            atoms: [cb, y, z],
                            ideal: SS_ANGLE.0,
                            esd: SS_ANGLE.1,
                            origin: Origin::Disulfide,
                        });
                    }
                }
            }
        }
    }
}

fn same_pair(a: &BondDef, b: &BondDef) -> bool {
    (a.a == b.a && a.b == b.b) || (a.a == b.b && a.b == b.a)
}

fn same_angle(a: &AngleDef, b: &AngleDef) -> bool {
    a.b == b.b && ((a.a == b.a && a.c == b.c) || (a.a == b.c && a.c == b.a))
}

/// Build every restraint of `model`: monomers, peptide links, disulfides, then the CDL.
pub(crate) fn build(model: &super::Model) -> Restraints {
    let mut r = Restraints::default();
    for res in &model.residues {
        if let Some(def) = library(&res.name) {
            r.add_residue(res, def);
        }
    }
    // Peptide links join consecutive standard residues of the same chain, in file order.
    let mut linked_to_next = vec![false; model.residues.len()];
    for k in 1..model.residues.len() {
        let (a, b) = (&model.residues[k - 1], &model.residues[k]);
        if a.id.chain != b.id.chain || library(&a.name).is_none() || library(&b.name).is_none() {
            continue;
        }
        linked_to_next[k - 1] = r.add_link(model, a, b);
    }
    r.add_disulfides(model);
    // The CDL needs a residue linked on both sides by a peptide bond (`are_linked` through the
    // restraints' bond table), with C, CA and N in the residue and in the next one
    // (`get_phi_psi_atoms`); the previous residue only has to supply its C.
    for k in 1..model.residues.len().saturating_sub(1) {
        if !(linked_to_next[k - 1] && linked_to_next[k]) {
            continue;
        }
        let (p, c, n) = (
            &model.residues[k - 1],
            &model.residues[k],
            &model.residues[k + 1],
        );
        if [c, n]
            .iter()
            .all(|r| r.atom("C").is_some() && r.atom("CA").is_some() && r.atom("N").is_some())
        {
            r.apply_cdl(model, p, c, n);
        }
    }
    r
}

/// Model value and signed deviation of each restraint, in cctbx's sign convention
/// (`delta = ideal − model`).
pub(crate) fn bond_value(model: &super::Model, r: &BondRestraint) -> (f64, f64) {
    let [a, b] = r.atoms;
    let d = (model.atoms[a].pos - model.atoms[b].pos).norm();
    (d, r.ideal - d)
}

pub(crate) fn angle_value(model: &super::Model, r: &AngleRestraint) -> (f64, f64) {
    let [a, b, c] = r.atoms.map(|i| model.atoms[i].pos);
    let (u, v) = (a - b, c - b);
    let cos = (u.dot(&v) / (u.norm() * v.norm())).clamp(-1.0, 1.0);
    let theta = cos.acos().to_degrees();
    (theta, r.ideal - theta)
}

/// Signed chiral volume of centre `p[0]` and neighbours `p[1..]` (cctbx `chirality`).
pub(crate) fn chiral_volume(p: [Vector3<f64>; 4]) -> f64 {
    let (d1, d2, d3) = (p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    d1.dot(&d2.cross(&d3))
}

pub(crate) fn chirality_value(model: &super::Model, r: &ChiralRestraint) -> (f64, f64) {
    let v = chiral_volume(r.atoms.map(|i| model.atoms[i].pos));
    let delta = if r.both_signs {
        r.ideal.abs() - v.abs()
    } else {
        r.ideal - v
    };
    (v, delta)
}

/// Per-atom distances from the weighted least-squares plane (cctbx `planarity`: weights
/// 1/esd², plane through the weighted centroid, normal along the smallest eigenvector).
pub(crate) fn plane_deltas(model: &super::Model, r: &PlaneRestraint) -> Vec<f64> {
    let pts: Vec<Vector3<f64>> = r.atoms.iter().map(|&i| model.atoms[i].pos).collect();
    let w: Vec<f64> = r.esds.iter().map(|e| 1.0 / (e * e)).collect();
    let wsum: f64 = w.iter().sum();
    let centroid = pts
        .iter()
        .zip(&w)
        .fold(Vector3::zeros(), |acc, (p, w)| acc + p * *w)
        / wsum;
    let scatter = pts
        .iter()
        .zip(&w)
        .fold(Matrix3::zeros(), |acc: Matrix3<f64>, (p, w)| {
            let d = p - centroid;
            acc + d * d.transpose() * *w
        });
    let eig = SymmetricEigen::new(scatter);
    let k = eig.eigenvalues.imin();
    let normal = eig.eigenvectors.column(k).into_owned();
    pts.iter().map(|p| (p - centroid).dot(&normal)).collect()
}
