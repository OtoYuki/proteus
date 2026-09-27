//! Shape complementarity (Sc) of two molecular surfaces, Lawrence & Colman (1993), J. Mol. Biol.
//! 234:946–950.
//!
//! Each molecule's Connolly surface (convex contact, toroidal re-entrant and concave probe dots,
//! Connolly 1983) is sampled at ~15 dots/Å² with a 1.7 Å probe. Dots buried by the other molecule
//! form the interface, and a 1.5 Å band at its edge is trimmed away. Each remaining dot is paired
//! with the nearest buried dot on the other surface and scored S = −n₁·n₂·exp(−w·d²), with
//! w = 0.5 Å⁻². Sc is the mean of the two surfaces' median S. It is 1 for a perfect fit and
//! around 0.6–0.75 for real protein–protein interfaces.
//!
//! Ported from `sc-rs` (<https://github.com/cytokineking/sc-rs>, MIT, © 2025, commit `befcc6c`),
//! which re-implements Rosetta's `ShapeComplementarityCalculator` and matches its values closely.
//! The notice is in `data/sc/NOTICE`. What changed in the port:
//! - Atoms are addressed by index instead of raw pointers, so the module has no `unsafe`.
//! - Only the single-threaded path is kept. Proteus runs structures in parallel instead.
//! - The radius table is compiled in. There is no environment override.
//! - The attention state that nothing ever assigned (`Consider`) is gone. The branches that
//!   tested it could never be taken.
//!
//! The algorithm, its constants and the order of operations are unchanged. The order matters:
//! which atoms are marked accessible, and so which dots exist, depends on it.

use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};

/// Probe radius, Å (Rosetta's default for Sc).
const PROBE_RADIUS: f64 = 1.7;
/// Dots per Å² (Lawrence & Colman: ~15 is enough; doubling it barely changes Sc).
const DOT_DENSITY: f64 = 15.0;
/// Width of the interface-edge band that is trimmed, Å.
const PERIPHERAL_BAND: f64 = 1.5;
/// An atom farther than this from the other molecule takes no part, Å.
const SEPARATION_CUTOFF: f64 = 8.0;
/// Gaussian weight w, Å⁻².
const GAUSSIAN_W: f64 = 0.5;

/// Per residue and atom radii, Å, in match order. `*` at the start of a pattern matches
/// anything, and `*` later in it matches any suffix. The last block is the fallback by element.
/// Generated from sc-rs `atomic_radii.json` (its order is the match order; do not hand-edit).
const RADII: &[(&str, &str, f64)] = &[
    ("ALA", "CB", 1.95),
    ("ARG", "NH*", 1.7),
    ("ARG", "CZ", 1.8),
    ("ARG", "NE", 1.65),
    ("ARG", "CD", 1.9),
    ("ARG", "CG", 1.9),
    ("ASN", "ND2", 1.7),
    ("ASN", "OD1", 1.6),
    ("ASN", "CG", 1.8),
    ("ASP", "OD*", 1.6),
    ("ASP", "CG", 1.8),
    ("GLN", "NE2", 1.7),
    ("GLN", "OE1", 1.6),
    ("GLN", "CD", 1.8),
    ("GLN", "CG", 1.9),
    ("GLU", "OE*", 1.6),
    ("GLU", "CD", 1.8),
    ("GLU", "CG", 1.9),
    ("GLY", "CA", 1.9),
    ("HIS", "CD2", 1.9),
    ("HIS", "NE2", 1.65),
    ("HIS", "CE1", 1.9),
    ("HIS", "ND1", 1.65),
    ("HIS", "CG", 1.8),
    ("HOH", "O**", 1.7),
    ("ILE", "CD1", 1.95),
    ("ILE", "CG1", 1.9),
    ("ILE", "CB", 1.85),
    ("ILE", "CG2", 1.95),
    ("LEU", "CD*", 1.95),
    ("LEU", "CG", 1.85),
    ("LYS", "NZ", 1.75),
    ("LYS", "CE", 1.9),
    ("LYS", "CD", 1.9),
    ("LYS", "CG", 1.9),
    ("MET", "CE", 1.95),
    ("MET", "CG", 1.9),
    ("PHE", "CD*", 1.9),
    ("PHE", "CE*", 1.9),
    ("PHE", "CZ", 1.9),
    ("PHE", "CG", 1.8),
    ("PRO", "CD", 1.9),
    ("PRO", "CG", 1.9),
    ("SER", "OG", 1.7),
    ("SUL", "S", 1.9),
    ("SUL", "O***", 1.65),
    ("THR", "CG2", 1.95),
    ("THR", "OG1", 1.7),
    ("THR", "CB", 1.85),
    ("TRP", "CE2", 1.8),
    ("TRP", "CE3", 1.9),
    ("TRP", "CD1", 1.9),
    ("TRP", "CD2", 1.8),
    ("TRP", "CZ*", 1.9),
    ("TRP", "CH2", 1.9),
    ("TRP", "NE1", 1.65),
    ("TRP", "CG", 1.8),
    ("TYR", "OH", 1.7),
    ("TYR", "CD*", 1.9),
    ("TYR", "CE*", 1.9),
    ("TYR", "CZ", 1.8),
    ("TYR", "CG", 1.8),
    ("VAL", "CG*", 1.95),
    ("VAL", "CB", 1.85),
    ("WAT", "O", 1.7),
    ("WAT", "O*", 1.7),
    ("***", "H", 0.5),
    ("***", "H*", 0.5),
    ("***", "H**", 0.5),
    ("***", "H***", 0.5),
    ("***", "CA", 1.85),
    ("***", "C", 1.8),
    ("***", "O", 1.6),
    ("***", "N", 1.65),
    ("***", "CB", 1.9),
    ("***", "OT*", 1.6),
    ("***", "OXT", 1.6),
    ("***", "S*", 1.9),
    ("***", "P", 1.8),
];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    fn norm2(self) -> f64 {
        self.dot(self)
    }
    fn normalized(self) -> Vec3 {
        let m = self.norm2().max(0.0).sqrt();
        if m > 0.0 {
            self / m
        } else {
            self
        }
    }
    fn dist2(self, o: Vec3) -> f64 {
        (self - o).norm2()
    }
    fn dist(self, o: Vec3) -> f64 {
        self.dist2(o).sqrt()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, r: Vec3) -> Vec3 {
        Vec3::new(self.x + r.x, self.y + r.y, self.z + r.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, r: Vec3) -> Vec3 {
        Vec3::new(self.x - r.x, self.y - r.y, self.z - r.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, r: f64) -> Vec3 {
        Vec3::new(self.x * r, self.y * r, self.z * r)
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, r: f64) -> Vec3 {
        Vec3::new(self.x / r, self.y / r, self.z / r)
    }
}

/// Why Sc could not be computed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ScError {
    #[error("one side of the interface has no atoms")]
    NoAtoms,
    #[error("no radius for atom {atom} of {residue}")]
    NoRadius { residue: String, atom: String },
    #[error("two atoms of one molecule coincide ({0})")]
    Coincident(String),
    #[error("degenerate surface construction at atom {0}")]
    Degenerate(usize),
    #[error("surface sampling did not converge")]
    TooManySubdivisions,
    #[error("the two molecules have no buried surface in common")]
    NoInterface,
}

/// An atom handed to [`shape_complementarity`]: residue name, atom name and position (Å).
#[derive(Debug, Clone)]
pub struct ScAtom {
    pub residue: String,
    pub name: String,
    pub coord: [f64; 3],
}

/// Sc and its by-products.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeComplementarity {
    /// Sc, the mean of the two surfaces' median scores.
    pub sc: f64,
    /// Mean of the two surfaces' median dot-to-dot separation, Å.
    pub median_distance: f64,
    /// Interface area after the edge band is trimmed, both surfaces summed, Å².
    pub trimmed_area: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Attention {
    Far,
    Buried,
}

struct Atom {
    molecule: usize,
    radius: f64,
    density: f64,
    attention: Attention,
    accessible: bool,
    coor: Vec3,
    neighbors: Vec<usize>,
}

struct Probe {
    atoms: [usize; 3],
    height: f64,
    point: Vec3,
    alt: Vec3,
}

struct Dot {
    coor: Vec3,
    outnml: Vec3,
    area: f64,
    buried: bool,
}

struct Surface {
    atoms: Vec<Atom>,
    probes: Vec<Probe>,
    dots: [Vec<Dot>; 2],
    radmax: f64,
    /// Atoms of each molecule, bucketed by `radmax + PROBE_RADIUS`, for the burial test.
    by_molecule: [Grid; 2],
}

/// Points bucketed in cubic cells, for "everything within one cell size" queries. Candidates
/// come back in ascending index order, so a loop over them visits points in the same order as
/// a scan of the whole list would: results, including which of two equidistant points wins,
/// are identical to the all-pairs version, only faster.
#[derive(Default)]
struct Grid {
    cell: f64,
    cells: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
}

impl Grid {
    fn new(cell: f64, points: impl IntoIterator<Item = (usize, Vec3)>) -> Self {
        let mut g = Grid {
            cell,
            cells: Default::default(),
        };
        for (i, p) in points {
            let k = g.key(p);
            g.cells.entry(k).or_default().push(i);
        }
        g
    }
    fn key(&self, p: Vec3) -> (i64, i64, i64) {
        (
            (p.x / self.cell).floor() as i64,
            (p.y / self.cell).floor() as i64,
            (p.z / self.cell).floor() as i64,
        )
    }
    /// Indices in the cube of cells `ring` cells around `p`'s cell, ascending. Every point
    /// within `ring × cell` of `p` is among them.
    fn around(&self, p: Vec3, ring: i64, out: &mut Vec<usize>) {
        out.clear();
        let (x, y, z) = self.key(p);
        for dx in -ring..=ring {
            for dy in -ring..=ring {
                for dz in -ring..=ring {
                    if let Some(v) = self.cells.get(&(x + dx, y + dy, z + dz)) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
        out.sort_unstable();
    }
}

fn wildcard_match(query: &str, pattern: &str) -> bool {
    if pattern.starts_with('*') {
        return true;
    }
    match pattern.find('*') {
        Some(star) => query.len() >= star && query[..star] == pattern[..star],
        None => query == pattern,
    }
}

fn radius_of(residue: &str, atom: &str) -> Option<f64> {
    RADII
        .iter()
        .find(|(r, a, _)| wildcard_match(residue, r) && wildcard_match(atom, a))
        .map(|&(_, _, radius)| radius)
        .or_else(|| {
            let element = atom.chars().find(|c| c.is_ascii_alphabetic())?;
            let element = element.to_ascii_uppercase().to_string();
            RADII
                .iter()
                .find(|(r, a, _)| r.starts_with("***") && *a == element)
                .map(|&(_, _, radius)| radius)
        })
}

/// Sc between molecule `a` and molecule `b` (heavy atoms; hydrogens are accepted and get Rosetta's
/// 0.5 Å radius).
pub fn shape_complementarity(a: &[ScAtom], b: &[ScAtom]) -> Result<ShapeComplementarity, ScError> {
    if a.is_empty() || b.is_empty() {
        return Err(ScError::NoAtoms);
    }
    let mut atoms = Vec::with_capacity(a.len() + b.len());
    for (molecule, side) in [a, b].into_iter().enumerate() {
        for at in side {
            let radius = radius_of(at.residue.trim(), at.name.trim())
                .filter(|r| *r > 0.0)
                .ok_or_else(|| ScError::NoRadius {
                    residue: at.residue.clone(),
                    atom: at.name.clone(),
                })?;
            atoms.push(Atom {
                molecule,
                radius,
                density: DOT_DENSITY,
                attention: Attention::Buried,
                accessible: false,
                coor: Vec3::new(at.coord[0], at.coord[1], at.coord[2]),
                neighbors: Vec::new(),
            });
        }
    }
    let mut s = Surface {
        atoms,
        probes: Vec::new(),
        dots: [Vec::new(), Vec::new()],
        radmax: 0.0,
        by_molecule: Default::default(),
    };
    s.assign_attention();
    s.generate_molecular_surfaces()?;
    if s.dots[0].is_empty() || s.dots[1].is_empty() {
        return Err(ScError::NoInterface);
    }
    let trimmed = [s.trim_peripheral_band(0), s.trim_peripheral_band(1)];
    let area: f64 = trimmed
        .iter()
        .enumerate()
        .map(|(m, idx)| idx.iter().map(|&i| s.dots[m][i].area).sum::<f64>())
        .sum();
    let (d0, s0) = s
        .neighbor_scores(&trimmed, 0, 1)
        .ok_or(ScError::NoInterface)?;
    let (d1, s1) = s
        .neighbor_scores(&trimmed, 1, 0)
        .ok_or(ScError::NoInterface)?;
    Ok(ShapeComplementarity {
        sc: (s0 + s1) / 2.0,
        median_distance: (d0 + d1) / 2.0,
        trimmed_area: area,
    })
}

/// The element at index `len / 2` after sorting, as `select_nth_unstable` picks it.
fn median(v: &mut [f64]) -> f64 {
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

impl Surface {
    fn assign_attention(&mut self) {
        let sep2 = SEPARATION_CUTOFF * SEPARATION_CUTOFF;
        let grid = Grid::new(
            SEPARATION_CUTOFF,
            self.atoms.iter().enumerate().map(|(i, a)| (i, a.coor)),
        );
        let mut near = Vec::new();
        for i in 0..self.atoms.len() {
            let (mol, c) = (self.atoms[i].molecule, self.atoms[i].coor);
            grid.around(c, 1, &mut near);
            let close = near
                .iter()
                .any(|&j| self.atoms[j].molecule != mol && c.dist2(self.atoms[j].coor) < sep2);
            self.atoms[i].attention = if close {
                Attention::Buried
            } else {
                Attention::Far
            };
        }
    }

    fn generate_molecular_surfaces(&mut self) -> Result<(), ScError> {
        self.radmax = self.atoms.iter().map(|a| a.radius).fold(0.0, f64::max);
        for m in 0..2 {
            self.by_molecule[m] = Grid::new(
                self.radmax + PROBE_RADIUS,
                self.atoms
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.molecule == m)
                    .map(|(i, a)| (i, a.coor)),
            );
        }
        let all = Grid::new(
            2.0 * (self.radmax + PROBE_RADIUS),
            self.atoms.iter().enumerate().map(|(i, a)| (i, a.coor)),
        );
        for i in 0..self.atoms.len() {
            if self.atoms[i].attention == Attention::Far {
                continue;
            }
            self.find_neighbors(i, &all)?;
            self.build_probes(i)?;
            if self.atoms[i].accessible {
                self.emit_contact_surface(i)?;
            }
        }
        self.generate_concave_surface()
    }

    /// Same-molecule atoms whose expanded spheres can share a probe, nearest first. `all` holds
    /// every atom at twice the largest expanded radius, so no candidate is missed.
    fn find_neighbors(&mut self, i: usize, all: &Grid) -> Result<(), ScError> {
        let a1 = &self.atoms[i];
        let mut candidates = Vec::new();
        all.around(a1.coor, 1, &mut candidates);
        let mut neighbors = Vec::new();
        for &j in &candidates {
            let a2 = &self.atoms[j];
            if j == i || a1.molecule != a2.molecule {
                continue;
            }
            let d2 = a1.coor.dist2(a2.coor);
            if d2 <= 0.0001 {
                return Err(ScError::Coincident(format!(
                    "atoms {} and {}",
                    i + 1,
                    j + 1
                )));
            }
            let bridge = a1.radius + a2.radius + 2.0 * PROBE_RADIUS;
            if d2 < bridge * bridge {
                neighbors.push(j);
            }
        }
        let center = a1.coor;
        let atoms = &self.atoms;
        // Unstable, with sc-rs's comparator: equidistant neighbours end up in the same order.
        neighbors.sort_unstable_by(|&x, &y| {
            let (d1, d2) = (atoms[x].coor.dist2(center), atoms[y].coor.dist2(center));
            if d1 < d2 {
                std::cmp::Ordering::Less
            } else if d1 > d2 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let a1 = &mut self.atoms[i];
        a1.accessible |= neighbors.is_empty();
        a1.neighbors = neighbors;
        Ok(())
    }

    fn build_probes(&mut self, i: usize) -> Result<(), ScError> {
        let neighbors = self.atoms[i].neighbors.clone();
        if neighbors.is_empty() {
            return Ok(());
        }
        let eri = self.atoms[i].radius + PROBE_RADIUS;
        for &j in &neighbors {
            if j <= i {
                continue;
            }
            let (ci, cj) = (self.atoms[i].coor, self.atoms[j].coor);
            let erj = self.atoms[j].radius + PROBE_RADIUS;
            let dij = ci.dist(cj);
            let uij = (cj - ci) / dij;
            let asym = (eri * eri - erj * erj) / dij;
            let tij = (ci + cj) * 0.5 + uij * (asym * 0.5);
            let far = (eri + erj) * (eri + erj) - dij * dij;
            if far <= 0.0 {
                continue;
            }
            let contain = dij * dij - (self.atoms[i].radius - self.atoms[j].radius).powi(2);
            if contain <= 0.0 {
                continue;
            }
            let rij = 0.5 * far.sqrt() * contain.sqrt() / dij;
            if neighbors.len() <= 1 {
                self.atoms[i].accessible = true;
                self.atoms[j].accessible = true;
                break;
            }
            self.build_probe_triplets(i, j, uij, tij, rij);
            let cusp = asym.abs() < dij;
            if self.atoms[i].attention != Attention::Far
                || self.atoms[j].attention != Attention::Far
            {
                self.emit_reentrant_surface(i, j, uij, tij, rij, cusp)?;
            }
        }
        Ok(())
    }

    fn build_probe_triplets(&mut self, i: usize, j: usize, uij: Vec3, tij: Vec3, rij: f64) {
        let neighbors = self.atoms[i].neighbors.clone();
        let (ci, cj) = (self.atoms[i].coor, self.atoms[j].coor);
        let eri = self.atoms[i].radius + PROBE_RADIUS;
        let erj = self.atoms[j].radius + PROBE_RADIUS;
        let mut made = false;
        for &k in &neighbors {
            if k <= j {
                continue;
            }
            let ck = self.atoms[k].coor;
            let erk = self.atoms[k].radius + PROBE_RADIUS;
            if cj.dist(ck) >= erj + erk {
                continue;
            }
            let dik = ci.dist(ck);
            if dik >= eri + erk {
                continue;
            }
            if [i, j, k]
                .iter()
                .all(|&x| self.atoms[x].attention == Attention::Far)
            {
                continue;
            }
            let uik = (ck - ci) / dik;
            let wedge = uij.dot(uik).acos();
            let swedge = wedge.sin();
            if swedge <= 0.0 {
                if tij.dist(ck) < erk * erk - rij * rij {
                    return;
                }
                continue;
            }
            let uijk = uij.cross(uik) / swedge;
            let utb = uijk.cross(uij);
            let asym_ik = (eri * eri - erk * erk) / dik;
            let tik = (ci + ck) * 0.5 + uik * (asym_ik * 0.5);
            let d = tik - tij;
            let sum = uik.x * d.x + uik.y * d.y + uik.z * d.z;
            let bijk = tij + utb * (sum / swedge);
            let h2 = eri * eri - bijk.dist2(ci);
            if h2 <= 0.0 {
                continue;
            }
            let height = h2.sqrt();
            for sign in [1.0, -1.0] {
                let pijk = bijk + uijk * (height * sign);
                if self.probe_collides(pijk, j, k, &neighbors) {
                    continue;
                }
                let atoms = if sign > 0.0 { [i, j, k] } else { [j, i, k] };
                self.probes.push(Probe {
                    atoms,
                    height,
                    point: pijk,
                    alt: uijk * sign,
                });
                made = true;
            }
        }
        if made {
            self.atoms[i].accessible = true;
        }
    }

    /// Whether a probe at `p` overlaps any neighbour of the first atom other than `j` and `k`.
    fn probe_collides(&self, p: Vec3, j: usize, k: usize, neighbors: &[usize]) -> bool {
        neighbors.iter().any(|&n| {
            n != j
                && n != k
                && p.dist2(self.atoms[n].coor) <= (self.atoms[n].radius + PROBE_RADIUS).powi(2)
        })
    }

    fn emit_reentrant_surface(
        &mut self,
        i: usize,
        j: usize,
        uij: Vec3,
        tij: Vec3,
        rij: f64,
        point_cusp: bool,
    ) -> Result<(), ScError> {
        let neighbors = self.atoms[i].neighbors.clone();
        let density = (self.atoms[i].density + self.atoms[j].density) / 2.0;
        let eri = self.atoms[i].radius + PROBE_RADIUS;
        let erj = self.atoms[j].radius + PROBE_RADIUS;
        let rci = rij * self.atoms[i].radius / eri;
        let rcj = rij * self.atoms[j].radius / erj;
        let rb = (rij - PROBE_RADIUS).max(0.0);
        let rs = (rci + 2.0 * rb + rcj) / 4.0;
        let e = rs / rij;
        let mut subs = Vec::new();
        let ts = sample_circle(tij, rij, uij, e * e * density, &mut subs)?;
        for sub in subs {
            let too_close = neighbors.iter().any(|&n| {
                n != j
                    && sub.dist2(self.atoms[n].coor) < (self.atoms[n].radius + PROBE_RADIUS).powi(2)
            });
            if too_close {
                continue;
            }
            self.atoms[i].accessible = true;
            self.atoms[j].accessible = true;
            let vpi = (self.atoms[i].coor - sub) / eri;
            let vpj = (self.atoms[j].coor - sub) / erj;
            let axis = vpi.cross(vpj).normalized();
            let cusp = PROBE_RADIUS * PROBE_RADIUS - rij * rij;
            let (arc_i, arc_j) = if cusp > 0.0 && point_cusp {
                let qij = tij - uij * cusp.sqrt();
                ((qij - sub) / PROBE_RADIUS, Vec3::ZERO)
            } else {
                let pq = (vpi + vpj).normalized();
                (pq, pq)
            };
            let dt = arc_i.dot(vpi);
            if dt >= 1.0 || dt <= -1.0 {
                return Ok(());
            }
            let dt = arc_j.dot(vpj);
            if dt >= 1.0 || dt <= -1.0 {
                return Ok(());
            }
            for (atom, from, to) in [(i, vpi, arc_i), (j, arc_j, vpj)] {
                if self.atoms[atom].attention == Attention::Far {
                    continue;
                }
                let mut points = Vec::new();
                let ps = sample_arc(sub, PROBE_RADIUS, axis, density, from, to, &mut points)?;
                for p in points {
                    let area = ps * ts * distance_to_line(tij, uij, p) / rij;
                    self.add_dot(self.atoms[atom].molecule, p, area, sub, atom);
                }
            }
        }
        Ok(())
    }

    fn emit_contact_surface(&mut self, i: usize) -> Result<(), ScError> {
        let neighbors = self.atoms[i].neighbors.clone();
        let ci = self.atoms[i].coor;
        let ri = self.atoms[i].radius;
        let eri = ri + PROBE_RADIUS;
        let mut north = Vec3::new(0.0, 0.0, 1.0);
        let mut south = Vec3::new(0.0, 0.0, -1.0);
        let mut eqv = Vec3::new(1.0, 0.0, 0.0);
        if let Some(&n0) = neighbors.first() {
            let (cn, rn) = (self.atoms[n0].coor, self.atoms[n0].radius);
            north = (ci - cn).normalized();
            let mut vtemp = Vec3::new(
                north.y * north.y + north.z * north.z,
                north.x * north.x + north.z * north.z,
                north.x * north.x + north.y * north.y,
            )
            .normalized();
            if vtemp.dot(north).abs() > 0.99 {
                vtemp = Vec3::new(1.0, 0.0, 0.0);
            }
            eqv = north.cross(vtemp).normalized();
            let erj = rn + PROBE_RADIUS;
            let dij = ci.dist(cn);
            let uij = (cn - ci) / dij;
            let asym = (eri * eri - erj * erj) / dij;
            let tij = (ci + cn) * 0.5 + uij * (asym * 0.5);
            let far = (eri + erj) * (eri + erj) - dij * dij;
            let contain = dij * dij - (ri - rn).powi(2);
            if far <= 0.0 || contain <= 0.0 {
                return Err(ScError::Degenerate(i + 1));
            }
            let rij = 0.5 * far.sqrt() * contain.sqrt() / dij;
            let pij = tij + eqv.cross(north) * rij;
            south = (pij - ci) / eri;
            if north.cross(south).dot(eqv) <= 0.0 {
                return Err(ScError::Degenerate(i + 1));
            }
        }
        let density = self.atoms[i].density;
        let mut lats = Vec::new();
        let cs = sample_arc(Vec3::ZERO, ri, eqv, density, north, south, &mut lats)?;
        let mut points = Vec::new();
        for lat in lats {
            let dt = lat.dot(north);
            let cen = ci + north * dt;
            let rad2 = ri * ri - dt * dt;
            if rad2 <= 0.0 {
                continue;
            }
            let ps = sample_circle(cen, rad2.sqrt(), north, density, &mut points)?;
            for &p in &points {
                let pcen = ci + (p - ci) * (eri / ri);
                // Rosetta skips the first (nearest) neighbour here; so does sc-rs.
                let collides = neighbors
                    .iter()
                    .skip(1)
                    .any(|&n| pcen.dist(self.atoms[n].coor) <= self.atoms[n].radius + PROBE_RADIUS);
                if !collides {
                    self.add_dot(self.atoms[i].molecule, p, ps * cs, pcen, i);
                }
            }
        }
        Ok(())
    }

    fn generate_concave_surface(&mut self) -> Result<(), ScError> {
        let low: Vec<usize> = (0..self.probes.len())
            .filter(|&p| self.probes[p].height < PROBE_RADIUS)
            .collect();
        for pi in 0..self.probes.len() {
            let Probe {
                atoms: ax,
                height: h,
                point: pijk,
                alt: uijk,
            } = self.probes[pi];
            let density = ax.iter().map(|&a| self.atoms[a].density).sum::<f64>() / 3.0;
            let nears: Vec<usize> = low
                .iter()
                .copied()
                .filter(|&lp| {
                    lp != pi
                        && pijk.dist2(self.probes[lp].point) <= 4.0 * PROBE_RADIUS * PROBE_RADIUS
                })
                .collect();
            let vp = ax.map(|a| (self.atoms[a].coor - pijk).normalized());
            let edges = [
                vp[0].cross(vp[1]).normalized(),
                vp[1].cross(vp[2]).normalized(),
                vp[2].cross(vp[0]).normalized(),
            ];
            let (mut dm, mut mm) = (-1.0, 0);
            for (k, v) in vp.iter().enumerate() {
                let dt = uijk.dot(*v);
                if dt > dm {
                    dm = dt;
                    mm = k;
                }
            }
            let south = uijk * -1.0;
            let arc_axis = vp[mm].cross(south).normalized();
            let mut lats = Vec::new();
            let cs = sample_arc(
                Vec3::ZERO,
                PROBE_RADIUS,
                arc_axis,
                density,
                vp[mm],
                south,
                &mut lats,
            )?;
            let mut points = Vec::new();
            for lat in lats {
                let dt = lat.dot(south);
                let cen = south * dt;
                let rad2 = PROBE_RADIUS * PROBE_RADIUS - dt * dt;
                if rad2 <= 0.0 {
                    continue;
                }
                let ps = sample_circle(cen, rad2.sqrt(), south, density, &mut points)?;
                for &p in &points {
                    if edges.iter().any(|e| p.dot(*e) >= 0.0) {
                        continue;
                    }
                    let p = p + pijk;
                    if h < PROBE_RADIUS
                        && nears
                            .iter()
                            .any(|&np| p.dist2(self.probes[np].point) < PROBE_RADIUS * PROBE_RADIUS)
                    {
                        continue;
                    }
                    let (mut dmin, mut mc) = (2.0 * PROBE_RADIUS, 0);
                    for (kk, &a) in ax.iter().enumerate() {
                        let d = p.dist(self.atoms[a].coor) - self.atoms[a].radius;
                        if d < dmin {
                            dmin = d;
                            mc = kk;
                        }
                    }
                    let atom = ax[mc];
                    self.add_dot(self.atoms[atom].molecule, p, ps * cs, pijk, atom);
                }
            }
        }
        Ok(())
    }

    fn add_dot(&mut self, molecule: usize, coor: Vec3, area: f64, pcen: Vec3, _atom: usize) {
        let outnml = (pcen - coor) / PROBE_RADIUS;
        let mut near = Vec::new();
        self.by_molecule[1 - molecule].around(pcen, 1, &mut near);
        let buried = near.iter().any(|&j| {
            let b = &self.atoms[j];
            pcen.dist2(b.coor) <= (b.radius + PROBE_RADIUS).powi(2)
        });
        self.dots[molecule].push(Dot {
            coor,
            outnml,
            area,
            buried,
        });
    }

    /// Buried dots of `m` that are not within the peripheral band of an exposed dot.
    fn trim_peripheral_band(&self, m: usize) -> Vec<usize> {
        let r2 = PERIPHERAL_BAND * PERIPHERAL_BAND;
        let dots = &self.dots[m];
        let exposed = Grid::new(
            PERIPHERAL_BAND,
            dots.iter()
                .enumerate()
                .filter(|(_, d)| !d.buried)
                .map(|(i, d)| (i, d.coor)),
        );
        let mut near = Vec::new();
        (0..dots.len())
            .filter(|&i| {
                if !dots[i].buried {
                    return false;
                }
                exposed.around(dots[i].coor, 1, &mut near);
                !near.iter().any(|&k| dots[i].coor.dist2(dots[k].coor) <= r2)
            })
            .collect()
    }

    /// Median distance and median score of `my` trimmed dots against `their` trimmed dots.
    fn neighbor_scores(
        &self,
        trimmed: &[Vec<usize>; 2],
        my: usize,
        their: usize,
    ) -> Option<(f64, f64)> {
        if trimmed[my].is_empty() || trimmed[their].is_empty() {
            return None;
        }
        let mut dists = Vec::with_capacity(trimmed[my].len());
        let mut scores = Vec::with_capacity(trimmed[my].len());
        // Positions in trimmed[their], bucketed; searched in growing cubes until the nearest
        // dot found is closer than anything outside the cube can be.
        const CELL: f64 = 2.0;
        let grid = Grid::new(
            CELL,
            trimmed[their]
                .iter()
                .enumerate()
                .map(|(pos, &d)| (pos, self.dots[their][d].coor)),
        );
        let mut near = Vec::new();
        for &pd in &trimmed[my] {
            let d1 = &self.dots[my][pd];
            let mut ring = 1;
            let (best2, nearest) = loop {
                grid.around(d1.coor, ring, &mut near);
                let mut best2 = 9.0e20_f64;
                let mut nearest: Option<&Dot> = None;
                // Ascending positions with `<=`: the last of equidistant dots wins, as in a scan.
                for &pos in &near {
                    let d2 = &self.dots[their][trimmed[their][pos]];
                    let x = d2.coor.dist2(d1.coor);
                    if x <= best2 {
                        best2 = x;
                        nearest = Some(d2);
                    }
                }
                let reach = ring as f64 * CELL;
                if (nearest.is_some() && best2 < reach * reach)
                    || near.len() == trimmed[their].len()
                {
                    break (best2, nearest);
                }
                ring += 1;
            };
            if let Some(n) = nearest {
                let dmin = best2.sqrt();
                let r = (d1.outnml.dot(n.outnml) * (-(dmin * dmin) * GAUSSIAN_W).exp())
                    .clamp(-0.999, 0.999);
                dists.push(dmin);
                scores.push(-r);
            }
        }
        if dists.is_empty() {
            return None;
        }
        Some((median(&mut dists), median(&mut scores)))
    }
}

fn distance_to_line(cen: Vec3, axis: Vec3, p: Vec3) -> f64 {
    let v = p - cen;
    let dt = v.dot(axis);
    (v.norm2() - dt * dt).max(0.0).sqrt()
}

fn sample_arc(
    cen: Vec3,
    rad: f64,
    axis: Vec3,
    density: f64,
    x: Vec3,
    v: Vec3,
    points: &mut Vec<Vec3>,
) -> Result<f64, ScError> {
    let y = axis.cross(x);
    let mut angle = v.dot(y).atan2(v.dot(x));
    if angle < 0.0 {
        angle += 2.0 * PI;
    }
    sample_arc_segment(cen, rad, x, y, angle, density, points)
}

fn sample_circle(
    cen: Vec3,
    rad: f64,
    axis: Vec3,
    density: f64,
    points: &mut Vec<Vec3>,
) -> Result<f64, ScError> {
    let mut v1 = Vec3::new(
        axis.y * axis.y + axis.z * axis.z,
        axis.x * axis.x + axis.z * axis.z,
        axis.x * axis.x + axis.y * axis.y,
    )
    .normalized();
    if v1.dot(axis).abs() > 0.99 {
        v1 = Vec3::new(1.0, 0.0, 0.0);
    }
    let v2 = axis.cross(v1).normalized();
    let x = axis.cross(v2).normalized();
    let y = axis.cross(x);
    sample_arc_segment(cen, rad, x, y, 2.0 * PI, density, points)
}

fn sample_arc_segment(
    cen: Vec3,
    rad: f64,
    x: Vec3,
    y: Vec3,
    angle: f64,
    density: f64,
    points: &mut Vec<Vec3>,
) -> Result<f64, ScError> {
    points.clear();
    if rad <= 0.0 {
        return Ok(0.0);
    }
    let delta = 1.0 / (density.sqrt() * rad);
    let mut a = -delta / 2.0;
    for _ in 0..100_000 {
        a += delta;
        if a > angle {
            break;
        }
        points.push(cen + x * (rad * a.cos()) + y * (rad * a.sin()));
    }
    if a + delta < angle {
        return Err(ScError::TooManySubdivisions);
    }
    Ok(if points.is_empty() {
        0.0
    } else {
        rad * angle / points.len() as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radii_follow_the_table_then_the_element() {
        assert_eq!(radius_of("ARG", "NH1"), Some(1.7));
        assert_eq!(radius_of("PHE", "CZ"), Some(1.9));
        assert_eq!(radius_of("PHE", "CG"), Some(1.8));
        assert_eq!(radius_of("ALA", "CA"), Some(1.85));
        assert_eq!(radius_of("ALA", "C"), Some(1.8));
        assert_eq!(radius_of("ALA", "OXT"), Some(1.6));
        // An unknown residue falls back to the element: CG is a carbon.
        assert_eq!(radius_of("XYZ", "CG"), Some(1.8));
    }

    #[test]
    fn two_distant_molecules_have_no_interface() {
        let atom = |x: f64| ScAtom {
            residue: "ALA".into(),
            name: "CA".into(),
            coord: [x, 0.0, 0.0],
        };
        assert_eq!(
            shape_complementarity(&[atom(0.0)], &[atom(50.0)]),
            Err(ScError::NoInterface)
        );
        assert_eq!(
            shape_complementarity(&[], &[atom(0.0)]),
            Err(ScError::NoAtoms)
        );
    }
}
