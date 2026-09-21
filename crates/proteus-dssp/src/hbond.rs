//! Backbone hydrogen placement and Kabsch–Sander H-bond energies.

use crate::assign::Residue;

/// −0.42 · 0.20 · 332 kcal·Å/mol (Kabsch & Sander 1983).
pub const COUPLING: f64 = -27.888;
/// Energies are clamped here (DSSP `kMinHBondEnergy`).
pub const MIN_ENERGY: f64 = -9.9;
/// A bond exists when E < −0.5 kcal/mol (DSSP `kMaxHBondEnergy`).
pub const MAX_BOND_ENERGY: f64 = -0.5;
/// Pairs with Cα–Cα distance at or above this are never bonded (DSSP `kMinimalCADistance`).
pub const MIN_CA_DISTANCE: f64 = 9.0;
/// Below this inter-atomic distance the energy saturates at [`MIN_ENERGY`].
const MIN_DISTANCE: f64 = 0.5;

pub type V = [f64; 3];

pub fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}

pub fn dist(a: V, b: V) -> f64 {
    norm(sub(a, b))
}

/// Amide hydrogen: H = N + unit(C(i−1) − O(i−1)). Proline and chain starts get H = N,
/// which makes the energy identically 0 (never a bond), matching DSSP.
pub fn place_hydrogens(res: &[Residue]) -> Vec<V> {
    let mut h = Vec::with_capacity(res.len());
    for (i, r) in res.iter().enumerate() {
        if i == 0 || r.is_proline || r.chain_break_before {
            h.push(r.n);
            continue;
        }
        let p = &res[i - 1];
        let d = sub(p.c, p.o);
        let l = norm(d);
        if l < 1e-6 {
            h.push(r.n);
        } else {
            h.push([r.n[0] + d[0] / l, r.n[1] + d[1] / l, r.n[2] + d[2] / l]);
        }
    }
    h
}

/// Energy of the N–H(donor) ··· O=C(acceptor) bond in kcal/mol.
pub fn energy(donor: &Residue, donor_h: V, acceptor: &Residue) -> f64 {
    let r_ho = dist(donor_h, acceptor.o);
    let r_hc = dist(donor_h, acceptor.c);
    let r_nc = dist(donor.n, acceptor.c);
    let r_no = dist(donor.n, acceptor.o);
    if r_ho < MIN_DISTANCE || r_hc < MIN_DISTANCE || r_nc < MIN_DISTANCE || r_no < MIN_DISTANCE {
        return MIN_ENERGY;
    }
    let e = COUPLING / r_ho - COUPLING / r_hc + COUPLING / r_nc - COUPLING / r_no;
    e.max(MIN_ENERGY)
}

#[derive(Clone, Copy, Debug)]
pub struct Partner {
    pub index: Option<usize>,
    pub energy: f64,
}

impl Default for Partner {
    fn default() -> Self {
        Partner {
            index: None,
            energy: 0.0,
        }
    }
}

/// Per residue: the two strongest acceptors of its N–H and the two strongest donors to its C=O.
#[derive(Clone, Debug, Default)]
pub struct Bonds {
    /// Residues whose C=O this residue's N–H bonds to.
    pub acceptor: [Partner; 2],
    /// Residues whose N–H bonds to this residue's C=O.
    pub donor: [Partner; 2],
}

fn insert(slot: &mut [Partner; 2], index: usize, energy: f64) {
    if energy < slot[0].energy {
        slot[1] = slot[0];
        slot[0] = Partner {
            index: Some(index),
            energy,
        };
    } else if energy < slot[1].energy {
        slot[1] = Partner {
            index: Some(index),
            energy,
        };
    }
}

/// Evaluate all candidate pairs (Cα–Cα < 9 Å) and keep the two best partners per direction.
pub fn compute_bonds(res: &[Residue]) -> Vec<Bonds> {
    let h = place_hydrogens(res);
    let mut bonds = vec![Bonds::default(); res.len()];
    for i in 0..res.len() {
        for j in (i + 1)..res.len() {
            if dist(res[i].ca, res[j].ca) >= MIN_CA_DISTANCE {
                continue;
            }
            let e = energy(&res[i], h[i], &res[j]);
            insert(&mut bonds[i].acceptor, j, e);
            insert(&mut bonds[j].donor, i, e);
            if j != i + 1 {
                let e = energy(&res[j], h[j], &res[i]);
                insert(&mut bonds[j].acceptor, i, e);
                insert(&mut bonds[i].donor, j, e);
            }
        }
    }
    bonds
}

/// True if residue `a`'s N–H donates to residue `b`'s C=O with E < −0.5 kcal/mol.
pub fn test_bond(bonds: &[Bonds], a: usize, b: usize) -> bool {
    bonds[a]
        .acceptor
        .iter()
        .any(|p| p.index == Some(b) && p.energy < MAX_BOND_ENERGY)
}
