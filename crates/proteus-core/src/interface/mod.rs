//! Protein–protein interface metrics for triaging designed binders.
//!
//! The binder-design filters in common use (BindCraft's defaults, and the features
//! Overath et al. 2025 found predictive over 3,766 wet-lab-tested binders) mix two kinds of
//! evidence:
//!
//! - **What the predictor is sure of about the interface**, from its PAE matrix: ipAE, ipSAE
//!   (Dunbrack 2025) and LIS (Kim et al. 2024). The strongest single predictor in the meta-analysis
//!   was ipSAE_min.
//! - **What the model's interface looks like**: buried surface (dSASA), shape complementarity
//!   (Sc, Lawrence & Colman 1993), residues in contact, and hydrogen bonds and salt bridges across it.
//!
//! Every one of these is computed without Rosetta. Rosetta's interface ΔG, packstat and
//! unsatisfied hydrogen bonds need an energy function and explicit hydrogens, and are not
//! computed.

pub mod sc;

use crate::error::CoreError;
use crate::io::{element_symbol, protein_heavy_atoms};
use crate::pae::PredictedAlignedError;
use crate::sasa::{compute_sasa, AtomDescriptor};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// A heavy atom within this distance of the other side puts its residue in the interface, Å
/// (BindCraft's `hotspot_residues` cutoff).
pub const INTERFACE_CONTACT_CUTOFF: f64 = 4.0;
/// Inter-chain PAE below this counts toward ipSAE, Å (Dunbrack 2025; used by Overath et al.).
pub const IPSAE_PAE_CUTOFF: f32 = 10.0;
/// Inter-chain PAE below this counts toward LIS, Å (Kim et al. 2024).
pub const LIS_PAE_CUTOFF: f32 = 12.0;

/// Which chains are the binder and which the target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterfaceSpec {
    pub binder: Vec<String>,
    /// Empty: every protein chain that is not the binder.
    pub target: Vec<String>,
}

impl InterfaceSpec {
    /// `A` (binder A against every other chain), `A:B` or `A:B,C` (binder A, target B and C).
    /// `H,L:A` makes a two-chain binder. An empty string means "the first chain".
    pub fn parse(s: &str) -> Result<Self, CoreError> {
        let split = |part: &str| -> Vec<String> {
            part.split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect()
        };
        let (binder, target) = match s.split_once(':') {
            Some((b, t)) => (split(b), split(t)),
            None => (split(s), Vec::new()),
        };
        if s.contains(':') && (binder.is_empty() || target.is_empty()) {
            return Err(CoreError::AnalysisError(format!(
                "interface '{s}': give chains on both sides of ':', e.g. A:B"
            )));
        }
        if binder.iter().any(|b| target.contains(b)) {
            return Err(CoreError::AnalysisError(format!(
                "interface '{s}': a chain cannot be both binder and target"
            )));
        }
        Ok(Self { binder, target })
    }
}

/// Interface metrics of one binder–target model. Chain lists are comma-joined ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceMetrics {
    pub binder_chains: String,
    pub target_chains: String,
    /// Binder residues with a heavy atom within [`INTERFACE_CONTACT_CUTOFF`] of the target.
    pub binder_interface_residues: usize,
    pub target_interface_residues: usize,
    /// SASA(binder) + SASA(target) − SASA(complex), Å² (Shrake–Rupley, Bondi radii, 1.4 Å probe).
    pub dsasa: f64,
    /// Shape complementarity; `None` when no interface surface forms.
    pub shape_complementarity: Option<f64>,
    /// Hydrogen bonds with donor and acceptor on opposite sides.
    pub interface_hbonds: usize,
    pub interface_salt_bridges: usize,
    /// Mean inter-chain PAE between binder and target residues, both directions, Å.
    pub ipae: Option<f64>,
    /// ipSAE over binder→target and target→binder chain pairs: the smallest and the largest.
    pub ipsae_min: Option<f64>,
    pub ipsae_max: Option<f64>,
    /// LIS, averaged over the binder→target and target→binder directions.
    pub lis: Option<f64>,
    /// Why the PAE metrics are empty, when a PAE was given but could not be matched to residues.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pae_note: Option<String>,
}

struct Side {
    atoms: Vec<(Vector3<f64>, String, String, String, usize)>, // coord, element, residue, atom, residue index
    residues: usize,
}

/// Interface metrics of `pdb` (first model, protein heavy atoms, first alternate location).
pub fn interface_metrics(
    pdb: &pdbtbx::PDB,
    spec: &InterfaceSpec,
    pae: Option<&PredictedAlignedError>,
) -> Result<InterfaceMetrics, CoreError> {
    let protein = protein_heavy_atoms(pdb);
    let chain_ids: Vec<String> = protein.chains().map(|c| c.id().to_string()).collect();
    let binder: Vec<String> = if spec.binder.is_empty() {
        chain_ids.first().cloned().into_iter().collect()
    } else {
        spec.binder.clone()
    };
    for b in binder.iter().chain(&spec.target) {
        if !chain_ids.contains(b) {
            return Err(CoreError::AnalysisError(format!(
                "no protein chain '{b}' (chains: {})",
                chain_ids.join(", ")
            )));
        }
    }
    let target: Vec<String> = if spec.target.is_empty() {
        chain_ids
            .iter()
            .filter(|c| !binder.contains(c))
            .cloned()
            .collect()
    } else {
        spec.target.clone()
    };
    if target.is_empty() {
        return Err(CoreError::AnalysisError(format!(
            "an interface needs two sides; the model has only chain {}",
            chain_ids.join(", ")
        )));
    }

    // Residue order over the whole model, as a predictor's PAE rows are laid out.
    let mut residue_chain: Vec<String> = Vec::new();
    let mut sides = [
        Side {
            atoms: Vec::new(),
            residues: 0,
        },
        Side {
            atoms: Vec::new(),
            residues: 0,
        },
    ];
    for chain in protein.chains() {
        let id = chain.id().to_string();
        let side = if binder.contains(&id) {
            Some(0)
        } else if target.contains(&id) {
            Some(1)
        } else {
            None
        };
        for residue in chain.residues() {
            let ridx = residue_chain.len();
            residue_chain.push(id.clone());
            let Some(s) = side else { continue };
            sides[s].residues += 1;
            let rname = residue.name().unwrap_or("UNK").to_string();
            for atom in residue.atoms() {
                let (x, y, z) = atom.pos();
                sides[s].atoms.push((
                    Vector3::new(x, y, z),
                    element_symbol(atom),
                    rname.clone(),
                    atom.name().trim().to_string(),
                    ridx,
                ));
            }
        }
    }

    let (binder_contacts, target_contacts) = contact_residues(&sides[0], &sides[1]);

    let descriptors = |s: &Side| -> Vec<AtomDescriptor> {
        s.atoms
            .iter()
            .map(|(c, e, ..)| AtomDescriptor::new(*c, e.clone()))
            .collect()
    };
    let (da, db) = (descriptors(&sides[0]), descriptors(&sides[1]));
    let sasa_a = compute_sasa(&da).total_sasa;
    let sasa_b = compute_sasa(&db).total_sasa;
    let complex: Vec<AtomDescriptor> = da.into_iter().chain(db).collect();
    let dsasa = sasa_a + sasa_b - compute_sasa(&complex).total_sasa;

    let sc_atoms = |s: &Side| -> Vec<sc::ScAtom> {
        s.atoms
            .iter()
            .map(|(c, _, r, a, _)| sc::ScAtom {
                residue: r.clone(),
                name: a.clone(),
                coord: [c.x, c.y, c.z],
            })
            .collect()
    };
    let shape_complementarity =
        sc::shape_complementarity(&sc_atoms(&sides[0]), &sc_atoms(&sides[1]))
            .ok()
            .map(|r| r.sc);

    let network = crate::interactions::compute_interaction_network(&protein);
    let across = |a: &str, b: &str| {
        (binder.iter().any(|c| c == a) && target.iter().any(|c| c == b))
            || (binder.iter().any(|c| c == b) && target.iter().any(|c| c == a))
    };
    let interface_hbonds = network
        .hbonds
        .iter()
        .filter(|h| across(&h.donor_chain_id, &h.acceptor_chain_id))
        .count();
    let interface_salt_bridges = network
        .salt_bridges
        .iter()
        .filter(|s| across(&s.cation_chain_id, &s.anion_chain_id))
        .count();

    let mut out = InterfaceMetrics {
        binder_chains: binder.join(","),
        target_chains: target.join(","),
        binder_interface_residues: binder_contacts,
        target_interface_residues: target_contacts,
        dsasa,
        shape_complementarity,
        interface_hbonds,
        interface_salt_bridges,
        ipae: None,
        ipsae_min: None,
        ipsae_max: None,
        lis: None,
        pae_note: None,
    };
    if let Some(pae) = pae {
        if pae.n != residue_chain.len() {
            out.pae_note = Some(format!(
                "the PAE matrix has {} rows and the model {} protein residues; PAE metrics need \
                 one row per residue in file order",
                pae.n,
                residue_chain.len()
            ));
        } else {
            let pm = pae_interface_metrics(pae, &residue_chain, &binder, &target);
            out.ipae = pm.ipae;
            out.ipsae_min = pm.ipsae_min;
            out.ipsae_max = pm.ipsae_max;
            out.lis = pm.lis;
        }
    }
    Ok(out)
}

/// Residues on each side with a heavy atom within [`INTERFACE_CONTACT_CUTOFF`] of the other side.
fn contact_residues(a: &Side, b: &Side) -> (usize, usize) {
    use std::collections::{HashMap, HashSet};
    let cell = INTERFACE_CONTACT_CUTOFF;
    let key = |c: &Vector3<f64>| {
        (
            (c.x / cell).floor() as i64,
            (c.y / cell).floor() as i64,
            (c.z / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, atom) in b.atoms.iter().enumerate() {
        grid.entry(key(&atom.0)).or_default().push(i);
    }
    let cut2 = cell * cell;
    let (mut ra, mut rb) = (HashSet::new(), HashSet::new());
    for atom in &a.atoms {
        let (x, y, z) = key(&atom.0);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = grid.get(&(x + dx, y + dy, z + dz)) else {
                        continue;
                    };
                    for &j in list {
                        if (atom.0 - b.atoms[j].0).norm_squared() <= cut2 {
                            ra.insert(atom.4);
                            rb.insert(b.atoms[j].4);
                        }
                    }
                }
            }
        }
    }
    (ra.len(), rb.len())
}

/// PAE-derived interface confidence.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PaeInterface {
    pub ipae: Option<f64>,
    pub ipsae_min: Option<f64>,
    pub ipsae_max: Option<f64>,
    pub lis: Option<f64>,
}

/// TM-score's d0 for `n` residues, floored at 26 residues and 1 Å, as ipSAE uses it
/// (Yang & Skolnick 2004; `calc_d0_array` in Dunbrack's `ipsae.py`).
fn ipsae_d0(n: usize) -> f64 {
    let l = n.max(26) as f64;
    (1.24 * (l - 15.0).cbrt() - 1.8).max(1.0)
}

/// ipSAE of the ordered chain pair (aligned chain `c1`, scored chain `c2`): for each residue of
/// `c1`, the pTM-style mean over the `c2` residues whose PAE is under the cutoff, with d0 from
/// how many there are; the pair's value is the best residue's.
fn ipsae_asym(pae: &PredictedAlignedError, chains: &[String], c1: &str, c2: &str) -> f64 {
    let mut best = 0.0_f64;
    for i in (0..pae.n).filter(|&i| chains[i] == c1) {
        let valid: Vec<f64> = (0..pae.n)
            .filter(|&j| chains[j] == c2)
            .map(|j| pae.get(i, j))
            .filter(|&v| v < IPSAE_PAE_CUTOFF)
            .map(f64::from)
            .collect();
        if valid.is_empty() {
            continue;
        }
        let d0 = ipsae_d0(valid.len());
        let score = valid
            .iter()
            .map(|v| 1.0 / (1.0 + (v / d0).powi(2)))
            .sum::<f64>()
            / valid.len() as f64;
        best = best.max(score);
    }
    best
}

/// LIS of the ordered chain pair: the mean of (12 − PAE)/12 over the pairs under 12 Å.
fn lis_asym(pae: &PredictedAlignedError, chains: &[String], c1: &str, c2: &str) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for i in (0..pae.n).filter(|&i| chains[i] == c1) {
        for j in (0..pae.n).filter(|&j| chains[j] == c2) {
            let v = pae.get(i, j);
            if v < LIS_PAE_CUTOFF {
                sum += f64::from((LIS_PAE_CUTOFF - v) / LIS_PAE_CUTOFF);
                n += 1;
            }
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// ipAE, ipSAE (min and max over binder↔target chain pairs, both directions) and LIS.
/// `chains[i]` is the chain of PAE row `i`.
pub fn pae_interface_metrics(
    pae: &PredictedAlignedError,
    chains: &[String],
    binder: &[String],
    target: &[String],
) -> PaeInterface {
    let is = |set: &[String], c: &str| set.iter().any(|s| s == c);
    // Both directions: binder rows against target columns and the reverse. This is what the
    // meta-analysis dataset holds (its af3_ipae agrees to 0.0005 Å on single-chain targets).
    let (mut sum, mut n) = (0.0, 0usize);
    for i in (0..pae.n).filter(|&i| is(binder, &chains[i])) {
        for j in (0..pae.n).filter(|&j| is(target, &chains[j])) {
            sum += f64::from(pae.get(i, j)) + f64::from(pae.get(j, i));
            n += 2;
        }
    }
    if n == 0 {
        return PaeInterface::default();
    }
    let mut ipsae = Vec::new();
    let mut lis = Vec::new();
    for b in binder {
        for t in target {
            ipsae.push(ipsae_asym(pae, chains, b, t));
            ipsae.push(ipsae_asym(pae, chains, t, b));
            lis.push(lis_asym(pae, chains, b, t));
            lis.push(lis_asym(pae, chains, t, b));
        }
    }
    PaeInterface {
        ipae: Some(sum / n as f64),
        ipsae_min: ipsae.iter().copied().reduce(f64::min),
        ipsae_max: ipsae.iter().copied().reduce(f64::max),
        lis: Some(lis.iter().sum::<f64>() / lis.len() as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pae(n: usize, f: impl Fn(usize, usize) -> f32) -> PredictedAlignedError {
        let mut values = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                values.push(f(i, j));
            }
        }
        PredictedAlignedError {
            n,
            values,
            max: 31.75,
        }
    }

    #[test]
    fn spec_parsing() {
        assert_eq!(
            InterfaceSpec::parse("A").unwrap(),
            InterfaceSpec {
                binder: vec!["A".into()],
                target: vec![]
            }
        );
        assert_eq!(
            InterfaceSpec::parse("H,L:A").unwrap(),
            InterfaceSpec {
                binder: vec!["H".into(), "L".into()],
                target: vec!["A".into()]
            }
        );
        assert!(InterfaceSpec::parse("A:").is_err());
        assert!(InterfaceSpec::parse("A:A").is_err());
        assert_eq!(
            InterfaceSpec::parse("").unwrap().binder,
            Vec::<String>::new()
        );
    }

    #[test]
    fn d0_matches_ipsae_py() {
        // calc_d0_array floors L at 26: 1.24·11^(1/3) − 1.8 = 0.9767 → floored to 1.0.
        assert_eq!(ipsae_d0(1), 1.0);
        assert_eq!(ipsae_d0(26), 1.0);
        let d100 = 1.24 * 85f64.cbrt() - 1.8;
        assert!((ipsae_d0(100) - d100).abs() < 1e-12);
    }

    /// Hand-computed: binder A = residues 0–1, target B = 2–4.
    #[test]
    fn pae_metrics_by_hand() {
        let chains: Vec<String> = ["A", "A", "B", "B", "B"].map(String::from).to_vec();
        // A→B entries 2 Å, B→A entries 20 Å (over the ipSAE and LIS cutoffs), intra-chain 1 Å.
        let m = pae(5, |i, j| match (chains[i].as_str(), chains[j].as_str()) {
            ("A", "B") => 2.0,
            ("B", "A") => 20.0,
            _ => 1.0,
        });
        let p = pae_interface_metrics(&m, &chains, &["A".into()], &["B".into()]);
        // (6 pairs × 2 Å + 6 pairs × 20 Å) / 12.
        assert_eq!(p.ipae, Some(11.0));
        // A→B: 3 valid pairs, d0 = 1.0 → 1/(1+4) = 0.2. B→A: nothing under 10 Å → 0.
        assert!((p.ipsae_max.unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(p.ipsae_min, Some(0.0));
        // LIS: A→B (12−2)/12 = 0.8333; B→A 0; mean of the two directions.
        assert!((p.lis.unwrap() - 10.0 / 24.0).abs() < 1e-6);
    }
}
