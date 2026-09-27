//! Covalent-geometry validation: how far a model's bond lengths, bond angles, chiral centres,
//! planar groups, Cβ positions and peptide ω angles are from ideal stereochemistry.
//!
//! This is the MolProbity/Phenix check set, reproduced from cctbx so that the numbers agree
//! with `phenix.molprobity` residue by residue (`validate/geometry_ref.py`,
//! `tests/geometry_validation.rs`):
//!
//! * **Restraints** — ideal values from Phenix geostd (Engh & Huber 1991 for the side chains),
//!   with the backbone replaced by the Conformation-Dependent Library v1.2 as Phenix and the
//!   wwPDB do by default, and Engh & Huber 1999 values for cis-proline. A restraint is an
//!   outlier beyond 4σ (`mmtbx/validation/restraints.py`).
//! * **Cβ deviation** — `mmtbx/validation/cbetadev.py`, outlier at ≥ 0.25 Å.
//! * **ω** — `mmtbx/validation/omegalyze.py`: cis within 30° of 0°, twisted between 30° and
//!   150°.
//! * **Rotamers** — [`crate::rotamer`], MolProbity Top8000 (`mmtbx/validation/rotalyze.py`).
//!
//! Only the 20 standard amino acids and selenomethionine are restrained; hydrogens are not used (predicted models
//! have none), so every check here is defined on heavy atoms. Input is the structure after
//! [`crate::io::protein_heavy_atoms`].

pub mod cbeta;
mod cdl;
#[rustfmt::skip]
mod library;
mod library_types;
pub mod omega;
mod restraints;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

pub use cbeta::CBETA_OUTLIER;
pub use omega::OmegaType;
pub use restraints::Origin as RestraintOrigin;

/// `mmtbx/validation/restraints.py`: restraints beyond this many σ are outliers.
pub const SIGMA_CUTOFF: f64 = 4.0;

/// Identity of a residue, as the PDB/mmCIF author numbering gives it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ResidueId {
    pub chain: String,
    pub seq_num: isize,
    /// Insertion code, empty when absent.
    pub insertion_code: String,
    pub name: String,
}

impl std::fmt::Display for ResidueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}{} {}",
            self.chain, self.seq_num, self.insertion_code, self.name
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Atom {
    pub pos: Vector3<f64>,
    pub name: String,
    pub residue: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Residue {
    pub id: ResidueId,
    pub name: String,
    atoms: Vec<(String, usize)>,
}

impl Residue {
    /// Index of the atom named `name` (first one if the file repeats a name).
    pub fn atom(&self, name: &str) -> Option<usize> {
        self.atoms.iter().find(|(n, _)| n == name).map(|&(_, i)| i)
    }
}

/// Atoms and residues of one structure in file order.
#[derive(Clone, Debug, Default)]
pub(crate) struct Model {
    pub atoms: Vec<Atom>,
    pub residues: Vec<Residue>,
}

impl Model {
    /// Atoms of the first model only (as `io::load_structure` keeps), so that a caller passing
    /// an NMR ensemble does not get restraints between copies of the same residue.
    pub fn from_pdb(pdb: &pdbtbx::PDB) -> Model {
        let mut m = Model::default();
        let Some(first) = pdb.models().next() else {
            return m;
        };
        for chain in first.chains() {
            for residue in chain.residues() {
                let name = residue
                    .name()
                    .map(|n| n.trim().to_uppercase())
                    .unwrap_or_default();
                let index = m.residues.len();
                let mut atoms = Vec::new();
                for atom in residue.atoms() {
                    let atom_name = atom.name().trim().to_uppercase();
                    atoms.push((atom_name.clone(), m.atoms.len()));
                    m.atoms.push(Atom {
                        pos: Vector3::new(atom.x(), atom.y(), atom.z()),
                        name: atom_name,
                        residue: index,
                    });
                }
                m.residues.push(Residue {
                    id: ResidueId {
                        chain: chain.id().to_string(),
                        seq_num: residue.serial_number(),
                        insertion_code: residue
                            .insertion_code()
                            .map(|c| c.trim().to_string())
                            .unwrap_or_default(),
                        name: name.clone(),
                    },
                    name,
                    atoms,
                });
            }
        }
        m
    }

    fn atom_label(&self, i: usize) -> String {
        let a = &self.atoms[i];
        format!("{} {}", self.residues[a.residue].id, a.name)
    }
}

/// How cctbx decides that a symmetric side chain is named against convention
/// (`iotbx.pdb.hierarchy.flip_symmetric_amino_acids`).
enum FlipTest {
    /// Flip when |dihedral| of these atoms exceeds 90°.
    Dihedral([&'static str; 4]),
    /// Flip when the chiral volume of these atoms is more than 2 Å³ from −2.5 Å³.
    Chiral([&'static str; 4]),
}

fn flip_rule(resname: &str) -> Option<(FlipTest, &'static [(&'static str, &'static str)])> {
    Some(match resname {
        "ARG" => (
            FlipTest::Dihedral(["CD", "NE", "CZ", "NH1"]),
            &[("NH1", "NH2")],
        ),
        "ASP" => (
            FlipTest::Dihedral(["CA", "CB", "CG", "OD1"]),
            &[("OD1", "OD2")],
        ),
        "GLU" => (
            FlipTest::Dihedral(["CB", "CG", "CD", "OE1"]),
            &[("OE1", "OE2")],
        ),
        "PHE" | "TYR" => (
            FlipTest::Dihedral(["CA", "CB", "CG", "CD1"]),
            &[("CD1", "CD2"), ("CE1", "CE2")],
        ),
        "VAL" => (
            FlipTest::Chiral(["CB", "CA", "CG1", "CG2"]),
            &[("CG1", "CG2")],
        ),
        "LEU" => (
            FlipTest::Chiral(["CG", "CB", "CD1", "CD2"]),
            &[("CD1", "CD2")],
        ),
        _ => return None,
    })
}

impl Model {
    /// Swap the coordinates of chemically equivalent atoms whose names break the IUPAC
    /// convention (Arg NH1/NH2, Asp OD1/OD2, Glu OE1/OE2, Phe/Tyr ring, Val/Leu methyls), as
    /// Phenix does before restraining (`pdb_interpretation.flip_symmetric_amino_acids = True`).
    /// Without it an arbitrary naming choice would read as a geometry outlier. Returns the
    /// number of residues renamed.
    fn flip_symmetric_amino_acids(&mut self) -> usize {
        let mut flips = 0;
        for res in &self.residues {
            let Some((test, pairs)) = flip_rule(&res.name) else {
                continue;
            };
            let pos = |n: &str| res.atom(n).map(|i| self.atoms[i].pos);
            let flip = match test {
                FlipTest::Dihedral(names) => {
                    let [a, b, c, d] = names.map(pos);
                    match (a, b, c, d) {
                        (Some(a), Some(b), Some(c), Some(d)) => {
                            restraints::dihedral(a, b, c, d).is_some_and(|w| w.abs() > 90.0)
                        }
                        _ => false,
                    }
                }
                FlipTest::Chiral(names) => {
                    let [a, b, c, d] = names.map(pos);
                    match (a, b, c, d) {
                        (Some(a), Some(b), Some(c), Some(d)) => {
                            (-2.5 - restraints::chiral_volume([a, b, c, d])).abs() > 2.0
                        }
                        _ => false,
                    }
                }
            };
            if !flip {
                continue;
            }
            // A pair with only one atom present leaves the residue as it is.
            let swaps: Option<Vec<(usize, usize)>> = pairs
                .iter()
                .filter_map(|&(x, y)| match (res.atom(x), res.atom(y)) {
                    (None, None) => None,
                    (Some(i), Some(j)) => Some(Some((i, j))),
                    _ => Some(None),
                })
                .collect();
            let Some(swaps) = swaps else { continue };
            for (i, j) in swaps {
                let (pi, pj) = (self.atoms[i].pos, self.atoms[j].pos);
                self.atoms[i].pos = pj;
                self.atoms[j].pos = pi;
            }
            flips += 1;
        }
        flips
    }
}

/// A copy of `pdb` (first model) with symmetric side-chain atoms named against the IUPAC
/// convention swapped, as Phenix does before validating, and the number of residues changed.
/// Every check in this module, and [`crate::rotamer`] through [`analyze`], sees this copy.
pub fn flip_symmetric_amino_acids(pdb: &pdbtbx::PDB) -> (pdbtbx::PDB, usize) {
    let mut model = Model::from_pdb(pdb);
    let flips = model.flip_symmetric_amino_acids();
    let mut out = pdb.clone();
    while out.model_count() > 1 {
        out.remove_model(1);
    }
    if flips > 0 {
        // Same traversal as `Model::from_pdb`, so atom k of the model is atom k here.
        let mut k = 0;
        if let Some(first) = out.models_mut().next() {
            for chain in first.chains_mut() {
                for residue in chain.residues_mut() {
                    for atom in residue.atoms_mut() {
                        let p = model.atoms[k].pos;
                        atom.set_pos((p.x, p.y, p.z)).expect("finite coordinates");
                        k += 1;
                    }
                }
            }
        }
    }
    (out, flips)
}

/// Kind of a covalent-geometry outlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum OutlierKind {
    Bond,
    Angle,
    /// Chiral volume beyond 4σ but below the handedness-swap threshold: a distorted centre.
    Tetrahedral,
    /// Chiral volume of the wrong sign: the centre is the mirror image (a D-amino acid, or an
    /// inverted Cβ of Ile/Thr). Both-sign centres (none among the standard residues) never
    /// count.
    HandednessSwap,
    /// The sign of a Val CB or Leu CG "centre" is swapped: the two methyls are misnamed.
    PseudochiralNaming,
    Planarity,
    Cbeta,
    /// A cis peptide bond before a residue other than proline.
    CisPeptide,
    TwistedPeptide,
    /// A side chain in a conformation seen in fewer than 0.3 % of Top8000 residues.
    Rotamer,
    /// A restraint with no defined value because two of its atoms coincide.
    Degenerate,
}

/// One outlier, located and quantified.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GeometryOutlier {
    pub kind: OutlierKind,
    /// Residue of the first atom (the central residue for Cβ and ω).
    pub residue: ResidueId,
    /// Atom labels ("A 12 LYS CA").
    pub atoms: Vec<String>,
    /// Ideal value (Å, °, Å³); 0 for planes; 0.25 Å for Cβ; 180° for ω; 0.3 % for rotamers.
    pub ideal: f64,
    /// Model value; the largest out-of-plane distance for planes; the Top8000 percentile for
    /// rotamers.
    pub model: f64,
    /// Signed (ideal − model)/σ for restraints, largest |deviation|/σ for planes, `None` for
    /// Cβ, ω and rotamers.
    pub z: Option<f64>,
}

impl GeometryOutlier {
    /// Ordering key, most severe first, on a common σ scale: restraints by |Z|; ω by its
    /// distance from 180° over the 5° esd of the peptide link's ω restraint (geostd TRANS);
    /// Cβ with 0.25 Å at 4σ; rotamer outliers at 4σ. Inverted chiral centres come first.
    fn severity(&self) -> f64 {
        match self.kind {
            OutlierKind::HandednessSwap => 1e6 + self.z.map_or(0.0, f64::abs),
            OutlierKind::TwistedPeptide | OutlierKind::CisPeptide => {
                (180.0 - self.model.abs()) / OMEGA_ESD
            }
            OutlierKind::Cbeta => self.model / CBETA_OUTLIER * SIGMA_CUTOFF,
            OutlierKind::Rotamer => SIGMA_CUTOFF,
            OutlierKind::Degenerate => 1e5,
            _ => self.z.map_or(0.0, f64::abs),
        }
    }
}

/// esd of the ω dihedral restraint of the geostd TRANS/CIS peptide links, degrees.
const OMEGA_ESD: f64 = 5.0;

/// Count, outliers and RMSZ of one restraint type.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RestraintStats {
    pub n: usize,
    /// Restraints deviating by more than [`SIGMA_CUTOFF`] σ.
    pub outliers: usize,
    /// Root-mean-square Z over all `n` restraints (about 1 for a well-refined model).
    pub rmsz: Option<f64>,
}

impl RestraintStats {
    fn from_z(z: impl Iterator<Item = f64>) -> RestraintStats {
        let (mut n, mut out, mut sum) = (0usize, 0usize, 0.0);
        for z in z {
            n += 1;
            sum += z * z;
            if z.abs() > SIGMA_CUTOFF {
                out += 1;
            }
        }
        RestraintStats {
            n,
            outliers: out,
            rmsz: (n > 0).then(|| (sum / n as f64).sqrt()),
        }
    }
}

/// Per-residue Cβ deviation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CbetaResult {
    pub residue: ResidueId,
    /// Å.
    pub deviation: f64,
}

/// ω of the peptide bond from the previous residue into `residue`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct OmegaResult {
    pub residue: ResidueId,
    /// Degrees, CA(i−1)–C(i−1)–N(i)–CA(i).
    pub omega: f64,
    pub omega_type: OmegaType,
    /// True when `residue` is a proline (cis is then unremarkable).
    pub proline: bool,
}

/// Everything the covalent-geometry checks found in one structure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CovalentGeometry {
    pub bonds: RestraintStats,
    pub angles: RestraintStats,
    pub chiralities: RestraintStats,
    pub planes: RestraintStats,
    /// Chiral centres of the wrong hand, excluding Val/Leu pseudo-chiral naming swaps.
    pub handedness_swaps: usize,
    pub cbeta_residues: usize,
    /// Residues with a Cβ deviation ≥ [`CBETA_OUTLIER`].
    pub cbeta_outliers: usize,
    pub peptides: usize,
    pub cis_proline: usize,
    pub cis_nonproline: usize,
    pub twisted: usize,
    /// Side chains evaluated against the Top8000 rotamer distributions.
    pub rotamer_residues: usize,
    /// Rotamers in the 0.3–2 % band.
    pub rotamer_allowed: usize,
    /// Rotamers below 0.3 %.
    pub rotamer_outliers: usize,
    /// Residues whose symmetric side-chain atoms were named against the IUPAC convention and
    /// were swapped before restraining (Arg, Asp, Glu, Phe, Tyr, Val, Leu).
    pub symmetric_flips: usize,
    /// Restraints left out of the statistics because their atoms coincide (listed as
    /// `Degenerate` outliers).
    #[serde(default)]
    pub degenerate: usize,
    /// Outliers, most severe first; possibly only the worst of them, see `outliers_total`.
    pub outliers: Vec<GeometryOutlier>,
    /// How many outliers there were before [`CovalentGeometry::keep_worst_outliers`].
    #[serde(default)]
    pub outliers_total: usize,
    /// Per-residue detail (not needed for summaries; skipped when serialising).
    #[serde(skip)]
    pub cbeta: Vec<CbetaResult>,
    #[serde(skip)]
    pub omegas: Vec<OmegaResult>,
}

impl CovalentGeometry {
    /// True when nothing could be measured: no restraint, Cβ, peptide or side chain (a
    /// C-alpha-only trace, or no standard residues).
    pub fn is_empty(&self) -> bool {
        self.bonds.n == 0
            && self.angles.n == 0
            && self.cbeta_residues == 0
            && self.peptides == 0
            && self.rotamer_residues == 0
    }

    /// Rotamer outliers as a percentage of evaluated side chains (MolProbity's figure).
    pub fn rotamer_outlier_pct(&self) -> Option<f64> {
        (self.rotamer_residues > 0)
            .then(|| self.rotamer_outliers as f64 * 100.0 / self.rotamer_residues as f64)
    }

    /// Cβ outliers as a percentage of residues with a Cβ.
    pub fn cbeta_outlier_pct(&self) -> Option<f64> {
        (self.cbeta_residues > 0)
            .then(|| self.cbeta_outliers as f64 * 100.0 / self.cbeta_residues as f64)
    }

    /// Keep only the `n` most severe outliers (counts are unaffected); `outliers_total`
    /// records how many there were.
    pub fn keep_worst_outliers(&mut self, n: usize) {
        self.outliers.truncate(n);
    }
}

/// Type of a chirality outlier. A centre is inverted when the model's chiral volume is on the
/// opposite side of zero by at least half the ideal magnitude; a volume near zero is a
/// flattened centre, not a mirror image, and stays a tetrahedral outlier. Val CB and Leu CG are not true centres, so their inversion
/// means the two methyls are misnamed. cctbx (`chirality.outlier_type`) instead calls any
/// |Z| > 20 (22 for Pro) a handedness swap, which also catches badly distorted centres of the
/// right hand (4HHB D:47 Asp CA: +8.5 Å³ against +2.5 Å³); here those stay tetrahedral outliers.
fn chirality_kind(model: &Model, centre: usize, ideal: f64, value: f64) -> OutlierKind {
    if ideal * value >= 0.0 || value.abs() < ideal.abs() / 2.0 {
        return OutlierKind::Tetrahedral;
    }
    let res = &model.residues[model.atoms[centre].residue].name;
    let name = model.atoms[centre].name.as_str();
    if (res == "VAL" && name == "CB") || (res == "LEU" && name == "CG") {
        OutlierKind::PseudochiralNaming
    } else {
        OutlierKind::HandednessSwap
    }
}

/// Kind of a covalent restraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum RestraintKind {
    Bond,
    Angle,
    Chirality,
    Planarity,
}

/// One restraint measured on the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RestraintDetail {
    pub kind: RestraintKind,
    pub origin: RestraintOrigin,
    /// Atom labels ("A 12 LYS CA"); the chiral centre first for chiralities.
    pub atoms: Vec<String>,
    /// Target: Å, °, signed Å³; 0 for planes.
    pub ideal: f64,
    /// σ of the restraint; for planes, σ of the atom that deviates most.
    pub esd: f64,
    /// Model value; for planes, the RMS out-of-plane distance (Å).
    pub model: f64,
    /// (ideal − model)/σ, or for planes the largest |out-of-plane distance|/σ.
    pub z: f64,
}

struct Evaluated {
    kind: RestraintKind,
    origin: RestraintOrigin,
    atoms: Vec<usize>,
    ideal: f64,
    esd: f64,
    model: f64,
    z: f64,
    /// Largest out-of-plane distance, planes only.
    max_delta: f64,
}

/// Every restraint measured, split into those with a defined value and the degenerate ones
/// (coincident atoms: an angle or plane with no defined value).
fn evaluate(model: &Model, r: &restraints::Restraints) -> (Vec<Evaluated>, Vec<Evaluated>) {
    let mut out =
        Vec::with_capacity(r.bonds.len() + r.angles.len() + r.chiralities.len() + r.planes.len());
    for b in &r.bonds {
        let (v, d) = restraints::bond_value(model, b);
        out.push(Evaluated {
            kind: RestraintKind::Bond,
            origin: b.origin,
            atoms: b.atoms.to_vec(),
            ideal: b.ideal,
            esd: b.esd,
            model: v,
            z: d / b.esd,
            max_delta: 0.0,
        });
    }
    for a in &r.angles {
        let (v, d) = restraints::angle_value(model, a);
        out.push(Evaluated {
            kind: RestraintKind::Angle,
            origin: a.origin,
            atoms: a.atoms.to_vec(),
            ideal: a.ideal,
            esd: a.esd,
            model: v,
            z: d / a.esd,
            max_delta: 0.0,
        });
    }
    for c in &r.chiralities {
        let (v, d) = restraints::chirality_value(model, c);
        out.push(Evaluated {
            kind: RestraintKind::Chirality,
            origin: RestraintOrigin::Residue,
            atoms: c.atoms.to_vec(),
            ideal: c.ideal,
            esd: restraints::CHIRAL_VOLUME_ESD,
            model: v,
            z: d / restraints::CHIRAL_VOLUME_ESD,
            max_delta: 0.0,
        });
    }
    for p in &r.planes {
        let deltas = restraints::plane_deltas(model, p);
        let (mut worst, mut esd) = (f64::NEG_INFINITY, p.esds[0]);
        for (d, e) in deltas.iter().zip(&p.esds) {
            if d.abs() / e > worst {
                worst = d.abs() / e;
                esd = *e;
            }
        }
        let rms = (deltas.iter().map(|d| d * d).sum::<f64>() / deltas.len() as f64).sqrt();
        out.push(Evaluated {
            kind: RestraintKind::Planarity,
            origin: p.origin,
            atoms: p.atoms.clone(),
            ideal: 0.0,
            esd,
            model: rms,
            z: worst,
            max_delta: deltas.iter().map(|d| d.abs()).fold(0.0, f64::max),
        });
    }
    out.into_iter()
        .partition(|e| e.z.is_finite() && e.model.is_finite() && e.max_delta.is_finite())
}

/// Every covalent restraint of the structure with its model value and Z, for export or
/// inspection. Input as for [`analyze`].
pub fn restraint_details(pdb: &pdbtbx::PDB) -> Vec<RestraintDetail> {
    let (flipped, _) = flip_symmetric_amino_acids(pdb);
    let model = Model::from_pdb(&flipped);
    let r = restraints::build(&model);
    evaluate(&model, &r)
        .0
        .into_iter()
        .map(|e| RestraintDetail {
            kind: e.kind,
            origin: e.origin,
            atoms: e.atoms.iter().map(|&i| model.atom_label(i)).collect(),
            ideal: e.ideal,
            esd: e.esd,
            model: e.model,
            z: e.z,
        })
        .collect()
}

/// Run every covalent-geometry check on a structure already reduced by
/// [`crate::io::protein_heavy_atoms`].
pub fn analyze(pdb: &pdbtbx::PDB) -> CovalentGeometry {
    let (flipped, symmetric_flips) = flip_symmetric_amino_acids(pdb);
    let pdb = &flipped;
    let model = Model::from_pdb(pdb);
    let r = restraints::build(&model);
    let (evaluated, degenerate) = evaluate(&model, &r);
    let mut g = CovalentGeometry {
        symmetric_flips,
        degenerate: degenerate.len(),
        ..CovalentGeometry::default()
    };
    for e in &degenerate {
        g.outliers.push(GeometryOutlier {
            kind: OutlierKind::Degenerate,
            residue: model.residues[model.atoms[e.atoms[0]].residue].id.clone(),
            atoms: e.atoms.iter().map(|&i| model.atom_label(i)).collect(),
            ideal: e.ideal,
            model: 0.0,
            z: None,
        });
    }
    let of = |k: RestraintKind| evaluated.iter().filter(move |e| e.kind == k);
    g.bonds = RestraintStats::from_z(of(RestraintKind::Bond).map(|e| e.z));
    g.angles = RestraintStats::from_z(of(RestraintKind::Angle).map(|e| e.z));
    g.chiralities = RestraintStats::from_z(of(RestraintKind::Chirality).map(|e| e.z));
    g.planes = RestraintStats::from_z(of(RestraintKind::Planarity).map(|e| e.z));
    for e in evaluated.iter().filter(|e| e.z.abs() > SIGMA_CUTOFF) {
        let kind = match e.kind {
            RestraintKind::Bond => OutlierKind::Bond,
            RestraintKind::Angle => OutlierKind::Angle,
            RestraintKind::Planarity => OutlierKind::Planarity,
            RestraintKind::Chirality => chirality_kind(&model, e.atoms[0], e.ideal, e.model),
        };
        if kind == OutlierKind::HandednessSwap {
            g.handedness_swaps += 1;
        }
        g.outliers.push(GeometryOutlier {
            kind,
            residue: model.residues[model.atoms[e.atoms[0]].residue].id.clone(),
            atoms: e.atoms.iter().map(|&i| model.atom_label(i)).collect(),
            ideal: e.ideal,
            model: if e.kind == RestraintKind::Planarity {
                e.max_delta
            } else {
                e.model
            },
            z: Some(e.z),
        });
    }

    for res in &model.residues {
        let pos = |n: &str| res.atom(n).map(|i| model.atoms[i].pos);
        let (Some(n), Some(ca), Some(c), Some(cb)) = (pos("N"), pos("CA"), pos("C"), pos("CB"))
        else {
            continue;
        };
        if let Some(d) = cbeta::deviation(&res.name, n, ca, c, cb) {
            g.cbeta_residues += 1;
            if d >= CBETA_OUTLIER {
                g.cbeta_outliers += 1;
                g.outliers.push(GeometryOutlier {
                    kind: OutlierKind::Cbeta,
                    residue: res.id.clone(),
                    atoms: vec![model.atom_label(res.atom("CB").unwrap())],
                    ideal: CBETA_OUTLIER,
                    model: d,
                    z: None,
                });
            }
            g.cbeta.push(CbetaResult {
                residue: res.id.clone(),
                deviation: d,
            });
        }
    }

    for w in model.residues.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.id.chain != b.id.chain {
            continue;
        }
        // Linked through C(i−1)–N(i) (`are_linked`); ω then needs both CAs as well.
        let pos = |r: &Residue, n: &str| r.atom(n).map(|i| model.atoms[i].pos);
        let (Some(ca1), Some(c1), Some(n2), Some(ca2)) =
            (pos(a, "CA"), pos(a, "C"), pos(b, "N"), pos(b, "CA"))
        else {
            continue;
        };
        if (c1 - n2).norm() >= omega::OMEGALYZE_LINK_CUTOFF {
            continue;
        }
        let Some(w) = restraints::dihedral(ca1, c1, n2, ca2) else {
            continue;
        };
        let t = OmegaType::classify(w);
        let proline = b.name == "PRO";
        g.peptides += 1;
        match (t, proline) {
            (OmegaType::Cis, true) => g.cis_proline += 1,
            (OmegaType::Cis, false) => g.cis_nonproline += 1,
            (OmegaType::Twisted, _) => g.twisted += 1,
            (OmegaType::Trans, _) => {}
        }
        // A cis proline is common (~5 %) and counted, but it is not a problem to list.
        if t == OmegaType::Twisted || (t == OmegaType::Cis && !proline) {
            g.outliers.push(GeometryOutlier {
                kind: if t == OmegaType::Cis {
                    OutlierKind::CisPeptide
                } else {
                    OutlierKind::TwistedPeptide
                },
                residue: b.id.clone(),
                atoms: vec![
                    format!("{} CA", a.id),
                    format!("{} C", a.id),
                    format!("{} N", b.id),
                    format!("{} CA", b.id),
                ],
                ideal: 180.0,
                model: w,
                z: None,
            });
        }
        g.omegas.push(OmegaResult {
            residue: b.id.clone(),
            omega: w,
            omega_type: t,
            proline,
        });
    }

    for rot in crate::rotamer::evaluate_rotamers(pdb) {
        g.rotamer_residues += 1;
        match rot.evaluation {
            crate::rotamer::RotamerEvaluation::Favored => {}
            crate::rotamer::RotamerEvaluation::Allowed => g.rotamer_allowed += 1,
            crate::rotamer::RotamerEvaluation::Outlier => {
                g.rotamer_outliers += 1;
                let id = ResidueId {
                    chain: rot.chain_id,
                    seq_num: rot.seq_num,
                    insertion_code: rot.insertion_code.unwrap_or_default(),
                    name: rot.resname,
                };
                g.outliers.push(GeometryOutlier {
                    kind: OutlierKind::Rotamer,
                    atoms: vec![format!(
                        "{id} χ {}",
                        rot.chi
                            .iter()
                            .map(|c| format!("{c:.0}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )],
                    residue: id,
                    ideal: 0.3,
                    model: rot.score,
                    z: None,
                });
            }
        }
    }

    g.outliers
        .sort_by(|a, b| b.severity().total_cmp(&a.severity()));
    g.outliers_total = g.outliers.len();
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    const CRAMBIN: &str = include_str!("../../tests/data/1crn.pdb");

    fn crambin() -> pdbtbx::PDB {
        let (pdb, _) = pdbtbx::ReadOptions::default()
            .set_format(pdbtbx::Format::Pdb)
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read_raw(std::io::BufReader::new(std::io::Cursor::new(
                CRAMBIN.as_bytes(),
            )))
            .unwrap();
        crate::io::protein_heavy_atoms(&pdb)
    }

    fn atom_mut<'a>(pdb: &'a mut pdbtbx::PDB, seq: isize, name: &str) -> &'a mut pdbtbx::Atom {
        pdb.residues_mut()
            .find(|r| r.serial_number() == seq)
            .unwrap()
            .atoms_mut()
            .find(|a| a.name().trim() == name)
            .unwrap()
    }

    fn pos(pdb: &pdbtbx::PDB, seq: isize, name: &str) -> (f64, f64, f64) {
        let a = pdb
            .residues()
            .find(|r| r.serial_number() == seq)
            .unwrap()
            .atoms()
            .find(|a| a.name().trim() == name)
            .unwrap();
        a.pos()
    }

    #[test]
    fn crambin_matches_cctbx_counts() {
        // validate/reference/geometry/1crn_pdb.json.gz
        let g = analyze(&crambin());
        assert_eq!((g.bonds.n, g.bonds.outliers), (337, 2));
        assert_eq!((g.angles.n, g.angles.outliers), (466, 10));
        assert_eq!(g.chiralities.n, 56);
        assert_eq!(g.planes.n, 61);
        assert!((g.bonds.rmsz.unwrap() - 1.49865).abs() < 1e-4);
        assert_eq!(g.handedness_swaps + g.cis_nonproline + g.twisted, 0);
    }

    #[test]
    fn coincident_atoms_are_reported_not_nan() {
        let mut pdb = crambin();
        let ca = pos(&pdb, 2, "CA");
        atom_mut(&mut pdb, 2, "CB").set_pos(ca).unwrap();
        let g = analyze(&pdb);
        assert!(g.degenerate > 0);
        assert!(g.outliers.iter().any(|o| o.kind == OutlierKind::Degenerate));
        for s in [&g.bonds, &g.angles, &g.chiralities, &g.planes] {
            assert!(s.rmsz.is_some_and(f64::is_finite));
        }
        // Everything serialises to JSON that reads back.
        let json = serde_json::to_string(&restraint_details(&pdb)).unwrap();
        let _: Vec<RestraintDetail> = serde_json::from_str(&json).unwrap();
        let _: CovalentGeometry =
            serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    }

    #[test]
    fn undefined_omega_is_not_a_cis_peptide() {
        let mut pdb = crambin();
        let n = pos(&pdb, 4, "N");
        atom_mut(&mut pdb, 4, "CA").set_pos(n).unwrap();
        let g = analyze(&pdb);
        assert_eq!(g.cis_nonproline, 0);
        assert!(g.cbeta.iter().all(|c| c.deviation.is_finite()));
    }

    #[test]
    fn only_the_first_model_is_checked() {
        let one = crambin();
        let mut two = one.clone();
        let mut copy = two.model(0).unwrap().clone();
        copy.set_serial_number(2);
        two.add_model(copy);
        assert_eq!(two.model_count(), 2);
        let (a, b) = (analyze(&one), analyze(&two));
        assert_eq!((a.bonds.n, a.bonds.outliers), (b.bonds.n, b.bonds.outliers));
        assert_eq!(a.cbeta_residues, b.cbeta_residues);
    }

    #[test]
    fn rotamers_are_scored_after_the_symmetric_flip() {
        // Swapping LEU 18's methyl names must not change its rotamer: Phenix renames them first.
        let pdb = crambin();
        let mut swapped = pdb.clone();
        let (d1, d2) = (pos(&pdb, 18, "CD1"), pos(&pdb, 18, "CD2"));
        atom_mut(&mut swapped, 18, "CD1").set_pos(d2).unwrap();
        atom_mut(&mut swapped, 18, "CD2").set_pos(d1).unwrap();
        let (a, b) = (analyze(&pdb), analyze(&swapped));
        assert_eq!(b.symmetric_flips, a.symmetric_flips + 1);
        assert_eq!(a.rotamer_outliers, b.rotamer_outliers);
        assert_eq!(a.rotamer_allowed, b.rotamer_allowed);
        assert_eq!(a.bonds, b.bonds);
    }
}
