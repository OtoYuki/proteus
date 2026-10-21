//! What a design candidate changed relative to its parent: which residues differ, how far the
//! structure moved around them, what the predictor's confidence is there, and which
//! non-covalent interactions were lost or gained.
//!
//! The interactions are [`compute_interaction_network`]'s, on both structures. Each one is keyed
//! by kind and endpoints (residue, and atom for hydrogen bonds) in the parent's numbering, with
//! the candidate's residues mapped through [`crate::pairing`]. An interaction present in only one
//! of the two is a change; for every change the same atoms are looked up in the other structure,
//! so "the side chain is gone" and "it moved 1.4 Å" are reported as the different findings they
//! are.
//!
//! Two crystal structures of one protein differ by noise at every cut-off, so each change also
//! carries its distance to the nearest changed site, and only changes within
//! [`DiffOptions::local_radius`] of one are *local*. Distal changes are counted, not narrated.

use crate::confidence::{detect_confidence_source, ConfidenceSource};
use crate::error::CoreError;
use crate::interactions::{compute_interaction_network, InteractionNetwork};
use crate::io::{load_structure_bytes, protein_heavy_atoms};
use crate::metrics::compute_kabsch_superposition;
use crate::pairing::{pair_best, ResidueId};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Changes within this distance (Å) of a changed site are local: a first and second shell.
pub const DEFAULT_LOCAL_RADIUS: f64 = 10.0;

/// A change is *clear* when an atom of it is gone from the other structure, or the other
/// structure's distance is past the interaction's cut-off by at least this much (Å). The rest
/// are *marginal*: the same atoms sit within this margin of the cut-off, or only an angle
/// criterion flipped, which two crystal structures of one protein do by noise alone.
pub const CLEAR_MARGIN: f64 = 0.5;

/// Fewest paired residues a diff is attempted on (three points fix a superposition).
pub const MIN_PAIRED_RESIDUES: usize = 3;

#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Restrict both structures to these chains (crystal files often hold several copies).
    pub chains: Option<Vec<String>>,
    /// See [`DEFAULT_LOCAL_RADIUS`].
    pub local_radius: f64,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            chains: None,
            local_radius: DEFAULT_LOCAL_RADIUS,
        }
    }
}

/// One residue with a C-alpha, and its heavy atoms.
#[derive(Debug, Clone)]
struct Residue {
    id: ResidueId,
    name: String,
    ca: Vector3<f64>,
    /// B-factor of the C-alpha (the predictor's confidence for a predicted model).
    b_factor: f64,
    atoms: Vec<(String, Vector3<f64>)>,
}

impl Residue {
    fn atom(&self, name: &str) -> Option<Vector3<f64>> {
        self.atoms.iter().find(|(n, _)| n == name).map(|&(_, p)| p)
    }

    /// The atom, or a chemically equivalent one under the other name (ASP OD1/OD2, …), which
    /// files assign either way round.
    fn atom_or_equivalent(&self, name: &str) -> Option<Vector3<f64>> {
        self.atom(name)
            .or_else(|| equivalent_atom(&self.name, name).and_then(|alt| self.atom(alt)))
    }

    fn ring_centroid(&self) -> Option<Vector3<f64>> {
        let names = ring_atoms(&self.name)?;
        let found: Vec<Vector3<f64>> = names.iter().filter_map(|n| self.atom(n)).collect();
        (found.len() >= 3).then(|| found.iter().sum::<Vector3<f64>>() / found.len() as f64)
    }
}

/// A structure read and measured once, so a parent can be compared with many candidates.
#[derive(Debug, Clone)]
pub struct PreparedStructure {
    residues: Vec<Residue>,
    /// For each residue of the protein in file order (the `res_idx` the interaction network
    /// gives), its index in `residues`; `None` for one without a C-alpha. Chain and number are
    /// not enough: an insertion code (`184A` beside `184`) shares the number.
    by_network_idx: Vec<Option<usize>>,
    confidence: ConfidenceSource,
    /// Factor from the B-factor column to pLDDT (100 for 0–1 files).
    plddt_scale: f64,
    network: InteractionNetwork,
}

impl PreparedStructure {
    /// Read a PDB or mmCIF file's bytes (gzipped or not). `hint` is the file name, for format
    /// detection.
    pub fn from_bytes(
        bytes: &[u8],
        hint: Option<&str>,
        chains: Option<&[String]>,
    ) -> Result<Self, CoreError> {
        Self::from_bytes_with_source(bytes, hint, chains, None)
    }

    /// [`Self::from_bytes`], with what the B-factor column holds given rather than detected
    /// when `source` is `Some` (see [`crate::pae::predicted_beside`]).
    pub fn from_bytes_with_source(
        bytes: &[u8],
        hint: Option<&str>,
        chains: Option<&[String]>,
        source: Option<ConfidenceSource>,
    ) -> Result<Self, CoreError> {
        let loaded = load_structure_bytes(bytes, hint)?;
        let mut protein = protein_heavy_atoms(&loaded.pdb);
        if let Some(keep) = chains {
            protein.remove_chains_by(|c| !keep.iter().any(|k| k == c.id().trim()));
        }
        let mut residues = Vec::new();
        let mut by_network_idx = Vec::new();
        for chain in protein.chains() {
            let chain_id = chain.id().trim().to_string();
            for residue in chain.residues() {
                // Every residue takes a network index, kept or not, so the two stay aligned.
                by_network_idx.push(None);
                let atoms: Vec<(String, Vector3<f64>)> = residue
                    .atoms()
                    .map(|a| {
                        (
                            a.name().trim().to_uppercase(),
                            Vector3::new(a.x(), a.y(), a.z()),
                        )
                    })
                    .collect();
                let Some(ca_atom) = residue.atoms().find(|a| a.name().trim() == "CA") else {
                    continue;
                };
                *by_network_idx.last_mut().expect("pushed above") = Some(residues.len());
                residues.push(Residue {
                    id: (
                        chain_id.clone(),
                        residue.serial_number(),
                        residue.insertion_code().map(str::to_string),
                    ),
                    name: residue
                        .name()
                        .map(|n| n.trim().to_uppercase())
                        .unwrap_or_else(|| "UNK".into()),
                    ca: Vector3::new(ca_atom.x(), ca_atom.y(), ca_atom.z()),
                    b_factor: ca_atom.b_factor(),
                    atoms,
                });
            }
        }
        if residues.len() < MIN_PAIRED_RESIDUES {
            return Err(CoreError::AnalysisError(match chains {
                Some(c) => format!(
                    "{} protein residue(s) with a C-alpha in chain(s) {}",
                    residues.len(),
                    c.join(",")
                ),
                None => format!("{} protein residue(s) with a C-alpha", residues.len()),
            }));
        }
        let b: Vec<f64> = residues.iter().map(|r| r.b_factor).collect();
        let confidence =
            source.unwrap_or_else(|| detect_confidence_source(&loaded.header_preview, &b));
        let max_b = b.iter().copied().fold(f64::MIN, f64::max);
        let plddt_scale = if max_b > 0.0 && max_b <= 1.0 {
            100.0
        } else {
            1.0
        };
        let network = compute_interaction_network(&protein);
        Ok(Self {
            residues,
            by_network_idx,
            confidence,
            plddt_scale,
            network,
        })
    }

    pub fn residue_count(&self) -> usize {
        self.residues.len()
    }

    pub fn confidence_source(&self) -> ConfidenceSource {
        self.confidence
    }

    fn confidence_at(&self, i: usize) -> Option<f64> {
        self.confidence
            .is_predicted()
            .then(|| self.residues[i].b_factor * self.plddt_scale)
    }

    /// The index in `residues` of the network's residue `res_idx`.
    fn index_of(&self, res_idx: usize) -> Option<usize> {
        self.by_network_idx.get(res_idx).copied().flatten()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionKind {
    HydrogenBond,
    SaltBridge,
    PiStacking,
    CationPi,
}

impl InteractionKind {
    pub const ALL: [InteractionKind; 4] = [
        InteractionKind::HydrogenBond,
        InteractionKind::SaltBridge,
        InteractionKind::PiStacking,
        InteractionKind::CationPi,
    ];

    /// Short plural label for captions and tables.
    /// The distance cut-off [`compute_interaction_network`] applies (Å): donor–acceptor for
    /// hydrogen bonds, the closest charged pair for salt bridges, ring centroids for π-stacks,
    /// cation to ring centroid for cation–π.
    pub fn cutoff(self) -> f64 {
        match self {
            InteractionKind::HydrogenBond => 3.5,
            InteractionKind::SaltBridge => 4.0,
            InteractionKind::PiStacking => 6.5,
            InteractionKind::CationPi => 6.0,
        }
    }

    pub fn label(self, n: usize) -> &'static str {
        match (self, n == 1) {
            (InteractionKind::HydrogenBond, true) => "H-bond",
            (InteractionKind::HydrogenBond, false) => "H-bonds",
            (InteractionKind::SaltBridge, true) => "salt bridge",
            (InteractionKind::SaltBridge, false) => "salt bridges",
            (InteractionKind::PiStacking, true) => "π-stack",
            (InteractionKind::PiStacking, false) => "π-stacks",
            (InteractionKind::CationPi, true) => "cation-π",
            (InteractionKind::CationPi, false) => "cation-π",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Change {
    Lost,
    Gained,
}

/// One end of an interaction, as named in the structure that has it. `atom` is `None` for an
/// aromatic ring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub chain_id: String,
    pub res_seq: isize,
    pub res_name: String,
    pub atom: Option<String>,
    /// Position in the candidate's frame (the parent is superposed onto the candidate).
    pub position: [f64; 3],
    /// The same residue in the candidate (chain, number), when it has one: for a lost
    /// interaction the endpoint above is named in the parent, whose numbering may differ.
    pub candidate_residue: Option<(String, isize)>,
    /// The same atom (or ring centre) in the other structure, in the candidate's frame, when
    /// it exists there: where a lost interaction's partner is now, or where a gained one's was.
    pub other_position: Option<[f64; 3]>,
}

impl Endpoint {
    pub fn label(&self) -> String {
        let one = crate::qc::one_letter(&self.res_name);
        match &self.atom {
            Some(a) => format!("{one}{}:{a}", self.res_seq),
            None => format!("{one}{} ring", self.res_seq),
        }
    }
}

/// An interaction present in one structure only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionChange {
    pub kind: InteractionKind,
    pub change: Change,
    pub a: Endpoint,
    pub b: Endpoint,
    /// Distance in the structure that has the interaction, Å (heavy atoms; ring centroids for
    /// π interactions; the closest charged pair for salt bridges).
    pub distance: f64,
    /// The same atoms' distance in the other structure, when both are present there.
    pub other_distance: Option<f64>,
    /// Atoms of this interaction absent from the other structure because their residue is a
    /// changed site, inserted or deleted ("ASP 39 OD1").
    pub removed: Vec<String>,
    /// Atoms absent from the other structure although their residue is the same there: a side
    /// chain the other file did not model. Such a change is never [clear](CLEAR_MARGIN).
    pub unmodelled: Vec<String>,
    /// Between two chains.
    pub interface: bool,
    /// Distance (Å) from the nearer endpoint to the nearest atom of a changed site, in the
    /// candidate's frame; `None` when nothing changed.
    pub site_distance: Option<f64>,
    /// Index into [`VariantDiff::sites`] of that nearest site.
    pub nearest_site: Option<usize>,
    pub local: bool,
    /// See [`CLEAR_MARGIN`].
    pub clear: bool,
}

/// A paired residue whose amino acid differs between parent and candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangedSite {
    /// `D39A`: parent residue, parent number, candidate residue.
    pub notation: String,
    pub parent_chain_id: String,
    pub parent_res_seq: isize,
    pub parent_name: String,
    pub candidate_chain_id: String,
    pub candidate_res_seq: isize,
    pub candidate_name: String,
    /// C-alpha RMSD (Å) of the paired residues within the local radius, after the global
    /// superposition.
    pub local_ca_rmsd: f64,
    /// The predictor's confidence (pLDDT) at this residue, when the file carries one.
    pub parent_confidence: Option<f64>,
    pub candidate_confidence: Option<f64>,
    /// Which [`VariantDiff::groups`] entry the site belongs to.
    pub group: usize,
    /// Heavy atoms in the candidate's frame, for drawing.
    pub candidate_atoms: Vec<(String, [f64; 3])>,
    pub parent_atoms: Vec<(String, [f64; 3])>,
}

/// Interaction counts by kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindCounts {
    pub hydrogen_bonds: usize,
    pub salt_bridges: usize,
    pub pi_stacking: usize,
    pub cation_pi: usize,
}

impl KindCounts {
    fn bump(&mut self, kind: InteractionKind) {
        match kind {
            InteractionKind::HydrogenBond => self.hydrogen_bonds += 1,
            InteractionKind::SaltBridge => self.salt_bridges += 1,
            InteractionKind::PiStacking => self.pi_stacking += 1,
            InteractionKind::CationPi => self.cation_pi += 1,
        }
    }

    pub fn get(&self, kind: InteractionKind) -> usize {
        match kind {
            InteractionKind::HydrogenBond => self.hydrogen_bonds,
            InteractionKind::SaltBridge => self.salt_bridges,
            InteractionKind::PiStacking => self.pi_stacking,
            InteractionKind::CationPi => self.cation_pi,
        }
    }

    pub fn total(&self) -> usize {
        self.hydrogen_bonds + self.salt_bridges + self.pi_stacking + self.cation_pi
    }
}

/// The counts a ranking can sort on.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffSummary {
    pub sites: usize,
    pub lost_local: KindCounts,
    pub gained_local: KindCounts,
    pub lost_distal: KindCounts,
    pub gained_distal: KindCounts,
    /// Local changes that are not [clear](CLEAR_MARGIN); not included in the counts above.
    pub marginal_local: usize,
    /// Local changes between chains.
    pub interface_lost_local: usize,
    pub interface_gained_local: usize,
    pub kept: KindCounts,
    /// Lowest candidate confidence over the changed sites, when the candidate has one.
    pub min_site_confidence: Option<f64>,
    /// Mean candidate − parent confidence over the changed sites, when both have one.
    pub mean_site_confidence_change: Option<f64>,
}

/// Rigid transform taking parent coordinates into the candidate's frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantDiff {
    /// How residues were paired (`residue id`, `residue number`, `sequence alignment`).
    pub pairing: String,
    pub paired: usize,
    pub parent_residues: usize,
    pub candidate_residues: usize,
    /// Candidate residues with no partner in the parent, and parent residues with none in the
    /// candidate.
    pub inserted: usize,
    pub deleted: usize,
    /// Fraction of paired residues with the same amino acid. A redesign can be well below 1;
    /// a value near random (~0.1) means the two are not parent and candidate at all.
    pub identity: f64,
    /// Paired residues over the larger of the two structures. Identity alone can mislead: an
    /// alignment of two unrelated proteins pairs few residues, mostly identical ones.
    pub coverage: f64,
    /// C-alpha RMSD (Å) over all paired residues after superposition.
    pub global_ca_rmsd: f64,
    pub parent_confidence: ConfidenceSource,
    pub candidate_confidence: ConfidenceSource,
    pub local_radius: f64,
    pub sites: Vec<ChangedSite>,
    /// Sites within the local radius of each other (single linkage on C-alphas), each list in
    /// candidate order, groups ordered by local interaction changes, most first.
    pub groups: Vec<Vec<usize>>,
    pub changes: Vec<InteractionChange>,
    pub summary: DiffSummary,
    pub parent_to_candidate: Transform,
}

/// Residue key of one interaction endpoint: a parent residue (paired or not), or a candidate
/// residue with no partner in the parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ResKey {
    Parent(usize),
    CandidateOnly(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    kind: InteractionKind,
    a: ResKey,
    /// Canonical atom name, for hydrogen bonds only.
    a_atom: Option<String>,
    b: ResKey,
    b_atom: Option<String>,
}

/// One interaction of one structure, located on its residues.
#[derive(Debug, Clone)]
struct Located {
    kind: InteractionKind,
    a: usize,
    a_atom: Option<String>,
    b: usize,
    b_atom: Option<String>,
    distance: f64,
}

/// ASP OD1/OD2, GLU OE1/OE2, ARG NH1/NH2: the same atom chemically, named either way round by
/// different files. (ASN/GLN/HIS amide and ring flips are *not* equivalent and are left alone.)
fn equivalent_atom(res: &str, atom: &str) -> Option<&'static str> {
    match (res, atom) {
        ("ASP", "OD1") => Some("OD2"),
        ("ASP", "OD2") => Some("OD1"),
        ("GLU", "OE1") => Some("OE2"),
        ("GLU", "OE2") => Some("OE1"),
        ("ARG", "NH1") => Some("NH2"),
        ("ARG", "NH2") => Some("NH1"),
        _ => None,
    }
}

fn canonical_atom(res: &str, atom: &str) -> String {
    match (res, atom) {
        ("ASP", "OD2") => "OD1".into(),
        ("GLU", "OE2") => "OE1".into(),
        ("ARG", "NH2") => "NH1".into(),
        _ => atom.to_string(),
    }
}

/// The atoms [`compute_interaction_network`] takes as each residue's aromatic ring.
fn ring_atoms(res: &str) -> Option<&'static [&'static str]> {
    match res {
        "PHE" | "TYR" => Some(&["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
        "TRP" => Some(&["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]),
        "HIS" => Some(&["CG", "ND1", "CD2", "CE1", "NE2"]),
        _ => None,
    }
}

/// Every interaction of `s`, on residue indices of `s`. Interactions whose residues are not in
/// `s.residues` (no C-alpha) are skipped.
fn locate(s: &PreparedStructure) -> Vec<Located> {
    let mut out = Vec::new();
    let n = &s.network;
    for h in &n.hbonds {
        if let (Some(a), Some(b)) = (s.index_of(h.donor_res_idx), s.index_of(h.acceptor_res_idx)) {
            out.push(Located {
                kind: InteractionKind::HydrogenBond,
                a,
                a_atom: Some(h.donor_atom_name.clone()),
                b,
                b_atom: Some(h.acceptor_atom_name.clone()),
                distance: h.distance,
            });
        }
    }
    for sb in &n.salt_bridges {
        if let (Some(a), Some(b)) = (s.index_of(sb.cation_res_idx), s.index_of(sb.anion_res_idx)) {
            out.push(Located {
                kind: InteractionKind::SaltBridge,
                a,
                a_atom: Some(sb.cation_atom_name.clone()),
                b,
                b_atom: Some(sb.anion_atom_name.clone()),
                distance: sb.distance,
            });
        }
    }
    for p in &n.pi_pi_stacks {
        if let (Some(a), Some(b)) = (s.index_of(p.ring1_res_idx), s.index_of(p.ring2_res_idx)) {
            out.push(Located {
                kind: InteractionKind::PiStacking,
                a,
                a_atom: None,
                b,
                b_atom: None,
                distance: p.centroid_distance,
            });
        }
    }
    for c in &n.cation_pi_interactions {
        if let (Some(a), Some(b)) = (s.index_of(c.cation_res_idx), s.index_of(c.ring_res_idx)) {
            out.push(Located {
                kind: InteractionKind::CationPi,
                a,
                a_atom: Some(c.cation_atom_name.clone()),
                b,
                b_atom: None,
                distance: c.distance_to_centroid,
            });
        }
    }
    out
}

/// The diff key of `l`. Hydrogen bonds are keyed by atom; salt bridges, π-stacks and cation–π
/// by residue (a salt bridge between two residues is one finding however many charged atom
/// pairs make it). π-stacking is symmetric, so its residues are ordered.
fn key_of(l: &Located, s: &PreparedStructure, res_key: impl Fn(usize) -> ResKey) -> Key {
    let (a, b) = (res_key(l.a), res_key(l.b));
    match l.kind {
        InteractionKind::HydrogenBond => Key {
            kind: l.kind,
            a,
            a_atom: l
                .a_atom
                .as_deref()
                .map(|x| canonical_atom(&s.residues[l.a].name, x)),
            b,
            b_atom: l
                .b_atom
                .as_deref()
                .map(|x| canonical_atom(&s.residues[l.b].name, x)),
        },
        InteractionKind::PiStacking => Key {
            kind: l.kind,
            a: a.min(b),
            a_atom: None,
            b: a.max(b),
            b_atom: None,
        },
        InteractionKind::SaltBridge | InteractionKind::CationPi => Key {
            kind: l.kind,
            a,
            a_atom: None,
            b,
            b_atom: None,
        },
    }
}

/// Interactions of `s` by key; where several share a key (the charged atom pairs of one salt
/// bridge), the shortest represents them.
fn keyed(s: &PreparedStructure, res_key: impl Fn(usize) -> ResKey) -> HashMap<Key, Located> {
    let mut map: HashMap<Key, Located> = HashMap::new();
    for l in locate(s) {
        let k = key_of(&l, s, &res_key);
        match map.get(&k) {
            Some(prev) if prev.distance <= l.distance => {}
            _ => {
                map.insert(k, l);
            }
        }
    }
    map
}

fn to_arr(v: Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// Position of one endpoint: the atom, or the ring centroid.
fn endpoint_position(r: &Residue, atom: Option<&str>) -> Option<Vector3<f64>> {
    match atom {
        Some(a) => r.atom_or_equivalent(a),
        None => r.ring_centroid(),
    }
}

/// Compare `candidate` with `parent`.
pub fn diff(
    parent: &PreparedStructure,
    candidate: &PreparedStructure,
    opts: &DiffOptions,
) -> Result<VariantDiff, CoreError> {
    let cand_ids: Vec<ResidueId> = candidate.residues.iter().map(|r| r.id.clone()).collect();
    let cand_names: Vec<String> = candidate.residues.iter().map(|r| r.name.clone()).collect();
    let par_ids: Vec<ResidueId> = parent.residues.iter().map(|r| r.id.clone()).collect();
    let par_names: Vec<String> = parent.residues.iter().map(|r| r.name.clone()).collect();
    let (pairs, pairing) = pair_best(&cand_ids, &cand_names, &par_ids, &par_names);
    if pairs.len() < MIN_PAIRED_RESIDUES {
        return Err(CoreError::AnalysisError(format!(
            "only {} residue(s) of the candidate could be paired with the parent (tried residue \
             ids, residue numbers and a sequence alignment); is it the same protein?",
            pairs.len()
        )));
    }
    let mut cand_to_par: HashMap<usize, usize> = HashMap::with_capacity(pairs.len());
    let mut par_to_cand: HashMap<usize, usize> = HashMap::with_capacity(pairs.len());
    for &(c, p) in &pairs {
        cand_to_par.insert(c, p);
        par_to_cand.insert(p, c);
    }

    // Parent onto candidate: the candidate keeps its own frame, as in the viewer.
    let p_ca: Vec<Vector3<f64>> = pairs.iter().map(|&(_, p)| parent.residues[p].ca).collect();
    let c_ca: Vec<Vector3<f64>> = pairs
        .iter()
        .map(|&(c, _)| candidate.residues[c].ca)
        .collect();
    let sup = compute_kabsch_superposition(&p_ca, &c_ca)?;
    let rot: Matrix3<f64> = sup.rotation;
    let tr = sup.translation;
    let move_p = |v: Vector3<f64>| rot * v + tr;

    // Changed sites, in candidate order.
    let mut site_pairs: Vec<(usize, usize)> = pairs
        .iter()
        .copied()
        .filter(|&(c, p)| candidate.residues[c].name != parent.residues[p].name)
        .collect();
    site_pairs.sort_unstable();

    let radius = opts.local_radius;
    let mut sites: Vec<ChangedSite> = Vec::with_capacity(site_pairs.len());
    for &(c, p) in &site_pairs {
        let cr = &candidate.residues[c];
        let pr = &parent.residues[p];
        let near: Vec<f64> = pairs
            .iter()
            .filter(|&&(ci, _)| (candidate.residues[ci].ca - cr.ca).norm() <= radius)
            .map(|&(ci, pi)| {
                (candidate.residues[ci].ca - move_p(parent.residues[pi].ca)).norm_squared()
            })
            .collect();
        let local = if near.is_empty() {
            0.0
        } else {
            (near.iter().sum::<f64>() / near.len() as f64).sqrt()
        };
        sites.push(ChangedSite {
            notation: format!(
                "{}{}{}",
                crate::qc::one_letter(&pr.name),
                pr.id.1,
                crate::qc::one_letter(&cr.name)
            ),
            parent_chain_id: pr.id.0.clone(),
            parent_res_seq: pr.id.1,
            parent_name: pr.name.clone(),
            candidate_chain_id: cr.id.0.clone(),
            candidate_res_seq: cr.id.1,
            candidate_name: cr.name.clone(),
            local_ca_rmsd: local,
            parent_confidence: parent.confidence_at(p),
            candidate_confidence: candidate.confidence_at(c),
            group: 0,
            candidate_atoms: cr
                .atoms
                .iter()
                .map(|(n, v)| (n.clone(), to_arr(*v)))
                .collect(),
            parent_atoms: pr
                .atoms
                .iter()
                .map(|(n, v)| (n.clone(), to_arr(move_p(*v))))
                .collect(),
        });
    }

    let site_parents: HashSet<usize> = site_pairs.iter().map(|&(_, p)| p).collect();
    let site_candidates: HashSet<usize> = site_pairs.iter().map(|&(c, _)| c).collect();

    // Interaction diff, keyed in parent numbering.
    let par_keyed = keyed(parent, ResKey::Parent);
    let cand_keyed = keyed(candidate, |c| match cand_to_par.get(&c) {
        Some(&p) => ResKey::Parent(p),
        None => ResKey::CandidateOnly(c),
    });

    // Every changed-site atom, both versions, in the candidate's frame.
    let site_atoms: Vec<(usize, Vector3<f64>)> = site_pairs
        .iter()
        .enumerate()
        .flat_map(|(si, &(c, p))| {
            let cand = candidate.residues[c]
                .atoms
                .iter()
                .map(move |&(_, v)| (si, v));
            let par = parent.residues[p]
                .atoms
                .iter()
                .map(move |&(_, v)| (si, move_p(v)));
            cand.chain(par)
        })
        .collect();
    let nearest_site = |pts: &[Vector3<f64>]| -> Option<(f64, usize)> {
        site_atoms
            .iter()
            .flat_map(|&(si, s)| pts.iter().map(move |p| ((p - s).norm(), si)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
    };

    let mut changes: Vec<InteractionChange> = Vec::new();
    let mut kept = KindCounts::default();
    for (k, l) in &par_keyed {
        if cand_keyed.contains_key(k) {
            kept.bump(l.kind);
            continue;
        }
        let (ra, rb) = (&parent.residues[l.a], &parent.residues[l.b]);
        let (Some(pa), Some(pb)) = (
            endpoint_position(ra, l.a_atom.as_deref()),
            endpoint_position(rb, l.b_atom.as_deref()),
        ) else {
            continue;
        };
        let (pa, pb) = (move_p(pa), move_p(pb));
        let other = |r: usize, atom: Option<&str>| {
            par_to_cand
                .get(&r)
                .and_then(|&c| endpoint_position(&candidate.residues[c], atom))
        };
        let (oa, ob) = (
            other(l.a, l.a_atom.as_deref()),
            other(l.b, l.b_atom.as_deref()),
        );
        let changed = |r: usize| !par_to_cand.contains_key(&r) || site_parents.contains(&r);
        let in_cand = |r: usize| {
            par_to_cand.get(&r).map(|&c| {
                let id = &candidate.residues[c].id;
                (id.0.clone(), id.1)
            })
        };
        changes.push(build_change(
            l,
            Change::Lost,
            (ra, pa, oa, changed(l.a), in_cand(l.a)),
            (rb, pb, ob, changed(l.b), in_cand(l.b)),
            &nearest_site,
            radius,
        ));
    }
    for (k, l) in &cand_keyed {
        if par_keyed.contains_key(k) {
            continue;
        }
        let (ra, rb) = (&candidate.residues[l.a], &candidate.residues[l.b]);
        let (Some(pa), Some(pb)) = (
            endpoint_position(ra, l.a_atom.as_deref()),
            endpoint_position(rb, l.b_atom.as_deref()),
        ) else {
            continue;
        };
        let other = |r: usize, atom: Option<&str>| {
            cand_to_par
                .get(&r)
                .and_then(|&p| endpoint_position(&parent.residues[p], atom))
                .map(move_p)
        };
        let (oa, ob) = (
            other(l.a, l.a_atom.as_deref()),
            other(l.b, l.b_atom.as_deref()),
        );
        let changed = |r: usize| !cand_to_par.contains_key(&r) || site_candidates.contains(&r);
        let own = |r: &Residue| Some((r.id.0.clone(), r.id.1));
        changes.push(build_change(
            l,
            Change::Gained,
            (ra, pa, oa, changed(l.a), own(ra)),
            (rb, pb, ob, changed(l.b), own(rb)),
            &nearest_site,
            radius,
        ));
    }
    // Deterministic order: local first, nearest first, then by residue.
    changes.sort_by(|x, y| {
        y.local
            .cmp(&x.local)
            .then(
                x.site_distance
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&y.site_distance.unwrap_or(f64::INFINITY)),
            )
            .then((x.a.chain_id.as_str(), x.a.res_seq).cmp(&(y.a.chain_id.as_str(), y.a.res_seq)))
            .then((x.b.chain_id.as_str(), x.b.res_seq).cmp(&(y.b.chain_id.as_str(), y.b.res_seq)))
            .then(x.a.atom.cmp(&y.a.atom))
            .then(x.b.atom.cmp(&y.b.atom))
            .then((x.kind as u8).cmp(&(y.kind as u8)))
            .then((x.change as u8).cmp(&(y.change as u8)))
    });

    // Site groups: single linkage on candidate C-alphas.
    let n_sites = site_pairs.len();
    let mut group_of: Vec<usize> = (0..n_sites).collect();
    fn find(g: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while g[r] != r {
            r = g[r];
        }
        let mut j = i;
        while g[j] != r {
            let next = g[j];
            g[j] = r;
            j = next;
        }
        r
    }
    for i in 0..n_sites {
        for j in (i + 1)..n_sites {
            let (ci, cj) = (site_pairs[i].0, site_pairs[j].0);
            if (candidate.residues[ci].ca - candidate.residues[cj].ca).norm() <= radius {
                let (ri, rj) = (find(&mut group_of, i), find(&mut group_of, j));
                if ri != rj {
                    group_of[rj.max(ri)] = ri.min(rj);
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut root_to_group: HashMap<usize, usize> = HashMap::new();
    for i in 0..n_sites {
        let r = find(&mut group_of, i);
        let g = *root_to_group.entry(r).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    let weight = |g: &[usize]| -> usize {
        let set: HashSet<usize> = g.iter().copied().collect();
        changes
            .iter()
            .filter(|ch| ch.local && ch.nearest_site.is_some_and(|s| set.contains(&s)))
            .count()
    };
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by(|&x, &y| weight(&groups[y]).cmp(&weight(&groups[x])).then(x.cmp(&y)));
    let groups: Vec<Vec<usize>> = order.into_iter().map(|g| groups[g].clone()).collect();
    for (gi, g) in groups.iter().enumerate() {
        for &s in g {
            sites[s].group = gi;
        }
    }

    let mut summary = DiffSummary {
        sites: sites.len(),
        kept,
        ..DiffSummary::default()
    };
    for ch in &changes {
        if ch.local && !ch.clear {
            summary.marginal_local += 1;
            continue;
        }
        match (ch.change, ch.local) {
            (Change::Lost, true) => {
                summary.lost_local.bump(ch.kind);
                summary.interface_lost_local += ch.interface as usize;
            }
            (Change::Gained, true) => {
                summary.gained_local.bump(ch.kind);
                summary.interface_gained_local += ch.interface as usize;
            }
            (Change::Lost, false) => summary.lost_distal.bump(ch.kind),
            (Change::Gained, false) => summary.gained_distal.bump(ch.kind),
        }
    }
    summary.min_site_confidence = sites
        .iter()
        .filter_map(|s| s.candidate_confidence)
        .min_by(f64::total_cmp);
    let deltas: Vec<f64> = sites
        .iter()
        .filter_map(|s| Some(s.candidate_confidence? - s.parent_confidence?))
        .collect();
    summary.mean_site_confidence_change =
        (!deltas.is_empty()).then(|| deltas.iter().sum::<f64>() / deltas.len() as f64);

    Ok(VariantDiff {
        pairing: pairing.to_string(),
        paired: pairs.len(),
        parent_residues: parent.residues.len(),
        candidate_residues: candidate.residues.len(),
        inserted: candidate.residues.len() - pairs.len(),
        deleted: parent.residues.len() - pairs.len(),
        identity: 1.0 - site_pairs.len() as f64 / pairs.len() as f64,
        coverage: pairs.len() as f64 / parent.residues.len().max(candidate.residues.len()) as f64,
        global_ca_rmsd: sup.rmsd,
        parent_confidence: parent.confidence,
        candidate_confidence: candidate.confidence,
        local_radius: radius,
        sites,
        groups,
        changes,
        summary,
        parent_to_candidate: Transform {
            rotation: [
                [rot[(0, 0)], rot[(0, 1)], rot[(0, 2)]],
                [rot[(1, 0)], rot[(1, 1)], rot[(1, 2)]],
                [rot[(2, 0)], rot[(2, 1)], rot[(2, 2)]],
            ],
            translation: to_arr(tr),
        },
    })
}

/// One side of an interaction: the residue in the structure that has it, its endpoint position
/// in the candidate's frame, the same endpoint in the other structure (candidate's frame),
/// whether that residue changed (a site, an insertion or a deletion), and the residue's identity
/// in the candidate.
type Side<'a> = (
    &'a Residue,
    Vector3<f64>,
    Option<Vector3<f64>>,
    bool,
    Option<(String, isize)>,
);

fn build_change(
    l: &Located,
    change: Change,
    a: Side<'_>,
    b: Side<'_>,
    nearest_site: &impl Fn(&[Vector3<f64>]) -> Option<(f64, usize)>,
    radius: f64,
) -> InteractionChange {
    let endpoint = |side: &Side<'_>, atom: &Option<String>| Endpoint {
        chain_id: side.0.id.0.clone(),
        res_seq: side.0.id.1,
        res_name: side.0.name.clone(),
        atom: atom.clone(),
        position: to_arr(side.1),
        candidate_residue: side.4.clone(),
        other_position: side.2.map(to_arr),
    };
    let describe = |r: &Residue, atom: &Option<String>| match atom {
        Some(x) => format!("{} {} {x}", r.name, r.id.1),
        None => format!("{} {} ring", r.name, r.id.1),
    };
    let (mut removed, mut unmodelled) = (Vec::new(), Vec::new());
    for (side, atom) in [(&a, &l.a_atom), (&b, &l.b_atom)] {
        if side.2.is_none() {
            let list = if side.3 {
                &mut removed
            } else {
                &mut unmodelled
            };
            list.push(describe(side.0, atom));
        }
    }
    let other_distance = match (a.2, b.2) {
        (Some(x), Some(y)) => Some((x - y).norm()),
        _ => None,
    };
    let near = nearest_site(&[a.1, b.1]);
    let clear = is_clear(l.kind, &removed, &unmodelled, other_distance);
    InteractionChange {
        kind: l.kind,
        change,
        a: endpoint(&a, &l.a_atom),
        b: endpoint(&b, &l.b_atom),
        distance: l.distance,
        other_distance,
        removed,
        unmodelled,
        interface: a.0.id.0 != b.0.id.0,
        site_distance: near.map(|n| n.0),
        nearest_site: near.map(|n| n.1),
        local: near.is_some_and(|n| n.0 <= radius),
        clear,
    }
}

/// See [`CLEAR_MARGIN`].
fn is_clear(
    kind: InteractionKind,
    removed: &[String],
    unmodelled: &[String],
    other_distance: Option<f64>,
) -> bool {
    if !removed.is_empty() {
        return true;
    }
    unmodelled.is_empty() && other_distance.is_some_and(|d| d >= kind.cutoff() + CLEAR_MARGIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTC: &str = include_str!("../tests/data/2ptc_EI.pdb");

    fn prepared(text: &str) -> PreparedStructure {
        PreparedStructure::from_bytes(text.as_bytes(), Some("x.pdb"), None).unwrap()
    }

    /// Rewrite the PDB text: residue `res_seq` of `chain` becomes `new_name`, keeping only the
    /// atoms in `keep` (a truncation such as K→A keeps N, CA, C, O, CB). A residue with an
    /// insertion code (`184A`) is a different residue and is left alone.
    fn mutate(text: &str, chain: char, res_seq: isize, new_name: &str, keep: &[&str]) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let is_atom = line.starts_with("ATOM") || line.starts_with("HETATM");
            if is_atom
                && line.len() > 26
                && line.as_bytes()[21] as char == chain
                && line[22..26].trim().parse::<isize>().ok() == Some(res_seq)
                && line.as_bytes()[26] == b' '
            {
                let atom = line[12..16].trim();
                if !keep.contains(&atom) {
                    continue;
                }
                out.push_str(&line[..17]);
                out.push_str(&format!("{new_name:>3}"));
                out.push_str(&line[20..]);
            } else {
                out.push_str(line);
            }
            out.push('\n');
        }
        out
    }

    /// Swap two atom names within one residue (a file naming OD1/OD2 the other way round).
    fn swap_names(text: &str, chain: char, res_seq: isize, x: &str, y: &str) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let hit = line.starts_with("ATOM")
                && line.len() > 26
                && line.as_bytes()[21] as char == chain
                && line[22..26].trim().parse::<isize>().ok() == Some(res_seq);
            if hit && line[12..16].trim() == x {
                out.push_str(&format!(
                    "{}{:<4}{}",
                    &line[..12],
                    format!(" {y}"),
                    &line[16..]
                ));
            } else if hit && line[12..16].trim() == y {
                out.push_str(&format!(
                    "{}{:<4}{}",
                    &line[..12],
                    format!(" {x}"),
                    &line[16..]
                ));
            } else {
                out.push_str(line);
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn a_structure_against_itself_changes_nothing() {
        let p = prepared(PTC);
        let d = diff(&p, &p, &DiffOptions::default()).unwrap();
        assert!(d.sites.is_empty());
        assert!(d.changes.is_empty());
        assert!(d.global_ca_rmsd < 1e-6);
        assert_eq!(d.inserted + d.deleted, 0);
        assert!(d.summary.kept.total() > 0);
    }

    /// BPTI Lys15 sits in trypsin's S1 pocket. Truncating it to alanine must remove exactly the
    /// interactions that used its side chain beyond CB, report them as lost at the site and at
    /// the interface, with the atoms that are gone, and change nothing else.
    #[test]
    fn truncating_bpti_lys15_loses_its_side_chain_interactions_and_nothing_else() {
        let parent = prepared(PTC);
        let mutant_text = mutate(PTC, 'I', 15, "ALA", &["N", "CA", "C", "O", "CB"]);
        let cand = prepared(&mutant_text);
        let d = diff(&parent, &cand, &DiffOptions::default()).unwrap();

        assert_eq!(d.sites.len(), 1);
        assert_eq!(d.sites[0].notation, "K15A");
        assert_eq!(d.sites[0].parent_chain_id, "I");
        assert_eq!(d.groups, vec![vec![0]]);
        assert!(d.global_ca_rmsd < 1e-6, "coordinates were not moved");

        let side_chain = ["CG", "CD", "CE", "NZ"];
        let expected: Vec<_> = locate(&parent)
            .into_iter()
            .filter(|l| {
                let touches = |r: usize, atom: &Option<String>| {
                    let res = &parent.residues[r];
                    res.id.0 == "I"
                        && res.id.1 == 15
                        && atom.as_deref().is_none_or(|a| side_chain.contains(&a))
                };
                touches(l.a, &l.a_atom) || touches(l.b, &l.b_atom)
            })
            .collect();
        assert!(!expected.is_empty(), "Lys15 makes interactions in 2PTC");
        assert!(d.changes.iter().all(|c| c.change == Change::Lost));
        assert!(d.changes.iter().all(|c| c.local && c.clear));
        assert!(
            d.changes
                .iter()
                .all(|c| !c.removed.is_empty() && c.unmodelled.is_empty()),
            "every lost interaction names the atom that is gone"
        );
        assert!(d.changes.iter().any(|c| c.interface));
        assert_eq!(d.summary.lost_distal.total(), 0);
        assert_eq!(d.summary.gained_local.total(), 0);
        // One change per key: hydrogen bonds by atom, the rest by residue pair.
        let lost_hb = d
            .changes
            .iter()
            .filter(|c| c.kind == InteractionKind::HydrogenBond)
            .count();
        let expected_hb = expected
            .iter()
            .filter(|l| l.kind == InteractionKind::HydrogenBond)
            .count();
        assert_eq!(lost_hb, expected_hb);
    }

    /// Trypsin numbers by chymotrypsin, so 2PTC has Gly E184A written before Tyr E184. An
    /// interaction is placed on the residue that made it, not on whichever of the two shares its
    /// number: truncating Tyr184 loses its OH hydrogen bond to Lys159 O.
    #[test]
    fn a_residue_beside_an_insertion_code_keeps_its_interactions() {
        let parent = prepared(PTC);
        let tyr = |s: &PreparedStructure| {
            s.residues
                .iter()
                .position(|r| r.id == ("E".to_string(), 184, None))
                .unwrap()
        };
        assert_eq!(parent.residues[tyr(&parent)].name, "TYR");
        let hb_of_tyr_oh = locate(&parent)
            .into_iter()
            .filter(|l| {
                l.kind == InteractionKind::HydrogenBond
                    && ((l.a == tyr(&parent) && l.a_atom.as_deref() == Some("OH"))
                        || (l.b == tyr(&parent) && l.b_atom.as_deref() == Some("OH")))
            })
            .count();
        assert!(
            hb_of_tyr_oh > 0,
            "Tyr E184 OH makes a hydrogen bond in 2PTC"
        );

        let cand = prepared(&mutate(PTC, 'E', 184, "ALA", &["N", "CA", "C", "O", "CB"]));
        let d = diff(&parent, &cand, &DiffOptions::default()).unwrap();
        assert_eq!(d.sites.len(), 1);
        assert_eq!(d.sites[0].notation, "Y184A");
        let lost_oh = d
            .changes
            .iter()
            .filter(|c| {
                c.change == Change::Lost
                    && c.kind == InteractionKind::HydrogenBond
                    && [&c.a, &c.b]
                        .iter()
                        .any(|e| e.res_name == "TYR" && e.atom.as_deref() == Some("OH"))
            })
            .count();
        assert_eq!(lost_oh, hb_of_tyr_oh, "{:#?}", d.changes);
    }

    /// A failed design (every residue under pLDDT 30, no header) reads as B-factors on its own;
    /// told it is predicted (its predictor's files are beside it), its sites carry pLDDT.
    #[test]
    fn a_declared_source_overrides_the_values() {
        let low: String = PTC
            .lines()
            .filter(|l| l.starts_with("ATOM"))
            .map(|l| format!("{}{:6.2}{}\n", &l[..60], 28.0, &l[66..]))
            .collect();
        let alone = prepared(&low);
        assert!(
            !alone.confidence_source().is_predicted(),
            "the values alone"
        );
        let told = PreparedStructure::from_bytes_with_source(
            low.as_bytes(),
            Some("x.pdb"),
            None,
            Some(ConfidenceSource::Predicted),
        )
        .unwrap();
        assert!(told.confidence_source().is_predicted());
        assert_eq!(told.confidence_at(0), Some(28.0));
    }

    /// A file that names an aspartate's carboxylate oxygens the other way round is the same
    /// structure; the diff must not report the swap as lost and gained bonds.
    #[test]
    fn equivalent_atom_names_swapped_are_not_a_change() {
        let parent = prepared(PTC);
        let asp_with_hbond = locate(&parent)
            .into_iter()
            .find(|l| {
                l.kind == InteractionKind::HydrogenBond
                    && parent.residues[l.b].name == "ASP"
                    && matches!(l.b_atom.as_deref(), Some("OD1") | Some("OD2"))
            })
            .expect("2PTC has an H-bond to an aspartate carboxylate");
        let r = &parent.residues[asp_with_hbond.b];
        let chain = r.id.0.chars().next().unwrap();
        let swapped = swap_names(PTC, chain, r.id.1, "OD1", "OD2");
        let cand = prepared(&swapped);
        let d = diff(&parent, &cand, &DiffOptions::default()).unwrap();
        assert!(d.sites.is_empty());
        assert!(
            d.changes.is_empty(),
            "swap reported as changes: {:?}",
            d.changes
                .iter()
                .map(|c| (c.a.label(), c.b.label()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_change_is_clear_only_when_an_atom_is_gone_or_it_moved_well_past_the_cutoff() {
        let gone = vec!["ASP 39 OD1".to_string()];
        let hb = InteractionKind::HydrogenBond;
        assert!(is_clear(hb, &gone, &[], None));
        assert!(
            !is_clear(hb, &[], &gone, None),
            "an unmodelled side chain is not a finding"
        );
        assert!(is_clear(hb, &[], &[], Some(4.0)));
        assert!(!is_clear(hb, &[], &[], Some(3.6)), "3.4→3.6 is noise");
        assert!(!is_clear(hb, &[], &[], Some(2.9)), "angle flip only");
        assert!(is_clear(InteractionKind::SaltBridge, &[], &[], Some(6.6)));
        assert!(!is_clear(InteractionKind::SaltBridge, &[], &[], Some(4.3)));
    }

    #[test]
    fn restricting_chains_keeps_only_those_chains() {
        let p = PreparedStructure::from_bytes(PTC.as_bytes(), Some("x.pdb"), Some(&["I".into()]))
            .unwrap();
        assert!(p.residues.iter().all(|r| r.id.0 == "I"));
        assert!(
            p.residue_count() > 50 && p.residue_count() < 70,
            "BPTI is 58 residues"
        );
    }

    #[test]
    fn a_different_protein_says_how_little_it_shares() {
        let crn = include_str!("../tests/data/1crn.pdb");
        let p = prepared(PTC);
        let c = PreparedStructure::from_bytes(crn.as_bytes(), Some("c.pdb"), None).unwrap();
        // Alignment always pairs something; the diff must say how, and how little is shared.
        let d = diff(&p, &c, &DiffOptions::default()).unwrap();
        assert!(d.coverage < 0.3, "coverage {}", d.coverage);
    }
}
