pub mod atoms;
pub mod brand;
pub mod error;
pub mod geometry;
pub mod rasterizer;
pub mod terminal;
pub mod tui;
pub mod web;

use error::RenderError;
use geometry::mesh::{generate_cartoon_mesh, TriangleMesh};
use nalgebra::Vector3;
use proteus_core::structure::assign_secondary_structure;
use rasterizer::buffer::{ColorRGB, Framebuffer};
use rasterizer::camera::OrbitCamera;
use rasterizer::pipeline::Rasterizer;
use rasterizer::shader::ColorScheme;
use terminal::braille::BrailleRenderer;
use terminal::halfblock::HalfBlockRenderer;
use terminal::kitty::KittyRenderer;
use terminal::TerminalBackend;

/// Disulfide bond connecting two cysteine residues via their sulfur atoms.
#[derive(Debug, Clone)]
pub struct DisulfideBond {
    pub res1_idx: usize,
    pub res2_idx: usize,
    pub p1: Vector3<f32>,
    pub p2: Vector3<f32>,
}

/// Structural rendering bundle containing the ribbon mesh, optional disulfide bridges, and camera.
#[derive(Debug, Clone)]
pub struct StructureRenderData {
    pub ribbon_mesh: TriangleMesh,
    pub disulfide_mesh: Option<TriangleMesh>,
    pub camera: OrbitCamera,
    pub num_residues: usize,
    pub num_disulfides: usize,
    pub metrics: Option<proteus_core::models::BiophysicalMetrics>,
    pub plddts: Vec<f64>,
    /// Factor applied to the file's B-factor column to get `plddts` (100 for 0–1 files, else 1).
    pub plddt_scale: f64,
    pub ramachandran_points: Vec<(
        Option<f64>,
        Option<f64>,
        proteus_core::structure::RamachandranRegion,
    )>,
    /// Per residue of the ribbon, in mesh `residue_index` order: chain ID, residue number,
    /// insertion code and three-letter name. What the browser viewer's hover shows.
    pub residue_labels: Vec<ResidueLabel>,
    /// Eight-state DSSP string, one character per residue of the ribbon.
    pub dssp: String,
    /// Heavy atoms and bonds of the ribbon's residues and of the ligands (browser sticks).
    pub atoms: atoms::AtomTable,
    /// Ligands, cofactors and ions; ligand `k` is residue `num_residues + k` in `atoms`.
    pub ligands: Vec<atoms::Ligand>,
    /// The analysis's findings located on the ribbon: what the viewer can point at.
    pub annotations: atoms::Annotations,
    /// PAE and pTM from the predictor's side files, when the caller attached them.
    pub confidence: Option<proteus_core::pae::PredictionConfidence>,
    /// Scores from a table (a mutational scan, a variant-effect predictor), placed on the
    /// residues, when the caller attached them.
    pub scores: Option<AttachedScores>,
    /// A reference structure superposed onto this one, when the caller attached it.
    pub comparison: Option<Comparison>,
}

/// A reference structure moved onto this one: its ribbon, in this structure's frame, and how
/// far each residue sits from its partner.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub name: String,
    pub reference_mesh: TriangleMesh,
    /// C-alpha distance to the paired reference residue after superposition, per residue of
    /// this structure; `None` for an unpaired residue.
    pub deviation: Vec<Option<f64>>,
    pub stats: SuperpositionStats,
}

/// A score table placed on a structure, with the direction its colours run in.
#[derive(Debug, Clone, PartialEq)]
pub struct AttachedScores {
    pub scores: proteus_core::scores::ResidueScores,
    /// True when a high value is the bad end (pathogenicity, ΔΔG of destabilisation); false for
    /// fitness and log-likelihood ratios, where low is deleterious.
    pub higher_is_worse: bool,
}

/// Whether a score column's name says that high values are the damaging end. Fitness,
/// enrichment and language-model log-likelihood ratios are low-is-bad, the default.
pub fn higher_is_worse(column: &str) -> bool {
    let c = column.to_ascii_lowercase();
    [
        "pathogen",
        "ddg",
        "damag",
        "deleterious",
        "destabil",
        "risk",
    ]
    .iter()
    .any(|k| c.contains(k))
}

/// Identity of one residue of the ribbon.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidueLabel {
    pub chain: String,
    pub number: isize,
    pub insertion_code: Option<String>,
    pub name: String,
}

/// Detect disulfide bonds by checking CYS sulfur-sulfur proximity (1.7Å - 2.6Å).
pub fn extract_disulfide_bonds(pdb: &pdbtbx::PDB) -> Vec<DisulfideBond> {
    let mut cys_sulfurs = Vec::new();

    for (res_idx, residue) in pdb.residues().enumerate() {
        let res_name = residue.name().map_or("", |s| s.trim());
        if res_name == "CYS" {
            for atom in residue.atoms() {
                if atom.name().trim() == "SG" {
                    cys_sulfurs.push((
                        res_idx,
                        Vector3::new(atom.x() as f32, atom.y() as f32, atom.z() as f32),
                    ));
                    break;
                }
            }
        }
    }

    let mut bonds = Vec::new();
    let n = cys_sulfurs.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (cys_sulfurs[i].1 - cys_sulfurs[j].1).norm();
            if (1.70..=2.60).contains(&dist) {
                bonds.push(DisulfideBond {
                    res1_idx: cys_sulfurs[i].0,
                    res2_idx: cys_sulfurs[j].0,
                    p1: cys_sulfurs[i].1,
                    p2: cys_sulfurs[j].1,
                });
            }
        }
    }
    bonds
}

/// Generate a triangle mesh of covalent cylinders and sulfur spheres for all disulfide bonds.
pub fn generate_disulfide_mesh(bonds: &[DisulfideBond]) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();
    for bond in bonds {
        let cylinder = crate::geometry::mesh::generate_cylinder_mesh(
            bond.p1,
            bond.p2,
            0.22,
            8,
            bond.res1_idx,
            proteus_core::structure::SecondaryStructure::Coil,
            95.0,
        );
        mesh.merge(cylinder);

        let sphere1 = crate::geometry::mesh::generate_sphere_mesh(
            bond.p1,
            0.45,
            6,
            8,
            bond.res1_idx,
            proteus_core::structure::SecondaryStructure::Coil,
            95.0,
        );
        let sphere2 = crate::geometry::mesh::generate_sphere_mesh(
            bond.p2,
            0.45,
            6,
            8,
            bond.res2_idx,
            proteus_core::structure::SecondaryStructure::Coil,
            95.0,
        );
        mesh.merge(sphere1);
        mesh.merge(sphere2);
    }
    mesh
}

/// One C-alpha per protein residue (first altloc, no ions, no waters), its B-factor/pLDDT, and
/// whether a peptide bond connects it to the previous entry — the same normalisation
/// `proteus analyze` applies, so the ribbon shows what the metrics measured.
struct Trace {
    ca: Vec<Vector3<f64>>,
    plddts: Vec<f64>,
    breaks: Vec<bool>,
    /// Ribbon wide-axis per residue, from the backbone carbonyl (see `ribbon_guides`).
    guides: Vec<Option<Vector3<f64>>>,
    /// Residue identity per entry: chain ID, residue number, insertion code.
    ids: Vec<ResidueId>,
    /// Three-letter residue name per entry.
    names: Vec<String>,
    protein: pdbtbx::PDB,
    plddt_scale: f64,
}

/// Chain ID, residue number and insertion code: what identifies a residue across two files.
type ResidueId = (String, isize, Option<String>);

fn trace_of(pdb: &pdbtbx::PDB) -> Trace {
    let protein = proteus_core::io::protein_heavy_atoms(pdb);
    let backbone = proteus_core::backbone::extract_backbone(&protein);
    let mut t = Trace {
        ca: Vec::with_capacity(backbone.len()),
        plddts: Vec::with_capacity(backbone.len()),
        breaks: Vec::with_capacity(backbone.len()),
        guides: Vec::with_capacity(backbone.len()),
        ids: Vec::with_capacity(backbone.len()),
        names: Vec::with_capacity(backbone.len()),
        protein,
        plddt_scale: 1.0,
    };
    for r in &backbone {
        if let Some(ca) = r.ca {
            t.ca.push(ca);
            t.plddts.push(r.b_factor);
            t.breaks.push(r.chain_break_before);
            t.ids
                .push((r.chain_id.clone(), r.seq_num, r.insertion_code.clone()));
            t.names.push(r.name.clone());
            t.guides.push(match (r.c, r.o) {
                (Some(c), Some(o)) if (o - c).norm() > 1e-6 => Some((o - c).normalize()),
                _ => None,
            });
        }
    }
    // Carson & Bugg flip correction: a beta strand's pleat alternates the carbonyl up and down
    // residue by residue, so the raw direction reverses every step. Flipping each guide to agree
    // with the previous one turns that alternation into a smoothly twisting ribbon axis; without
    // it the ribbon would corkscrew 180 degrees per residue. Measured on the corpus this takes
    // the consecutive-guide angle in strands from a median 57 degrees to 21, the remainder being
    // the strand's real twist. A chain break restarts the chain of comparisons.
    let mut previous: Option<Vector3<f64>> = None;
    for i in 0..t.guides.len() {
        if t.breaks.get(i).copied().unwrap_or(false) {
            previous = None;
        }
        if let Some(g) = t.guides[i] {
            let corrected = match previous {
                Some(p) if g.dot(&p) < 0.0 => -g,
                _ => g,
            };
            t.guides[i] = Some(corrected);
            previous = Some(corrected);
        }
    }

    // Normalize pLDDT if in [0.0, 1.0] range (e.g., raw ESMFold outputs)
    let max_plddt = t.plddts.iter().copied().fold(f64::MIN, f64::max);
    if max_plddt <= 1.0 && max_plddt > 0.0 {
        for v in &mut t.plddts {
            *v *= 100.0;
        }
        t.plddt_scale = 100.0;
    }
    t
}

/// Cartoon mesh built segment by segment: the spline never crosses a chain break or a gap in
/// the model, so no tube is drawn between chains or across missing residues. Vertex residue
/// indices stay global so rainbow colouring runs over the whole structure.
fn segmented_cartoon_mesh(
    ca: &[Vector3<f64>],
    ss: &[proteus_core::structure::SecondaryStructure],
    plddts: &[f64],
    breaks: &[bool],
    guides: &[Option<Vector3<f64>>],
) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();
    let mut start = 0usize;
    let n = ca.len();
    for end in 1..=n {
        if end == n || breaks[end] {
            if end - start >= 2 {
                let mut part = generate_cartoon_mesh(
                    &ca[start..end],
                    &ss[start..end],
                    &plddts[start..end],
                    &guides[start..end],
                    4,
                );
                for v in &mut part.vertices {
                    v.residue_index += start;
                }
                mesh.merge(part);
            }
            start = end;
        }
    }
    mesh
}

/// Parse PDB/mmCIF text into the render bundle: ribbon mesh, disulfide mesh, camera, metrics.
pub fn parse_pdb_structure(pdb_content: &str) -> Result<StructureRenderData, RenderError> {
    let pdb = proteus_core::io::open_structure_bytes(pdb_content.as_bytes(), None)
        .map_err(|e| RenderError::PdbParse(e.to_string()))?;

    let trace = trace_of(&pdb);
    let ca_coords = trace.ca;
    let plddts = trace.plddts;

    if ca_coords.len() < 2 {
        return Err(RenderError::PdbParse(
            "PDB must contain at least 2 C-alpha residues for ribbon rendering".into(),
        ));
    }

    let ss_summary =
        assign_secondary_structure(&proteus_core::backbone::extract_backbone(&trace.protein));

    // Compute bounding sphere
    let n = ca_coords.len() as f64;
    let sum_pos: Vector3<f64> = ca_coords.iter().sum();
    let center_f64 = sum_pos / n;
    let center = Vector3::new(
        center_f64.x as f32,
        center_f64.y as f32,
        center_f64.z as f32,
    );

    let max_radius = ca_coords
        .iter()
        .map(|p| (p - center_f64).norm())
        .fold(0.0f64, f64::max) as f32;

    let ribbon_mesh = segmented_cartoon_mesh(
        &ca_coords,
        &ss_summary.assignment,
        &plddts,
        &trace.breaks,
        &trace.guides,
    );
    let ca_f32: Vec<Vector3<f32>> = ca_coords
        .iter()
        .map(|p| Vector3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let camera = OrbitCamera::oriented(center, max_radius, &ca_f32);

    let ds_bonds = extract_disulfide_bonds(&trace.protein);
    let num_disulfides = ds_bonds.len();
    let disulfide_mesh = if !ds_bonds.is_empty() {
        Some(generate_disulfide_mesh(&ds_bonds))
    } else {
        None
    };

    let analysis = proteus_core::metrics::analyze_pdb_detailed(&trace.protein, None).ok();
    let (metrics, detailed_plddts, rama_points) = if let Some(a) = analysis {
        (Some(a.metrics), a.plddts, a.ramachandran_points)
    } else {
        (None, plddts.clone(), Vec::new())
    };

    let (atom_table, ligands) = atoms::build_atoms(&trace.protein, &pdb, &trace.ids);
    let annotations = match &metrics {
        Some(m) => {
            let backbone_ids: Vec<ResidueId> =
                proteus_core::backbone::extract_backbone(&trace.protein)
                    .iter()
                    .map(|r| (r.chain_id.clone(), r.seq_num, r.insertion_code.clone()))
                    .collect();
            atoms::annotate(
                m,
                &rama_points,
                &backbone_ids,
                &trace.protein,
                &atom_table,
                &trace.ids,
                &trace.names,
            )
        }
        None => atoms::Annotations::default(),
    };

    let residue_labels = trace
        .ids
        .iter()
        .zip(&trace.names)
        .map(|((chain, number, icode), name)| ResidueLabel {
            chain: chain.clone(),
            number: *number,
            insertion_code: icode.clone(),
            name: name.clone(),
        })
        .collect();
    Ok(StructureRenderData {
        residue_labels,
        dssp: ss_summary.dssp.clone(),
        ribbon_mesh,
        disulfide_mesh,
        camera,
        num_residues: ca_coords.len(),
        num_disulfides,
        metrics,
        plddts: detailed_plddts,
        plddt_scale: trace.plddt_scale,
        ramachandran_points: rama_points,
        atoms: atom_table,
        ligands,
        annotations,
        confidence: None,
        scores: None,
        comparison: None,
    })
}

/// Colour scheme to use when the caller did not choose one: pLDDT only when the B-factor
/// column really is a confidence (predicted model); otherwise secondary structure, so that an
/// experimental structure is not painted "very low confidence" because its B-factors are small.
pub fn default_color_scheme(
    source: Option<proteus_core::confidence::ConfidenceSource>,
) -> ColorScheme {
    match source {
        Some(proteus_core::confidence::ConfidenceSource::Predicted) => ColorScheme::Plddt,
        _ => ColorScheme::SecondaryStructure,
    }
}

impl StructureRenderData {
    /// See [`default_color_scheme`]; uses this bundle's analysis.
    pub fn default_color_scheme(&self) -> ColorScheme {
        default_color_scheme(self.metrics.as_ref().map(|m| m.confidence_source))
    }

    /// What one rendered pixel covers, in Ångströms, for a terminal viewport of `cols` × `rows`
    /// cells on `backend`. See [`OrbitCamera::angstroms_per_pixel`].
    pub fn angstroms_per_pixel(&self, cols: usize, rows: usize, backend: TerminalBackend) -> f64 {
        match viewport_pixels(cols, rows, backend) {
            Ok((w, h)) => self.camera.angstroms_per_pixel(w, h),
            Err(_) => f64::NAN,
        }
    }

    /// A one-line caveat when the viewport cannot resolve what the structure contains, or
    /// `None` when it can. Consecutive C-alphas are 3.8 Å apart, so a pixel wider than that
    /// cannot separate neighbouring residues however good the rasteriser is.
    pub fn resolution_note(
        &self,
        cols: usize,
        rows: usize,
        backend: TerminalBackend,
    ) -> Option<String> {
        const CA_SPACING: f64 = 3.8;
        let a_per_px = self.angstroms_per_pixel(cols, rows, backend);
        if !a_per_px.is_finite() || a_per_px < CA_SPACING / 2.0 {
            return None;
        }
        Some(format!(
            "note: {a_per_px:.1} Å per pixel at {cols}×{rows} — neighbouring residues are 3.8 Å \
             apart, so this shows the fold's outline and not per-residue detail. Enlarge the \
             terminal, or use --backend braille (2×4 subpixels per cell) or kitty."
        ))
    }

    /// Attach a predictor's PAE and pTM. A PAE matrix whose size is not the ribbon's residue
    /// count (a complex with ligand tokens, the wrong file) is dropped rather than drawn against
    /// the wrong residues; the returned note says why.
    pub fn attach_confidence(
        &mut self,
        mut confidence: proteus_core::pae::PredictionConfidence,
    ) -> Option<String> {
        let mut note = None;
        if let Some(p) = &confidence.pae {
            if p.n != self.num_residues {
                note = Some(format!(
                    "note: the PAE matrix covers {} positions but the structure has {} \
                     residues; PAE is not shown",
                    p.n, self.num_residues
                ));
                confidence.pae = None;
            }
        }
        if !confidence.is_empty() {
            self.confidence = Some(confidence);
        }
        note
    }

    /// Superpose `reference_pdb` onto this structure (parsed from `self_pdb`) and keep its ribbon
    /// and the per-residue deviation. The reference moves; this structure, its atoms and its
    /// findings stay where they are.
    pub fn attach_comparison(
        &mut self,
        self_pdb: &str,
        reference_pdb: &str,
        name: &str,
    ) -> Result<SuperpositionStats, RenderError> {
        let tgt_pdb = proteus_core::io::open_structure_bytes(self_pdb.as_bytes(), None)
            .map_err(|e| RenderError::PdbParse(e.to_string()))?;
        let ref_pdb = proteus_core::io::open_structure_bytes(reference_pdb.as_bytes(), None)
            .map_err(|e| RenderError::PdbParse(format!("Reference structure parse failed: {e}")))?;
        let tgt = trace_of(&tgt_pdb);
        let refr = trace_of(&ref_pdb);
        if tgt.ca.len() != self.num_residues {
            return Err(RenderError::Geometry(
                "the structure text does not match this bundle".into(),
            ));
        }
        let (pairs, pairing) = pair_best(&tgt, &refr);
        if pairs.len() < MIN_SUPERPOSITION_PAIRS {
            return Err(RenderError::Geometry(format!(
                "only {} residue(s) could be paired with the reference, and at least \
                 {MIN_SUPERPOSITION_PAIRS} are needed to superpose",
                pairs.len()
            )));
        }
        let paired_tgt: Vec<Vector3<f64>> = pairs.iter().map(|&(i, _)| tgt.ca[i]).collect();
        let paired_ref: Vec<Vector3<f64>> = pairs.iter().map(|&(_, j)| refr.ca[j]).collect();
        // Fit the reference onto the target, so the target keeps its own frame.
        let sup = proteus_core::metrics::compute_kabsch_superposition(&paired_ref, &paired_tgt)
            .map_err(|e| RenderError::Geometry(e.to_string()))?;
        let moved: Vec<Vector3<f64>> = refr
            .ca
            .iter()
            .map(|p| sup.rotation * p + sup.translation)
            .collect();
        let guides: Vec<Option<Vector3<f64>>> = refr
            .guides
            .iter()
            .map(|g| g.map(|g| sup.rotation * g))
            .collect();
        let ref_ss =
            assign_secondary_structure(&proteus_core::backbone::extract_backbone(&refr.protein));
        let reference_mesh = segmented_cartoon_mesh(
            &moved,
            &ref_ss.assignment,
            &refr.plddts,
            &refr.breaks,
            &guides,
        );
        let mut deviation = vec![None; self.num_residues];
        for &(i, j) in &pairs {
            deviation[i] = Some((tgt.ca[i] - moved[j]).norm());
        }
        let stats = SuperpositionStats {
            rmsd: sup.rmsd,
            paired: pairs.len(),
            target_residues: tgt.ca.len(),
            reference_residues: refr.ca.len(),
            mismatched_names: pairs
                .iter()
                .filter(|&&(i, j)| tgt.names[i] != refr.names[j])
                .count(),
            pairing,
        };
        // Frame both: widen the camera to the reference's reach.
        let c = self.camera.center;
        let c64 = Vector3::new(c.x as f64, c.y as f64, c.z as f64);
        let reach = moved
            .iter()
            .map(|p| (p - c64).norm())
            .fold(0.0f64, f64::max) as f32;
        if reach > self.camera.bounding_radius {
            self.camera.bounding_radius = reach;
        }
        self.comparison = Some(Comparison {
            name: name.to_string(),
            reference_mesh,
            deviation,
            stats,
        });
        Ok(stats)
    }

    /// Place a score table on this structure's residues (see
    /// [`proteus_core::scores::ScoreTable::place`]). The returned line says how it matched.
    pub fn attach_scores(
        &mut self,
        table: &proteus_core::scores::ScoreTable,
        higher_is_worse: Option<bool>,
    ) -> Result<String, proteus_core::CoreError> {
        let numbers: Vec<isize> = self.residue_labels.iter().map(|l| l.number).collect();
        let letters: Vec<char> = self
            .residue_labels
            .iter()
            .map(|l| atoms::one_letter(&l.name))
            .collect();
        let scores = table.place(&numbers, &letters)?;
        let note = format!(
            "scores '{}': {} of {} residues, matched by {}{}",
            scores.column,
            scores.covered,
            self.num_residues,
            scores.matched_by,
            if scores.mismatches > 0 {
                format!(
                    " ({} wild-type letters differ from the structure)",
                    scores.mismatches
                )
            } else {
                String::new()
            }
        );
        let worse = higher_is_worse.unwrap_or_else(|| crate::higher_is_worse(&scores.column));
        self.scores = Some(AttachedScores {
            scores,
            higher_is_worse: worse,
        });
        Ok(note)
    }

    /// True when the B-factor column is a predictor's confidence.
    pub fn is_predicted(&self) -> bool {
        self.metrics
            .as_ref()
            .is_some_and(|m| m.confidence_source.is_predicted())
    }
}

/// Largest viewport side accepted, in terminal cells. Well past any real terminal.
pub const MAX_VIEWPORT_CELLS: usize = 4096;

/// Largest framebuffer accepted, in pixels (2²⁵: a 4096 × 4096-cell half-block frame). At 7
/// bytes a pixel (colour and depth) that is ~235 MB; a true-pixel backend reaches it at a far
/// smaller cell count, since each cell is 8 × 16 pixels.
pub const MAX_FRAMEBUFFER_PIXELS: usize = 1 << 25;

/// Framebuffer pixels behind a `cols` × `rows`-cell viewport on `backend`, or an error when
/// the viewport is empty or too large to allocate. Half-block packs 1×2 pixels per cell,
/// Braille 2×4, and kitty/Sixel blit true pixels (8×16 per cell).
///
/// Without this a `--width 100000 --height 100000` aborted the process on allocation, and a
/// width near `usize::MAX / 8` overflowed to an empty kitty image printed with exit status 0.
pub fn viewport_pixels(
    cols: usize,
    rows: usize,
    backend: TerminalBackend,
) -> Result<(usize, usize), RenderError> {
    if cols == 0 || rows == 0 {
        return Err(RenderError::InvalidViewport(format!(
            "{cols}×{rows} cells is empty; width and height must be at least 1"
        )));
    }
    if cols > MAX_VIEWPORT_CELLS || rows > MAX_VIEWPORT_CELLS {
        return Err(RenderError::InvalidViewport(format!(
            "{cols}×{rows} cells exceeds the maximum of {MAX_VIEWPORT_CELLS} per side"
        )));
    }
    let (px_w, px_h) = match backend {
        TerminalBackend::HalfBlock => (1, 2),
        TerminalBackend::Braille => (2, 4),
        // Sixel and kitty both blit true pixels, so both get a full cell's worth.
        TerminalBackend::Kitty | TerminalBackend::Sixel => (8, 16),
    };
    let (w, h) = (cols * px_w, rows * px_h);
    match w.checked_mul(h) {
        Some(n) if n <= MAX_FRAMEBUFFER_PIXELS => Ok((w, h)),
        _ => Err(RenderError::InvalidViewport(format!(
            "{cols}×{rows} cells is {w}×{h} pixels on this backend, over the \
             {MAX_FRAMEBUFFER_PIXELS}-pixel limit; use a smaller --width/--height"
        ))),
    }
}

/// Render a single static snapshot string from PDB content with SSAO and disulfide bridges.
pub fn render_pdb_snapshot(
    pdb_content: &str,
    width: usize,
    height: usize,
    backend: TerminalBackend,
    scheme: ColorScheme,
) -> Result<String, RenderError> {
    let structure = parse_pdb_structure(pdb_content)?;
    render_structure_snapshot(&structure, width, height, backend, scheme)
}

/// As [`render_pdb_snapshot`], for an already parsed structure bundle.
pub fn render_structure_snapshot(
    structure: &StructureRenderData,
    width: usize,
    height: usize,
    backend: TerminalBackend,
    scheme: ColorScheme,
) -> Result<String, RenderError> {
    // Pixel dimensions based on backend
    let (px_width, px_height) = viewport_pixels(width, height, backend)?;
    let mut fb = Framebuffer::new(px_width, px_height);
    fb.clear(ColorRGB::BLACK);

    let mut rasterizer = Rasterizer::new(scheme);
    if let Some(sc) = tui::ScoreColors::of(structure) {
        rasterizer.residue_colors = sc.residue_colors;
    }
    rasterizer.rasterize_mesh(&structure.ribbon_mesh, &structure.camera, &mut fb, scheme);

    if let Some(ref ds_mesh) = structure.disulfide_mesh {
        let gold = brand::structure::DISULFIDE; // Chartreuse
        rasterizer.rasterize_mesh(
            ds_mesh,
            &structure.camera,
            &mut fb,
            ColorScheme::Solid(gold),
        );
    }

    rasterizer.apply_post_processing(&mut fb);

    match backend {
        TerminalBackend::HalfBlock => Ok(HalfBlockRenderer::render_snapshot(&fb)),
        TerminalBackend::Braille => Ok(BrailleRenderer::render_snapshot(&fb)),
        TerminalBackend::Sixel => Ok(crate::terminal::SixelRenderer::render_snapshot(&fb)),
        TerminalBackend::Kitty => {
            let mut out = Vec::new();
            KittyRenderer::render(&fb, &mut out)?;
            String::from_utf8(out).map_err(|e| RenderError::Terminal(e.to_string()))
        }
    }
}

/// Data bundle for dual-structure superposition rendering.
pub struct SuperpositionRenderData {
    pub target_mesh: TriangleMesh,
    pub ref_mesh: TriangleMesh,
    pub camera: OrbitCamera,
    pub rmsd: f64,
    /// How the residues were paired, which the RMSD is only meaningful alongside.
    pub stats: SuperpositionStats,
}

/// What a superposition compared: residues are paired by identity (chain ID, residue number,
/// insertion code), and the RMSD covers the paired C-alphas only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuperpositionStats {
    /// C-alpha RMSD after Kabsch superposition of the paired residues, in Å.
    pub rmsd: f64,
    /// Residues present in both structures.
    pub paired: usize,
    pub target_residues: usize,
    pub reference_residues: usize,
    /// Paired residues whose amino acid differs between the two structures.
    pub mismatched_names: usize,
    /// How residues were paired: `"residue id"` (chain, number, insertion code), `"residue
    /// number"` (single-chain files with different chain ids), or `"sequence alignment"`
    /// (renumbered or differently numbered files).
    pub pairing: &'static str,
}

impl SuperpositionStats {
    /// True when every residue of each structure has a partner with the same amino acid, so
    /// the RMSD compares two conformations of one sequence.
    pub fn same_sequence(&self) -> bool {
        self.mismatched_names == 0
            && self.paired == self.target_residues
            && self.paired == self.reference_residues
    }
}

/// Fewest shared residues a superposition is attempted on: three points fix a rotation.
pub const MIN_SUPERPOSITION_PAIRS: usize = 3;

/// Index pairs `(target, reference)` of residues with the same identity, in target order. A
/// duplicated identity (which a well-formed file does not have) pairs its first occurrence.
fn pair_residues(target: &[ResidueId], reference: &[ResidueId]) -> Vec<(usize, usize)> {
    let mut by_id: std::collections::HashMap<&ResidueId, usize> =
        std::collections::HashMap::with_capacity(reference.len());
    for (j, id) in reference.iter().enumerate() {
        by_id.entry(id).or_insert(j);
    }
    let mut used = vec![false; reference.len()];
    let mut pairs = Vec::new();
    for (i, id) in target.iter().enumerate() {
        if let Some(&j) = by_id.get(id) {
            if !used[j] {
                used[j] = true;
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Global alignment (Needleman–Wunsch) of two residue-name sequences; returns the aligned
/// index pairs. Match +2, mismatch −1, gap −1: identical stretches dominate, a point mutation
/// stays aligned rather than opening two gaps.
fn align_by_sequence(a: &[String], b: &[String]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    let mut score = vec![vec![0i32; m + 1]; n + 1];
    for (i, row) in score.iter_mut().enumerate() {
        row[0] = -(i as i32);
    }
    for (j, cell) in score[0].iter_mut().enumerate() {
        *cell = -(j as i32);
    }
    for i in 1..=n {
        for j in 1..=m {
            let s = if a[i - 1] == b[j - 1] { 2 } else { -1 };
            score[i][j] = (score[i - 1][j - 1] + s)
                .max(score[i - 1][j] - 1)
                .max(score[i][j - 1] - 1);
        }
    }
    let (mut i, mut j, mut pairs) = (n, m, Vec::new());
    while i > 0 && j > 0 {
        let s = if a[i - 1] == b[j - 1] { 2 } else { -1 };
        if score[i][j] == score[i - 1][j - 1] + s {
            pairs.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if score[i][j] == score[i - 1][j] - 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    pairs
}

/// Pair residues of two traces the way a reader would: by identity when the files share a
/// numbering, by residue number when each is a single chain with a different chain id (a
/// predicted model against a deposited entry), and by sequence alignment when the numbering
/// differs altogether. The pairing that matches the most identical amino acids wins; ties go
/// to the more literal method.
fn pair_best(tgt: &Trace, refr: &Trace) -> (Vec<(usize, usize)>, &'static str) {
    let matching = |pairs: &[(usize, usize)]| {
        pairs
            .iter()
            .filter(|&&(i, j)| tgt.names[i] == refr.names[j])
            .count()
    };
    let mut best = (pair_residues(&tgt.ids, &refr.ids), "residue id");
    let single_chain = |ids: &[ResidueId]| ids.windows(2).all(|w| w[0].0 == w[1].0);
    if single_chain(&tgt.ids) && single_chain(&refr.ids) {
        let strip = |ids: &[ResidueId]| -> Vec<ResidueId> {
            ids.iter()
                .map(|(_, n, ic)| (String::new(), *n, ic.clone()))
                .collect()
        };
        let by_number = pair_residues(&strip(&tgt.ids), &strip(&refr.ids));
        if matching(&by_number) > matching(&best.0) {
            best = (by_number, "residue number");
        }
    }
    // The alignment table is n×m; beyond ~4000×4000 residues it is not worth the memory.
    if tgt.names.len() * refr.names.len() <= 16_000_000 {
        let aligned = align_by_sequence(&tgt.names, &refr.names);
        if matching(&aligned) > matching(&best.0) {
            best = (aligned, "sequence alignment");
        }
    }
    best
}

/// Parse and superimpose two structures using Kabsch optimal alignment.
///
/// Residues are paired by chain ID, residue number and insertion code, and only residues
/// present in both are superposed. Pairing by position — the i-th C-alpha of one file with
/// the i-th of the other — silently compared the wrong residues as soon as one file had a
/// residue the other lacked (1UBQ against itself minus its first residue gave 3.77 Å instead
/// of 0), and truncated the longer structure without saying so. Both structures are still drawn
/// whole.
pub fn prepare_superposition_for_rendering(
    target_pdb: &str,
    reference_pdb: &str,
) -> Result<SuperpositionRenderData, RenderError> {
    let tgt_pdb = proteus_core::io::open_structure_bytes(target_pdb.as_bytes(), None)
        .map_err(|e| RenderError::PdbParse(format!("Target structure parse failed: {e}")))?;
    let ref_pdb = proteus_core::io::open_structure_bytes(reference_pdb.as_bytes(), None)
        .map_err(|e| RenderError::PdbParse(format!("Reference structure parse failed: {e}")))?;

    let tgt = trace_of(&tgt_pdb);
    let refr = trace_of(&ref_pdb);

    let (pairs, pairing) = pair_best(&tgt, &refr);
    if pairs.len() < MIN_SUPERPOSITION_PAIRS {
        return Err(RenderError::Geometry(format!(
            "only {} residue(s) could be paired between the structures (target {}, reference \
             {} residues; tried residue ids, residue numbers and a sequence alignment), and \
             at least {MIN_SUPERPOSITION_PAIRS} are needed to superpose",
            pairs.len(),
            tgt.ca.len(),
            refr.ca.len()
        )));
    }
    let paired_tgt: Vec<Vector3<f64>> = pairs.iter().map(|&(i, _)| tgt.ca[i]).collect();
    let paired_ref: Vec<Vector3<f64>> = pairs.iter().map(|&(_, j)| refr.ca[j]).collect();
    let mismatched_names = pairs
        .iter()
        .filter(|&&(i, j)| tgt.names[i] != refr.names[j])
        .count();

    // Align target onto reference frame using Kabsch algorithm
    let sup = proteus_core::metrics::compute_kabsch_superposition(&paired_tgt, &paired_ref)
        .map_err(|e| RenderError::Geometry(e.to_string()))?;

    // The whole target moves with the transform fitted on the pairs, carbonyl guides included
    // (they are directions, so they rotate but do not translate).
    let aligned_tgt_ca: Vec<Vector3<f64>> = tgt
        .ca
        .iter()
        .map(|p| sup.rotation * p + sup.translation)
        .collect();
    let aligned_tgt_guides: Vec<Option<Vector3<f64>>> = tgt
        .guides
        .iter()
        .map(|g| g.map(|g| sup.rotation * g))
        .collect();
    // Secondary structure is invariant under rigid superposition: assign on the originals.
    let tgt_ss =
        assign_secondary_structure(&proteus_core::backbone::extract_backbone(&tgt.protein));
    let ref_ss =
        assign_secondary_structure(&proteus_core::backbone::extract_backbone(&refr.protein));

    let target_mesh = segmented_cartoon_mesh(
        &aligned_tgt_ca,
        &tgt_ss.assignment,
        &tgt.plddts,
        &tgt.breaks,
        &aligned_tgt_guides,
    );
    let ref_mesh = segmented_cartoon_mesh(
        &refr.ca,
        &ref_ss.assignment,
        &refr.plddts,
        &refr.breaks,
        &refr.guides,
    );

    // Frame both structures: centre and radius over every drawn C-alpha.
    let all: Vec<Vector3<f64>> = refr.ca.iter().chain(&aligned_tgt_ca).copied().collect();
    let center_f64 = all.iter().sum::<Vector3<f64>>() / all.len() as f64;
    let center = Vector3::new(
        center_f64.x as f32,
        center_f64.y as f32,
        center_f64.z as f32,
    );

    let max_radius = all
        .iter()
        .map(|p| (p - center_f64).norm())
        .fold(0.0f64, f64::max) as f32;

    let all_f32: Vec<Vector3<f32>> = all
        .iter()
        .map(|p| Vector3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let camera = OrbitCamera::oriented(center, max_radius * 1.15, &all_f32);

    Ok(SuperpositionRenderData {
        target_mesh,
        ref_mesh,
        camera,
        rmsd: sup.rmsd,
        stats: SuperpositionStats {
            rmsd: sup.rmsd,
            paired: pairs.len(),
            target_residues: tgt.ca.len(),
            reference_residues: refr.ca.len(),
            mismatched_names,
            pairing,
        },
    })
}

/// Render a single static snapshot string of two superimposed structures.
/// The target is drawn in Tide and the reference in Clay (`brand::structure`).
pub fn render_superposition_snapshot(
    target_pdb: &str,
    reference_pdb: &str,
    width: usize,
    height: usize,
    backend: TerminalBackend,
) -> Result<(String, SuperpositionStats), RenderError> {
    let (px_width, px_height) = viewport_pixels(width, height, backend)?;
    let data = prepare_superposition_for_rendering(target_pdb, reference_pdb)?;

    let mut fb = Framebuffer::new(px_width, px_height);
    fb.clear(ColorRGB::BLACK);

    let mut rasterizer = Rasterizer::new(ColorScheme::Plddt);
    // 1. Render the reference
    let ref_color = brand::structure::REFERENCE;
    rasterizer.rasterize_mesh(
        &data.ref_mesh,
        &data.camera,
        &mut fb,
        ColorScheme::Solid(ref_color),
    );

    // 2. Render the aligned target into the same depth buffer
    let target_color = brand::structure::TARGET;
    rasterizer.rasterize_mesh(
        &data.target_mesh,
        &data.camera,
        &mut fb,
        ColorScheme::Solid(target_color),
    );

    // 3. Screen-space ambient occlusion and silhouette cel outlines
    rasterizer.apply_post_processing(&mut fb);

    let output_str = match backend {
        TerminalBackend::HalfBlock => HalfBlockRenderer::render_snapshot(&fb),
        TerminalBackend::Braille => BrailleRenderer::render_snapshot(&fb),
        TerminalBackend::Sixel => crate::terminal::SixelRenderer::render_snapshot(&fb),
        TerminalBackend::Kitty => {
            let mut out = Vec::new();
            KittyRenderer::render(&fb, &mut out)?;
            String::from_utf8(out).map_err(|e| RenderError::Terminal(e.to_string()))?
        }
    };

    Ok((output_str, data.stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CRAMBIN_PDB: &str = include_str!("../../proteus-core/tests/data/1crn.pdb");

    /// Two copies of crambin, chain A and chain B, 60 Å apart.
    fn two_chains_far_apart() -> String {
        let mut out = String::new();
        for line in CRAMBIN_PDB.lines().filter(|l| l.starts_with("ATOM")) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("TER\n");
        for line in CRAMBIN_PDB.lines().filter(|l| l.starts_with("ATOM")) {
            let x: f64 = line[30..38].trim().parse().unwrap();
            let shifted = format!(
                "{}B{}{:8.3}{}",
                &line[..21],
                &line[22..30],
                x + 60.0,
                &line[38..]
            );
            out.push_str(&shifted);
            out.push('\n');
        }
        out.push_str("END\n");
        out
    }

    #[test]
    fn default_colour_is_plddt_only_for_predicted_models() {
        use proteus_core::confidence::ConfidenceSource as C;
        assert_eq!(default_color_scheme(Some(C::Predicted)), ColorScheme::Plddt);
        assert_eq!(
            default_color_scheme(Some(C::ExperimentalBFactor)),
            ColorScheme::SecondaryStructure
        );
        assert_eq!(
            default_color_scheme(Some(C::Unknown)),
            ColorScheme::SecondaryStructure
        );
        assert_eq!(default_color_scheme(None), ColorScheme::SecondaryStructure);
        // The bundle carries the analysis, so the choice can be made from it directly.
        let crambin = parse_pdb_structure(CRAMBIN_PDB).unwrap();
        assert_eq!(
            crambin.default_color_scheme(),
            ColorScheme::SecondaryStructure
        );
    }

    /// The simulated runner writes C-alpha-only files; they must still produce a ribbon.
    #[test]
    fn ca_only_trace_renders_a_ribbon() {
        let mut text = String::from("HEADER    SYNTHETIC STRUCTURE\n");
        for i in 0..20 {
            // Ideal alpha-helix C-alpha trace: 2.3 Å radius, 100° per residue, 1.5 Å rise.
            let theta = (i as f64) * 100.0f64.to_radians();
            text += &format!(
                "ATOM  {:5}  CA  ALA A{:4}    {:8.3}{:8.3}{:8.3}  1.00 80.00           C\n",
                i + 1,
                i + 1,
                2.3 * theta.cos(),
                2.3 * theta.sin(),
                1.5 * i as f64
            );
        }
        text += "END\n";
        let data = parse_pdb_structure(&text).unwrap();
        assert_eq!(data.num_residues, 20);
        assert!(
            !data.ribbon_mesh.vertices.is_empty(),
            "C-alpha-only structure produced an empty ribbon mesh"
        );
        let frame = render_structure_snapshot(
            &data,
            80,
            24,
            TerminalBackend::HalfBlock,
            ColorScheme::Plddt,
        )
        .unwrap();
        let drawn = frame
            .chars()
            .filter(|c| matches!(c, '▀' | '▄' | '█'))
            .count();
        assert!(drawn > 0, "nothing was drawn for a C-alpha-only helix");
    }

    /// The ribbon's flat face must follow the backbone, not an arbitrary axis.
    ///
    /// Checked against the validated interaction network rather than against the ribbon's own
    /// inputs, so this measures a physical property and not the plumbing: **residues that are
    /// backbone H-bond partners across a β-sheet should present near-parallel ribbon faces.**
    /// That coplanarity is what makes a sheet read as a sheet instead of a bundle of tubes.
    ///
    /// Measured over β-rich corpus structures, carbonyl-guided orientation gives a consistent
    /// 21–31° (1CRN 21.1, 1UBQ 21.5, 1TEN 25.9, 1PGB 26.3, 2CI2 30.8) — which is the real
    /// twist of a β-sheet. Pure parallel transport, which is all a renderer that never reads
    /// C and O can do, gives 25–84° on the same pairs: sometimes right by luck of the seed,
    /// never reliably. The consistency is the evidence, not the magnitude.
    #[test]
    fn hbonded_sheet_partners_share_a_ribbon_plane() {
        use proteus_core::structure::SecondaryStructure;
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let pdb = proteus_core::io::open_structure_bytes(text.as_bytes(), None).unwrap();
        let trace = trace_of(&pdb);
        let backbone = proteus_core::backbone::extract_backbone(&trace.protein);
        let ss = proteus_core::structure::assign_secondary_structure(&backbone);
        let net = proteus_core::interactions::compute_interaction_network(&pdb);

        let index: std::collections::HashMap<(String, isize), usize> = backbone
            .iter()
            .enumerate()
            .map(|(i, r)| ((r.chain_id.clone(), r.seq_num), i))
            .collect();

        let mut angles: Vec<f64> = Vec::new();
        for h in &net.hbonds {
            let (Some(&i), Some(&j)) = (
                index.get(&(h.donor_chain_id.clone(), h.donor_res_seq)),
                index.get(&(h.acceptor_chain_id.clone(), h.acceptor_res_seq)),
            ) else {
                continue;
            };
            // Both ends in a strand, and far enough apart in sequence to be a sheet contact
            // rather than a local turn.
            if ss.assignment.get(i) != Some(&SecondaryStructure::Strand)
                || ss.assignment.get(j) != Some(&SecondaryStructure::Strand)
                || (i as isize - j as isize).abs() < 3
            {
                continue;
            }
            if let (Some(a), Some(b)) = (trace.guides[i], trace.guides[j]) {
                // Sign is free: the face is a plane, not a direction.
                angles.push(a.dot(&b).abs().clamp(0.0, 1.0).acos().to_degrees());
            }
        }

        assert!(
            !angles.is_empty(),
            "no sheet H-bond pairs found to measure — has the interaction network or the \
             secondary-structure assignment changed?"
        );
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = angles[angles.len() / 2];
        assert!(
            median < 40.0,
            "ribbon faces of H-bonded sheet partners are {median:.1}° apart over {} pairs; a \
             β-sheet should be near-coplanar (observed 21–31° across the corpus). This is what \
             an arbitrary frame orientation looks like.",
            angles.len()
        );
    }

    /// The ribbon face must twist smoothly through a residue boundary, not snap there.
    ///
    /// With one fixed carbonyl guide per residue, the wide axis held still across a residue
    /// and then turned by the whole guide-to-guide angle between the last ring of one residue
    /// and the first of the next. Measured as the angle between the wide axes (vertex 0 minus
    /// vertex 4) of consecutive 8-vertex rings of the same secondary structure, each boundary
    /// step is compared with the step right after it, inside the next residue: both straddle
    /// the same C-alpha knot, where the spline tangent turns fastest, so the tangent's share
    /// largely cancels and what is left is the snap. On 1CRN the difference was a median 9.7°,
    /// p90 61.6° and max 76.8° with one guide per residue; guides anchored mid-peptide and
    /// interpolated give 6.3°, 12.4° and 16.5°.
    #[test]
    fn ribbon_face_twists_smoothly_across_residue_boundaries() {
        let data = parse_pdb_structure(CRAMBIN_PDB).unwrap();
        let v = &data.ribbon_mesh.vertices;
        // Wide-axis turn from ring k-1 to ring k, or None across a segment or SS change.
        let step = |k: usize| -> Option<(f32, bool)> {
            let (a0, a4, b0, b4) = (v[8 * (k - 1)], v[8 * (k - 1) + 4], v[8 * k], v[8 * k + 4]);
            let (wa, wb) = (a0.position - a4.position, b0.position - b4.position);
            let centre_step = ((a0.position + a4.position) - (b0.position + b4.position)) * 0.5;
            if wa.norm() < 1e-3
                || wb.norm() < 1e-3
                || centre_step.norm() > 3.0
                || a0.secondary_structure != b0.secondary_structure
            {
                return None;
            }
            let angle = wa
                .normalize()
                .dot(&wb.normalize())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            Some((angle, a0.residue_index != b0.residue_index))
        };
        // Only full rings: the two cap-centre vertices sit after the last ring.
        let rings = v.len() / 8;
        let mut excess: Vec<f32> = Vec::new();
        for k in 1..rings - 1 {
            if let (Some((across, true)), Some((next, false))) = (step(k), step(k + 1)) {
                excess.push((across - next).abs());
            }
        }
        assert!(
            excess.len() > 25,
            "only {} boundaries measured",
            excess.len()
        );
        excess.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = excess[excess.len() / 2];
        let p90 = excess[excess.len() * 9 / 10];
        let max = excess[excess.len() - 1];
        assert!(
            median < 8.0 && p90 < 20.0 && max < 30.0,
            "at a residue boundary the ribbon face turns differently from one ring later by a \
             median {median:.1}° (p90 {p90:.1}°, max {max:.1}°) — it snaps at each residue \
             instead of twisting"
        );
    }

    /// Secondary structure must survive a file that does not declare it.
    ///
    /// **Predicted structures carry no `HELIX`/`SHEET` records** — an ESMFold response has
    /// none, and neither does the offline simulator's output. A viewer that reads secondary
    /// structure from those records therefore has nothing to read for exactly the files a
    /// protein-engineering tool exists to look at, and falls back to a guess.
    ///
    /// Proteus runs Kabsch–Sander DSSP on the coordinates, so the render is identical whether
    /// the annotations are present or not. Measured against the deposited 1PGB with its five
    /// `HELIX`/`SHEET` records stripped, the rendered colour composition does not move at all;
    /// a viewer that reads the records saw its β-strand coverage nearly halve on the same pair
    /// of files.
    #[test]
    fn secondary_structure_survives_a_file_that_does_not_declare_it() {
        let with_records = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        ))
        .unwrap();
        assert!(
            with_records
                .lines()
                .any(|l| l.starts_with("HELIX") || l.starts_with("SHEET")),
            "fixture no longer carries the annotations this test strips"
        );
        let stripped: String = with_records
            .lines()
            .filter(|l| !l.starts_with("HELIX") && !l.starts_with("SHEET"))
            .map(|l| format!("{l}\n"))
            .collect();

        let a = parse_pdb_structure(&with_records).unwrap();
        let b = parse_pdb_structure(&stripped).unwrap();

        // Same count of vertices in each secondary-structure class, not merely "some SS".
        let tally = |d: &StructureRenderData| {
            let mut counts = [0usize; 3];
            for v in &d.ribbon_mesh.vertices {
                counts[match v.secondary_structure {
                    proteus_core::structure::SecondaryStructure::Helix => 0,
                    proteus_core::structure::SecondaryStructure::Strand => 1,
                    proteus_core::structure::SecondaryStructure::Coil => 2,
                }] += 1;
            }
            counts
        };
        let (ta, tb) = (tally(&a), tally(&b));
        assert_eq!(
            ta, tb,
            "the render changed when HELIX/SHEET records were removed: {ta:?} vs {tb:?} — \
             secondary structure must come from the coordinates, not the annotations"
        );
        assert!(
            ta[0] > 0 && ta[1] > 0,
            "crambin has both a helix and a sheet; got {ta:?}"
        );
    }

    /// Sixel exists for reach, but it is also the cheapest true-pixel path on the wire, which
    /// is what matters when the terminal is at the other end of an SSH session. The kitty
    /// protocol sends raw RGB, so its payload is a fixed function of the viewport; Sixel
    /// run-length encodes, and a protein render is mostly background.
    ///
    /// Measured at 110×30 cells: 1CRN 20.9× smaller, 1PGB 16.9×, 1TEN 14.3×, 4HHB 6.2× (a
    /// four-chain structure fills more of the frame, so it compresses least).
    #[test]
    fn sixel_is_cheaper_on_the_wire_than_kitty() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        ))
        .unwrap();
        let data = parse_pdb_structure(&text).unwrap();
        let scheme = data.default_color_scheme();
        let sixel =
            render_structure_snapshot(&data, 110, 30, TerminalBackend::Sixel, scheme).unwrap();
        let kitty =
            render_structure_snapshot(&data, 110, 30, TerminalBackend::Kitty, scheme).unwrap();
        assert!(
            sixel.len() * 4 < kitty.len(),
            "sixel {} bytes vs kitty {} — the run-length encoding is not working",
            sixel.len(),
            kitty.len()
        );
        // Both describe the same framebuffer, so the resolution advice must be the same too.
        assert_eq!(
            data.angstroms_per_pixel(110, 30, TerminalBackend::Sixel),
            data.angstroms_per_pixel(110, 30, TerminalBackend::Kitty),
            "sixel and kitty blit the same pixels and must report the same resolution"
        );
    }

    /// A picture that cannot separate neighbouring residues must say so — and the advice it
    /// gives has to be true, which is the part that is easy to get wrong.
    #[test]
    fn a_viewport_too_small_to_resolve_residues_says_so() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        ))
        .unwrap();
        let data = parse_pdb_structure(&text).unwrap();

        // 46 residues across a normal terminal resolves fine: no note, no crying wolf.
        assert!(
            data.resolution_note(80, 24, TerminalBackend::HalfBlock)
                .is_none(),
            "warned about a small protein that renders fine"
        );

        // The same structure squeezed into a tiny viewport cannot, and must say so.
        let note = data
            .resolution_note(10, 4, TerminalBackend::HalfBlock)
            .expect("no note at 10x4, where a pixel spans several residues");
        assert!(note.contains("Å per pixel"), "{note}");
        assert!(note.contains("3.8 Å apart"), "{note}");

        // The advice must hold: braille packs 2x4 subpixels per cell against half-block's 1x2,
        // so at the same cell count it genuinely resolves more. A note that recommended a
        // worse option would be worse than silence.
        let half = data.angstroms_per_pixel(10, 4, TerminalBackend::HalfBlock);
        let braille = data.angstroms_per_pixel(10, 4, TerminalBackend::Braille);
        let kitty = data.angstroms_per_pixel(10, 4, TerminalBackend::Kitty);
        assert!(
            braille < half && kitty < braille,
            "the note recommends finer backends, so they must actually be finer: \
             half-block {half:.2} Å/px, braille {braille:.2}, kitty {kitty:.2}"
        );

        // And a bigger terminal must help, which is the other half of the advice.
        assert!(
            data.angstroms_per_pixel(160, 48, TerminalBackend::HalfBlock) < half,
            "enlarging the terminal did not improve the resolution"
        );
    }

    /// A C-alpha-only trace has no carbonyl to orient by, so it must fall back to parallel
    /// transport rather than producing nothing. This is the offline simulator's output and any
    /// coarse-grained model.
    #[test]
    fn ca_only_traces_fall_back_to_parallel_transport() {
        let mut text = String::from("HEADER    CA ONLY\n");
        for i in 0..24 {
            let theta = (i as f64) * 100.0f64.to_radians();
            text += &format!(
                "ATOM  {:5}  CA  ALA A{:4}    {:8.3}{:8.3}{:8.3}  1.00 80.00           C\n",
                i + 1,
                i + 1,
                2.3 * theta.cos(),
                2.3 * theta.sin(),
                1.5 * i as f64
            );
        }
        text += "END\n";
        let pdb = proteus_core::io::open_structure_bytes(text.as_bytes(), None).unwrap();
        let trace = trace_of(&pdb);
        assert!(
            trace.guides.iter().all(Option::is_none),
            "a C-alpha-only trace cannot have carbonyl guides"
        );
        let data = parse_pdb_structure(&text).unwrap();
        assert!(
            !data.ribbon_mesh.vertices.is_empty(),
            "fallback did not produce a ribbon"
        );
    }

    /// A viewport too large to allocate, or empty, is refused with an error rather than
    /// aborting on allocation, overflowing, or printing nothing with success.
    #[test]
    fn oversized_or_empty_viewports_are_refused() {
        use TerminalBackend::*;
        let data = parse_pdb_structure(CRAMBIN_PDB).unwrap();
        let scheme = ColorScheme::SecondaryStructure;
        for (w, h, backend) in [
            (100_000, 100_000, HalfBlock),
            (100_000, 100_000, Kitty),
            (usize::MAX / 8 + 1, 1, Kitty), // w * 8 wraps to 0
            (usize::MAX, usize::MAX, Braille),
            (0, 10, Kitty),
            (10, 0, HalfBlock),
            (MAX_VIEWPORT_CELLS + 1, 10, HalfBlock),
            (MAX_VIEWPORT_CELLS, MAX_VIEWPORT_CELLS, Sixel), // within cells, over pixels
        ] {
            let err = render_structure_snapshot(&data, w, h, backend, scheme)
                .expect_err(&format!("{w}x{h} {backend:?} was accepted"));
            assert!(
                matches!(err, RenderError::InvalidViewport(_)),
                "{w}x{h} {backend:?}: {err}"
            );
            let err = render_superposition_snapshot(CRAMBIN_PDB, CRAMBIN_PDB, w, h, backend)
                .expect_err(&format!("superposition {w}x{h} {backend:?} was accepted"));
            assert!(matches!(err, RenderError::InvalidViewport(_)), "{err}");
            // The resolution advice must not panic on the same input either.
            assert!(data.resolution_note(w, h, backend).is_none());
        }
        // The limits themselves are usable.
        assert!(render_structure_snapshot(&data, MAX_VIEWPORT_CELLS, 2, HalfBlock, scheme).is_ok());
        assert!(viewport_pixels(MAX_VIEWPORT_CELLS, MAX_VIEWPORT_CELLS, HalfBlock).is_ok());
    }

    #[test]
    fn ribbon_does_not_bridge_chains() {
        let data = parse_pdb_structure(&two_chains_far_apart()).unwrap();
        assert_eq!(data.num_residues, 92);
        let pdb = proteus_core::io::open_structure_bytes(two_chains_far_apart().as_bytes(), None)
            .unwrap();
        let ca: Vec<Vector3<f32>> = pdb
            .atoms()
            .filter(|a| a.name() == "CA")
            .map(|a| Vector3::new(a.x() as f32, a.y() as f32, a.z() as f32))
            .collect();
        // Every ribbon vertex must sit near some C-alpha; a tube bridging the 60 Å gap has
        // vertices ~30 Å from everything.
        let worst = data
            .ribbon_mesh
            .vertices
            .iter()
            .map(|v| {
                ca.iter()
                    .map(|c| (v.position - c).norm())
                    .fold(f32::INFINITY, f32::min)
            })
            .fold(0.0f32, f32::max);
        assert!(
            worst < 6.0,
            "a ribbon vertex is {worst:.1} Å from the nearest C-alpha"
        );
    }

    #[test]
    fn altloc_and_ion_ca_atoms_do_not_enter_the_ribbon() {
        // Duplicate one CA as an alternate conformation and add a calcium ion named CA.
        let mut text = String::new();
        for line in CRAMBIN_PDB.lines().filter(|l| l.starts_with("ATOM")) {
            if &line[12..16] == " CA " && &line[22..26] == "   3" {
                text.push_str(&format!("{}A{}\n", &line[..16], &line[17..]));
                text.push_str(&format!("{}B{}\n", &line[..16], &line[17..]));
            } else {
                text.push_str(line);
                text.push('\n');
            }
        }
        text.push_str(
            "HETATM 9999 CA    CA A 101      50.000  50.000  50.000  1.00 10.00          CA\nEND\n",
        );
        let data = parse_pdb_structure(&text).unwrap();
        assert_eq!(data.num_residues, 46);
        let far = data
            .ribbon_mesh
            .vertices
            .iter()
            .filter(|v| (v.position - Vector3::new(50.0, 50.0, 50.0)).norm() < 8.0)
            .count();
        assert_eq!(far, 0, "ribbon reached the calcium ion");
    }

    #[test]
    fn test_parse_and_render_crambin_halfblock() {
        let snapshot = render_pdb_snapshot(
            CRAMBIN_PDB,
            80,
            24,
            TerminalBackend::HalfBlock,
            ColorScheme::Plddt,
        )
        .expect("Failed to render HalfBlock snapshot");

        assert!(!snapshot.is_empty());
        assert!(snapshot.contains('▀'));
    }

    #[test]
    fn test_parse_and_render_crambin_braille() {
        let snapshot = render_pdb_snapshot(
            CRAMBIN_PDB,
            80,
            24,
            TerminalBackend::Braille,
            ColorScheme::SecondaryStructure,
        )
        .expect("Failed to render Braille snapshot");

        assert!(!snapshot.is_empty());
    }

    #[test]
    fn test_parse_and_render_crambin_kitty() {
        let snapshot = render_pdb_snapshot(
            CRAMBIN_PDB,
            40,
            20,
            TerminalBackend::Kitty,
            ColorScheme::Rainbow,
        )
        .expect("Failed to render Kitty snapshot");

        assert!(!snapshot.is_empty());
        assert!(snapshot.contains("\x1b_Ga=T"));
    }

    #[test]
    fn test_render_superposition_snapshot() {
        let (snapshot, stats) = render_superposition_snapshot(
            CRAMBIN_PDB,
            CRAMBIN_PDB,
            80,
            24,
            TerminalBackend::Braille,
        )
        .expect("Failed to render superposition snapshot");

        assert!(!snapshot.is_empty());
        assert!(stats.rmsd < 1e-6); // Identical structures have 0.0 RMSD
        assert!(stats.same_sequence());
        assert_eq!(stats.paired, 46);
    }

    /// Keep `ATOM` records of crambin for which `keep(resSeq)` holds, with `edit` applied.
    fn crambin_where(keep: impl Fn(i32) -> bool, edit: impl Fn(&str) -> String) -> String {
        let mut out = String::new();
        for line in CRAMBIN_PDB.lines().filter(|l| l.starts_with("ATOM")) {
            if keep(line[22..26].trim().parse().unwrap()) {
                out.push_str(&edit(line));
                out.push('\n');
            }
        }
        out.push_str("END\n");
        out
    }

    /// `--compare` must pair residues by identity, not by position. Removing one residue from
    /// the front of a structure shifted every later pairing by one: the same coordinates
    /// superposed at 3.77 Å (1UBQ). A longer reference was silently truncated, and nothing
    /// said how many residues the RMSD covered or that the sequences differed.
    #[test]
    fn superposition_falls_back_when_chain_ids_or_numbering_differ() {
        // Reported: a predicted model (chain A, 1..N) against a deposited entry with another
        // chain id or numbering found 0 shared residues and refused to superpose.
        let chain_b = crambin_where(|_| true, |l| format!("{}B{}", &l[..21], &l[22..]));
        let s = prepare_superposition_for_rendering(&chain_b, CRAMBIN_PDB)
            .unwrap()
            .stats;
        assert!(s.rmsd < 1e-6 && s.paired == 46, "{s:?}");
        assert_eq!(s.pairing, "residue number");
        assert!(s.same_sequence());

        let renumbered = crambin_where(
            |_| true,
            |l| {
                let n: isize = l[22..26].trim().parse().unwrap();
                format!("{}{:>4}{}", &l[..22], n + 100, &l[26..])
            },
        );
        let s = prepare_superposition_for_rendering(&renumbered, CRAMBIN_PDB)
            .unwrap()
            .stats;
        assert!(s.rmsd < 1e-6 && s.paired == 46, "{s:?}");
        assert_eq!(s.pairing, "sequence alignment");
    }

    #[test]
    fn superposition_pairs_residues_by_identity() {
        let same = |l: &str| l.to_string();
        // One residue fewer at the N-terminus: 45 pairs, the same coordinates, RMSD 0.
        let shorter = crambin_where(|n| n != 1, same);
        let s = prepare_superposition_for_rendering(&shorter, CRAMBIN_PDB)
            .unwrap()
            .stats;
        assert!(
            s.rmsd < 1e-6,
            "positional pairing is back: RMSD {:.3}",
            s.rmsd
        );
        assert_eq!(
            (s.paired, s.target_residues, s.reference_residues),
            (45, 45, 46)
        );
        assert!(!s.same_sequence(), "an unpaired residue went unreported");

        // A second chain in the reference has no partner and must not be superposed.
        let s = prepare_superposition_for_rendering(CRAMBIN_PDB, &two_chains_far_apart())
            .unwrap()
            .stats;
        assert!(s.rmsd < 1e-6, "RMSD {:.3}", s.rmsd);
        assert_eq!((s.paired, s.reference_residues), (46, 92));

        // A point mutation pairs every residue but is reported as a sequence difference.
        let mutant = crambin_where(
            |_| true,
            |l| {
                if &l[22..26] == "  10" {
                    format!("{}ALA{}", &l[..17], &l[20..])
                } else {
                    l.to_string()
                }
            },
        );
        let s = prepare_superposition_for_rendering(&mutant, CRAMBIN_PDB)
            .unwrap()
            .stats;
        assert_eq!((s.paired, s.mismatched_names), (46, 1));
        assert!(!s.same_sequence());

        // The PDB and mmCIF of one entry pair completely.
        let cif = include_str!("../../proteus-core/tests/data/1crn.cif");
        let s = prepare_superposition_for_rendering(CRAMBIN_PDB, cif)
            .unwrap()
            .stats;
        assert!(s.same_sequence(), "{s:?}");
        assert!(s.rmsd < 1e-6);

        // Nothing in common — another chain and no residue name in common — is refused, not
        // superposed on nothing. (Another chain with the same sequence pairs by number; see
        // `superposition_falls_back_when_chain_ids_or_numbering_differ`.)
        let other_chain = crambin_where(|_| true, |l| format!("{}UNK B{}", &l[..17], &l[22..]));
        let err = prepare_superposition_for_rendering(&other_chain, CRAMBIN_PDB)
            .err()
            .expect("structures with no residue in common were superposed");
        assert!(err.to_string().contains("at least 3"), "{err}");
        let two = crambin_where(|n| n <= 2, same);
        assert!(prepare_superposition_for_rendering(&two, CRAMBIN_PDB).is_err());
    }

    #[test]
    fn test_disulfide_extraction_and_rendering() {
        let structure = parse_pdb_structure(CRAMBIN_PDB)
            .expect("Failed to parse Crambin structure for rendering");

        // Crambin (1CRN) contains 3 invariant disulfide bridges:
        // Cys3-Cys40, Cys4-Cys32, and Cys16-Cys26
        assert_eq!(structure.num_disulfides, 3);
        assert!(structure.disulfide_mesh.is_some());

        let ds_mesh = structure.disulfide_mesh.as_ref().unwrap();
        assert!(ds_mesh.triangle_count() > 0);
        assert!(!ds_mesh.vertices.is_empty());

        // Snapshot rendering with disulfides & SSAO enabled should complete successfully
        let snapshot = render_pdb_snapshot(
            CRAMBIN_PDB,
            80,
            24,
            TerminalBackend::HalfBlock,
            ColorScheme::SecondaryStructure,
        )
        .expect("Failed to render snapshot with disulfides");
        assert!(!snapshot.is_empty());
    }
}
