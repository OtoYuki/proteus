pub mod error;
pub mod geometry;
pub mod rasterizer;
pub mod terminal;
pub mod tui;

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

/// Generate a triangle mesh of golden covalent cylinders and sulfur spheres for all disulfide bonds.
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
    protein: pdbtbx::PDB,
    plddt_scale: f64,
}

fn trace_of(pdb: &pdbtbx::PDB) -> Trace {
    let protein = proteus_core::io::protein_heavy_atoms(pdb);
    let backbone = proteus_core::backbone::extract_backbone(&protein);
    let mut t = Trace {
        ca: Vec::with_capacity(backbone.len()),
        plddts: Vec::with_capacity(backbone.len()),
        breaks: Vec::with_capacity(backbone.len()),
        protein,
        plddt_scale: 1.0,
    };
    for r in &backbone {
        if let Some(ca) = r.ca {
            t.ca.push(ca);
            t.plddts.push(r.b_factor);
            t.breaks.push(r.chain_break_before);
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
) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();
    let mut start = 0usize;
    let n = ca.len();
    for end in 1..=n {
        if end == n || breaks[end] {
            if end - start >= 2 {
                let mut part =
                    generate_cartoon_mesh(&ca[start..end], &ss[start..end], &plddts[start..end], 4);
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

/// Parse PDB string content into a high-fidelity structure bundle with ribbons and disulfides.
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

    let ribbon_mesh =
        segmented_cartoon_mesh(&ca_coords, &ss_summary.assignment, &plddts, &trace.breaks);
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

    Ok(StructureRenderData {
        ribbon_mesh,
        disulfide_mesh,
        camera,
        num_residues: ca_coords.len(),
        num_disulfides,
        metrics,
        plddts: detailed_plddts,
        plddt_scale: trace.plddt_scale,
        ramachandran_points: rama_points,
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

    /// True when the B-factor column is a predictor's confidence.
    pub fn is_predicted(&self) -> bool {
        self.metrics
            .as_ref()
            .is_some_and(|m| m.confidence_source.is_predicted())
    }
}

/// Parse PDB string content and construct the 3D ribbon mesh and initial orbit camera.
pub fn parse_pdb_for_rendering(
    pdb_content: &str,
) -> Result<(TriangleMesh, OrbitCamera), RenderError> {
    let data = parse_pdb_structure(pdb_content)?;
    Ok((data.ribbon_mesh, data.camera))
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
    let (px_width, px_height) = match backend {
        TerminalBackend::HalfBlock => (width, height * 2),
        TerminalBackend::Braille => (width * 2, height * 4),
        TerminalBackend::Kitty => (width * 8, height * 16),
    };

    let mut fb = Framebuffer::new(px_width, px_height);
    fb.clear(ColorRGB::BLACK);

    let mut rasterizer = Rasterizer::new(scheme);
    rasterizer.rasterize_mesh(&structure.ribbon_mesh, &structure.camera, &mut fb, scheme);

    if let Some(ref ds_mesh) = structure.disulfide_mesh {
        let gold = ColorRGB::new(251, 191, 36);
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
}

/// Parse and superimpose two PDB structures using Kabsch optimal alignment.
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
    let (tgt_ca, tgt_plddts) = (tgt.ca, tgt.plddts);
    let (ref_ca, ref_plddts) = (refr.ca, refr.plddts);

    let common_len = tgt_ca.len().min(ref_ca.len());
    if common_len < 2 {
        return Err(RenderError::PdbParse(
            "Both structures must contain at least 2 C-alpha atoms for superposition".into(),
        ));
    }

    // Align target onto reference frame using Kabsch algorithm
    let sup = proteus_core::metrics::compute_kabsch_superposition(
        &tgt_ca[..common_len],
        &ref_ca[..common_len],
    )
    .map_err(|e| RenderError::Geometry(e.to_string()))?;

    let aligned_tgt_ca = sup.aligned_coords;
    // Secondary structure is invariant under rigid superposition: assign on the originals.
    let tgt_ss =
        assign_secondary_structure(&proteus_core::backbone::extract_backbone(&tgt.protein));
    let ref_ss =
        assign_secondary_structure(&proteus_core::backbone::extract_backbone(&refr.protein));

    let target_mesh = segmented_cartoon_mesh(
        &aligned_tgt_ca,
        &tgt_ss.assignment[..common_len],
        &tgt_plddts[..common_len],
        &tgt.breaks[..common_len],
    );
    let ref_mesh = segmented_cartoon_mesh(
        &ref_ca[..common_len],
        &ref_ss.assignment[..common_len],
        &ref_plddts[..common_len],
        &refr.breaks[..common_len],
    );

    // Compute bounding center and radius over the combined structures
    let n = common_len as f64;
    let sum_pos: Vector3<f64> = ref_ca[..common_len].iter().sum();
    let center_f64 = sum_pos / n;
    let center = Vector3::new(
        center_f64.x as f32,
        center_f64.y as f32,
        center_f64.z as f32,
    );

    let max_radius = ref_ca[..common_len]
        .iter()
        .map(|p| (p - center_f64).norm())
        .fold(0.0f64, f64::max) as f32;

    let ref_f32: Vec<Vector3<f32>> = ref_ca[..common_len]
        .iter()
        .map(|p| Vector3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let camera = OrbitCamera::oriented(center, max_radius * 1.15, &ref_f32);

    Ok(SuperpositionRenderData {
        target_mesh,
        ref_mesh,
        camera,
        rmsd: sup.rmsd,
    })
}

/// Render a single static snapshot string of two superimposed structures.
/// Target is rendered in Cyan (#06B6D4), Reference is rendered in Ruby (#F43F5E).
pub fn render_superposition_snapshot(
    target_pdb: &str,
    reference_pdb: &str,
    width: usize,
    height: usize,
    backend: TerminalBackend,
) -> Result<(String, f64), RenderError> {
    let data = prepare_superposition_for_rendering(target_pdb, reference_pdb)?;

    let (px_width, px_height) = match backend {
        TerminalBackend::HalfBlock => (width, height * 2),
        TerminalBackend::Braille => (width * 2, height * 4),
        TerminalBackend::Kitty => (width * 8, height * 16),
    };

    let mut fb = Framebuffer::new(px_width, px_height);
    fb.clear(ColorRGB::BLACK);

    let mut rasterizer = Rasterizer::new(ColorScheme::Plddt);
    // 1. Render reference in Ruby
    let ref_color = ColorRGB::new(244, 63, 94);
    rasterizer.rasterize_mesh(
        &data.ref_mesh,
        &data.camera,
        &mut fb,
        ColorScheme::Solid(ref_color),
    );

    // 2. Render aligned target in Cyan into the same depth buffer
    let target_color = ColorRGB::new(6, 182, 212);
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
        TerminalBackend::Kitty => {
            let mut out = Vec::new();
            KittyRenderer::render(&fb, &mut out)?;
            String::from_utf8(out).map_err(|e| RenderError::Terminal(e.to_string()))?
        }
    };

    Ok((output_str, data.rmsd))
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
        let (snapshot, rmsd) = render_superposition_snapshot(
            CRAMBIN_PDB,
            CRAMBIN_PDB,
            80,
            24,
            TerminalBackend::Braille,
        )
        .expect("Failed to render superposition snapshot");

        assert!(!snapshot.is_empty());
        assert!(rmsd < 1e-6); // Identical structures have 0.0 RMSD
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
