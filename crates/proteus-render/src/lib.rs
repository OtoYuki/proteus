pub mod error;
pub mod geometry;
pub mod rasterizer;
pub mod terminal;
pub mod tui;

use error::RenderError;
use geometry::mesh::{generate_cartoon_mesh, TriangleMesh};
use nalgebra::Vector3;
use pdbtbx::{open_raw, StrictnessLevel};
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

/// Parse PDB string content into a high-fidelity structure bundle with ribbons and disulfides.
pub fn parse_pdb_structure(pdb_content: &str) -> Result<StructureRenderData, RenderError> {
    let cursor = std::io::Cursor::new(pdb_content.as_bytes());
    let reader = std::io::BufReader::new(cursor);
    let (pdb, _errors) = open_raw(reader, StrictnessLevel::Loose)
        .map_err(|e| RenderError::PdbParse(format!("{e:?}")))?;

    let mut ca_coords = Vec::new();
    let mut plddts = Vec::new();

    for residue in pdb.residues() {
        for atom in residue.atoms() {
            if atom.name().trim() == "CA" {
                ca_coords.push(Vector3::new(atom.x(), atom.y(), atom.z()));
                plddts.push(atom.b_factor());
            }
        }
    }

    if ca_coords.len() < 2 {
        return Err(RenderError::PdbParse(
            "PDB must contain at least 2 C-alpha residues for ribbon rendering".into(),
        ));
    }

    // Normalize pLDDT if in [0.0, 1.0] range (e.g., raw ESMFold outputs)
    let max_plddt = plddts.iter().copied().fold(f64::MIN, f64::max);
    if max_plddt <= 1.0 && max_plddt > 0.0 {
        for v in &mut plddts {
            *v *= 100.0;
        }
    }

    let ss_summary = assign_secondary_structure(&ca_coords);

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

    let ribbon_mesh = generate_cartoon_mesh(&ca_coords, &ss_summary.assignment, &plddts, 4);
    let camera = OrbitCamera::new(center, max_radius);

    let ds_bonds = extract_disulfide_bonds(&pdb);
    let num_disulfides = ds_bonds.len();
    let disulfide_mesh = if !ds_bonds.is_empty() {
        Some(generate_disulfide_mesh(&ds_bonds))
    } else {
        None
    };

    let analysis = proteus_core::metrics::analyze_pdb_detailed(&pdb, None).ok();
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
        ramachandran_points: rama_points,
    })
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
    let cursor_tgt = std::io::Cursor::new(target_pdb.as_bytes());
    let (tgt_pdb, _) = open_raw(std::io::BufReader::new(cursor_tgt), StrictnessLevel::Loose)
        .map_err(|e| RenderError::PdbParse(format!("Target PDB parse failed: {e:?}")))?;

    let cursor_ref = std::io::Cursor::new(reference_pdb.as_bytes());
    let (ref_pdb, _) = open_raw(std::io::BufReader::new(cursor_ref), StrictnessLevel::Loose)
        .map_err(|e| RenderError::PdbParse(format!("Reference PDB parse failed: {e:?}")))?;

    let mut tgt_ca = Vec::new();
    let mut tgt_plddts = Vec::new();
    for r in tgt_pdb.residues() {
        for a in r.atoms() {
            if a.name().trim() == "CA" {
                tgt_ca.push(Vector3::new(a.x(), a.y(), a.z()));
                tgt_plddts.push(a.b_factor());
            }
        }
    }

    let mut ref_ca = Vec::new();
    let mut ref_plddts = Vec::new();
    for r in ref_pdb.residues() {
        for a in r.atoms() {
            if a.name().trim() == "CA" {
                ref_ca.push(Vector3::new(a.x(), a.y(), a.z()));
                ref_plddts.push(a.b_factor());
            }
        }
    }

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
    let tgt_ss = assign_secondary_structure(&aligned_tgt_ca);
    let ref_ss = assign_secondary_structure(&ref_ca[..common_len]);

    let target_mesh = generate_cartoon_mesh(
        &aligned_tgt_ca,
        &tgt_ss.assignment,
        &tgt_plddts[..common_len],
        4,
    );
    let ref_mesh = generate_cartoon_mesh(
        &ref_ca[..common_len],
        &ref_ss.assignment,
        &ref_plddts[..common_len],
        4,
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

    let camera = OrbitCamera::new(center, max_radius * 1.15);

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
