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

/// Parse PDB string content and construct the 3D ribbon mesh and initial orbit camera.
pub fn parse_pdb_for_rendering(
    pdb_content: &str,
) -> Result<(TriangleMesh, OrbitCamera), RenderError> {
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

    let mesh = generate_cartoon_mesh(&ca_coords, &ss_summary.assignment, &plddts, 4);
    let camera = OrbitCamera::new(center, max_radius);

    Ok((mesh, camera))
}

/// Render a single static snapshot string from PDB content.
pub fn render_pdb_snapshot(
    pdb_content: &str,
    width: usize,
    height: usize,
    backend: TerminalBackend,
    scheme: ColorScheme,
) -> Result<String, RenderError> {
    let (mesh, camera) = parse_pdb_for_rendering(pdb_content)?;

    // Pixel dimensions based on backend
    let (px_width, px_height) = match backend {
        TerminalBackend::HalfBlock => (width, height * 2),
        TerminalBackend::Braille => (width * 2, height * 4),
        TerminalBackend::Kitty => (width * 8, height * 16),
    };

    let mut fb = Framebuffer::new(px_width, px_height);
    fb.clear(ColorRGB::BLACK);

    let mut rasterizer = Rasterizer::new(scheme);
    rasterizer.render(&mesh, &camera, &mut fb);

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
    rasterizer.render_with_scheme(
        &data.ref_mesh,
        &data.camera,
        &mut fb,
        ColorScheme::Solid(ref_color),
    );

    // 2. Render aligned target in Cyan into the same depth buffer
    let target_color = ColorRGB::new(6, 182, 212);
    rasterizer.render_with_scheme(
        &data.target_mesh,
        &data.camera,
        &mut fb,
        ColorScheme::Solid(target_color),
    );

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
}
