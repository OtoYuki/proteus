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
}
