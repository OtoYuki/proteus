use super::buffer::ColorRGB;
use nalgebra::Vector3;
use proteus_core::structure::SecondaryStructure;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    Plddt,
    SecondaryStructure,
    Rainbow,
    Solid(ColorRGB),
}

/// Map AlphaFold / ESMFold pLDDT score (0.0 - 100.0) to standard colors:
/// - >= 90: Deep Royal Blue (#0053D6)
/// - 70 - 90: Light Cyan (#65CBF3)
/// - 50 - 70: Yellow (#FFDB13)
/// - < 50: Orange-Red (#FF7D45)
pub fn plddt_to_color(plddt: f32) -> ColorRGB {
    if plddt >= 90.0 {
        ColorRGB::new(0, 83, 214)
    } else if plddt >= 70.0 {
        // Interpolate between cyan (70) and deep blue (90)
        let t = (plddt - 70.0) / 20.0;
        ColorRGB::lerp(ColorRGB::new(101, 203, 243), ColorRGB::new(0, 83, 214), t)
    } else if plddt >= 50.0 {
        // Interpolate between yellow (50) and cyan (70)
        let t = (plddt - 50.0) / 20.0;
        ColorRGB::lerp(ColorRGB::new(255, 219, 19), ColorRGB::new(101, 203, 243), t)
    } else {
        // Interpolate between red-orange (0) and yellow (50)
        let t = (plddt / 50.0).clamp(0.0, 1.0);
        ColorRGB::lerp(ColorRGB::new(255, 125, 69), ColorRGB::new(255, 219, 19), t)
    }
}

pub fn secondary_structure_to_color(ss: SecondaryStructure) -> ColorRGB {
    match ss {
        // The Proteus structure colours (Clay, Tide, pale cream): chosen to stay apart under
        // deutan, protan and tritan vision (brand::tests).
        SecondaryStructure::Helix => crate::brand::structure::HELIX,
        SecondaryStructure::Strand => crate::brand::structure::STRAND,
        SecondaryStructure::Coil => crate::brand::structure::COIL,
    }
}

pub fn rainbow_color(residue_index: usize, total_residues: usize) -> ColorRGB {
    if total_residues == 0 {
        return ColorRGB::WHITE;
    }
    let t = (residue_index as f32 / total_residues as f32).clamp(0.0, 1.0);
    // Standard rainbow spectrum: Blue -> Cyan -> Green -> Yellow -> Red
    let hue = (1.0 - t) * 240.0; // 240 (blue) down to 0 (red)
    hsv_to_rgb(hue, 0.85, 0.95)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> ColorRGB {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    ColorRGB::new(
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Shade a surface point with Blinn-Phong lighting model (key light + fill light + ambient)
pub fn shade_blinn_phong(base_color: ColorRGB, normal: Vector3<f32>) -> ColorRGB {
    let key_light = Vector3::new(0.5, 0.8, 1.0).normalize();
    let fill_light = Vector3::new(-0.6, -0.4, 0.5).normalize();
    let view_dir = Vector3::new(0.0, 0.0, 1.0); // Facing viewer in camera view

    let n = if normal.norm_squared() < 1e-6 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        normal.normalize()
    };

    let ambient = 0.30;
    let diff_key = n.dot(&key_light).max(0.0) * 0.65;
    let diff_fill = n.dot(&fill_light).max(0.0) * 0.25;

    let half_vec = (key_light + view_dir).normalize();
    let spec = n.dot(&half_vec).max(0.0).powf(16.0) * 0.25;

    let intensity = (ambient + diff_key + diff_fill + spec).clamp(0.0, 1.3);

    ColorRGB::new(
        ((base_color.r as f32 * intensity).min(255.0)) as u8,
        ((base_color.g as f32 * intensity).min(255.0)) as u8,
        ((base_color.b as f32 * intensity).min(255.0)) as u8,
    )
}
