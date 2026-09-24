use super::buffer::ColorRGB;
use nalgebra::Vector3;
use proteus_core::structure::SecondaryStructure;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    Plddt,
    SecondaryStructure,
    Rainbow,
    Solid(ColorRGB),
    /// Per-residue colours from an attached score table (`Rasterizer::residue_colors`).
    Scores,
}

/// ColorBrewer RdBu, from the damaging end (red) to the tolerated end (blue).
pub const SCORE_STOPS: [ColorRGB; 7] = [
    ColorRGB {
        r: 178,
        g: 24,
        b: 43,
    },
    ColorRGB {
        r: 239,
        g: 138,
        b: 98,
    },
    ColorRGB {
        r: 253,
        g: 219,
        b: 199,
    },
    ColorRGB {
        r: 247,
        g: 247,
        b: 247,
    },
    ColorRGB {
        r: 209,
        g: 229,
        b: 240,
    },
    ColorRGB {
        r: 103,
        g: 169,
        b: 207,
    },
    ColorRGB {
        r: 33,
        g: 102,
        b: 172,
    },
];

/// ColorBrewer Greens, reversed: AlphaFold DB's PAE colours, 0 Å darkest.
pub const PAE_STOPS: [ColorRGB; 9] = [
    ColorRGB { r: 0, g: 68, b: 27 },
    ColorRGB {
        r: 0,
        g: 109,
        b: 44,
    },
    ColorRGB {
        r: 35,
        g: 139,
        b: 69,
    },
    ColorRGB {
        r: 65,
        g: 171,
        b: 93,
    },
    ColorRGB {
        r: 116,
        g: 196,
        b: 118,
    },
    ColorRGB {
        r: 161,
        g: 217,
        b: 155,
    },
    ColorRGB {
        r: 199,
        g: 233,
        b: 192,
    },
    ColorRGB {
        r: 229,
        g: 245,
        b: 224,
    },
    ColorRGB {
        r: 247,
        g: 252,
        b: 245,
    },
];

/// A PAE value on AlphaFold DB's scale, 0 → `max` Å.
pub fn pae_color(v: f32, max: f32) -> ColorRGB {
    let t = (v / max.max(1e-3)).clamp(0.0, 1.0) * (PAE_STOPS.len() - 1) as f32;
    let i = (t.floor() as usize).min(PAE_STOPS.len() - 2);
    ColorRGB::lerp(PAE_STOPS[i], PAE_STOPS[i + 1], t - i as f32)
}

/// A residue with no score: a neutral grey that no end of the scale uses.
pub const NO_SCORE: ColorRGB = ColorRGB {
    r: 96,
    g: 96,
    b: 90,
};

/// The colour scale of a score column: 2nd–98th percentile limits, centred on zero when the
/// values change sign (a log-likelihood ratio, a fitness relative to wild type), with red at
/// the damaging end whichever way the column runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreScale {
    pub lo: f64,
    pub hi: f64,
    pub diverging: bool,
    pub higher_is_worse: bool,
}

impl ScoreScale {
    pub fn fit(values: &[Option<f64>], higher_is_worse: bool) -> Self {
        let mut v: Vec<f64> = values.iter().flatten().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f64| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v[((v.len() - 1) as f64 * q).round() as usize]
            }
        };
        let (p2, p98) = (pct(0.02), pct(0.98));
        let diverging = p2 < 0.0 && p98 > 0.0;
        let (lo, hi) = if diverging {
            let m = p2.abs().max(p98.abs());
            (-m, m)
        } else if p98 > p2 {
            (p2, p98)
        } else {
            (p2 - 0.5, p2 + 0.5)
        };
        Self {
            lo,
            hi,
            diverging,
            higher_is_worse,
        }
    }

    /// Colour of one value; `None` is [`NO_SCORE`].
    pub fn color(&self, value: Option<f64>) -> ColorRGB {
        let Some(v) = value else { return NO_SCORE };
        let mut t = ((v - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) as f32;
        if self.higher_is_worse {
            t = 1.0 - t;
        }
        let x = t * (SCORE_STOPS.len() - 1) as f32;
        let i = (x.floor() as usize).min(SCORE_STOPS.len() - 2);
        ColorRGB::lerp(SCORE_STOPS[i], SCORE_STOPS[i + 1], x - i as f32)
    }
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
