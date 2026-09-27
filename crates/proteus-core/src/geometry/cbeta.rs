//! Cβ deviation (Lovell et al. 2003): distance of the modelled CB from the position the
//! backbone N, CA and C imply for it. A deviation of 0.25 Å or more flags a backbone or
//! side chain forced into a strained conformation.
//!
//! A line-by-line port of cctbx `mmtbx/validation/cbetadev.py`
//! (`calculate_ideal_and_deviation`, `idealized_calpha_angles`, `construct_fourth`) and
//! `scitbx.matrix.rotate_point_around_axis`.

use nalgebra::Vector3;

use super::restraints::chiral_volume;

/// `cbetadev`: a residue is an outlier at this deviation or more.
pub const CBETA_OUTLIER: f64 = 0.25;

/// Ideal CA–CB distance and the angles/dihedrals the two constructions of the ideal CB use
/// (`idealized_calpha_angles`), for L-amino acids.
struct Ideal {
    dist: f64,
    angle_cab: f64,
    dihedral_ncab: f64,
    angle_nab: f64,
    dihedral_cnab: f64,
}

fn idealized(resname: &str) -> Ideal {
    match resname {
        "ALA" | "DAL" => Ideal {
            dist: 1.536,
            angle_cab: 110.1,
            dihedral_ncab: 122.9,
            angle_nab: 110.6,
            dihedral_cnab: -122.6,
        },
        "PRO" | "DPR" => Ideal {
            dist: 1.530,
            angle_cab: 112.2,
            dihedral_ncab: 115.1,
            angle_nab: 103.0,
            dihedral_cnab: -120.7,
        },
        "VAL" | "THR" | "ILE" | "DVA" | "DTH" | "DIL" => Ideal {
            dist: 1.540,
            angle_cab: 109.1,
            dihedral_ncab: 123.4,
            angle_nab: 111.5,
            dihedral_cnab: -122.0,
        },
        _ => Ideal {
            dist: 1.530,
            angle_cab: 110.1,
            dihedral_ncab: 122.8,
            angle_nab: 110.5,
            dihedral_cnab: -122.6,
        },
    }
}

/// Rotate `point` by `angle` degrees about the axis from `p1` to `p2` (right-handed).
fn rotate(p1: Vector3<f64>, p2: Vector3<f64>, point: Vector3<f64>, angle: f64) -> Vector3<f64> {
    let angle = angle.to_radians();
    let l = p2 - p1;
    let dlsq = l.norm_squared();
    let dl = dlsq.sqrt();
    let ca = angle.cos();
    let dsa = angle.sin() / dl;
    let oca = (1.0 - ca) / dlsq;
    let (xl, yl, zl) = (l.x, l.y, l.z);
    let m = point - p1;
    Vector3::new(
        m.x * (xl * xl * oca + ca)
            + m.y * (xl * yl * oca - zl * dsa)
            + m.z * (xl * zl * oca + yl * dsa),
        m.x * (xl * yl * oca + zl * dsa)
            + m.y * (yl * yl * oca + ca)
            + m.z * (yl * zl * oca - xl * dsa),
        m.x * (xl * zl * oca - yl * dsa)
            + m.y * (yl * zl * oca + xl * dsa)
            + m.z * (zl * zl * oca + ca),
    ) + p1
}

/// `construct_fourth`: place a fourth atom at `dist` from `r2`, making `angle` at `r2` and
/// `dihedral` with `r0`–`r1`–`r2`.
fn construct_fourth(
    r0: Vector3<f64>,
    r1: Vector3<f64>,
    r2: Vector3<f64>,
    dist: f64,
    angle: f64,
    dihedral: f64,
) -> Vector3<f64> {
    let mut c = (r2 - r1).cross(&(r0 - r1));
    let cmag = c.norm();
    if cmag > 0.000001 {
        c *= dist / cmag;
    }
    let d = c + r2;
    let new_d = rotate(r1, r2, d, dihedral - 90.0);
    let mut c = (new_d - r2).cross(&(r1 - r2));
    let cmag = c.norm();
    if cmag > 0.000001 {
        c *= dist / cmag;
    }
    let b = c + r2;
    if b == r2 {
        return new_d;
    }
    rotate(r2, b, new_d, 90.0 - angle)
}

/// cctbx's `common_amino_acid` class: the 20 standard residues and selenomethionine.
const STANDARD: [&str; 21] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
];

/// D-amino-acid residue names cctbx knows (`iotbx.pdb.amino_acid_codes`).
const D_AMINO_ACIDS: [&str; 19] = [
    "DAL", "DAR", "DAS", "DCY", "DGL", "DGN", "DHI", "DIL", "DLE", "DLY", "DPN", "DPR", "DSG",
    "DSN", "DTH", "DTR", "DTY", "DVA", "MED",
];

/// Deviation of `cb` from the ideal CB built on `n`, `ca`, `c` for residue `resname`.
/// `None` for glycine.
///
/// The targets are for L-amino acids; they are mirrored for a D-amino-acid name and, for a
/// residue name that is neither standard nor D, when its CA–N–CB–C chiral volume is positive
/// (`idealized_calpha_angles`), so a misnamed standard residue still shows up as an outlier.
pub fn deviation(
    resname: &str,
    n: Vector3<f64>,
    ca: Vector3<f64>,
    c: Vector3<f64>,
    cb: Vector3<f64>,
) -> Option<f64> {
    if resname == "GLY" {
        return None;
    }
    let mut ideal = idealized(resname);
    let mirrored = !STANDARD.contains(&resname)
        && (D_AMINO_ACIDS.contains(&resname) || chiral_volume([ca, n, cb, c]) > 0.0);
    if mirrored {
        ideal.dihedral_ncab = -ideal.dihedral_ncab;
        ideal.dihedral_cnab = -ideal.dihedral_cnab;
    }
    let ncab = construct_fourth(n, c, ca, ideal.dist, ideal.angle_cab, ideal.dihedral_ncab);
    let cnab = construct_fourth(c, n, ca, ideal.dist, ideal.angle_nab, ideal.dihedral_cnab);
    let mut beta = (ncab + cnab) / 2.0;
    let betadist = (ca - beta).norm();
    if betadist == 0.0 {
        return None;
    }
    if betadist != ideal.dist {
        beta = ca + (beta - ca) * ideal.dist / betadist;
    }
    Some((cb - beta).norm()).filter(|d| d.is_finite())
}
