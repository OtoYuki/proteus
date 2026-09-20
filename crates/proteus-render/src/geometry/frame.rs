use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct BishopFrame {
    pub tangent: Vector3<f64>,
    pub normal1: Vector3<f64>,
    pub normal2: Vector3<f64>,
}

/// Compute singularity-free moving orthonormal coordinate frames along a 3D curve
/// using the Bishop Frame (Parallel Transport) algorithm.
pub fn compute_bishop_frames(tangents: &[Vector3<f64>]) -> Vec<BishopFrame> {
    let n = tangents.len();
    if n == 0 {
        return Vec::new();
    }

    let mut frames = Vec::with_capacity(n);

    // Initial frame at k = 0
    let t0 = tangents[0].normalize();
    let ref_axis = if t0.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let n1_0 = t0.cross(&ref_axis).normalize();
    let n2_0 = t0.cross(&n1_0).normalize();

    frames.push(BishopFrame {
        tangent: t0,
        normal1: n1_0,
        normal2: n2_0,
    });

    for k in 0..(n - 1) {
        let t_cur = frames[k].tangent;
        let t_next = tangents[k + 1].normalize();
        let n1_cur = frames[k].normal1;

        let axis = t_cur.cross(&t_next);
        let axis_len = axis.norm();

        let n1_next = if axis_len < 1e-6 {
            // Straight line segment: parallel transport is identical
            n1_cur
        } else {
            let u_axis = axis / axis_len;
            let dot = t_cur.dot(&t_next).clamp(-1.0, 1.0);
            let theta = dot.acos();

            // Rodrigues' rotation formula
            n1_cur * theta.cos()
                + u_axis.cross(&n1_cur) * theta.sin()
                + u_axis * u_axis.dot(&n1_cur) * (1.0 - theta.cos())
        }
        .normalize();

        let n2_next = t_next.cross(&n1_next).normalize();

        frames.push(BishopFrame {
            tangent: t_next,
            normal1: n1_next,
            normal2: n2_next,
        });
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bishop_orthonormality() {
        let tangents = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0).normalize(),
            Vector3::new(0.0, 1.0, 1.0).normalize(),
        ];

        let frames = compute_bishop_frames(&tangents);
        assert_eq!(frames.len(), 3);

        for f in &frames {
            // Check dot products are near 0 (orthogonal)
            assert!(f.tangent.dot(&f.normal1).abs() < 1e-5);
            assert!(f.tangent.dot(&f.normal2).abs() < 1e-5);
            assert!(f.normal1.dot(&f.normal2).abs() < 1e-5);

            // Check norms are 1.0 (normalized)
            assert!((f.tangent.norm() - 1.0).abs() < 1e-5);
            assert!((f.normal1.norm() - 1.0).abs() < 1e-5);
            assert!((f.normal2.norm() - 1.0).abs() < 1e-5);
        }
    }
}
