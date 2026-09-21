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

    /// The point of a Bishop (rotation-minimising) frame: the normal must rotate by no *more*
    /// than the tangent does. A Frenet frame violates this at an inflection point — its normal
    /// swings through the curvature reversal — which is what makes ribbons built on one flip.
    #[test]
    fn normals_rotate_no_faster_than_the_tangent() {
        // A path with an inflection: curvature reverses sign halfway through.
        let tangents: Vec<Vector3<f64>> = (0..40)
            .map(|i| {
                let t = (i as f64 - 20.0) * 0.15;
                Vector3::new(1.0, t, t * t * 0.5 - 1.0).normalize()
            })
            .collect();
        let frames = compute_bishop_frames(&tangents);
        for k in 0..frames.len() - 1 {
            let (a, b) = (&frames[k], &frames[k + 1]);
            let tangent_turn = a.tangent.dot(&b.tangent).clamp(-1.0, 1.0).acos();
            let normal_turn = a.normal1.dot(&b.normal1).clamp(-1.0, 1.0).acos();
            assert!(
                normal_turn <= tangent_turn + 1e-6,
                "frame {k}: normal turned {normal_turn} rad while the tangent turned \
                 {tangent_turn} rad — the frame is twisting"
            );
        }
    }

    /// Orthonormality must survive a long, highly curved path: the transport step renormalises,
    /// so drift should not accumulate over hundreds of samples.
    #[test]
    fn frames_do_not_drift_over_a_long_path() {
        let tangents: Vec<Vector3<f64>> = (0..500)
            .map(|i| {
                let t = i as f64 * 0.11;
                Vector3::new(t.cos(), t.sin(), (t * 0.37).sin() * 0.6).normalize()
            })
            .collect();
        let frames = compute_bishop_frames(&tangents);
        let last = frames.last().unwrap();
        assert!(
            last.tangent.dot(&last.normal1).abs() < 1e-9,
            "drifted out of orthogonality"
        );
        assert!(last.normal1.dot(&last.normal2).abs() < 1e-9);
        assert!((last.normal1.norm() - 1.0).abs() < 1e-9);
    }

    /// A straight run must carry the same frame throughout — the degenerate branch of the
    /// parallel-transport step (|t × t'| ≈ 0) is the one that silently produces NaNs.
    #[test]
    fn straight_segments_keep_a_constant_finite_frame() {
        let tangents = vec![Vector3::new(0.0, 0.0, 1.0); 8];
        let frames = compute_bishop_frames(&tangents);
        assert_eq!(frames.len(), 8);
        for f in &frames {
            assert!(f.normal1.iter().all(|v| v.is_finite()));
            assert!((f.normal1 - frames[0].normal1).norm() < 1e-9);
            assert!((f.normal2 - frames[0].normal2).norm() < 1e-9);
        }
    }

    /// (tangent, normal1, normal2) must be right-handed everywhere, or the extruded ribbon
    /// turns inside out and the surface normals point into the mesh.
    #[test]
    fn frames_are_right_handed() {
        let tangents: Vec<Vector3<f64>> = (0..16)
            .map(|i| {
                let t = i as f64 * 0.4;
                Vector3::new(t.cos(), t.sin() * 0.5, (t * 0.3).sin()).normalize()
            })
            .collect();
        for (k, f) in compute_bishop_frames(&tangents).iter().enumerate() {
            let handedness = f.normal1.cross(&f.normal2).dot(&f.tangent);
            assert!(
                handedness < -0.99 || handedness > 0.99,
                "frame {k} is degenerate (|n1 × n2 · t| = {})",
                handedness.abs()
            );
            assert!(
                handedness.signum()
                    == compute_bishop_frames(&tangents)[0]
                        .normal1
                        .cross(&compute_bishop_frames(&tangents)[0].normal2)
                        .dot(&compute_bishop_frames(&tangents)[0].tangent)
                        .signum(),
                "frame {k} flipped handedness relative to the first"
            );
        }
    }

    #[test]
    fn empty_and_single_tangent_inputs_are_handled() {
        assert!(compute_bishop_frames(&[]).is_empty());
        let one = compute_bishop_frames(&[Vector3::new(0.0, 1.0, 0.0)]);
        assert_eq!(one.len(), 1);
        assert!((one[0].tangent.dot(&one[0].normal1)).abs() < 1e-9);
    }
}
