use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct SplinePoint {
    pub position: Vector3<f64>,
    pub tangent: Vector3<f64>,
    pub residue_index: usize,
    pub parameter: f64, // Normalized progress along backbone segment [0.0, 1.0]
}

/// Interpolates a series of 3D control points (typically C-alpha backbone atoms)
/// using a Centripetal Catmull-Rom Spline (alpha = 0.5).
/// Centripetal parameterization guarantees no self-intersections or cusps.
pub fn interpolate_catmull_rom(points: &[Vector3<f64>], subdivisions: usize) -> Vec<SplinePoint> {
    let n = points.len();
    if n < 2 {
        return points
            .iter()
            .enumerate()
            .map(|(i, &p)| SplinePoint {
                position: p,
                tangent: Vector3::new(0.0, 0.0, 1.0),
                residue_index: i,
                parameter: 0.0,
            })
            .collect();
    }

    let mut result = Vec::new();

    for i in 0..(n - 1) {
        let p0 = if i == 0 {
            points[0] + (points[0] - points[1])
        } else {
            points[i - 1]
        };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 < n {
            points[i + 2]
        } else {
            p2 + (p2 - p1)
        };

        let alpha = 0.5; // Centripetal
        let t0 = 0.0;
        let t1 = t0 + (p1 - p0).norm().powf(alpha).max(1e-4);
        let t2 = t1 + (p2 - p1).norm().powf(alpha).max(1e-4);
        let t3 = t2 + (p3 - p2).norm().powf(alpha).max(1e-4);

        let steps = if i == n - 2 {
            subdivisions + 1
        } else {
            subdivisions
        };

        let pts = [p0, p1, p2, p3];
        let ts = [t0, t1, t2, t3];

        for s in 0..steps {
            let u = s as f64 / subdivisions as f64;
            let t = t1 + u * (t2 - t1);

            let pos = eval_barry_goldman(pts, ts, t);

            // Numerical tangent with central difference
            let dt = 1e-4 * (t2 - t1);
            let pos_fwd = eval_barry_goldman(pts, ts, t + dt);
            let pos_bwd = eval_barry_goldman(pts, ts, t - dt);
            let delta = pos_fwd - pos_bwd;
            // With fully coincident control points the central difference vanishes and
            // `normalize()` yields NaN, which would poison the frames, the mesh and every
            // vertex downstream. The knot-spacing clamp above keeps realistic structures off
            // this path — a PDB with one duplicated C-alpha still renders — but the function is
            // public API, so carry the previous tangent rather than emit NaN.
            let tangent = if delta.norm() > 1e-12 {
                delta.normalize()
            } else {
                result
                    .last()
                    .map(|prev: &SplinePoint| prev.tangent)
                    .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0))
            };

            result.push(SplinePoint {
                position: pos,
                tangent,
                residue_index: i,
                parameter: u,
            });
        }
    }

    result
}

#[inline]
fn eval_barry_goldman(pts: [Vector3<f64>; 4], ts: [f64; 4], t: f64) -> Vector3<f64> {
    let [p0, p1, p2, p3] = pts;
    let [t0, t1, t2, t3] = ts;

    let a1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
    let a2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
    let a3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));

    let b1 = a1 * ((t2 - t) / (t2 - t0)) + a2 * ((t - t0) / (t2 - t0));
    let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));

    b1 * ((t2 - t) / (t2 - t1)) + b2 * ((t - t1) / (t2 - t1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spline_endpoints_interpolation() {
        let pts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(3.0, 2.0, 0.0),
        ];

        let curve = interpolate_catmull_rom(&pts, 8);
        assert!(!curve.is_empty());

        // First point should match pts[0] closely
        assert!((curve.first().unwrap().position - pts[0]).norm() < 1e-3);
        // Last point should match pts[3] closely
        assert!((curve.last().unwrap().position - pts[3]).norm() < 1e-3);
    }

    /// Catmull–Rom is *interpolating*: the curve must pass through every control point, not
    /// just the two ends. A wrong tangent scale or an off-by-one in the knot windows shows up
    /// here and nowhere else.
    #[test]
    fn spline_passes_through_every_control_point() {
        let pts = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, -1.0),
            Vector3::new(2.0, 0.0, 1.5),
            Vector3::new(3.5, 2.0, 0.0),
            Vector3::new(5.0, -1.0, 2.0),
        ];
        let curve = interpolate_catmull_rom(&pts, 8);
        for (i, p) in pts.iter().enumerate() {
            let closest = curve
                .iter()
                .map(|c| (c.position - p).norm())
                .fold(f64::MAX, f64::min);
            assert!(
                closest < 1e-6,
                "control point {i} is {closest} from the curve"
            );
        }
    }

    /// Tangents must be unit length and point along the direction of travel, or the Bishop
    /// frames built from them are meaningless.
    #[test]
    fn tangents_are_unit_and_forward_facing() {
        let pts: Vec<Vector3<f64>> = (0..6)
            .map(|i| Vector3::new(i as f64, (i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let curve = interpolate_catmull_rom(&pts, 6);
        for (i, sp) in curve.iter().enumerate() {
            assert!(
                (sp.tangent.norm() - 1.0).abs() < 1e-6,
                "tangent {i} has length {}",
                sp.tangent.norm()
            );
            if i + 1 < curve.len() {
                let step = curve[i + 1].position - sp.position;
                if step.norm() > 1e-9 {
                    assert!(
                        sp.tangent.dot(&step.normalize()) > 0.5,
                        "tangent {i} points away from the next sample"
                    );
                }
            }
        }
    }

    /// Sample density and monotone progress: a spline over N control points with S subdivisions
    /// must produce strictly advancing samples, never a duplicate or a reversal.
    #[test]
    fn samples_advance_monotonically() {
        let pts: Vec<Vector3<f64>> = (0..5)
            .map(|i| Vector3::new(i as f64 * 3.8, 0.0, 0.0))
            .collect();
        let curve = interpolate_catmull_rom(&pts, 4);
        assert!(
            curve.len() >= pts.len(),
            "curve is coarser than its control polygon"
        );
        for w in curve.windows(2) {
            assert!(
                w[1].position.x > w[0].position.x - 1e-9,
                "sample went backwards: {} then {}",
                w[0].position.x,
                w[1].position.x
            );
        }
        assert!(curve.iter().all(|c| c.residue_index < pts.len()));
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert!(interpolate_catmull_rom(&[], 4).is_empty());
        assert_eq!(interpolate_catmull_rom(&[Vector3::zeros()], 4).len(), 1);
        // Coincident control points: zero-length segments must not produce NaN tangents.
        let same = vec![Vector3::new(1.0, 1.0, 1.0); 4];
        let curve = interpolate_catmull_rom(&same, 4);
        assert!(curve
            .iter()
            .all(|c| c.tangent.iter().all(|v| v.is_finite())));
    }
}
