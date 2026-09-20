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
            let tangent = (pos_fwd - pos_bwd).normalize();

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
}
