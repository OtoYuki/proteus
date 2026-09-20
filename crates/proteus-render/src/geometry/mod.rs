pub mod frame;
pub mod mesh;
pub mod spline;

pub use frame::{compute_bishop_frames, BishopFrame};
pub use mesh::{generate_cartoon_mesh, TriangleMesh, Vertex3D};
pub use spline::{interpolate_catmull_rom, SplinePoint};
