//! The ribbon as a binary glTF 2.0 file (`.glb`), for Blender, Keynote, three.js, a slide.
//!
//! The mesh is the one every Proteus renderer draws (Bishop-frame cartoon, DSSP, disulfide
//! sticks), with the colour scheme baked in as vertex colours from [`Rasterizer::base_color`],
//! so an export in `plddt` matches the terminal and the browser page. Shading is left to the
//! importer: the colours are unlit base colours, the material is a plain dielectric.
//!
//! Units are ångström and the coordinates are the file's own (not centred), so two exports of
//! superposed structures still line up. glTF's nominal unit is the metre; scale on import if a
//! tool cares. Colours are converted from sRGB to linear, as glTF vertex colours require.

use crate::brand;
use crate::geometry::mesh::TriangleMesh;
use crate::rasterizer::{buffer::ColorRGB, shader::ColorScheme, Rasterizer};
use crate::{tui, StructureRenderData};

/// Binary glTF of `structure`'s ribbon (and disulfides, when any) coloured by `scheme`.
pub fn structure_glb(structure: &StructureRenderData, scheme: ColorScheme) -> Vec<u8> {
    let mut rasterizer = Rasterizer::new(scheme);
    if let Some(v) = structure.default_interface() {
        rasterizer.interface_colors = v.residue_colors(&structure.residue_labels);
    }
    if let Some(sc) = tui::ScoreColors::of(structure) {
        rasterizer.residue_colors = sc.residue_colors;
    }
    let mut parts: Vec<(&str, &TriangleMesh, ColorScheme)> =
        vec![("ribbon", &structure.ribbon_mesh, scheme)];
    if let Some(ds) = &structure.disulfide_mesh {
        parts.push((
            "disulfides",
            ds,
            ColorScheme::Solid(brand::structure::DISULFIDE),
        ));
    }
    let meshes: Vec<GlbMesh> = parts
        .into_iter()
        .filter(|(_, m, _)| !m.vertices.is_empty() && !m.indices.is_empty())
        .map(|(name, mesh, scheme)| {
            let total = mesh
                .vertices
                .iter()
                .map(|v| v.residue_index)
                .max()
                .unwrap_or(0)
                + 1;
            GlbMesh {
                name,
                positions: mesh
                    .vertices
                    .iter()
                    .map(|v| [v.position.x, v.position.y, v.position.z])
                    .collect(),
                normals: mesh
                    .vertices
                    .iter()
                    .map(|v| {
                        let n = v.normal.try_normalize(1e-12).unwrap_or(v.normal);
                        [n.x, n.y, n.z]
                    })
                    .collect(),
                colors: mesh
                    .vertices
                    .iter()
                    .map(|v| linear(rasterizer.base_color(scheme, v, total)))
                    .collect(),
                indices: mesh.indices.iter().flatten().copied().collect(),
            }
        })
        .collect();
    write_glb(&meshes)
}

struct GlbMesh<'a> {
    name: &'a str,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

fn linear(c: ColorRGB) -> [f32; 3] {
    let f = |u: u8| {
        let s = u as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    [f(c.r), f(c.g), f(c.b)]
}

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;

/// Assemble the GLB container: a JSON chunk and one BIN chunk holding every attribute.
fn write_glb(meshes: &[GlbMesh]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut gltf_meshes = Vec::new();
    let mut nodes = Vec::new();

    let mut push_view = |bin: &mut Vec<u8>, bytes: &[u8], target: u32| -> usize {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{},"target":{target}}}"#,
            bytes.len()
        ));
        views.len() - 1
    };

    for (mi, m) in meshes.iter().enumerate() {
        let vec3 = |v: &[[f32; 3]]| -> Vec<u8> {
            v.iter()
                .flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes()))
                .collect()
        };
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &m.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let n = m.positions.len();
        let pv = push_view(&mut bin, &vec3(&m.positions), ARRAY_BUFFER);
        let nv = push_view(&mut bin, &vec3(&m.normals), ARRAY_BUFFER);
        let cv = push_view(&mut bin, &vec3(&m.colors), ARRAY_BUFFER);
        let idx: Vec<u8> = m.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let iv = push_view(&mut bin, &idx, ELEMENT_ARRAY_BUFFER);
        let base = accessors.len();
        accessors.push(format!(
            r#"{{"bufferView":{pv},"componentType":{FLOAT},"count":{n},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        ));
        accessors.push(format!(
            r#"{{"bufferView":{nv},"componentType":{FLOAT},"count":{n},"type":"VEC3"}}"#
        ));
        accessors.push(format!(
            r#"{{"bufferView":{cv},"componentType":{FLOAT},"count":{n},"type":"VEC3"}}"#
        ));
        accessors.push(format!(
            r#"{{"bufferView":{iv},"componentType":{UNSIGNED_INT},"count":{},"type":"SCALAR"}}"#,
            m.indices.len()
        ));
        gltf_meshes.push(format!(
            r#"{{"name":"{}","primitives":[{{"attributes":{{"POSITION":{},"NORMAL":{},"COLOR_0":{}}},"indices":{},"material":0,"mode":4}}]}}"#,
            m.name,
            base,
            base + 1,
            base + 2,
            base + 3
        ));
        nodes.push(format!(r#"{{"name":"{}","mesh":{mi}}}"#, m.name));
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let scene_nodes: Vec<String> = (0..nodes.len()).map(|i| i.to_string()).collect();
    let mut json = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"proteus {}","extras":{{"units":"angstrom"}}}},"#,
            r#""scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}],"meshes":[{}],"#,
            r#""materials":[{{"name":"proteus","pbrMetallicRoughness":{{"baseColorFactor":[1,1,1,1],"metallicFactor":0,"roughnessFactor":0.45}},"doubleSided":true}}],"#,
            r#""accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#
        ),
        env!("CARGO_PKG_VERSION"),
        scene_nodes.join(","),
        nodes.join(","),
        gltf_meshes.join(","),
        accessors.join(","),
        views.join(","),
        bin.len()
    );
    while !json.len().is_multiple_of(4) {
        json.push(' ');
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crambin() -> StructureRenderData {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        ))
        .unwrap();
        crate::parse_pdb_structure(&text).unwrap()
    }

    fn chunks(glb: &[u8]) -> (serde_json::Value, &[u8]) {
        let u = |o: usize| u32::from_le_bytes(glb[o..o + 4].try_into().unwrap()) as usize;
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u(4), 2, "glTF version");
        assert_eq!(u(8), glb.len(), "header length is the file length");
        let jlen = u(12);
        assert_eq!(&glb[16..20], b"JSON");
        assert_eq!(jlen % 4, 0);
        let json: serde_json::Value = serde_json::from_slice(&glb[20..20 + jlen]).unwrap();
        let b = 20 + jlen;
        assert_eq!(&glb[b + 4..b + 8], b"BIN\0");
        (json, &glb[b + 8..b + 8 + u(b)])
    }

    /// The file is valid GLB, its accessors agree with the mesh it came from, and every index
    /// points at a vertex. Crambin has three disulfides, so it carries two meshes.
    #[test]
    fn crambin_exports_as_well_formed_glb() {
        let s = crambin();
        let glb = structure_glb(&s, ColorScheme::SecondaryStructure);
        let (json, bin) = chunks(&glb);
        assert_eq!(json["asset"]["version"], "2.0");
        let meshes = json["meshes"].as_array().unwrap();
        assert_eq!(meshes.len(), 1 + usize::from(s.disulfide_mesh.is_some()));
        let acc = json["accessors"].as_array().unwrap();
        let views = json["bufferViews"].as_array().unwrap();
        assert_eq!(acc[0]["count"], s.ribbon_mesh.vertices.len());
        assert_eq!(acc[3]["count"], s.ribbon_mesh.indices.len() * 3);
        assert_eq!(json["buffers"][0]["byteLength"], bin.len());
        // indices in range
        let iv = &views[acc[3]["bufferView"].as_u64().unwrap() as usize];
        let off = iv["byteOffset"].as_u64().unwrap() as usize;
        let len = iv["byteLength"].as_u64().unwrap() as usize;
        let n = s.ribbon_mesh.vertices.len() as u32;
        assert!(bin[off..off + len]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|c| u32::from_le_bytes(*c) < n));
        // POSITION min/max bound the vertices
        let x0 = acc[0]["min"][0].as_f64().unwrap() as f32;
        assert!(s.ribbon_mesh.vertices.iter().all(|v| v.position.x >= x0));
    }

    /// The export tessellation follows the same residues as the screen one, only finer.
    #[test]
    fn export_detail_is_finer_over_the_same_residues() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../proteus-core/tests/data/1crn.pdb"
        ))
        .unwrap();
        let screen = crate::parse_pdb_structure(&text).unwrap();
        let export = crate::parse_pdb_structure_at(&text, crate::RibbonDetail::EXPORT).unwrap();
        assert!(export.ribbon_mesh.vertices.len() > 5 * screen.ribbon_mesh.vertices.len());
        let span = |m: &TriangleMesh| {
            let r = m.vertices.iter().map(|v| v.residue_index);
            (r.clone().min(), r.max())
        };
        assert_eq!(span(&screen.ribbon_mesh), span(&export.ribbon_mesh));
        assert_eq!(screen.dssp, export.dssp);
    }

    /// Vertex colours are the renderer's base colours, linearised: the first vertex's colour
    /// read back from the file converts to the sRGB the terminal would start from.
    #[test]
    fn vertex_colours_are_the_renderers_base_colours_in_linear() {
        let s = crambin();
        let glb = structure_glb(&s, ColorScheme::SecondaryStructure);
        let (json, bin) = chunks(&glb);
        let cv =
            &json["bufferViews"][json["accessors"][2]["bufferView"].as_u64().unwrap() as usize];
        let off = cv["byteOffset"].as_u64().unwrap() as usize;
        let f =
            |k: usize| f32::from_le_bytes(bin[off + 4 * k..off + 4 * k + 4].try_into().unwrap());
        let r = Rasterizer::new(ColorScheme::SecondaryStructure);
        let want = r.base_color(
            ColorScheme::SecondaryStructure,
            &s.ribbon_mesh.vertices[0],
            1,
        );
        let lin = linear(want);
        for (k, want) in lin.iter().enumerate() {
            assert!((f(k) - want).abs() < 1e-6);
        }
        assert!((0..3).all(|k| (0.0..=1.0).contains(&f(k))));
    }
}
