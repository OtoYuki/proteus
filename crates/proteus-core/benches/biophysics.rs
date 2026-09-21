//! Criterion benchmarks for the biophysics kernels, on the same structures and metrics as
//! `bench/python_baseline.py` so the two tables are comparable. Run `bench/run.sh`.

use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// (label, path relative to the workspace root). Corpus files are optional (skipped if absent).
const STRUCTURES: &[(&str, &str)] = &[
    ("1crn", "crates/proteus-core/tests/data/1crn.pdb"),
    ("1ubq", "validate/corpus/1ubq.pdb"),
    ("4hhb", "validate/corpus/4hhb.pdb"),
    ("6vxx", "validate/corpus/6vxx.cif"),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Loaded {
    label: &'static str,
    pdb: pdbtbx::PDB,
    atoms: Vec<proteus_core::sasa::AtomDescriptor>,
    backbone: Vec<proteus_core::backbone::BackboneResidue>,
    n_atoms: u64,
}

fn load_all() -> Vec<Loaded> {
    STRUCTURES
        .iter()
        .filter_map(|(label, rel)| {
            let path = root().join(rel);
            if !path.exists() {
                eprintln!(
                    "skip {label}: {} missing (run `make fetch`)",
                    path.display()
                );
                return None;
            }
            let raw = proteus_core::io::open_structure(&path).unwrap();
            let pdb = proteus_core::io::protein_heavy_atoms(&raw);
            let atoms: Vec<_> = pdb
                .atoms()
                .map(|a| {
                    proteus_core::sasa::AtomDescriptor::new(
                        nalgebra::Vector3::new(a.x(), a.y(), a.z()),
                        proteus_core::io::element_symbol(a),
                    )
                })
                .collect();
            let backbone = proteus_core::backbone::extract_backbone(&pdb);
            let n_atoms = atoms.len() as u64;
            Some(Loaded {
                label,
                pdb,
                atoms,
                backbone,
                n_atoms,
            })
        })
        .collect()
}

fn bench_kernels(c: &mut Criterion) {
    let structures = load_all();

    let mut g = c.benchmark_group("sasa_shrake_rupley_960");
    for s in &structures {
        g.throughput(Throughput::Elements(s.n_atoms));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::sasa::compute_sasa_with_points(&s.atoms, 960))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("sasa_shrake_rupley_96");
    for s in &structures {
        g.throughput(Throughput::Elements(s.n_atoms));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::sasa::compute_sasa_with_points(&s.atoms, 96))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("dssp");
    for s in &structures {
        g.throughput(Throughput::Elements(s.backbone.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::structure::assign_secondary_structure(&s.backbone))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("phi_psi_ramachandran");
    for s in &structures {
        g.throughput(Throughput::Elements(s.backbone.len() as u64));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| {
                let bb = &s.backbone;
                let mut pts = Vec::with_capacity(bb.len());
                for i in 0..bb.len() {
                    let phi = if i > 0 {
                        proteus_core::backbone::phi(&bb[i - 1], &bb[i])
                    } else {
                        None
                    };
                    let psi = if i + 1 < bb.len() {
                        proteus_core::backbone::psi(&bb[i], &bb[i + 1])
                    } else {
                        None
                    };
                    let omega = if i > 0 {
                        proteus_core::backbone::omega(&bb[i - 1], &bb[i])
                    } else {
                        None
                    };
                    let next = if i + 1 < bb.len() {
                        Some(bb[i + 1].name.as_str())
                    } else {
                        None
                    };
                    let class =
                        proteus_core::rama8000::RamaClass::classify(&bb[i].name, next, omega);
                    pts.push((phi, psi, class));
                }
                proteus_core::structure::evaluate_ramachandran(&pts)
            })
        });
    }
    g.finish();

    let mut g = c.benchmark_group("steric_overlap");
    for s in &structures {
        g.throughput(Throughput::Elements(s.n_atoms));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::clash::compute_steric_overlap(&s.pdb))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("interaction_network");
    for s in &structures {
        g.throughput(Throughput::Elements(s.n_atoms));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::interactions::compute_interaction_network(&s.pdb))
        });
    }
    g.finish();

    let mut g = c.benchmark_group("full_profile");
    for s in &structures {
        g.throughput(Throughput::Elements(s.n_atoms));
        g.bench_with_input(BenchmarkId::from_parameter(s.label), s, |b, s| {
            b.iter(|| proteus_core::metrics::analyze_pdb_detailed(&s.pdb, None).unwrap())
        });
    }
    g.finish();

    let mut g = c.benchmark_group("kabsch_rmsd");
    for s in &structures {
        let ca: Vec<nalgebra::Vector3<f64>> = s.backbone.iter().filter_map(|r| r.ca).collect();
        // Rotate + translate a copy so the alignment is non-trivial.
        let rot = nalgebra::Rotation3::from_euler_angles(0.3, -0.7, 1.1);
        let moved: Vec<nalgebra::Vector3<f64>> = ca
            .iter()
            .map(|p| rot * p + nalgebra::Vector3::new(5.0, -3.0, 2.0))
            .collect();
        g.throughput(Throughput::Elements(ca.len() as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(s.label),
            &(ca, moved),
            |b, (p, q)| b.iter(|| proteus_core::metrics::compute_kabsch_rmsd(p, q).unwrap()),
        );
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_kernels
}
criterion_main!(benches);
