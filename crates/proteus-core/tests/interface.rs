//! Interface metrics on trypsin–BPTI (2PTC, chains E and I).

use proteus_core::interface::sc::{shape_complementarity, ScAtom};
use proteus_core::interface::{interface_metrics, InterfaceSpec};
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// The atoms `sc-rs`'s own command line takes from a PDB file: ATOM records, first altloc, and
/// its hydrogen filter (which also drops names ending in H, such as TYR OH).
fn sc_rs_atoms(chain: &str) -> Vec<ScAtom> {
    std::fs::read_to_string(data("2ptc_EI.pdb"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("ATOM") && l.len() >= 54 && &l[21..22] == chain)
        .filter(|l| matches!(l.as_bytes()[16], b' ' | b'A'))
        .filter_map(|l| {
            let name = l[12..16].trim().to_string();
            let element = l.get(76..78).unwrap_or("").trim();
            let hydrogen = element.eq_ignore_ascii_case("H")
                || name.starts_with('H')
                || name.ends_with('H')
                || (name.contains('H') && name.starts_with(|c: char| c.is_ascii_digit()));
            (!hydrogen).then(|| ScAtom {
                residue: l[17..20].trim().into(),
                name,
                coord: [
                    l[30..38].trim().parse().unwrap(),
                    l[38..46].trim().parse().unwrap(),
                    l[46..54].trim().parse().unwrap(),
                ],
            })
        })
        .collect()
}

/// The port reproduces sc-rs (commit befcc6c) on the same atoms: `sc 2ptc_EI.pdb E I --json`
/// printed sc 0.758366028510063, median distance 0.46998312487395555, trimmed area
/// 968.7313951893929 on 1 619 + 450 atoms.
#[test]
fn shape_complementarity_matches_sc_rs() {
    let (e, i) = (sc_rs_atoms("E"), sc_rs_atoms("I"));
    assert_eq!((e.len(), i.len()), (1619, 450));
    let r = shape_complementarity(&e, &i).unwrap();
    assert!((r.sc - 0.758366028510063).abs() < 1e-12, "{}", r.sc);
    assert!((r.median_distance - 0.46998312487395555).abs() < 1e-12);
    assert!((r.trimmed_area - 968.7313951893929).abs() < 1e-9);
    // The two surfaces in the other order score the same interface.
    let swapped = shape_complementarity(&i, &e).unwrap();
    assert!(
        (swapped.sc - r.sc).abs() < 0.02,
        "{} vs {}",
        swapped.sc,
        r.sc
    );
}

#[test]
fn trypsin_bpti_interface() {
    let pdb = proteus_core::io::open_structure(&data("2ptc_EI.pdb")).unwrap();
    let m = interface_metrics(&pdb, &InterfaceSpec::parse("I:E").unwrap(), None).unwrap();
    assert_eq!(
        (m.binder_chains.as_str(), m.target_chains.as_str()),
        ("I", "E")
    );
    // BPTI buries ~1 400 Å² against trypsin (Janin & Chothia); Lys15 in the S1 pocket.
    assert!((1200.0..1700.0).contains(&m.dsasa), "dSASA {}", m.dsasa);
    let sc = m.shape_complementarity.unwrap();
    assert!((0.70..0.80).contains(&sc), "Sc {sc}");
    assert!(m.binder_interface_residues >= 10 && m.target_interface_residues >= 15);
    assert!(m.interface_hbonds >= 5, "{}", m.interface_hbonds);
    assert_eq!(m.ipae, None);
    // "I" alone means BPTI against everything else, which here is trypsin.
    let auto = interface_metrics(&pdb, &InterfaceSpec::parse("I").unwrap(), None).unwrap();
    assert_eq!(auto.target_chains, "E");
    assert_eq!(auto.dsasa, m.dsasa);
    // A chain that is not there is named.
    let err = interface_metrics(&pdb, &InterfaceSpec::parse("X:E").unwrap(), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no protein chain 'X'"), "{err}");
}

/// A float32 `.npy` of `shape`, as numpy writes it.
fn npy_f32(shape: &[usize], values: &[f32]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        dims.join(", ")
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((header.len() as u16).to_le_bytes());
    b.extend(header.as_bytes());
    for v in values {
        b.extend(v.to_le_bytes());
    }
    b
}

/// A Boltz-style `pae_<model>.npz` beside the model switches the PAE columns on, and the numbers
/// follow from the matrix: BPTI (rows 223..281) sees trypsin at 2 Å, trypsin sees BPTI at 20 Å.
#[test]
fn a_pae_file_beside_the_model_gives_ipsae() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("complex_model_0.pdb");
    std::fs::copy(data("2ptc_EI.pdb"), &model).unwrap();
    let pdb = proteus_core::io::open_structure(&model).unwrap();
    let chains: Vec<(String, usize)> = proteus_core::io::protein_heavy_atoms(&pdb)
        .chains()
        .map(|c| (c.id().to_string(), c.residue_count()))
        .collect();
    assert_eq!(chains, vec![("E".to_string(), 223), ("I".to_string(), 58)]);
    let n = 281;
    let chain = |i: usize| if i < 223 { 'E' } else { 'I' };
    let mut values = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            values.push(match (chain(i), chain(j)) {
                ('I', 'E') => 2.0,
                ('E', 'I') => 20.0,
                _ => 1.0,
            });
        }
    }
    let mut zip_bytes = std::io::Cursor::new(Vec::new());
    {
        let mut w = zip::ZipWriter::new(&mut zip_bytes);
        w.start_file("pae.npy", zip::write::SimpleFileOptions::default())
            .unwrap();
        std::io::Write::write_all(&mut w, &npy_f32(&[n, n], &values)).unwrap();
        w.finish().unwrap();
    }
    std::fs::write(
        dir.path().join("pae_complex_model_0.npz"),
        zip_bytes.into_inner(),
    )
    .unwrap();

    let spec = InterfaceSpec::parse("I:E").unwrap();
    let qc = proteus_core::qc::structure_qc_with(
        &model,
        &proteus_core::qc::QcOptions {
            interface: Some(&spec),
            ..Default::default()
        },
    )
    .unwrap();
    let i = &qc.interface;
    // Both directions: (2 + 20) / 2 over equal numbers of pairs.
    assert_eq!(i.ipae, Some(11.0));
    // I→E: 223 partners under 10 Å, d0 = 1.24·208^(1/3) − 1.8; E→I: none under 10 Å.
    let d0 = 1.24 * 208f64.cbrt() - 1.8;
    let expected = 1.0 / (1.0 + (2.0 / d0).powi(2));
    assert!(
        (i.ipsae_max.unwrap() - expected).abs() < 1e-9,
        "{:?}",
        i.ipsae_max
    );
    assert_eq!(i.ipsae_min, Some(0.0));
    assert!((i.lis.unwrap() - (10.0 / 12.0) / 2.0).abs() < 1e-6);
    assert!(i.interface_sc.is_some() && i.interface_dsasa.is_some());
}

/// One design from the binder meta-analysis dataset, end to end: its Boltz-1 files under their
/// original names, `structure_qc_with` finding the PAE and confidences beside the model, and the
/// dataset's own values for this design (final_dataset.csv): boltz1_ipSAE_min 0.75,
/// boltz1_ipSAE_max 0.804, boltz1_ipae 3.91, boltz1_iptm_model_0 0.891. It was measured as a
/// non-binder: a confident interface is not a guarantee. `make validate-binders` does the same
/// over the AlphaFold 3 models of all 3 669 designs.
#[test]
fn a_dataset_design_reproduces_the_dataset_values() {
    use std::io::Read;
    let name = "ems_3hC_1044_0001_000000014_0001";
    let dir = tempfile::tempdir().unwrap();
    let src = data("binders");
    let model = dir.path().join(format!("{name}_model_0.cif"));
    let mut text = Vec::new();
    flate2::read::GzDecoder::new(
        &std::fs::read(src.join(format!("{name}_model_0.cif.gz"))).unwrap()[..],
    )
    .read_to_end(&mut text)
    .unwrap();
    std::fs::write(&model, text).unwrap();
    for f in [
        format!("pae_{name}_model_0.npz"),
        format!("confidence_{name}_model_0.json"),
    ] {
        std::fs::copy(src.join(&f), dir.path().join(&f)).unwrap();
    }
    let spec = InterfaceSpec::parse("A").unwrap();
    let qc = proteus_core::qc::structure_qc_with(
        &model,
        &proteus_core::qc::QcOptions {
            interface: Some(&spec),
            ..Default::default()
        },
    )
    .unwrap();
    let i = &qc.interface;
    assert_eq!(i.interface_target.as_deref(), Some("B"));
    let near = |ours: Option<f64>, theirs: f64, tol: f64| {
        let v = ours.unwrap();
        assert!((v - theirs).abs() <= tol, "{v} vs {theirs}");
    };
    // The CSV rounds to three decimals.
    near(i.ipsae_min, 0.75, 0.0006);
    near(i.ipsae_max, 0.804, 0.0006);
    near(i.ipae, 3.91, 0.0006);
    near(i.iptm, 0.891, 0.0006);
}

/// Unpack a directory of predictor outputs into a temporary one, un-gzipping `*.gz`.
fn unpack(src: &std::path::Path) -> tempfile::TempDir {
    use std::io::Read;
    let dir = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let bytes = std::fs::read(e.path()).unwrap();
        match name.strip_suffix(".gz") {
            Some(plain) => {
                let mut out = Vec::new();
                flate2::read::GzDecoder::new(&bytes[..])
                    .read_to_end(&mut out)
                    .unwrap();
                std::fs::write(dir.path().join(plain), out).unwrap();
            }
            None => std::fs::write(dir.path().join(name), bytes).unwrap(),
        }
    }
    dir
}

/// Protenix output as written (tests/data/predictors/NOTICE): the files are found beside the
/// model, the SEP's per-atom tokens and the ATP's are skipped, and the values equal those
/// computed from Protenix's own atom-to-token map.
#[test]
fn protenix_output_with_a_modified_residue_and_a_ligand() {
    let dir = unpack(&data("predictors/protenix"));
    let spec = InterfaceSpec::parse("A:B").unwrap();
    let qc = proteus_core::qc::structure_qc_with(
        &dir.path().join("sepatp_sample_0.cif"),
        &proteus_core::qc::QcOptions {
            interface: Some(&spec),
            ..Default::default()
        },
    )
    .unwrap();
    let i = &qc.interface;
    assert_eq!(i.interface_note, None);
    let near = |ours: Option<f64>, theirs: f64| {
        let v = ours.unwrap();
        assert!((v - theirs).abs() < 1e-6, "{v} vs {theirs}");
    };
    near(i.ipsae_max, 0.016426);
    near(i.ipsae_min, 0.013679);
    near(i.lis, 0.168540);
    near(i.ipae, 15.095940);
    near(i.iptm, 0.265748);
}
