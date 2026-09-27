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
