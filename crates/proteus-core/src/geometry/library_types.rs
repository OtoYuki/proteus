//! Shapes of the generated restraint tables in `library.rs`.

/// An ideal bond length: atom names, target (Å) and estimated standard deviation (Å).
#[derive(Debug, Clone, Copy)]
pub(crate) struct BondDef {
    pub a: &'static str,
    pub b: &'static str,
    pub ideal: f64,
    pub esd: f64,
}

/// An ideal bond angle a–b–c (vertex `b`), degrees.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AngleDef {
    pub a: &'static str,
    pub b: &'static str,
    pub c: &'static str,
    pub ideal: f64,
    pub esd: f64,
}

/// A chiral centre `atoms[0]` with its three neighbours, and the signed ideal chiral volume
/// (Å³) for that atom order, computed by cctbx from the monomer's own ideal bonds and angles.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChiralDef {
    pub atoms: [&'static str; 4],
    pub ideal: f64,
    pub both_signs: bool,
}

/// A planar group: atom names with their out-of-plane esd (Å).
pub(crate) type PlaneDef = &'static [(&'static str, f64)];

/// Heavy-atom restraints of one monomer.
#[derive(Debug)]
pub(crate) struct ResidueDef {
    pub name: &'static str,
    pub bonds: &'static [BondDef],
    pub angles: &'static [AngleDef],
    pub chiralities: &'static [ChiralDef],
    pub planes: &'static [PlaneDef],
}

/// An atom of a link: `1` for the preceding residue, `2` for the following one.
pub(crate) type LinkAtom = (u8, &'static str);

#[derive(Debug, Clone, Copy)]
pub(crate) struct LinkBondDef {
    pub a: LinkAtom,
    pub b: LinkAtom,
    pub ideal: f64,
    pub esd: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LinkAngleDef {
    pub atoms: [LinkAtom; 3],
    pub ideal: f64,
    pub esd: f64,
}

/// A peptide link between consecutive residues.
#[derive(Debug)]
pub(crate) struct LinkDef {
    pub id: &'static str,
    pub bonds: &'static [LinkBondDef],
    pub angles: &'static [LinkAngleDef],
    pub planes: &'static [&'static [(u8, &'static str, f64)]],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModBondDef {
    /// `true` adds the restraint, `false` replaces the monomer's restraint on the same atoms.
    pub add: bool,
    pub bond: BondDef,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ModAngleDef {
    pub add: bool,
    pub angle: AngleDef,
}

/// A monomer modification (here only the C-terminal carboxylate).
#[derive(Debug)]
pub(crate) struct ModDef {
    pub bonds: &'static [ModBondDef],
    pub angles: &'static [ModAngleDef],
    pub planes: &'static [PlaneDef],
}
