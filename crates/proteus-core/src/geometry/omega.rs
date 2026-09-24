//! Peptide ω classification (MolProbity omegalyze; Williams et al. 2018): every peptide bond is
//! trans, cis or twisted, and anything but trans is reported. Cis bonds before a proline are
//! common (~5 % of prolines); cis bonds elsewhere occur about once in 3,000 residues, and
//! twisted bonds are almost always modelling errors.
//!
//! A port of cctbx `mmtbx/validation/omegalyze.py` (`find_omega_type`, `get_omega_atoms`) and
//! the residue linkage it uses (`LinkedResidues.are_linked` without a restraints manager).

use serde::{Deserialize, Serialize};

/// Residues are consecutive for omegalyze when C(i)–N(i+1) is shorter than this
/// (`are_linked(bond_cut_off=2.)`).
pub const OMEGALYZE_LINK_CUTOFF: f64 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum OmegaType {
    Trans,
    Cis,
    Twisted,
}

impl OmegaType {
    /// `find_omega_type`: cis within 30° of 0, trans within 30° of 180, twisted in between.
    pub fn classify(omega: f64) -> OmegaType {
        if omega > -30.0 && omega < 30.0 {
            OmegaType::Cis
        } else if !(-150.0..=150.0).contains(&omega) {
            OmegaType::Trans
        } else {
            OmegaType::Twisted
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholds() {
        assert_eq!(OmegaType::classify(0.0), OmegaType::Cis);
        assert_eq!(OmegaType::classify(-29.99), OmegaType::Cis);
        assert_eq!(OmegaType::classify(30.0), OmegaType::Twisted);
        assert_eq!(OmegaType::classify(150.0), OmegaType::Twisted);
        assert_eq!(OmegaType::classify(150.01), OmegaType::Trans);
        assert_eq!(OmegaType::classify(-179.0), OmegaType::Trans);
    }
}
