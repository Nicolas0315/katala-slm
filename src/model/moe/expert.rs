//! Medical Domain Experts
//!
//! Each expert is a lightweight FFN specialized for a medical domain.
//! In a 0.6B model with 8 experts, each expert is ~75M effective params
//! but only 2 are activated per token → constant compute cost.

use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Medical specialties that experts are trained on
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedicalDomain {
    Cardiology,
    InfectiousDisease,
    Endocrinology,
    Neurology,
    Oncology,
    Pulmonology,
    Gastroenterology,
    GeneralMedicine,
}

impl MedicalDomain {
    pub fn all() -> &'static [MedicalDomain] {
        &[
            Self::Cardiology,
            Self::InfectiousDisease,
            Self::Endocrinology,
            Self::Neurology,
            Self::Oncology,
            Self::Pulmonology,
            Self::Gastroenterology,
            Self::GeneralMedicine,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Cardiology => 0,
            Self::InfectiousDisease => 1,
            Self::Endocrinology => 2,
            Self::Neurology => 3,
            Self::Oncology => 4,
            Self::Pulmonology => 5,
            Self::Gastroenterology => 6,
            Self::GeneralMedicine => 7,
        }
    }

    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Cardiology,
            1 => Self::InfectiousDisease,
            2 => Self::Endocrinology,
            3 => Self::Neurology,
            4 => Self::Oncology,
            5 => Self::Pulmonology,
            6 => Self::Gastroenterology,
            _ => Self::GeneralMedicine,
        }
    }
}

/// A single medical expert (SwiGLU FFN)
pub struct MedicalExpert {
    pub domain: MedicalDomain,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MedicalExpert {
    pub fn new(
        domain: MedicalDomain,
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let prefix = format!("expert_{}", domain.index());
        Ok(Self {
            domain,
            gate_proj: linear_no_bias(
                hidden_size,
                intermediate_size,
                vb.pp(format!("{prefix}.gate")),
            )?,
            up_proj: linear_no_bias(
                hidden_size,
                intermediate_size,
                vb.pp(format!("{prefix}.up")),
            )?,
            down_proj: linear_no_bias(
                intermediate_size,
                hidden_size,
                vb.pp(format!("{prefix}.down")),
            )?,
        })
    }

    /// Forward pass with SwiGLU activation
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let activated = gate.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

/// Expert output with routing weight
pub struct ExpertOutput {
    pub output: Tensor,
    pub domain: MedicalDomain,
    pub routing_weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_roundtrip() {
        for (i, domain) in MedicalDomain::all().iter().enumerate() {
            assert_eq!(domain.index(), i);
            assert_eq!(MedicalDomain::from_index(i), *domain);
        }
    }

    #[test]
    fn all_domains_count() {
        assert_eq!(MedicalDomain::all().len(), 8);
    }
}
