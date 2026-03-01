//! MoME Layer — Mixture of Medical Experts
//!
//! Replaces the standard FFN in transformer blocks with a sparse MoE layer.
//! Each token is routed to top-K medical domain experts.
//!
//! Key properties:
//! - Same compute per token as single FFN (only top-K experts activated)
//! - 4x total parameters but constant inference cost
//! - Domain-specialized knowledge in each expert
//! - Shared attention layers, expert-specific FFN

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use super::{
    expert::{MedicalDomain, MedicalExpert},
    router::{ExpertRouter, RouterConfig, RouterOutput},
};

/// MoME layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoMEConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub router: RouterConfig,
}

impl Default for MoMEConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            intermediate_size: 1408,
            router: RouterConfig::default(),
        }
    }
}

/// A Mixture of Medical Experts layer
pub struct MoMELayer {
    router: ExpertRouter,
    experts: Vec<MedicalExpert>,
    config: MoMEConfig,
}

impl MoMELayer {
    pub fn new(config: MoMEConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let router = ExpertRouter::new(config.hidden_size, config.router.clone(), vb.pp("mome"))?;

        let num_experts = config.router.num_experts.min(MedicalDomain::all().len());
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let domain = MedicalDomain::from_index(i);
            let expert = MedicalExpert::new(
                domain,
                config.hidden_size,
                config.intermediate_size,
                vb.pp("mome"),
            )?;
            experts.push(expert);
        }

        Ok(Self {
            router,
            experts,
            config,
        })
    }

    /// Forward pass: route tokens to experts and combine outputs
    pub fn forward(&self, hidden_states: &Tensor, training: bool) -> Result<MoMEOutput> {
        let (batch, seq, hidden) = hidden_states.dims3()?;
        let num_tokens = batch * seq;

        // Route tokens to experts
        let router_output = self.router.route(hidden_states, training)?;

        // Flatten input for per-token processing
        let flat_hidden = hidden_states.reshape((num_tokens, hidden))?;

        // Dispatch tokens to experts and combine
        let mut output_accum =
            Tensor::zeros((num_tokens, hidden), DType::F32, hidden_states.device())?;

        for token_idx in 0..num_tokens {
            let token_hidden = flat_hidden.narrow(0, token_idx, 1)?;
            let expert_indices = &router_output.expert_indices[token_idx];
            let weights = &router_output.routing_weights[token_idx];

            let mut token_output = Tensor::zeros((1, hidden), DType::F32, hidden_states.device())?;

            for (_k, (&expert_idx, &weight)) in
                expert_indices.iter().zip(weights.iter()).enumerate()
            {
                if let Some(expert) = self.experts.get(expert_idx) {
                    let expert_out = expert.forward(&token_hidden)?;
                    let weighted = expert_out.affine(weight as f64, 0.0)?;
                    token_output = token_output.broadcast_add(&weighted)?;
                }
            }

            // Scatter back
            let _current = output_accum.narrow(0, token_idx, 1)?;
            output_accum =
                output_accum.slice_assign(&[token_idx..token_idx + 1, 0..hidden], &token_output)?;
        }

        let output = output_accum.reshape((batch, seq, hidden))?;

        // Collect expert utilization stats
        let expert_utilization: Vec<f32> = router_output
            .expert_counts
            .iter()
            .map(|&c| c as f32 / (num_tokens as f32 * self.config.router.top_k as f32))
            .collect();

        Ok(MoMEOutput {
            hidden_states: output,
            aux_loss: router_output.aux_loss,
            expert_utilization,
            router_output,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    pub fn active_experts_per_token(&self) -> usize {
        self.config.router.top_k
    }

    /// Effective parameter count (only active experts counted)
    pub fn effective_params(&self) -> usize {
        let expert_params = self.config.hidden_size * self.config.intermediate_size * 3; // gate + up + down
        expert_params * self.config.router.top_k
    }

    /// Total parameter count (all experts)
    pub fn total_params(&self) -> usize {
        let expert_params = self.config.hidden_size * self.config.intermediate_size * 3;
        expert_params * self.experts.len()
    }
}

/// MoME layer output
pub struct MoMEOutput {
    pub hidden_states: Tensor,
    pub aux_loss: f32,
    pub expert_utilization: Vec<f32>,
    pub router_output: RouterOutput,
}

/// Format expert utilization as a readable string
pub fn format_expert_utilization(utilization: &[f32]) -> String {
    let mut s = String::from("Expert Utilization:\n");
    for (i, &util) in utilization.iter().enumerate() {
        let domain = MedicalDomain::from_index(i);
        let bar_len = (util * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        s.push_str(&format!("  {:?}: {:>5.1}% {}\n", domain, util * 100.0, bar));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_mome(hidden: usize) -> Result<MoMELayer> {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = MoMEConfig {
            hidden_size: hidden,
            intermediate_size: hidden * 4,
            router: RouterConfig {
                num_experts: 4,
                top_k: 2,
                ..Default::default()
            },
        };
        MoMELayer::new(config, vb)
    }

    #[test]
    fn mome_forward_shape() -> Result<()> {
        let mome = make_mome(32)?;
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1f32, (1, 4, 32), &device)?;

        let output = mome.forward(&input, false)?;
        assert_eq!(output.hidden_states.dims(), &[1, 4, 32]);
        Ok(())
    }

    #[test]
    fn mome_expert_counts() -> Result<()> {
        let mome = make_mome(32)?;
        assert_eq!(mome.num_experts(), 4);
        assert_eq!(mome.active_experts_per_token(), 2);
        // Total > effective (all experts vs active only)
        assert!(mome.total_params() > mome.effective_params());
        assert_eq!(mome.total_params(), mome.effective_params() * 2); // 4/2 = 2x
        Ok(())
    }

    #[test]
    fn expert_utilization_is_bounded() -> Result<()> {
        let mome = make_mome(32)?;
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1f32, (2, 8, 32), &device)?;

        let output = mome.forward(&input, false)?;
        for &util in &output.expert_utilization {
            assert!(
                util >= 0.0 && util <= 1.0,
                "Utilization should be in [0,1], got {util}"
            );
        }
        Ok(())
    }

    #[test]
    fn utilization_format() {
        let utils = vec![0.3, 0.25, 0.2, 0.25];
        let formatted = format_expert_utilization(&utils);
        assert!(formatted.contains("Cardiology"));
        assert!(formatted.contains("InfectiousDisease"));
    }
}
