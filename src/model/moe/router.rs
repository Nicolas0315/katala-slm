//! Expert Router — Top-K Sparse Gating
//!
//! Routes each token to the top-K medical experts based on learned gating.
//! Includes load balancing auxiliary loss to prevent expert collapse.

use candle_core::{DType, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts activated per token
    pub top_k: usize,
    /// Noise factor for load balancing during training
    pub noise_factor: f32,
    /// Load balancing loss coefficient
    pub lb_loss_coeff: f32,
    /// Capacity factor (max fraction of tokens per expert)
    pub capacity_factor: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            noise_factor: 0.1,
            lb_loss_coeff: 0.01,
            capacity_factor: 1.25,
        }
    }
}

/// Router output: expert indices, weights, and aux loss
#[derive(Debug)]
pub struct RouterOutput {
    /// Expert indices per token: [batch_size * seq_len, top_k]
    pub expert_indices: Vec<Vec<usize>>,
    /// Routing weights per token: [batch_size * seq_len, top_k]
    pub routing_weights: Vec<Vec<f32>>,
    /// Load balancing auxiliary loss
    pub aux_loss: f32,
    /// Expert utilization counts
    pub expert_counts: Vec<usize>,
}

/// Learned gating network
pub struct ExpertRouter {
    gate: Linear,
    config: RouterConfig,
}

impl ExpertRouter {
    pub fn new(hidden_size: usize, config: RouterConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, config.num_experts, vb.pp("router_gate"))?;
        Ok(Self { gate, config })
    }

    /// Route tokens to experts
    pub fn route(&self, hidden_states: &Tensor, training: bool) -> Result<RouterOutput> {
        let (batch, seq, _hidden) = hidden_states.dims3()?;
        let num_tokens = batch * seq;
        let flat = hidden_states.reshape((num_tokens, ()))?;

        // Compute gating logits
        let logits = self.gate.forward(&flat)?; // [num_tokens, num_experts]
        let logits_vec: Vec<Vec<f32>> =
            self.tensor_to_2d(&logits, num_tokens, self.config.num_experts)?;

        // Add noise during training for exploration
        let logits_noisy = if training && self.config.noise_factor > 0.0 {
            self.add_noise(&logits_vec)
        } else {
            logits_vec
        };

        // Softmax over experts
        let probs = self.softmax_2d(&logits_noisy);

        // Top-K selection
        let mut expert_indices = Vec::with_capacity(num_tokens);
        let mut routing_weights = Vec::with_capacity(num_tokens);
        let mut expert_counts = vec![0usize; self.config.num_experts];

        for token_probs in &probs {
            let (top_k_idx, top_k_weights) = self.top_k(token_probs, self.config.top_k);

            // Normalize top-K weights to sum to 1
            let weight_sum: f32 = top_k_weights.iter().sum::<f32>().max(1e-10);
            let normalized: Vec<f32> = top_k_weights.iter().map(|w| w / weight_sum).collect();

            for &idx in &top_k_idx {
                expert_counts[idx] += 1;
            }

            expert_indices.push(top_k_idx);
            routing_weights.push(normalized);
        }

        // Load balancing auxiliary loss
        let aux_loss = self.compute_lb_loss(&probs, &expert_counts, num_tokens);

        Ok(RouterOutput {
            expert_indices,
            routing_weights,
            aux_loss,
            expert_counts,
        })
    }

    fn tensor_to_2d(&self, t: &Tensor, _rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
        let flat: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(flat.chunks(cols).map(|c| c.to_vec()).collect())
    }

    fn add_noise(&self, logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simple deterministic noise based on position (for reproducibility)
        logits
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let noise = ((i * 7 + j * 13) % 100) as f32 / 100.0 - 0.5;
                        v + noise * self.config.noise_factor
                    })
                    .collect()
            })
            .collect()
    }

    fn softmax_2d(&self, logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
        logits
            .iter()
            .map(|row| {
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
                let sum: f32 = exp.iter().sum::<f32>().max(1e-12);
                exp.iter().map(|v| v / sum).collect()
            })
            .collect()
    }

    fn top_k(&self, probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        let k = k.min(probs.len());
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let weights: Vec<f32> = indexed[..k].iter().map(|(_, w)| *w).collect();
        (indices, weights)
    }

    /// Switch Transformer-style load balancing loss
    ///
    /// L_lb = N * Σ_i (f_i * P_i)
    /// where f_i = fraction of tokens routed to expert i
    ///       P_i = fraction of routing probability assigned to expert i
    fn compute_lb_loss(
        &self,
        probs: &[Vec<f32>],
        expert_counts: &[usize],
        num_tokens: usize,
    ) -> f32 {
        if num_tokens == 0 || probs.is_empty() {
            return 0.0;
        }

        let n = self.config.num_experts as f32;
        let num_tokens_f = num_tokens as f32;

        // f_i: fraction of tokens dispatched to expert i
        let f: Vec<f32> = expert_counts
            .iter()
            .map(|&c| c as f32 / num_tokens_f)
            .collect();

        // P_i: mean routing probability for expert i across all tokens
        let mut p = vec![0.0f32; self.config.num_experts];
        for token_probs in probs {
            for (i, &prob) in token_probs.iter().enumerate() {
                p[i] += prob;
            }
        }
        for pi in &mut p {
            *pi /= num_tokens_f;
        }

        // L = N * Σ(f_i * P_i)
        let loss: f32 = f.iter().zip(p.iter()).map(|(fi, pi)| fi * pi).sum();
        n * loss * self.config.lb_loss_coeff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_router(hidden: usize, num_experts: usize, top_k: usize) -> Result<ExpertRouter> {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = RouterConfig {
            num_experts,
            top_k,
            ..Default::default()
        };
        ExpertRouter::new(hidden, config, vb)
    }

    #[test]
    fn router_selects_top_k_experts() -> Result<()> {
        let router = make_router(64, 8, 2)?;
        let device = Device::Cpu;
        let hidden = Tensor::randn(0f32, 1f32, (1, 4, 64), &device)?;

        let output = router.route(&hidden, false)?;
        assert_eq!(output.expert_indices.len(), 4); // 1*4 tokens
        for indices in &output.expert_indices {
            assert_eq!(indices.len(), 2); // top_k = 2
                                          // Each index should be valid
            for &idx in indices {
                assert!(idx < 8);
            }
        }
        Ok(())
    }

    #[test]
    fn routing_weights_sum_to_one() -> Result<()> {
        let router = make_router(32, 4, 2)?;
        let device = Device::Cpu;
        let hidden = Tensor::randn(0f32, 1f32, (2, 3, 32), &device)?;

        let output = router.route(&hidden, false)?;
        for weights in &output.routing_weights {
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Weights should sum to ~1.0, got {sum}"
            );
        }
        Ok(())
    }

    #[test]
    fn load_balancing_loss_is_non_negative() -> Result<()> {
        let router = make_router(64, 8, 2)?;
        let device = Device::Cpu;
        let hidden = Tensor::randn(0f32, 1f32, (1, 10, 64), &device)?;

        let output = router.route(&hidden, true)?;
        assert!(
            output.aux_loss >= 0.0,
            "LB loss should be non-negative, got {}",
            output.aux_loss
        );
        Ok(())
    }

    #[test]
    fn expert_counts_match_token_count() -> Result<()> {
        let router = make_router(32, 4, 2)?;
        let device = Device::Cpu;
        let hidden = Tensor::randn(0f32, 1f32, (1, 8, 32), &device)?;

        let output = router.route(&hidden, false)?;
        let total_assignments: usize = output.expert_counts.iter().sum();
        // Each token routes to top_k=2 experts, 8 tokens total
        assert_eq!(total_assignments, 16);
        Ok(())
    }
}
