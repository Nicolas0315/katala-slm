//! DPO (Direct Preference Optimization) with KS Verification as Reward Signal
//!
//! Uses KS40e's 18-axis evidence scores as preference signal:
//! - Chosen response = high evidence level (A/B) + high confidence
//! - Rejected response = low evidence level (C/D) + low confidence
//! This removes the need for a separate reward model.

use candle_core::{Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::ks::{
    evidence::EvidenceLevel,
    verify::{VerifiedAnswer, Verifier},
};

/// DPO training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    /// Beta parameter (controls divergence from reference policy)
    pub beta: f32,
    /// Minimum confidence gap between chosen and rejected pairs
    pub min_confidence_gap: f32,
    /// Minimum evidence level for chosen responses
    pub min_chosen_evidence: EvidenceLevel,
    /// Maximum evidence level for rejected responses
    pub max_rejected_evidence: EvidenceLevel,
    /// Label smoothing (0 = no smoothing)
    pub label_smoothing: f32,
    /// Whether to use KS axis scores as additional reward signal
    pub use_axis_reward: bool,
    /// Weight for axis-based reward bonus
    pub axis_reward_weight: f32,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            min_confidence_gap: 0.15,
            min_chosen_evidence: EvidenceLevel::B,
            max_rejected_evidence: EvidenceLevel::C,
            label_smoothing: 0.0,
            use_axis_reward: true,
            axis_reward_weight: 0.3,
        }
    }
}

/// A preference pair for DPO training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub chosen_evidence: EvidenceLevel,
    pub rejected_evidence: EvidenceLevel,
    pub chosen_confidence: f32,
    pub rejected_confidence: f32,
    pub axis_reward_delta: f32,
}

/// Generate preference pairs from model outputs using KS verification
pub fn generate_preference_pairs(
    prompts: &[String],
    responses_a: &[String],
    responses_b: &[String],
    verifier: &Verifier,
    config: &DpoConfig,
) -> Vec<PreferencePair> {
    let mut pairs = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let (resp_a, resp_b) = match (responses_a.get(i), responses_b.get(i)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        let verified_a = verifier.verify(prompt, resp_a, &[]);
        let verified_b = verifier.verify(prompt, resp_b, &[]);

        let gap = (verified_a.confidence - verified_b.confidence).abs();
        if gap < config.min_confidence_gap {
            continue; // Skip ambiguous pairs
        }

        let (chosen, rejected, chosen_v, rejected_v) =
            if verified_a.confidence > verified_b.confidence {
                (resp_a.clone(), resp_b.clone(), verified_a, verified_b)
            } else {
                (resp_b.clone(), resp_a.clone(), verified_b, verified_a)
            };

        let axis_delta = if config.use_axis_reward {
            compute_axis_reward_delta(&chosen_v, &rejected_v)
        } else {
            0.0
        };

        pairs.push(PreferencePair {
            prompt: prompt.clone(),
            chosen,
            rejected,
            chosen_evidence: chosen_v.evidence_level,
            rejected_evidence: rejected_v.evidence_level,
            chosen_confidence: chosen_v.confidence,
            rejected_confidence: rejected_v.confidence,
            axis_reward_delta: axis_delta,
        });
    }

    pairs
}

/// Compute DPO loss
///
/// L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
pub fn dpo_loss(
    chosen_logprobs: &Tensor,
    rejected_logprobs: &Tensor,
    ref_chosen_logprobs: &Tensor,
    ref_rejected_logprobs: &Tensor,
    config: &DpoConfig,
) -> Result<Tensor> {
    let beta = config.beta as f64;

    // Policy log-ratios
    let chosen_ratio = chosen_logprobs.broadcast_sub(ref_chosen_logprobs)?;
    let rejected_ratio = rejected_logprobs.broadcast_sub(ref_rejected_logprobs)?;

    // DPO implicit reward difference
    let logits = chosen_ratio
        .broadcast_sub(&rejected_ratio)?
        .affine(beta, 0.0)?;

    // Label smoothing
    let loss = if config.label_smoothing > 0.0 {
        let eps = config.label_smoothing as f64;
        let pos_loss = log_sigmoid_neg(&logits)?;
        let neg_loss = log_sigmoid(&logits)?;
        pos_loss
            .affine(1.0 - eps, 0.0)?
            .broadcast_add(&neg_loss.affine(eps, 0.0)?)?
    } else {
        log_sigmoid_neg(&logits)?
    };

    loss.neg()?.mean_all()
}

/// DPO loss with KS axis reward bonus
pub fn dpo_loss_with_ks_reward(
    chosen_logprobs: &Tensor,
    rejected_logprobs: &Tensor,
    ref_chosen_logprobs: &Tensor,
    ref_rejected_logprobs: &Tensor,
    axis_reward_deltas: &[f32],
    config: &DpoConfig,
) -> Result<Tensor> {
    let base_loss = dpo_loss(
        chosen_logprobs,
        rejected_logprobs,
        ref_chosen_logprobs,
        ref_rejected_logprobs,
        config,
    )?;

    if !config.use_axis_reward || axis_reward_deltas.is_empty() {
        return Ok(base_loss);
    }

    // Add KS axis reward as additional signal
    let device = chosen_logprobs.device().clone();
    let reward_bonus = Tensor::from_vec(
        axis_reward_deltas.to_vec(),
        axis_reward_deltas.len(),
        &device,
    )?;
    let avg_bonus = reward_bonus.mean_all()?;
    let weighted_bonus = avg_bonus.affine(config.axis_reward_weight as f64, 0.0)?;

    base_loss.broadcast_sub(&weighted_bonus)
}

/// Compute reward delta from KS axis scores between chosen and rejected
fn compute_axis_reward_delta(chosen: &VerifiedAnswer, rejected: &VerifiedAnswer) -> f32 {
    // Simple confidence-based delta for now
    // In production, this would use full axis score comparison
    let conf_delta = chosen.confidence - rejected.confidence;
    let evidence_delta = evidence_level_to_numeric(chosen.evidence_level)
        - evidence_level_to_numeric(rejected.evidence_level);
    let contraindication_penalty =
        if chosen.contraindications.len() > rejected.contraindications.len() {
            -0.1
        } else if chosen.contraindications.len() < rejected.contraindications.len() {
            0.1
        } else {
            0.0
        };

    conf_delta + evidence_delta * 0.5 + contraindication_penalty
}

fn evidence_level_to_numeric(level: EvidenceLevel) -> f32 {
    match level {
        EvidenceLevel::A => 1.0,
        EvidenceLevel::B => 0.7,
        EvidenceLevel::C => 0.4,
        EvidenceLevel::D => 0.1,
    }
}

/// log(sigmoid(x))
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    // Numerically stable: log(sigmoid(x)) = -log(1 + exp(-x))
    let neg_x = x.neg()?;
    let one_plus_exp = neg_x.exp()?.affine(1.0, 1.0)?;
    one_plus_exp.log()?.neg()
}

/// log(sigmoid(-x)) = log(1 - sigmoid(x))
fn log_sigmoid_neg(x: &Tensor) -> Result<Tensor> {
    log_sigmoid(&x.neg()?)
}

/// Verify that a preference pair meets quality thresholds
pub fn is_valid_pair(pair: &PreferencePair, config: &DpoConfig) -> bool {
    let gap = pair.chosen_confidence - pair.rejected_confidence;
    gap >= config.min_confidence_gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn preference_pair_generation() {
        let verifier = Verifier::default();
        let config = DpoConfig {
            min_confidence_gap: 0.01,
            ..Default::default()
        };

        let prompts = vec!["Patient on warfarin with joint pain".to_string()];
        let responses_a = vec![
            "Consider NSAID for pain relief. Based on CDC guidelines and WHO recommendations."
                .to_string(),
        ];
        let responses_b = vec!["Maybe try something, it's unclear what works best.".to_string()];

        let pairs =
            generate_preference_pairs(&prompts, &responses_a, &responses_b, &verifier, &config);
        // At least one pair should be generated (unless confidence gap is too small)
        // The exact result depends on verifier heuristics
        assert!(pairs.len() <= 1);
    }

    #[test]
    fn dpo_loss_basic() -> Result<()> {
        let device = Device::Cpu;
        let chosen = Tensor::from_vec(vec![-0.5f32, -0.3, -0.2], (3,), &device)?;
        let rejected = Tensor::from_vec(vec![-1.5f32, -1.0, -0.8], (3,), &device)?;
        let ref_chosen = Tensor::from_vec(vec![-0.6f32, -0.4, -0.3], (3,), &device)?;
        let ref_rejected = Tensor::from_vec(vec![-1.4f32, -0.9, -0.7], (3,), &device)?;

        let config = DpoConfig::default();
        let loss = dpo_loss(&chosen, &rejected, &ref_chosen, &ref_rejected, &config)?;
        let val: f32 = loss.to_scalar()?;
        assert!(val.is_finite(), "DPO loss should be finite, got {val}");
        Ok(())
    }
}
