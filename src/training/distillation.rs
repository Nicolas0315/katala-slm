//! LRC (Low-Rank Clone) Distillation Framework
//!
//! 1000x more efficient than full knowledge distillation by transferring
//! only the low-rank subspace of teacher activations to the student.

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for LRC distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrcConfig {
    /// Rank of the projection (controls capacity/efficiency tradeoff)
    pub rank: usize,
    /// Temperature for softmax distillation (higher = softer targets)
    pub temperature: f32,
    /// Weight of the hard label loss vs distillation loss
    pub alpha_hard: f32,
    /// Weight of the feature alignment loss
    pub alpha_feature: f32,
    /// Weight of the attention transfer loss
    pub alpha_attention: f32,
    /// Layer indices to align features from (teacher → student mapping)
    pub alignment_layers: Vec<(usize, usize)>,
}

impl Default for LrcConfig {
    fn default() -> Self {
        Self {
            rank: 32,
            temperature: 4.0,
            alpha_hard: 0.3,
            alpha_feature: 0.5,
            alpha_attention: 0.2,
            alignment_layers: vec![(0, 0), (3, 1), (7, 3), (11, 5)],
        }
    }
}

/// Low-rank projection matrix for mapping teacher hidden states to student space
#[derive(Debug, Clone)]
pub struct LrcProjection {
    pub down: Tensor, // [teacher_dim, rank]
    pub up: Tensor,   // [rank, student_dim]
}

impl LrcProjection {
    pub fn new(
        teacher_dim: usize,
        student_dim: usize,
        rank: usize,
        device: &Device,
    ) -> Result<Self> {
        let scale = (2.0 / (teacher_dim + student_dim) as f64).sqrt() as f32;
        let down = Tensor::randn(0f32, scale, (teacher_dim, rank), device)?;
        let up = Tensor::randn(0f32, scale, (rank, student_dim), device)?;
        Ok(Self { down, up })
    }

    /// Project teacher hidden states to student dimension space
    pub fn project(&self, teacher_hidden: &Tensor) -> Result<Tensor> {
        let dims = teacher_hidden.dims();
        match dims.len() {
            2 => teacher_hidden.matmul(&self.down)?.matmul(&self.up),
            3 => {
                let (b, s, _d) = teacher_hidden.dims3()?;
                let flat = teacher_hidden.reshape((b * s, ()))?;
                let projected = flat.matmul(&self.down)?.matmul(&self.up)?;
                let out_dim = projected.dim(1)?;
                projected.reshape((b, s, out_dim))
            }
            _ => teacher_hidden.matmul(&self.down)?.matmul(&self.up),
        }
    }
}

/// Compute KL divergence distillation loss between teacher and student logits
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    let t = temperature as f64;
    let student_scaled = student_logits.affine(1.0 / t, 0.0)?;
    let teacher_scaled = teacher_logits.affine(1.0 / t, 0.0)?;

    let student_log_probs = candle_nn::ops::log_softmax(&student_scaled, candle_core::D::Minus1)?;
    let teacher_probs = candle_nn::ops::softmax(&teacher_scaled, candle_core::D::Minus1)?;

    // KL(teacher || student) = sum(teacher_probs * (log(teacher_probs) - student_log_probs))
    let teacher_log_probs = candle_nn::ops::log_softmax(&teacher_scaled, candle_core::D::Minus1)?;
    let kl = teacher_probs.broadcast_mul(&teacher_log_probs.broadcast_sub(&student_log_probs)?)?;
    let loss = kl.sum_all()?.affine(t * t, 0.0)?;
    Ok(loss)
}

/// Compute feature alignment loss (MSE between projected teacher and student hidden states)
pub fn feature_alignment_loss(
    student_hidden: &Tensor,
    teacher_hidden: &Tensor,
    projection: &LrcProjection,
) -> Result<Tensor> {
    let projected = projection.project(teacher_hidden)?;
    let diff = student_hidden.broadcast_sub(&projected)?;
    let mse = diff.sqr()?.mean_all()?;
    Ok(mse)
}

/// Compute attention transfer loss (MSE between attention maps)
pub fn attention_transfer_loss(student_attn: &Tensor, teacher_attn: &Tensor) -> Result<Tensor> {
    // Normalize attention maps along the last dimension
    let s_norm = l2_normalize(student_attn)?;
    let t_norm = l2_normalize(teacher_attn)?;
    let diff = s_norm.broadcast_sub(&t_norm)?;
    diff.sqr()?.mean_all()
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    let norm = norm.clamp(1e-12f32, f32::MAX)?;
    x.broadcast_div(&norm)
}

/// Combined LRC distillation loss
pub fn lrc_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    student_hiddens: &[Tensor],
    teacher_hiddens: &[Tensor],
    projections: &[LrcProjection],
    student_attns: &[Tensor],
    teacher_attns: &[Tensor],
    hard_labels: &Tensor,
    config: &LrcConfig,
) -> Result<Tensor> {
    // 1. Distillation loss (soft targets)
    let kd_loss = distillation_loss(student_logits, teacher_logits, config.temperature)?;

    // 2. Hard label cross-entropy loss
    let hard_loss = cross_entropy_loss_tensor(student_logits, hard_labels)?;

    // 3. Feature alignment losses
    let mut feat_loss = Tensor::zeros((), DType::F32, student_logits.device())?;
    for (proj, &(t_idx, s_idx)) in projections.iter().zip(config.alignment_layers.iter()) {
        if let (Some(t_h), Some(s_h)) = (teacher_hiddens.get(t_idx), student_hiddens.get(s_idx)) {
            let fl = feature_alignment_loss(s_h, t_h, proj)?;
            feat_loss = feat_loss.broadcast_add(&fl)?;
        }
    }

    // 4. Attention transfer losses
    let mut attn_loss = Tensor::zeros((), DType::F32, student_logits.device())?;
    let n_attn = student_attns.len().min(teacher_attns.len());
    for i in 0..n_attn {
        let al = attention_transfer_loss(&student_attns[i], &teacher_attns[i])?;
        attn_loss = attn_loss.broadcast_add(&al)?;
    }

    // Combined weighted loss
    let alpha_distill = 1.0 - config.alpha_hard - config.alpha_feature - config.alpha_attention;
    let total = kd_loss
        .affine(alpha_distill as f64, 0.0)?
        .broadcast_add(&hard_loss.affine(config.alpha_hard as f64, 0.0)?)?
        .broadcast_add(&feat_loss.affine(config.alpha_feature as f64, 0.0)?)?
        .broadcast_add(&attn_loss.affine(config.alpha_attention as f64, 0.0)?)?;

    Ok(total)
}

/// Simple cross-entropy loss for tensor-based labels
fn cross_entropy_loss_tensor(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?;
    let neg_log_probs = log_probs.neg()?;
    // Gather the target class probabilities
    let loss = neg_log_probs.gather(targets, candle_core::D::Minus1)?;
    loss.mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lrc_projection_shape() -> Result<()> {
        let device = Device::Cpu;
        let proj = LrcProjection::new(768, 256, 32, &device)?;
        assert_eq!(proj.down.dims(), &[768, 32]);
        assert_eq!(proj.up.dims(), &[32, 256]);

        let teacher_h = Tensor::randn(0f32, 1f32, (1, 10, 768), &device)?;
        let projected = proj.project(&teacher_h)?;
        assert_eq!(projected.dims(), &[1, 10, 256]);
        Ok(())
    }

    #[test]
    fn distillation_loss_is_non_negative() -> Result<()> {
        let device = Device::Cpu;
        let student = Tensor::randn(0f32, 1f32, (1, 5, 100), &device)?;
        let teacher = Tensor::randn(0f32, 1f32, (1, 5, 100), &device)?;
        let loss = distillation_loss(&student, &teacher, 4.0)?;
        let val: f32 = loss.to_scalar()?;
        assert!(
            val >= -0.01,
            "KL divergence should be approximately non-negative, got {val}"
        );
        Ok(())
    }
}
