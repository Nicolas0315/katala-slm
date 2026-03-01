pub mod distillation;
pub mod dpo;
pub mod optimizer;
pub mod qlora;

use optimizer::{AdamW, TrainableParameter};

#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub input: Vec<f32>,
    pub target_class: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingMetrics {
    pub avg_loss: f32,
    pub step_count: usize,
    pub learning_rate: f32,
}

pub trait TrainableModel {
    fn forward(&self, input: &[f32]) -> Vec<f32>;
    fn backward(&mut self, input: &[f32], grad_output: &[f32]);
    fn parameters_mut(&mut self) -> &mut [TrainableParameter];
}

pub fn cross_entropy_loss(logits: &[f32], target_class: usize) -> (f32, Vec<f32>) {
    if logits.is_empty() {
        return (0.0, vec![]);
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max_logit).exp()).collect();
    let exp_sum: f32 = exp.iter().sum::<f32>().max(1e-12);
    let probs: Vec<f32> = exp.iter().map(|v| v / exp_sum).collect();

    let target = target_class.min(probs.len() - 1);
    let loss = -(probs[target].max(1e-12)).ln();

    let mut grad = probs;
    grad[target] -= 1.0;

    (loss, grad)
}

pub fn train_epoch<M: TrainableModel>(
    model: &mut M,
    batches: &[TrainingBatch],
    optimizer: &mut AdamW,
    grad_accum_steps: usize,
) -> TrainingMetrics {
    if batches.is_empty() {
        return TrainingMetrics {
            avg_loss: 0.0,
            step_count: optimizer.step_count(),
            learning_rate: optimizer.learning_rate,
        };
    }

    let accum = grad_accum_steps.max(1);
    let mut total_loss = 0.0;

    for (idx, batch) in batches.iter().enumerate() {
        let logits = model.forward(&batch.input);
        let (loss, mut grad_output) = cross_entropy_loss(&logits, batch.target_class);
        total_loss += loss;

        for g in &mut grad_output {
            *g /= accum as f32;
        }

        model.backward(&batch.input, &grad_output);

        let is_accum_boundary = (idx + 1) % accum == 0;
        let is_last = idx + 1 == batches.len();
        if is_accum_boundary || is_last {
            optimizer.step(model.parameters_mut());
        }
    }

    TrainingMetrics {
        avg_loss: total_loss / batches.len() as f32,
        step_count: optimizer.step_count(),
        learning_rate: optimizer.learning_rate,
    }
}
