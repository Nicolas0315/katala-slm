use katala_slm::training::{
    optimizer::{AdamW, StatePrecision, TrainableParameter},
    qlora::QLoRAConfig,
    train_epoch, TrainableModel, TrainingBatch,
};

#[derive(Debug, Clone)]
struct TinyLinearModel {
    params: Vec<TrainableParameter>,
    in_dim: usize,
    out_dim: usize,
}

impl TinyLinearModel {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut values = vec![0.0; in_dim * out_dim];
        for (idx, v) in values.iter_mut().enumerate() {
            *v = (idx as f32 + 1.0) * 0.01;
        }
        Self {
            params: vec![TrainableParameter::new(values)],
            in_dim,
            out_dim,
        }
    }
}

impl TrainableModel for TinyLinearModel {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0; self.out_dim];
        for (o, logit) in logits.iter_mut().enumerate().take(self.out_dim) {
            let mut sum = 0.0;
            for (i, x) in input.iter().copied().enumerate().take(self.in_dim) {
                sum += self.params[0].values[o * self.in_dim + i] * x;
            }
            *logit = sum;
        }
        logits
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        for (o, go) in grad_output.iter().copied().enumerate().take(self.out_dim) {
            for (i, x) in input.iter().copied().enumerate().take(self.in_dim) {
                self.params[0].grads[o * self.in_dim + i] += go * x;
            }
        }
    }

    fn parameters_mut(&mut self) -> &mut [TrainableParameter] {
        &mut self.params
    }
}

#[test]
fn qlora_config_has_reasonable_defaults() {
    let cfg = QLoRAConfig::default();
    assert!(cfg.rank > 0);
    assert!(cfg.alpha > 0.0);
    assert!(!cfg.target_modules.is_empty());
}

#[test]
fn training_loop_updates_parameters_with_gradient_accumulation() {
    let mut model = TinyLinearModel::new(2, 2);
    let before = model.params[0].values.clone();

    let batches = vec![
        TrainingBatch {
            input: vec![1.0, 0.5],
            target_class: 0,
        },
        TrainingBatch {
            input: vec![0.5, 1.0],
            target_class: 1,
        },
    ];

    let mut optimizer = AdamW::new(0.05, 0.0, StatePrecision::EightBitConceptual);
    let metrics = train_epoch(&mut model, &batches, &mut optimizer, 2);

    assert!(metrics.avg_loss >= 0.0);
    assert_eq!(metrics.step_count, 1);
    assert_ne!(before, model.params[0].values);
}
