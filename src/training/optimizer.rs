#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePrecision {
    EightBitConceptual,
    F32,
}

#[derive(Debug, Clone)]
pub struct TrainableParameter {
    pub values: Vec<f32>,
    pub grads: Vec<f32>,
}

impl TrainableParameter {
    pub fn new(values: Vec<f32>) -> Self {
        let grads = vec![0.0; values.len()];
        Self { values, grads }
    }

    pub fn zero_grad(&mut self) {
        self.grads.fill(0.0);
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub state_precision: StatePrecision,
    step: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32, state_precision: StatePrecision) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            state_precision,
            step: 0,
            m: vec![],
            v: vec![],
        }
    }

    pub fn step(&mut self, params: &mut [TrainableParameter]) {
        self.step = self.step.saturating_add(1);
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0; p.values.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.values.len()]).collect();
        }

        let step_i32 = self.step as i32;
        let bias_correction1 = 1.0 - self.beta1.powi(step_i32);
        let bias_correction2 = 1.0 - self.beta2.powi(step_i32);

        for (idx, param) in params.iter_mut().enumerate() {
            for i in 0..param.values.len() {
                let grad = param.grads[i] + self.weight_decay * param.values[i];

                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * grad;
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * grad * grad;

                let m_hat = self.m[idx][i] / bias_correction1.max(1e-12);
                let v_hat = self.v[idx][i] / bias_correction2.max(1e-12);

                param.values[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            param.zero_grad();
        }
    }

    pub fn step_count(&self) -> usize {
        self.step
    }
}
