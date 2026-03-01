//! Elastic Weight Consolidation (EWC)
//!
//! Prevents catastrophic forgetting when learning new medical knowledge.
//! Key idea: penalize changes to parameters that were important for
//! previously learned tasks, measured by Fisher Information.
//!
//! L_total = L_new + λ/2 * Σ_i F_i * (θ_i - θ*_i)²
//!
//! where F_i = Fisher Information for parameter i
//!       θ*_i = optimal parameter value for old tasks
//!       λ = EWC regularization strength

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// EWC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwcConfig {
    /// Regularization strength (higher = more resistance to forgetting)
    pub lambda: f32,
    /// Number of samples to estimate Fisher Information
    pub fisher_samples: usize,
    /// Whether to use online EWC (accumulate Fisher across tasks)
    pub online: bool,
    /// Decay factor for online EWC (γ in F_new = γ*F_old + F_current)
    pub gamma: f32,
    /// Minimum Fisher value (prevents division by zero)
    pub fisher_floor: f32,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 5000.0,
            fisher_samples: 200,
            online: true,
            gamma: 0.95,
            fisher_floor: 1e-8,
        }
    }
}

/// Fisher Information for a single parameter set
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Parameter name → Fisher diagonal values
    pub fisher_diag: HashMap<String, Vec<f32>>,
    /// Parameter name → optimal parameter values (θ*)
    pub optimal_params: HashMap<String, Vec<f32>>,
    /// Task name this Fisher was computed for
    pub task_name: String,
    /// Number of samples used to compute this Fisher
    pub num_samples: usize,
}

impl FisherInformation {
    pub fn new(task_name: &str) -> Self {
        Self {
            fisher_diag: HashMap::new(),
            optimal_params: HashMap::new(),
            task_name: task_name.to_string(),
            num_samples: 0,
        }
    }

    /// Accumulate gradient squared for Fisher estimation
    /// F_i ≈ E[∂log p(y|x,θ)/∂θ_i)²]
    pub fn accumulate_grad_squared(&mut self, param_name: &str, grad_squared: &[f32]) {
        let entry = self
            .fisher_diag
            .entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; grad_squared.len()]);
        for (f, &g) in entry.iter_mut().zip(grad_squared) {
            *f += g;
        }
        self.num_samples += 1;
    }

    /// Finalize Fisher by averaging over samples
    pub fn finalize(&mut self, config: &EwcConfig) {
        let n = self.num_samples.max(1) as f32;
        for fisher_vals in self.fisher_diag.values_mut() {
            for f in fisher_vals.iter_mut() {
                *f = (*f / n).max(config.fisher_floor);
            }
        }
    }

    /// Store current parameters as optimal (θ*)
    pub fn store_optimal(&mut self, param_name: &str, values: &[f32]) {
        self.optimal_params
            .insert(param_name.to_string(), values.to_vec());
    }
}

/// EWC regularization engine
#[derive(Debug, Clone)]
pub struct EwcRegularizer {
    config: EwcConfig,
    /// Accumulated Fisher information across tasks
    accumulated_fisher: HashMap<String, Vec<f32>>,
    /// Reference parameters (θ*)
    reference_params: HashMap<String, Vec<f32>>,
    /// Number of tasks consolidated
    pub tasks_consolidated: usize,
}

impl EwcRegularizer {
    pub fn new(config: EwcConfig) -> Self {
        Self {
            config,
            accumulated_fisher: HashMap::new(),
            reference_params: HashMap::new(),
            tasks_consolidated: 0,
        }
    }

    /// Consolidate a new task's Fisher information
    pub fn consolidate(&mut self, fisher: &FisherInformation) {
        for (param_name, fisher_vals) in &fisher.fisher_diag {
            if self.config.online && self.accumulated_fisher.contains_key(param_name) {
                // Online EWC: F_new = γ * F_old + F_current
                let existing = self.accumulated_fisher.get_mut(param_name).unwrap();
                for (e, &f) in existing.iter_mut().zip(fisher_vals) {
                    *e = self.config.gamma * *e + f;
                }
            } else {
                self.accumulated_fisher
                    .insert(param_name.clone(), fisher_vals.clone());
            }
        }

        // Update reference parameters
        for (param_name, param_vals) in &fisher.optimal_params {
            self.reference_params
                .insert(param_name.clone(), param_vals.clone());
        }

        self.tasks_consolidated += 1;
    }

    /// Compute EWC penalty: λ/2 * Σ_i F_i * (θ_i - θ*_i)²
    pub fn penalty(&self, param_name: &str, current_params: &[f32]) -> f32 {
        let fisher = match self.accumulated_fisher.get(param_name) {
            Some(f) => f,
            None => return 0.0,
        };
        let reference = match self.reference_params.get(param_name) {
            Some(r) => r,
            None => return 0.0,
        };

        let mut penalty = 0.0f32;
        let len = current_params.len().min(fisher.len()).min(reference.len());
        for i in 0..len {
            let diff = current_params[i] - reference[i];
            penalty += fisher[i] * diff * diff;
        }

        self.config.lambda / 2.0 * penalty
    }

    /// Total EWC penalty across all parameter sets
    pub fn total_penalty(&self, current_params: &HashMap<String, Vec<f32>>) -> f32 {
        current_params
            .iter()
            .map(|(name, values)| self.penalty(name, values))
            .sum()
    }

    /// Compute EWC gradient contribution for a parameter
    /// ∂L_ewc/∂θ_i = λ * F_i * (θ_i - θ*_i)
    pub fn penalty_gradient(&self, param_name: &str, current_params: &[f32]) -> Vec<f32> {
        let fisher = match self.accumulated_fisher.get(param_name) {
            Some(f) => f,
            None => return vec![0.0; current_params.len()],
        };
        let reference = match self.reference_params.get(param_name) {
            Some(r) => r,
            None => return vec![0.0; current_params.len()],
        };

        let len = current_params.len().min(fisher.len()).min(reference.len());
        let mut grad = vec![0.0f32; current_params.len()];
        for i in 0..len {
            grad[i] = self.config.lambda * fisher[i] * (current_params[i] - reference[i]);
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fisher_accumulation_and_finalization() {
        let config = EwcConfig::default();
        let mut fisher = FisherInformation::new("task_1");

        fisher.accumulate_grad_squared("layer.weight", &[1.0, 4.0, 9.0]);
        fisher.accumulate_grad_squared("layer.weight", &[3.0, 0.0, 1.0]);
        fisher.finalize(&config);

        let vals = fisher.fisher_diag.get("layer.weight").unwrap();
        // Average: [(1+3)/2, (4+0)/2, (9+1)/2] = [2.0, 2.0, 5.0]
        assert!((vals[0] - 2.0).abs() < 0.01);
        assert!((vals[1] - 2.0).abs() < 0.01);
        assert!((vals[2] - 5.0).abs() < 0.01);
    }

    #[test]
    fn ewc_penalty_increases_with_divergence() {
        let config = EwcConfig {
            lambda: 100.0,
            ..Default::default()
        };
        let mut regularizer = EwcRegularizer::new(config);

        let mut fisher = FisherInformation::new("old_knowledge");
        fisher.fisher_diag.insert("w".into(), vec![1.0, 1.0, 1.0]);
        fisher
            .optimal_params
            .insert("w".into(), vec![0.5, 0.5, 0.5]);
        regularizer.consolidate(&fisher);

        // Close to reference → small penalty
        let small = regularizer.penalty("w", &[0.5, 0.5, 0.5]);
        // Far from reference → large penalty
        let large = regularizer.penalty("w", &[5.0, 5.0, 5.0]);

        assert!(small < large, "Penalty should increase with divergence");
        assert!(small < 0.01, "Penalty at reference should be near zero");
    }

    #[test]
    fn online_ewc_accumulates_across_tasks() {
        let config = EwcConfig {
            lambda: 1.0,
            gamma: 0.9,
            online: true,
            ..Default::default()
        };
        let mut regularizer = EwcRegularizer::new(config);

        let mut f1 = FisherInformation::new("task1");
        f1.fisher_diag.insert("w".into(), vec![1.0]);
        f1.optimal_params.insert("w".into(), vec![0.0]);
        regularizer.consolidate(&f1);

        let after_first = regularizer.accumulated_fisher.get("w").unwrap()[0];

        let mut f2 = FisherInformation::new("task2");
        f2.fisher_diag.insert("w".into(), vec![2.0]);
        f2.optimal_params.insert("w".into(), vec![1.0]);
        regularizer.consolidate(&f2);

        let after_second = regularizer.accumulated_fisher.get("w").unwrap()[0];
        // Should be 0.9 * 1.0 + 2.0 = 2.9
        assert!(
            (after_second - 2.9).abs() < 0.01,
            "Online EWC accumulation: expected 2.9, got {after_second}"
        );
        assert_eq!(regularizer.tasks_consolidated, 2);
    }

    #[test]
    fn penalty_gradient_direction() {
        let config = EwcConfig {
            lambda: 10.0,
            ..Default::default()
        };
        let mut regularizer = EwcRegularizer::new(config);
        let mut fisher = FisherInformation::new("base");
        fisher.fisher_diag.insert("w".into(), vec![1.0, 1.0]);
        fisher.optimal_params.insert("w".into(), vec![0.0, 0.0]);
        regularizer.consolidate(&fisher);

        // Current params > reference → positive gradient (pulls back)
        let grad = regularizer.penalty_gradient("w", &[2.0, -1.0]);
        assert!(grad[0] > 0.0, "Gradient should push back toward reference");
        assert!(grad[1] < 0.0, "Gradient should push back toward reference");
    }
}
