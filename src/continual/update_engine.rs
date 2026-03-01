//! Continuous Update Engine
//!
//! Orchestrates the full knowledge update pipeline:
//! 1. Ingest new medical data (PubMed, guidelines)
//! 2. Compute Fisher Information on current model (EWC preparation)
//! 3. Fine-tune on new data with EWC regularization
//! 4. Run benchmark validation (MedQA regression check)
//! 5. Version the knowledge update
//! 6. Roll back if regression detected

use serde::{Deserialize, Serialize};

use super::{
    ewc::{EwcConfig, EwcRegularizer, FisherInformation},
    knowledge_version::{KnowledgeSource, KnowledgeUpdate, KnowledgeVersionStore, UpdateMetrics},
};

/// Update engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateEngineConfig {
    /// EWC configuration
    pub ewc: EwcConfig,
    /// Maximum regression in MedQA accuracy before rollback (e.g., 0.02 = 2%)
    pub max_regression: f32,
    /// Maximum regression in safety score before rollback
    pub max_safety_regression: f32,
    /// Minimum training samples per update
    pub min_samples: usize,
    /// Maximum training samples per update (to limit drift)
    pub max_samples: usize,
    /// Learning rate for knowledge updates (usually lower than initial training)
    pub update_lr: f32,
    /// Number of epochs per update
    pub update_epochs: usize,
    /// Auto-rollback on regression
    pub auto_rollback: bool,
}

impl Default for UpdateEngineConfig {
    fn default() -> Self {
        Self {
            ewc: EwcConfig::default(),
            max_regression: 0.02,
            max_safety_regression: 0.01,
            min_samples: 50,
            max_samples: 5000,
            update_lr: 1e-5,
            update_epochs: 3,
            auto_rollback: true,
        }
    }
}

/// Result of a knowledge update attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    pub version: String,
    pub status: UpdateStatus,
    pub pre_metrics: UpdateMetrics,
    pub post_metrics: UpdateMetrics,
    pub sources_incorporated: usize,
    pub samples_trained: usize,
    pub regression_detected: bool,
    pub rolled_back: bool,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateStatus {
    Success,
    RolledBack,
    Skipped,
    Failed,
}

/// The continuous update engine
#[derive(Debug)]
pub struct UpdateEngine {
    config: UpdateEngineConfig,
    regularizer: EwcRegularizer,
    version_store: KnowledgeVersionStore,
    update_count: usize,
}

impl UpdateEngine {
    pub fn new(config: UpdateEngineConfig) -> Self {
        let regularizer = EwcRegularizer::new(config.ewc.clone());
        Self {
            config,
            regularizer,
            version_store: KnowledgeVersionStore::new(),
            update_count: 0,
        }
    }

    /// Plan a knowledge update from new sources
    pub fn plan_update(&self, sources: Vec<KnowledgeSource>, num_samples: usize) -> UpdatePlan {
        let filtered_sources: Vec<_> = sources
            .into_iter()
            .filter(|s| !self.version_store.has_source(&s.identifier))
            .collect();

        let effective_samples = num_samples
            .max(self.config.min_samples)
            .min(self.config.max_samples);

        let version = format!(
            "knowledge-v{}.{}.0",
            self.update_count / 10 + 1,
            self.update_count % 10
        );

        UpdatePlan {
            version,
            sources: filtered_sources,
            num_samples: effective_samples,
            epochs: self.config.update_epochs,
            learning_rate: self.config.update_lr,
            parent_version: self.version_store.current_version.clone(),
        }
    }

    /// Execute a knowledge update (simulated — real training requires model access)
    ///
    /// In production, this would:
    /// 1. Compute Fisher on old data
    /// 2. Fine-tune with EWC on new data
    /// 3. Benchmark and validate
    /// 4. Version the result
    pub fn execute_update(
        &mut self,
        plan: UpdatePlan,
        pre_metrics: UpdateMetrics,
        post_metrics: UpdateMetrics,
        fisher: Option<FisherInformation>,
    ) -> UpdateResult {
        // 1. Consolidate Fisher if provided
        if let Some(fisher) = fisher {
            self.regularizer.consolidate(&fisher);
        }

        // 2. Check for regression
        let accuracy_regression =
            pre_metrics.medqa_accuracy.unwrap_or(0.0) - post_metrics.medqa_accuracy.unwrap_or(0.0);
        let safety_regression =
            pre_metrics.safety_score.unwrap_or(0.0) - post_metrics.safety_score.unwrap_or(0.0);

        let regression_detected = accuracy_regression > self.config.max_regression
            || safety_regression > self.config.max_safety_regression;

        let rolled_back = regression_detected && self.config.auto_rollback;

        let status = if rolled_back {
            UpdateStatus::RolledBack
        } else if regression_detected {
            UpdateStatus::Success // Manual review needed
        } else {
            UpdateStatus::Success
        };

        // 3. Version the update (unless rolled back)
        if !rolled_back {
            let update = KnowledgeUpdate {
                version: plan.version.clone(),
                timestamp: chrono_now_stub(),
                description: format!(
                    "Knowledge update with {} sources, {} samples",
                    plan.sources.len(),
                    plan.num_samples
                ),
                sources: plan.sources.clone(),
                num_samples: plan.num_samples,
                metrics: Some(post_metrics.clone()),
                parent_version: plan.parent_version,
                fisher_hash: None,
            };
            self.version_store.register_update(update);
            self.update_count += 1;
        }

        let summary = if rolled_back {
            format!(
                "⚠️ Update {} ROLLED BACK — accuracy regression: {:.1}%, safety regression: {:.1}%",
                plan.version,
                accuracy_regression * 100.0,
                safety_regression * 100.0
            )
        } else {
            format!(
                "✅ Update {} applied — {} sources, {} samples, eval loss: {:.4}",
                plan.version,
                plan.sources.len(),
                plan.num_samples,
                post_metrics.eval_loss
            )
        };

        UpdateResult {
            version: plan.version,
            status,
            pre_metrics,
            post_metrics,
            sources_incorporated: plan.sources.len(),
            samples_trained: plan.num_samples,
            regression_detected,
            rolled_back,
            summary,
        }
    }

    /// Get the knowledge version store
    pub fn version_store(&self) -> &KnowledgeVersionStore {
        &self.version_store
    }

    /// Get the EWC regularizer (for training integration)
    pub fn regularizer(&self) -> &EwcRegularizer {
        &self.regularizer
    }

    /// Number of successful updates
    pub fn update_count(&self) -> usize {
        self.update_count
    }
}

/// A planned knowledge update (before execution)
#[derive(Debug, Clone)]
pub struct UpdatePlan {
    pub version: String,
    pub sources: Vec<KnowledgeSource>,
    pub num_samples: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub parent_version: Option<String>,
}

fn chrono_now_stub() -> String {
    // In production, use chrono::Utc::now().to_rfc3339()
    "2026-03-02T00:00:00Z".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continual::knowledge_version::KnowledgeSourceType;

    fn test_source(id: &str) -> KnowledgeSource {
        KnowledgeSource {
            source_type: KnowledgeSourceType::PubMed,
            identifier: id.to_string(),
            title: format!("Article {id}"),
            pub_date: Some("2026-01-01".to_string()),
            domains: vec!["cardiology".to_string()],
        }
    }

    fn good_metrics() -> UpdateMetrics {
        UpdateMetrics {
            train_loss: 0.5,
            eval_loss: 0.6,
            medqa_accuracy: Some(0.75),
            safety_score: Some(0.95),
            ewc_penalty: 0.1,
        }
    }

    #[test]
    fn plan_filters_existing_sources() {
        let config = UpdateEngineConfig::default();
        let mut engine = UpdateEngine::new(config);

        // First update with source A
        let plan = engine.plan_update(vec![test_source("A")], 100);
        let result = engine.execute_update(plan, good_metrics(), good_metrics(), None);
        assert_eq!(result.status, UpdateStatus::Success);

        // Second plan: source A should be filtered, B should remain
        let plan2 = engine.plan_update(vec![test_source("A"), test_source("B")], 100);
        assert_eq!(plan2.sources.len(), 1);
        assert_eq!(plan2.sources[0].identifier, "B");
    }

    #[test]
    fn regression_triggers_rollback() {
        let config = UpdateEngineConfig {
            max_regression: 0.02,
            auto_rollback: true,
            ..Default::default()
        };
        let mut engine = UpdateEngine::new(config);

        let pre = UpdateMetrics {
            train_loss: 0.5,
            eval_loss: 0.6,
            medqa_accuracy: Some(0.80),
            safety_score: Some(0.95),
            ewc_penalty: 0.1,
        };
        let post = UpdateMetrics {
            train_loss: 0.4,
            eval_loss: 0.7,
            medqa_accuracy: Some(0.75), // 5% regression > 2% threshold
            safety_score: Some(0.95),
            ewc_penalty: 0.3,
        };

        let plan = engine.plan_update(vec![test_source("X")], 100);
        let result = engine.execute_update(plan, pre, post, None);

        assert!(result.regression_detected);
        assert!(result.rolled_back);
        assert_eq!(result.status, UpdateStatus::RolledBack);
        assert_eq!(engine.update_count(), 0); // Not counted
    }

    #[test]
    fn successful_update_versions_correctly() {
        let mut engine = UpdateEngine::new(UpdateEngineConfig::default());

        let plan = engine.plan_update(vec![test_source("S1"), test_source("S2")], 500);
        let result = engine.execute_update(plan, good_metrics(), good_metrics(), None);

        assert_eq!(result.status, UpdateStatus::Success);
        assert!(!result.rolled_back);
        assert_eq!(engine.update_count(), 1);
        assert!(engine.version_store().has_source("S1"));
        assert!(engine.version_store().has_source("S2"));
    }

    #[test]
    fn version_numbering() {
        let mut engine = UpdateEngine::new(UpdateEngineConfig::default());

        for i in 0..3 {
            let plan = engine.plan_update(vec![test_source(&format!("src_{i}"))], 100);
            engine.execute_update(plan, good_metrics(), good_metrics(), None);
        }

        assert_eq!(engine.update_count(), 3);
        let lineage = engine.version_store().lineage();
        assert_eq!(lineage.len(), 3);
    }

    #[test]
    fn ewc_integration() {
        let mut engine = UpdateEngine::new(UpdateEngineConfig::default());

        let mut fisher = FisherInformation::new("base_knowledge");
        fisher
            .fisher_diag
            .insert("layer1.weight".into(), vec![1.0, 2.0, 3.0]);
        fisher
            .optimal_params
            .insert("layer1.weight".into(), vec![0.1, 0.2, 0.3]);

        let plan = engine.plan_update(vec![test_source("F1")], 200);
        let result = engine.execute_update(plan, good_metrics(), good_metrics(), Some(fisher));

        assert_eq!(result.status, UpdateStatus::Success);
        assert_eq!(engine.regularizer().tasks_consolidated, 1);

        // EWC should penalize deviation from optimal params
        let penalty = engine
            .regularizer()
            .penalty("layer1.weight", &[5.0, 5.0, 5.0]);
        assert!(
            penalty > 0.0,
            "EWC penalty should be positive for deviated params"
        );
    }
}
