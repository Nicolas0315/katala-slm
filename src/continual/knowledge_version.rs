//! Knowledge Version Control
//!
//! Tracks what medical knowledge the model has learned and when.
//! Every update is versioned so we can:
//! - Know which guidelines were incorporated
//! - Roll back if a knowledge update causes regression
//! - Report "this answer is based on guidelines as of 2026-03"
//! - Audit the training data provenance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single knowledge update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeUpdate {
    /// Unique version identifier (semver-ish: "knowledge-v1.3.0")
    pub version: String,
    /// ISO 8601 timestamp of when this knowledge was incorporated
    pub timestamp: String,
    /// Human-readable description
    pub description: String,
    /// Sources incorporated in this update
    pub sources: Vec<KnowledgeSource>,
    /// Number of training samples in this update
    pub num_samples: usize,
    /// Training metrics after this update
    pub metrics: Option<UpdateMetrics>,
    /// Parent version (for lineage tracking)
    pub parent_version: Option<String>,
    /// EWC Fisher info hash (for reproducibility)
    pub fisher_hash: Option<String>,
}

/// A knowledge source with provenance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSource {
    /// Source type (PubMed, Guideline, ClinicalTrial, etc.)
    pub source_type: KnowledgeSourceType,
    /// Identifier (PMID, DOI, URL, etc.)
    pub identifier: String,
    /// Title
    pub title: String,
    /// Publication date (ISO 8601)
    pub pub_date: Option<String>,
    /// Medical domain tags
    pub domains: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeSourceType {
    PubMed,
    ClinicalGuideline,
    ClinicalTrial,
    Cochrane,
    JapaneseGuideline,
    WHOGuideline,
    CDCGuidance,
    ExpertConsensus,
    CaseReport,
}

/// Training metrics snapshot for a knowledge version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMetrics {
    pub train_loss: f32,
    pub eval_loss: f32,
    pub medqa_accuracy: Option<f32>,
    pub safety_score: Option<f32>,
    pub ewc_penalty: f32,
}

/// Knowledge version store — the "git" for model knowledge
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeVersionStore {
    /// All versions in chronological order
    pub versions: Vec<KnowledgeUpdate>,
    /// Current active version
    pub current_version: Option<String>,
    /// Domain → latest version that updated it
    pub domain_versions: HashMap<String, String>,
}

impl KnowledgeVersionStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new knowledge update
    pub fn register_update(&mut self, update: KnowledgeUpdate) {
        // Track which domains were updated
        for source in &update.sources {
            for domain in &source.domains {
                self.domain_versions
                    .insert(domain.clone(), update.version.clone());
            }
        }

        self.current_version = Some(update.version.clone());
        self.versions.push(update);
    }

    /// Get the current knowledge version
    pub fn current(&self) -> Option<&KnowledgeUpdate> {
        self.current_version
            .as_ref()
            .and_then(|v| self.versions.iter().rfind(|u| u.version == *v))
    }

    /// Get all updates for a specific medical domain
    pub fn domain_history(&self, domain: &str) -> Vec<&KnowledgeUpdate> {
        self.versions
            .iter()
            .filter(|u| {
                u.sources
                    .iter()
                    .any(|s| s.domains.iter().any(|d| d == domain))
            })
            .collect()
    }

    /// Get the version lineage (version chain from current back to initial)
    pub fn lineage(&self) -> Vec<&KnowledgeUpdate> {
        let mut chain = Vec::new();
        let mut current = self.current_version.as_deref();
        while let Some(version) = current {
            if let Some(update) = self.versions.iter().find(|u| u.version == version) {
                chain.push(update);
                current = update.parent_version.as_deref();
            } else {
                break;
            }
        }
        chain
    }

    /// Check if a specific source has been incorporated
    pub fn has_source(&self, identifier: &str) -> bool {
        self.versions
            .iter()
            .any(|v| v.sources.iter().any(|s| s.identifier == identifier))
    }

    /// Total number of training samples across all versions
    pub fn total_samples(&self) -> usize {
        self.versions.iter().map(|v| v.num_samples).sum()
    }

    /// Generate a knowledge provenance report for a response
    pub fn provenance_report(&self, domains: &[String]) -> String {
        let mut report = String::from("📚 Knowledge Provenance:\n");
        report.push_str(&format!(
            "  Current version: {}\n",
            self.current_version.as_deref().unwrap_or("none")
        ));
        report.push_str(&format!("  Total updates: {}\n", self.versions.len()));
        report.push_str(&format!("  Total samples: {}\n", self.total_samples()));

        if !domains.is_empty() {
            report.push_str("  Domain-specific versions:\n");
            for domain in domains {
                if let Some(version) = self.domain_versions.get(domain) {
                    report.push_str(&format!("    {}: {}\n", domain, version));
                } else {
                    report.push_str(&format!("    {}: (no specialized knowledge)\n", domain));
                }
            }
        }

        report
    }

    /// Serialize the store to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_update(version: &str, parent: Option<&str>) -> KnowledgeUpdate {
        KnowledgeUpdate {
            version: version.to_string(),
            timestamp: "2026-03-02T00:00:00Z".to_string(),
            description: format!("Test update {version}"),
            sources: vec![KnowledgeSource {
                source_type: KnowledgeSourceType::PubMed,
                identifier: format!("PMID:{version}"),
                title: "Test Article".to_string(),
                pub_date: Some("2026-01-01".to_string()),
                domains: vec!["cardiology".to_string()],
            }],
            num_samples: 100,
            metrics: None,
            parent_version: parent.map(String::from),
            fisher_hash: None,
        }
    }

    #[test]
    fn version_registration_and_lookup() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));
        store.register_update(sample_update("v1.1.0", Some("v1.0.0")));

        assert_eq!(store.current_version.as_deref(), Some("v1.1.0"));
        assert_eq!(store.versions.len(), 2);
        assert_eq!(store.total_samples(), 200);
    }

    #[test]
    fn lineage_tracking() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));
        store.register_update(sample_update("v1.1.0", Some("v1.0.0")));
        store.register_update(sample_update("v1.2.0", Some("v1.1.0")));

        let lineage = store.lineage();
        assert_eq!(lineage.len(), 3);
        assert_eq!(lineage[0].version, "v1.2.0");
        assert_eq!(lineage[1].version, "v1.1.0");
        assert_eq!(lineage[2].version, "v1.0.0");
    }

    #[test]
    fn domain_history() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));

        let cardio = store.domain_history("cardiology");
        assert_eq!(cardio.len(), 1);

        let neuro = store.domain_history("neurology");
        assert_eq!(neuro.len(), 0);
    }

    #[test]
    fn source_tracking() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));

        assert!(store.has_source("PMID:v1.0.0"));
        assert!(!store.has_source("PMID:unknown"));
    }

    #[test]
    fn json_roundtrip() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));

        let json = store.to_json().unwrap();
        let restored = KnowledgeVersionStore::from_json(&json).unwrap();
        assert_eq!(restored.versions.len(), 1);
        assert_eq!(restored.current_version, Some("v1.0.0".to_string()));
    }

    #[test]
    fn provenance_report() {
        let mut store = KnowledgeVersionStore::new();
        store.register_update(sample_update("v1.0.0", None));

        let report = store.provenance_report(&["cardiology".to_string(), "neurology".to_string()]);
        assert!(report.contains("v1.0.0"));
        assert!(report.contains("cardiology"));
        assert!(report.contains("no specialized knowledge"));
    }
}
