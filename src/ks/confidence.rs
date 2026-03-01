use serde::{Deserialize, Serialize};

use super::evidence::EvidenceLevel;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceScore {
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct ConfidenceScorer;

impl ConfidenceScorer {
    pub fn score(
        &self,
        evidence_level: EvidenceLevel,
        contraindications: &[String],
        answer: &str,
    ) -> ConfidenceScore {
        let base = match evidence_level {
            EvidenceLevel::A => 0.9,
            EvidenceLevel::B => 0.75,
            EvidenceLevel::C => 0.55,
            EvidenceLevel::D => 0.3,
        };

        let contraindication_penalty = (contraindications.len() as f32 * 0.1).min(0.4);
        let hedge_penalty = if answer.to_lowercase().contains("consult") {
            0.03
        } else {
            0.0
        };
        let score = (base - contraindication_penalty - hedge_penalty).clamp(0.0, 1.0);
        ConfidenceScore { score }
    }
}
