use serde::{Deserialize, Serialize};

use super::{
    confidence::{ConfidenceScore, ConfidenceScorer},
    evidence::{AxisScores, EvidenceClassifier, EvidenceLevel},
    source::{SourceAttribution, SourceAttributionTracker},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedAnswer {
    pub answer: String,
    pub evidence_level: EvidenceLevel,
    pub axis_scores: AxisScores,
    pub composite_score: f32,
    pub sources: Vec<SourceAttribution>,
    pub confidence: f32,
    pub contraindications: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Verifier {
    evidence: EvidenceClassifier,
    confidence: ConfidenceScorer,
    source_tracker: SourceAttributionTracker,
}

impl Verifier {
    pub fn verify(&self, prompt: &str, answer: &str, extra_sources: &[String]) -> VerifiedAnswer {
        let sources = self.source_tracker.from_context(prompt);
        let mut source_labels: Vec<String> = sources.iter().map(|s| s.title.clone()).collect();
        source_labels.extend(extra_sources.iter().cloned());

        let contraindications = self.detect_contraindications(prompt, answer);
        let assessment = self
            .evidence
            .classify_with_context(prompt, answer, &source_labels);
        let ConfidenceScore { score } =
            self.confidence
                .score(assessment.evidence_level, &contraindications, answer);

        VerifiedAnswer {
            answer: answer.to_string(),
            evidence_level: assessment.evidence_level,
            axis_scores: assessment.axis_scores,
            composite_score: assessment.composite_score,
            sources,
            confidence: score,
            contraindications,
        }
    }

    fn detect_contraindications(&self, prompt: &str, answer: &str) -> Vec<String> {
        let p = prompt.to_lowercase();
        let a = answer.to_lowercase();
        let mut found = Vec::new();

        if (p.contains("pregnan") || p.contains("妊娠")) && a.contains("isotretinoin") {
            found.push("Potential teratogenic risk noted for pregnancy".to_string());
        }
        if (p.contains("warfarin") || p.contains("anticoagulant")) && a.contains("nsaid") {
            found.push("Potential bleeding risk with NSAID and anticoagulant use".to_string());
        }
        if p.contains("renal") && a.contains("metformin") {
            found.push("Renal impairment may contraindicate metformin continuation".to_string());
        }

        found
    }
}
