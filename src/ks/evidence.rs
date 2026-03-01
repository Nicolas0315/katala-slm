use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceLevel {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Default)]
pub struct EvidenceClassifier;

impl EvidenceClassifier {
    pub fn classify(&self, answer: &str, sources: &[String]) -> EvidenceLevel {
        let lower = answer.to_lowercase();
        let has_guideline = sources.iter().any(|s| {
            let s = s.to_lowercase();
            s.contains("guideline") || s.contains("who") || s.contains("cdc") || s.contains("nejm")
        });
        let has_uncertainty = lower.contains("unclear")
            || lower.contains("unknown")
            || lower.contains("insufficient")
            || lower.contains("may");

        match (has_guideline, sources.len(), has_uncertainty) {
            (true, n, false) if n >= 2 => EvidenceLevel::A,
            (true, _, _) => EvidenceLevel::B,
            (false, n, false) if n >= 1 => EvidenceLevel::C,
            _ => EvidenceLevel::D,
        }
    }
}
