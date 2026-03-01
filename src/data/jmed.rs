use serde::{Deserialize, Serialize};

use crate::{data::dataset::DatasetRecord, ks::evidence::EvidenceLevel};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JapaneseMedicalGuideline {
    pub id: String,
    pub title: String,
    pub organization: Option<String>,
    pub section: Option<String>,
    pub recommendation_text: String,
    pub evidence_level: Option<String>,
}

pub fn parse_jmed_json(input: &str) -> serde_json::Result<Vec<JapaneseMedicalGuideline>> {
    if let Ok(items) = serde_json::from_str::<Vec<JapaneseMedicalGuideline>>(input) {
        return Ok(items);
    }

    #[derive(Debug, Deserialize)]
    struct Wrapper {
        guidelines: Vec<JapaneseMedicalGuideline>,
    }

    serde_json::from_str::<Wrapper>(input).map(|w| w.guidelines)
}

pub fn extract_evidence_level_from_text(text: &str) -> EvidenceLevel {
    let t = text.to_lowercase();
    if t.contains("grade a")
        || t.contains("level a")
        || t.contains("推奨a")
        || t.contains("強く推奨")
    {
        EvidenceLevel::A
    } else if t.contains("grade b") || t.contains("level b") || t.contains("推奨b") {
        EvidenceLevel::B
    } else if t.contains("grade c") || t.contains("level c") || t.contains("推奨c") {
        EvidenceLevel::C
    } else {
        EvidenceLevel::D
    }
}

impl JapaneseMedicalGuideline {
    pub fn inferred_evidence_level(&self) -> EvidenceLevel {
        if let Some(level) = &self.evidence_level {
            extract_evidence_level_from_text(level)
        } else {
            extract_evidence_level_from_text(&self.recommendation_text)
        }
    }

    pub fn to_dataset_record(&self) -> DatasetRecord {
        DatasetRecord {
            id: format!("jmed:{}", self.id),
            prompt: self.title.clone(),
            answer: self.recommendation_text.clone(),
            evidence_level: self.inferred_evidence_level(),
            sources: vec![self
                .organization
                .clone()
                .unwrap_or_else(|| "JMED".to_string())],
        }
    }
}
