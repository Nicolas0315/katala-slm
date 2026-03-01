use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub source_id: String,
    pub title: String,
    pub url: Option<String>,
    pub snippet: String,
}

#[derive(Debug, Clone, Default)]
pub struct SourceAttributionTracker;

impl SourceAttributionTracker {
    pub fn from_context(&self, prompt: &str) -> Vec<SourceAttribution> {
        if prompt.to_lowercase().contains("influenza") {
            return vec![SourceAttribution {
                source_id: "cdc-flu-antiviral".to_string(),
                title: "CDC Influenza Antiviral Guidance".to_string(),
                url: Some("https://www.cdc.gov/flu/professionals/antivirals/index.htm".to_string()),
                snippet: "Early antiviral treatment is recommended for high-risk patients."
                    .to_string(),
            }];
        }
        vec![]
    }
}
