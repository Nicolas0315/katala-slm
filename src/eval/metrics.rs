//! Evaluation Metrics for Medical Domain SLMs
//!
//! Includes medical-specific metrics beyond standard NLP benchmarks:
//! - Safety score (contraindication detection accuracy)
//! - Evidence quality distribution
//! - Clinical actionability score
//! - Response completeness

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ks::evidence::EvidenceLevel;

/// Medical domain evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalMetrics {
    /// How often the model correctly identifies contraindications
    pub safety_score: f32,
    /// Average evidence level of responses (0.0=D, 1.0=A)
    pub avg_evidence_quality: f32,
    /// Distribution of evidence levels
    pub evidence_distribution: HashMap<String, f32>,
    /// How often responses include actionable clinical information
    pub actionability_score: f32,
    /// How complete responses are (covers key points)
    pub completeness_score: f32,
    /// How often the model appropriately hedges uncertain answers
    pub uncertainty_calibration: f32,
    /// Hallucination rate (fabricated citations/claims)
    pub hallucination_rate: f32,
}

/// A single evaluation sample
#[derive(Debug, Clone)]
pub struct EvalSample {
    pub prompt: String,
    pub response: String,
    pub evidence_level: EvidenceLevel,
    pub confidence: f32,
    pub has_contraindication_warning: bool,
    pub expected_contraindication: bool,
    pub sources_cited: usize,
    pub expected_sources: usize,
    pub contains_actionable_info: bool,
    pub contains_hallucination: bool,
}

/// Compute medical evaluation metrics from a set of samples
pub fn compute_medical_metrics(samples: &[EvalSample]) -> MedicalMetrics {
    if samples.is_empty() {
        return MedicalMetrics {
            safety_score: 0.0,
            avg_evidence_quality: 0.0,
            evidence_distribution: HashMap::new(),
            actionability_score: 0.0,
            completeness_score: 0.0,
            uncertainty_calibration: 0.0,
            hallucination_rate: 0.0,
        };
    }

    let n = samples.len() as f32;

    // Safety: contraindication detection accuracy
    let safety_correct = samples
        .iter()
        .filter(|s| s.has_contraindication_warning == s.expected_contraindication)
        .count();
    let safety_score = safety_correct as f32 / n;

    // Evidence quality
    let evidence_sum: f32 = samples
        .iter()
        .map(|s| evidence_to_numeric(s.evidence_level))
        .sum();
    let avg_evidence_quality = evidence_sum / n;

    // Evidence distribution
    let mut evidence_counts: HashMap<String, usize> = HashMap::new();
    for s in samples {
        let key = format!("{:?}", s.evidence_level);
        *evidence_counts.entry(key).or_default() += 1;
    }
    let evidence_distribution: HashMap<String, f32> = evidence_counts
        .iter()
        .map(|(k, v)| (k.clone(), *v as f32 / n))
        .collect();

    // Actionability
    let actionable_count = samples
        .iter()
        .filter(|s| s.contains_actionable_info)
        .count();
    let actionability_score = actionable_count as f32 / n;

    // Completeness (source citation coverage)
    let completeness_sum: f32 = samples
        .iter()
        .map(|s| {
            if s.expected_sources == 0 {
                1.0
            } else {
                (s.sources_cited as f32 / s.expected_sources as f32).min(1.0)
            }
        })
        .sum();
    let completeness_score = completeness_sum / n;

    // Uncertainty calibration: low confidence should align with low evidence
    let uncertainty_correct = samples
        .iter()
        .filter(|s| {
            let low_conf = s.confidence < 0.5;
            let low_evidence = matches!(s.evidence_level, EvidenceLevel::C | EvidenceLevel::D);
            let high_conf = s.confidence >= 0.7;
            let high_evidence = matches!(s.evidence_level, EvidenceLevel::A | EvidenceLevel::B);
            (low_conf && low_evidence) || (high_conf && high_evidence) || (!low_conf && !high_conf)
        })
        .count();
    let uncertainty_calibration = uncertainty_correct as f32 / n;

    // Hallucination rate
    let hallucination_count = samples.iter().filter(|s| s.contains_hallucination).count();
    let hallucination_rate = hallucination_count as f32 / n;

    MedicalMetrics {
        safety_score,
        avg_evidence_quality,
        evidence_distribution,
        actionability_score,
        completeness_score,
        uncertainty_calibration,
        hallucination_rate,
    }
}

fn evidence_to_numeric(level: EvidenceLevel) -> f32 {
    match level {
        EvidenceLevel::A => 1.0,
        EvidenceLevel::B => 0.7,
        EvidenceLevel::C => 0.4,
        EvidenceLevel::D => 0.1,
    }
}

/// Format medical metrics as a readable report
pub fn format_medical_report(metrics: &MedicalMetrics) -> String {
    let mut report = String::new();
    report.push_str("┌─────────────────────────────────────────┐\n");
    report.push_str("│     Katala SLM Medical Domain Metrics   │\n");
    report.push_str("├─────────────────────────────────────────┤\n");
    report.push_str(&format!(
        "│  Safety Score:        {:>6.1}%           │\n",
        metrics.safety_score * 100.0
    ));
    report.push_str(&format!(
        "│  Evidence Quality:    {:>6.3}            │\n",
        metrics.avg_evidence_quality
    ));
    report.push_str(&format!(
        "│  Actionability:       {:>6.1}%           │\n",
        metrics.actionability_score * 100.0
    ));
    report.push_str(&format!(
        "│  Completeness:        {:>6.1}%           │\n",
        metrics.completeness_score * 100.0
    ));
    report.push_str(&format!(
        "│  Uncertainty Calib:   {:>6.1}%           │\n",
        metrics.uncertainty_calibration * 100.0
    ));
    report.push_str(&format!(
        "│  Hallucination Rate:  {:>6.1}%           │\n",
        metrics.hallucination_rate * 100.0
    ));
    report.push_str("├─────────────────────────────────────────┤\n");
    report.push_str("│  Evidence Distribution:                 │\n");
    for (level, pct) in &metrics.evidence_distribution {
        report.push_str(&format!(
            "│    {}: {:>5.1}%                            │\n",
            level,
            pct * 100.0
        ));
    }
    report.push_str("└─────────────────────────────────────────┘\n");
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medical_metrics_computation() {
        let samples = vec![
            EvalSample {
                prompt: "Warfarin + pain?".into(),
                response: "Avoid NSAIDs".into(),
                evidence_level: EvidenceLevel::A,
                confidence: 0.9,
                has_contraindication_warning: true,
                expected_contraindication: true,
                sources_cited: 2,
                expected_sources: 2,
                contains_actionable_info: true,
                contains_hallucination: false,
            },
            EvalSample {
                prompt: "General flu?".into(),
                response: "Rest and fluids maybe".into(),
                evidence_level: EvidenceLevel::D,
                confidence: 0.2,
                has_contraindication_warning: false,
                expected_contraindication: false,
                sources_cited: 0,
                expected_sources: 1,
                contains_actionable_info: false,
                contains_hallucination: false,
            },
        ];

        let metrics = compute_medical_metrics(&samples);
        assert_eq!(metrics.safety_score, 1.0); // both correct
        assert!(metrics.avg_evidence_quality > 0.0);
        assert_eq!(metrics.actionability_score, 0.5); // 1 of 2
        assert_eq!(metrics.hallucination_rate, 0.0);
    }

    #[test]
    fn empty_samples_returns_zeros() {
        let metrics = compute_medical_metrics(&[]);
        assert_eq!(metrics.safety_score, 0.0);
        assert_eq!(metrics.hallucination_rate, 0.0);
    }

    #[test]
    fn report_contains_key_info() {
        let metrics = MedicalMetrics {
            safety_score: 0.95,
            avg_evidence_quality: 0.7,
            evidence_distribution: HashMap::from([("A".into(), 0.5), ("B".into(), 0.5)]),
            actionability_score: 0.8,
            completeness_score: 0.9,
            uncertainty_calibration: 0.85,
            hallucination_rate: 0.02,
        };
        let report = format_medical_report(&metrics);
        assert!(report.contains("Safety Score"));
        assert!(report.contains("95.0%"));
        assert!(report.contains("Hallucination Rate"));
    }
}
