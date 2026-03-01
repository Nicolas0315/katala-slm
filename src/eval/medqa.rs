//! MedQA / JMED Benchmark Harness
//!
//! Supports USMLE-style multiple choice questions (MedQA)
//! and Japanese medical licensing exam questions (JMED).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ks::evidence::EvidenceLevel;

/// A single benchmark question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkQuestion {
    pub id: String,
    pub question: String,
    pub options: Vec<String>,
    pub correct_answer: usize,
    pub category: Option<String>,
    pub difficulty: Option<QuestionDifficulty>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuestionDifficulty {
    Easy,
    Medium,
    Hard,
}

/// Model's answer with KS verification metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkAnswer {
    pub question_id: String,
    pub selected_option: usize,
    pub confidence: f32,
    pub evidence_level: EvidenceLevel,
    pub reasoning: String,
    pub latency_ms: f64,
}

/// Overall benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub dataset_name: String,
    pub total_questions: usize,
    pub correct: usize,
    pub accuracy: f32,
    pub avg_confidence: f32,
    pub avg_latency_ms: f64,
    pub evidence_distribution: HashMap<String, usize>,
    pub category_accuracy: HashMap<String, f32>,
    pub calibration_error: f32,
    pub confidence_correct_correlation: f32,
}

/// A benchmark harness that evaluates model answers
#[derive(Debug, Default)]
pub struct BenchmarkHarness {
    questions: Vec<BenchmarkQuestion>,
    answers: Vec<BenchmarkAnswer>,
}

impl BenchmarkHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_questions(&mut self, questions: Vec<BenchmarkQuestion>) {
        self.questions = questions;
    }

    /// Load MedQA-format JSONL data
    pub fn load_medqa_jsonl(jsonl: &str) -> Vec<BenchmarkQuestion> {
        jsonl
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.is_empty() {
                    return None;
                }
                serde_json::from_str::<BenchmarkQuestion>(line).ok()
            })
            .collect()
    }

    /// Load JMED-format questions (Japanese medical exam)
    pub fn load_jmed_jsonl(jsonl: &str) -> Vec<BenchmarkQuestion> {
        // Same format, different dataset
        Self::load_medqa_jsonl(jsonl)
    }

    pub fn record_answer(&mut self, answer: BenchmarkAnswer) {
        self.answers.push(answer);
    }

    /// Evaluate all recorded answers against questions
    pub fn evaluate(&self, dataset_name: &str) -> BenchmarkResults {
        let mut correct = 0;
        let mut total_confidence = 0.0f32;
        let mut total_latency = 0.0f64;
        let mut evidence_dist: HashMap<String, usize> = HashMap::new();
        let mut category_correct: HashMap<String, (usize, usize)> = HashMap::new();
        let mut confidence_correct_pairs: Vec<(f32, bool)> = Vec::new();

        let question_map: HashMap<&str, &BenchmarkQuestion> =
            self.questions.iter().map(|q| (q.id.as_str(), q)).collect();

        for answer in &self.answers {
            let is_correct = question_map
                .get(answer.question_id.as_str())
                .map(|q| q.correct_answer == answer.selected_option)
                .unwrap_or(false);

            if is_correct {
                correct += 1;
            }

            total_confidence += answer.confidence;
            total_latency += answer.latency_ms;

            let evidence_key = format!("{:?}", answer.evidence_level);
            *evidence_dist.entry(evidence_key).or_default() += 1;

            if let Some(q) = question_map.get(answer.question_id.as_str()) {
                if let Some(cat) = &q.category {
                    let entry = category_correct.entry(cat.clone()).or_insert((0, 0));
                    entry.1 += 1;
                    if is_correct {
                        entry.0 += 1;
                    }
                }
            }

            confidence_correct_pairs.push((answer.confidence, is_correct));
        }

        let n = self.answers.len().max(1);
        let accuracy = correct as f32 / n as f32;
        let avg_confidence = total_confidence / n as f32;
        let avg_latency = total_latency / n as f64;

        let category_accuracy: HashMap<String, f32> = category_correct
            .iter()
            .map(|(k, (c, t))| (k.clone(), *c as f32 / (*t).max(1) as f32))
            .collect();

        let calibration_error = compute_ece(&confidence_correct_pairs, 10);
        let correlation = compute_correlation(&confidence_correct_pairs);

        BenchmarkResults {
            dataset_name: dataset_name.to_string(),
            total_questions: self.questions.len(),
            correct,
            accuracy,
            avg_confidence,
            avg_latency_ms: avg_latency,
            evidence_distribution: evidence_dist,
            category_accuracy,
            calibration_error,
            confidence_correct_correlation: correlation,
        }
    }

    /// Generate a formatted report
    pub fn report(&self, results: &BenchmarkResults) -> String {
        let mut report = String::new();
        report.push_str(&format!("╔══════════════════════════════════════════╗\n"));
        report.push_str(&format!(
            "║  Katala SLM Benchmark: {:>17} ║\n",
            results.dataset_name
        ));
        report.push_str(&format!("╠══════════════════════════════════════════╣\n"));
        report.push_str(&format!(
            "║  Accuracy:         {:>6.1}% ({}/{})    ║\n",
            results.accuracy * 100.0,
            results.correct,
            results.total_questions
        ));
        report.push_str(&format!(
            "║  Avg Confidence:   {:>6.3}              ║\n",
            results.avg_confidence
        ));
        report.push_str(&format!(
            "║  Avg Latency:      {:>6.1}ms             ║\n",
            results.avg_latency_ms
        ));
        report.push_str(&format!(
            "║  ECE (calibration):{:>6.3}              ║\n",
            results.calibration_error
        ));
        report.push_str(&format!(
            "║  Conf-Acc Corr:    {:>6.3}              ║\n",
            results.confidence_correct_correlation
        ));
        report.push_str(&format!("╠══════════════════════════════════════════╣\n"));
        report.push_str("║  Evidence Distribution:                  ║\n");
        for (level, count) in &results.evidence_distribution {
            report.push_str(&format!(
                "║    {}: {:>4}                              ║\n",
                level, count
            ));
        }
        report.push_str(&format!("╚══════════════════════════════════════════╝\n"));
        report
    }
}

/// Expected Calibration Error (ECE)
fn compute_ece(pairs: &[(f32, bool)], n_bins: usize) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }

    let mut bins: Vec<Vec<(f32, bool)>> = vec![vec![]; n_bins];
    for &(conf, correct) in pairs {
        let bin = ((conf * n_bins as f32) as usize).min(n_bins - 1);
        bins[bin].push((conf, correct));
    }

    let total = pairs.len() as f32;
    let mut ece = 0.0f32;

    for bin in &bins {
        if bin.is_empty() {
            continue;
        }
        let avg_conf: f32 = bin.iter().map(|(c, _)| c).sum::<f32>() / bin.len() as f32;
        let avg_acc: f32 = bin.iter().filter(|(_, c)| *c).count() as f32 / bin.len() as f32;
        ece += (bin.len() as f32 / total) * (avg_conf - avg_acc).abs();
    }

    ece
}

/// Pearson correlation between confidence and correctness
fn compute_correlation(pairs: &[(f32, bool)]) -> f32 {
    if pairs.len() < 2 {
        return 0.0;
    }
    let n = pairs.len() as f32;
    let confs: Vec<f32> = pairs.iter().map(|(c, _)| *c).collect();
    let accs: Vec<f32> = pairs
        .iter()
        .map(|(_, c)| if *c { 1.0 } else { 0.0 })
        .collect();

    let mean_conf = confs.iter().sum::<f32>() / n;
    let mean_acc = accs.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_conf = 0.0f32;
    let mut var_acc = 0.0f32;

    for i in 0..pairs.len() {
        let dc = confs[i] - mean_conf;
        let da = accs[i] - mean_acc;
        cov += dc * da;
        var_conf += dc * dc;
        var_acc += da * da;
    }

    let denom = (var_conf * var_acc).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_harness_evaluates_correctly() {
        let mut harness = BenchmarkHarness::new();
        harness.load_questions(vec![
            BenchmarkQuestion {
                id: "q1".into(),
                question: "What is the first-line treatment for influenza?".into(),
                options: vec![
                    "Oseltamivir".into(),
                    "Amoxicillin".into(),
                    "Metformin".into(),
                    "Aspirin".into(),
                ],
                correct_answer: 0,
                category: Some("Infectious Disease".into()),
                difficulty: Some(QuestionDifficulty::Easy),
            },
            BenchmarkQuestion {
                id: "q2".into(),
                question: "HbA1c target for most diabetic patients?".into(),
                options: vec!["<5%".into(), "<7%".into(), "<10%".into(), "<12%".into()],
                correct_answer: 1,
                category: Some("Endocrinology".into()),
                difficulty: Some(QuestionDifficulty::Medium),
            },
        ]);

        harness.record_answer(BenchmarkAnswer {
            question_id: "q1".into(),
            selected_option: 0,
            confidence: 0.85,
            evidence_level: EvidenceLevel::A,
            reasoning: "Oseltamivir is first-line per CDC guidelines".into(),
            latency_ms: 42.5,
        });
        harness.record_answer(BenchmarkAnswer {
            question_id: "q2".into(),
            selected_option: 2, // wrong
            confidence: 0.4,
            evidence_level: EvidenceLevel::C,
            reasoning: "Unclear target".into(),
            latency_ms: 38.0,
        });

        let results = harness.evaluate("MedQA-test");
        assert_eq!(results.correct, 1);
        assert_eq!(results.total_questions, 2);
        assert!((results.accuracy - 0.5).abs() < 0.01);
        assert!(results.avg_confidence > 0.0);
        assert!(results.avg_latency_ms > 0.0);
    }

    #[test]
    fn ece_is_bounded() {
        let pairs = vec![
            (0.9, true),
            (0.8, true),
            (0.3, false),
            (0.2, false),
            (0.6, true),
        ];
        let ece = compute_ece(&pairs, 10);
        assert!(
            ece >= 0.0 && ece <= 1.0,
            "ECE should be in [0,1], got {ece}"
        );
    }

    #[test]
    fn perfect_calibration_has_low_ece() {
        // Perfectly calibrated: high confidence → correct, low → incorrect
        let pairs = vec![
            (0.95, true),
            (0.90, true),
            (0.85, true),
            (0.10, false),
            (0.05, false),
        ];
        let ece = compute_ece(&pairs, 10);
        assert!(
            ece < 0.2,
            "Well-calibrated model should have low ECE, got {ece}"
        );
    }

    #[test]
    fn report_format() {
        let mut harness = BenchmarkHarness::new();
        harness.load_questions(vec![BenchmarkQuestion {
            id: "q1".into(),
            question: "Test?".into(),
            options: vec!["A".into(), "B".into()],
            correct_answer: 0,
            category: None,
            difficulty: None,
        }]);
        harness.record_answer(BenchmarkAnswer {
            question_id: "q1".into(),
            selected_option: 0,
            confidence: 0.9,
            evidence_level: EvidenceLevel::A,
            reasoning: "Correct".into(),
            latency_ms: 10.0,
        });
        let results = harness.evaluate("Test");
        let report = harness.report(&results);
        assert!(report.contains("Katala SLM Benchmark"));
        assert!(report.contains("100.0%"));
    }

    #[test]
    fn medqa_jsonl_parsing() {
        let jsonl = r#"{"id":"q1","question":"Test?","options":["A","B","C","D"],"correct_answer":2,"category":"Cardiology","difficulty":"Medium"}
{"id":"q2","question":"Another?","options":["X","Y"],"correct_answer":0,"category":null,"difficulty":null}"#;

        let questions = BenchmarkHarness::load_medqa_jsonl(jsonl);
        assert_eq!(questions.len(), 2);
        assert_eq!(questions[0].id, "q1");
        assert_eq!(questions[0].correct_answer, 2);
        assert_eq!(questions[1].options.len(), 2);
    }
}
