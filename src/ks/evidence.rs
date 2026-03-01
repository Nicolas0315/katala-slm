use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationAxis {
    Accuracy,
    Relevance,
    Recency,
    SourceQuality,
    Reproducibility,
    SampleSize,
    StudyDesign,
    BiasAssessment,
    ConflictOfInterest,
    PeerReview,
    Consistency,
    Generalizability,
    ClinicalSignificance,
    SafetyProfile,
    DoseResponse,
    MechanisticPlausibility,
    PatientPopulation,
    OutcomeMeasurement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceLevel {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisScores {
    pub accuracy: f32,
    pub relevance: f32,
    pub recency: f32,
    pub source_quality: f32,
    pub reproducibility: f32,
    pub sample_size: f32,
    pub study_design: f32,
    pub bias_assessment: f32,
    pub conflict_of_interest: f32,
    pub peer_review: f32,
    pub consistency: f32,
    pub generalizability: f32,
    pub clinical_significance: f32,
    pub safety_profile: f32,
    pub dose_response: f32,
    pub mechanistic_plausibility: f32,
    pub patient_population: f32,
    pub outcome_measurement: f32,
}

impl Default for AxisScores {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl AxisScores {
    pub fn new(initial: f32) -> Self {
        let v = initial.clamp(0.0, 1.0);
        Self {
            accuracy: v,
            relevance: v,
            recency: v,
            source_quality: v,
            reproducibility: v,
            sample_size: v,
            study_design: v,
            bias_assessment: v,
            conflict_of_interest: v,
            peer_review: v,
            consistency: v,
            generalizability: v,
            clinical_significance: v,
            safety_profile: v,
            dose_response: v,
            mechanistic_plausibility: v,
            patient_population: v,
            outcome_measurement: v,
        }
    }

    pub fn composite_score(&self) -> f32 {
        let weighted = [
            (self.accuracy, 1.3),
            (self.relevance, 1.2),
            (self.recency, 0.9),
            (self.source_quality, 1.3),
            (self.reproducibility, 1.0),
            (self.sample_size, 1.0),
            (self.study_design, 1.2),
            (self.bias_assessment, 0.9),
            (self.conflict_of_interest, 0.7),
            (self.peer_review, 1.0),
            (self.consistency, 1.0),
            (self.generalizability, 0.9),
            (self.clinical_significance, 1.1),
            (self.safety_profile, 1.1),
            (self.dose_response, 0.8),
            (self.mechanistic_plausibility, 0.8),
            (self.patient_population, 0.9),
            (self.outcome_measurement, 1.0),
        ];
        let (sum, total_weight) = weighted
            .into_iter()
            .fold((0.0_f32, 0.0_f32), |(s, w), (score, weight)| {
                (s + score.clamp(0.0, 1.0) * weight, w + weight)
            });
        (sum / total_weight).clamp(0.0, 1.0)
    }

    pub fn to_evidence_level(&self) -> EvidenceLevel {
        let score = self.composite_score();
        if score >= 0.62 {
            EvidenceLevel::A
        } else if score >= 0.52 {
            EvidenceLevel::B
        } else if score >= 0.38 {
            EvidenceLevel::C
        } else {
            EvidenceLevel::D
        }
    }

    fn clamp_all(&mut self) {
        self.accuracy = self.accuracy.clamp(0.0, 1.0);
        self.relevance = self.relevance.clamp(0.0, 1.0);
        self.recency = self.recency.clamp(0.0, 1.0);
        self.source_quality = self.source_quality.clamp(0.0, 1.0);
        self.reproducibility = self.reproducibility.clamp(0.0, 1.0);
        self.sample_size = self.sample_size.clamp(0.0, 1.0);
        self.study_design = self.study_design.clamp(0.0, 1.0);
        self.bias_assessment = self.bias_assessment.clamp(0.0, 1.0);
        self.conflict_of_interest = self.conflict_of_interest.clamp(0.0, 1.0);
        self.peer_review = self.peer_review.clamp(0.0, 1.0);
        self.consistency = self.consistency.clamp(0.0, 1.0);
        self.generalizability = self.generalizability.clamp(0.0, 1.0);
        self.clinical_significance = self.clinical_significance.clamp(0.0, 1.0);
        self.safety_profile = self.safety_profile.clamp(0.0, 1.0);
        self.dose_response = self.dose_response.clamp(0.0, 1.0);
        self.mechanistic_plausibility = self.mechanistic_plausibility.clamp(0.0, 1.0);
        self.patient_population = self.patient_population.clamp(0.0, 1.0);
        self.outcome_measurement = self.outcome_measurement.clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone)]
pub struct EvidenceAssessment {
    pub axis_scores: AxisScores,
    pub composite_score: f32,
    pub evidence_level: EvidenceLevel,
}

#[derive(Debug, Clone, Default)]
pub struct EvidenceClassifier;

impl EvidenceClassifier {
    pub fn classify(&self, answer: &str, sources: &[String]) -> EvidenceLevel {
        self.classify_with_context("", answer, sources)
            .evidence_level
    }

    pub fn classify_with_context(
        &self,
        prompt: &str,
        answer: &str,
        sources: &[String],
    ) -> EvidenceAssessment {
        let text = format!(
            "{}\n{}\n{}",
            prompt.to_lowercase(),
            answer.to_lowercase(),
            sources.join(" ").to_lowercase()
        );
        let mut axis_scores = AxisScores::new(0.5);

        axis_scores.accuracy = self.keyword_score(
            &text,
            &["accurate", "validated", "confirmed", "gold standard"],
            &["unclear", "unknown", "speculative"],
        );
        axis_scores.relevance = self.keyword_score(
            &text,
            &[
                "patient",
                "treatment",
                "diagnosis",
                "clinical",
                "management",
            ],
            &["off-topic", "unrelated"],
        );
        axis_scores.recency = self.keyword_score(
            &text,
            &["recent", "latest", "2023", "2024", "2025", "2026"],
            &["obsolete", "outdated", "historical only"],
        );
        axis_scores.source_quality = self.keyword_score(
            &text,
            &[
                "guideline",
                "who",
                "cdc",
                "nejm",
                "lancet",
                "cochrane",
                "consensus",
            ],
            &["blog", "anecdotal", "social media"],
        );
        axis_scores.reproducibility = self.keyword_score(
            &text,
            &["replicated", "reproducible", "independent validation"],
            &["single-center only", "not replicated"],
        );
        axis_scores.sample_size = self.keyword_score(
            &text,
            &[
                "meta-analysis",
                "multi-center",
                "multicenter",
                "large cohort",
                "n=",
            ],
            &["small sample", "case series", "few patients"],
        );
        axis_scores.study_design = self.keyword_score(
            &text,
            &[
                "randomized",
                "double-blind",
                "controlled trial",
                "systematic review",
                "rct",
            ],
            &["case report", "expert opinion only"],
        );
        axis_scores.bias_assessment = self.keyword_score(
            &text,
            &["risk of bias", "confounding adjusted", "blinded"],
            &["selection bias", "high risk of bias"],
        );
        axis_scores.conflict_of_interest = self.keyword_score(
            &text,
            &["no conflict", "independent funding", "disclosure"],
            &["industry sponsored", "financial conflict"],
        );
        axis_scores.peer_review = self.keyword_score(
            &text,
            &["peer-reviewed", "published in"],
            &["preprint", "not peer reviewed"],
        );
        axis_scores.consistency = self.keyword_score(
            &text,
            &["consistent", "convergent", "across studies"],
            &["inconsistent", "heterogeneous", "conflicting"],
        );
        axis_scores.generalizability = self.keyword_score(
            &text,
            &[
                "diverse",
                "population-based",
                "external validation",
                "real-world",
            ],
            &["narrow population", "single-site"],
        );
        axis_scores.clinical_significance = self.keyword_score(
            &text,
            &["mortality", "absolute risk", "nnt", "clinically meaningful"],
            &["not clinically significant", "surrogate only"],
        );
        axis_scores.safety_profile = self.keyword_score(
            &text,
            &["safe", "well tolerated", "adverse events monitored"],
            &["serious adverse event", "toxicity", "contraindicated"],
        );
        axis_scores.dose_response = self.keyword_score(
            &text,
            &["dose-response", "dose dependent", "titration"],
            &["no dose response"],
        );
        axis_scores.mechanistic_plausibility = self.keyword_score(
            &text,
            &[
                "mechanism",
                "pathway",
                "biological plausibility",
                "target engagement",
            ],
            &["mechanism unclear"],
        );
        axis_scores.patient_population = self.keyword_score(
            &text,
            &[
                "adults",
                "pediatric",
                "elderly",
                "pregnan",
                "comorbidity",
                "renal impairment",
            ],
            &["population unspecified"],
        );
        axis_scores.outcome_measurement = self.keyword_score(
            &text,
            &[
                "primary endpoint",
                "outcome",
                "follow-up",
                "validated scale",
            ],
            &["outcome not reported"],
        );

        self.apply_global_adjustments(answer, sources, &mut axis_scores);
        axis_scores.clamp_all();

        let composite_score = axis_scores.composite_score();
        let evidence_level = axis_scores.to_evidence_level();
        EvidenceAssessment {
            axis_scores,
            composite_score,
            evidence_level,
        }
    }

    fn keyword_score(&self, text: &str, positives: &[&str], negatives: &[&str]) -> f32 {
        let mut score: f32 = 0.5;
        for keyword in positives {
            if text.contains(keyword) {
                score += 0.1;
            }
        }
        for keyword in negatives {
            if text.contains(keyword) {
                score -= 0.15;
            }
        }
        score.clamp(0.0, 1.0)
    }

    fn apply_global_adjustments(
        &self,
        answer: &str,
        sources: &[String],
        axis_scores: &mut AxisScores,
    ) {
        let lower = answer.to_lowercase();
        let has_guideline = sources.iter().any(|s| {
            let s = s.to_lowercase();
            s.contains("guideline") || s.contains("who") || s.contains("cdc") || s.contains("nejm")
        });
        let has_uncertainty = lower.contains("unclear")
            || lower.contains("unknown")
            || lower.contains("insufficient")
            || lower.contains("may");

        if has_guideline {
            axis_scores.source_quality += 0.20;
            axis_scores.peer_review += 0.10;
            axis_scores.accuracy += 0.10;
            axis_scores.consistency += 0.08;
        }

        if sources.len() >= 2 {
            axis_scores.sample_size += 0.10;
            axis_scores.reproducibility += 0.10;
            axis_scores.generalizability += 0.08;
            axis_scores.consistency += 0.07;
        }

        if has_guideline && sources.len() >= 2 && !has_uncertainty {
            axis_scores.accuracy += 0.20;
            axis_scores.source_quality += 0.18;
            axis_scores.peer_review += 0.12;
            axis_scores.study_design += 0.12;
            axis_scores.outcome_measurement += 0.12;
            axis_scores.consistency += 0.10;
            axis_scores.relevance += 0.08;
        }

        if sources.is_empty() {
            axis_scores.source_quality -= 0.25;
            axis_scores.peer_review -= 0.20;
            axis_scores.reproducibility -= 0.10;
            axis_scores.sample_size -= 0.12;
            axis_scores.study_design -= 0.12;
            axis_scores.outcome_measurement -= 0.10;
        }

        if has_uncertainty {
            axis_scores.accuracy -= 0.20;
            axis_scores.consistency -= 0.10;
            axis_scores.clinical_significance -= 0.10;
            axis_scores.relevance -= 0.08;
            axis_scores.mechanistic_plausibility -= 0.08;
        }

        if sources.is_empty() && has_uncertainty {
            axis_scores.accuracy -= 0.12;
            axis_scores.source_quality -= 0.15;
            axis_scores.reproducibility -= 0.12;
        }
    }
}
