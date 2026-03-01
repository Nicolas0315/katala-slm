use katala_slm::ks::{
    evidence::{AxisScores, EvidenceClassifier, EvidenceLevel},
    verify::Verifier,
};

#[test]
fn composite_mapping_respects_thresholds() {
    let mut low = AxisScores::new(0.2);
    assert_eq!(low.to_evidence_level(), EvidenceLevel::D);

    low.accuracy = 0.8;
    low.source_quality = 0.8;
    low.relevance = 0.8;
    low.study_design = 0.7;
    low.sample_size = 0.7;
    low.outcome_measurement = 0.7;
    assert!(matches!(
        low.to_evidence_level(),
        EvidenceLevel::C | EvidenceLevel::B
    ));
}

#[test]
fn keyword_heuristic_boosts_study_design_and_sample_size() {
    let classifier = EvidenceClassifier;
    let assessment = classifier.classify_with_context(
        "",
        "Randomized double-blind controlled trial with meta-analysis across studies.",
        &["peer-reviewed journal".to_string()],
    );

    assert!(assessment.axis_scores.study_design > 0.6);
    assert!(assessment.axis_scores.sample_size >= 0.6);
}

#[test]
fn verifier_returns_axis_breakdown() {
    let verifier = Verifier::default();
    let result = verifier.verify(
        "Patient on warfarin asks about pain control",
        "Guideline-based recommendation avoids NSAID and references randomized trial data.",
        &["WHO guideline".to_string(), "meta-analysis".to_string()],
    );

    assert!((0.0..=1.0).contains(&result.axis_scores.accuracy));
    assert!((0.0..=1.0).contains(&result.composite_score));
}
