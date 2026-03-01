use katala_slm::ks::{
    confidence::ConfidenceScorer,
    evidence::{EvidenceClassifier, EvidenceLevel},
    verify::Verifier,
};

#[test]
fn evidence_classifier_ranks_correctly() {
    let classifier = EvidenceClassifier;

    // Multiple guideline sources → A
    let level = classifier.classify(
        "Treatment is well-established.",
        &["WHO guideline".to_string(), "CDC guidance".to_string()],
    );
    assert_eq!(level, EvidenceLevel::A);

    // Single guideline → B
    let level = classifier.classify("Standard treatment.", &["NEJM review".to_string()]);
    assert_eq!(level, EvidenceLevel::B);

    // No guidelines but has source → C
    let level = classifier.classify("Limited data suggests...", &["PubMed case report".to_string()]);
    assert_eq!(level, EvidenceLevel::C);

    // No sources, uncertain → D
    let level = classifier.classify("The mechanism is unclear.", &[]);
    assert_eq!(level, EvidenceLevel::D);
}

#[test]
fn confidence_decreases_with_contraindications() {
    let scorer = ConfidenceScorer;
    let high = scorer.score(EvidenceLevel::A, &[], "Safe treatment.");
    let low = scorer.score(
        EvidenceLevel::A,
        &["renal failure".to_string(), "allergy".to_string()],
        "Safe treatment.",
    );
    assert!(high.score > low.score);
}

#[test]
fn verifier_detects_warfarin_nsaid_interaction() {
    let verifier = Verifier::default();
    let result = verifier.verify(
        "Patient on warfarin with joint pain",
        "Consider NSAID for pain relief.",
        &[],
    );
    assert!(!result.contraindications.is_empty());
    assert!(result.contraindications[0].contains("bleeding"));
}
