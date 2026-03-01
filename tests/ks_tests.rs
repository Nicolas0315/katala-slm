use katala_slm::ks::{evidence::EvidenceLevel, verify::Verifier};

#[test]
fn verifier_produces_structured_output() {
    let verifier = Verifier::default();
    let result = verifier.verify(
        "Patient on warfarin asks about pain control",
        "Avoid NSAID when possible and consult physician.",
        &["Clinical guideline".to_string()],
    );

    assert!(matches!(
        result.evidence_level,
        EvidenceLevel::A | EvidenceLevel::B | EvidenceLevel::C | EvidenceLevel::D
    ));
    assert!((0.0..=1.0).contains(&result.confidence));
}
