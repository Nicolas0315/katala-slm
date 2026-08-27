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

#[test]
fn verified_answer_never_claims_clinical_use() {
    // The crate labels output; it does not gate it. verify() returns the answer
    // unchanged even when a contraindication fires, and the contraindication set is
    // three keyword rules rather than a drug-interaction database. A consumer reading
    // only the struct has to be told that, so the flag travels with every response.
    let verifier = katala_slm::ks::verify::Verifier::default();

    let flagged = verifier.verify(
        "pregnant patient with severe acne",
        "Consider isotretinoin for severe acne",
        &[],
    );
    assert!(
        !flagged.contraindications.is_empty(),
        "expected the pregnancy rule to fire"
    );
    // The answer is returned untouched. This is the property the README now states,
    // pinned here so a later change cannot quietly turn labelling into gating — or
    // claim gating that does not exist.
    assert_eq!(flagged.answer, "Consider isotretinoin for severe acne");
    assert!(!flagged.clinical_use);

    // Phrasing that avoids the literal keywords produces an empty list. That means
    // "not checked", and the flag must still say this is not for clinical use.
    let unchecked = verifier.verify(
        "patient on blood thinners with joint pain",
        "an anti-inflammatory pain reliever",
        &[],
    );
    assert!(unchecked.contraindications.is_empty());
    assert!(!unchecked.clinical_use);
}
