use katala_slm::data::{
    jmed::{extract_evidence_level_from_text, parse_jmed_json},
    pubmed::parse_pubmed_xml,
};
use katala_slm::ks::evidence::EvidenceLevel;

#[test]
fn parse_pubmed_xml_extracts_expected_fields() {
    let xml = r#"
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Clinical efficacy of treatment X</ArticleTitle>
        <Abstract>
          <AbstractText>Randomized controlled trial showed improved outcomes.</AbstractText>
        </Abstract>
        <Journal><Title>New England Journal of Medicine</Title></Journal>
        <AuthorList>
          <Author><ForeName>Jane</ForeName><LastName>Doe</LastName></Author>
          <Author><CollectiveName>Trial Group</CollectiveName></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Hypertension</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"#;

    let parsed = parse_pubmed_xml(xml);
    assert_eq!(parsed.len(), 1);
    let article = &parsed[0];
    assert_eq!(article.pmid, "12345");
    assert!(article.title.contains("treatment X"));
    assert!(article.authors.iter().any(|a| a.contains("Jane Doe")));

    let record = article.to_dataset_record();
    assert_eq!(record.evidence_level, EvidenceLevel::B);
}

#[test]
fn parse_jmed_json_and_extract_evidence() {
    let json = r#"[
  {
    "id": "g1",
    "title": "高血圧治療ガイドライン",
    "organization": "日本高血圧学会",
    "section": "薬物療法",
    "recommendation_text": "推奨A: ACE阻害薬を第一選択とする。",
    "evidence_level": null
  }
]"#;

    let guidelines = parse_jmed_json(json).expect("valid guideline JSON");
    assert_eq!(guidelines.len(), 1);
    assert_eq!(guidelines[0].inferred_evidence_level(), EvidenceLevel::A);
    assert_eq!(
        extract_evidence_level_from_text("grade c"),
        EvidenceLevel::C
    );

    let record = guidelines[0].to_dataset_record();
    assert_eq!(record.evidence_level, EvidenceLevel::A);
}
