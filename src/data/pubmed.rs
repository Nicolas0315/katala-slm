use serde::{Deserialize, Serialize};

use crate::{data::dataset::DatasetRecord, ks::evidence::EvidenceLevel};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PubMedArticle {
    pub title: String,
    pub abstract_text: String,
    pub authors: Vec<String>,
    pub journal: String,
    pub pmid: String,
    pub mesh_terms: Vec<String>,
    pub publication_type: Vec<String>,
}

pub fn parse_pubmed_xml(xml: &str) -> Vec<PubMedArticle> {
    extract_blocks(xml, "PubmedArticle")
        .into_iter()
        .map(|block| parse_pubmed_article(&block))
        .collect()
}

pub fn publication_type_to_evidence_level(publication_types: &[String]) -> EvidenceLevel {
    let lower: Vec<String> = publication_types.iter().map(|v| v.to_lowercase()).collect();
    if lower.iter().any(|p| {
        p.contains("meta-analysis") || p.contains("systematic review") || p.contains("guideline")
    }) {
        EvidenceLevel::A
    } else if lower.iter().any(|p| {
        p.contains("randomized controlled trial")
            || p.contains("clinical trial")
            || p.contains("cohort")
    }) {
        EvidenceLevel::B
    } else if lower
        .iter()
        .any(|p| p.contains("case") || p.contains("observational"))
    {
        EvidenceLevel::C
    } else {
        EvidenceLevel::D
    }
}

impl PubMedArticle {
    pub fn to_dataset_record(&self) -> DatasetRecord {
        let evidence_level = publication_type_to_evidence_level(&self.publication_type);
        DatasetRecord {
            id: format!("pubmed:{}", self.pmid),
            prompt: self.title.clone(),
            answer: self.abstract_text.clone(),
            evidence_level,
            sources: vec![
                format!("Journal: {}", self.journal),
                format!("PMID: {}", self.pmid),
            ],
        }
    }
}

fn parse_pubmed_article(block: &str) -> PubMedArticle {
    let title = extract_first_tag_text(block, "ArticleTitle").unwrap_or_default();
    let abstract_text = extract_all_tag_text(block, "AbstractText").join(" ");
    let journal = extract_first_tag_text(block, "Title").unwrap_or_default();
    let pmid = extract_first_tag_text(block, "PMID").unwrap_or_default();

    let authors = extract_blocks(block, "Author")
        .into_iter()
        .filter_map(|author| {
            let collective = extract_first_tag_text(&author, "CollectiveName").unwrap_or_default();
            if !collective.is_empty() {
                return Some(collective);
            }
            let first = extract_first_tag_text(&author, "ForeName").unwrap_or_default();
            let last = extract_first_tag_text(&author, "LastName").unwrap_or_default();
            let combined = format!("{} {}", first, last).trim().to_string();
            if combined.is_empty() {
                None
            } else {
                Some(combined)
            }
        })
        .collect();

    let mesh_terms = extract_all_tag_text(block, "DescriptorName");
    let publication_type = extract_all_tag_text(block, "PublicationType");

    PubMedArticle {
        title,
        abstract_text,
        authors,
        journal,
        pmid,
        mesh_terms,
        publication_type,
    }
}

fn extract_blocks(input: &str, tag: &str) -> Vec<String> {
    let start_tag = format!("<{tag}");
    let end_tag = format!("</{tag}>");

    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(start_rel) = input[cursor..].find(&start_tag) {
        let start_idx = cursor + start_rel;
        let Some(open_end_rel) = input[start_idx..].find('>') else {
            break;
        };
        let content_start = start_idx + open_end_rel + 1;

        let Some(end_rel) = input[content_start..].find(&end_tag) else {
            break;
        };
        let end_idx = content_start + end_rel;
        out.push(clean_text(&input[content_start..end_idx]));
        cursor = end_idx + end_tag.len();
    }
    out
}

fn extract_first_tag_text(input: &str, tag: &str) -> Option<String> {
    extract_all_tag_text(input, tag).into_iter().next()
}

fn extract_all_tag_text(input: &str, tag: &str) -> Vec<String> {
    extract_blocks(input, tag)
        .into_iter()
        .map(|s| decode_xml_entities(&s))
        .filter(|s| !s.is_empty())
        .collect()
}

fn clean_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn decode_xml_entities(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}
