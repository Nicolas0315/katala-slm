use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ks::evidence::EvidenceLevel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub id: String,
    pub prompt: String,
    pub answer: String,
    pub evidence_level: EvidenceLevel,
    pub sources: Vec<String>,
}

pub trait Dataset {
    fn len(&self) -> usize;
    fn iter(&self) -> Box<dyn Iterator<Item = &DatasetRecord> + '_>;
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryDataset {
    pub records: Vec<DatasetRecord>,
}

impl InMemoryDataset {
    pub fn map_parallel<F>(&self, f: F) -> Vec<DatasetRecord>
    where
        F: Fn(&DatasetRecord) -> DatasetRecord + Sync + Send,
    {
        self.records.par_iter().map(f).collect()
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &DatasetRecord> + '_> {
        Box::new(self.records.iter())
    }
}
