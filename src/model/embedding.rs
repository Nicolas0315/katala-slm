use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

pub struct EmbeddingLayer {
    token_embedding: Embedding,
}

impl EmbeddingLayer {
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let token_embedding = embedding(vocab_size, hidden_size, vb.pp("token_embedding"))?;
        Ok(Self { token_embedding })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.token_embedding.forward(input_ids)
    }
}
