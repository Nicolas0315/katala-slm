use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 512,
            intermediate_size: 1_536,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            max_position_embeddings: 2_048,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }
}
