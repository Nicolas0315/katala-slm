use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QLoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub dropout: f32,
}

impl Default for QLoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            dropout: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoRAWeights {
    pub a: Tensor,
    pub b: Tensor,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
}

impl LoRAWeights {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        config: &QLoRAConfig,
        device: &Device,
    ) -> Result<Self> {
        let rank = config.rank.max(1);
        let a = Tensor::randn(0f32, 0.02f32, (input_dim, rank), device)?;
        let b = Tensor::zeros((rank, output_dim), DType::F32, device)?;

        Ok(Self {
            a,
            b,
            rank,
            alpha: config.alpha,
            dropout: config.dropout,
        })
    }

    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn lora_update(&self, input: &Tensor) -> Result<Tensor> {
        input.matmul(&self.a)?.matmul(&self.b)
    }
}
