use candle_core::{Result, Tensor};
use candle_nn::{rms_norm, Module, RmsNorm, VarBuilder};

pub struct RMSNormLayer {
    norm: RmsNorm,
}

impl RMSNormLayer {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let norm = rms_norm(hidden_size, eps, vb.pp("rms_norm"))?;
        Ok(Self { norm })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.norm.forward(x)
    }
}
