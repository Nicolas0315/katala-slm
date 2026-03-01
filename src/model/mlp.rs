use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, ops, Linear, Module, VarBuilder};

pub struct SwiGluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGluMlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }
}
