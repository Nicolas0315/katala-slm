use std::collections::VecDeque;

use candle_core::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    Int8,
    Int4,
}

#[derive(Default)]
pub struct LayerKvCache {
    pub keys: VecDeque<Tensor>,
    pub values: VecDeque<Tensor>,
}

pub struct KvCache {
    pub layers: Vec<LayerKvCache>,
    pub quantization: Quantization,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, max_seq_len: usize, quantization: Quantization) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerKvCache::default()).collect(),
            quantization,
            max_seq_len,
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    pub fn push(&mut self, layer_idx: usize, key: Tensor, value: Tensor) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            if layer.keys.len() >= self.max_seq_len {
                layer.keys.pop_front();
                layer.values.pop_front();
            }
            layer.keys.push_back(key);
            layer.values.push_back(value);
        }
    }
}
