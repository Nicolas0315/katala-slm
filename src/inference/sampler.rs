use candle_core::{Result, Tensor, D};

#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temperature: f64,
    pub top_p: f64,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64) -> Self {
        Self {
            temperature: temperature.max(1e-5),
            top_p: top_p.clamp(0.01, 1.0),
        }
    }

    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_vec1::<f32>()?;
        let scaled: Vec<f32> = logits
            .iter()
            .map(|v| (*v as f64 / self.temperature) as f32)
            .collect();

        let max = scaled
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let exp: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
        let sum = exp.iter().sum::<f32>().max(1e-8);
        let mut probs: Vec<(usize, f32)> =
            exp.iter().enumerate().map(|(i, p)| (i, p / sum)).collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0f32;
        let mut cutoff = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cumulative += *p;
            if cumulative >= self.top_p as f32 {
                cutoff = idx + 1;
                break;
            }
        }
        let chosen = probs[..cutoff]
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| *i)
            .unwrap_or(0);

        Ok(chosen as u32)
    }

    pub fn last_token_logits(logits: &Tensor) -> Result<Tensor> {
        let (_, seq, vocab) = logits.dims3()?;
        logits.narrow(D::Minus2, seq - 1, 1)?.reshape((vocab,))
    }
}
