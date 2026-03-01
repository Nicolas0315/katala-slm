use std::sync::{Arc, Mutex};

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::{
    data::tokenizer::TokenizerWrapper,
    ks::verify::{VerifiedAnswer, Verifier},
    model::{config::ModelConfig, transformer::TransformerModel},
};

use super::{
    kv_cache::{KvCache, Quantization},
    sampler::Sampler,
};

#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.8,
            top_p: 0.95,
        }
    }
}

pub struct InferenceEngine {
    pub model: Arc<TransformerModel>,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub verifier: Arc<Verifier>,
    kv_cache: Mutex<KvCache>,
}

impl InferenceEngine {
    pub fn new(config: ModelConfig, force_cpu: bool) -> Result<Self> {
        let model = Arc::new(TransformerModel::new(config.clone(), force_cpu)?);
        let tokenizer = Arc::new(TokenizerWrapper::from_builtin()?);
        let verifier = Arc::new(Verifier::default());
        let kv_cache = Mutex::new(KvCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            Quantization::Int8,
        ));

        Ok(Self {
            model,
            tokenizer,
            verifier,
            kv_cache,
        })
    }

    pub fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<String> {
        let mut token_ids = self.tokenizer.encode(prompt)?;
        let sampler = Sampler::new(options.temperature, options.top_p);

        for _ in 0..options.max_new_tokens {
            let input =
                Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &self.model.device)?
                    .to_dtype(DType::U32)?;
            let logits = self.model.forward(&input)?;
            let last = Sampler::last_token_logits(&logits)?;
            let next = sampler.sample(&last)?;
            token_ids.push(next);
            if next == self.tokenizer.eos_token_id() {
                break;
            }
        }

        self.tokenizer.decode(&token_ids)
    }

    pub fn generate_verified(
        &self,
        prompt: &str,
        options: GenerateOptions,
    ) -> Result<VerifiedAnswer> {
        let mut cache = self.kv_cache.lock().expect("kv cache mutex poisoned");
        cache.clear();
        drop(cache);

        let answer = self.generate(prompt, options)?;
        Ok(self.verifier.verify(prompt, &answer, &[]))
    }
}
