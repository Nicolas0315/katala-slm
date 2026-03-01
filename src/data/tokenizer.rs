use anyhow::{Context, Result};
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl TokenizerWrapper {
    pub fn from_file(path: &str, eos_token_id: u32) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            tokenizer,
            eos_token_id,
        })
    }

    pub fn from_builtin() -> Result<Self> {
        let mut tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
        ));
        Ok(Self {
            tokenizer,
            eos_token_id: 2,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("tokenizer encode failed")?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<u32> = ids.iter().copied().collect();
        self.tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("tokenizer decode failed")
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}
