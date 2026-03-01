use anyhow::Result;
use katala_slm::model::{config::ModelConfig, transformer::TransformerModel};
use candle_core::{DType, Tensor};

#[test]
fn model_generates_logits() -> Result<()> {
    let config = ModelConfig {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };
    let model = TransformerModel::new(config.clone(), true)?;
    let input = Tensor::from_vec(vec![1u32, 5, 10, 20], (1, 4), &model.device)?
        .to_dtype(DType::U32)?;
    let logits = model.forward(&input)?;
    let (b, s, v) = logits.dims3()?;
    assert_eq!(b, 1);
    assert_eq!(s, 4);
    assert_eq!(v, config.vocab_size);
    Ok(())
}
