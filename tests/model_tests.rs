use anyhow::Result;
use candle_core::{DType, Tensor};
use katala_slm::model::{config::ModelConfig, transformer::TransformerModel};

#[test]
fn transformer_forward_has_expected_shape() -> Result<()> {
    let config = ModelConfig {
        vocab_size: 128,
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 128,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };
    let model = TransformerModel::new(config.clone(), true)?;

    let input =
        Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &model.device)?.to_dtype(DType::U32)?;
    let logits = model.forward(&input)?;
    assert_eq!(logits.dims3()?, (1, 4, config.vocab_size));
    Ok(())
}
