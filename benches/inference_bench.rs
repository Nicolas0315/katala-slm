use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Tensor};
use katala_slm::{
    ks::verify::Verifier,
    model::{config::ModelConfig, transformer::TransformerModel},
};

fn main() -> Result<()> {
    let config = ModelConfig {
        vocab_size: 512,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 512,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };

    let model = TransformerModel::new(config.clone(), true)?;
    let verifier = Verifier::default();

    println!("{:<34} {:>12}", "Benchmark", "Time (ms)");
    println!("{}", "-".repeat(50));

    for seq_len in [32usize, 128, 512] {
        let input_tokens: Vec<u32> = (0..seq_len as u32)
            .map(|v| v % config.vocab_size as u32)
            .collect();
        let input =
            Tensor::from_vec(input_tokens, (1, seq_len), &model.device)?.to_dtype(DType::U32)?;

        let start = Instant::now();
        for _ in 0..5 {
            let _ = model.forward(&input)?;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "{:<34} {:>12.2}",
            format!("Forward pass (seq={seq_len})"),
            elapsed
        );
    }

    let ks_prompt = "Patient on warfarin asks about pain control";
    let ks_answer = "Randomized controlled trial and guideline evidence suggest avoiding NSAID.";
    let ks_sources = vec!["WHO guideline".to_string(), "Meta-analysis".to_string()];

    let start = Instant::now();
    for _ in 0..1_000 {
        let _ = verifier.verify(ks_prompt, ks_answer, &ks_sources);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
    println!("{:<34} {:>12.2}", "KS verification x1000", elapsed);

    for gen_tokens in [10usize, 50, 100] {
        let mut token_ids: Vec<u32> = vec![1, 2, 3, 4];
        let start = Instant::now();
        for _ in 0..gen_tokens {
            let input = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &model.device)?
                .to_dtype(DType::U32)?;
            let _logits = model.forward(&input)?;
            let next = (token_ids.last().copied().unwrap_or(0) + 1) % config.vocab_size as u32;
            token_ids.push(next);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "{:<34} {:>12.2}",
            format!("Token loop ({gen_tokens} tokens)"),
            elapsed
        );
    }

    Ok(())
}
