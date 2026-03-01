use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use clap::{Parser, Subcommand};
use katala_slm::{
    inference::engine::{GenerateOptions, InferenceEngine},
    model::config::ModelConfig,
    serve::api::run_server,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(author, version, about = "Katala SLM CLI")]
struct Cli {
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Infer {
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
    },
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = ModelConfig::default();
    let engine = Arc::new(InferenceEngine::new(config, cli.cpu)?);

    match cli.command {
        Commands::Infer {
            model: _,
            prompt,
            max_new_tokens,
            temperature,
            top_p,
        } => {
            let options = GenerateOptions {
                max_new_tokens,
                temperature,
                top_p,
            };
            let answer = engine.generate_verified(&prompt, options)?;
            println!("{}", serde_json::to_string_pretty(&answer)?);
        }
        Commands::Serve { port } => {
            run_server(([0, 0, 0, 0], port).into(), engine).await?;
        }
    }

    Ok(())
}
