use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::{
    inference::engine::{GenerateOptions, InferenceEngine},
    ks::verify::VerifiedAnswer,
};

#[derive(Clone)]
struct ApiState {
    engine: Arc<InferenceEngine>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

pub async fn run_server(addr: SocketAddr, engine: Arc<InferenceEngine>) -> Result<()> {
    let state = ApiState { engine };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/medical/generate", post(generate_medical))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(HealthResponse { status: "ok" })
}

async fn generate_medical(
    State(state): State<ApiState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let options = GenerateOptions {
        max_new_tokens: req.max_new_tokens.unwrap_or(128),
        temperature: req.temperature.unwrap_or(0.8),
        top_p: req.top_p.unwrap_or(0.95),
    };

    let response: VerifiedAnswer = state
        .engine
        .generate_verified(&req.prompt, options)
        .unwrap_or_else(|err| VerifiedAnswer {
            answer: format!("Generation failed: {err}"),
            evidence_level: crate::ks::evidence::EvidenceLevel::D,
            sources: vec![],
            confidence: 0.0,
            contraindications: vec!["internal_error".to_string()],
        });

    Json(response)
}
