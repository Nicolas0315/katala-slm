# Katala SLM

![Rust](https://img.shields.io/badge/rust-1.78%2B-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Candle](https://img.shields.io/badge/ML-Candle-green)

Katala SLM is a Rust-first medical-domain small language model framework with a KS verification layer.

## Features
- Decoder-only transformer core (GQA attention + RoPE + SwiGLU + RMSNorm)
- Candle-based forward pass and inference loop
- KS verification pipeline with:
  - Evidence-level classifier (`A/B/C/D`)
  - Confidence scoring (`0.0-1.0`)
  - Source attribution
  - Contraindication checks
- Axum REST API with structured medical output
- CLI modes for local inference and API serving

## Architecture Overview
- `src/model`: model configuration and transformer components
- `src/inference`: generation engine, KV cache, sampling
- `src/ks`: evidence classification, confidence, attribution, verification
- `src/data`: tokenizer wrapper and dataset abstractions
- `src/serve`: HTTP server and endpoints

## Build
```bash
# CPU
cargo build --release

# CUDA
cargo build --release --features cuda
```

## Usage
```bash
# Inference CLI
cargo run --release -- infer --prompt "Influenza treatment options?"

# API server
cargo run --release -- serve --port 8080
```

### API Request
```bash
curl -X POST http://localhost:8080/v1/medical/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is first-line treatment for influenza?"}'
```

### API Response
```json
{
  "answer": "...",
  "evidence_level": "B",
  "sources": [
    {
      "source_id": "cdc-flu-antiviral",
      "title": "CDC Influenza Antiviral Guidance",
      "url": "https://www.cdc.gov/flu/professionals/antivirals/index.htm",
      "snippet": "Early antiviral treatment is recommended for high-risk patients."
    }
  ],
  "confidence": 0.72,
  "contraindications": []
}
```
