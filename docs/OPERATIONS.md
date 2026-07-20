# Operations

See `ARCHITECTURE.md` for the model/serving design. This covers the run/publish/incident
surface.

## Before any publish or push

```powershell
scripts/publish-preflight.ps1
```

Runs, in order: `gitleaks detect --source .` (if installed) then a pattern scan over
all git-tracked files (excluding the script itself) for: local machine paths
(`C:\Users\ogosh`, `/Users/s30519`), Tailscale IPs (`100.x.x.x`), API-key-shaped
strings (`sk-...`, `ghp_...`), and PEM/OpenSSH private-key headers. Any match should
stop the publish — this is a medical-domain project (KS verification layer), so the
bar for accidental leakage is higher than a typical repo.

## Incident Response

| Symptom | Likely cause | Action |
|---|---|---|
| `publish-preflight.ps1` throws "gitleaks failed" | A tracked file matches a known secret pattern | Do not publish. Remove/rotate the secret; if already pushed, treat as a secret-exposure incident (second opinion required before any history rewrite) |
| Pattern scan flags a local path or Tailscale IP | Fleet-specific path/IP leaked into a tracked file (config, log, comment) | Scrub before publishing — these identify the operator's private infrastructure |
| KS verification pipeline returns a low-confidence (`0.0-1.0`) or `D`-level evidence classification in production-facing output | Working as designed — the pipeline is meant to flag exactly this | Do not suppress or auto-upgrade the confidence score; surface it to the caller. A pattern of unexpectedly low scores across normal inputs is a model/data regression worth investigating separately |
| Contraindication check fails to trigger on a known-bad input during testing | Regression in the KS verification layer | Treat as a correctness bug in the safety-critical path — do not deploy until `benches/` and `tests/` both cover the missed case |

## Retention

No retention rule declared for `benches/` output or inference logs. Given the
medical-domain, safety-relevant nature of this project, recommend keeping evaluation
run history (not just pass/fail) longer than a typical project — flagged during the
2026-07 IT-system-management-guideline audit for an explicit decision.
