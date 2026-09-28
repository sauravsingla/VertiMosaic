# Tutorial 1 — First VFL run in 10 minutes

This tutorial uses the built-in CPU-only synthetic benchmark. The synthetic dataset binds explicit ordered entity identifiers, and the official logistic experiment path enables strict alignment checks.

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run the smoke demo

```bash
vertimosaic demo --rows 2000 --seed 42
```

For a paper-style run with reproducibility artifacts:

```bash
python - <<'PY'
from vertimosaic.experiments.pipeline import run_synthetic_experiment

result = run_synthetic_experiment(
    rows=2000,
    seed=42,
    model_name="logistic",
    bootstrap_replicates=200,
    write_run=True,
)
print(result["metrics"])
print(result["run_directory"])
PY
```

## What happened

- the active Bank party owns labels and one feature block;
- Telecom, Insurance and Retail own other feature blocks;
- raw passive feature matrices stay inside their party objects;
- party-local logits and residual-related messages cross the explicit transport boundary;
- entity order is checked using bound identifiers rather than row count alone;
- validation predictions select the operating threshold;
- the test partition produces held-out metrics;
- the run directory records configuration, provenance, predictions, metrics, communication metadata, environment information and hashes.

## Verify the evidence

Inspect the generated `runs/<run_id>/` directory. Do not copy a metric into a report without retaining the measured run artifact that produced it.

## Important privacy boundary

This run demonstrates raw-feature locality. It is not end-to-end cryptographically secure VFL. Residuals, logits and other derived messages can leak information; see `docs/threat_model.md` and `docs/limitations.md`.
