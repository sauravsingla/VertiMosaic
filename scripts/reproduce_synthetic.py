from pathlib import Path

from vertimosaic.training import run_demo

for model in ("logistic", "vfl-hist-gbdt"):
    run_demo(rows=10_000, seed=42, model=model, output=Path("results") / f"synthetic_{model}")
