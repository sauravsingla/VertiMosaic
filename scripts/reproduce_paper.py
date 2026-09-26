from pathlib import Path

from vertimosaic.training import run_demo

out = Path("results/paper_smoke")
run_demo(rows=10_000, seed=42, model="logistic", output=out / "logistic")
run_demo(rows=10_000, seed=42, model="vfl-hist-gbdt", output=out / "vfl_hist_gbdt")
