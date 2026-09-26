from pathlib import Path

from vertimosaic.training import run_demo

if __name__ == "__main__":
    print(run_demo(rows=5000, seed=42, model="logistic", output=Path("results/demo")))
