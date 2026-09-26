from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

from vertimosaic.training import run_demo


def main() -> None:
    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    records = []
    for rows in (10_000, 30_000, 50_000, 100_000):
        start = time.perf_counter()
        result = run_demo(rows=rows, seed=42, model="logistic")
        elapsed = time.perf_counter() - start
        records.append(
            {
                "rows": rows,
                "model": "logistic",
                "wall_seconds": elapsed,
                "roc_auc": result["metrics"]["roc_auc"],
                "pr_auc": result["metrics"]["pr_auc"],
                "estimated_communication_bytes": result["communication"]["estimated_bytes"],
            }
        )
    pd.DataFrame(records).to_csv(out / "results.csv", index=False)
    environment = {
        "python": sys.version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "seed": 42,
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
