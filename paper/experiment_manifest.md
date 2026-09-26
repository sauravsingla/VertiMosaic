# Experiment manifest

| RQ | Benchmark | Command | Primary outputs |
|---|---|---|---|
| RQ1/RQ3 | synthetic-scale | `vertimosaic demo --model logistic` / `--model vfl-hist-gbdt` | `results/demo/metrics.json` |
| RQ2 | controlled linear | `pytest tests/test_logistic_centralized_equivalence.py` | test result |
| RQ8 | synthetic-scale | `python scripts/run_benchmarks.py` | `benchmarks/results.csv` |
| External provenance | public sources | `vertimosaic datasets list` and explicit download commands | `data/provenance/*.json` |

Additional overlap, dropout, drift, party-utility, and external benchmark commands are planned as the experiment suite expands. Publication claims must be based only on generated outputs.
