# VertiMosaic experiment manifest

| RQ | Command | Dataset/mode | Models | Primary outputs |
|---|---|---|---|---|
| RQ1 | `vertimosaic external-demo` | observed/distributed external | logistic, VFLHistGBDT | `reports/external_demo.json` |
| RQ2 | `pytest tests/test_logistic_centralized_equivalence.py` | controlled synthetic | VFL logistic vs equivalent centralized objective | test output |
| RQ3 | `vertimosaic train --model vfl-hist-gbdt` | synthetic_scale | logistic, VFLHistGBDT | `runs/<run_id>/metrics.json` |
| RQ4 | `vertimosaic contribution` | synthetic_scale | logistic | `results/party_contribution.csv` |
| RQ5 | `vertimosaic overlap` | synthetic_scale | configurable | `results/partial_overlap.csv` |
| RQ6 | `vertimosaic dropout` | synthetic_scale | logistic | `results/party_dropout.csv` |
| RQ7 | `vertimosaic drift` | synthetic_scale | logistic | `results/feature_drift.csv` |
| RQ8 | `vertimosaic benchmark` | synthetic_scale | configurable | `benchmarks/results.csv`, `benchmarks/environment.json` |
| RQ9 | mode-specific commands | all four modes | applicable | run/report artifacts |

Tables and figures must be generated from result files; no metric should be manually typed into paper artifacts.
