# Formal benchmark matrix

`vertimosaic-benchmark-matrix` produces a machine-readable and paper-ready comparison of:

- centralized all-feature **NON-FEDERATED** logistic regression;
- single-party Bank-only **NON-FEDERATED** logistic regression;
- `VFLLogisticRegression`; and
- `VFLHistGBDT`.

The core matrix reports ROC-AUC, PR-AUC, F1, Brier score, training/inference wall-clock time, process RSS, estimated protocol payload bytes/message count, and a reproducible confidence-membership attack baseline.

With robustness enabled, the VFL rows also report 50% entity-overlap behavior. Logistic VFL reports documented inference-time party-dropout behavior. Histogram GBDT does not silently omit a party after training; the table explicitly reports that such a condition requires a separately retrained/evaluated availability configuration.

## Interpretation boundaries

Communication values are VertiMosaic protocol-payload accounting, not packet captures and not a network-stack benchmark. RSS is process-level observed resident memory, not a profiler-derived allocation peak. The membership attack is a deliberately simple empirical baseline, not a privacy proof. Centralized rows deliberately pool selected feature columns and must never be described as federated.
