# External VFL comparators

VertiMosaic's internal centralized and single-party baselines are correctness/utility anchors, not substitutes for comparison with established vertical federated learning systems. v0.3 therefore defines a framework-neutral comparator contract for FATE, SecretFlow, or another VFL implementation.

## Fair-comparison contract

Every external comparison must use the same frozen benchmark manifest and must report at least:

- model/algorithm name;
- ROC-AUC;
- PR-AUC;
- training wall-clock seconds;
- inference wall-clock seconds.

Recommended additional fields are F1, Brier score, log loss, peak RSS, logical communication bytes/messages where the framework exposes them, framework version, Python version, CPU model, process/container topology, privacy mechanism configuration, and seed.

Use:

```bash
python scripts/run_external_comparator.py \
  --backend fate \
  --command "python /path/to/fate_runner.py" \
  --manifest benchmarks/exchange/manifest.json \
  --output benchmarks/comparators/fate.json
```

or:

```bash
python scripts/run_external_comparator.py \
  --backend secretflow \
  --command "python /path/to/secretflow_runner.py" \
  --manifest benchmarks/exchange/manifest.json \
  --output benchmarks/comparators/secretflow.json
```

The external runner receives `--manifest <path> --output <path>` and must emit JSON containing the required fields. The wrapper validates the record, hashes the manifest/result, records the execution environment, and preserves an explicit interpretation boundary.

## FATE

FATE supports heterogeneous/vertical federated learning and can be run in standalone or multi-node configurations. A FATE comparison should name the exact component/version used (for example the corresponding heterogeneous logistic/boosting implementation in the installed FATE release), deployment mode, cryptographic/protection settings, and any framework-specific preprocessing.

Do not label FATE communication/runtime measurements equivalent to VertiMosaic payload accounting unless the same measurement layer is actually being observed.

## SecretFlow

SecretFlow supports vertically partitioned ML through its device/algorithm layers. A SecretFlow comparison must report the exact SecretFlow version, device configuration, vertical algorithm, cryptographic device/protocol if used, and execution topology.

Do not describe a comparison as "privacy-equivalent" merely because both systems are vertical federated learning frameworks. Compare utility and resource measurements separately from security guarantees.

## Required publication table

A paper-ready comparison should contain rows for:

1. Bank-only non-federated baseline;
2. centralized all-feature non-federated baseline;
3. VertiMosaic VFL logistic;
4. VertiMosaic VFL histogram GBDT;
5. at least one FATE or SecretFlow vertical baseline.

The external row is considered complete only after the normalized JSON result exists in a released evidence bundle. Until then the manuscript must state that external-framework comparison is pending rather than inventing numbers.
