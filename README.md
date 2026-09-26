# VertiMosaic

**CPU-Only Cross-Industry Vertical Federated Learning for Tabular Data**

VertiMosaic is an open-source research framework for training machine-learning models across organizations that hold complementary attributes about overlapping entities while keeping their raw feature tables local.

| Party | Private information represented |
|---|---|
| Bank | Financial/payment behaviour + target |
| Telecom | Communications/service behaviour |
| Insurance | Claims/risk behaviour |
| Retail | Purchase behaviour |

**Federation type:** Vertical Federated Learning  
**Raw feature tables pooled:** No  
**CPU supported:** Yes  
**GPU required:** No  
**Graph ML required:** No  
**External real-world datasets:** Yes  
**Four source datasets contain the same actual people:** No  
**Cross-industry linkage:** Explicitly semi-synthetic  
**Reference implementation:** NumPy/scikit-learn ecosystem

VertiMosaic distinguishes carefully between data locality, pseudonymization, synthetic linkage and cryptographic privacy guarantees.

## What VertiMosaic is

Vertical federated learning (VFL) assumes the same or overlapping entities are represented across parties, while each party owns different feature columns. VertiMosaic models that column-partitioned setting. Raw passive-party feature tables remain inside `PassiveParty` objects and are not concatenated during federated training.

The four primary public data sources used by the external benchmark do **not** describe the same real people. The cross-industry benchmark is therefore described as an **externally grounded semi-synthetic cross-industry VFL benchmark**. Linkage is target-blind in observed-target mode.

## What VertiMosaic is not

The default simulator is not cryptographically secure VFL. It provides raw-feature locality, local preprocessing, protocol separation and pseudonymous identifiers. It does not automatically provide PSI, MPC, homomorphic encryption, collusion resistance, malicious-party security, formal differential privacy, or protection against all gradient/routing leakage.

## Quick start

```bash
python -m pip install -e .
vertimosaic demo --rows 2000 --seed 42
```

Run checks:

```bash
ruff check .
ruff format --check .
mypy src/vertimosaic
pytest -q
python -m build
twine check dist/*
```

## Reference algorithms

1. **VFLLogisticRegression** — first-principles NumPy vertical logistic regression. Each party computes local logits and gradients over its own features.
2. **VFLHistGBDT** — CPU histogram-based vertical gradient boosting research implementation. Parties compute local candidate histogram statistics; the active party chooses a split, and the owning party returns routing only.

## External data

See [DATA_LICENSES.md](DATA_LICENSES.md). Source-code licensing does not relicense any dataset. IEEE-CIS is optional, local-only, and never redistributed.

## Scientific honesty

Generated metrics are written by experiment code. Documentation does not hard-code claims that VFL improves performance. Negative findings are retained. Communication numbers are simulated payload estimates, not measured network traffic.

## License

Source code: Apache-2.0. Dataset licenses remain with their providers.
