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

VertiMosaic distinguishes carefully between data locality, pseudonymization, synthetic linkage, and cryptographic privacy guarantees.

## True vertical federated learning

Horizontal FL generally trains across different entities that share a similar feature schema. Vertical FL aligns the same or overlapping entities while parties own different feature columns. VertiMosaic implements the latter. During federated training, passive-party raw feature matrices remain inside their party objects and are not pooled into the Bank.

The four primary public sources do **not** describe the same real people. The four-industry benchmark is therefore an **externally grounded semi-synthetic cross-industry VFL benchmark**. In observed-target mode, linkage is target-blind. In distributed-signal mode, the target is generated only after linked profiles are formed.

## Privacy boundary

The default simulator provides raw-feature locality, party-local computation/preprocessing, explicit protocol boundaries, metadata-only communication auditing, and research pseudonymization. It is **not cryptographically secure VFL** and does not automatically provide PSI, MPC, homomorphic encryption, secure aggregation, collusion resistance, malicious-party security, or formal differential privacy. Gradients, Hessians, local logits, residuals, entity membership, routing information, and derived split statistics may leak information. See `docs/threat_model.md` and `docs/privacy_boundaries.md`.

## Quick start

```bash
python -m pip install -e .
vertimosaic --help
vertimosaic demo --rows 2000 --seed 42
```

Reference commands:

```bash
vertimosaic datasets list
vertimosaic datasets describe bank
vertimosaic datasets verify
vertimosaic datasets download bank
vertimosaic datasets download-all
vertimosaic prepare-external
vertimosaic generate-synthetic
vertimosaic train --model logistic
vertimosaic train --model vfl-hist-gbdt
vertimosaic evaluate
vertimosaic ablation
vertimosaic contribution
vertimosaic overlap
vertimosaic dropout
vertimosaic drift
vertimosaic benchmark
vertimosaic report
vertimosaic external-demo --model vfl-hist-gbdt --seed 42
```

Optional IEEE-CIS files are authorized local inputs only:

```bash
vertimosaic prepare-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv
```

## Reference algorithms

`VFLLogisticRegression` is the first-principles NumPy reference protocol. Each party computes local logits and local gradients over only its own features. `VFLHistGBDT` is a CPU vertical histogram-gradient-boosting research implementation in which parties compute local candidate statistics and the owning party performs routing.

Centralized models exist only as **NON-FEDERATED BASELINES** for research comparison and are never relabelled as VFL.

## Benchmark modes

- `observed_target_external`: Bank is the anchor population and uses the published observed default target; external profiles are linked target-blind.
- `distributed_signal_external`: real transformed source-domain features are linked first, then a clearly disclosed semi-synthetic target depends on all four parties.
- `synthetic_scale`: fully controlled scaling, overlap, dropout, drift, noise, and distributed-signal experiments.
- `ieee_cis_linked`: optional genuinely linked local benchmark preparation when authorized IEEE-CIS transaction and identity files are supplied; restricted source files are never downloaded or redistributed by VertiMosaic.

## External data and provenance

The primary sources are UCI Default of Credit Card Clients (350), UCI Iranian Churn (563), OpenML `freMTPL2freq` (41214) / `freMTPL2sev` (41215), and UCI Online Retail (352). Retail is aggregated to customer level before VFL and uses temporal cutoffs to prevent future-data leakage. Source-code licensing does not relicense datasets; see `DATA_LICENSES.md`.

`vertimosaic datasets verify` checks the static UCI attribution fields and queries OpenML's official JSON metadata API for the two insurance licenses. Verification fails conservatively when required provider license metadata cannot be obtained. Downloaded artifact metadata retains the complete per-source provenance schema, and unknown raw-provider values remain explicit `null` values rather than being fabricated.

Full experiment runs can record configuration, dataset/linkage provenance, environment, predictions, training history, communication metadata, and feature provenance under `runs/<run_id>/`.

## Evaluation and scientific honesty

Entity-level splits default to 70% train, 15% validation, and 15% test. Threshold selection uses validation data only. Test evaluation supports ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, ECE, confusion counts, deterministic bootstrap intervals, and paired bootstrap differences.

No benchmark conclusion is hard-coded. Negative and uncertain findings are retained. Communication quantities are **simulated payload estimates**, not measured network traffic or latency. Result tables and reports should be generated from measured artifacts rather than manually typed metric values.

## Verification

```bash
ruff check .
ruff format --check .
mypy src/vertimosaic
pytest -q --cov=vertimosaic --cov-report=term-missing
bandit -r src
pip-audit
vertimosaic demo --rows 2000 --seed 42
python scripts/reproduce_paper.py --smoke
python -m build
twine check dist/*
```

Protocol-critical modules target at least 90% coverage; coverage is not inflated merely to chase 100%.

## Documentation

See `docs/architecture.md`, `docs/protocol.md`, `docs/vertical_vs_horizontal_fl.md`, `docs/datasets.md`, `docs/linkage.md`, `docs/privacy_boundaries.md`, `docs/threat_model.md`, `docs/reproducibility.md`, `docs/limitations.md`, and `paper/experiment_manifest.md`.

## License

VertiMosaic source code is licensed under Apache-2.0. Dataset licenses and provider terms remain separate.
