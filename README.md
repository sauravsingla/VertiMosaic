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

The quick-start demo is a smoke experiment and uses 100 bootstrap replicates. Normal research experiment and benchmark commands default to 1,000 bootstrap replicates. Experiment commands create a standard reproducibility bundle under `runs/<run_id>/`.

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
vertimosaic benchmark --mode synthetic_scale --sizes 10000,30000,50000,100000,250000
vertimosaic benchmark --mode observed_target_external --model vfl-hist-gbdt
vertimosaic report
vertimosaic external-demo --model vfl-hist-gbdt --seed 42
```

Optional IEEE-CIS files are authorized local inputs only. VertiMosaic never downloads or redistributes these competition files and ordinary CI never depends on them. Preparation is available separately:

```bash
vertimosaic prepare-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv
```

To run the genuinely linked two-party benchmark on authorized local files:

```bash
vertimosaic run-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv \
  --model logistic \
  --seed 42
```

The IEEE-CIS mode uses the exact `TransactionID` intersection, keeps `isFraud` with the active transaction party, fits preprocessing independently on each party's training rows, evaluates on an entity-level held-out test set, pseudonymizes exported test identifiers, records source-file SHA-256 checksums, and writes the same reproducibility bundle as other experiments. It is a two-party linked sanity benchmark and must not be described as a four-industry benchmark.

## Reference algorithms

`VFLLogisticRegression` is the first-principles NumPy reference protocol. Each party computes local logits and local gradients over only its own features. `VFLHistGBDT` is a CPU vertical histogram-gradient-boosting research implementation in which every party fits and retains its own quantile-bin representation, computes aggregate gradient/Hessian/count candidate statistics from those local bins, and applies the chosen split locally through an opaque feature/bin reference. Numeric split thresholds and raw feature matrices are not carried in coordinator-facing candidate records.

`LocalTabularPreprocessor` supports party-local median imputation, robust/standard scaling, categorical imputation, one-hot encoding with unknown-category handling, configurable winsorization, and a train-only local quantile-binning helper for tree-model preprocessing.

Centralized models exist only as **NON-FEDERATED BASELINES** for research comparison and are never relabelled as VFL. The centralized baseline API explicitly supports every Bank-anchored party subset as well as Bank-only and all-party comparisons for logistic and histogram-gradient-boosting models.

## Benchmark modes

- `observed_target_external`: Bank is the anchor population and uses the published observed default target; external profiles are linked target-blind.
- `distributed_signal_external`: real transformed source-domain features are linked first, then a clearly disclosed semi-synthetic target depends on all four parties.
- `synthetic_scale`: fully controlled scaling, overlap, dropout, drift, noise, and distributed-signal experiments.
- `ieee_cis_linked`: optional genuinely linked two-party VFL benchmark using authorized local IEEE-CIS transaction and identity files joined by `TransactionID`; restricted source files are never downloaded, committed, or redistributed by VertiMosaic.

Synthetic CPU scaling uses the full 10K/30K/50K/100K/250K grid. External CPU benchmarking runs once at the **actual available Bank anchor size** and writes the same timing, memory, communication and metric fields into `benchmarks/results.csv`; it does not fabricate a synthetic external row count.

## Partial entity overlap

`vertimosaic overlap` evaluates 100%, 90%, 75%, 50% and 25% controlled Bank-to-passive overlap. The same deterministic availability realization and the same global train/validation/test entity split are used to compare `intersection_only` against missing-party-aware `availability_indicator` training. The output records both retained coverage and true common-intersection coverage, and selects F1 thresholds using validation predictions only.

## External data and provenance

The primary sources are UCI Default of Credit Card Clients (350), UCI Iranian Churn (563), OpenML `freMTPL2freq` (41214) / `freMTPL2sev` (41215), and UCI Online Retail (352). Retail is aggregated to customer level using an explicit source-time feature cutoff **before** cross-domain linkage, so post-cutoff transactions are excluded from the benchmark snapshot. Source-code licensing does not relicense datasets; see `DATA_LICENSES.md`.

`vertimosaic datasets verify` checks the static UCI attribution fields and queries OpenML's official JSON metadata API for the two insurance licenses. Verification fails conservatively when required provider license metadata cannot be obtained. Each retrieved source capture records a SHA-256 content checksum before transformation, sampling, or temporal cutoff as applicable, plus raw/processed row counts and per-source license metadata. Processed feature artifacts have a separately scoped SHA-256 checksum; source-capture hashes are not falsely presented as provider-published file checksums.

Every experiment run bundle records `config.yaml`, dataset/linkage provenance, environment and dependency versions, measured metrics, predictions, training history, communication metadata, feature provenance, artifact hashes, configuration hash, timestamps, seed, and Git SHA under `runs/<run_id>/`.

## Evaluation and scientific honesty

Entity-level splits default to 70% train, 15% validation, and 15% test. Threshold selection uses validation data only. Test evaluation supports ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, ECE, confusion counts, deterministic bootstrap intervals, and paired bootstrap differences.

Normal research runs use 1,000 deterministic bootstrap replicates where practical; smoke CI uses 100. Important ROC-AUC baseline comparisons also calculate the PR-AUC delta and 95% paired interval on the same paired bootstrap resamples. No benchmark conclusion is hard-coded. Negative and uncertain findings are retained. Communication quantities are **simulated payload estimates**, not measured network traffic or latency. Result tables and reports should be generated from measured artifacts rather than manually typed metric values.

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
