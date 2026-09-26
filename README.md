# VertiMosaic

**Cross-industry vertical federated learning for tabular data**

VertiMosaic is an open-source research framework for training machine-learning models across organizations that hold complementary attributes about overlapping entities while keeping their raw feature tables local.

| Party | Private information represented |
|---|---|
| Bank | Financial / payment behaviour + target |
| Telecom | Communications / service behaviour |
| Insurance | Claims / risk behaviour |
| Retail | Purchase behaviour |

- **Federation type:** Vertical Federated Learning
- **Raw feature tables pooled during VFL training:** No
- **Graph ML required:** No
- **External real-world datasets supported:** Yes
- **Four source datasets contain the same actual people:** No
- **Cross-industry linkage:** Explicitly semi-synthetic and documented
- **Execution:** Designed for ordinary Python environments without accelerator-specific dependencies

VertiMosaic distinguishes data locality, pseudonymization, synthetic linkage, and cryptographic privacy guarantees. The default simulator provides **raw-feature locality and protocol separation**; it does **not** claim secure PSI, homomorphic encryption, MPC, malicious-party security, or formal differential privacy.

## Why vertical, not horizontal?

**Horizontal FL** usually means different entities/rows with a similar feature schema.  
**Vertical FL** means the same or overlapping entities with different feature columns owned by different parties.

VertiMosaic implements the second case.

```text
                    shared / aligned entity
                              |
       +----------------------+----------------------+
       |                      |                      |
      Bank                 Telecom               Insurance              Retail
  payment features      service features       claims/risk          purchase features
      + label                 |                      |                      |
       +---------------------- private feature mosaic --------------------+
                              |
                              v
                    vertical federated model

                    raw feature tables stay local
```

## Models

### `VFLLogisticRegression`
A first-principles NumPy implementation of vertically partitioned logistic regression. Each party computes its local logit and local gradient; passive-party feature matrices stay inside their party object.

### `VFLHistGBDT`
A histogram-based vertical gradient-boosted tree research implementation. Parties bin features locally and exchange aggregated gradient/Hessian histograms plus routing information. This is more expressive for nonlinear tabular relationships, but gradient/Hessian and routing signals can leak information; see the threat model.

## External datasets

The main externally grounded benchmark uses independent public sources:

| Party | Dataset | Identifier |
|---|---|---|
| Bank | UCI Default of Credit Card Clients | UCI 350, DOI `10.24432/C55S3H` |
| Telecom | UCI Iranian Churn | UCI 563, DOI `10.24432/C5JW3Z` |
| Insurance | French Motor Third-Party Liability Claims | OpenML `41214` / `41215` |
| Retail | UCI Online Retail | UCI 352, DOI `10.24432/C5BW33` |

These four sources do **not** describe the same real people. VertiMosaic therefore treats the four-industry benchmark as an **externally grounded semi-synthetic cross-industry VFL benchmark**: real source-domain feature distributions, explicit synthetic linkage. Numeric identifiers are never treated as cross-dataset matches merely because they happen to coincide.

An optional IEEE-CIS transaction/identity benchmark can be prepared from user-supplied authorized local files. It is not bundled and CI does not depend on it.

See [DATA_LICENSES.md](DATA_LICENSES.md) and [docs/datasets.md](docs/datasets.md).

## Quick start

```bash
python -m pip install -e ".[data,dev]"
vertimosaic --help
vertimosaic demo --rows 5000 --seed 42 --model logistic
vertimosaic demo --rows 5000 --seed 42 --model vfl-hist-gbdt
```

List public-source dataset definitions:

```bash
vertimosaic datasets list
```

Download an external dataset explicitly:

```bash
vertimosaic datasets download bank
vertimosaic datasets download telecom
vertimosaic datasets download insurance_freq
vertimosaic datasets download insurance_sev
vertimosaic datasets download retail
```

Raw and processed datasets are git-ignored.

## Research questions

The repository is organized to study:

1. Whether complementary vertically partitioned tabular information improves measured predictive performance relative to an active-party-only baseline.
2. How a federated linear objective compares with its centralized mathematical counterpart.
3. How nonlinear vertical histogram boosting behaves on distributed nonlinear signals.
4. Predictive utility contributed by each industry.
5. Sensitivity to partial entity overlap, missing parties, and party-specific drift.
6. Compute, memory, and simulated communication costs.
7. Differences between observed-target, distributed-signal semi-synthetic, synthetic-scale, and optionally real-linked benchmarks.

No positive result is assumed in advance. Generated reports should retain negative or inconclusive findings.

## Development checks

```bash
ruff check .
ruff format --check .
mypy src/vertimosaic
pytest -q
python -m build
twine check dist/*
```

Additional security/dependency checks:

```bash
bandit -r src
pip-audit
make sbom
```

## Privacy scope

Implemented by the default simulator:

- raw passive-party feature locality;
- local preprocessing;
- vertically separated feature ownership;
- metadata-only audit logs;
- pseudonymous research identifiers.

Not automatically provided:

- secure private set intersection;
- cryptographic confidentiality;
- malicious-party security;
- collusion resistance;
- formal differential privacy;
- encrypted gradient exchange.

See [docs/threat_model.md](docs/threat_model.md).

## License

Source code is licensed under Apache-2.0. External datasets keep their own terms and are not relicensed by this repository.
