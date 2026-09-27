# Privacy boundaries

Implemented properties:

- passive raw feature tables remain party-local;
- preprocessing is party-local;
- logistic and histogram GBDT passive computations can run behind authenticated remote process/network boundaries;
- GBDT numeric split thresholds and histogram bins remain inside the owning party service;
- federated communication has explicit message boundaries;
- audit logging retains metadata rather than private arrays;
- modelling outputs use pseudonymous entity identifiers where identifiers are needed; and
- selected leakage surfaces are measured through reproducible empirical experiments.

## Empirical leakage research

The baseline privacy audit measures confidence-based membership inference, logistic residual-label inference, and passive-tree routing/entity exposure. The multi-factor research suite extends those measurements across:

- dataset size;
- number of participating parties;
- logistic epochs and GBDT boosting rounds;
- model regularization;
- logistic versus histogram GBDT;
- mitigation/utility trade-offs; and
- multiple deterministic seeds with reported confidence intervals.

`VFLLogisticRegression.residual_noise_std` is an empirical mitigation control that adds Gaussian noise to residual signals sent to passive parties during training. The active party retains its clean target-owned residual for its own gradient and intercept update. This control is useful for measuring leakage/utility trade-offs, but it is **not** a differential-privacy mechanism and has no epsilon/delta guarantee.

GBDT mitigation experiments compare structural controls such as shallower trees, larger leaves, and stronger leaf regularization. These are empirical exposure/utility comparisons, not cryptographic defenses.

Run the reduced research grid with:

```bash
python scripts/run_privacy_research.py --smoke
```

Run the complete configured multi-seed grid with:

```bash
python scripts/run_privacy_research.py
```

The runner writes raw per-seed measurements and aggregate confidence-interval summaries to JSON and CSV so results can be independently inspected rather than inferred from documentation claims.

## Not guaranteed by default

- cryptographic confidentiality;
- Private Set Intersection;
- malicious-party security;
- collusion resistance;
- formal differential privacy;
- secure aggregation, MPC, homomorphic encryption, or secret sharing; or
- protection against all gradient, activation, routing, model-inversion, or membership-inference attacks.

Remote transport and mTLS protect the communication channel and establish a real process/network boundary; they do not remove information already present in legitimate protocol payloads.

SHA-256 pseudonymization is not a Private Set Intersection protocol. Empirical attack results are measurements for the tested populations/configurations and must not be generalized into a formal privacy guarantee.
