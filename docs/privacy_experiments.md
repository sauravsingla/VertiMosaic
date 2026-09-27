# Empirical privacy experiments

VertiMosaic documents privacy boundaries separately from empirical attack measurements. The measurements in `vertimosaic.privacy` are reproducible leakage baselines; they are **not formal privacy guarantees**.

## Current baselines

### Confidence membership inference

The attack compares prediction confidence on training and held-out entities. It reports ROC-AUC, best threshold attack advantage, and sample counts. This is intentionally a simple baseline rather than a claim that confidence thresholding is the strongest membership-inference attack.

### Logistic residual label inference

For the plain binary logistic VFL residual `p - y`, the sign can expose the active party's binary label to a passive party: a negative residual implies `y=1`, while a positive residual implies `y=0` for non-degenerate probabilities. The audit measures exposed/ambiguous rows and inference accuracy.

### Passive GBDT routing exposure

For passive-party-owned tree splits, the active party transmits node/entity membership so the owning party can route the split. The audit counts routed entity events, unique exposed entities, exposure fraction, and repeated exposure.

## Reproduce

```bash
python scripts/run_privacy_audit.py \
  --rows 1200 \
  --seed 42 \
  --output reports/privacy_audit.json
```

Tagged release evidence includes the resulting JSON. CI runs a smaller deterministic smoke audit.

## Interpretation

These experiments make known leakage surfaces measurable. They do not provide PSI, MPC, homomorphic encryption, secure aggregation, malicious-party security, collusion resistance, or formal differential privacy. Stronger attack implementations and mitigations can be added behind the same measurement API and compared against the baseline protocol.
