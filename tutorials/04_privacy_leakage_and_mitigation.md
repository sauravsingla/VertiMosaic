# Tutorial 4 — Privacy leakage and a scoped mitigation

This tutorial compares ordinary logistic VFL with the v0.3 clipped-Gaussian residual-release mechanism. The mechanism protects one message family; it does not establish end-to-end VFL differential privacy.

```python
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.privacy import ClippedGaussianDPBackend

active, passive = make_vertical_synthetic(n_rows=800, seed=42)

ordinary = VFLLogisticRegression(
    max_iter=20,
    learning_rate=0.05,
    require_entity_ids=True,
    seed=42,
)
ordinary.fit(active, passive)
ordinary_probability = ordinary.predict_proba([active, *passive])[:, 1]

protected = VFLLogisticRegression(
    max_iter=20,
    learning_rate=0.05,
    require_entity_ids=True,
    residual_dp_backend=ClippedGaussianDPBackend(
        clip_l2_norm=1.0,
        noise_multiplier=2.0,
        adjacency="replace_one",
        seed=42,
    ),
    seed=42,
)
protected.fit(active, passive)
protected_probability = protected.predict_proba([active, *passive])[:, 1]

print(protected.privacy_report(delta=1e-6))
```

## Run the empirical privacy suites

```bash
python scripts/run_privacy_audit.py \
  --rows 1200 --seed 42 \
  --output reports/privacy_audit.json

vertimosaic-advanced-privacy-audit \
  --rows 1200 --seed 42 \
  --output reports/advanced_privacy_audit.json

python scripts/run_privacy_research.py \
  --output reports/privacy_research.json \
  --csv reports/privacy_research_summary.csv
```

## Protected PSI + residual path

With the optional crypto dependencies installed, `fit_protected_logistic` can first intersect party entity sets using a PSI backend, canonically reorder the common entities, enforce strict entity alignment, and then train logistic VFL with clipped-Gaussian residual releases.

See `docs/protected_logistic.md` for the complete API and guarantee boundary.

## Interpretation

A meaningful privacy experiment reports both sides of the trade-off:

- held-out predictive metrics;
- attack metrics;
- number of protected releases;
- clip norm and adjacency definition;
- noise multiplier, epsilon and delta;
- runtime and communication overhead;
- dataset/model/seed configuration.

Do not replace this with a generic statement such as "DP enabled" or "secure VFL".
