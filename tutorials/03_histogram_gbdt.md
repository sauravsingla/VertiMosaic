# Tutorial 3 — Vertical histogram GBDT

VertiMosaic's histogram GBDT keeps train-derived numeric split thresholds at the party that owns each feature. The coordinator sees aggregate candidate statistics plus opaque feature/bin references.

```python
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLHistGBDT
from vertimosaic.parties import PassiveParty

active, passive = make_vertical_synthetic(n_rows=2000, seed=42)
split = entity_level_split(active.labels, seed=42)
train_active, train_passive = slice_parties(active, passive, split.train)
val_active, val_passive = slice_parties(active, passive, split.validation)
test_active, test_passive = slice_parties(active, passive, split.test)

model = VFLHistGBDT(
    n_estimators=20,
    max_depth=3,
    min_samples_leaf=20,
    early_stopping_rounds=3,
    seed=42,
)
model.fit(train_active, train_passive, val_active, val_passive)

training_parties: list[PassiveParty] = [train_active, *train_passive]
test_parties: list[PassiveParty] = [test_active, *test_passive]
for source, target in zip(training_parties, test_parties, strict=True):
    source.share_histogram_routing_state_with(target)

probability = model.predict_proba(test_parties)[:, 1]
print(probability[:10])
print("messages:", model.transport.message_count)
print("logical payload bytes:", model.transport.estimated_payload_bytes)
```

## Why routing state is shared explicitly

Validation/test partitions must use the thresholds learned from the training partition. The training party therefore shares immutable threshold state only with another object representing the same owning organization. The model/coordinator retains an opaque routing handle rather than the literal threshold array.

## What remains visible

The protocol still exposes target-derived gradient/Hessian signals, node membership, aggregate histogram statistics, opaque split references and routed entity indices. Hiding numeric thresholds is useful separation of responsibilities, not a complete privacy guarantee.

## Recommended next checks

```bash
pytest -q tests/test_vfl_gbdt_vs_reference.py
pytest -q tests/test_vfl_gbdt_routing.py
pytest -q tests/test_histogram_state_isolation.py
```
