# Tutorial 2 — Two-party VFL logistic with strict entity alignment

This example shows the lowest-level logistic API and the v0.3 alignment guard.

```python
import numpy as np

from vertimosaic.alignment import bind_entity_ids
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

entity_ids = np.asarray([f"customer-{i:04d}" for i in range(8)])

bank = ActiveParty(
    "bank",
    np.asarray([[0.1], [0.2], [0.4], [0.5], [0.7], [0.8], [1.0], [1.1]]),
    np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
)
telecom = PassiveParty(
    "telecom",
    np.asarray([[1.2], [1.0], [0.9], [0.8], [0.5], [0.4], [0.2], [0.1]]),
)

bind_entity_ids(bank, entity_ids)
bind_entity_ids(telecom, entity_ids)

model = VFLLogisticRegression(
    learning_rate=0.05,
    max_iter=100,
    require_entity_ids=True,
    missing_party_policy="error",
    seed=42,
)
model.fit(bank, [telecom])
probability = model.predict_proba([bank, telecom])[:, 1]
print(probability)
```

## Demonstrate the row-order guard

The following must fail even though both matrices contain eight rows:

```python
bad_telecom = PassiveParty("telecom", telecom._x[[1, 0, 2, 3, 4, 5, 6, 7]])
bind_entity_ids(
    bad_telecom,
    entity_ids[[1, 0, 2, 3, 4, 5, 6, 7]],
)

bad_model = VFLLogisticRegression(max_iter=5, require_entity_ids=True)
bad_model.fit(bank, [bad_telecom])
```

VertiMosaic raises an entity-order mismatch before optimization starts.

## Demonstrate the missing-party guard

After training with Bank and Telecom, this call fails by default:

```python
model.predict_proba([bank])
```

If you are deliberately studying service degradation, create a separately evaluated model configuration with:

```python
model = VFLLogisticRegression(missing_party_policy="zero_contribution")
```

That fallback is an experimental assumption, not a generally safe replacement for the missing party.
