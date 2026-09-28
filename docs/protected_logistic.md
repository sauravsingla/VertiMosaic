# Protected logistic research path

VertiMosaic v0.3 includes one intentionally narrow, executable privacy-enhanced VFL path instead of treating independent privacy primitives as if they automatically secured the full protocol.

The path combines:

1. **PSI-based entity intersection** through a configured `PSIBackend` (the optional `OpenMinedPSIBackend` is the packaged adapter);
2. **exact ordered entity binding** after the intersection, enforced by the model boundary;
3. **VFL logistic training with all raw feature matrices remaining party-local**;
4. **L2 clipping + Gaussian noise on every active-to-passive residual message** through `ClippedGaussianDPBackend`;
5. **zCDP composition converted to `(epsilon, delta)`** for those residual releases.

## Example

```python
from vertimosaic.privacy import OpenMinedPSIBackend, fit_protected_logistic

run = fit_protected_logistic(
    active=bank_party,
    active_entity_ids=bank_ids,
    passive=[telecom_party, insurance_party],
    passive_entity_ids=[telecom_ids, insurance_ids],
    psi_backend=OpenMinedPSIBackend(),
    clip_l2_norm=1.0,
    noise_multiplier=2.0,
    seed=42,
    model_kwargs={"max_iter": 200, "learning_rate": 0.05},
)

report = run.privacy_report(delta=1e-6)
```

Install the optional PSI adapter with:

```bash
pip install "vertimosaic[privacy-crypto]"
```

## Exact guarantee boundary

This is **not** a generic `secure=True` switch and must not be described as end-to-end cryptographically secure VFL.

- PSI protects the set-intersection operation according to the selected PSI backend's threat model.
- Ordered entity digests detect accidental row-order mismatch after alignment; hashing is an integrity/correctness check, not PSI.
- Clipped Gaussian accounting covers residual messages routed through the configured backend.
- Passive-to-active logits, model parameters, timing, message sizes, and other protocol metadata are not automatically differentially private.
- The path does not provide malicious-party security, collusion resistance, Byzantine robustness, encrypted model training, or dropout-resilient secure aggregation.

## Evaluation requirements

Any paper/report using this path should report side-by-side:

- ordinary VFL utility;
- protected-path utility;
- PSI intersection size and coverage;
- number of protected residual releases;
- clipping norm, adjacency definition, noise multiplier, epsilon and delta;
- wall-clock overhead;
- logical communication overhead;
- the same empirical privacy attacks used elsewhere in the repository;
- the exact guarantee-boundary paragraph above or equivalent wording.

The objective is to make the privacy/utility/resource trade-off measurable without expanding the claim beyond the mechanisms actually executed.
