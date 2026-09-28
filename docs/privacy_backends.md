# Optional privacy research backends

VertiMosaic v0.2.0 provides several explicit research mechanisms. None is silently enabled in the default VFL protocols, and the presence of these primitives must not be described as an end-to-end cryptographic-security guarantee.

## Differential privacy releases

`GaussianDPBackend` retains the original research API where the caller supplies an L2-sensitivity bound. It adds Gaussian noise with standard deviation `noise_multiplier * l2_sensitivity` and composes releases with zCDP.

`ClippedGaussianDPBackend` makes the message-level sensitivity contract executable:

- every released vector is L2-clipped to `clip_l2_norm`;
- add/remove adjacency uses sensitivity `clip_l2_norm`;
- replace-one adjacency uses sensitivity `2 * clip_l2_norm`;
- Gaussian noise is calibrated from that enforced sensitivity; and
- repeated releases compose through the zCDP accountant and can be converted to `(epsilon, delta)`.

This is still a **message-level** mechanism. End-to-end VFL differential privacy requires a concrete neighboring-dataset definition and coverage/accounting of every sensitive data-dependent release in the protocol.

## Secure aggregation

`PairwiseMaskSecureAggregation` is a dependency-light reference secure-sum primitive for integer vectors. Every pair of parties receives the same out-of-band secret; deterministic per-round pairwise masks are added with opposite signs and cancel in the aggregate.

Scope: honest-but-curious aggregation research. It does **not** implement dropout recovery, malicious-party verification, or collusion resistance.

## Additive secret sharing / MPC primitive

`AdditiveSecretSharingSum` splits integer vectors into additive shares over a large prime field and reconstructs their sum. It is useful for studying an MPC-style sum boundary but is deliberately **not a general-purpose MPC runtime** and supplies no malicious-party verification.

## Private set intersection

`OpenMinedPSIBackend` is an optional adapter for OpenMined's ECDH-based private-set-intersection implementation. Install the optional dependencies with:

```bash
pip install "vertimosaic[privacy-crypto]"
```

The adapter protects the set-intersection operation according to the upstream PSI implementation's threat model. It does not protect VFL residuals, gradients, Hessians, logits, routing messages, or other training traffic.

## Homomorphic addition

`PaillierHomomorphicSum` is an optional Paillier adapter for bounded signed integer-vector sums. It demonstrates additive homomorphic aggregation. It does not provide encrypted comparisons, encrypted tree construction, key-management infrastructure, or end-to-end homomorphically encrypted VFL training.

## What remains outside the guarantee

The default VFL algorithms remain raw-feature-local research protocols. VertiMosaic does not claim malicious-party security, collusion resistance, dropout-resilient secure aggregation, general-purpose MPC, end-to-end HE training, or end-to-end differential privacy by default. Every experiment should name the exact mechanism, observer and protected release rather than using the generic phrase "secure VFL".
