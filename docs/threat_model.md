# Threat model

VertiMosaic is designed primarily for an honest-but-curious research setting. The active party may observe protocol messages required for training; passive parties may receive target-derived residuals, gradients, or Hessians according to the model. The simulator and reference remote services do not defend against malicious parties or collusion.

## Risks considered

- gradient and Hessian leakage;
- activation/logit leakage;
- split-routing leakage;
- entity-membership and identifier leakage;
- model inversion and membership inference;
- data and feature poisoning;
- malicious-message manipulation and collusion.

## Implemented controls

Raw feature locality, local preprocessing, explicit protocol separation, metadata-only audit logs, pseudonymous research identifiers, authenticated HTTPS/mTLS-capable reference transport, and process-isolated passive-party RPC for both logistic and histogram GBDT feature-dependent computations.

Histogram bins and numeric split thresholds remain inside their owning party service. The coordinator sees aggregate candidate statistics, opaque split references, opaque routing handles, and routed entity indices. These controls reduce unnecessary raw-data exposure but do not make legitimate protocol messages information-free.

## Empirical attack measurement

The repository includes reproducible measurements for confidence-based membership inference, logistic residual-label inference, and passive-tree entity-routing exposure. The extended privacy research suite varies dataset size, party count, training duration, regularization, model family, and mitigation strength across multiple seeds, reporting raw results and confidence intervals.

The logistic residual-noise option and GBDT structural mitigation settings are empirical research controls. They are not claimed to provide differential privacy, cryptographic confidentiality, or resistance to an adaptive adversary.

## Not automatically guaranteed

Cryptographic confidentiality, malicious-party security, collusion resistance, PSI, formal differential privacy, secure aggregation, MPC, homomorphic encryption, or secret sharing.

Those mechanisms are potential extensions and must not be claimed unless separately implemented and validated. Likewise, a measured reduction in a particular attack metric under a tested configuration must not be interpreted as a general privacy proof.
