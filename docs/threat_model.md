# Threat model

VertiMosaic is designed primarily for an honest-but-curious research setting. The active party may observe protocol messages required for training; passive parties may receive target-derived residuals, gradients, or Hessians according to the model. The simulator does not defend against malicious parties or collusion.

## Risks considered

- gradient and Hessian leakage;
- activation/logit leakage;
- split-routing leakage;
- entity-membership and identifier leakage;
- model inversion and membership inference;
- data and feature poisoning;
- malicious-message manipulation and collusion.

## Implemented controls

Raw feature locality, local preprocessing, explicit protocol separation, metadata-only audit logs, and pseudonymous research identifiers.

## Not automatically guaranteed

Cryptographic confidentiality, malicious-party security, collusion resistance, PSI, formal differential privacy, secure aggregation, MPC, homomorphic encryption, or secret sharing.

Those mechanisms are potential extensions and must not be claimed unless separately implemented and validated.
