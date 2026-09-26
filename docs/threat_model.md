# Threat model

## Considered risks

- honest-but-curious active or passive parties;
- malicious parties;
- collusion;
- gradient / Hessian leakage;
- activation/logit leakage;
- tree routing leakage;
- entity-membership and identifier leakage;
- model inversion and membership inference;
- data and feature poisoning.

## Implemented properties

- raw feature locality;
- party-local preprocessing;
- explicit protocol separation;
- pseudonymous identifiers for research workflows;
- metadata-only audit history.

## Not automatically guaranteed

- cryptographic confidentiality;
- secure private set intersection;
- malicious-party security;
- collusion resistance;
- formal differential privacy;
- encrypted gradients, Hessians, or routing decisions.

Potential extensions include MPC, homomorphic encryption, secret sharing, secure aggregation, and differential privacy. Such properties must not be claimed until implemented and tested.
