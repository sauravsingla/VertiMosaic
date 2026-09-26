# Privacy boundaries

Implemented properties:

- passive raw feature tables remain in party objects;
- preprocessing is party-local;
- federated communication has explicit message boundaries;
- audit logging retains metadata rather than private arrays;
- modelling outputs use pseudonymous entity identifiers where identifiers are needed.

Not guaranteed by default:

- cryptographic confidentiality;
- Private Set Intersection;
- malicious-party security;
- collusion resistance;
- formal differential privacy;
- protection against all gradient, activation, routing, model-inversion, or membership-inference attacks.

SHA-256 pseudonymization is not a Private Set Intersection protocol.
