# Federated protocols

## Logistic regression

Each party computes a local logit contribution `X_p @ w_p`. The active party aggregates contributions, computes binary-logistic residuals from its target, and returns the residual signal required for each party to compute its local gradient. Raw feature matrices stay local.

The reference implementation supports entity-aligned mini-batches, L1/L2 or elastic-net regularization, class weighting, gradient clipping, deterministic shuffling, learning-rate schedules, warm start, validation monitoring and early stopping. Convergence telemetry is recorded as measured training/validation loss rather than inferred from final metrics.

## Vertical histogram GBDT

The active party maintains predictions and computes per-row gradients/Hessians. Before tree construction, each party fits and retains its own quantile-bin representation of its local training matrix. Node-level histogram candidates are then accumulated from those retained party-local bin codes using aggregate gradient/Hessian/count statistics rather than recomputing and exposing raw feature thresholds for every node.

Passive candidate records expose aggregate statistics plus an opaque party-local feature/bin reference. The coordinator computes gains and selects the best party/reference; the owning party resolves the reference and performs routing locally. Numeric thresholds remain encapsulated in the in-process party-local split reference and are not coordinator-facing candidate fields. The implementation also supports party-local aggregation of GBDT split-count/gain feature importance so publication reports can receive aggregates without raw rows.

Gradient, Hessian, activation, entity-membership and routing messages can leak information. The default protocol is therefore a raw-feature-locality-preserving research simulator, not cryptographically secure VFL; it does not claim PSI, secure aggregation, MPC, homomorphic encryption, differential privacy, collusion resistance or malicious-party security.
