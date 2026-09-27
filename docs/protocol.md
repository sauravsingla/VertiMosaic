# Federated protocols

## Logistic regression

Each party computes a local logit contribution `X_p @ w_p`. The active party aggregates contributions, computes binary-logistic residuals from its target, and returns the residual signal required for each party to compute its local gradient. Raw feature matrices stay local.

The reference implementation supports entity-aligned mini-batches, L1/L2 or elastic-net regularization, class weighting, gradient clipping, deterministic shuffling, learning-rate schedules, warm start, validation monitoring and early stopping. Convergence telemetry is recorded as measured training/validation loss rather than inferred from final metrics.

## Vertical histogram GBDT

The active party maintains predictions and owns the labels. It computes per-row gradients and Hessians for binary logistic loss. The gradients and Hessians needed by each passive party are delivered through the simulated `Message`/`InMemoryTransport` boundary; the passive party therefore receives target-derived optimization signals, which is an explicit leakage surface rather than a cryptographic privacy guarantee.

Before tree construction, every party fits and retains a quantile-bin representation of its **training** matrix. Numeric bin thresholds are kept in immutable party-local routing state. Coordinator-visible `OpaqueSplitReference` objects contain only local `feature_ref` and `bin_ref` integers; they contain no threshold and no feature name. Validation and test rows are routed with the training-derived routing state, so evaluation data do not refit histogram thresholds.

For each tree node, the active party sends the node-membership indices and any feature-subsample references needed by a passive party through `Message` objects. Each passive party then computes local candidate histograms from its retained bin codes. The real coordinator-facing candidate payload—not a placeholder—is returned through `Message`/`InMemoryTransport` and contains only:

- opaque feature/bin reference;
- left/right gradient sums;
- left/right Hessian sums;
- left/right counts.

The active party calculates gains and chooses the best party/reference. For a passive-owned split, the selected opaque reference plus the aligned node membership are sent back through the transport. The owning party resolves the reference against its local training-derived routing state, applies the split locally, and returns only the partition-routing indices required to continue tree construction. Training, validation and inference routing use this same explicit message path for passive-owned splits.

`AuditEvent` persists only communication metadata—message type, sender/receiver role, direction, stage/step, shape, scalar count and estimated bytes. Ephemeral payload values are not retained by the transport or audit log. Communication volume is therefore a **simulated payload-size estimate**, not observed network traffic.

Feature-importance reporting can ask each owning party to aggregate its opaque split references into `split_count`, `gain_sum` and `gain_mean`; raw rows, numeric thresholds and feature names are not needed by the coordinator for that aggregation.

Gradient, Hessian, activation, entity-membership and routing messages can leak information. The default protocol is therefore a raw-feature-locality-preserving research simulator, not cryptographically secure VFL. The in-process Python implementation provides protocol separation, not process isolation, and it does not claim PSI, secure aggregation, MPC, homomorphic encryption, formal differential privacy, collusion resistance or malicious-party security.
