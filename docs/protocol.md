# Federated protocols

## Logistic regression

Each party computes a local logit contribution `X_p @ w_p`. The active party aggregates contributions, computes binary-logistic residuals from its target, and returns the residual signal required for each party to compute its local gradient. Raw feature matrices stay local.

## Vertical histogram GBDT

The active party maintains predictions and computes per-row gradients/Hessians. Each party builds local split histograms. Only aggregated candidate statistics and opaque ownership references are considered centrally; the split-owning party performs routing locally.

Gradient, Hessian, activation, and routing messages can leak information. The default protocol is therefore a raw-feature-locality-preserving research simulator, not cryptographically secure VFL.
