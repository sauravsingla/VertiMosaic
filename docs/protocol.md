# Protocol

## Logistic VFL
Party `p` computes `z_p = X_p w_p`. The active party aggregates logits, computes probabilities and residuals from its local label, and sends the residual signal. Each party computes `X_p^T residual` locally and updates its local weights.

## Histogram GBDT
The active party computes per-sample binary-log-loss gradients and Hessians. Parties bin features locally and compute per-feature/per-bin gradient/Hessian/count histograms. The coordinator compares split gain summaries. The winning party applies the split locally and supplies routing information required to continue tree construction.

The protocol intentionally exposes some intermediate values. This is a data-locality simulator, not a cryptographically secure VFL implementation.
