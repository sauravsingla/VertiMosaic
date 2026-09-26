# Model card

## Intended use

Research into CPU-only vertical federated learning on tabular data, including correctness, partial overlap, missing parties, drift, party utility, and communication/computation trade-offs.

## Models

`VFLLogisticRegression` is the transparent reference protocol. `VFLHistGBDT` is the nonlinear vertical histogram-boosting research model. Centralized baselines are explicitly non-federated research comparators.

## Privacy

Raw feature locality is provided by protocol separation. Cryptographic privacy is not provided by default.

## Evaluation

Use entity-disjoint train/validation/test splits, validation-only threshold selection, test bootstrap confidence intervals, paired bootstrap comparisons, and data-driven interpretation. Negative results must be retained.
