# Model card

## Intended use
Research, education, reproducible benchmarking, and protocol experimentation for vertically partitioned tabular learning.

## Out of scope
Production deployment involving regulated personal data without a separate security, privacy, legal, and operational review.

## Models
- VFL Logistic: transparent reference objective.
- VFLHistGBDT: nonlinear tabular research model based on local binning and aggregated split statistics.

## Reporting
Use ROC-AUC and PR-AUC alongside calibration and threshold-dependent metrics. Thresholds should be selected on validation data rather than the test set.
