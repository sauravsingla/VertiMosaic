# Cross-domain linkage

VertiMosaic supports target-blind rank/Gaussian-copula-style semi-synthetic linkage. Party-local unsupervised row summaries are converted to empirical Gaussian scores; controlled latent dependence guides donor selection while preserving source marginals as much as practical. Donor reuse is recorded.

In `observed_target_external`, the Bank target is prohibited from linkage. In `distributed_signal_external`, the semi-synthetic target is created only after linked profiles exist.
