"""Run the documented paper artifact pipeline explicitly; external downloads may be slow."""

from vertimosaic.experiments import (
    run_ablation_study,
    run_contribution_study,
    run_drift_study,
    run_dropout_study,
    run_overlap_study,
)

if __name__ == "__main__":
    run_ablation_study()
    run_contribution_study()
    run_overlap_study()
    run_dropout_study()
    run_drift_study()
