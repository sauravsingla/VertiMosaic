# SPDX-License-Identifier: Apache-2.0
import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.privacy import privacy_research_summary_rows, run_privacy_research


_REQUIRED_EXPERIMENTS = {
    "attack_vs_dataset_size",
    "attack_vs_party_count",
    "attack_vs_training_rounds",
    "attack_vs_regularization",
    "logistic_vs_gbdt",
    "mitigation_utility_tradeoff",
}


def test_privacy_research_smoke_covers_every_requested_axis_with_uncertainty() -> None:
    result = run_privacy_research(smoke=True)
    assert result["schema_version"] == 2
    assert result["formal_privacy_guarantee"] is False
    experiments = result["experiments"]
    assert set(experiments) == _REQUIRED_EXPERIMENTS

    for experiment in experiments.values():
        assert experiment["raw"]
        assert experiment["summary"]
        for summary in experiment["summary"]:
            assert summary["seed_count"] == 2
            measured = [
                value
                for key, value in summary.items()
                if key
                in {
                    "membership_roc_auc",
                    "membership_advantage",
                    "label_inference_accuracy",
                    "routing_exposure_fraction",
                    "routing_events_per_entity",
                    "holdout_roc_auc",
                }
            ]
            assert measured
            for interval in measured:
                assert interval["n"] == 2
                assert interval["ci95_low"] <= interval["mean"] <= interval["ci95_high"]

    flattened = privacy_research_summary_rows(result)
    assert flattened
    assert {row["experiment"] for row in flattened} == _REQUIRED_EXPERIMENTS


def test_residual_noise_is_training_time_mitigation_not_reporting_only() -> None:
    active, passive = make_vertical_synthetic(240, seed=91)
    active_noisy, passive_noisy = make_vertical_synthetic(240, seed=91)
    baseline = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=20,
        tolerance=0.0,
        gradient_clip=None,
        residual_noise_std=0.0,
        seed=7,
    )
    mitigated = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=20,
        tolerance=0.0,
        gradient_clip=None,
        residual_noise_std=0.2,
        seed=7,
    )
    baseline.fit(active, passive)
    mitigated.fit(active_noisy, passive_noisy)

    passive_name = passive[0].name
    assert not np.allclose(
        baseline.weights_[passive_name],
        mitigated.weights_[passive_name],
    )
    assert np.isfinite(mitigated.predict_proba([active_noisy, *passive_noisy])).all()
