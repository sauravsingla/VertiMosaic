from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.experiments import run_overlap_study
from vertimosaic.parties import PassiveParty
from vertimosaic.preprocessing import LocalTabularPreprocessor
from vertimosaic.reporting import build_data_driven_report


def test_histogram_party_retains_local_bins_across_candidate_calls() -> None:
    values = np.array([[0.1], [0.2], [0.4], [0.7], [0.8], [0.9]])
    party = PassiveParty("telecom", values)
    party.prepare_histogram_bins(4)
    assert party.histogram_bins_ready
    first_bins = party._histogram_bins
    gradients = np.array([-1.0, -0.5, -0.2, 0.2, 0.5, 1.0])
    hessians = np.ones(6)
    first = party.candidate_histograms(
        gradients,
        hessians,
        np.arange(6),
        max_bins=4,
        min_samples_leaf=1,
    )
    second = party.candidate_histograms(
        gradients,
        hessians,
        np.array([0, 1, 2, 3]),
        max_bins=4,
        min_samples_leaf=1,
    )
    assert first and second
    assert party._histogram_bins is first_bins
    assert all("threshold" not in candidate and "feature" not in candidate for candidate in first)


def test_local_preprocessor_supports_train_only_quantile_bins() -> None:
    train = pd.DataFrame({"amount": [0.0, 1.0, 2.0, 3.0, np.nan]})
    future = pd.DataFrame({"amount": [-100.0, 1.5, 100.0, np.nan]})
    preprocessor = LocalTabularPreprocessor(["amount"], scaling="none")
    preprocessor.fit_quantile_bins(train, max_bins=4)
    learned_edges = preprocessor.quantile_bin_edges_["amount"].copy()
    transformed = preprocessor.transform_quantile_bins(future)
    assert transformed.shape == (4, 1)
    assert np.array_equal(preprocessor.quantile_bin_edges_["amount"], learned_edges)
    assert transformed[-1, 0] == len(learned_edges)
    assert transformed[0, 0] == 0
    assert transformed[2, 0] == len(learned_edges)


def test_overlap_study_compares_intersection_and_missing_aware(tmp_path: Path) -> None:
    frame = run_overlap_study(
        rows=400,
        seed=17,
        fractions=(1.0, 0.5),
        output=tmp_path / "partial_overlap.csv",
        write_run=False,
    )
    assert set(frame["method"]) == {"intersection_only", "availability_indicator"}
    assert len(frame) == 4
    half = frame.loc[frame["overlap_fraction"] == 0.5]
    assert np.allclose(half["intersection_coverage"], 0.5)
    intersection = half.loc[half["method"] == "intersection_only"].iloc[0]
    missing_aware = half.loc[half["method"] == "availability_indicator"].iloc[0]
    assert np.isclose(float(intersection["coverage"]), 0.5)
    assert np.isclose(float(missing_aware["coverage"]), 1.0)
    assert half["threshold_selected_on_validation"].between(0.0, 1.0).all()


def test_report_interprets_overlap_methods_separately(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame(
        [
            {
                "method": "intersection_only",
                "overlap_fraction": 1.0,
                "intersection_coverage": 1.0,
                "coverage": 1.0,
                "pr_auc": 0.8,
            },
            {
                "method": "intersection_only",
                "overlap_fraction": 0.5,
                "intersection_coverage": 0.5,
                "coverage": 0.5,
                "pr_auc": 0.7,
            },
            {
                "method": "availability_indicator",
                "overlap_fraction": 1.0,
                "intersection_coverage": 1.0,
                "coverage": 1.0,
                "pr_auc": 0.81,
            },
            {
                "method": "availability_indicator",
                "overlap_fraction": 0.5,
                "intersection_coverage": 0.5,
                "coverage": 1.0,
                "pr_auc": 0.76,
            },
        ]
    ).to_csv(results / "partial_overlap.csv", index=False)
    report = build_data_driven_report(
        {"metrics": {"pr_auc": 0.8}, "comparisons": []},
        results_directory=results,
        benchmark_directory=tmp_path / "benchmarks",
        case_study_path=tmp_path / "missing_case_study.json",
    )
    overlap_observations = [
        item for item in report["observations"] if item.startswith("Partial-overlap study")
    ]
    assert len(overlap_observations) == 2
    assert any("intersection_only" in item for item in overlap_observations)
    assert any("availability_indicator" in item for item in overlap_observations)
