# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from vertimosaic.experiments.feature_importance_study import run_feature_importance_study
from vertimosaic.experiments.overlap_study import run_overlap_study
from vertimosaic.experiments.studies import run_ablation_study


def test_feature_importance_gbdt_predicts_on_fresh_test_partition(tmp_path: Path) -> None:
    frames = run_feature_importance_study(
        rows=240,
        seed=11,
        directory=tmp_path / "results",
        write_run=True,
        runs_root=tmp_path / "runs",
    )
    assert set(frames) == {"bank", "telecom", "insurance", "retail"}


def test_gbdt_ablation_predicts_on_fresh_test_partitions(tmp_path: Path) -> None:
    frame = run_ablation_study(
        rows=240,
        seed=13,
        model_name="vfl-hist-gbdt",
        output=tmp_path / "ablation.csv",
        write_run=False,
    )
    assert len(frame) == 8
    assert frame["roc_auc"].notna().all()


def test_gbdt_overlap_predicts_on_fresh_validation_and_test_partitions(tmp_path: Path) -> None:
    frame = run_overlap_study(
        rows=300,
        seed=17,
        model_name="vfl-hist-gbdt",
        fractions=(1.0,),
        methods=("intersection_only",),
        output=tmp_path / "overlap.csv",
        write_run=False,
    )
    assert len(frame) == 1
    assert frame["threshold_selected_on_validation"].notna().all()
