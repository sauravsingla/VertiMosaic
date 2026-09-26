from pathlib import Path

from vertimosaic.training import run_demo


def test_demo_runs(tmp_path: Path) -> None:
    result = run_demo(rows=1200, seed=3, model="logistic", output=tmp_path)
    assert result["metrics"]["roc_auc"] > 0.65
    assert (tmp_path / "metrics.json").exists()
