from __future__ import annotations

from pathlib import Path

from vertimosaic.experiments.formal_benchmark import run_formal_benchmark


def test_formal_benchmark_has_four_explicit_protocol_rows(tmp_path: Path) -> None:
    output = tmp_path / "formal.csv"
    frame = run_formal_benchmark(
        rows=240,
        seed=5,
        include_robustness=False,
        logistic_max_iter=2,
        gbdt_estimators=1,
        output=output,
    )
    assert frame["protocol"].tolist() == [
        "centralized_all_features",
        "single_party_bank",
        "vfl_logistic",
        "vfl_hist_gbdt",
    ]
    assert frame.loc[frame["federated"], "estimated_communication_bytes"].gt(0).all()
    assert frame.loc[~frame["federated"], "estimated_communication_bytes"].eq(0).all()
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert output.with_suffix(".md").exists()
