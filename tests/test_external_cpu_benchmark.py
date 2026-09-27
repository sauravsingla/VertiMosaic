from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from vertimosaic.experiments import benchmarks


def test_external_cpu_benchmark_uses_actual_anchor_size(monkeypatch, tmp_path) -> None:
    def fake_fetch(name: str, **_: object) -> SimpleNamespace:
        rows = 321 if name == "bank" else 50
        return SimpleNamespace(features=pd.DataFrame({"x": range(rows)}))

    def fake_run(**_: object) -> dict[str, object]:
        return {
            "metrics": {"roc_auc": 0.7, "pr_auc": 0.6, "brier": 0.2, "f1": 0.5},
            "communication": {
                "message_count": 4,
                "scalar_count": 100,
                "forward_estimated_bytes": 600,
                "backward_estimated_bytes": 400,
                "traffic_type": "SIMULATED PAYLOAD SIZE",
            },
            "data_preparation_seconds": 1.0,
            "entity_alignment_seconds": 2.0,
            "preprocessing_seconds": 3.0,
            "training_seconds": 4.0,
            "inference_seconds": 5.0,
            "training_steps": 6,
            "peak_rss_bytes": 7,
            "estimated_communication_bytes": 1000,
            "run_directory": "runs/fake",
        }

    monkeypatch.setattr(benchmarks, "fetch_external_party", fake_fetch)
    monkeypatch.setattr(benchmarks, "run_external_experiment", fake_run)
    frame = benchmarks.run_external_cpu_benchmark(
        mode="observed_target_external",
        model_name="logistic",
        bootstrap_replicates=10,
        directory=tmp_path,
        write_runs=False,
        append=False,
    )
    assert len(frame) == 1
    assert int(frame.iloc[0]["rows"]) == 321
    assert frame.iloc[0]["mode"] == "observed_target_external"
    assert frame.iloc[0]["traffic_type"] == "SIMULATED PAYLOAD SIZE"
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "environment.json").exists()
