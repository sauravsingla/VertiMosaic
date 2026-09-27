import pytest

from vertimosaic.reporting import (
    MissingMeasuredResultsError,
    build_data_driven_report,
    interpret_delta,
    write_final_report,
)


def test_interpreter_retains_negative_result() -> None:
    text = interpret_delta("roc_auc", -0.01, -0.03, 0.01)
    assert "did not outperform" in text


def test_report_is_data_driven() -> None:
    report = build_data_driven_report(
        {
            "metrics": {"brier": 0.2, "ece": 0.05},
            "comparisons": [
                {"metric": "roc_auc", "delta": 0.01, "lower": -0.01, "upper": 0.03}
            ],
            "estimated_communication_bytes": 1024,
        }
    )
    joined = " ".join(report["observations"])
    assert "sampling variability" in joined
    assert "not measurements of real network traffic" in joined


def test_final_report_refuses_absent_measured_payload(tmp_path) -> None:
    with pytest.raises(MissingMeasuredResultsError):
        write_final_report(
            {"metrics": {}, "comparisons": []},
            tmp_path,
            results_directory=tmp_path / "results",
            benchmark_directory=tmp_path / "benchmarks",
            case_study_path=tmp_path / "case_study.json",
        )
    assert not (tmp_path / "final_report.json").exists()
    assert not (tmp_path / "final_report.md").exists()
