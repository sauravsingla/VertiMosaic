from pathlib import Path

from vertimosaic.experiments import run_dropout_study


def test_dropout_study_covers_inference_and_training_time_availability(tmp_path: Path) -> None:
    frame = run_dropout_study(
        rows=320,
        seed=17,
        max_iter=5,
        output=tmp_path / "party_dropout.csv",
    )
    assert set(frame["phase"]) == {
        "inference_time_dropout",
        "training_and_inference_availability",
    }
    assert len(frame) == 10
    assert set(frame["scenario"]) == {
        "none",
        "telecom_absent",
        "insurance_absent",
        "retail_absent",
        "telecom_retail_absent",
    }
    assert (tmp_path / "party_dropout.csv").exists()
    assert frame["estimated_communication_bytes"].ge(0).all()
