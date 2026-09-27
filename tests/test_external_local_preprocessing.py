from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.external import ExternalBenchmark
from vertimosaic.experiments.external_preprocessing import prepare_external_splits_locally
from vertimosaic.parties import ActiveParty, PassiveParty


def test_external_preprocessing_is_party_local_and_persisted(tmp_path: Path) -> None:
    rows = 40
    labels = np.tile(np.array([0.0, 1.0]), rows // 2)
    frames = {
        name: pd.DataFrame(
            {
                "numeric": np.arange(rows, dtype=float) + offset,
                "category": ["a" if index % 3 else "b" for index in range(rows)],
            }
        )
        for offset, name in enumerate(("bank", "telecom", "insurance", "retail"))
    }
    benchmark = ExternalBenchmark(
        active=ActiveParty("bank", np.zeros((rows, 1)), labels),
        passive=[
            PassiveParty("telecom", np.zeros((rows, 1))),
            PassiveParty("insurance", np.zeros((rows, 1))),
            PassiveParty("retail", np.zeros((rows, 1))),
        ],
        linkage_manifests={},
        mode="observed_target_external",
        feature_frames=frames,
    )
    split = entity_level_split(labels, seed=9)
    prepared = prepare_external_splits_locally(
        benchmark,
        split,
        artifact_directory=tmp_path / "artifacts",
    )

    assert prepared.train_active.n_rows == len(split.train)
    assert prepared.validation_active.n_rows == len(split.validation)
    assert prepared.test_active.n_rows == len(split.test)
    assert [party.name for party in prepared.train_passive] == [
        "telecom",
        "insurance",
        "retail",
    ]
    assert set(prepared.preprocessor_paths) == {"bank", "telecom", "insurance", "retail"}
    for party in ("bank", "telecom", "insurance", "retail"):
        assert (tmp_path / "artifacts" / f"{party}_preprocessor.joblib").exists()
