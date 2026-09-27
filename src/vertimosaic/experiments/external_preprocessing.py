from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.evaluation import SplitIndices
from vertimosaic.experiments.external import ExternalBenchmark
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.preprocessing import LocalTabularPreprocessor


@dataclass(frozen=True)
class ExternalPreparedSplits:
    train_active: ActiveParty
    train_passive: list[PassiveParty]
    validation_active: ActiveParty
    validation_passive: list[PassiveParty]
    test_active: ActiveParty
    test_passive: list[PassiveParty]
    preprocessing_seconds: float
    preprocessor_paths: dict[str, str]
    feature_metadata: dict[str, list[dict[str, str]]]


def _column_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in frame.columns if name not in numeric]
    return numeric, categorical


def prepare_external_splits_locally(
    benchmark: ExternalBenchmark,
    split: SplitIndices,
    *,
    artifact_directory: Path = Path("artifacts"),
) -> ExternalPreparedSplits:
    """Fit one preprocessor per party on TRAIN only and transform all entity splits."""
    transformed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    paths: dict[str, str] = {}
    feature_metadata: dict[str, list[dict[str, str]]] = {}
    start = time.perf_counter()

    for party in ("bank", "telecom", "insurance", "retail"):
        frame = benchmark.feature_frames[party].reset_index(drop=True)
        numeric, categorical = _column_types(frame)
        preprocessor = LocalTabularPreprocessor(
            numeric_columns=numeric,
            categorical_columns=categorical,
            scaling="robust",
        )
        train_frame = frame.iloc[split.train]
        validation_frame = frame.iloc[split.validation]
        test_frame = frame.iloc[split.test]
        train_x = preprocessor.fit_transform(train_frame)
        validation_x = preprocessor.transform(validation_frame)
        test_x = preprocessor.transform(test_frame)
        path = artifact_directory / f"{party}_preprocessor.joblib"
        preprocessor.save(path)
        transformed[party] = (train_x, validation_x, test_x)
        paths[party] = str(path)
        feature_metadata[party] = preprocessor.output_feature_metadata()

    preprocessing_seconds = time.perf_counter() - start
    bank_train, bank_validation, bank_test = transformed["bank"]
    train_active = ActiveParty("bank", bank_train, benchmark.active.labels[split.train])
    validation_active = ActiveParty(
        "bank", bank_validation, benchmark.active.labels[split.validation]
    )
    test_active = ActiveParty("bank", bank_test, benchmark.active.labels[split.test])

    train_passive: list[PassiveParty] = []
    validation_passive: list[PassiveParty] = []
    test_passive: list[PassiveParty] = []
    for party in ("telecom", "insurance", "retail"):
        train_x, validation_x, test_x = transformed[party]
        train_passive.append(PassiveParty(party, train_x))
        validation_passive.append(PassiveParty(party, validation_x))
        test_passive.append(PassiveParty(party, test_x))

    return ExternalPreparedSplits(
        train_active=train_active,
        train_passive=train_passive,
        validation_active=validation_active,
        validation_passive=validation_passive,
        test_active=test_active,
        test_passive=test_passive,
        preprocessing_seconds=preprocessing_seconds,
        preprocessor_paths=paths,
        feature_metadata=feature_metadata,
    )
