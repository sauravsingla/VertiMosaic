from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_ALLOWED_SOURCE_TYPES = {
    "real_external",
    "real_external_aggregated",
    "real_external_derived",
    "semi_synthetic_derived",
    "fully_synthetic",
}


@dataclass(frozen=True)
class FeatureProvenance:
    party: str
    feature: str
    external_dataset: str
    source_column: str
    transformation: str
    source_type: str
    observed_or_derived: str
    semi_synthetic: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if self.source_type not in _ALLOWED_SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if self.observed_or_derived not in {"observed", "derived"}:
            raise ValueError("observed_or_derived must be observed or derived")


def write_feature_provenance(records: list[FeatureProvenance], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "party",
        "feature",
        "external_dataset",
        "source_column",
        "transformation",
        "source_type",
        "observed_or_derived",
        "semi_synthetic",
        "notes",
    ]
    frame = pd.DataFrame([asdict(item) for item in records], columns=columns)
    frame.to_csv(path, index=False)
