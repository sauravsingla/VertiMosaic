"""External dataset registry and provenance metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    party: str
    name: str
    provider: str
    dataset_id: str
    doi: str | None
    source_url: str
    expected_license: str | None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


DATASETS: dict[str, DatasetSpec] = {
    "bank": DatasetSpec(
        "bank", "bank", "Default of Credit Card Clients", "UCI", "350",
        "10.24432/C55S3H", "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients", "CC BY 4.0",
    ),
    "telecom": DatasetSpec(
        "telecom", "telecom", "Iranian Churn", "UCI", "563",
        "10.24432/C5JW3Z", "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset", "CC BY 4.0",
    ),
    "insurance_freq": DatasetSpec(
        "insurance_freq", "insurance", "freMTPL2freq", "OpenML", "41214", None,
        "https://www.openml.org/d/41214", None,
    ),
    "insurance_sev": DatasetSpec(
        "insurance_sev", "insurance", "freMTPL2sev", "OpenML", "41215", None,
        "https://www.openml.org/d/41215", None,
    ),
    "retail": DatasetSpec(
        "retail", "retail", "Online Retail", "UCI", "352",
        "10.24432/C5BW33", "https://archive.ics.uci.edu/dataset/352/online+retail", "CC BY 4.0",
    ),
}


def list_specs() -> list[DatasetSpec]:
    return list(DATASETS.values())
