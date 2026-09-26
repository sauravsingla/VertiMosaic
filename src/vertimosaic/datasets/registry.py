from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetRecord:
    party: str
    dataset_name: str
    provider: str
    dataset_id: str
    doi: str | None
    license: str | None
    retrieval_method: str


REGISTRY: dict[str, DatasetRecord] = {
    "bank": DatasetRecord("bank", "Default of Credit Card Clients", "UCI", "350", "10.24432/C55S3H", "CC BY 4.0", "ucimlrepo"),
    "telecom": DatasetRecord("telecom", "Iranian Churn", "UCI", "563", "10.24432/C5JW3Z", "CC BY 4.0", "ucimlrepo"),
    "insurance_freq": DatasetRecord("insurance", "freMTPL2freq", "OpenML", "41214", None, None, "sklearn.fetch_openml"),
    "insurance_sev": DatasetRecord("insurance", "freMTPL2sev", "OpenML", "41215", None, None, "sklearn.fetch_openml"),
    "retail": DatasetRecord("retail", "Online Retail", "UCI", "352", "10.24432/C5BW33", "CC BY 4.0", "ucimlrepo"),
}


class DatasetRegistry:
    def list(self) -> list[dict[str, str | None]]:
        return [asdict(record) for record in REGISTRY.values()]

    def get(self, name: str) -> DatasetRecord:
        try:
            return REGISTRY[name]
        except KeyError as exc:
            raise KeyError(f"unknown dataset: {name}") from exc
