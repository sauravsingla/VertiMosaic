from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class DatasetRecord:
    party: str
    dataset_name: str
    provider: str
    dataset_id: str
    doi: str | None
    license: str | None
    retrieval_method: str
    provider_url: str
    license_url: str | None = None
    citation: str | None = None
    retrieval_date: str | None = None
    checksum: str | None = None
    raw_rows: int | None = None
    processed_rows: int | None = None


REGISTRY: dict[str, DatasetRecord] = {
    "bank": DatasetRecord(
        party="bank",
        dataset_name="Default of Credit Card Clients",
        provider="UCI Machine Learning Repository",
        dataset_id="350",
        doi="10.24432/C55S3H",
        license="CC BY 4.0",
        retrieval_method="ucimlrepo.fetch_ucirepo(id=350)",
        provider_url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Yeh, I-Cheng. Default of Credit Card Clients. UCI Machine Learning Repository.",
    ),
    "telecom": DatasetRecord(
        party="telecom",
        dataset_name="Iranian Churn",
        provider="UCI Machine Learning Repository",
        dataset_id="563",
        doi="10.24432/C5JW3Z",
        license="CC BY 4.0",
        retrieval_method="ucimlrepo.fetch_ucirepo(id=563)",
        provider_url="https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Iranian Churn Dataset. UCI Machine Learning Repository.",
    ),
    "insurance_freq": DatasetRecord(
        party="insurance",
        dataset_name="freMTPL2freq",
        provider="OpenML",
        dataset_id="41214",
        doi=None,
        license=None,
        retrieval_method="sklearn.datasets.fetch_openml(data_id=41214)",
        provider_url="https://www.openml.org/d/41214",
        citation="French Motor Third-Party Liability frequency data via OpenML.",
    ),
    "insurance_sev": DatasetRecord(
        party="insurance",
        dataset_name="freMTPL2sev",
        provider="OpenML",
        dataset_id="41215",
        doi=None,
        license=None,
        retrieval_method="sklearn.datasets.fetch_openml(data_id=41215)",
        provider_url="https://www.openml.org/d/41215",
        citation="French Motor Third-Party Liability severity data via OpenML.",
    ),
    "retail": DatasetRecord(
        party="retail",
        dataset_name="Online Retail",
        provider="UCI Machine Learning Repository",
        dataset_id="352",
        doi="10.24432/C5BW33",
        license="CC BY 4.0",
        retrieval_method="ucimlrepo.fetch_ucirepo(id=352)",
        provider_url="https://archive.ics.uci.edu/dataset/352/online+retail",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        citation="Chen, D. Online Retail. UCI Machine Learning Repository.",
    ),
}


class DatasetRegistry:
    def list(self) -> list[dict[str, str | int | None]]:
        return [asdict(record) for record in REGISTRY.values()]

    def get(self, name: str) -> DatasetRecord:
        try:
            return REGISTRY[name]
        except KeyError as exc:
            raise KeyError(f"unknown dataset: {name}") from exc

    def describe(self, name: str) -> dict[str, str | int | None]:
        return asdict(self.get(name))

    def with_retrieval_date(self, name: str) -> dict[str, str | int | None]:
        data = self.describe(name)
        data["retrieval_date"] = date.today().isoformat()
        return data

    def verify_license_metadata(self, name: str) -> bool:
        record = self.get(name)
        if record.provider == "OpenML":
            return record.license is not None
        return record.license is not None and record.license_url is not None
