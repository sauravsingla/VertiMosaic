from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


def _openml_license_from_api(dataset_id: str, *, timeout_seconds: float = 10.0) -> str | None:
    """Read license metadata from OpenML's official JSON API without downloading data."""
    if not dataset_id.isdigit():
        return None
    url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "VertiMosaic/0.1"})
    try:
        # The URL is restricted above to a fixed HTTPS OpenML host and numeric dataset id.
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            payload = json.load(response)
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    description = payload.get("data_set_description")
    if not isinstance(description, dict):
        return None
    value = description.get("licence") or description.get("license")
    return value.strip() if isinstance(value, str) and value.strip() else None


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

    def runtime_license(self, name: str) -> str | None:
        """Return a verified provider license, querying OpenML when it is runtime-only."""
        record = self.get(name)
        if record.license is not None:
            return record.license
        if record.provider == "OpenML":
            return _openml_license_from_api(record.dataset_id)
        return None

    def verify_license_metadata(self, name: str) -> bool:
        record = self.get(name)
        license_value = self.runtime_license(name)
        if record.provider == "OpenML":
            return license_value is not None
        return license_value is not None and record.license_url is not None
