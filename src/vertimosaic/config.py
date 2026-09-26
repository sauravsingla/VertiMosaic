from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_MODES = {
    "observed_target_external",
    "distributed_signal_external",
    "synthetic_scale",
    "ieee_cis_linked",
}
_ALLOWED_MODELS = {"logistic", "vfl-hist-gbdt"}
_ALLOWED_MISSING = {
    "intersection_only",
    "zero_contribution",
    "availability_indicator",
    "learned_party_bias",
}


@dataclass(frozen=True)
class ExperimentConfig:
    benchmark_mode: str = "synthetic_scale"
    model: str = "logistic"
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    cross_party_correlation: float = 0.25
    overlap_fraction: float = 1.0
    missing_party_method: str = "intersection_only"
    insurance_sample_size: int | None = None
    bootstrap_replicates: int = 1000

    def __post_init__(self) -> None:
        if self.benchmark_mode not in _ALLOWED_MODES:
            raise ValueError(f"unknown benchmark_mode: {self.benchmark_mode}")
        if self.model not in _ALLOWED_MODELS:
            raise ValueError(f"unknown model: {self.model}")
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train/validation/test fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("all split fractions must be positive")
        if not 0.0 <= self.cross_party_correlation <= 1.0:
            raise ValueError("cross_party_correlation must be in [0, 1]")
        if not 0.0 < self.overlap_fraction <= 1.0:
            raise ValueError("overlap_fraction must be in (0, 1]")
        if self.missing_party_method not in _ALLOWED_MISSING:
            raise ValueError(f"unknown missing_party_method: {self.missing_party_method}")
        if self.insurance_sample_size is not None and self.insurance_sample_size <= 0:
            raise ValueError("insurance_sample_size must be positive")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = yaml.safe_dump(self.to_dict(), sort_keys=True).encode("utf-8")
        return sha256(payload).hexdigest()

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration YAML must contain a mapping")
        return cls(**data)
