"""Feature provenance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    party: str
    feature: str
    external_dataset: str | None
    source_column: str | None
    transformation: str
    source_type: str
    observed_or_derived: str
    semi_synthetic: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
