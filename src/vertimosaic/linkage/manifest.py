from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinkageManifest:
    method: str
    seed: int
    cross_party_correlation: float
    anchor_rows: int
    donor_rows: int
    unique_donors: int
    donor_reuse_fraction: float
    target_blind: bool
    sampled_with_replacement: bool

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
