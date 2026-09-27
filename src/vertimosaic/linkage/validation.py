from __future__ import annotations

from collections.abc import Sequence


def validate_target_blind_inputs(feature_names: Sequence[str], target_name: str | None) -> None:
    """Reject accidental inclusion of the target in linkage inputs."""
    if target_name is None:
        return
    normalized = {name.strip().casefold() for name in feature_names}
    if target_name.strip().casefold() in normalized:
        raise ValueError("target column must not be used during target-blind linkage")
