from __future__ import annotations

import json
from pathlib import Path
from typing import SupportsFloat, SupportsIndex, cast

from vertimosaic.experiments.external_run import run_external_experiment


def _numeric(value: object) -> float:
    """Narrow measured scalar values from the generic case-study payload."""
    return float(cast(str | SupportsFloat | SupportsIndex, value))


def run_distributed_signal_case_study(
    *,
    seed: int = 42,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    bootstrap_replicates: int = 1000,
    runs_root: Path = Path("runs"),
    output_json: Path = Path("reports/case_study.json"),
    output_markdown: Path = Path("reports/case_study.md"),
) -> dict[str, object]:
    """Generate a measured sanitized case study plus the standard experiment run bundle."""
    payload = run_external_experiment(
        mode="distributed_signal_external",
        model_name="logistic",
        seed=seed,
        cross_party_correlation=cross_party_correlation,
        insurance_sample_size=insurance_sample_size,
        bootstrap_replicates=bootstrap_replicates,
        write_run=True,
        runs_root=runs_root,
    )
    raw_case = payload.get("case_study")
    if not isinstance(raw_case, dict):
        raise RuntimeError("distributed-signal external run did not produce a case study")
    result: dict[str, object] = dict(raw_case)
    result["benchmark_mode"] = "distributed_signal_external"
    result["cross_party_correlation"] = cross_party_correlation
    result["raw_passive_features_exposed"] = False
    result["preprocessing_fit_scope"] = "TRAIN only, independently per party"
    result["preprocessor_artifacts"] = payload["preprocessor_artifacts"]
    result["run_directory"] = payload["run_directory"]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contributions = result["contribution_summary"]
    if not isinstance(contributions, dict):
        raise TypeError("case-study contribution summary must be a mapping")
    lines = [
        "# Sanitized model-generated case study",
        "",
        f"Entity: `{result['entity_id']}`",
        f"Bank-only risk: {_numeric(result['bank_only_risk']):.6f}",
        f"Four-party VFL risk: {_numeric(result['four_party_vfl_risk']):.6f}",
        f"Validation-selected threshold: {_numeric(result['threshold']):.6f}",
        "",
        "## Predictive contribution summary",
        "",
    ]
    for party, value in contributions.items():
        lines.append(f"- {party}: {_numeric(value):.6f}")
    lines.extend(
        [
            "",
            "These are party-representation permutation probability deltas, not causal importance.",
            "No passive raw feature values are included in this report.",
            f"Full reproducibility bundle: `{result['run_directory']}`",
        ]
    )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
