from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vertimosaic.reporting.interpreter import (
    comparison_observation,
    interpret_calibration,
    interpret_costs,
)


def build_data_driven_report(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    comparisons = payload.get("comparisons", [])
    observations = [comparison_observation(item) for item in comparisons]
    if "brier" in metrics and "ece" in metrics:
        observations.append(interpret_calibration(float(metrics["brier"]), float(metrics["ece"])))
    observations.append(
        interpret_costs(
            float(payload["training_seconds"]) if "training_seconds" in payload else None,
            float(payload["estimated_communication_bytes"])
            if "estimated_communication_bytes" in payload
            else None,
        )
    )
    return {
        "observations": observations,
        "interpretations": [
            "Conclusions are restricted to measured outputs from the supplied run artifacts.",
            "Negative or uncertain comparisons are retained rather than rewritten as improvements.",
        ],
        "limitations": [
            "Raw-feature locality does not imply cryptographic confidentiality.",
            (
                "The four-industry external benchmark uses explicitly semi-synthetic "
                "cross-domain linkage."
            ),
            "Simulated payload bytes are not real network-traffic measurements.",
        ],
        "source": payload,
    }


def write_final_report(
    payload: dict[str, Any], directory: Path = Path("reports")
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    report = build_data_driven_report(payload)
    json_path = directory / "final_report.json"
    md_path = directory / "final_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# VertiMosaic measured-result report", "", "## Observations", ""]
    lines.extend(f"- {item}" for item in report["observations"])
    lines.extend(["", "## Interpretations", ""])
    lines.extend(f"- {item}" for item in report["interpretations"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
