from __future__ import annotations

from typing import Any


def interpret_delta(metric: str, delta: float, ci_low: float, ci_high: float) -> str:
    if delta > 0 and ci_low > 0:
        return (
            f"VFL produced a measured improvement of {delta:.6f} in {metric}; "
            "the paired bootstrap interval did not include zero."
        )
    if delta > 0:
        return (
            f"Point performance was higher by {delta:.6f} in {metric}, but this experiment "
            "does not provide clear evidence that the improvement is distinguishable from "
            "sampling variability."
        )
    return f"VFL did not outperform the corresponding baseline on {metric} (delta={delta:.6f})."


def interpret_calibration(brier: float, ece: float) -> str:
    return (
        f"Measured calibration statistics were Brier={brier:.6f} and ECE={ece:.6f}. "
        "These values are descriptive; they are not, by themselves, evidence of "
        "calibrated deployment risk."
    )


def interpret_costs(training_seconds: float | None, payload_bytes: float | None) -> str:
    parts: list[str] = []
    if training_seconds is not None:
        parts.append(f"CPU training wall time was {training_seconds:.3f} seconds")
    if payload_bytes is not None:
        parts.append(
            f"the simulated federated payload estimate was {payload_bytes / (1024 * 1024):.3f} MiB"
        )
    if not parts:
        return "No runtime or communication measurements were supplied for interpretation."
    return "; ".join(parts) + ". Payload estimates are not measurements of real network traffic."


def comparison_observation(comparison: dict[str, Any]) -> str:
    required = {"metric", "delta", "lower", "upper"}
    missing = required - comparison.keys()
    if missing:
        raise ValueError(f"comparison is missing fields: {sorted(missing)}")
    observations = [
        interpret_delta(
            str(comparison["metric"]),
            float(comparison["delta"]),
            float(comparison["lower"]),
            float(comparison["upper"]),
        )
    ]
    secondary_delta = comparison.get("paired_pr_auc_delta")
    secondary_lower = comparison.get("paired_pr_auc_lower")
    secondary_upper = comparison.get("paired_pr_auc_upper")
    if secondary_delta is not None and secondary_lower is not None and secondary_upper is not None:
        observations.append(
            interpret_delta(
                "pr_auc",
                float(secondary_delta),
                float(secondary_lower),
                float(secondary_upper),
            )
        )
    return " ".join(observations)
