from __future__ import annotations


def interpret_delta(metric: str, delta: float, ci_low: float, ci_high: float) -> str:
    if delta > 0 and ci_low > 0:
        return f"VFL produced a measured improvement of {delta:.6f} in {metric}; the paired bootstrap interval did not include zero."
    if delta > 0:
        return f"Point performance was higher by {delta:.6f} in {metric}, but this experiment does not provide clear evidence that the improvement is distinguishable from sampling variability."
    return f"VFL did not outperform the corresponding baseline on {metric} (delta={delta:.6f})."
