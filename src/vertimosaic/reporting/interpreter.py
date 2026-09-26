"""Conservative, data-driven result interpretation."""

from __future__ import annotations


def interpret_delta(metric: str, delta: float, ci_low: float, ci_high: float) -> str:
    if delta > 0 and ci_low > 0:
        return (
            f"VertiMosaic produced a measured improvement of {delta:.4f} in {metric}; "
            "the paired bootstrap interval did not include zero."
        )
    if delta > 0:
        return (
            f"Point performance was higher by {delta:.4f} in {metric}, but the interval "
            f"[{ci_low:.4f}, {ci_high:.4f}] includes zero, so this experiment does not provide "
            "clear evidence that the improvement is distinguishable from sampling variability."
        )
    return (
        f"VertiMosaic did not outperform the corresponding baseline on {metric}; "
        f"the measured delta was {delta:.4f}."
    )
