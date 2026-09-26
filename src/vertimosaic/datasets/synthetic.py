"""Controlled four-party synthetic benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(rows: int = 10_000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    entity_id = np.array([f"e{i:08d}" for i in range(rows)])
    financial = rng.normal(size=rows)
    digital = 0.25 * financial + rng.normal(size=rows)
    claims = 0.15 * financial + rng.normal(size=rows)
    retail_latent = 0.2 * digital + rng.normal(size=rows)

    bank = pd.DataFrame({
        "entity_id": entity_id,
        "payment_pressure": financial + rng.normal(0, 0.5, rows),
        "utilization": 0.7 * financial + rng.normal(0, 0.7, rows),
        "repayment_delay": 0.8 * financial + rng.normal(0, 0.6, rows),
        "balance_volatility": np.abs(financial + rng.normal(0, 0.7, rows)),
    })
    telecom = pd.DataFrame({
        "entity_id": entity_id,
        "usage_change": digital + rng.normal(0, 0.5, rows),
        "call_failure": 0.6 * digital + rng.normal(0, 0.8, rows),
        "service_instability": np.abs(digital + rng.normal(0, 0.5, rows)),
    })
    insurance = pd.DataFrame({
        "entity_id": entity_id,
        "claim_frequency": claims + rng.normal(0, 0.5, rows),
        "bonus_malus": 0.7 * claims + rng.normal(0, 0.7, rows),
        "claim_severity": np.abs(claims + rng.normal(0, 0.8, rows)),
    })
    retail = pd.DataFrame({
        "entity_id": entity_id,
        "spend_volatility": retail_latent + rng.normal(0, 0.5, rows),
        "purchase_velocity": 0.7 * retail_latent + rng.normal(0, 0.7, rows),
        "return_ratio": np.abs(retail_latent + rng.normal(0, 0.7, rows)),
    })
    logit = (
        0.9 * financial + 0.7 * digital + 0.7 * claims + 0.8 * retail_latent
        + 0.35 * digital * retail_latent + rng.normal(0, 0.8, rows) - 3.2
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    bank["target"] = rng.binomial(1, probability)
    return {"bank": bank, "telecom": telecom, "insurance": insurance, "retail": retail}
