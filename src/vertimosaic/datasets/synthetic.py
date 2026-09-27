from __future__ import annotations

import numpy as np

from vertimosaic.parties import ActiveParty, PassiveParty


def make_vertical_synthetic(
    n_rows: int = 2000,
    seed: int = 42,
    n_features_per_party: int = 4,
) -> tuple[ActiveParty, list[PassiveParty]]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_rows, 4))
    xs: list[np.ndarray] = []
    for party_idx in range(4):
        base = latent[:, [party_idx]]
        noise = rng.normal(scale=0.8, size=(n_rows, n_features_per_party))
        x = base @ np.linspace(0.5, 1.2, n_features_per_party)[None, :] + noise
        xs.append(x)
    logit = (
        0.9 * latent[:, 0]
        + 0.8 * latent[:, 1]
        + 0.7 * latent[:, 2]
        + 0.6 * latent[:, 3]
        + 0.35 * latent[:, 0] * latent[:, 1]
        - 0.25 * latent[:, 2] * latent[:, 3]
        + rng.normal(scale=0.5, size=n_rows)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, p).astype(float)
    active = ActiveParty("bank", xs[0], y)
    passive = [
        PassiveParty("telecom", xs[1]),
        PassiveParty("insurance", xs[2]),
        PassiveParty("retail", xs[3]),
    ]
    return active, passive
