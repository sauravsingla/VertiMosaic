"""Target-blind rank/correlation based semi-synthetic cross-domain linkage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _latent_score(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.select_dtypes(include=np.number)
    if numeric.shape[1] == 0:
        return np.zeros(len(frame))
    arr = numeric.to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    arr = np.where(np.isnan(arr), med, arr)
    scale = np.nanstd(arr, axis=0)
    scale[scale < 1e-12] = 1.0
    z = (arr - np.nanmean(arr, axis=0)) / scale
    score = np.mean(z, axis=1)
    ranks = rankdata(score, method="average") / (len(score) + 1.0)
    return norm.ppf(np.clip(ranks, 1e-5, 1 - 1e-5))


@dataclass(frozen=True, slots=True)
class LinkageManifest:
    method: str
    seed: int
    cross_party_correlation: float
    target_blind: bool
    anchor_rows: int
    donor_reuse: dict[str, int]


class CopulaLinker:
    """Align independent public-source rows without pretending they are the same people."""

    def __init__(self, correlation: float = 0.25, seed: int = 42) -> None:
        if not 0 <= correlation < 1:
            raise ValueError("correlation must be in [0, 1)")
        self.correlation = correlation
        self.seed = seed

    def link(
        self,
        anchor: pd.DataFrame,
        donors: dict[str, pd.DataFrame],
        *,
        target_column: str | None = None,
    ) -> tuple[dict[str, pd.DataFrame], LinkageManifest]:
        """Create target-blind donor assignments to anchor entities.

        `target_column` is explicitly excluded from the anchor linkage score.
        """
        rng = np.random.default_rng(self.seed)
        anchor_features = anchor.drop(columns=[target_column], errors="ignore")
        anchor_latent = _latent_score(anchor_features)
        shared = np.sqrt(self.correlation) * anchor_latent + np.sqrt(
            1.0 - self.correlation
        ) * rng.normal(size=len(anchor))
        linked: dict[str, pd.DataFrame] = {}
        reuse: dict[str, int] = {}
        for name, donor in donors.items():
            donor_latent = _latent_score(donor)
            order = np.argsort(donor_latent)
            donor_sorted = donor_latent[order]
            idx = np.searchsorted(donor_sorted, shared, side="left")
            idx = np.clip(idx, 0, len(order) - 1)
            choice = order[idx]
            out = donor.iloc[choice].reset_index(drop=True).copy()
            out.insert(0, "entity_id", anchor["entity_id"].to_numpy())
            linked[name] = out
            counts = np.bincount(choice, minlength=len(donor))
            reuse[name] = int(counts.max(initial=0))
        return linked, LinkageManifest(
            method="rank_gaussian_copula_nearest_donor",
            seed=self.seed,
            cross_party_correlation=self.correlation,
            target_blind=True,
            anchor_rows=len(anchor),
            donor_reuse=reuse,
        )
