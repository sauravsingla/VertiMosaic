from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from vertimosaic.datasets import ExternalDatasetBundle, fetch_external_party
from vertimosaic.linkage import GaussianCopulaLinker, LinkageManifest
from vertimosaic.parties import ActiveParty, PassiveParty


@dataclass(frozen=True)
class ExternalBenchmark:
    active: ActiveParty
    passive: list[PassiveParty]
    linkage_manifests: dict[str, LinkageManifest]
    mode: str


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Locally turn one party's prepared frame into a finite numeric research matrix."""
    columns: list[np.ndarray] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        else:
            values = pd.factorize(series.astype("string"), sort=True)[0].astype(float)
        finite = np.isfinite(values)
        fill = float(np.median(values[finite])) if finite.any() else 0.0
        columns.append(np.where(finite, values, fill))
    if not columns:
        raise ValueError("prepared party frame contains no usable features")
    return np.column_stack(columns)


def _link_passive(
    anchor_matrix: np.ndarray,
    bundle: ExternalDatasetBundle,
    *,
    correlation: float,
    seed: int,
) -> tuple[np.ndarray, LinkageManifest]:
    donor_matrix = _numeric_matrix(bundle.features)
    result = GaussianCopulaLinker(cross_party_correlation=correlation, seed=seed).link(
        anchor_matrix, donor_matrix
    )
    linked = donor_matrix[result.donor_indices]
    manifest = LinkageManifest(
        method="gaussian_copula_rank_proximity",
        seed=seed,
        cross_party_correlation=correlation,
        anchor_rows=result.anchor_rows,
        donor_rows=result.donor_rows,
        unique_donors=result.unique_donors,
        donor_reuse_fraction=result.donor_reuse_fraction,
        target_blind=True,
        sampled_with_replacement=result.unique_donors < result.anchor_rows,
    )
    return linked, manifest


def _standardized_signal(values: np.ndarray) -> np.ndarray:
    signal = np.mean(values, axis=1)
    std = float(signal.std())
    return (signal - float(signal.mean())) / (std if std > 1e-12 else 1.0)


def prepare_external_benchmark(
    *,
    mode: str = "observed_target_external",
    cross_party_correlation: float = 0.25,
    seed: int = 42,
    insurance_sample_size: int | None = None,
    bundles: dict[str, ExternalDatasetBundle] | None = None,
) -> ExternalBenchmark:
    """Build the explicitly semi-synthetic four-industry external VFL benchmark."""
    if mode not in {"observed_target_external", "distributed_signal_external"}:
        raise ValueError(
            "external benchmark mode must be observed_target_external "
            "or distributed_signal_external"
        )
    if bundles is None:
        bundles = {
            name: fetch_external_party(name, insurance_sample_size=insurance_sample_size, seed=seed)
            for name in ("bank", "telecom", "insurance", "retail")
        }
    bank = bundles["bank"]
    bank_matrix = _numeric_matrix(bank.features)
    if bank.target is None:
        raise ValueError("Bank external bundle must contain the observed target")
    observed_target = np.asarray(bank.target, dtype=float).reshape(-1)
    if len(observed_target) != len(bank_matrix):
        raise ValueError("Bank target and prepared features must have equal row count")
    linked: dict[str, np.ndarray] = {}
    manifests: dict[str, LinkageManifest] = {}
    for offset, name in enumerate(("telecom", "insurance", "retail"), start=1):
        linked[name], manifests[name] = _link_passive(
            bank_matrix,
            bundles[name],
            correlation=cross_party_correlation,
            seed=seed + offset,
        )
    if mode == "observed_target_external":
        y = observed_target
    else:
        rng = np.random.default_rng(seed)
        bank_signal = _standardized_signal(bank_matrix)
        telecom_signal = _standardized_signal(linked["telecom"])
        insurance_signal = _standardized_signal(linked["insurance"])
        retail_signal = _standardized_signal(linked["retail"])
        logit = (
            0.8 * bank_signal
            + 0.7 * telecom_signal
            + 0.65 * insurance_signal
            + 0.6 * retail_signal
            + 0.25 * bank_signal * telecom_signal
            - 0.20 * insurance_signal * retail_signal
            + rng.normal(scale=0.6, size=len(bank_signal))
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0)))
        y = rng.binomial(1, probability).astype(float)
    return ExternalBenchmark(
        active=ActiveParty("bank", bank_matrix, y),
        passive=[
            PassiveParty("telecom", linked["telecom"]),
            PassiveParty("insurance", linked["insurance"]),
            PassiveParty("retail", linked["retail"]),
        ],
        linkage_manifests=manifests,
        mode=mode,
    )


def linkage_manifest_dict(benchmark: ExternalBenchmark) -> dict[str, dict[str, object]]:
    return {name: asdict(manifest) for name, manifest in benchmark.linkage_manifests.items()}
