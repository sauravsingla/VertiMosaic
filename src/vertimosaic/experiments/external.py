from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from vertimosaic.datasets import DatasetRegistry, ExternalDatasetBundle, fetch_external_party
from vertimosaic.linkage import GaussianCopulaLinker, LinkageManifest
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.provenance import FeatureProvenance


@dataclass(frozen=True)
class ExternalBenchmark:
    active: ActiveParty
    passive: list[PassiveParty]
    linkage_manifests: dict[str, LinkageManifest]
    mode: str
    feature_frames: dict[str, pd.DataFrame]
    source_metadata: dict[str, dict[str, Any]]
    source_provenance: dict[str, list[FeatureProvenance]]
    entity_alignment_seconds: float


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Create a finite numeric linkage representation, not a fitted model preprocessor."""
    columns: list[np.ndarray] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, 0.0)
        else:
            text = series.astype("string").fillna("<missing>")
            hashed = pd.util.hash_pandas_object(text, index=False).to_numpy(dtype=np.uint64)
            values = (hashed % np.uint64(1_000_003)).astype(float)
        columns.append(values)
    if not columns:
        raise ValueError("prepared party frame contains no usable features")
    return np.column_stack(columns)


def _link_passive(
    anchor_matrix: np.ndarray,
    bundle: ExternalDatasetBundle,
    *,
    correlation: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, LinkageManifest]:
    donor_matrix = _numeric_matrix(bundle.features)
    result = GaussianCopulaLinker(cross_party_correlation=correlation, seed=seed).link(
        anchor_matrix, donor_matrix
    )
    linked_frame = bundle.features.iloc[result.donor_indices].reset_index(drop=True)
    linked_matrix = donor_matrix[result.donor_indices]
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
    return linked_frame, linked_matrix, manifest


def _standardized_signal(values: np.ndarray) -> np.ndarray:
    signal = np.mean(values, axis=1)
    std = float(signal.std())
    return (signal - float(signal.mean())) / (std if std > 1e-12 else 1.0)


def _verify_fetched_license_metadata(bundles: dict[str, ExternalDatasetBundle]) -> None:
    """Require verifiable provider license metadata before external modelling."""
    registry = DatasetRegistry()
    for name in ("bank", "telecom", "retail"):
        if not registry.verify_license_metadata(name):
            raise RuntimeError(f"license metadata could not be verified for {name}")
    insurance_license = bundles["insurance"].metadata.get("license")
    if not isinstance(insurance_license, str) or not insurance_license.strip():
        raise RuntimeError(
            "OpenML insurance license metadata could not be verified; "
            "external modelling is stopped conservatively"
        )


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
    _verify_fetched_license_metadata(bundles)
    bank = bundles["bank"]
    bank_frame = bank.features.reset_index(drop=True)
    bank_matrix = _numeric_matrix(bank_frame)
    if bank.target is None:
        raise ValueError("Bank external bundle must contain the observed target")
    observed_target = np.asarray(bank.target, dtype=float).reshape(-1)
    if len(observed_target) != len(bank_matrix):
        raise ValueError("Bank target and prepared features must have equal row count")

    linked_frames: dict[str, pd.DataFrame] = {}
    linked_matrices: dict[str, np.ndarray] = {}
    manifests: dict[str, LinkageManifest] = {}
    alignment_start = time.perf_counter()
    for offset, name in enumerate(("telecom", "insurance", "retail"), start=1):
        linked_frames[name], linked_matrices[name], manifests[name] = _link_passive(
            bank_matrix,
            bundles[name],
            correlation=cross_party_correlation,
            seed=seed + offset,
        )
    entity_alignment_seconds = time.perf_counter() - alignment_start

    if mode == "observed_target_external":
        y = observed_target
    else:
        rng = np.random.default_rng(seed)
        bank_signal = _standardized_signal(bank_matrix)
        telecom_signal = _standardized_signal(linked_matrices["telecom"])
        insurance_signal = _standardized_signal(linked_matrices["insurance"])
        retail_signal = _standardized_signal(linked_matrices["retail"])
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

    feature_frames = {
        "bank": bank_frame,
        "telecom": linked_frames["telecom"],
        "insurance": linked_frames["insurance"],
        "retail": linked_frames["retail"],
    }
    return ExternalBenchmark(
        active=ActiveParty("bank", bank_matrix, y),
        passive=[
            PassiveParty("telecom", linked_matrices["telecom"]),
            PassiveParty("insurance", linked_matrices["insurance"]),
            PassiveParty("retail", linked_matrices["retail"]),
        ],
        linkage_manifests=manifests,
        mode=mode,
        feature_frames=feature_frames,
        source_metadata={name: dict(bundle.metadata) for name, bundle in bundles.items()},
        source_provenance={name: list(bundle.provenance) for name, bundle in bundles.items()},
        entity_alignment_seconds=entity_alignment_seconds,
    )


def linkage_manifest_dict(benchmark: ExternalBenchmark) -> dict[str, dict[str, object]]:
    return {name: asdict(manifest) for name, manifest in benchmark.linkage_manifests.items()}
