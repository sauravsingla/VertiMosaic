from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vertimosaic.alignment import EntityAligner
from vertimosaic.evaluation import communication_event_frame
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.provenance import FeatureProvenance
from vertimosaic.reproducibility import RunArtifacts


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _synthetic_dataset_hashes(
    active: ActiveParty,
    passive: list[PassiveParty],
) -> dict[str, str]:
    hashes = {
        "bank_features": _array_sha256(active._x),
        "bank_labels": _array_sha256(active.labels),
    }
    hashes.update({f"{party.name}_features": _array_sha256(party._x) for party in passive})
    return hashes


def _synthetic_feature_provenance(
    active: ActiveParty,
    passive: list[PassiveParty],
) -> pd.DataFrame:
    records: list[FeatureProvenance] = []
    for party in [active, *passive]:
        for index in range(party.n_features):
            records.append(
                FeatureProvenance(
                    party=party.name,
                    feature=f"feature_{index}",
                    external_dataset="synthetic_scale",
                    source_column=f"synthetic_column_{index}",
                    transformation="controlled synthetic generator",
                    source_type="fully_synthetic",
                    observed_or_derived="derived",
                    semi_synthetic=False,
                )
            )
    return pd.DataFrame([asdict(item) for item in records])


def prediction_frame(
    entity_indices: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    *,
    seed: int,
    condition: str,
) -> pd.DataFrame:
    """Create a pseudonymous per-condition prediction frame for a measured study."""
    aligner = EntityAligner(salt=f"vertimosaic-study-{seed}")
    return pd.DataFrame(
        {
            "entity_id": [aligner.pseudonymize(str(index)) for index in entity_indices],
            "condition": condition,
            "target": target,
            "probability": probability,
        }
    )


def model_training_frame(
    model: VFLLogisticRegression | VFLHistGBDT,
    *,
    condition: str,
) -> pd.DataFrame:
    """Return convergence telemetry tagged with the study condition."""
    if isinstance(model, VFLLogisticRegression):
        frame = pd.DataFrame(
            {"iteration": np.arange(len(model.loss_history_)), "loss": model.loss_history_}
        )
        if model.validation_loss_history_:
            values = model.validation_loss_history_[: len(frame)]
            frame.loc[: len(values) - 1, "validation_loss"] = values
    else:
        frame = pd.DataFrame(
            {
                "tree": np.arange(len(model.training_loss_history_)),
                "training_loss": model.training_loss_history_,
            }
        )
        if model.validation_loss_history_:
            frame["validation_loss"] = model.validation_loss_history_[: len(frame)]
    frame.insert(0, "condition", condition)
    return frame


def model_communication_frame(
    model: VFLLogisticRegression | VFLHistGBDT,
    *,
    condition: str,
) -> pd.DataFrame:
    """Return metadata-only simulated communication events tagged by condition."""
    frame = communication_event_frame(model.transport.audit_log)
    frame.insert(0, "condition", condition)
    return frame


def write_synthetic_study_run(
    *,
    study_name: str,
    seed: int,
    active: ActiveParty,
    passive: list[PassiveParty],
    config: dict[str, Any],
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    training_history: pd.DataFrame,
    communication: pd.DataFrame,
    runs_root: Path = Path("runs"),
) -> tuple[str, Path]:
    """Persist one measured synthetic study using the standard run-bundle contract."""
    if predictions.empty:
        raise ValueError("study reproducibility requires measured predictions")
    run = RunArtifacts.create(root=runs_root)
    metrics = {
        "mode": "synthetic_scale",
        "study": study_name,
        "seed": seed,
        "results": results.to_dict(orient="records"),
    }
    directory = run.finalize(
        config={"study": study_name, "seed": seed, **config},
        seed=seed,
        dataset_provenance={
            "source_type": "fully_synthetic",
            "rows": active.n_rows,
            "dataset_hashes": _synthetic_dataset_hashes(active, passive),
        },
        linkage_manifest={
            "method": "shared synthetic latent entities",
            "target_blind": True,
        },
        metrics=metrics,
        predictions=predictions,
        training_history=training_history,
        communication=communication,
        feature_provenance=_synthetic_feature_provenance(active, passive),
    )
    return run.run_id, directory
