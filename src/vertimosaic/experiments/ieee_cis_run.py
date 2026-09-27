from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from vertimosaic.alignment import EntityAligner
from vertimosaic.baselines import fit_centralized_baseline
from vertimosaic.evaluation import (
    binary_metrics,
    bootstrap_confidence_intervals,
    communication_event_frame,
    communication_totals,
    confusion_at_threshold,
    entity_level_split,
    paired_bootstrap_difference,
    select_f1_threshold,
)
from vertimosaic.experiments.ieee_cis import prepare_ieee_cis
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.preprocessing import LocalTabularPreprocessor
from vertimosaic.provenance import FeatureProvenance
from vertimosaic.reproducibility import RunArtifacts, file_sha256

_IEEE_CIS_PROVIDER_URL = "https://www.kaggle.com/c/ieee-fraud-detection"
_IEEE_CIS_RULES_URL = "https://www.kaggle.com/c/ieee-fraud-detection/rules"


def _column_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]
    categorical = [name for name in frame.columns if name not in numeric]
    return numeric, categorical


def _model(name: str, seed: int) -> VFLLogisticRegression | VFLHistGBDT:
    if name == "logistic":
        return VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=500,
            l2=1e-3,
            early_stopping_rounds=5,
            seed=seed,
        )
    if name == "vfl-hist-gbdt":
        return VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping_rounds=3,
            seed=seed,
        )
    raise ValueError("model_name must be logistic or vfl-hist-gbdt")


def _training_frame(model: VFLLogisticRegression | VFLHistGBDT) -> pd.DataFrame:
    if isinstance(model, VFLLogisticRegression):
        frame = pd.DataFrame(
            {"iteration": np.arange(len(model.loss_history_)), "loss": model.loss_history_}
        )
        if model.validation_loss_history_:
            values = model.validation_loss_history_[: len(frame)]
            frame.loc[: len(values) - 1, "validation_loss"] = values
        return frame
    frame = pd.DataFrame(
        {
            "tree": np.arange(len(model.training_loss_history_)),
            "training_loss": model.training_loss_history_,
        }
    )
    if model.validation_loss_history_:
        frame["validation_loss"] = model.validation_loss_history_[: len(frame)]
    return frame


def _preprocess_party(
    frame: pd.DataFrame,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    artifact_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    numeric, categorical = _column_types(frame)
    preprocessor = LocalTabularPreprocessor(
        numeric_columns=numeric,
        categorical_columns=categorical,
        scaling="robust",
    )
    train_x = preprocessor.fit_transform(frame.iloc[train])
    validation_x = preprocessor.transform(frame.iloc[validation])
    test_x = preprocessor.transform(frame.iloc[test])
    preprocessor.save(artifact_path)
    return train_x, validation_x, test_x, preprocessor.output_feature_metadata()


def _feature_provenance(
    party: str,
    metadata: list[dict[str, str]],
) -> list[FeatureProvenance]:
    return [
        FeatureProvenance(
            party=party,
            feature=item["feature"],
            external_dataset="IEEE-CIS Fraud Detection",
            source_column=item["source_column"],
            transformation=item["transformation"],
            source_type="real_external",
            observed_or_derived="derived",
            semi_synthetic=False,
            notes=(
                "authorized local Kaggle competition data; VertiMosaic does not redistribute "
                "source rows or raw identifiers"
            ),
        )
        for item in metadata
    ]


def _source_record(
    *,
    name: str,
    dataset_id: str,
    path: Path,
    raw_rows: int,
    processed_rows: int,
) -> dict[str, Any]:
    return {
        "dataset_name": name,
        "provider": "Kaggle / IEEE-CIS Fraud Detection competition",
        "dataset_id": dataset_id,
        "doi": None,
        "provider_url": _IEEE_CIS_PROVIDER_URL,
        "license": "Kaggle competition rules; user authorization required",
        "license_url": _IEEE_CIS_RULES_URL,
        "citation": "IEEE-CIS Fraud Detection competition dataset",
        "retrieval_method": "user-provided authorized local file",
        "retrieval_date": date.today().isoformat(),
        "checksum": file_sha256(path),
        "checksum_algorithm": "sha256",
        "checksum_scope": "authorized_local_file_bytes",
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "local_filename": path.name,
        "redistributed_by_vertimosaic": False,
    }


def run_ieee_cis_experiment(
    transaction_path: Path,
    identity_path: Path,
    *,
    model_name: str = "logistic",
    seed: int = 42,
    bootstrap_replicates: int = 1000,
    artifact_directory: Path = Path("artifacts"),
    runs_root: Path = Path("runs"),
    output: Path | None = None,
) -> dict[str, Any]:
    """Run the optional genuinely linked two-party IEEE-CIS VFL benchmark."""
    preparation_start = time.perf_counter()
    prepared = prepare_ieee_cis(transaction_path, identity_path)
    total_preparation_seconds = time.perf_counter() - preparation_start
    data_preparation_seconds = max(
        0.0,
        total_preparation_seconds - prepared.entity_alignment_seconds,
    )
    if len(prepared.target) < 200:
        raise ValueError("IEEE-CIS linked benchmark requires at least 200 overlapping rows")
    if prepared.target.nunique() < 2:
        raise ValueError("IEEE-CIS linked benchmark target must contain both classes")

    split = entity_level_split(prepared.target.to_numpy(dtype=float), seed=seed)
    preprocessing_start = time.perf_counter()
    tx_train, tx_validation, tx_test, tx_feature_metadata = _preprocess_party(
        prepared.transaction.reset_index(drop=True),
        train=split.train,
        validation=split.validation,
        test=split.test,
        artifact_path=artifact_directory / "ieee_cis_transaction_preprocessor.joblib",
    )
    id_train, id_validation, id_test, id_feature_metadata = _preprocess_party(
        prepared.identity.reset_index(drop=True),
        train=split.train,
        validation=split.validation,
        test=split.test,
        artifact_path=artifact_directory / "ieee_cis_identity_preprocessor.joblib",
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    labels = prepared.target.to_numpy(dtype=float)
    train_active = ActiveParty("transaction", tx_train, labels[split.train])
    validation_active = ActiveParty("transaction", tx_validation, labels[split.validation])
    test_active = ActiveParty("transaction", tx_test, labels[split.test])
    train_passive = [PassiveParty("identity", id_train)]
    validation_passive = [PassiveParty("identity", id_validation)]
    test_passive = [PassiveParty("identity", id_test)]

    model = _model(model_name, seed)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    training_start = time.perf_counter()
    model.fit(train_active, train_passive, validation_active, validation_passive)
    training_seconds = time.perf_counter() - training_start
    peak_rss_bytes = max(rss_before, process.memory_info().rss)

    validation_probability = model.predict_proba([validation_active, *validation_passive])[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_probability)
    inference_start = time.perf_counter()
    test_probability = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    metrics = binary_metrics(test_active.labels, test_probability, threshold=threshold)
    intervals = bootstrap_confidence_intervals(
        test_active.labels,
        test_probability,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    baseline_name = "logistic" if model_name == "logistic" else "hist-gbdt"
    transaction_baseline = fit_centralized_baseline(
        [tx_train],
        train_active.labels,
        [tx_test],
        model=baseline_name,
        seed=seed,
    )
    centralized_baseline = fit_centralized_baseline(
        [tx_train, id_train],
        train_active.labels,
        [tx_test, id_test],
        model=baseline_name,
        seed=seed,
    )
    transaction_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_probability,
        transaction_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    transaction_comparison["baseline"] = "transaction_only_non_federated"
    centralized_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_probability,
        centralized_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    centralized_comparison["baseline"] = "centralized_transaction_plus_identity_non_federated"

    communication = communication_totals(model.transport.audit_log)
    training_steps = (
        model.n_iter_ if isinstance(model, VFLLogisticRegression) else len(model.trees_)
    )
    overlap_rows = len(prepared.target)
    overlap_coverage = overlap_rows / max(prepared.transaction_source_rows, 1)
    payload: dict[str, Any] = {
        "benchmark_description": "genuinely linked authorized-local IEEE-CIS VFL sanity benchmark",
        "mode": "ieee_cis_linked",
        "parties": ["transaction", "identity"],
        "four_industry_benchmark": False,
        "target_description": "observed IEEE-CIS isFraud competition target",
        "model": model_name,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "overlap_rows": overlap_rows,
        "overlap_coverage_against_transaction_rows": overlap_coverage,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "comparisons": [transaction_comparison, centralized_comparison],
        "confusion_matrix": confusion_at_threshold(
            test_active.labels,
            test_probability,
            threshold,
        ),
        "threshold_selected_on_validation": threshold,
        "data_preparation_seconds": data_preparation_seconds,
        "entity_alignment_seconds": prepared.entity_alignment_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "training_steps": training_steps,
        "peak_rss_bytes": int(peak_rss_bytes),
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication": communication,
        "preprocessing_fit_scope": "TRAIN only, independently per party",
        "raw_source_ids_exported": False,
        "source_data_redistributed": False,
    }

    transaction_source = _source_record(
        name="IEEE-CIS train_transaction",
        dataset_id="ieee-cis/train_transaction",
        path=transaction_path,
        raw_rows=prepared.transaction_source_rows,
        processed_rows=overlap_rows,
    )
    identity_source = _source_record(
        name="IEEE-CIS train_identity",
        dataset_id="ieee-cis/train_identity",
        path=identity_path,
        raw_rows=prepared.identity_source_rows,
        processed_rows=overlap_rows,
    )
    dataset_hashes = {
        "train_transaction": transaction_source["checksum"],
        "train_identity": identity_source["checksum"],
    }
    raw_ids = prepared.transaction.index.to_numpy()[split.test]
    aligner = EntityAligner(salt=f"vertimosaic-ieee-cis-{seed}")
    predictions = pd.DataFrame(
        {
            "entity_id": [aligner.pseudonymize(str(value)) for value in raw_ids],
            "target": test_active.labels,
            "probability": test_probability,
        }
    )
    feature_provenance = pd.DataFrame(
        [
            *[asdict(item) for item in _feature_provenance("transaction", tx_feature_metadata)],
            *[asdict(item) for item in _feature_provenance("identity", id_feature_metadata)],
        ]
    )
    run = RunArtifacts.create(root=runs_root)
    run_directory = run.finalize(
        config={
            "mode": "ieee_cis_linked",
            "model": model_name,
            "seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
            "entity_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "overlap_policy": "TransactionID intersection only",
        },
        seed=seed,
        dataset_provenance={
            "sources": [transaction_source, identity_source],
            "dataset_hashes": dataset_hashes,
            "authorization_required": True,
            "source_data_redistributed": False,
            "raw_source_ids_exported": False,
        },
        linkage_manifest={
            "method": "exact TransactionID intersection from authorized local competition files",
            "real_linkage": True,
            "intersection_only": True,
            "overlap_rows": overlap_rows,
            "raw_source_ids_exported": False,
            "pseudonymization": "SHA-256 research pseudonymization; not PSI",
        },
        metrics=payload,
        predictions=predictions,
        training_history=_training_frame(model),
        communication=communication_event_frame(model.transport.audit_log),
        feature_provenance=feature_provenance,
    )
    payload["run_directory"] = str(run_directory)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
