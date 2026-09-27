from __future__ import annotations

import json
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from vertimosaic.alignment import EntityAligner
from vertimosaic.baselines import fit_centralized_baseline
from vertimosaic.datasets import ExternalDatasetBundle
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
from vertimosaic.experiments.external import linkage_manifest_dict, prepare_external_benchmark
from vertimosaic.experiments.external_preprocessing import (
    ExternalPreparedSplits,
    prepare_external_splits_locally,
)
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.provenance import FeatureProvenance
from vertimosaic.reporting import select_sanitized_case_study
from vertimosaic.reproducibility import RunArtifacts


def _external_model(
    model_name: str,
    seed: int,
    *,
    early_stopping: bool,
) -> VFLLogisticRegression | VFLHistGBDT:
    if model_name == "logistic":
        return VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=500,
            l2=1e-3,
            early_stopping_rounds=5 if early_stopping else None,
            seed=seed,
        )
    if model_name == "vfl-hist-gbdt":
        return VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping_rounds=3 if early_stopping else None,
            seed=seed,
        )
    raise ValueError(f"unknown model: {model_name}")


def _fit_external_model(
    model: VFLLogisticRegression | VFLHistGBDT,
    train_active: ActiveParty,
    train_passive: list[PassiveParty],
    validation_active: ActiveParty,
    validation_passive: list[PassiveParty],
) -> None:
    model.fit(train_active, train_passive, validation_active, validation_passive)


def _share_gbdt_state(
    model: VFLLogisticRegression | VFLHistGBDT,
    source: list[PassiveParty],
    target: list[PassiveParty],
) -> None:
    if not isinstance(model, VFLHistGBDT):
        return
    for source_party, target_party in zip(source, target, strict=True):
        source_party.share_histogram_routing_state_with(target_party)


def _party_permuted_probabilities(
    model: VFLLogisticRegression | VFLHistGBDT,
    active: ActiveParty,
    passive: list[PassiveParty],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    all_names = [active.name, *[party.name for party in passive]]
    source_parties: list[PassiveParty] = [active, *passive]
    for party_index, party_name in enumerate(all_names):
        rng = np.random.default_rng(seed + 500 + party_index)
        permutation = rng.permutation(active.n_rows)
        permuted_active = ActiveParty(active.name, active._x.copy(), active.labels)
        permuted_passive = [PassiveParty(item.name, item._x.copy()) for item in passive]
        if party_name == active.name:
            permuted_active = ActiveParty(active.name, active._x[permutation], active.labels)
        else:
            permuted_passive = [
                PassiveParty(
                    item.name,
                    item._x[permutation] if item.name == party_name else item._x,
                )
                for item in passive
            ]
        target_parties: list[PassiveParty] = [permuted_active, *permuted_passive]
        _share_gbdt_state(model, source_parties, target_parties)
        output[party_name] = model.predict_proba(target_parties)[:, 1]
    return output


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


def _frame_sha256(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return sha256(hashed.tobytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _external_feature_provenance(
    benchmark_source_provenance: dict[str, list[FeatureProvenance]],
    prepared: ExternalPreparedSplits,
) -> pd.DataFrame:
    records: list[FeatureProvenance] = []
    for party, transformed_features in prepared.feature_metadata.items():
        source_lookup = {
            record.feature: record for record in benchmark_source_provenance.get(party, [])
        }
        for transformed in transformed_features:
            source_feature = transformed["source_column"]
            original = source_lookup.get(source_feature)
            if original is None:
                fallback_notes = (
                    "passive-party row assignment is semi-synthetic" if party != "bank" else ""
                )
                records.append(
                    FeatureProvenance(
                        party=party,
                        feature=transformed["feature"],
                        external_dataset=party,
                        source_column=source_feature,
                        transformation=transformed["transformation"],
                        source_type="real_external_derived",
                        observed_or_derived="derived",
                        semi_synthetic=party != "bank",
                        notes=fallback_notes,
                    )
                )
                continue
            linkage_note = "semi-synthetic donor linkage -> " if party != "bank" else ""
            records.append(
                FeatureProvenance(
                    party=party,
                    feature=transformed["feature"],
                    external_dataset=original.external_dataset,
                    source_column=original.source_column,
                    transformation=(
                        f"{linkage_note}{original.transformation} -> "
                        f"{transformed['transformation']}"
                    ),
                    source_type=original.source_type,
                    observed_or_derived="derived",
                    semi_synthetic=party != "bank",
                    notes=(
                        "source feature values come from the public dataset; cross-domain entity "
                        "assignment is semi-synthetic"
                        if party != "bank"
                        else ""
                    ),
                )
            )
    return pd.DataFrame([asdict(record) for record in records])


def run_external_experiment(
    *,
    mode: str = "observed_target_external",
    model_name: str = "vfl-hist-gbdt",
    seed: int = 42,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    bootstrap_replicates: int = 1000,
    output: Path | None = None,
    write_run: bool = True,
    runs_root: Path = Path("runs"),
    bundles: dict[str, ExternalDatasetBundle] | None = None,
) -> dict[str, Any]:
    preparation_start = time.perf_counter()
    benchmark = prepare_external_benchmark(
        mode=mode,
        cross_party_correlation=cross_party_correlation,
        seed=seed,
        insurance_sample_size=insurance_sample_size,
        bundles=bundles,
    )
    benchmark_preparation_seconds = time.perf_counter() - preparation_start
    data_preparation_seconds = max(
        0.0,
        benchmark_preparation_seconds - benchmark.entity_alignment_seconds,
    )
    split = entity_level_split(benchmark.active.labels, seed=seed)
    prepared = prepare_external_splits_locally(benchmark, split)
    train_active = prepared.train_active
    train_passive = prepared.train_passive
    val_active = prepared.validation_active
    val_passive = prepared.validation_passive
    test_active = prepared.test_active
    test_passive = prepared.test_passive

    model = _external_model(model_name, seed, early_stopping=True)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    _fit_external_model(model, train_active, train_passive, val_active, val_passive)
    _share_gbdt_state(
        model,
        [train_active, *train_passive],
        [test_active, *test_passive],
    )
    training_seconds = time.perf_counter() - start
    peak_rss_bytes = max(rss_before, process.memory_info().rss)
    validation_p = model.predict_proba([val_active, *val_passive])[:, 1]
    threshold = select_f1_threshold(val_active.labels, validation_p)
    inference_start = time.perf_counter()
    test_p = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    metrics = binary_metrics(test_active.labels, test_p, threshold=threshold)
    intervals = bootstrap_confidence_intervals(
        test_active.labels,
        test_p,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    centralized_name = "logistic" if model_name == "logistic" else "hist-gbdt"
    bank_baseline = fit_centralized_baseline(
        [train_active._x],
        train_active.labels,
        [test_active._x],
        model=centralized_name,
        seed=seed,
    )
    all_party_baseline = fit_centralized_baseline(
        [train_active._x, *[item._x for item in train_passive]],
        train_active.labels,
        [test_active._x, *[item._x for item in test_passive]],
        model=centralized_name,
        seed=seed,
    )
    bank_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_p,
        bank_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    bank_comparison["baseline"] = "bank_only_non_federated"
    all_party_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_p,
        all_party_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    all_party_comparison["baseline"] = "centralized_all_party_non_federated"
    communication = communication_totals(model.transport.audit_log)
    training_steps = (
        model.n_iter_ if isinstance(model, VFLLogisticRegression) else len(model.trees_)
    )
    payload: dict[str, Any] = {
        "benchmark_description": "externally grounded semi-synthetic cross-industry VFL benchmark",
        "mode": mode,
        "target_description": "observed Bank default target"
        if mode == "observed_target_external"
        else "semi_synthetic_cross_industry_risk",
        "model": model_name,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "comparisons": [bank_comparison, all_party_comparison],
        "confusion_matrix": confusion_at_threshold(test_active.labels, test_p, threshold),
        "threshold_selected_on_validation": threshold,
        "data_preparation_seconds": data_preparation_seconds,
        "entity_alignment_seconds": benchmark.entity_alignment_seconds,
        "preprocessing_seconds": prepared.preprocessing_seconds,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "training_steps": training_steps,
        "peak_rss_bytes": int(peak_rss_bytes),
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication": communication,
        "preprocessor_artifacts": prepared.preprocessor_paths,
        "preprocessing_fit_scope": "TRAIN only, independently per party",
        "linkage": linkage_manifest_dict(benchmark),
        "four_sources_same_real_people": False,
    }
    if mode == "distributed_signal_external":
        bank_model = _external_model(model_name, seed + 1, early_stopping=True)
        _fit_external_model(bank_model, train_active, [], val_active, [])
        bank_test_active = ActiveParty("bank", test_active._x.copy(), test_active.labels)
        _share_gbdt_state(bank_model, [train_active], [bank_test_active])
        bank_only_p = bank_model.predict_proba([bank_test_active])[:, 1]
        party_permuted = _party_permuted_probabilities(
            model,
            test_active,
            test_passive,
            seed=seed,
        )
        payload["case_study"] = select_sanitized_case_study(
            split.test,
            bank_only_p,
            test_p,
            party_permuted,
            threshold=threshold,
        )

    if write_run:
        run = RunArtifacts.create(root=runs_root)
        aligner = EntityAligner(salt=f"vertimosaic-external-{seed}")
        predictions = pd.DataFrame(
            {
                "entity_id": [aligner.pseudonymize(str(index)) for index in split.test],
                "target": test_active.labels,
                "probability": test_p,
            }
        )
        dataset_hashes = {
            f"{party}_linked_features": _frame_sha256(frame)
            for party, frame in benchmark.feature_frames.items()
        }
        dataset_hashes["bank_target"] = _array_sha256(benchmark.active.labels)
        directory = run.finalize(
            config={
                "mode": mode,
                "model": model_name,
                "seed": seed,
                "cross_party_correlation": cross_party_correlation,
                "insurance_sample_size": insurance_sample_size,
                "bootstrap_replicates": bootstrap_replicates,
                "entity_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
            seed=seed,
            dataset_provenance={
                "benchmark_description": (
                    "externally grounded semi-synthetic cross-industry VFL benchmark"
                ),
                "sources": benchmark.source_metadata,
                "dataset_hashes": dataset_hashes,
                "four_sources_same_real_people": False,
                "raw_source_ids_exported": False,
            },
            linkage_manifest={
                "method": "gaussian_copula_rank_proximity",
                "target_blind": True,
                "target_generated_after_linkage": mode == "distributed_signal_external",
                "parties": linkage_manifest_dict(benchmark),
                "pseudonymization": "SHA-256 research pseudonymization; not PSI",
            },
            metrics=payload,
            predictions=predictions,
            training_history=_training_frame(model),
            communication=communication_event_frame(model.transport.audit_log),
            feature_provenance=_external_feature_provenance(
                benchmark.source_provenance,
                prepared,
            ),
        )
        payload["run_directory"] = str(directory)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
