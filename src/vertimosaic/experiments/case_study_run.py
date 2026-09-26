from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vertimosaic.evaluation import entity_level_split, select_f1_threshold
from vertimosaic.experiments.external import prepare_external_benchmark
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.reporting import select_sanitized_case_study


def run_distributed_signal_case_study(
    *,
    seed: int = 42,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    output_json: Path = Path("reports/case_study.json"),
    output_markdown: Path = Path("reports/case_study.md"),
) -> dict[str, object]:
    """Generate one measured, sanitized case study from distributed-signal external data."""
    benchmark = prepare_external_benchmark(
        mode="distributed_signal_external",
        cross_party_correlation=cross_party_correlation,
        seed=seed,
        insurance_sample_size=insurance_sample_size,
    )
    split = entity_level_split(benchmark.active.labels, seed=seed)
    train_active, train_passive = slice_parties(benchmark.active, benchmark.passive, split.train)
    val_active, val_passive = slice_parties(benchmark.active, benchmark.passive, split.validation)
    test_active, test_passive = slice_parties(benchmark.active, benchmark.passive, split.test)

    full_model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=500,
        l2=1e-3,
        seed=seed,
    )
    full_model.fit(train_active, train_passive)
    bank_model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=500,
        l2=1e-3,
        seed=seed,
    )
    bank_model.fit(train_active, [])

    validation_probability = full_model.predict_proba([val_active, *val_passive])[:, 1]
    threshold = select_f1_threshold(val_active.labels, validation_probability)
    full_probability = full_model.predict_proba([test_active, *test_passive])[:, 1]
    bank_probability = bank_model.predict_proba([test_active])[:, 1]

    rng = np.random.default_rng(seed + 10_000)
    permuted: dict[str, np.ndarray] = {}
    for party_name in ("bank", "telecom", "insurance", "retail"):
        permutation = rng.permutation(test_active.n_rows)
        if party_name == "bank":
            active_eval = ActiveParty(
                "bank",
                test_active._x[permutation],
                test_active.labels,
            )
            passive_eval = test_passive
        else:
            active_eval = test_active
            passive_eval = [
                PassiveParty(
                    party.name,
                    party._x[permutation] if party.name == party_name else party._x,
                )
                for party in test_passive
            ]
        permuted[party_name] = full_model.predict_proba([active_eval, *passive_eval])[:, 1]

    result = select_sanitized_case_study(
        split.test,
        bank_probability,
        full_probability,
        permuted,
        threshold=threshold,
    )
    result["benchmark_mode"] = "distributed_signal_external"
    result["cross_party_correlation"] = cross_party_correlation
    result["raw_passive_features_exposed"] = False

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contributions = result["contribution_summary"]
    if not isinstance(contributions, dict):
        raise TypeError("case-study contribution summary must be a mapping")
    lines = [
        "# Sanitized model-generated case study",
        "",
        f"Entity: `{result['entity_id']}`",
        f"Bank-only risk: {float(result['bank_only_risk']):.6f}",
        f"Four-party VFL risk: {float(result['four_party_vfl_risk']):.6f}",
        f"Validation-selected threshold: {float(result['threshold']):.6f}",
        "",
        "## Predictive contribution summary",
        "",
    ]
    for party, value in contributions.items():
        lines.append(f"- {party}: {float(value):.6f}")
    lines.extend(
        [
            "",
            "These are party-representation permutation probability deltas, not causal importance.",
            "No passive raw feature values are included in this report.",
        ]
    )
    output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
