# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.privacy import (
    gradient_label_inference,
    message_feature_reconstruction,
    privacy_attack_scenarios,
)


def run_advanced_privacy_audit(
    *,
    rows: int = 1200,
    seed: int = 42,
    output: Path = Path("reports/advanced_privacy_audit.json"),
) -> dict[str, Any]:
    if rows < 300:
        raise ValueError("advanced privacy audit requires at least 300 rows")
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    validation_active, validation_passive = slice_parties(active, passive, split.validation)
    test_active, test_passive = slice_parties(active, passive, split.test)

    model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=120,
        l2=1e-3,
        early_stopping_rounds=5,
        seed=seed,
    )
    model.fit(train_active, train_passive, validation_active, validation_passive)

    train_probability = model.predict_proba([train_active, *train_passive])[:, 1]
    gradient_attack = gradient_label_inference(
        train_probability - train_active.labels,
        train_active.labels,
    )

    first_validation = validation_passive[0]
    first_test = test_passive[0]
    weights = model.weights_[first_validation.name]
    validation_message = first_validation.local_logits(weights)
    test_message = first_test.local_logits(weights)
    reconstruction = message_feature_reconstruction(
        validation_message,
        first_validation._x,
        test_message,
        first_test._x,
    )

    payload: dict[str, Any] = {
        "audit": "advanced_protocol_message_privacy_baselines",
        "rows": rows,
        "seed": seed,
        "scenarios": list(privacy_attack_scenarios()),
        "passive_observer_gradient_label_inference": gradient_attack.to_dict(),
        "active_observer_passive_feature_reconstruction": reconstruction.to_dict(),
        "feature_reconstruction_party": first_test.name,
        "claim_boundary": (
            "These are empirical baseline attacks against explicit reference-protocol messages. "
            "Positive leakage demonstrates an attack surface; weak baseline results do not "
            "prove privacy."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run advanced VertiMosaic message-leakage baselines."
    )
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/advanced_privacy_audit.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_advanced_privacy_audit(rows=args.rows, seed=args.seed, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
