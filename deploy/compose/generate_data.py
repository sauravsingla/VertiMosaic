# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic isolated Compose data")
    parser.add_argument("--root", type=Path, default=Path("/data"))
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bank = rng.normal(size=(args.rows, 2))
    telecom = rng.normal(size=(args.rows, 3))
    insurance = rng.normal(size=(args.rows, 2))
    retail = rng.normal(size=(args.rows, 2))
    score = (
        0.75 * bank[:, 0]
        - 0.35 * bank[:, 1]
        - 0.65 * telecom[:, 1]
        + 0.55 * insurance[:, 0]
        + 0.45 * retail[:, 1]
        + rng.normal(scale=0.35, size=args.rows)
    )
    target = (score > np.median(score)).astype(float)

    payloads = {
        "bank": {"features.npy": bank, "labels.npy": target},
        "telecom": {"features.npy": telecom},
        "insurance": {"features.npy": insurance},
        "retail": {"features.npy": retail},
    }
    for party, files in payloads.items():
        directory = args.root / party
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            np.save(directory / name, value, allow_pickle=False)
    print(f"generated {args.rows} aligned rows with seed={args.seed}")


if __name__ == "__main__":
    main()
