# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd

from vertimosaic.datasets import build_npi_linked_frames


def test_npi_builder_exactly_links_real_entity_keys_across_three_sources() -> None:
    n = 120
    ids = [f"{1000000000 + index:010d}" for index in range(n)]
    prior = pd.DataFrame(
        {
            "npi": ids,
            "amount": np.linspace(10.0, 100.0, n),
        }
    )
    target = pd.DataFrame(
        {
            "npi": list(reversed(ids)),
            "amount": np.linspace(20.0, 200.0, n),
        }
    )
    nppes = pd.DataFrame(
        {
            "NPI": ids[20:] + ids[:20],
            "taxonomy": [f"t{index % 4}" for index in range(n)],
        }
    )
    care = pd.DataFrame(
        {
            "provider_npi": ids[50:] + ids[:50],
            "quality": np.linspace(0.1, 0.9, n),
        }
    )

    linked = build_npi_linked_frames(
        prior,
        target,
        nppes,
        care,
        prior_npi_column="npi",
        target_npi_column="npi",
        nppes_npi_column="NPI",
        care_npi_column="provider_npi",
        prior_amount_column="amount",
        target_amount_column="amount",
        nppes_feature_columns=["taxonomy"],
        care_feature_columns=["quality"],
        target_quantile=0.5,
    )

    assert len(linked.entity_ids) == n
    assert linked.entity_ids == tuple(sorted(ids))
    assert linked.active.shape == (n, 3)
    assert linked.nppes.shape == (n, 1)
    assert linked.care_compare.shape == (n, 1)
    assert linked.target.shape == (n,)
    assert set(np.unique(linked.target)) == {0.0, 1.0}
    assert linked.manifest["linkage"] == "exact 10-digit National Provider Identifier intersection"
