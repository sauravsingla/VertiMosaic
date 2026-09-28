# Public exact-NPI multi-source benchmark

VertiMosaic v0.3 adds an optional **genuinely exact-linked public multi-source benchmark** built around the US National Provider Identifier (NPI). It is intended to close the gap between single-source vertical partitioning and semi-synthetic cross-domain linkage without overstating what public government data demonstrates.

## Sources

The benchmark joins real provider records from three public data products using the same authoritative 10-digit NPI:

1. **Open Payments** — prior-period payment aggregates form the active-party features; a later period supplies the observed target.
2. **NPPES** — passive-party provider/demographic/taxonomy fields.
3. **CMS Care Compare / provider-data** — a second passive-party provider/quality/service feature set.

The builder does not hard-code download URLs or column schemas because CMS refreshes the underlying files and schemas over time. Download the desired public releases from the official CMS/NPPES data portals and pass the relevant columns to `build_npi_linked_frames`.

## Why this is stronger than the semi-synthetic four-industry benchmark

The same NPI refers to the same real provider across the participating public source files. Entity alignment is therefore exact and authoritative; no copula, rank, similarity, or synthetic matching rule creates the cross-source relationship.

## What it still does not prove

This benchmark is **not** evidence of a private cross-company federation. The data are public US government/provider data products, and the benchmark runs locally after public files have been downloaded. It demonstrates real multi-source entity linkage and vertical feature partitioning, not confidential bilateral data collaboration.

## Leakage-resistant temporal target design

Use an earlier Open Payments period for active-party features and a later period for the target. The included builder aggregates prior-period payment count/sum/mean and defines the later-period binary target from a configurable quantile of later total payments.

For a paper-quality benchmark:

- freeze exact source release dates/years;
- record source URLs, checksums and data dictionaries;
- split entities before fitting imputers/encoders/scalers;
- fit categorical vocabularies and numeric preprocessing on training entities only;
- bind the ordered NPI IDs to all VFL parties with `bind_entity_ids`;
- use `require_entity_ids=True` for VFL training;
- record source-specific missingness and intersection coverage;
- keep the public-data scope statement visible in tables and captions.

## Example builder

```python
from vertimosaic.datasets import build_npi_linked_frames

linked = build_npi_linked_frames(
    prior_payments=payments_2024,
    target_payments=payments_2025,
    nppes=nppes_snapshot,
    care_compare=provider_snapshot,
    prior_npi_column="<Open Payments NPI column>",
    target_npi_column="<Open Payments NPI column>",
    nppes_npi_column="NPI",
    care_npi_column="<provider-data NPI column>",
    prior_amount_column="<payment amount column>",
    target_amount_column="<payment amount column>",
    nppes_feature_columns=["<selected NPPES columns>"],
    care_feature_columns=["<selected provider-data columns>"],
)
```

Column placeholders are deliberate: the repository should not silently guess an external release's schema.
