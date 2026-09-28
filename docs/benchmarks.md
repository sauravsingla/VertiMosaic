# Benchmarks

VertiMosaic deliberately separates **exact entity linkage**, **single-source vertical partitioning**, and **semi-synthetic cross-domain linkage**. These categories answer different research questions and should not be collapsed into one claim.

## Benchmark taxonomy

| Benchmark | Linkage type | What it supports | What it does not establish |
|---|---|---|---|
| NPI-linked public provider benchmark | Exact multi-source public linkage | Real entities linked across distinct public data products using authoritative NPI | Private cross-company collaboration or production PSI |
| MovieLens 1M | Exact multi-table linkage | Real users linked across public tables from one service | Cross-organization entity resolution |
| IEEE-CIS | Authorized local exact linkage | Exact transaction/identity joins on `TransactionID` | Redistribution of restricted competition data |
| UCI Credit | Exact-row single-source vertical partition | Deterministic vertical feature partitioning with exact entity rows | Real cross-source linkage |
| Bank/Telecom/Insurance/Retail | Semi-synthetic cross-domain linkage | Controlled cross-industry VFL research with heterogeneous source domains | Same-real-person linkage across the four domains |

## Exact-NPI multi-source public benchmark

VertiMosaic v0.3 adds a builder that joins real provider records across:

- Open Payments,
- NPPES, and
- CMS Care Compare / provider-data.

The authoritative 10-digit NPI provides exact entity identity across the participating public sources. A leakage-resistant design uses an earlier Open Payments period for active-party features and a later period for the observed target.

See [Exact-NPI Benchmark](npi_linked_benchmark.md).

## MovieLens 1M

The MovieLens path links `users.dat`, `ratings.dat`, and `movies.dat` using observed service identifiers. Historical behavior can be used for passive features while later rating events define a target excluded from those passive features.

This is genuine exact linkage across public tables, but all tables originate from one service.

## IEEE-CIS

The optional IEEE-CIS path uses locally authorized files and exact `TransactionID` intersection. VertiMosaic does not download or redistribute those competition files.

```bash
vertimosaic prepare-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv
```

## UCI Credit

The UCI path retrieves the public source dataset and partitions disjoint columns between parties while preserving exact entity rows. It is useful for protocol sanity checks because alignment is exact and reproducible, but it does not model cross-organization entity resolution.

```bash
vertimosaic-linked-uci
```

## Four-industry benchmark

The Bank/Telecom/Insurance/Retail benchmark combines heterogeneous public source domains through an explicitly documented research linkage mechanism. The four sources do not describe the same real individuals.

Use this benchmark for questions about heterogeneous cross-domain feature value, missing parties, contribution studies, communication cost, and VFL behavior under controlled linkage assumptions—not as evidence of real-world cross-company identity matching.

## External framework comparison

VertiMosaic includes a deterministic exchange/normalization contract for measured comparisons against FATE, SecretFlow, or another VFL framework. The repository intentionally avoids fabricating third-party benchmark results.

See [External Comparators](external_comparators.md).

## Reporting rules

When publishing or presenting benchmark results:

1. name the linkage category explicitly;
2. record source provenance and licensing constraints;
3. keep train/validation/test entities separated before fitting preprocessing state;
4. bind exact entity IDs where the strict protocol path supports them;
5. report runtime and communication metadata alongside predictive metrics; and
6. do not generalize a public or semi-synthetic benchmark into a private-production deployment claim.
