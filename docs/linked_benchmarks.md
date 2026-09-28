# Exact-linked public benchmarks

VertiMosaic keeps linkage categories separate so a reproducible vertical split is never presented as stronger evidence than the source data actually support.

## MovieLens 1M exact multi-table linkage

`vertimosaic-linked-movielens` is the fully public, provider-backed multi-table linked benchmark in v0.2.0. It downloads MovieLens 1M directly from GroupLens at runtime and uses the dataset's observed identifiers:

- `UserID` links `users.dat` to `ratings.dat`;
- `MovieID` links `ratings.dat` to `movies.dat`;
- the active party receives demographic attributes from `users.dat`;
- the passive party receives historical-rating aggregates and genre behavior derived from the earliest 80% of each user's observed ratings;
- the binary target is derived only from the latest 20% of that user's observed ratings: whether the late-period mean rating is at least four stars;
- the target-defining late rating events are excluded from passive-party features;
- train/validation/test splitting is entity-level and all preprocessing is fitted on training entities only;
- the output records the downloaded archive SHA-256, provider/license metadata, environment, sampled peak RSS, protocol communication metadata, bootstrap uncertainty and pseudonymous test predictions.

This is **genuine exact linkage across multiple public tables describing the same service users**. It is still a **single-service dataset**, not cross-organization linkage, and it does not demonstrate PSI. MovieLens source files are not committed or redistributed by VertiMosaic; they are downloaded from GroupLens for research use at runtime.

## IEEE-CIS transaction/identity

Authorized local IEEE-CIS files can be joined on the exact `TransactionID` intersection. This is a genuinely linked two-table transaction/identity benchmark, but the competition files are not downloaded or redistributed by VertiMosaic.

## UCI Credit exact-row vertical partition

`vertimosaic-linked-uci` provides a public exact-link sanity benchmark using UCI **Default of Credit Card Clients** (dataset 350). The benchmark:

- retrieves the source directly from UCI at runtime;
- uses the exact same real source rows on both parties;
- partitions **disjoint raw source columns** between an active and passive party;
- fits imputation/scaling on training rows only;
- preserves a held-out entity split;
- writes pseudonymous test identifiers, source checksum, license metadata, metrics and a SHA-256 manifest.

This benchmark is intentionally labeled **single-source exact-row vertical partitioning**. It does **not** claim that the two feature partitions originated from independent organizations, and it is not evidence of cross-organization entity resolution.

## Cross-industry four-party benchmark

The Bank/Telecom/Insurance/Retail public-source benchmark remains an externally grounded **semi-synthetic linkage construction**. The source datasets do not describe the same real people. That benchmark tests cross-domain modeling assumptions; it must not be conflated with MovieLens/IEEE-CIS exact linkage.
