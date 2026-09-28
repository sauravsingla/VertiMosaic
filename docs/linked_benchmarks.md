# Exact-linked public benchmarks

VertiMosaic keeps two categories separate:

## IEEE-CIS transaction/identity

Authorized local IEEE-CIS files can be joined on the exact `TransactionID` intersection. This remains the strongest genuinely linked multi-table benchmark currently supported, but the competition files are not downloaded or redistributed by VertiMosaic.

## UCI Credit exact-row vertical partition

`vertimosaic-linked-uci` adds a public, reproducible exact-link sanity benchmark using UCI **Default of Credit Card Clients** (dataset 350). The benchmark:

- retrieves the source directly from UCI at runtime;
- uses the exact same real source rows on both parties;
- partitions **disjoint raw source columns** between an active and passive party;
- fits imputation/scaling on training rows only;
- preserves a held-out entity split;
- writes pseudonymous test identifiers, source checksum, license metadata, metrics and a SHA-256 manifest.

This benchmark is intentionally labeled **single-source exact-row vertical partitioning**. It does **not** claim that the two feature partitions originated from independent organizations, and it is not evidence of cross-organization entity resolution. That claim boundary is included in every output bundle.
