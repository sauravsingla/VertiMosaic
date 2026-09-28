# Dataset licenses and attribution

VertiMosaic source code is Apache-2.0. That license does **not** relicense external data.

| Party / benchmark | Dataset | Provider / ID | DOI | License / handling |
|---|---|---|---|---|
| Bank | Default of Credit Card Clients | UCI 350 | 10.24432/C55S3H | UCI reports CC BY 4.0 |
| Telecom | Iranian Churn | UCI 563 | 10.24432/C5JW3Z | UCI reports CC BY 4.0 |
| Insurance | freMTPL2freq / freMTPL2sev | OpenML 41214 / 41215 | provider metadata | License metadata is queried from OpenML's official JSON API at verification/runtime; redistribution claims are not hard-coded |
| Retail | Online Retail | UCI 352 | 10.24432/C5BW33 | UCI reports CC BY 4.0 |
| Public exact-linked | MovieLens 1M | GroupLens Research, University of Minnesota | n/a | Research use with acknowledgement; GroupLens source data are downloaded at runtime and are **not redistributed by VertiMosaic**; commercial/revenue-bearing use and redistribution require provider permission under the supplied README terms |
| Optional exact-linked | IEEE-CIS Fraud Detection | Kaggle competition data | n/a | User-supplied authorized local files only; never redistributed |

Downloaded artifact metadata combines machine-readable registry fields with the runtime retrieval date and processed row count. Each source record retains the available provenance fields and values that a provider or loader does not expose remain explicit rather than being invented.

For MovieLens 1M, the benchmark records the SHA-256 of the exact ZIP downloaded from the fixed GroupLens HTTPS endpoint, the source-table row counts, and a claim boundary stating that the benchmark is genuine exact **multi-table linkage from one service**, not cross-organization linkage. The source ZIP and raw `*.dat` files are not committed to this repository or republished in release artifacts.

VertiMosaic records a SHA-256 checksum of the **processed feature parquet it creates** for the four-party external benchmark and labels that checksum scope explicitly. It does not present this as a provider-supplied raw-source checksum. Raw-source checksums that are not exposed by a provider remain `null`.

`vertimosaic datasets verify` checks the static UCI attribution fields and queries OpenML's official JSON metadata endpoint for the two insurance licenses. If that runtime metadata cannot be obtained or is empty, verification fails conservatively.
