# Dataset licenses and attribution

VertiMosaic source code is Apache-2.0. That license does **not** relicense external data.

| Party | Dataset | Provider / ID | DOI | License / handling |
|---|---|---|---|---|
| Bank | Default of Credit Card Clients | UCI 350 | 10.24432/C55S3H | UCI reports CC BY 4.0 |
| Telecom | Iranian Churn | UCI 563 | 10.24432/C5JW3Z | UCI reports CC BY 4.0 |
| Insurance | freMTPL2freq / freMTPL2sev | OpenML 41214 / 41215 | provider metadata | License metadata is queried from OpenML's official JSON API at verification/runtime; redistribution claims are not hard-coded |
| Retail | Online Retail | UCI 352 | 10.24432/C5BW33 | UCI reports CC BY 4.0 |
| Optional | IEEE-CIS Fraud Detection | Kaggle competition data | n/a | User-supplied authorized local files only; never redistributed |

Downloaded artifact metadata combines the machine-readable registry fields with the runtime retrieval date and processed row count. Each source record retains the complete registry/provenance schema, including explicit `raw_rows`, `processed_rows`, and `checksum` fields. Values that the provider or loader does not expose remain `null` rather than being invented.

VertiMosaic records a SHA-256 checksum of the **processed feature parquet it creates** and labels that checksum scope explicitly. It does not present this as a provider-supplied raw-source checksum. Raw-source checksums that are not exposed by the provider remain `null`.

`vertimosaic datasets verify` checks the static UCI attribution fields and queries OpenML's official JSON metadata endpoint for the two insurance licenses. If that runtime metadata cannot be obtained or is empty, verification fails conservatively.
