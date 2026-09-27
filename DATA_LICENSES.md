# Dataset licenses and attribution

VertiMosaic source code is Apache-2.0. That license does **not** relicense external data.

| Party | Dataset | Provider / ID | DOI | License / handling |
|---|---|---|---|---|
| Bank | Default of Credit Card Clients | UCI 350 | 10.24432/C55S3H | UCI reports CC BY 4.0 |
| Telecom | Iranian Churn | UCI 563 | 10.24432/C5JW3Z | UCI reports CC BY 4.0 |
| Insurance | freMTPL2freq / freMTPL2sev | OpenML 41214 / 41215 | provider metadata | Retrieved programmatically; redistribution claims are not hard-coded |
| Retail | Online Retail | UCI 352 | 10.24432/C5BW33 | UCI reports CC BY 4.0 |
| Optional | IEEE-CIS Fraud Detection | Kaggle competition data | n/a | User-supplied authorized local files only; never redistributed |

Downloaded artifact metadata combines the machine-readable registry fields with the runtime retrieval date and processed row count. VertiMosaic also records a SHA-256 checksum of the **processed feature parquet it creates** and labels that checksum scope explicitly. It does not present this as a provider-supplied raw-source checksum. Raw-source checksums or raw row counts that are not exposed by the provider/loader remain `null` rather than being invented.
