# Dataset licenses and access

VertiMosaic's **source code** is Apache-2.0. External datasets keep their own licenses and terms.
The project does not relicense external data.

| Party / benchmark | Dataset | Identifier | Access / license handling |
|---|---|---|---|
| Bank | UCI Default of Credit Card Clients | UCI 350, DOI 10.24432/C55S3H | Retrieved through UCI tooling; record provider metadata and CC BY 4.0 attribution reported by UCI. |
| Telecom | UCI Iranian Churn | UCI 563, DOI 10.24432/C5JW3Z | Retrieved through UCI tooling; record provider metadata and CC BY 4.0 attribution reported by UCI. |
| Insurance | freMTPL2freq / freMTPL2sev | OpenML 41214 / 41215 | Retrieved through `sklearn.datasets.fetch_openml`; preserve provider metadata and do not assert redistribution rights beyond source metadata. |
| Retail | UCI Online Retail | UCI 352, DOI 10.24432/C5BW33 | Retrieved through UCI tooling; record provider metadata and CC BY 4.0 attribution reported by UCI. |
| Optional linked benchmark | IEEE-CIS Fraud Detection | Kaggle competition | User-supplied authorized local files only. Never bundled, mirrored, or auto-downloaded by CI. |

Users are responsible for complying with source terms. `data/raw/` and `data/processed/` are git-ignored.
