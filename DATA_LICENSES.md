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
| Public exact-NPI multi-source | Open Payments + NPPES + CMS Care Compare/provider-data | Centers for Medicare & Medicaid Services / NPPES | n/a | Public government/provider data downloaded by the user from official portals. VertiMosaic does not redistribute source files and does not hard-code a blanket license claim; users must verify the release-specific terms, data dictionaries and permitted use at retrieval time. NPPES downloadable provider data are FOIA-disclosable; Open Payments and CMS provider-data products are publicly available. |

Downloaded artifact metadata combines machine-readable registry fields with the runtime retrieval date and processed row count. Each source record retains the available provenance fields and values that a provider or loader does not expose remain explicit rather than being invented.

For MovieLens 1M, the benchmark records the SHA-256 of the exact ZIP downloaded from the fixed GroupLens HTTPS endpoint, the source-table row counts, and a claim boundary stating that the benchmark is genuine exact **multi-table linkage from one service**, not cross-organization linkage. The source ZIP and raw `*.dat` files are not committed to this repository or republished in release artifacts.

For the exact-NPI benchmark, the repository provides a local builder rather than an automatic source downloader. This is deliberate: Open Payments, NPPES and CMS provider-data releases are refreshed over time and their file schemas/packaging can change. A paper-quality run should record the exact release dates/years, official download locations, local SHA-256 checksums, selected columns, intersection coverage and source data dictionaries. The resulting benchmark is genuine exact linkage across public source products describing the same providers, but it is not evidence of confidential private-company data sharing.

VertiMosaic records a SHA-256 checksum of the **processed feature parquet it creates** for the four-party external benchmark and labels that checksum scope explicitly. It does not present this as a provider-supplied raw-source checksum. Raw-source checksums that are not exposed by a provider remain `null`.

`vertimosaic datasets verify` checks the static UCI attribution fields and queries OpenML's official JSON metadata endpoint for the two insurance licenses. If that runtime metadata cannot be obtained or is empty, verification fails conservatively.
