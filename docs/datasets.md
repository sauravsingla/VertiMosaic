# External datasets

The reference external benchmark is grounded in four public domains:

- Bank: UCI Default of Credit Card Clients (ID 350; observed default target).
- Telecom: UCI Iranian Churn (ID 563).
- Insurance: OpenML `freMTPL2freq` (41214) and `freMTPL2sev` (41215).
- Retail: UCI Online Retail (ID 352), aggregated to customer level using transactions before the feature cutoff.

These sources do **not** describe the same people. They are independently preprocessed and connected only through explicitly semi-synthetic target-blind linkage. Dataset licensing is separate from the Apache-2.0 source license; see `DATA_LICENSES.md`.
