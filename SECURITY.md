# Security policy

VertiMosaic is a research simulator, not a cryptographically secure federated-learning system. Do not use it as the sole protection for sensitive production data.

Please report security issues privately to the repository maintainer rather than publishing exploit details first. Reports should include affected version/commit, reproduction steps, impact, and suggested remediation where known.

CI runs Bandit and dependency auditing. Raw rows, credentials, tokens, Kaggle secrets, and dataset passwords must never be committed or logged.
