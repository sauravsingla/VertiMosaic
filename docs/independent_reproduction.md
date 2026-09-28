# Independent reproduction protocol

This protocol is designed for a person or organization that is independent of the VertiMosaic author/maintainer.

## Clean environment

Create a new environment and install a released wheel from PyPI, not an editable checkout:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vertimosaic==<released-version>
vertimosaic-reproduce --expected-version <released-version> --output evidence
```

On Windows, activate the virtual environment with the platform-appropriate command.

The command writes:

- `formal_comparison.csv`, `.json`, and `.md`;
- `environment.json` with OS, CPU, RAM, Python and dependency versions;
- `installed-distributions.txt`;
- `reproduction.json` with a stable result digest; and
- `SHA256SUMS` covering the evidence files.

## Publishing independent evidence

Open the **Independent reproduction report** issue template and attach or link the complete evidence directory. State your affiliation and relationship, if any, to the project.

A maintainer-run GitHub Action, a maintainer machine, or an author-generated evidence bundle is useful reproducibility evidence, but it is **not** described as independent external validation. The repository only marks evidence as independent when an unaffiliated reproducer publishes it.

## Expected comparison

The formal table contains four explicitly separated protocols:

1. centralized all-feature non-federated baseline;
2. Bank-only/single-party non-federated baseline;
3. vertical federated logistic regression; and
4. vertical federated histogram GBDT.

Utility, runtime, RSS, protocol communication accounting and an empirical confidence-membership attack are recorded together. The optional full benchmark additionally records 50% overlap behavior and documented dropout behavior.
