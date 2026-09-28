# Independent reproductions

This directory is the public index for reproduction evidence produced by people who are not acting as VertiMosaic maintainers for the run being reported.

A maintainer-run GitHub Actions job is useful build evidence, but it is **not** an independent reproduction. An independent record must disclose the reproducer's relationship to the project and must be produced from a released package in a clean environment.

## Reproduce a release

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install vertimosaic==0.2.0
vertimosaic-reproduce --expected-version 0.2.0 --output evidence
```

For the release under test, replace `0.2.0` with the exact published version.

## Submission contents

Create `reproductions/<github-login>/<version>/` containing:

- `attestation.json` following `reproductions/attestation.schema.json`;
- `environment.json` from the generated evidence bundle;
- `installed-distributions.txt`;
- `formal_comparison.csv`;
- `reproduction.json`;
- `SHA256SUMS`.

Do not edit measured files to make them match another run. Negative reproductions and portability failures are valid evidence.

## Independence rule

The attestation must state:

- GitHub/login or public identity used for the reproduction;
- affiliation;
- relationship to VertiMosaic and its maintainer(s);
- whether any maintainer supplied unpublished code, data, expected metrics, or debugging changes;
- the released package version installed;
- the source of the package (`PyPI` is preferred for release reproduction).

A record can be indexed as `independent=true` only when the reproducer states that they are unaffiliated with the project for this evaluation and the evidence was generated from a public release without unpublished maintainer modifications.

## Validation

```bash
python scripts/validate_reproduction.py reproductions/<github-login>/<version>
```

The validator checks the attestation, required files, hashes where available, and the release/version declaration. It cannot prove social independence; that statement remains an auditable human attestation.
