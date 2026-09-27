# Immutable release evidence

VertiMosaic separates normal CI artifacts from durable publication evidence.

When a `v*` tag is pushed, `.github/workflows/release-evidence.yml` installs the exact CPython 3.12 reproduction lock, runs the CPU smoke evidence pipeline and empirical privacy audit, records the installed environment, creates a per-file SHA-256 manifest, packages the available results/reports/benchmarks/figures/tables/runs, and uploads the bundle to the corresponding GitHub Release.

Published assets include:

- `vertimosaic-release-evidence.tar.gz`;
- `vertimosaic-release-evidence.tar.gz.sha256`;
- `manifest.json` with Git SHA, tag, platform, lock hash, artifact sizes and hashes; and
- `pip-freeze.txt` recording the exact installed environment.

The flexible dependency ranges in `pyproject.toml` remain appropriate for library users. `reproducibility/requirements-py312.lock` is intentionally exact and is used only for research/release reproduction.

## Zenodo

The GitHub Release bundle is structured so a repository-level Zenodo integration can archive tagged releases and mint a DOI. Zenodo authorization is account-level and is intentionally not embedded in repository code or secrets. After the repository is connected to Zenodo, creating a tagged GitHub Release through this workflow provides the immutable source/evidence payload for DOI archival.
