# Reproducibility environment

`requirements-py312.lock` is the exact dependency set used by the tagged-release evidence workflow. The package itself intentionally keeps bounded dependency ranges in `pyproject.toml`; research releases use this separate exact lock so the published evidence can be recreated without converting the library into an over-pinned application.

A tagged release records:

- the Git commit SHA and tag;
- the SHA-256 of this lock file;
- the complete `pip freeze` environment;
- generated benchmark/report/figure/table/run artifacts that exist for the release run;
- a per-file SHA-256 manifest; and
- an archive SHA-256 published beside the release evidence tarball.

The lock targets CPython 3.12. Updating it is a deliberate research-environment change and should be reviewed separately from ordinary runtime dependency-range maintenance.
