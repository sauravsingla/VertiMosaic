# Release and archival policy

## v0.1.x freeze

The `v0.1.x` line is frozen as the first public research-evidence line. New features are developed for v0.2+. A v0.1.x patch is reserved for security, packaging, data-license/provenance, or reproducibility defects that materially affect the archived evidence.

Published tags and release assets are treated as immutable evidence points. Corrections are made through a new version/tag rather than rewriting a published release.

## Evidence attached to releases

The release-evidence workflow records the exact Python environment, runs the reproducible evidence pipeline and privacy research, builds the formal comparison matrix, creates SHA-256 manifests, and attaches the evidence bundle to the GitHub Release.

## Zenodo

The repository includes complete software metadata for archival. Zenodo/GitHub integration must be enabled by the repository owner in Zenodo; that account-level authorization cannot be performed by repository code. Once enabled, a new GitHub release can be archived and assigned a DOI. The DOI is added back to `CITATION.cff`, the README citation section, and the release notes in the next metadata-only revision.

Do not state that VertiMosaic has a Zenodo DOI until the Zenodo record actually exists.
