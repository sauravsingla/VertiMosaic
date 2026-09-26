# Reproducibility

Experiments use deterministic seeds and entity-level train/validation/test splits. Full runs can create `runs/<run_id>/` bundles containing configuration, dataset provenance, linkage manifest, environment, metrics, predictions, training history, communication metadata, feature provenance, and a run manifest.

Environment snapshots record OS, architecture, Python, CPU counts, RAM where available, dependency versions, git SHA, and seed. Unknown hardware metadata is represented as null rather than invented.
