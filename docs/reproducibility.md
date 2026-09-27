# Reproducibility

Experiments use deterministic seeds and entity-level train/validation/test splits. Full runs can create `runs/<run_id>/` bundles containing configuration, dataset provenance, linkage manifest, environment, metrics, predictions, training history, communication metadata, feature provenance, and a run manifest.

Environment snapshots record OS, architecture, Python, CPU counts, RAM where available, dependency versions, git SHA, and seed. Unknown hardware metadata is represented as null rather than invented.

## Privacy research evidence

The multi-factor privacy research runner is deterministic for a fixed configuration and records every individual seed result before aggregation. It reports the configured seed list, dataset sizes, party counts, training-round grids, regularization settings, and mitigation strengths in the JSON output.

```bash
python scripts/run_privacy_research.py --smoke
python scripts/run_privacy_research.py
```

The default outputs are:

- `reports/privacy_research.json` — configuration, raw per-seed measurements, confidence-interval summaries, and interpretation metadata;
- `reports/privacy_research_summary.csv` — flattened summary rows suitable for paper tables and independent analysis.

Confidence intervals are computed from independent deterministic seed runs and the exact interval convention is stored in the result metadata. CI uses the reduced grid to keep verification bounded; tagged release evidence runs the complete configured grid and includes the resulting JSON/CSV files in the hashed release bundle.

The original lightweight `scripts/run_privacy_audit.py` remains available as a fast baseline leakage smoke test. The research suite is intentionally separate so a fast protocol regression test is not confused with the larger multi-factor empirical study.
