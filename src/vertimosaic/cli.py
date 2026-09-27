from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from vertimosaic.datasets import (
    DatasetRegistry,
    fetch_external_party,
    make_vertical_synthetic,
    save_bundle,
)
from vertimosaic.experiments import (
    prepare_external_benchmark,
    prepare_ieee_cis,
    run_ablation_study,
    run_contribution_study,
    run_cpu_benchmarks,
    run_demo,
    run_drift_study,
    run_dropout_study,
    run_external_cpu_benchmark,
    run_external_experiment,
    run_ieee_cis_experiment,
    run_overlap_study,
    run_synthetic_experiment,
    write_demo_report,
)
from vertimosaic.reporting import write_final_report

app = typer.Typer(help="VertiMosaic: CPU-first vertical federated learning research framework")
datasets_app = typer.Typer(help="Dataset registry and retrieval commands")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def datasets_list() -> None:
    typer.echo(json.dumps(DatasetRegistry().list(), indent=2))


@datasets_app.command("describe")
def datasets_describe(name: str) -> None:
    typer.echo(json.dumps(DatasetRegistry().describe(name), indent=2, sort_keys=True))


@datasets_app.command("verify")
def datasets_verify() -> None:
    registry = DatasetRegistry()
    names = ("bank", "telecom", "insurance_freq", "insurance_sev", "retail")
    result = {name: registry.verify_license_metadata(name) for name in names}
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not all(result.values()):
        raise typer.Exit(code=1)


@datasets_app.command("download")
def datasets_download(
    name: str,
    output: Path = Path("data/processed"),
    insurance_sample_size: int | None = typer.Option(None, min=1),
    seed: int = 42,
) -> None:
    bundle = fetch_external_party(name, insurance_sample_size=insurance_sample_size, seed=seed)
    paths = save_bundle(bundle, output)
    typer.echo(
        json.dumps(
            {"party": bundle.party, "outputs": paths, "metadata": bundle.metadata},
            indent=2,
            default=str,
        )
    )


@datasets_app.command("download-all")
def datasets_download_all(
    output: Path = Path("data/processed"),
    insurance_sample_size: int | None = typer.Option(None, min=1),
    seed: int = 42,
) -> None:
    result: dict[str, object] = {}
    for name in ("bank", "telecom", "insurance", "retail"):
        bundle = fetch_external_party(name, insurance_sample_size=insurance_sample_size, seed=seed)
        result[name] = save_bundle(bundle, output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command("data-summary")
def data_summary(directory: Path = Path("data/processed")) -> None:
    rows: list[dict[str, object]] = []
    if directory.exists():
        for path in sorted(directory.glob("*.parquet")):
            frame = pd.read_parquet(path)
            rows.append({"file": path.name, "rows": len(frame), "columns": len(frame.columns)})
    typer.echo(json.dumps(rows, indent=2, sort_keys=True))


@app.command("generate-synthetic")
def generate_synthetic(
    rows: int = typer.Option(2000, min=200),
    seed: int = 42,
    output: Path = Path("data/processed/synthetic"),
) -> None:
    active, passive = make_vertical_synthetic(rows, seed)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(active._x).to_parquet(output / "bank_features.parquet", index=False)
    pd.DataFrame({"target": active.labels}).to_parquet(output / "bank_target.parquet", index=False)
    for party in passive:
        pd.DataFrame(party._x).to_parquet(output / f"{party.name}_features.parquet", index=False)
    typer.echo(str(output))


@app.command("prepare-external")
def prepare_external(
    mode: str = typer.Option("observed_target_external"),
    seed: int = 42,
    cross_party_correlation: float = typer.Option(0.25, min=0.0, max=1.0),
    insurance_sample_size: int | None = typer.Option(None, min=1),
    output: Path = Path("reports/external_prepare.json"),
) -> None:
    benchmark = prepare_external_benchmark(
        mode=mode,
        seed=seed,
        cross_party_correlation=cross_party_correlation,
        insurance_sample_size=insurance_sample_size,
    )
    payload = {
        "mode": benchmark.mode,
        "rows": benchmark.active.n_rows,
        "parties": [benchmark.active.name, *[party.name for party in benchmark.passive]],
        "four_sources_same_real_people": False,
        "linkage": {
            name: asdict(manifest) for name, manifest in benchmark.linkage_manifests.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command()
def demo(
    rows: int = typer.Option(2000, min=200),
    seed: int = 42,
    model: str = typer.Option("logistic", help="logistic or vfl-hist-gbdt"),
    output: Path = Path("reports/demo_metrics.json"),
) -> None:
    metrics = run_demo(rows=rows, seed=seed, model_name=model)
    write_demo_report(metrics, output)
    typer.echo(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("train")
def train(
    model: str = typer.Option("logistic", help="logistic or vfl-hist-gbdt"),
    rows: int = typer.Option(2000, min=200),
    seed: int = 42,
    bootstrap_replicates: int = typer.Option(1000, min=10),
    write_run: bool = True,
) -> None:
    payload = run_synthetic_experiment(
        rows=rows,
        seed=seed,
        model_name=model,
        bootstrap_replicates=bootstrap_replicates,
        write_run=write_run,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command("evaluate")
def evaluate(input_path: Path = Path("reports/demo_metrics.json")) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("ablation")
def ablation(
    rows: int = typer.Option(1200, min=400), seed: int = 42, model: str = "logistic"
) -> None:
    frame = run_ablation_study(rows=rows, seed=seed, model_name=model)
    typer.echo(frame.to_csv(index=False))


@app.command("contribution")
def contribution(rows: int = typer.Option(800, min=400), seed: int = 42) -> None:
    frame = run_contribution_study(rows=rows, seed=seed)
    typer.echo(frame.to_csv(index=False))


@app.command("overlap")
def overlap(
    rows: int = typer.Option(2000, min=400), seed: int = 42, model: str = "logistic"
) -> None:
    frame = run_overlap_study(rows=rows, seed=seed, model_name=model)
    typer.echo(frame.to_csv(index=False))


@app.command("dropout")
def dropout(rows: int = typer.Option(1600, min=400), seed: int = 42) -> None:
    frame = run_dropout_study(rows=rows, seed=seed)
    typer.echo(frame.to_csv(index=False))


@app.command("drift")
def drift(rows: int = typer.Option(1600, min=400), seed: int = 42) -> None:
    frame = run_drift_study(rows=rows, seed=seed)
    typer.echo(frame.to_csv(index=False))


@app.command("benchmark")
def benchmark(
    sizes: str = typer.Option("10000,30000,50000,100000,250000"),
    model: str = "logistic",
    mode: str = typer.Option(
        "synthetic_scale",
        help="synthetic_scale, observed_target_external or distributed_signal_external",
    ),
    seed: int = 42,
    bootstrap_replicates: int = typer.Option(1000, min=10),
    cross_party_correlation: float = typer.Option(0.25, min=0.0, max=1.0),
    insurance_sample_size: int | None = typer.Option(None, min=1),
) -> None:
    if mode == "synthetic_scale":
        parsed = [int(item.strip()) for item in sizes.split(",") if item.strip()]
        if any(item < 200 for item in parsed):
            raise typer.BadParameter("benchmark sizes must be at least 200")
        frame = run_cpu_benchmarks(
            parsed,
            seed=seed,
            model_name=model,
            bootstrap_replicates=bootstrap_replicates,
        )
    elif mode in {"observed_target_external", "distributed_signal_external"}:
        frame = run_external_cpu_benchmark(
            mode=mode,
            seed=seed,
            model_name=model,
            bootstrap_replicates=bootstrap_replicates,
            cross_party_correlation=cross_party_correlation,
            insurance_sample_size=insurance_sample_size,
        )
    else:
        raise typer.BadParameter(f"unsupported benchmark mode: {mode}")
    typer.echo(frame.to_csv(index=False))


@app.command("report")
def report(
    input_path: Path = Path("reports/experiment_payload.json"),
    output_directory: Path = Path("reports"),
) -> None:
    if not input_path.exists():
        raise typer.BadParameter(
            f"measured experiment payload does not exist: {input_path}; run an experiment first"
        )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    md_path, json_path = write_final_report(payload, output_directory)
    typer.echo(json.dumps({"markdown": str(md_path), "json": str(json_path)}, indent=2))


@app.command("external-demo")
def external_demo(
    model: str = typer.Option("vfl-hist-gbdt"),
    mode: str = typer.Option("observed_target_external"),
    seed: int = 42,
    cross_party_correlation: float = typer.Option(0.25, min=0.0, max=1.0),
    insurance_sample_size: int | None = typer.Option(None, min=1),
    bootstrap_replicates: int = typer.Option(1000, min=10),
    output: Path = Path("reports/external_demo.json"),
) -> None:
    payload = run_external_experiment(
        mode=mode,
        model_name=model,
        seed=seed,
        cross_party_correlation=cross_party_correlation,
        insurance_sample_size=insurance_sample_size,
        bootstrap_replicates=bootstrap_replicates,
        output=output,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command("prepare-ieee-cis")
def prepare_ieee_cis_command(
    transaction: Annotated[Path, typer.Option(exists=True, readable=True)],
    identity: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Path = Path("data/processed/ieee_cis_local"),
) -> None:
    prepared = prepare_ieee_cis(transaction, identity)
    output.mkdir(parents=True, exist_ok=True)
    prepared.transaction.to_parquet(output / "transaction_party.parquet")
    prepared.identity.to_parquet(output / "identity_party.parquet")
    pd.DataFrame({"target": prepared.target}).to_parquet(output / "target.parquet", index=False)
    typer.echo(
        json.dumps(
            {
                "rows": len(prepared.target),
                "output": str(output),
                "note": "Local authorized IEEE-CIS files are not redistributed by VertiMosaic.",
            },
            indent=2,
        )
    )


@app.command("run-ieee-cis")
def run_ieee_cis_command(
    transaction: Annotated[Path, typer.Option(exists=True, readable=True)],
    identity: Annotated[Path, typer.Option(exists=True, readable=True)],
    model: str = typer.Option("logistic", help="logistic or vfl-hist-gbdt"),
    seed: int = 42,
    bootstrap_replicates: int = typer.Option(1000, min=10),
    output: Path = Path("reports/ieee_cis_linked.json"),
) -> None:
    payload = run_ieee_cis_experiment(
        transaction,
        identity,
        model_name=model,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        output=output,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    app()
