"""VertiMosaic command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from vertimosaic.datasets import download_dataset, list_specs
from vertimosaic.training import run_demo

app = typer.Typer(help="VertiMosaic: vertical federated learning for tabular data.")
datasets_app = typer.Typer(help="External dataset registry and retrieval.")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def datasets_list() -> None:
    for spec in list_specs():
        typer.echo(f"{spec.key:18} {spec.party:10} {spec.name} [{spec.provider} {spec.dataset_id}]")


@datasets_app.command("download")
def datasets_download(key: str, root: Path = Path("data")) -> None:
    path = download_dataset(key, root)
    typer.echo(str(path))


@datasets_app.command("download-all")
def datasets_download_all(root: Path = Path("data")) -> None:
    for key in ["bank", "telecom", "insurance_freq", "insurance_sev", "retail"]:
        path = download_dataset(key, root)
        typer.echo(f"{key}: {path}")


@app.command()
def demo(
    rows: int = typer.Option(5000, min=500),
    seed: int = 42,
    model: str = typer.Option("logistic", help="logistic or vfl-hist-gbdt"),
    output: Path = Path("results/demo"),
) -> None:
    """Run a fully local four-party synthetic VFL smoke experiment."""
    result = run_demo(rows=rows, seed=seed, model=model, output=output)
    typer.echo(json.dumps(result, indent=2))


@app.command("generate-synthetic")
def generate_synthetic_cmd(rows: int = 10_000, seed: int = 42) -> None:
    from vertimosaic.datasets import generate_synthetic

    data = generate_synthetic(rows, seed)
    out = Path("data/processed/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    for party, frame in data.items():
        frame.to_parquet(out / f"{party}.parquet", index=False)
    typer.echo(str(out))


@app.command()
def report() -> None:
    path = Path("results/demo/metrics.json")
    if not path.exists():
        raise typer.BadParameter("Run `vertimosaic demo` first.")
    payload = json.loads(path.read_text())
    metrics = payload["metrics"]
    typer.echo("VertiMosaic measured results")
    typer.echo(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    typer.echo(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    typer.echo(
        "Interpretation is descriptive; no claim of superiority is made without a paired baseline interval."
    )


if __name__ == "__main__":
    app()
