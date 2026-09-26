from __future__ import annotations

import json
from pathlib import Path

import typer

from vertimosaic.datasets import DatasetRegistry
from vertimosaic.experiments import run_demo, write_demo_report

app = typer.Typer(help="VertiMosaic: CPU-first vertical federated learning research framework")
datasets_app = typer.Typer(help="Dataset registry and retrieval commands")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def datasets_list() -> None:
    typer.echo(json.dumps(DatasetRegistry().list(), indent=2))


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


@app.command("report")
def report() -> None:
    typer.echo(
        "Reporting primitives are available; run an experiment to generate measured outputs."
    )


if __name__ == "__main__":
    app()
