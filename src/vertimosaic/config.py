"""Project configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Validated model training configuration."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    test_size: float = Field(default=0.15, gt=0.0, lt=0.5)
    validation_size: float = Field(default=0.15, gt=0.0, lt=0.5)
    model: Literal["logistic", "vfl-hist-gbdt"] = "logistic"
    learning_rate: float = Field(default=0.1, gt=0.0)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=512, ge=1)
    l2: float = Field(default=1e-4, ge=0.0)
    max_depth: int = Field(default=3, ge=1, le=10)
    n_estimators: int = Field(default=50, ge=1)
    max_bins: int = Field(default=32, ge=4, le=255)
    min_samples_leaf: int = Field(default=20, ge=2)


class PathsConfig(BaseModel):
    """Repository-local paths used by commands."""

    root: Path = Path(".")
    data: Path = Path("data")
    reports: Path = Path("reports")
    results: Path = Path("results")
    runs: Path = Path("runs")
