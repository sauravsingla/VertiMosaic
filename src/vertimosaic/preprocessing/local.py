from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


@dataclass
class LocalTabularPreprocessor:
    """Fit preprocessing locally using only one party's training rows."""

    numeric_columns: list[str]
    categorical_columns: list[str] = field(default_factory=list)
    scaling: str = "robust"
    winsor_quantile: float | None = None
    transformer_: ColumnTransformer | None = field(default=None, init=False)
    lower_bounds_: pd.Series | None = field(default=None, init=False)
    upper_bounds_: pd.Series | None = field(default=None, init=False)
    quantile_bin_edges_: dict[str, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    quantile_bin_count_: int | None = field(default=None, init=False, repr=False)

    def _validate(self) -> None:
        if self.scaling not in {"robust", "standard", "none"}:
            raise ValueError("scaling must be robust, standard or none")
        if self.winsor_quantile is not None and not 0.0 < self.winsor_quantile < 0.5:
            raise ValueError("winsor_quantile must be in (0, 0.5)")

    def _winsor_fit(self, frame: pd.DataFrame) -> None:
        if self.winsor_quantile is None or not self.numeric_columns:
            return
        q = self.winsor_quantile
        self.lower_bounds_ = frame[self.numeric_columns].quantile(q)
        self.upper_bounds_ = frame[self.numeric_columns].quantile(1.0 - q)

    def _winsor_apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            out.loc[:, self.numeric_columns] = out[self.numeric_columns].clip(
                lower=self.lower_bounds_, upper=self.upper_bounds_, axis="columns"
            )
        return out

    def fit(self, frame: pd.DataFrame) -> LocalTabularPreprocessor:
        self._validate()
        missing = (set(self.numeric_columns) | set(self.categorical_columns)) - set(frame.columns)
        if missing:
            raise ValueError(f"missing preprocessing columns: {sorted(missing)}")
        self._winsor_fit(frame)
        train = self._winsor_apply(frame)
        numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
        if self.scaling == "robust":
            numeric_steps.append(("scale", RobustScaler()))
        elif self.scaling == "standard":
            numeric_steps.append(("scale", StandardScaler()))
        transformers: list[tuple[str, object, list[str]]] = []
        if self.numeric_columns:
            transformers.append(("numeric", Pipeline(numeric_steps), self.numeric_columns))
        if self.categorical_columns:
            categorical = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                    ),
                ]
            )
            transformers.append(("categorical", categorical, self.categorical_columns))
        self.transformer_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        self.transformer_.fit(train)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer_ is None:
            raise RuntimeError("preprocessor is not fitted")
        transformed = self.transformer_.transform(self._winsor_apply(frame))
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def fit_quantile_bins(
        self,
        frame: pd.DataFrame,
        *,
        max_bins: int = 16,
    ) -> LocalTabularPreprocessor:
        """Fit party-local numeric quantile bins using only the supplied training frame.

        The learned cut points remain inside this party preprocessor. Missing values are
        assigned to a dedicated final bin during transformation. This helper is intended
        for local tree-model preprocessing and never pools quantiles across parties.
        """
        self._validate()
        if max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        missing = set(self.numeric_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing preprocessing columns: {sorted(missing)}")
        if self.winsor_quantile is not None and self.lower_bounds_ is None:
            self._winsor_fit(frame)
        local = self._winsor_apply(frame)
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        edges: dict[str, np.ndarray] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(local[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            edges[column] = (
                np.unique(np.quantile(finite, quantiles)).astype(float)
                if finite.size and quantiles.size
                else np.empty(0, dtype=float)
            )
        self.quantile_bin_edges_ = edges
        self.quantile_bin_count_ = max_bins
        return self

    def transform_quantile_bins(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply previously fitted party-local quantile bins to numeric columns."""
        if self.quantile_bin_count_ is None:
            raise RuntimeError("quantile bins are not fitted")
        missing = set(self.numeric_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"missing preprocessing columns: {sorted(missing)}")
        local = self._winsor_apply(frame)
        output = np.zeros((len(local), len(self.numeric_columns)), dtype=np.int32)
        for column_index, column in enumerate(self.numeric_columns):
            edges = self.quantile_bin_edges_[column]
            values = pd.to_numeric(local[column], errors="coerce").to_numpy(dtype=float)
            codes = np.searchsorted(edges, values, side="left").astype(np.int32)
            codes[~np.isfinite(values)] = len(edges)
            output[:, column_index] = codes
        return output

    def output_feature_metadata(self) -> list[dict[str, str]]:
        """Describe each transformed column without exposing any row-level values."""
        if self.transformer_ is None:
            raise RuntimeError("preprocessor is not fitted")
        output_names = [str(name) for name in self.transformer_.get_feature_names_out()]
        metadata: list[dict[str, str]] = []
        cursor = 0
        numeric_transform = "median imputation"
        if self.scaling != "none":
            numeric_transform += f" + {self.scaling} scaling"
        if self.winsor_quantile is not None:
            numeric_transform = f"winsorization + {numeric_transform}"
        for source_column in self.numeric_columns:
            metadata.append(
                {
                    "feature": output_names[cursor],
                    "source_column": source_column,
                    "transformation": numeric_transform,
                }
            )
            cursor += 1
        if self.categorical_columns:
            categorical_pipeline = self.transformer_.named_transformers_["categorical"]
            encoder = categorical_pipeline.named_steps["onehot"]
            for source_column, categories in zip(
                self.categorical_columns,
                encoder.categories_,
                strict=True,
            ):
                for _ in categories:
                    metadata.append(
                        {
                            "feature": output_names[cursor],
                            "source_column": source_column,
                            "transformation": (
                                "most-frequent imputation + one-hot encoding "
                                "(unknown categories ignored)"
                            ),
                        }
                    )
                    cursor += 1
        if cursor != len(output_names):
            raise RuntimeError("preprocessor feature metadata does not match transformed columns")
        return metadata

    def save(self, path: Path) -> None:
        if self.transformer_ is None:
            raise RuntimeError("preprocessor is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
