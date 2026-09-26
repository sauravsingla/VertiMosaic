"""Party-local preprocessing fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class LocalPreprocessor:
    transformer: ColumnTransformer | None = None

    def fit(self, frame: pd.DataFrame) -> LocalPreprocessor:
        numeric = list(frame.select_dtypes(include=np.number).columns)
        categorical = [c for c in frame.columns if c not in numeric]
        self.transformer = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                    ),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        return np.asarray(self.transformer.transform(frame), dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def feature_names(self) -> tuple[str, ...]:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        return tuple(map(str, self.transformer.get_feature_names_out()))
