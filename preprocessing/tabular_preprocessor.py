"""
preprocessing/tabular_preprocessor.py — Fold-local preprocessing transformer.

Median imputation, missing indicators, 1st/99th-percentile winsorisation and
standardisation for continuous columns; nominal columns pass through as
integer codes. It is the first step of every imblearn Pipeline, so all of
these statistics are re-estimated inside each cross-validation fold and on
the full training partition for the final refit. No statistic ever sees a
validation fold or the test partition.

Output column order is fixed at construction time
(continuous, missing indicators, nominals) so positional indices used by the
nominal-aware sampler and TabNet embeddings are stable across folds.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        continuous_columns: Sequence[str] = (),
        indicator_columns: Sequence[str] = (),
        nominal_columns: Sequence[str] = (),
        winsor_quantiles: Tuple[float, float] = (0.01, 0.99),
        scale: bool = True,
    ):
        self.continuous_columns = continuous_columns
        self.indicator_columns = indicator_columns
        self.nominal_columns = nominal_columns
        self.winsor_quantiles = winsor_quantiles
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None):
        X = self._frame(X)
        cont = list(self.continuous_columns)
        self.medians_ = X[cont].median().fillna(0.0)
        filled = X[cont].fillna(self.medians_)
        lo, hi = self.winsor_quantiles
        self.lower_ = filled.quantile(lo)
        self.upper_ = filled.quantile(hi)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.mean_ = clipped.mean()
        std = clipped.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.scale_ = std
        self.feature_names_out_ = (
            cont + [f"{c}_missing" for c in self.indicator_columns] + list(self.nominal_columns)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._frame(X)
        cont = list(self.continuous_columns)
        out = pd.DataFrame(index=X.index)
        block = X[cont].astype("float64")
        missing = block.isna()
        block = block.fillna(self.medians_).clip(self.lower_, self.upper_, axis=1)
        if self.scale:
            block = (block - self.mean_) / self.scale_
        for c in cont:
            out[c] = block[c].to_numpy()
        for c in self.indicator_columns:
            out[f"{c}_missing"] = missing[c].to_numpy().astype("int8")
        for c in self.nominal_columns:
            out[c] = X[c].to_numpy().astype("int32")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        columns = list(self.continuous_columns) + list(self.nominal_columns)
        return pd.DataFrame(np.asarray(X), columns=columns)
