"""
preprocessing/imbalance_handler.py — Class-imbalance treatment strategies.

Strategies
----------
  • None             no resampling, no reweighting (plain baseline)
  • ClassWeight      cost-sensitive learning: inverse-frequency class weights
                     (scale_pos_weight / class_weight / auto_class_weights);
                     implemented on the model side, see models.build_model
  • SMOTE            (Chawla et al., 2002)
  • BorderlineSMOTE  (Han et al., 2005)
  • SVMSMOTE         (Nguyen et al., 2011)
  • ADASYN           (He et al., 2008)

Nominal columns are never interpolated. Oversamplers operate on the continuous
block only; each synthetic row receives the nominal values that are most
frequent among its k nearest original minority neighbours (the SMOTE-NC rule,
Chawla et al. 2002, §6.1), so every synthetic row carries a real category code.

All samplers are exposed through :class:`NominalAwareOverSampler`, an
imbalanced-learn compatible sampler, so they can sit inside an
``imblearn.pipeline.Pipeline`` and be re-fitted inside every cross-validation
fold (no synthetic rows ever reach a validation fold).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.base import BaseSampler
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE
from sklearn.neighbors import NearestNeighbors

log = logging.getLogger(__name__)

SAMPLER_REGISTRY: Dict[str, Optional[type]] = {
    "SMOTE"           : SMOTE,
    "BorderlineSMOTE" : BorderlineSMOTE,
    "SVMSMOTE"        : SVMSMOTE,
    "ADASYN"          : ADASYN,
    "None"            : None,
    "ClassWeight"     : None,
}
RESAMPLING_STRATEGIES = ("SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN")


class NominalAwareOverSampler(BaseSampler):
    """
    Wrap any imbalanced-learn oversampler so nominal columns are not interpolated.

    Parameters
    ----------
    strategy : name from SAMPLER_REGISTRY. "None"/"ClassWeight" pass data through.
    nominal_indices : positional indices of nominal (integer-coded) columns.
    sampler_kwargs : forwarded to the base sampler constructor.
    k_neighbors_nominal : neighbours used to vote the nominal values of a
        synthetic row (mode of the k nearest original minority rows).
    """

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}

    def __init__(
        self,
        strategy: str = "SMOTE",
        nominal_indices: Sequence[int] = (),
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        k_neighbors_nominal: int = 5,
        sampling_strategy="auto",
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.strategy = strategy
        self.nominal_indices = nominal_indices
        self.sampler_kwargs = sampler_kwargs
        self.k_neighbors_nominal = k_neighbors_nominal

    def _check_X_y(self, X, y, accept_sparse=None):
        # Keep pandas containers; nothing here needs sparse support.
        y, binarize_y = self._check_y_binary(y)
        return X, y, binarize_y

    @staticmethod
    def _check_y_binary(y):
        from imblearn.utils import check_target_type
        return check_target_type(y, indicate_one_vs_all=True)

    def fit_resample(self, X, y):
        if self.strategy not in SAMPLER_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Choose from {list(SAMPLER_REGISTRY)}."
            )
        if SAMPLER_REGISTRY[self.strategy] is None:
            return X, y
        columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        index_name = y.name if isinstance(y, pd.Series) else None
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)

        nominal = np.asarray(sorted(set(self.nominal_indices)), dtype=int)
        continuous = np.asarray([i for i in range(X_arr.shape[1]) if i not in set(nominal.tolist())], dtype=int)
        if continuous.size == 0:
            raise ValueError("Oversampling requires at least one continuous column.")

        sampler = SAMPLER_REGISTRY[self.strategy](**(self.sampler_kwargs or {}))
        X_cont_res, y_res = sampler.fit_resample(X_arr[:, continuous], y_arr)
        y_res = np.asarray(y_res).astype(int)

        n_original = len(y_arr)
        n_synthetic = len(y_res) - n_original
        X_res = np.empty((len(y_res), X_arr.shape[1]), dtype=float)
        X_res[:, continuous] = X_cont_res
        X_res[:n_original, nominal] = X_arr[:, nominal]

        if n_synthetic > 0 and nominal.size > 0:
            synthetic_cont = X_cont_res[n_original:]
            synthetic_y = y_res[n_original:]
            for cls in np.unique(synthetic_y):
                real_mask = y_arr == cls
                syn_mask = synthetic_y == cls
                real_cont = X_arr[real_mask][:, continuous]
                real_nom = X_arr[real_mask][:, nominal]
                k = int(min(self.k_neighbors_nominal, len(real_cont)))
                nn = NearestNeighbors(n_neighbors=k).fit(real_cont)
                _, neighbours = nn.kneighbors(synthetic_cont[syn_mask])
                votes = real_nom[neighbours]                      # (n_syn, k, n_nom)
                modes = np.empty((votes.shape[0], votes.shape[2]))
                for j in range(votes.shape[2]):
                    col = votes[:, :, j].astype(int)
                    for r in range(col.shape[0]):
                        vals, counts = np.unique(col[r], return_counts=True)
                        # tie-break towards the nearest neighbour's value
                        best = vals[counts == counts.max()]
                        modes[r, j] = col[r, 0] if col[r, 0] in best else best[0]
                rows = np.where(syn_mask)[0] + n_original
                X_res[np.ix_(rows, nominal)] = modes
        elif n_synthetic > 0:
            pass  # no nominal columns: nothing to restore

        if columns is not None:
            X_out = pd.DataFrame(X_res, columns=columns)
            X_out = X_out.astype({columns[i]: "int32" for i in nominal})
            return X_out, pd.Series(y_res, name=index_name)
        return X_res, y_res

    def _fit_resample(self, X, y):  # pragma: no cover - BaseSampler API
        return self.fit_resample(X, y)


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "SMOTE",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    nominal_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply one strategy to a training partition (nominal-aware)."""
    if strategy not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(SAMPLER_REGISTRY.keys())}."
        )
    _log_distribution("Before resampling", y_train)
    if SAMPLER_REGISTRY[strategy] is None:
        log.info("Strategy '%s': no resampling applied.", strategy)
        return X_train, y_train
    nominal_idx = [X_train.columns.get_loc(c) for c in (nominal_columns or []) if c in X_train.columns]
    sampler = NominalAwareOverSampler(strategy, nominal_idx, sampler_kwargs)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    y_res = pd.Series(np.asarray(y_res).astype(int), name=y_train.name)
    _log_distribution(f"After {strategy}", y_res)
    return X_res, y_res


def _log_distribution(label: str, y: pd.Series) -> None:
    counts = pd.Series(y).value_counts().sort_index()
    if len(counts) < 2:
        log.info("%s | single class present (n=%d)", label, len(y))
        return
    ir = counts.iloc[0] / counts.iloc[1]
    log.info(
        "%s | class 0: %d (%.1f%%)  class 1: %d (%.1f%%)  IR=%.2f",
        label, counts.iloc[0], counts.iloc[0] / len(y) * 100,
        counts.iloc[1], counts.iloc[1] / len(y) * 100, ir,
    )


def imbalance_statistics(y: pd.Series) -> Dict[str, float]:
    """Imbalance summary for Table I."""
    counts = pd.Series(y).value_counts().sort_index()
    n_maj = int(counts.iloc[0])
    n_min = int(counts.iloc[1])
    return {
        "n_total"         : int(len(y)),
        "n_majority"      : n_maj,
        "n_minority"      : n_min,
        "imbalance_ratio" : round(n_maj / n_min, 4),
        "minority_pct"    : round(n_min / len(y) * 100, 2),
    }
