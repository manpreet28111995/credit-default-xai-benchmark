"""
preprocessing/imbalance_handler.py — Class-imbalance treatment strategies.

Implements and benchmarks:
  • SMOTE          (Chawla et al., 2002)
  • BorderlineSMOTE (Han et al., 2005)
  • SVM-SMOTE       (Nguyen et al., 2011)
  • ADASYN          (He et al., 2008)
  • Baseline (no resampling)

All samplers are wrapped in a consistent API that also logs before/after
class distributions to aid reporting in the paper.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import (
    SMOTE,
    BorderlineSMOTE,
    SVMSMOTE,
    ADASYN,
)

log = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────

SAMPLER_REGISTRY: Dict[str, type] = {
    "SMOTE"           : SMOTE,
    "BorderlineSMOTE" : BorderlineSMOTE,
    "SVMSMOTE"        : SVMSMOTE,
    "ADASYN"          : ADASYN,
    "None"            : None,               # passthrough
}


# ── Core function ─────────────────────────────────────────────────────────────

def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "SMOTE",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply the chosen over-sampling strategy to the training set.

    Parameters
    ──────────
    X_train         : Training features (DataFrame)
    y_train         : Training labels  (Series, binary 0/1)
    strategy        : One of 'SMOTE', 'BorderlineSMOTE', 'SVMSMOTE',
                      'ADASYN', or 'None'.
    sampler_kwargs  : Extra kwargs forwarded to the sampler constructor.

    Returns
    ───────
    X_res, y_res    : Resampled arrays wrapped back in DataFrame / Series.
    """
    if strategy not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from {list(SAMPLER_REGISTRY.keys())}."
        )

    _log_distribution("Before resampling", y_train)

    if strategy == "None" or SAMPLER_REGISTRY[strategy] is None:
        log.info("No resampling applied.")
        return X_train, y_train

    kwargs   = sampler_kwargs or {}
    sampler  = SAMPLER_REGISTRY[strategy](**kwargs)

    X_res_arr, y_res_arr = sampler.fit_resample(X_train.values, y_train.values)

    X_res = pd.DataFrame(X_res_arr, columns=X_train.columns)
    y_res = pd.Series(y_res_arr.astype(int), name=y_train.name)

    _log_distribution(f"After {strategy}", y_res)

    return X_res, y_res


def _log_distribution(label: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    ir     = counts.iloc[0] / counts.iloc[1]          # imbalance ratio (maj/min)
    log.info(
        "%s | class 0: %d (%.1f%%)  class 1: %d (%.1f%%)  IR=%.2f",
        label,
        counts.iloc[0], counts.iloc[0] / len(y) * 100,
        counts.iloc[1], counts.iloc[1] / len(y) * 100,
        ir,
    )


# ── Ablation: run all strategies ──────────────────────────────────────────────

def compare_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategies: Optional[list[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Run every resampling strategy and return a dict of resampled sets.
    Used by the ablation study in the pipeline.

    Returns
    ───────
    { strategy_name : (X_res, y_res) }
    """
    if strategies is None:
        strategies = list(SAMPLER_REGISTRY.keys())

    from config import SMOTE_STRATEGIES

    results: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for name in strategies:
        kwargs = SMOTE_STRATEGIES.get(name) or {}
        X_res, y_res = apply_resampling(X_train, y_train, strategy=name,
                                        sampler_kwargs=kwargs)
        results[name] = (X_res, y_res)

    return results


# ── Imbalance diagnostics ─────────────────────────────────────────────────────

def imbalance_statistics(y: pd.Series) -> Dict[str, float]:
    """
    Return a dictionary of imbalance metrics for inclusion in Table I of the
    paper.

    Metrics
    ───────
    n_total          Total sample count
    n_majority       Count of majority class (0)
    n_minority       Count of minority class (1)
    imbalance_ratio  n_majority / n_minority
    minority_pct     Fraction of minority class (%)
    """
    counts = y.value_counts().sort_index()
    n_maj  = int(counts.iloc[0])
    n_min  = int(counts.iloc[1])
    return {
        "n_total"         : len(y),
        "n_majority"      : n_maj,
        "n_minority"      : n_min,
        "imbalance_ratio" : round(n_maj / n_min, 4),
        "minority_pct"    : round(n_min / len(y) * 100, 2),
    }
