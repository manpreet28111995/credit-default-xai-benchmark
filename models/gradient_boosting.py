"""
models/gradient_boosting.py — Gradient boosting and baseline classifier wrappers.

Unified sklearn-compatible API so the pipeline (and imbalanced-learn
Pipelines inside RandomizedSearchCV) can swap models without branching.

Protocol v3 additions
---------------------
* Early stopping never touches data outside the fit call: when no explicit
  eval set is supplied, each fit carves a stratified
  ``EARLY_STOPPING_FRACTION`` slice from its *own* training data. This keeps
  early stopping fold-local inside cross-validation.
* ``cost_sensitive=True`` switches on inverse-frequency class weighting
  (``scale_pos_weight`` for XGBoost/LightGBM, ``auto_class_weights`` for
  CatBoost, ``class_weight`` for scikit-learn models). The default is *no*
  weighting so that the "None" imbalance arm is a genuine plain baseline.
* ``nominal_indices`` lets the logistic-regression wrapper one-hot encode
  integer-coded nominal columns internally; tree models split on the codes.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder

log = logging.getLogger(__name__)


def _import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")


def _import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        raise ImportError("Install lightgbm: pip install lightgbm")


def _import_cat():
    try:
        import catboost as cb
        return cb
    except ImportError:
        raise ImportError("Install catboost: pip install catboost")


def positive_class_weight(y) -> float:
    """Inverse-frequency weight for the positive class (n_neg / n_pos)."""
    y = np.asarray(y).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos)


class _WrapperBase(ClassifierMixin, BaseEstimator):
    """Shared plumbing: parameter dict, fold-local early-stopping split."""

    DEFAULTS: dict = {}
    WRAPPER_KEYS = ("cost_sensitive", "nominal_indices", "early_stopping_fraction")

    def __init__(self, **params):
        self.params = {**self.DEFAULTS, **params}
        self._model = None

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def _wrapper_options(self) -> dict:
        from config import EARLY_STOPPING_FRACTION
        return {
            "cost_sensitive": bool(self.params.get("cost_sensitive", False)),
            "nominal_indices": list(self.params.get("nominal_indices") or []),
            "early_stopping_fraction": float(
                self.params.get("early_stopping_fraction", EARLY_STOPPING_FRACTION)
            ),
        }

    def _model_params(self) -> dict:
        return {k: v for k, v in self.params.items() if k not in self.WRAPPER_KEYS}

    def _early_stopping_split(self, X, y, fraction: float, seed: int):
        """Stratified slice of the fit's own data used only for early stopping."""
        y_arr = np.asarray(y).astype(int)
        if fraction <= 0 or len(y_arr) < 50 or min(np.bincount(y_arr)) < 5:
            return X, y, None, None
        idx_fit, idx_es = train_test_split(
            np.arange(len(y_arr)), test_size=fraction, stratify=y_arr, random_state=seed,
        )
        take = (lambda A, idx: A.iloc[idx]) if hasattr(X, "iloc") else (lambda A, idx: A[idx])
        return take(X, idx_fit), y_arr[idx_fit], take(X, idx_es), y_arr[idx_es]

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def classes_(self):
        return self._model.classes_

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        return self


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostClassifier(_WrapperBase):
    """XGBClassifier with fold-local early stopping and optional class weighting."""

    def __init__(self, **params):
        from config import XGBOOST_PARAMS
        self.DEFAULTS = XGBOOST_PARAMS
        super().__init__(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostClassifier":
        xgb = _import_xgb()
        opts = self._wrapper_options()
        kw = self._model_params()
        early = kw.pop("early_stopping_rounds", None)
        if opts["cost_sensitive"]:
            kw["scale_pos_weight"] = positive_class_weight(y_train)

        if early and X_val is None:
            X_train, y_train, X_val, y_val = self._early_stopping_split(
                X_train, y_train, opts["early_stopping_fraction"], int(kw.get("random_state", 0)),
            )
        self._model = xgb.XGBClassifier(**kw)
        if X_val is not None and early:
            self._model.set_params(early_stopping_rounds=early)
            self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self._model.fit(X_train, y_train)
        log.debug("XGBoost trained | best_iteration=%s", getattr(self._model, "best_iteration", "N/A"))
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LGBMClassifierWrapper(_WrapperBase):
    """LightGBM with fold-local early stopping and optional class weighting."""

    def __init__(self, **params):
        from config import LGBM_PARAMS
        self.DEFAULTS = LGBM_PARAMS
        super().__init__(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LGBMClassifierWrapper":
        lgb = _import_lgb()
        opts = self._wrapper_options()
        kw = self._model_params()
        if opts["cost_sensitive"]:
            kw["scale_pos_weight"] = positive_class_weight(y_train)
        if X_val is None:
            X_train, y_train, X_val, y_val = self._early_stopping_split(
                X_train, y_train, opts["early_stopping_fraction"], int(kw.get("random_state", 0)),
            )
        self._model = lgb.LGBMClassifier(**kw)
        if X_val is not None:
            self._model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
            )
        else:
            self._model.fit(X_train, y_train)
        log.debug("LightGBM trained | best_iteration=%s", getattr(self._model, "best_iteration_", "N/A"))
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


# ── CatBoost ──────────────────────────────────────────────────────────────────

class CatBoostWrapper(_WrapperBase):
    """CatBoost with fold-local early stopping and optional class weighting."""

    def __init__(self, **params):
        from config import CATBOOST_PARAMS
        self.DEFAULTS = CATBOOST_PARAMS
        super().__init__(**params)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "CatBoostWrapper":
        cb = _import_cat()
        opts = self._wrapper_options()
        kw = self._model_params()
        early = kw.pop("early_stopping_rounds", None)
        if opts["cost_sensitive"]:
            kw["auto_class_weights"] = "Balanced"
        kw.setdefault("allow_writing_files", False)
        if early and X_val is None:
            X_train, y_train, X_val, y_val = self._early_stopping_split(
                X_train, y_train, opts["early_stopping_fraction"], int(kw.get("random_seed", 0)),
            )
        if X_val is not None and early:
            kw["early_stopping_rounds"] = early
        self._model = cb.CatBoostClassifier(**kw)
        if X_val is not None:
            self._model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        else:
            self._model.fit(X_train, y_train)
        log.debug("CatBoost trained")
        return self

    @property
    def feature_importances_(self):
        return self._model.get_feature_importance()

    def save(self, path: Path):
        self._model.save_model(str(path))


# ── Baselines ─────────────────────────────────────────────────────────────────

class LogisticRegressionWrapper(_WrapperBase):
    """L2 logistic regression; nominal codes are one-hot encoded internally."""

    DEFAULTS = {"max_iter": 2000, "solver": "lbfgs", "C": 1.0}

    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)

    def fit(self, X, y, **_):
        opts = self._wrapper_options()
        kw = self._model_params()
        if opts["cost_sensitive"]:
            kw["class_weight"] = "balanced"
        clf = LogisticRegression(**kw)
        if opts["nominal_indices"]:
            pre = ColumnTransformer(
                [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), opts["nominal_indices"])],
                remainder="passthrough", verbose_feature_names_out=False,
            )
            self._model = SkPipeline([("pre", pre), ("clf", clf)])
        else:
            self._model = clf
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=float))

    @property
    def feature_importances_(self):
        clf = self._model[-1] if isinstance(self._model, SkPipeline) else self._model
        return np.abs(clf.coef_[0])


class RandomForestWrapper(_WrapperBase):
    DEFAULTS = {"n_estimators": 300, "n_jobs": -1}

    def __init__(self, random_state: int = 42, **kwargs):
        super().__init__(random_state=random_state, **kwargs)

    def fit(self, X, y, **_):
        opts = self._wrapper_options()
        kw = self._model_params()
        if opts["cost_sensitive"]:
            kw["class_weight"] = "balanced"
        self._model = RandomForestClassifier(**kw)
        self._model.fit(X, y)
        return self

    @property
    def feature_importances_(self):
        return self._model.feature_importances_


# ── Factory ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "XGBoost"       : XGBoostClassifier,
    "LightGBM"      : LGBMClassifierWrapper,
    "CatBoost"      : CatBoostWrapper,
    "Logistic Reg." : LogisticRegressionWrapper,
    "Random Forest" : RandomForestWrapper,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](**kwargs)
