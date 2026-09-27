"""
models/gradient_boosting.py — Gradient boosting and ensemble classifier wrappers.

Wraps XGBoost, LightGBM, CatBoost, and sklearn baselines in a unified
sklearn-compatible API so the pipeline can swap them without branching logic.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log = logging.getLogger(__name__)


# ── Lazy imports (optional heavy deps) ───────────────────────────────────────

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


# ── XGBoost wrapper ───────────────────────────────────────────────────────────

class XGBoostClassifier(ClassifierMixin, BaseEstimator):
    """
    Thin sklearn-compatible wrapper around XGBClassifier with early stopping
    support and IEEE-table-ready feature importance extraction.
    """

    def __init__(self, **params):
        from config import XGBOOST_PARAMS
        self.params = {**XGBOOST_PARAMS, **params}
        self._model  = None

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
    ) -> "XGBoostClassifier":
        xgb = _import_xgb()
        kw  = dict(self.params)
        early = kw.pop("early_stopping_rounds", None)

        self._model = xgb.XGBClassifier(**kw)

        if X_val is not None and early:
            self._model.set_params(early_stopping_rounds=early)
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self._model.fit(X_train, y_train)

        log.info("XGBoost trained | best_iteration=%s",
                 getattr(self._model, "best_iteration", "N/A"))
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

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


# ── LightGBM wrapper ──────────────────────────────────────────────────────────

class LGBMClassifierWrapper(ClassifierMixin, BaseEstimator):
    """LightGBM wrapper with early stopping via callbacks."""

    def __init__(self, **params):
        from config import LGBM_PARAMS
        self.params = {**LGBM_PARAMS, **params}
        self._model  = None

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
    ) -> "LGBMClassifierWrapper":
        lgb = _import_lgb()
        kw  = dict(self.params)

        self._model = lgb.LGBMClassifier(**kw)

        callbacks = [lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)]

        if X_val is not None:
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self._model.fit(X_train, y_train)

        log.info("LightGBM trained | best_iteration=%s",
                 getattr(self._model, "best_iteration_", "N/A"))
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def classes_(self):
        return self._model.classes_

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self._model, f)


# ── CatBoost wrapper ──────────────────────────────────────────────────────────

class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """CatBoost wrapper."""

    def __init__(self, **params):
        from config import CATBOOST_PARAMS
        self.params = {**CATBOOST_PARAMS, **params}
        self._model  = None

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(
        self,
        X_train, y_train,
        X_val=None, y_val=None,
    ) -> "CatBoostWrapper":
        cb  = _import_cat()
        kw  = dict(self.params)
        early = kw.pop("early_stopping_rounds", None)
        if X_val is not None and early:
            kw["early_stopping_rounds"] = early
        self._model = cb.CatBoostClassifier(**kw)

        if X_val is not None:
            self._model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
        else:
            self._model.fit(X_train, y_train)

        log.info("CatBoost trained")
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.get_feature_importance()

    @property
    def classes_(self):
        return self._model.classes_

    def save(self, path: Path):
        self._model.save_model(str(path))


# ── Baseline models ───────────────────────────────────────────────────────────

class LogisticRegressionWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state: int = 42, **kwargs):
        self.params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": random_state,
            **kwargs,
        }
        self._model = LogisticRegression(
            **self.params
        )

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y, **_):
        self._model = LogisticRegression(**self.params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return np.abs(self._model.coef_[0])

    @property
    def classes_(self):
        return self._model.classes_


class RandomForestWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, random_state: int = 42, **kwargs):
        self.params = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": random_state,
            **kwargs,
        }
        self._model = RandomForestClassifier(
            **self.params
        )

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, X, y, **_):
        self._model = RandomForestClassifier(**self.params)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def feature_importances_(self):
        return self._model.feature_importances_

    @property
    def classes_(self):
        return self._model.classes_


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
