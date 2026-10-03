"""Protocol v3 regression tests: nominal handling, censoring-aware labels,
rolling-origin splits, fold-local resampling, and OOF helpers.

Run:  .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.data_loader import (  # noqa: E402
    build_prosper_horizon_label, load_dataset, _encode_nominals, ORIGINATION_COL, OBSERVABLE_COL,
)
from preprocessing.imbalance_handler import NominalAwareOverSampler, apply_resampling  # noqa: E402
from preprocessing.tabular_preprocessor import TabularPreprocessor  # noqa: E402
from evaluation.metrics import oof_predict_proba, optimal_threshold, friedman_nemenyi, delong_test  # noqa: E402
from evaluation.revision_analysis import perturb_test_set, run_controlled_ablation  # noqa: E402


# ── Nominal-aware oversampling ────────────────────────────────────────────────

def _toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "cat": rng.integers(0, 4, n)})
    y = pd.Series((rng.random(n) < 0.15).astype(int), name="y")
    return X, y


@pytest.mark.parametrize("strategy", ["SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN"])
def test_synthetic_rows_keep_real_category_codes(strategy):
    X, y = _toy()
    Xr, yr = apply_resampling(X, y, strategy, {"random_state": 0}, nominal_columns=["cat"])
    assert len(yr) > len(y)
    assert set(Xr["cat"].unique()) <= set(X["cat"].unique())
    assert Xr["cat"].dtype.kind in "iu"
    # original rows are untouched and come first
    pd.testing.assert_frame_equal(Xr.iloc[: len(X)].reset_index(drop=True).astype(float),
                                  X.reset_index(drop=True).astype(float))


def test_passthrough_strategies_do_not_change_data():
    X, y = _toy()
    for strategy in ("None", "ClassWeight"):
        Xr, yr = apply_resampling(X, y, strategy, None, nominal_columns=["cat"])
        assert len(yr) == len(y)


def test_sampler_is_clonable_and_fold_local():
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from imblearn.pipeline import Pipeline

    X, y = _toy()
    sampler = NominalAwareOverSampler("SMOTE", [2], {"random_state": 0})
    clone(sampler)
    pipe = Pipeline([("s", sampler), ("m", LogisticRegression(max_iter=500))])
    proba, covered = oof_predict_proba(pipe, X, y, StratifiedKFold(3, shuffle=True, random_state=0))
    assert covered.all() and proba.shape == (len(y),)
    # validation folds are never resampled: OOF vector has exactly one score per original row
    assert np.isfinite(proba).all()


# ── Prosper horizon label ─────────────────────────────────────────────────────

def _prosper_frame():
    return pd.DataFrame({
        "LoanOriginationDate": ["2011-01-15", "2011-01-15", "2013-06-01", "2011-01-15", "2011-01-15", "2013-09-01"],
        "LoanStatus": ["Defaulted", "Chargedoff", "Current", "Completed", "Current", "Completed"],
        "ClosedDate": ["2011-09-01", "2013-02-01", None, "2012-01-01", None, "2014-01-01"],
        "LoanFirstDefaultedCycleNumber": [7, 24, None, None, None, None],
    })


def test_prosper_label_positive_negative_censored():
    label, snapshot = build_prosper_horizon_label(_prosper_frame(), 12, snapshot=pd.Timestamp("2014-03-12"))
    assert label.tolist() == [1, 0, -1, 0, 0, 0]


def test_prosper_default_with_unknown_time_is_censored():
    frame = _prosper_frame()
    frame.loc[0, "LoanFirstDefaultedCycleNumber"] = None
    frame.loc[0, "ClosedDate"] = None
    label, _ = build_prosper_horizon_label(frame, 12, snapshot=pd.Timestamp("2014-03-12"))
    assert label.iloc[0] == -1
    # defaulted after the horizon counts as a negative (observed >= 12 months)
    # still-open loan with < 12 months history is censored
    # completed loans are resolved negatives regardless of age


def test_prosper_label_falls_back_to_closed_date_when_cycle_missing():
    frame = _prosper_frame()
    frame.loc[0, "LoanFirstDefaultedCycleNumber"] = None
    label, _ = build_prosper_horizon_label(frame, 12, snapshot=pd.Timestamp("2014-03-12"))
    assert label.iloc[0] == 1


# ── Rolling-origin split on the real Prosper file (skipped when absent) ───────

@pytest.mark.skipif(not (ROOT / "data" / "prosperLoanData.csv").exists(), reason="prosper file missing")
def test_prosper_temporal_split_is_embargoed_and_disjoint():
    loaded = load_dataset("prosper", data_dir=ROOT / "data", split_mode="temporal", cutoff="2011-07",
                          test_window_months=6, horizon_months=12)
    meta = loaded.meta
    assert pd.Timestamp(meta["train_origination_range"][1]) <= pd.Timestamp("2010-07-31")
    assert pd.Timestamp(meta["test_origination_range"][0]) >= pd.Timestamp("2011-07-01")
    assert pd.Timestamp(meta["test_origination_range"][1]) < pd.Timestamp("2012-01-01")
    assert not set(loaded.X_train.index) & set(loaded.X_test.index)
    assert set(loaded.nominal_columns) <= set(loaded.X_train.columns)
    for col in loaded.nominal_columns:
        assert loaded.X_train[col].dtype.kind in "iu"
    # raw features keep NaNs; the fold-local preprocessor removes them
    prep = TabularPreprocessor(**loaded.prep_spec).fit(loaded.X_train)
    Xt = prep.transform(loaded.X_test)
    assert Xt.isna().sum().sum() == 0
    assert Xt.columns.tolist() == loaded.feature_names
    assert Xt.columns[-len(loaded.nominal_columns):].tolist() == loaded.nominal_columns
    assert "MonthlyLoanPayment" not in loaded.X_train.columns and "BorrowerRate" not in loaded.X_train.columns


@pytest.mark.skipif(not (ROOT / "data" / "prosperLoanData.csv").exists(), reason="prosper file missing")
def test_prosper_rejects_unobservable_test_window():
    with pytest.raises(ValueError, match="not fully observable"):
        load_dataset("prosper", data_dir=ROOT / "data", split_mode="temporal", cutoff="2013-06",
                     test_window_months=6, horizon_months=12)


# ── South German: nominal codes, age_group, no age as fairness group ──────────

@pytest.mark.skipif(not (ROOT / "data" / "SouthGermanCredit.asc").exists(), reason="south german file missing")
def test_south_german_groups_are_categorical_only():
    loaded = load_dataset("south_german", data_dir=ROOT / "data", random_state=1)
    assert "age" not in loaded.groups_train.columns
    assert set(loaded.groups_train["age_group"].unique()) <= {1, 2}
    assert loaded.groups_train.shape[0] == loaded.X_train.shape[0]
    assert len(loaded.nominal_columns) == 13
    for col in loaded.nominal_columns:
        assert loaded.X_train[col].dtype.kind in "iu"
        assert loaded.X_train[col].min() >= 0


def test_preprocessor_is_fold_local_and_layout_stable():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "cat": rng.integers(0, 3, 100)})
    X.loc[::10, "a"] = np.nan
    spec = {"continuous_columns": ["a", "b"], "indicator_columns": ["a"], "nominal_columns": ["cat"]}
    p1 = TabularPreprocessor(**spec).fit(X.iloc[:50])
    p2 = TabularPreprocessor(**spec).fit(X.iloc[50:])
    assert p1.medians_["a"] != p2.medians_["a"] or p1.mean_["b"] != p2.mean_["b"]   # statistics are fit-specific
    out = p1.transform(X)
    assert out.columns.tolist() == ["a", "b", "a_missing", "cat"]
    assert out.isna().sum().sum() == 0 and out["cat"].dtype.kind in "iu"
    assert out["a_missing"].sum() == X["a"].isna().sum()
    assert abs(p1.transform(X.iloc[:50])["b"].mean()) < 1e-9          # standardised on its own fit
    from sklearn.base import clone
    clone(p1)


def test_unknown_categories_map_to_extra_code():
    train = pd.DataFrame({"c": ["x", "y", "x"]})
    test = pd.DataFrame({"c": ["y", "z", None]})
    (tr, te), cats = _encode_nominals([train, test], ["c"])
    assert cats["c"] == ["x", "y"]
    assert te["c"].tolist() == [1, 2, 2]


# ── Perturbations and thresholds ──────────────────────────────────────────────

def test_nominal_columns_never_perturbed():
    X, _ = _toy(200)
    for kind in ("missingness", "correlated_missingness", "numeric_noise", "covariate_shift"):
        Xp = perturb_test_set(X, X, kind, 0.2, 3, nominal_columns=["cat"])
        assert (Xp["cat"].to_numpy() == X["cat"].to_numpy()).all(), kind


def test_cost_threshold_and_ablation_use_only_oof():
    rng = np.random.default_rng(1)
    y_oof = rng.integers(0, 2, 400)
    oof = np.clip(0.5 * y_oof + 0.5 * rng.random(400), 0, 1)
    y_test = rng.integers(0, 2, 200)
    test = np.clip(0.5 * y_test + 0.5 * rng.random(200), 0, 1)
    t, _ = optimal_threshold(y_oof, oof, "cost", cost_fp=1, cost_fn=5)
    assert 0.05 <= t < 0.95
    df = run_controlled_ablation({"None": {"oof_proba": oof, "y_oof": y_oof, "test_proba": test}}, y_test, "cost")
    assert set(df["threshold_rule"]) == {"fixed_0.50", "oof_cost", "isotonic_oof"}


def test_delong_symmetric_and_friedman_ranks():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    a = np.clip(0.6 * y + 0.4 * rng.random(300), 0, 1)
    b = np.clip(0.3 * y + 0.7 * rng.random(300), 0, 1)
    ab, ba = delong_test(y, a, b), delong_test(y, b, a)
    assert np.isclose(ab["z"], -ba["z"]) and ab["p_value"] < 0.05
    out = friedman_nemenyi(pd.DataFrame({"m1": [.8, .82, .81, .79], "m2": [.7, .71, .69, .72], "m3": [.75, .76, .74, .77]}))
    assert list(out["average_rank"]) == ["m1", "m3", "m2"]
