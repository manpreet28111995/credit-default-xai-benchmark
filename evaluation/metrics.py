"""
evaluation/metrics.py — Evaluation framework for imbalanced credit default prediction.

Implements all metrics required for Table II of the IEEE Access paper:
  • AUROC, AUPRC (Average Precision)
  • F1-Score, Precision, Recall, Specificity
  • G-Mean (geometric mean of sensitivity and specificity)
  • Matthews Correlation Coefficient (MCC)
  • Brier Score
  • Threshold-agnostic and threshold-sweep analysis
  • Cross-validation with stratified k-fold
  • McNemar's test and DeLong's test for model comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef,
    brier_score_loss, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2_contingency
import warnings

log = logging.getLogger(__name__)


# ── Core metric computation ───────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Return a flat dictionary of all evaluation metrics at a given threshold.

    Parameters
    ──────────
    y_true     : ground-truth binary labels
    y_pred     : hard predictions (0/1) — may be re-derived from y_proba
    y_proba    : predicted probability for class 1
    threshold  : decision boundary for converting proba → hard prediction
    """
    y_pred_t = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_t)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn + 1e-10)   # Recall
    specificity = tn / (tn + fp + 1e-10)
    g_mean      = np.sqrt(sensitivity * specificity)

    metrics = {
        "AUROC"      : roc_auc_score(y_true, y_proba),
        "AUPRC"      : average_precision_score(y_true, y_proba),
        "F1"         : f1_score(y_true, y_pred_t, zero_division=0),
        "Precision"  : precision_score(y_true, y_pred_t, zero_division=0),
        "Recall"     : sensitivity,
        "Specificity": specificity,
        "G-Mean"     : g_mean,
        "MCC"        : matthews_corrcoef(y_true, y_pred_t),
        "Brier"      : brier_score_loss(y_true, y_proba),
        "TP"         : int(tp),
        "FP"         : int(fp),
        "FN"         : int(fn),
        "TN"         : int(tn),
        "threshold"  : threshold,
    }
    # Full precision: rounding here would quantise bootstrap distributions.
    return {k: (int(v) if k in {"TP", "FP", "FN", "TN"} else float(v)) for k, v in metrics.items()}


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    criterion: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Tuple[float, float]:
    """
    Sweep probability thresholds and pick the one maximising `criterion`.

    Parameters
    ──────────
    criterion : 'f1' | 'g_mean' | 'youden' | 'cost'
                'cost' maximises expected utility per row:
                (TP - cost_fp*FP - cost_fn*FN) / n, consistent with the
                fairness--utility analysis.

    Returns
    ───────
    (best_threshold, best_score)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    thresholds = np.arange(0.01, 0.99, 0.01)
    scores     = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        cm       = confusion_matrix(y_true, y_pred_t)
        if cm.size < 4:
            scores.append(0.0)
            continue
        tn, fp, fn, tp = cm.ravel()
        sens    = tp / (tp + fn + 1e-10)
        spec    = tn / (tn + fp + 1e-10)

        if criterion == "f1":
            prec  = tp / (tp + fp + 1e-10)
            score = 2 * prec * sens / (prec + sens + 1e-10)
        elif criterion == "g_mean":
            score = np.sqrt(sens * spec)
        elif criterion == "youden":
            score = sens + spec - 1
        elif criterion == "cost":
            score = (tp - cost_fp * fp - cost_fn * fn) / max(len(y_true), 1)
        else:
            raise ValueError(f"Unknown criterion '{criterion}'.")
        scores.append(score)

    best_idx = np.argmax(scores)
    return float(thresholds[best_idx]), float(scores[best_idx])


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    model_cls,
    model_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42,
    fit_kwargs: Optional[dict] = None,
    resample_fn: Optional[
        Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
    ] = None,
) -> pd.DataFrame:
    """
    Stratified k-fold cross-validation with full metric suite per fold.

    If supplied, resample_fn is applied to the training partition inside
    each fold only. That keeps SMOTE-style resampling fold-local and avoids
    validation leakage.

    Returns
    ───────
    DataFrame where each row is one fold's metrics.
    """
    skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fit_kw   = fit_kwargs or {}
    records  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        if resample_fn is not None:
            X_tr, y_tr = resample_fn(X_tr, y_tr)

        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr, **fit_kw)

        y_prob = model.predict_proba(X_vl)[:, 1]
        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)
        m      = compute_metrics(y_vl.values, y_pred, y_prob, threshold=threshold)
        m["fold"] = fold
        records.append(m)
        log.info("Fold %d | AUROC=%.4f  F1=%.4f  G-Mean=%.4f",
                 fold, m["AUROC"], m["F1"], m["G-Mean"])

    df = pd.DataFrame(records)
    summary = df.drop(columns=["fold"]).agg(["mean", "std"])
    log.info("\nCV Summary:\n%s", summary.to_string())
    return df


# ── Model comparison ──────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's test for statistical significance between two binary classifiers.

    Returns chi2 statistic and p-value.
    Ref: McNemar, Q. (1947). Psychological Bulletin, 44(2), 153–167.
    """
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = (correct_a & ~correct_b).sum()
    c = (~correct_a & correct_b).sum()

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": int(b), "c": int(c)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)   # with continuity correction
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2, df=1)
    return {"chi2": float(chi2), "p_value": float(p_val), "b": int(b), "c": int(c)}


def delong_test(
    y_true: np.ndarray,
    y_proba_a: np.ndarray,
    y_proba_b: np.ndarray,
) -> Dict[str, float]:
    """
    DeLong's test for comparing two ROC curves (AUC difference).

    Implementation follows:
      Sun, X., & Xu, W. (2014). Fast Implementation of DeLong's Algorithm
      for Comparing the Areas Under Correlated Receiver Operating
      Characteristic Curves. IEEE Signal Processing Letters, 21(11).

    Returns z-statistic and two-tailed p-value.
    """
    from scipy.stats import norm

    def _structural_components(y_true, y_score):
        """Compute placement values for DeLong."""
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        m, n = len(pos), len(neg)

        # Placement values
        vp = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
        vn = np.array([np.mean(pos > p) + 0.5 * np.mean(pos == p) for p in neg])
        return vp, vn, m, n

    vp_a, vn_a, m, n = _structural_components(y_true, y_proba_a)
    vp_b, vn_b, _, _ = _structural_components(y_true, y_proba_b)

    auc_a = vp_a.mean()
    auc_b = vp_b.mean()

    # Covariance
    # Unbiased (ddof=1) estimates throughout, matching np.cov's default.
    s11 = (np.var(vp_a, ddof=1) / m + np.var(vn_a, ddof=1) / n)
    s22 = (np.var(vp_b, ddof=1) / m + np.var(vn_b, ddof=1) / n)
    s12 = (np.cov(vp_a, vp_b)[0, 1] / m + np.cov(vn_a, vn_b)[0, 1] / n)

    var_diff = s11 + s22 - 2 * s12
    if var_diff <= 0:
        return {"z": 0.0, "p_value": 1.0, "auc_a": auc_a, "auc_b": auc_b}

    z     = (auc_a - auc_b) / np.sqrt(var_diff)
    p_val = 2 * (1 - norm.cdf(abs(z)))

    return {
        "z"      : float(z),
        "p_value": float(p_val),
        "auc_a"  : float(auc_a),
        "auc_b"  : float(auc_b),
    }


# ── Results table builder ─────────────────────────────────────────────────────

def build_results_table(
    results: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convert a {model_name: metrics_dict} mapping into a publication-ready
    DataFrame (Table II layout).
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Keep only publication-relevant columns
    keep = ["AUROC", "AUPRC", "F1", "Precision", "Recall",
            "Specificity", "G-Mean", "MCC", "Brier"]
    df   = df[[c for c in keep if c in df.columns]]
    return df.round(4)


# ── Out-of-fold prediction (works for partition and forward-chaining splitters) ──

def oof_predict_proba(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    fit_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Out-of-fold positive-class probabilities.

    Unlike ``cross_val_predict`` this accepts splitters whose test folds do not
    partition the data (e.g. TimeSeriesSplit). Rows never held out receive NaN;
    the boolean ``covered`` mask marks rows with a prediction.

    The estimator is cloned per fold, so any sampler/preprocessing inside an
    imblearn Pipeline is re-fitted fold-locally.
    """
    from sklearn.base import clone

    y_arr = np.asarray(y).astype(int)
    proba = np.full(len(y_arr), np.nan, dtype=float)
    for train_idx, test_idx in cv.split(X, y_arr):
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y_arr[train_idx], **(fit_params or {}))
        proba[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]
    covered = ~np.isnan(proba)
    return proba, covered


# ── Cross-run model comparison (Demšar 2006) ──────────────────────────────────

# Studentised range statistic q_alpha (alpha = 0.05) divided by sqrt(2), k = 2..10
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman_nemenyi(
    scores: pd.DataFrame,
    higher_is_better: bool = True,
) -> Dict[str, object]:
    """
    Friedman test with Nemenyi critical difference across runs.

    Parameters
    ──────────
    scores : DataFrame indexed by run (seed or cutoff), one column per model.

    Returns
    ───────
    dict with average ranks, Friedman chi², p-value, Nemenyi CD (alpha 0.05),
    and the pairwise |rank difference| matrix flagged against the CD.
    """
    from scipy.stats import friedmanchisquare

    frame = scores.dropna(axis=0, how="any")
    n_runs, k = frame.shape
    if n_runs < 2 or k < 2:
        return {"status": "insufficient_runs_or_models", "n_runs": int(n_runs), "k": int(k)}
    ranks = frame.rank(axis=1, ascending=not higher_is_better)
    avg_rank = ranks.mean(axis=0).sort_values()
    if k == 2:
        chi2, p = np.nan, np.nan
    else:
        chi2, p = friedmanchisquare(*[frame[c].to_numpy() for c in frame.columns])
    q = _NEMENYI_Q05.get(int(k))
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n_runs))) if q else float("nan")
    diff = pd.DataFrame(
        np.abs(avg_rank.to_numpy()[:, None] - avg_rank.to_numpy()[None, :]),
        index=avg_rank.index, columns=avg_rank.index,
    )
    return {
        "status": "ok",
        "n_runs": int(n_runs), "k": int(k),
        "average_rank": avg_rank.to_dict(),
        "friedman_chi2": float(chi2) if np.isfinite(chi2) else None,
        "friedman_p": float(p) if np.isfinite(p) else None,
        "nemenyi_cd_alpha_0.05": cd,
        "pairwise_rank_difference": diff.round(4).to_dict(),
        "significant_pairs_alpha_0.05": [
            [a, b] for i, a in enumerate(avg_rank.index) for b in avg_rank.index[i + 1:]
            if np.isfinite(cd) and diff.loc[a, b] > cd
        ],
    }
