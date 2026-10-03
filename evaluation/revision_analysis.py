"""Reproducible analyses requested during manuscript revision.

The functions in this module are deliberately model-agnostic. They consume the
already-frozen test predictions and only use training/validation data for
imputation statistics, explanation references, and threshold selection.
"""

from __future__ import annotations

import itertools
import re
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from evaluation.metrics import compute_metrics, optimal_threshold


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else default


def perturb_test_set(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    kind: str,
    level: float,
    seed: int,
    protected_columns: Iterable[str] = ("SEX", "EDUCATION", "MARRIAGE"),
    nominal_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Create a deterministic, bounded test perturbation.

    Missing values use train-fold medians. Noise is expressed in train-fold
    standard deviations. Group columns and nominal (integer-coded) columns are
    never perturbed: Gaussian noise or an additive shift on a category code
    has no meaning.
    """
    out = X_test.copy()
    rng = np.random.default_rng(seed)
    protected = set(protected_columns) | set(nominal_columns)
    numeric = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    numeric = [c for c in numeric if c not in protected]

    if kind == "missingness":
        mask = rng.random((len(out), len(numeric))) < level
        medians = X_train[numeric].median().fillna(0.0)
        values = out[numeric].to_numpy(copy=True)
        values[mask] = np.broadcast_to(medians.to_numpy(), values.shape)[mask]
        out.loc[:, numeric] = values
    elif kind == "correlated_missingness":
        # A common operational failure: one statement/repayment block is absent
        # together. The mask is row-wise and group-wise, not iid per cell.
        groups = {}
        for column in numeric:
            key = "repayment" if any(token in column.upper() for token in ("PAY", "BILL", "UTIL")) else "other"
            groups.setdefault(key, []).append(column)
        medians = X_train[numeric].median().fillna(0.0)
        values = out[numeric].to_numpy(copy=True)
        for columns in groups.values():
            mask = rng.random(len(out)) < level
            indices = [numeric.index(column) for column in columns]
            values[np.ix_(mask, indices)] = medians[columns].to_numpy()
        out.loc[:, numeric] = values
    elif kind == "numeric_noise":
        scale = X_train[numeric].std(ddof=0).replace(0, 1.0).to_numpy()
        noise = rng.normal(0.0, level, size=(len(out), len(numeric))) * scale
        # Assign column-wise after widening integer/float32 columns. Pandas
        # otherwise rejects the valid float perturbation as a lossy cast.
        for index, column in enumerate(numeric):
            out[column] = out[column].astype(float).to_numpy() + noise[:, index]
    elif kind == "covariate_shift":
        # Simulate a portfolio mix shift in monetary/utilisation variables while
        # preserving the observed covariance structure of each row.
        shifted = [
            column for column in numeric
            if any(token in column.upper() for token in
                   ("BILL", "PAY", "LIMIT", "UTIL", "AMOUNT", "INCOME", "DTI", "FICO"))
        ] or numeric
        scale = X_train[shifted].std(ddof=0).replace(0, 1.0).fillna(1.0)
        for column in shifted:
            out[column] = out[column].astype(float) + float(level) * float(scale[column])
    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")
    return out


def evaluate_robustness(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    seed: int,
    levels: Iterable[float] = (0.05, 0.10, 0.20),
    nominal_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Evaluate performance retention under missingness and numeric noise."""
    base_proba = model.predict_proba(X_test)[:, 1]
    base = compute_metrics(
        y_test.to_numpy(),
        base_proba >= threshold,
        base_proba,
        threshold=threshold,
    )
    base_prediction = base_proba >= threshold
    rows = [{
        "perturbation": "none", "level": 0.0, "seed": seed, **base,
        "mean_abs_probability_shift": 0.0, "decision_flip_rate": 0.0,
    }]
    for kind in ("missingness", "correlated_missingness", "numeric_noise", "covariate_shift"):
        for level in levels:
            X_shift = perturb_test_set(X_train, X_test, kind, level, seed, nominal_columns=nominal_columns)
            proba = model.predict_proba(X_shift)[:, 1]
            metrics = compute_metrics(y_test.to_numpy(), proba >= threshold, proba, threshold)
            rows.append({
                "perturbation": kind, "level": level, "seed": seed, **metrics,
                "mean_abs_probability_shift": float(np.mean(np.abs(proba - base_proba))),
                "decision_flip_rate": float(np.mean((proba >= threshold) != base_prediction)),
            })
    result = pd.DataFrame(rows)
    base_auc = float(base["AUROC"])
    base_brier = max(float(base["Brier"]), 1e-12)
    result["AUROC_retention"] = result["AUROC"] / max(base_auc, 1e-12)
    result["Brier_retention"] = 1.0 - (result["Brier"] - base_brier) / base_brier
    return result


def _gini(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    total = values.sum()
    if total <= 0 or len(values) < 2:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, len(ordered) + 1)
    return float((2 * np.sum(index * ordered) - (len(ordered) + 1) * total) /
                 (len(ordered) * total))


def explanation_parsimony(
    values: np.ndarray,
    top_mass: float = 0.8,
) -> dict[str, float]:
    """Summarise attribution concentration and support size."""
    arr = np.abs(np.asarray(values, dtype=float))
    if arr.ndim == 1:
        arr = arr[None, :]
    ginis, supports = [], []
    for row in arr:
        total = row.sum()
        order = np.argsort(row)[::-1]
        ginis.append(_gini(row))
        if total <= 0:
            supports.append(len(row))
        else:
            supports.append(int(np.searchsorted(np.cumsum(row[order]), top_mass * total) + 1))
    return {
        "gini_mean": _safe_mean(ginis),
        "features_for_mass_mean": _safe_mean(supports),
        "parsimony_score": _safe_mean(ginis),
    }


def _lime_vector(series: pd.Series, feature_names: list[str]) -> np.ndarray:
    vector = np.zeros(len(feature_names), dtype=float)
    for condition, value in series.items():
        for i, name in enumerate(feature_names):
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", str(condition)):
                vector[i] += float(value)
                break
    return vector


def _pairwise_stability(matrix: np.ndarray, top_k: int = 10) -> tuple[float, float]:
    if len(matrix) < 2:
        return 1.0, 1.0
    correlations, overlaps = [], []
    for a, b in itertools.combinations(matrix, 2):
        rho = spearmanr(np.abs(a), np.abs(b)).statistic
        correlations.append(0.0 if not np.isfinite(rho) else rho)
        ia = set(np.argsort(np.abs(a))[-top_k:])
        ib = set(np.argsort(np.abs(b))[-top_k:])
        overlaps.append(len(ia & ib) / max(len(ia | ib), 1))
    return _safe_mean(correlations), _safe_mean(overlaps)


def explanation_metrics(
    shap_values: np.ndarray,
    lime_explanations: list[pd.Series],
    feature_names: list[str],
    repeated_lime: list[list[pd.Series]] | None = None,
    lime_scores: list[float] | None = None,
) -> pd.DataFrame:
    """Return formal SHAP/LIME parsimony, agreement, and stability metrics."""
    shap = np.asarray(shap_values, dtype=float)
    rows = []
    for i, lime in enumerate(lime_explanations[:len(shap)]):
        lime_vec = _lime_vector(lime, feature_names)
        shared = np.abs(lime_vec) > 0
        rho = spearmanr(np.abs(shap[i][shared]), np.abs(lime_vec[shared])).statistic if shared.sum() >= 3 else np.nan
        shap_p = explanation_parsimony(shap[i])
        lime_p = explanation_parsimony(lime_vec)
        row = {
            "instance": i,
            "shap_parsimony": shap_p["parsimony_score"],
            "shap_features_for_80pct": shap_p["features_for_mass_mean"],
            "lime_parsimony": lime_p["parsimony_score"],
            "lime_features_for_80pct": lime_p["features_for_mass_mean"],
            "shap_lime_abs_rank_rho": float(rho) if np.isfinite(rho) else np.nan,
            "shap_lime_signed_abs_attribution_rho": float(rho) if np.isfinite(rho) else np.nan,
            "lime_local_r2": (
                float(lime_scores[i]) if lime_scores is not None and i < len(lime_scores) else np.nan
            ),
        }
        if repeated_lime and i < len(repeated_lime):
            vectors = np.vstack([_lime_vector(x, feature_names) for x in repeated_lime[i]])
            row["lime_stability_rho"], row["lime_top10_jaccard"] = _pairwise_stability(vectors)
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_utility_curve(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    groups: pd.DataFrame,
    thresholds: Iterable[float] = np.arange(0.20, 0.61, 0.05),
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    benefit_tp: float = 1.0,
    benefit_tn: float = 0.0,
) -> pd.DataFrame:
    """Compute utility and group disparity at each candidate threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    rows = []
    for threshold in thresholds:
        metrics = compute_metrics(y, p >= threshold, p, float(threshold))
        row = {
            "threshold": float(threshold),
            "expected_utility": float(
                (metrics["TP"] * benefit_tp + metrics["TN"] * benefit_tn
                 - metrics["FP"] * cost_fp - metrics["FN"] * cost_fn) / max(len(y), 1)
            ),
            **metrics,
        }
        for column in groups.columns:
            values = groups[column].to_numpy()
            rates = []
            for group in sorted(pd.Series(values).dropna().unique(), key=str):
                mask = values == group
                pred = p[mask] >= threshold
                pos = y[mask] == 1
                neg = y[mask] == 0
                rates.append({
                    "selection": float(pred.mean()) if len(pred) else np.nan,
                    "tpr": float(pred[pos].mean()) if pos.any() else np.nan,
                    "fpr": float(pred[neg].mean()) if neg.any() else np.nan,
                })
            for rate_name in ("selection", "tpr", "fpr"):
                observed = [r[rate_name] for r in rates if np.isfinite(r[rate_name])]
                row[f"{column}_{rate_name}_gap"] = (
                    float(max(observed) - min(observed)) if len(observed) >= 2 else float("nan")
                )
        gap_columns = [key for key in row if key.endswith("_selection_gap") or key.endswith("_tpr_gap") or key.endswith("_fpr_gap")]
        valid_gaps = [float(row[key]) for key in gap_columns if np.isfinite(row[key])]
        row["max_group_gap"] = max(valid_gaps) if valid_gaps else float("nan")
        rows.append(row)
    return fairness_utility_pareto(pd.DataFrame(rows))


def calibration_diagnostics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return calibration/probability diagnostics and equal-width reliability bins.

    Brier score is reported as overall probability error. ECE and the logistic
    calibration intercept/slope are separate diagnostics; none is treated as a
    stand-alone proof that probabilities are calibrated.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("y_true and probabilities must be finite, non-empty, and equal length")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain binary 0/1 labels")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0 - eps)
    prevalence = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))
    base_brier = prevalence * (1.0 - prevalence)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    bins = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y[mask].mean())
        error = abs(mean_pred - observed)
        ece += count / len(y) * error
        mce = max(mce, error)
        bins.append({
            "bin": index,
            "bin_lower": float(edges[index]),
            "bin_upper": float(edges[index + 1]),
            "n": count,
            "mean_predicted": mean_pred,
            "observed_rate": observed,
            "absolute_gap": error,
        })

    logit_p = logit(p)
    def _fit_logistic(offset: bool) -> tuple[float, float]:
        design = np.column_stack([np.ones(len(y)), logit_p]) if not offset else np.ones((len(y), 1))
        initial = np.array([0.0, 1.0]) if not offset else np.array([0.0])
        def objective(params):
            eta = design @ params + (logit_p if offset else 0.0)
            return float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        result = minimize(objective, initial, method="L-BFGS-B", bounds=[(-30, 30)] * len(initial))
        if not result.success or not np.isfinite(result.fun):
            return float("nan"), float("nan")
        if offset:
            return float(result.x[0]), 1.0
        return float(result.x[0]), float(result.x[1])

    intercept, slope = _fit_logistic(offset=False)
    intercept_fixed_slope, _ = _fit_logistic(offset=True)
    summary = {
        "n": int(len(y)),
        "prevalence": prevalence,
        "brier": brier,
        "base_rate_brier": base_brier,
        "brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else float("nan"),
        "ece_equal_width": float(ece),
        "mce_equal_width": float(mce),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_in_the_large": intercept_fixed_slope,
    }
    return summary, pd.DataFrame(bins)


def _group_key(value: object) -> str:
    return "__MISSING__" if pd.isna(value) else str(value)


def fit_equal_opportunity_threshold_policy(
    y_validation: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    groups: pd.DataFrame,
    group_column: str,
    max_tpr_gap: float = 0.10,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    target_grid: Iterable[float] = np.linspace(0.0, 1.0, 101),
    threshold_candidates: int = 201,
) -> dict[str, object]:
    """Fit validation-only group thresholds for an equal-opportunity baseline.

    For each target TPR, choose each group's threshold nearest that target, then
    select the feasible policy with greatest validation utility. This is a
    transparent threshold post-processor, not a legal fairness guarantee.
    """
    y = np.asarray(y_validation, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError("validation labels, scores, and groups must have equal length")
    if group_column not in groups:
        raise KeyError(f"Unknown group column: {group_column}")
    values = groups[group_column].to_numpy()
    keys = np.asarray([_group_key(value) for value in values], dtype=object)
    levels = sorted(set(keys), key=str)
    fallback_threshold, _ = optimal_threshold(y, p, criterion="f1")
    candidates_by_group: dict[str, list[tuple[float, float, float]]] = {}
    for level in levels:
        mask = keys == level
        positives = int((y[mask] == 1).sum())
        negatives = int((y[mask] == 0).sum())
        if positives == 0 or negatives == 0:
            continue
        quantiles = np.linspace(0.0, 1.0, max(2, int(threshold_candidates)))
        thresholds = np.unique(np.concatenate(([-1e-12], np.quantile(p[mask], quantiles), [1.0 + 1e-12])))
        rows = []
        for threshold in thresholds:
            pred = p[mask] >= threshold
            tpr = float(pred[y[mask] == 1].mean())
            utility = float((pred[y[mask] == 1].sum() - cost_fp * pred[y[mask] == 0].sum()
                             - cost_fn * ((y[mask] == 1) & ~pred).sum()) / int(mask.sum()))
            rows.append((float(threshold), tpr, utility))
        candidates_by_group[level] = rows
    if len(candidates_by_group) < 2:
        return {
            "status": "insufficient_groups_with_both_classes",
            "group_column": group_column,
            "thresholds_by_group": {},
            "validation_tpr_gap": float("nan"),
            "validation_utility": float("nan"),
            "feasible": False,
            "max_tpr_gap": float(max_tpr_gap),
        }

    policies = []
    for target in target_grid:
        selected = {}
        group_tpr = {}
        for level, rows in candidates_by_group.items():
            chosen = min(rows, key=lambda row: (abs(row[1] - float(target)), -row[2], -row[0]))
            selected[level] = chosen[0]
            group_tpr[level] = chosen[1]
        full_thresholds = {key: float(fallback_threshold) for key in levels}
        full_thresholds.update(selected)
        threshold_vector = np.asarray([full_thresholds[key] for key in keys])
        pred = p >= threshold_vector
        yy = y
        tn = int(((yy == 0) & ~pred).sum())
        fp = int(((yy == 0) & pred).sum())
        fn = int(((yy == 1) & ~pred).sum())
        tp = int(((yy == 1) & pred).sum())
        utility = float((tp - cost_fp * fp - cost_fn * fn) / max(len(yy), 1))
        gap = float(max(group_tpr.values()) - min(group_tpr.values()))
        policies.append({
            "target_tpr": float(target), "thresholds": selected,
            "validation_tpr_gap": gap, "validation_utility": utility,
            "feasible": gap <= max_tpr_gap,
        })
    feasible = [row for row in policies if row["feasible"]]
    chosen = max(feasible, key=lambda row: (row["validation_utility"], -row["validation_tpr_gap"])) if feasible else min(
        policies, key=lambda row: (row["validation_tpr_gap"], -row["validation_utility"])
    )
    return {
        "status": "fit",
        "group_column": group_column,
        "validation_group_count": int(len(levels)),
        "validation_groups_compared": int(len(candidates_by_group)),
        "validation_groups_without_both_classes": sorted(set(levels) - set(candidates_by_group)),
        "target_tpr": chosen["target_tpr"],
        "thresholds_by_group": {**{key: float(fallback_threshold) for key in levels}, **chosen["thresholds"]},
        "fallback_threshold": float(fallback_threshold),
        "validation_tpr_gap": chosen["validation_tpr_gap"],
        "validation_utility": chosen["validation_utility"],
        "feasible": chosen["feasible"],
        "max_tpr_gap": float(max_tpr_gap),
        "validation_only_fit": True,
        "postprocessor": "group-specific thresholds targeting equal opportunity",
    }


def evaluate_group_threshold_policy(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    groups: pd.DataFrame,
    policy: Mapping[str, object],
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> dict[str, object]:
    """Evaluate a frozen group-threshold policy on held-out data."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    column = str(policy["group_column"])
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError("test labels, scores, and groups must have equal length")
    thresholds = {str(key): float(value) for key, value in dict(policy["thresholds_by_group"]).items()}
    keys = np.asarray([_group_key(value) for value in groups[column].to_numpy()], dtype=object)
    fallback = float(policy.get("fallback_threshold", 0.5))
    row_thresholds = np.asarray([thresholds.get(key, fallback) for key in keys], dtype=float)
    pred = p >= row_thresholds
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    observed_tpr = []
    observed_fpr = []
    observed_selection = []
    per_group = {}
    for level in sorted(set(keys), key=str):
        mask = keys == level
        pos, neg = mask & (y == 1), mask & (y == 0)
        tpr = float(pred[pos].mean()) if pos.any() else float("nan")
        fpr = float(pred[neg].mean()) if neg.any() else float("nan")
        selection = float(pred[mask].mean()) if mask.any() else float("nan")
        if np.isfinite(tpr): observed_tpr.append(tpr)
        if np.isfinite(fpr): observed_fpr.append(fpr)
        if np.isfinite(selection): observed_selection.append(selection)
        per_group[level] = {
            "n": int(mask.sum()), "tpr": tpr, "fpr": fpr,
            "selection_rate": selection,
        }
    test_gaps = []
    for observed in (observed_tpr, observed_fpr, observed_selection):
        if len(observed) >= 2:
            test_gaps.append(float(max(observed) - min(observed)))
    max_test_gap = max(test_gaps) if test_gaps else float("nan")
    return {
        "test_accuracy": float((pred == y).mean()),
        "test_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "test_brier": float(np.mean((p - y) ** 2)),
        "test_utility": float((tp - cost_fp * fp - cost_fn * fn) / max(len(y), 1)),
        "test_tpr_gap": float(max(observed_tpr) - min(observed_tpr)) if len(observed_tpr) >= 2 else float("nan"),
        "test_fpr_gap": float(max(observed_fpr) - min(observed_fpr)) if len(observed_fpr) >= 2 else float("nan"),
        "test_selection_gap": float(max(observed_selection) - min(observed_selection)) if len(observed_selection) >= 2 else float("nan"),
        "test_confusion_tn": tn, "test_confusion_fp": fp,
        "test_confusion_fn": fn, "test_confusion_tp": tp,
        "test_max_group_gap": max_test_gap,
        "fallback_group_count": int(sum(key not in thresholds for key in keys)),
        "thresholds_by_group": thresholds,
        "per_group": per_group,
        "test_predictions": pred,
    }


def bootstrap_fairness_policy_differences(
    y_true: pd.Series | np.ndarray,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    groups: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> pd.DataFrame:
    """Paired row-bootstrap CIs for post-processing changes in utility and gaps."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    reference = np.asarray(reference_predictions, dtype=bool).reshape(-1)
    candidate = np.asarray(candidate_predictions, dtype=bool).reshape(-1)
    if not (len(y) == len(reference) == len(candidate) == len(groups)):
        raise ValueError("labels, both prediction vectors, and groups must have equal length")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "utility", "max_group_gap"]
    for column in groups.columns:
        metrics.extend(f"{column}_{rate}_gap" for rate in ("selection", "tpr", "fpr"))
    draws = {metric: [] for metric in metrics}
    group_frame = groups.reset_index(drop=True)

    def snapshot(yy, prediction, gg):
        tn = int(((yy == 0) & ~prediction).sum())
        fp = int(((yy == 0) & prediction).sum())
        fn = int(((yy == 1) & ~prediction).sum())
        tp = int(((yy == 1) & prediction).sum())
        result = {
            "accuracy": float((yy == prediction).mean()),
            "utility": float((tp - cost_fp * fp - cost_fn * fn) / max(len(yy), 1)),
        }
        all_gaps = []
        for column in gg.columns:
            values = gg[column].to_numpy()
            gaps = {}
            for rate in ("selection", "tpr", "fpr"):
                observed = []
                for level in pd.unique(values):
                    mask = pd.isna(values) if pd.isna(level) else values == level
                    mask = np.asarray(mask, dtype=bool)
                    eligible = mask if rate == "selection" else mask & (yy == (1 if rate == "tpr" else 0))
                    if eligible.any():
                        observed.append(float(prediction[eligible].mean()))
                gap = float(max(observed) - min(observed)) if len(observed) >= 2 else float("nan")
                result[f"{column}_{rate}_gap"] = gap
                if np.isfinite(gap):
                    all_gaps.append(gap)
                gaps[rate] = gap
        result["max_group_gap"] = float(max(all_gaps)) if all_gaps else float("nan")
        return result

    for _ in range(int(n_boot)):
        index = rng.integers(0, len(y), len(y))
        yy = y[index]
        gg = group_frame.iloc[index]
        left = snapshot(yy, reference[index], gg)
        right = snapshot(yy, candidate[index], gg)
        for metric in metrics:
            if np.isfinite(left[metric]) and np.isfinite(right[metric]):
                draws[metric].append(right[metric] - left[metric])
    rows = []
    for metric, values in draws.items():
        if not values:
            continue
        rows.append({
            "metric": metric,
            "difference_candidate_minus_reference": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "valid_bootstrap_reps": int(len(values)),
            "requested_bootstrap_reps": int(n_boot),
            "seed": int(seed),
        })
    return pd.DataFrame(rows)

def fairness_utility_pareto(curve: pd.DataFrame) -> pd.DataFrame:
    """Mark thresholds that are not dominated in utility/fairness space."""
    out = curve.copy()
    out["pareto_optimal"] = False
    for i, current in out.iterrows():
        dominates = (
            (out["expected_utility"] >= current["expected_utility"])
            & (out["max_group_gap"] <= current["max_group_gap"])
            & ((out["expected_utility"] > current["expected_utility"])
               | (out["max_group_gap"] < current["max_group_gap"]))
        )
        out.loc[i, "pareto_optimal"] = not bool(dominates.any())
    return out


def reliability_constrained_selection(
    validation_rows: pd.DataFrame,
    min_robustness: float = 0.90,
    max_fairness_gap: float = 0.10,
) -> pd.DataFrame:
    """Rank candidates using separate ranking, Brier-skill, robustness, and gap terms."""
    out = validation_rows.copy()
    base_brier = float(out["base_rate_brier"].iloc[0])
    gap = pd.to_numeric(out["max_fairness_gap"], errors="coerce")
    out["ranking_component"] = np.clip(out["AUROC"].astype(float), 0.0, 1.0)
    out["probability_quality_component"] = np.clip(
        1.0 - out["Brier"].astype(float) / max(base_brier, 1e-12), 0.0, 1.0
    )
    out["robustness_component"] = np.clip(out["robustness_retention"].astype(float), 0.0, 1.0)
    out["inverse_group_gap_component"] = np.clip(1.0 - gap, 0.0, 1.0)
    components = ["ranking_component", "probability_quality_component",
                  "robustness_component", "inverse_group_gap_component"]
    matrix = out[components].to_numpy(dtype=float)
    score = np.prod(matrix, axis=1) ** (1.0 / len(components))
    out["reliability_score"] = np.where(np.isfinite(matrix).all(axis=1), score, np.nan)
    out["fairness_constraint_ok"] = gap.notna() & gap.le(max_fairness_gap)
    out["robustness_constraint_ok"] = out["robustness_retention"].astype(float).ge(min_robustness)
    out["feasible"] = out["fairness_constraint_ok"] & out["robustness_constraint_ok"]
    gap_violation = np.where(gap.notna(), np.maximum(gap - max_fairness_gap, 0.0), 1.0)
    out["constraint_violation"] = (
        gap_violation + np.maximum(min_robustness - out["robustness_retention"].astype(float), 0.0)
    )
    out = out.sort_values(
        ["feasible", "reliability_score", "constraint_violation"],
        ascending=[False, False, True], na_position="last",
    ).reset_index(drop=True)
    out["selected"] = False
    if not out.empty:
        out.loc[0, "selected"] = True
    out["selection_role"] = np.where(
        out["selected"] & out["feasible"], "feasible_candidate",
        np.where(out["selected"], "infeasible_fallback", "not_selected"),
    )
    return out


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Test-sample CIs conditional on fitted models and selected thresholds."""
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    bootstrap_indices = [rng.integers(0, len(y), len(y)) for _ in range(int(n_boot))]
    rows = []
    for model, proba in predictions.items():
        p = np.asarray(proba, dtype=float)
        values = {metric: [] for metric in ("AUROC", "AUPRC", "Brier", "F1", "MCC")}
        for index in bootstrap_indices:
            if len(np.unique(y[index])) < 2:
                continue
            metrics = compute_metrics(y[index], p[index] >= thresholds[model], p[index], thresholds[model])
            for metric in values:
                values[metric].append(float(metrics[metric]))
        for metric, samples in values.items():
            if not samples:
                continue
            rows.append({
                "model": model,
                "metric": metric,
                "estimate": float(np.mean(samples)),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_reps": len(samples),
                "inference_scope": "conditional_on_fitted_models_and_selected_thresholds",
            })
    return pd.DataFrame(rows)


def paired_bootstrap_differences(
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Conditional test-sample CIs; model and threshold selection are not repeated."""
    y = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    indices = [rng.integers(0, len(y), len(y)) for _ in range(int(n_boot))]
    rows = []
    names = list(predictions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            differences = {metric: [] for metric in ("AUROC", "AUPRC", "Brier", "F1", "MCC")}
            for index in indices:
                if len(np.unique(y[index])) < 2:
                    continue
                left_metrics = compute_metrics(
                    y[index], predictions[left][index] >= thresholds[left],
                    predictions[left][index], thresholds[left],
                )
                right_metrics = compute_metrics(
                    y[index], predictions[right][index] >= thresholds[right],
                    predictions[right][index], thresholds[right],
                )
                for metric in differences:
                    differences[metric].append(left_metrics[metric] - right_metrics[metric])
            for metric, samples in differences.items():
                if samples:
                    rows.append({
                        "model_a": left,
                        "model_b": right,
                        "metric": metric,
                        "difference_mean": float(np.mean(samples)),
                        "ci_low": float(np.quantile(samples, 0.025)),
                        "ci_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_reps": len(samples),
                        "inference_scope": "conditional_on_fitted_models_and_selected_thresholds",
                    })
    return pd.DataFrame(rows)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down multiplicity adjustment."""
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def explanation_fidelity(
    model,
    X: pd.DataFrame,
    X_reference: pd.DataFrame,
    attributions: np.ndarray,
    top_fraction: float = 0.20,
    nominal_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Measure deletion/insertion faithfulness of local attributions.

    The neutral baseline is the train median for continuous columns and the
    train mode for nominal columns (a median category code is meaningless).
    """
    if len(X) == 0:
        return pd.DataFrame()
    baseline = X_reference.median(numeric_only=True).reindex(X.columns).fillna(0.0)
    for column in nominal_columns:
        if column in X.columns:
            baseline[column] = float(X_reference[column].mode().iloc[0])
    median = baseline.to_numpy(dtype=float)
    rows = []
    for i, (row, attribution) in enumerate(zip(X.to_numpy(dtype=float), np.asarray(attributions))):
        k = max(1, int(round(len(row) * top_fraction)))
        top = np.argsort(np.abs(attribution))[-k:]
        full = float(model.predict_proba(pd.DataFrame([row], columns=X.columns))[:, 1][0])
        deleted = row.copy()
        deleted[top] = median[top]
        inserted = median.copy()
        inserted[top] = row[top]
        p_deleted = float(model.predict_proba(pd.DataFrame([deleted], columns=X.columns))[:, 1][0])
        p_inserted = float(model.predict_proba(pd.DataFrame([inserted], columns=X.columns))[:, 1][0])
        rows.append({
            "instance": i,
            "top_fraction": top_fraction,
            "comprehensiveness": abs(full - p_deleted),
            "sufficiency": abs(full - p_inserted),
            "full_probability": full,
        })
    return pd.DataFrame(rows)


def calibrated_predictions(y_val, val_proba, test_proba) -> np.ndarray:
    """Fit calibration on validation predictions only and transform test scores."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(val_proba), np.asarray(y_val))
    return calibrator.predict(np.asarray(test_proba))


def reliability_score(
    predictive_quality: float,
    robustness_retention: float,
    explanation_diagnostics: float,
    inverse_group_gap: float | None = None,
) -> dict[str, float | str]:
    """Descriptive geometric audit score; omit score when fairness is unavailable."""
    components = {
        "predictive_quality": predictive_quality,
        "robustness_retention": robustness_retention,
        "explanation_diagnostics": explanation_diagnostics,
    }
    if inverse_group_gap is None or not np.isfinite(inverse_group_gap):
        return {**components, "inverse_group_gap": float("nan"),
                "reliability_score": float("nan"),
                "score_status": "not_scored_fairness_unavailable"}
    components["inverse_group_gap"] = inverse_group_gap
    clipped = {k: float(np.clip(v, 0.0, 1.0)) for k, v in components.items()}
    score = float(np.prod(list(clipped.values())) ** (1 / len(clipped)))
    return {**clipped, "reliability_score": score, "score_status": "descriptive_only"}


def run_controlled_ablation(
    strategy_runs: Mapping[str, Mapping[str, np.ndarray]],
    y_test: pd.Series | np.ndarray,
    threshold_criterion: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> pd.DataFrame:
    """Compare imbalance strategies under fixed, OOF-tuned, and calibrated thresholds.

    ``strategy_runs[strategy]`` holds ``oof_proba`` and ``y_oof`` (out-of-fold
    predictions inside the training partition, used for every selection) and
    ``test_proba`` (the refit model's held-out scores). The test set never
    chooses a threshold or a calibration map.
    """
    y_test_arr = np.asarray(y_test).astype(int)
    rows = []
    for strategy, run in strategy_runs.items():
        oof = np.asarray(run["oof_proba"], dtype=float)
        y_oof = np.asarray(run["y_oof"]).astype(int)
        test_proba = np.asarray(run["test_proba"], dtype=float)
        tuned, _ = optimal_threshold(y_oof, oof, criterion=threshold_criterion, cost_fp=cost_fp, cost_fn=cost_fn)
        raw_calibration, _ = calibration_diagnostics(y_test_arr, test_proba)
        for label, threshold in (("fixed_0.50", 0.50), (f"oof_{threshold_criterion}", tuned)):
            metrics = compute_metrics(y_test_arr, test_proba >= threshold, test_proba, threshold)
            rows.append({
                "strategy": strategy, "threshold_rule": label, **metrics,
                **{f"probability_{key}": value for key, value in raw_calibration.items()
                   if key not in {"n", "prevalence", "brier", "base_rate_brier"}},
            })
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof, y_oof)
        calibrated_oof = calibrator.predict(oof)
        calibrated = calibrator.predict(test_proba)
        cal_threshold, _ = optimal_threshold(
            y_oof, calibrated_oof, criterion=threshold_criterion, cost_fp=cost_fp, cost_fn=cost_fn,
        )
        cal_metrics = compute_metrics(y_test_arr, calibrated >= cal_threshold, calibrated, cal_threshold)
        calibrated_summary, _ = calibration_diagnostics(y_test_arr, calibrated)
        rows.append({
            "strategy": strategy, "threshold_rule": "isotonic_oof", **cal_metrics,
            **{f"probability_{key}": value for key, value in calibrated_summary.items()
               if key not in {"n", "prevalence", "brier", "base_rate_brier"}},
        })
    return pd.DataFrame(rows)
