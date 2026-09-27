"""
explainability/lime_explainer.py — LIME local explanations for credit default.

LIME (Local Interpretable Model-agnostic Explanations, Ribeiro et al. 2016)
fits a local linear surrogate model around each query instance to provide
interpretable explanations for individual predictions.

This module provides:
  • A wrapper around lime.lime_tabular.LimeTabularExplainer
  • Per-instance explanation extraction
  • Agreement / disagreement analysis between SHAP and LIME rankings
  • Figure generation matching the paper's layout (Fig. 7)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)


def _import_lime():
    try:
        from lime.lime_tabular import LimeTabularExplainer
        return LimeTabularExplainer
    except ImportError:
        raise ImportError("Install lime: pip install lime")


# ── LimeWrapper ───────────────────────────────────────────────────────────────

class CreditLimeExplainer:
    """
    Convenience wrapper around LimeTabularExplainer tailored for the
    credit default experiment.

    Parameters
    ──────────
    X_train        : Training set used to estimate feature statistics.
    feature_names  : Column names of X_train.
    predict_fn     : Model's predict_proba (or equivalent) callable.
    class_names    : Labels for each class output.
    num_features   : Max features per local explanation.
    num_samples    : Perturbation samples for surrogate fitting.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        feature_names: list[str],
        predict_fn: Callable,
        class_names: Optional[list[str]] = None,
        num_features: int = 15,
        num_samples: int = 1000,
        random_state: int = 42,
    ):
        LimeTabularExplainer = _import_lime()

        self.predict_fn    = predict_fn
        self.num_features  = num_features
        self.num_samples   = num_samples
        self.feature_names = feature_names

        self._explainer = LimeTabularExplainer(
            training_data   = X_train.values.astype("float32"),
            feature_names   = feature_names,
            class_names     = class_names or ["Non-default", "Default"],
            discretize_continuous = True,
            random_state    = random_state,
            verbose         = False,
        )
        log.info("LIME tabular explainer initialised | features=%d", len(feature_names))

    def explain_instance(
        self,
        x: np.ndarray | pd.Series,
        label: int = 1,
    ):
        """
        Return a LIME Explanation object for a single instance.

        Parameters
        ──────────
        x      : 1-D feature vector (or pd.Series).
        label  : Class to explain (1 = default).
        """
        if isinstance(x, pd.Series):
            x = x.values.astype("float32")
        else:
            x = np.asarray(x, dtype="float32")

        exp = self._explainer.explain_instance(
            data_row           = x,
            predict_fn         = self.predict_fn,
            num_features       = self.num_features,
            num_samples        = self.num_samples,
            labels             = (label,),
        )
        return exp

    def explanation_to_series(
        self,
        exp,
        label: int = 1,
    ) -> pd.Series:
        """
        Convert LIME explanation to a signed importance Series.

        Returns
        ───────
        pd.Series with feature_condition → signed weight, sorted by |weight|.
        """
        raw    = dict(exp.as_list(label=label))
        series = pd.Series(raw, name="lime_weight")
        return series.reindex(series.abs().sort_values(ascending=False).index)

    def batch_explain(
        self,
        X: pd.DataFrame,
        label: int = 1,
        n: Optional[int] = None,
        return_scores: bool = False,
    ) -> list[pd.Series] | tuple[list[pd.Series], list[float]]:
        """Explain multiple instances, optionally returning local surrogate R²."""
        n   = n or len(X)
        out = []
        scores = []
        for i in range(min(n, len(X))):
            exp = self.explain_instance(X.iloc[i].values, label=label)
            out.append(self.explanation_to_series(exp, label=label))
            scores.append(float(getattr(exp, "score", np.nan)))
        log.info("LIME batch complete | n=%d", len(out))
        return (out, scores) if return_scores else out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_lime_explanation(
    exp,
    label: int = 1,
    title: str = "LIME Local Explanation",
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Horizontal bar chart of LIME local feature weights.
    Green = pushes towards non-default; Red = pushes towards default.
    """
    feats_weights = exp.as_list(label=label)
    conditions    = [fw[0] for fw in feats_weights]
    weights       = [fw[1] for fw in feats_weights]
    colors        = ["#d73027" if w > 0 else "#4575b4" for w in weights]

    fig, ax = plt.subplots(figsize=(9, max(4, len(conditions) * 0.4 + 1)))
    y_pos   = np.arange(len(conditions))
    ax.barh(y_pos, weights, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight (signed contribution to P(Default))", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)

    # Add a legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#d73027", label="Increases default probability"),
        Patch(facecolor="#4575b4", label="Decreases default probability"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved LIME explanation → %s", save_path)
    plt.close()
    return fig


def plot_lime_multi_instance(
    exps: list,
    titles: list[str],
    label: int = 1,
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    2×2 grid of LIME explanations for TP, FP, FN, TN instances.
    Used as Fig. 7 in the paper.
    """
    n    = min(4, len(exps))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (exp, title) in enumerate(zip(exps[:n], titles[:n])):
        ax = axes[i]
        feats_weights = exp.as_list(label=label)
        if not feats_weights:
            ax.set_visible(False)
            continue
        conditions = [fw[0] for fw in feats_weights][:10]
        weights    = [fw[1] for fw in feats_weights][:10]
        colors     = ["#d73027" if w > 0 else "#4575b4" for w in weights]

        y_pos = np.arange(len(conditions))
        ax.barh(y_pos, weights, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conditions, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("LIME Weight", fontsize=9)

    for j in range(n, 4):
        axes[j].set_visible(False)

    plt.suptitle("LIME Local Explanations: TP / FP / FN / TN Instances",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved LIME multi-instance grid → %s", save_path)
    plt.close()


# ── SHAP–LIME agreement analysis ─────────────────────────────────────────────

def compute_shap_lime_agreement(
    shap_vals: np.ndarray,
    lime_explanations: list[pd.Series],
    feature_names: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Compute Spearman rank correlation between SHAP and LIME feature rankings
    for each instance.

    Returns
    ───────
    DataFrame with columns: instance, spearman_rho, p_value
    Used for Table IV of the paper.
    """
    records = []
    for i, lime_ser in enumerate(lime_explanations):
        if i >= len(shap_vals):
            break

        # SHAP ranking for this instance
        shap_row  = np.abs(shap_vals[i])
        shap_rank = pd.Series(shap_row, index=feature_names).rank(ascending=False)

        # LIME ranking — map condition strings back to feature names
        lime_abs   = lime_ser.abs()
        # Extract feature name from LIME condition string (e.g. "PAY_0 <= 0.00")
        lime_clean = {}
        for cond, val in lime_abs.items():
            for fn in feature_names:
                if fn in cond:
                    lime_clean[fn] = lime_clean.get(fn, 0) + val
                    break

        if not lime_clean:
            continue

        lime_rank = pd.Series(lime_clean).rank(ascending=False)

        # Align on shared features
        shared    = shap_rank.index.intersection(lime_rank.index)
        if len(shared) < 3:
            continue

        rho, pval = spearmanr(shap_rank[shared], lime_rank[shared])
        records.append({"instance": i, "spearman_rho": round(rho, 4), "p_value": round(pval, 4)})

    df = pd.DataFrame(records)
    if not df.empty:
        log.info(
            "SHAP–LIME agreement | mean ρ=%.3f ± %.3f",
            df["spearman_rho"].mean(), df["spearman_rho"].std(),
        )
    return df
