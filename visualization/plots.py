"""
visualization/plots.py — Publication-quality figures for the IEEE Access paper.

Generates:
  Fig. 1  — Dataset class distribution and imbalance overview
  Fig. 2  — ROC curves (all models, all SMOTE strategies)
  Fig. 3  — Precision-Recall curves
  Fig. 4  — Confusion matrices (best model, each strategy)
  Fig. 5  — SHAP summary + bar plots  [delegated to shap_explainer.py]
  Fig. 6  — SHAP dependence plots     [delegated to shap_explainer.py]
  Fig. 7  — LIME local explanations   [delegated to lime_explainer.py]
  Fig. 8  — Feature importance comparison (SHAP vs. model built-in)
  Fig. 9  — Threshold sweep (F1, G-Mean, Precision, Recall)
  Fig. 10 — Ablation: SMOTE strategy vs. AUROC grouped bar chart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, average_precision_score,
)

log = logging.getLogger(__name__)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Serif",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "legend.fontsize"  : 9,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

from config import PALETTE


# ── Fig 1: Class distribution ─────────────────────────────────────────────────

def plot_class_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
    resampled: Optional[Dict[str, pd.Series]] = None,
    save_path: Optional[Path] = None,
):
    """
    Side-by-side bar charts of class distribution before and after each
    resampling strategy.  Matches the style of Figure 1 in the paper.
    """
    splits = {"Train (original)": y_train, "Test": y_test}
    if resampled:
        for name, y_res in resampled.items():
            splits[f"After {name}"] = y_res

    n    = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, y) in zip(axes, splits.items()):
        counts  = y.value_counts().sort_index()
        pct     = counts / counts.sum() * 100
        colors  = ["#4575b4", "#d73027"]
        bars    = ax.bar(["Non-default\n(0)", "Default\n(1)"],
                         counts.values, color=colors, edgecolor="white", width=0.5)
        for bar, p in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Class Distribution Across Dataset Splits", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 1 saved → %s", save_path)
    plt.close()


# ── Fig 2: ROC Curves ─────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    title: str = "ROC Curves — All Models",
):
    """
    Overlay ROC curves for multiple models.

    `results` format:
      { model_name: {"y_true": ..., "y_proba": ...} }
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        roc_auc     = auc(fpr, tpr)
        color       = PALETTE.get(model_name, None)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{model_name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 2 saved → %s", save_path)
    plt.close()


# ── Fig 3: Precision-Recall Curves ───────────────────────────────────────────

def plot_pr_curves(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curves — All Models",
):
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, data in results.items():
        prec, rec, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        ap           = average_precision_score(data["y_true"], data["y_proba"])
        color        = PALETTE.get(model_name, None)
        ax.plot(rec, prec, linewidth=2, color=color,
                label=f"{model_name}  (AP = {ap:.4f})")

    # Baseline = prevalence
    prevalence = results[list(results.keys())[0]]["y_true"].mean()
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (prevalence = {prevalence:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 3 saved → %s", save_path)
    plt.close()


# ── Fig 4: Confusion Matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_path: Optional[Path] = None,
    normalize: bool = True,
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Non-default (0)", "Default (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix")

    fmt = ".2f" if normalize else "d"
    thresh = cm_norm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            raw = cm[i, j]
            ax.text(j, i,
                    f"{val:{fmt}}\n(n={raw:,})",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 4 saved → %s", save_path)
    plt.close()


# ── Fig 8: Feature importance comparison ─────────────────────────────────────

def plot_feature_importance_comparison(
    shap_importance: pd.Series,
    model_importance: pd.Series,
    top_k: int = 15,
    save_path: Optional[Path] = None,
):
    """
    Dual horizontal bar chart: SHAP mean |φ| vs model built-in gain importance.
    Highlights agreement / disagreement in top-k rankings.
    """
    shap_top  = shap_importance.nlargest(top_k).sort_values()
    model_top = model_importance.nlargest(top_k).sort_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Normalise to [0, 1] for side-by-side comparability
    def _norm(s): return s / s.max()

    ax1.barh(range(len(shap_top)), _norm(shap_top), color="#e41a1c", height=0.6)
    ax1.set_yticks(range(len(shap_top)))
    ax1.set_yticklabels(shap_top.index, fontsize=9)
    ax1.set_xlabel("Normalised Mean |SHAP|")
    ax1.set_title("SHAP Feature Importance")

    ax2.barh(range(len(model_top)), _norm(model_top), color="#377eb8", height=0.6)
    ax2.set_yticks(range(len(model_top)))
    ax2.set_yticklabels(model_top.index, fontsize=9)
    ax2.set_xlabel("Normalised Model Gain")
    ax2.set_title("Model Built-in Importance")

    fig.suptitle(f"Feature Importance: SHAP vs. Model Gain (Top {top_k})",
                 fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 8 saved → %s", save_path)
    plt.close()


# ── Fig 9: Threshold sweep ────────────────────────────────────────────────────

def plot_threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "",
    save_path: Optional[Path] = None,
):
    """
    Plot F1, G-Mean, Precision, and Recall as a function of decision threshold.
    Marks the threshold that maximises F1.
    """
    from evaluation.metrics import compute_metrics

    thresholds = np.arange(0.05, 0.95, 0.01)
    records    = []
    for t in thresholds:
        m = compute_metrics(y_true, (y_proba >= t).astype(int), y_proba, t)
        records.append({"threshold": t, "F1": m["F1"],
                         "G-Mean": m["G-Mean"],
                         "Precision": m["Precision"],
                         "Recall": m["Recall"]})

    df       = pd.DataFrame(records)
    best_idx = df["F1"].idxmax()
    best_t   = df.loc[best_idx, "threshold"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in zip(["F1", "G-Mean", "Precision", "Recall"],
                           ["#e41a1c", "#4daf4a", "#377eb8", "#ff7f00"]):
        ax.plot(df["threshold"], df[col], label=col, color=color, linewidth=2)

    ax.axvline(best_t, color="gray", linestyle="--", linewidth=1.5,
               label=f"Optimal threshold = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Threshold Sweep — {model_name}")
    ax.legend()
    ax.set_xlim([0.05, 0.95]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 9 saved → %s", save_path)
    plt.close()
    return float(best_t)


# ── Fig 10: SMOTE ablation ────────────────────────────────────────────────────

def plot_smote_ablation(
    ablation_df: pd.DataFrame,
    metric: str = "AUROC",
    save_path: Optional[Path] = None,
):
    """
    Grouped bar chart: one group per model, bars = SMOTE strategies.

    ablation_df columns: Model, Strategy, <metric>
    """
    strategies = ablation_df["Strategy"].unique()
    models     = ablation_df["Model"].unique()
    n_strategies = len(strategies)
    x            = np.arange(len(models))
    width        = 0.8 / n_strategies

    colors = sns.color_palette("Set2", n_strategies)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))

    for i, (strat, color) in enumerate(zip(strategies, colors)):
        sub    = ablation_df[ablation_df["Strategy"] == strat]
        vals   = [sub[sub["Model"] == m][metric].values[0]
                  if len(sub[sub["Model"] == m]) > 0 else 0
                  for m in models]
        offset = (i - n_strategies / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=strat,
                        color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Ablation Study: SMOTE Strategy vs. {metric}")
    ax.legend(title="Resampling Strategy", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)
    ax.set_ylim([ablation_df[metric].min() - 0.02, 1.0])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        log.info("Fig 10 saved → %s", save_path)
    plt.close()
