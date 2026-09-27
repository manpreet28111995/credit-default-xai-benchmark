"""
explainability/shap_explainer.py — SHAP-based global and local explanations.

Implements:
  • TreeExplainer    for tree-based models (XGBoost, LightGBM, CatBoost, RF)
  • KernelExplainer  fallback for black-box models
  • DeepExplainer    stub for neural models (TabNet)

Generates Figures 3–6 from the paper:
  • SHAP summary (beeswarm) plot — global feature ranking
  • SHAP bar plot — mean |SHAP| values
  • SHAP dependence plots — feature interaction
  • SHAP waterfall (force) plots — individual prediction explanation
  • SHAP decision plot — multi-sample paths
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _import_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")


# ── Explainer factory ─────────────────────────────────────────────────────────

def get_shap_explainer(model, X_background: pd.DataFrame, model_type: str = "tree"):
    """
    Return the appropriate SHAP explainer for a given model type.

    Parameters
    ──────────
    model           : fitted classifier (with predict_proba)
    X_background    : background / reference dataset (subset of train)
    model_type      : 'tree' | 'kernel' | 'deep'
    """
    shap = _import_shap()

    if model_type == "tree":
        inner = getattr(model, "_model", model)   # unwrap our wrapper
        try:
            explainer = shap.TreeExplainer(
                inner,
                data                    = shap.sample(X_background, 200),
                feature_perturbation    = "interventional",
                model_output            = "probability",
            )
            
            # SHAP defers the unsupported categorical check until shap_values() is called.
            # We execute a single dummy row calculation to proactively catch the error now.
            _ = explainer.shap_values(X_background.iloc[:1])
            
        except NotImplementedError as exc:
            # Fallback for models trained with native categorical features (LightGBM/XGBoost/CatBoost)
            if "Categorical split" in str(exc):
                log.warning(
                    "Categorical splits detected. Falling back to tree_path_dependent "
                    "perturbation. Note: SHAP values will be in margin (log-odds) space."
                )
                explainer = shap.TreeExplainer(
                    inner,
                    feature_perturbation="tree_path_dependent"
                )
            else:
                raise
        except ValueError as exc:
            # Known SHAP / XGBoost incompatibility: XGBoost >= 3.0/3.1
            # serialises base_score as a bracketed array string (e.g.
            # "[2.21E-1]") instead of a plain float. Older SHAP releases
            # call float() on it directly and crash. There's no reliable
            # way to patch this from the outside — XGBoost re-derives the
            # bracketed form internally regardless of what's written back
            # via load_config(), so the actual fix is upgrading SHAP to a
            # release that already handles it (confirmed fixed as of SHAP
            # 0.52.0). Surface a clear, actionable message instead of the
            # raw stack trace.
            if "could not convert string to float" in str(exc) and hasattr(inner, "get_booster"):
                raise RuntimeError(
                    "SHAP can't read this XGBoost model due to a known "
                    "version incompatibility: XGBoost >= 3.0 changed how "
                    "it serialises 'base_score', and your installed SHAP "
                    "version predates the upstream fix. Run:\n\n"
                    "    pip install --upgrade shap\n\n"
                    "then re-run the pipeline. "
                    f"(Original error: {exc})"
                ) from exc
            raise
        log.info("SHAP TreeExplainer initialised")

    elif model_type == "kernel":
        background = shap.kmeans(X_background.values, 50)
        explainer  = shap.KernelExplainer(
            model.predict_proba, background
        )
        log.info("SHAP KernelExplainer initialised (slow — use tree when possible)")

    elif model_type == "deep":
        # Requires pytorch model accessible via model._model
        raise NotImplementedError(
            "DeepExplainer for TabNet requires extracting the PyTorch model. "
            "Use model.explain() for TabNet-native attention masks instead."
        )
    else:
        raise ValueError(f"model_type must be 'tree', 'kernel', or 'deep'. Got '{model_type}'.")

    return explainer


# ── SHAP value computation ────────────────────────────────────────────────────

def compute_shap_values(
    explainer,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_samples: int = 2000,
) -> np.ndarray:
    """
    Compute SHAP values for class-1 (default) probability.

    Returns
    ───────
    shap_vals : (n_samples, n_features) array of SHAP values for the
                positive class.
    """
    X_sub = X.iloc[:max_samples] if len(X) > max_samples else X
    shap   = _import_shap()

    if model_type == "tree":
        sv = explainer.shap_values(X_sub)
    else:
        sv = explainer.shap_values(X_sub.values)

    shap_vals = _positive_class_shap_values(sv)
    log.info("SHAP values computed | shape=%s", shap_vals.shape)
    return shap_vals


def _positive_class_shap_values(shap_values) -> np.ndarray:
    """
    Normalise SHAP outputs to a 2-D class-1 matrix.

    SHAP versions differ by explainer/model:
      - older binary/multiclass explainers return list[class] arrays;
      - newer explainers can return (n_samples, n_features, n_classes);
      - some binary explainers return (n_samples, n_features) directly.
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1, :, :]
        else:
            raise ValueError(
                "Cannot infer positive-class SHAP slice from shape "
                f"{arr.shape}; expected class axis of length 2."
            )

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D SHAP values after class slicing, got shape {arr.shape}."
        )

    return arr


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    title: str = "SHAP Summary Plot",
    save_path: Optional[Path] = None,
    max_display: int = 20,
    dpi: int = 300,
):
    """Beeswarm summary plot — global feature importance + direction."""
    shap = _import_shap()
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X, max_display=max_display,
        show=False, plot_type="dot"
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved SHAP summary → %s", save_path)
    plt.close()


def plot_shap_bar(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    title: str = "Mean |SHAP| Feature Importance",
    save_path: Optional[Path] = None,
    max_display: int = 20,
    dpi: int = 300,
):
    """Horizontal bar chart of mean absolute SHAP values."""
    shap = _import_shap()
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_vals, X, max_display=max_display,
        plot_type="bar", show=False
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved SHAP bar → %s", save_path)
    plt.close()


def plot_shap_dependence(
    shap_vals: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """Dependence plot for a single feature, optionally coloured by interaction."""
    shap = _import_shap()
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature,
        shap_vals,
        X,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(title or f"SHAP Dependence: {feature}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved SHAP dependence → %s", save_path)
    plt.close()


def plot_shap_waterfall(
    explainer,
    X_instance: pd.DataFrame,
    title: str = "SHAP Waterfall (Local Explanation)",
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Waterfall plot explaining a single prediction.
    Uses the new shap.plots.waterfall API (SHAP >= 0.41).
    """
    shap = _import_shap()
    sv    = explainer(X_instance)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv[0, :, 1], show=False)   # class 1 slice
    plt.title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved SHAP waterfall → %s", save_path)
    plt.close()


def plot_shap_decision(
    explainer,
    X_samples: pd.DataFrame,
    title: str = "SHAP Decision Plot",
    save_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Decision plot showing cumulative SHAP value paths for multiple instances.
    Highlights defaulters vs non-defaulters.
    """
    shap = _import_shap()
    sv    = explainer.shap_values(X_samples)
    sv_c1 = sv[1] if isinstance(sv, list) else sv

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.decision_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list)
            else explainer.expected_value,
        sv_c1,
        X_samples,
        show=False,
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved SHAP decision plot → %s", save_path)
    plt.close()


# ── Utility: top-k features ───────────────────────────────────────────────────

def top_k_features(
    shap_vals: np.ndarray,
    feature_names: list[str],
    k: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame ranking features by mean absolute SHAP value.
    Used to populate Table III of the paper.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df       = pd.DataFrame({
        "feature"    : feature_names,
        "mean_|shap|": mean_abs,
    }).sort_values("mean_|shap|", ascending=False).head(k)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)[["rank", "feature", "mean_|shap|"]]


# ── SHAP consistency check ─────────────────────────────────────────────────────

def shap_consistency_check(
    shap_vals: np.ndarray,
    y_pred_proba: np.ndarray,
    expected_value: float,
    tol: float = 0.05,
    n_samples: int = 200,
) -> dict:
    """
    Verify that SHAP values reconstruct model output (additivity property).

    Σ φᵢ(x) + E[f(x)] ≈ f(x)   for each sample.

    Returns a dict with mean absolute error and pass/fail flag.
    """
    n = min(n_samples, len(shap_vals))
    pred_from_shap = shap_vals[:n].sum(axis=1) + expected_value
    pred_model     = y_pred_proba[:n]

    # --- NEW LOGIC ---
    # If the predictions from SHAP are well outside the [0, 1] range, 
    # it means they were calculated in margin (log-odds) space.
    # We apply the sigmoid function to convert log-odds back to probabilities.
    if pred_from_shap.max() > 1.0 or pred_from_shap.min() < 0.0:
        log.debug("SHAP values appear to be in log-odds space. Applying sigmoid for check.")
        pred_from_shap = 1 / (1 + np.exp(-pred_from_shap))
    # -----------------

    mae  = np.abs(pred_from_shap - pred_model).mean()
    result = {
        "mae"  : float(mae),
        "pass" : bool(mae < tol),
        "tol"  : tol,
    }
    log.info(
        "SHAP additivity check | MAE=%.5f | %s",
        mae, "PASS ✓" if result["pass"] else "FAIL ✗",
    )
    return result