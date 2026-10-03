"""
pipeline.py — Training-only CV selection, OOF postprocessing, held-out evaluation.

Stages
  1. Data loading, labelling, and partitioning (train / test only)
  2. Imbalance-strategy ablation, selected on out-of-fold (OOF) AUROC
  3. Model training: RandomizedSearchCV over an imblearn Pipeline
     (preprocessor -> sampler -> model) so imputation, winsorisation,
     scaling and resampling are all re-fitted inside every fold; OOF
     predictions from the tuned pipeline; refit on the full train partition
  4. Evaluation: thresholds, calibration maps, champion, and fairness
     thresholds are all chosen on OOF predictions; the test partition is
     used for reporting, not selection
  5. SHAP global and local explanations (random test subsample)
  6. LIME local explanations and SHAP–LIME agreement
  7. Revision analyses (robustness, XAI quality, controlled ablations,
     fairness–utility, descriptive reliability diagnostics)
  8. Multi-run summary with descriptive model ranks across dependent runs

Replication unit
  * random split   : one run per seed (split + model RNG)
  * temporal split : rolling-origin blocks; one run per (cutoff, seed).
                     Training rows have labels observable at the cutoff;
                     test rows are originated in the following window.

Run with:
  python pipeline.py --dataset uci --seeds 42 99 ...
  python pipeline.py --dataset prosper --split-mode temporal \
                     --cutoffs 2010-01 2010-07 2011-01 2011-07 2012-01 2012-07 --seeds 42
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import platform
import pickle
import signal
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from config import (
    DATASET, TEST_SIZE, DEFAULT_SEEDS,
    SMOTE_STRATEGIES, FIG_DIR, RES_DIR, MDL_DIR, DATA_DIR, OUTPUTS_DIR, ORIGINAL_OUTPUTS_DIR, RUNS_DIR,
    RANDOM_SEARCH_SPACES, RANDOM_SEARCH_N_ITER, RANDOM_SEARCH_SCORING,
    INNER_CV_FOLDS, INNER_TEMPORAL_FOLDS, THRESHOLD_CRITERION,
    SHAP_BACKGROUND_SAMPLES, LIME_NUM_FEATURES, LIME_NUM_SAMPLES, FIGURE_EXT,
    PROSPER_DEFAULT_CUTOFFS, PROSPER_HORIZON_MONTHS, PROSPER_TEST_WINDOW_MONTHS,
    PROSPER_FILE,
)
from data.data_loader import load_dataset, LoadedData
from preprocessing.imbalance_handler import (
    NominalAwareOverSampler, apply_resampling, imbalance_statistics, SAMPLER_REGISTRY,
)
from preprocessing.tabular_preprocessor import TabularPreprocessor
from models.gradient_boosting import build_model, MODEL_REGISTRY
from evaluation.metrics import (
    compute_metrics, optimal_threshold, build_results_table,
    mcnemar_test, delong_test, oof_predict_proba,
)
from evaluation.revision_analysis import (
    evaluate_robustness, explanation_metrics, fairness_utility_curve,
    reliability_score, run_controlled_ablation, explanation_fidelity,
    reliability_constrained_selection, bootstrap_metric_intervals,
    paired_bootstrap_differences, holm_adjust, calibration_diagnostics,
    fit_equal_opportunity_threshold_policy, evaluate_group_threshold_policy,
    bootstrap_fairness_policy_differences,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from evaluation.temporal_cv import ObservableTimeSeriesSplit
from rerun_support import PROTOCOL, training_key, load_checkpoint, save_checkpoint, descriptive_ranks
from visualization.plots import (
    plot_class_distribution, plot_roc_curves, plot_pr_curves,
    plot_confusion_matrix, plot_feature_importance_comparison,
    plot_threshold_sweep, plot_smote_ablation,
)
from explainability.shap_explainer import (
    get_shap_explainer, compute_shap_values,
    plot_shap_summary, plot_shap_bar, plot_shap_dependence,
    top_k_features, shap_consistency_check,
)
from explainability.lime_explainer import (
    CreditLimeExplainer, plot_lime_explanation,
    plot_lime_multi_instance, compute_shap_lime_agreement,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("pipeline")

ALL_MODELS = ["XGBoost", "LightGBM", "CatBoost", "Random Forest", "Logistic Reg."]
FAST_MODELS = ["XGBoost", "LightGBM", "Logistic Reg."]
TREE_MODELS = {"XGBoost", "LightGBM", "CatBoost", "Random Forest"}


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def _time_limit(seconds: int, label: str = ""):
    """Best-effort SIGALRM wall-clock ceiling (no-op where SIGALRM is missing)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s" + (f" ({label})" if label else ""))

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _model_seed_kwargs(name: str, seed: int) -> dict:
    if name in {"XGBoost", "LightGBM", "Random Forest", "Logistic Reg."}:
        return {"random_state": seed}
    if name == "CatBoost":
        return {"random_seed": seed}
    if name == "TabNet":
        return {"seed": seed}
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if "__pycache__" in path.parts or ".venv" in path.parts:
            continue
        digest.update(str(path.relative_to(Path(__file__).parent)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_checksums(dataset_names: list[str]) -> dict[str, str | None]:
    candidates = {
        "uci": [DATA_DIR / "uci_credit.csv"],
        "lending_club": [DATA_DIR / "lending_club_subsample_50k_by_issue_date.csv", DATA_DIR / "loan.csv"],
        "south_german": [DATA_DIR / "south_german_credit.csv", DATA_DIR / "german_credit.csv",
                         DATA_DIR / "SouthGermanCredit.asc"],
        "german": [DATA_DIR / "SouthGermanCredit.asc"],
        "prosper": [DATA_DIR / PROSPER_FILE],
    }
    result = {}
    for name in dict.fromkeys(dataset_names):
        path = next((c for c in candidates.get(name, []) if c.exists()), None)
        result[name] = _sha256(path) if path else None
    return result


def _package_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ["numpy", "pandas", "sklearn", "imblearn", "xgboost", "lightgbm",
                   "catboost", "shap", "lime", "torch", "pytorch_tabnet", "scipy"]:
        try:
            versions[module] = getattr(importlib.import_module(module), "__version__", "unknown")
        except Exception:
            versions[module] = None
    return versions


def _build_any_model(name: str, seed: int, data: dict | None = None,
                     cost_sensitive: bool = False, **params):
    """Fresh untrained model with dataset-aware nominal handling."""
    kwargs = {**_model_seed_kwargs(name, seed), "cost_sensitive": cost_sensitive, **params}
    if data is not None:
        kwargs["nominal_indices"] = list(data["nominal_idx"])
        if name == "TabNet":
            kwargs["cat_idxs"] = list(data["nominal_idx"])
            kwargs["cat_dims"] = list(data["cat_dims"])
    if name == "TabNet":
        from models.tabnet_model import TabNetWrapper
        kwargs.pop("nominal_indices", None)
        return TabNetWrapper(**kwargs)
    return build_model(name, **kwargs)


def _smote_kwargs(name: str, seed: int) -> dict:
    kwargs = dict(SMOTE_STRATEGIES.get(name) or {})
    if "random_state" in kwargs:
        kwargs["random_state"] = seed
    return kwargs


def _make_pipeline(strategy: str, model_name: str, seed: int, data: dict, **model_params):
    """imblearn Pipeline(prep -> sampler -> model); every step re-fits inside each CV fold."""
    prep = TabularPreprocessor(**data["prep_spec"])
    sampler = NominalAwareOverSampler(
        strategy=strategy, nominal_indices=list(data["nominal_idx"]),
        sampler_kwargs=_smote_kwargs(strategy, seed),
    )
    model = _build_any_model(
        model_name, seed, data, cost_sensitive=(strategy == "ClassWeight"), **model_params,
    )
    return ImbPipeline([("prep", prep), ("sampler", sampler), ("model", model)])


def _data_dict(loaded: LoadedData, split_mode: str) -> dict:
    """Pipeline-facing view of a LoadedData record (positions refer to preprocessor output)."""
    n_lead = len(loaded.prep_spec["continuous_columns"]) + len(loaded.prep_spec["indicator_columns"])
    return dict(
        X_train=loaded.X_train, X_test=loaded.X_test, y_train=loaded.y_train, y_test=loaded.y_test,
        feature_names=loaded.feature_names, prep_spec=loaded.prep_spec,
        nominal_columns=loaded.nominal_columns,
        nominal_idx=[n_lead + i for i in range(len(loaded.nominal_columns))],
        cat_dims=[len(loaded.meta["nominal_categories"][c]) + 1 for c in loaded.nominal_columns],
        nominal_categories=loaded.meta["nominal_categories"], split_mode=split_mode,
        train_dates=loaded.train_dates,
    )


def _inner_cv(data: dict, args):
    if data["split_mode"] == "temporal":
        return ObservableTimeSeriesSplit(data["train_dates"], n_splits=args.inner_temporal_folds)
    return StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)


def _oof(pipeline, data: dict, args) -> Tuple[np.ndarray, np.ndarray]:
    # Probe, family selection and ablations often request the identical fit.
    # Cache only unfitted-estimator recipes with the complete training key.
    if "cache_key" not in data:
        return oof_predict_proba(pipeline, data["X_train"], data["y_train"], _inner_cv(data, args))
    recipe = hashlib.sha256(pickle.dumps(pipeline, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    key = data["cache_key"] + recipe
    path = MDL_DIR / "oof" / f"{recipe}.pkl"
    cached = load_checkpoint(path, key)
    if cached is not None:
        log.info("Reusing identical OOF pipeline fit.")
        return cached
    result = oof_predict_proba(pipeline, data["X_train"], data["y_train"], _inner_cv(data, args))
    save_checkpoint(path, key, result)
    return result


def _threshold_from_oof(y_oof, oof_proba, args) -> Tuple[float, float]:
    return optimal_threshold(
        y_oof, oof_proba, criterion=args.threshold_criterion,
        cost_fp=args.cost_fp, cost_fn=args.cost_fn,
    )


def _select_representative_instances(X_test, y_test, y_pred, y_proba) -> Dict[str, int]:
    y_true = y_test.values
    instances = {}
    for kind, cond in {
        "TP": (y_true == 1) & (y_pred == 1),
        "FP": (y_true == 0) & (y_pred == 1),
        "FN": (y_true == 1) & (y_pred == 0),
        "TN": (y_true == 0) & (y_pred == 0),
    }.items():
        idxs = np.where(cond)[0]
        if len(idxs) > 0:
            instances[kind] = int(idxs[np.argmin(np.abs(y_proba[idxs] - 0.5))])
        else:
            log.warning("No %s instances found in test set.", kind)
    return instances


def _save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    table_df = df.reset_index() if df.index.name else df
    n_cols = len(table_df.columns)
    latex = table_df.to_latex(
        index=False, float_format="%.4f", caption=caption, label=label,
        column_format="l" + "c" * (n_cols - 1),
    )
    latex = latex.replace("\\begin{table}", "\\begin{table}\n\\centering", 1)
    path.write_text(latex)
    log.info("LaTeX table → %s", path)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray, pd.Series)):
        return [_to_builtin(v) for v in obj.tolist()]
    try:
        if not isinstance(obj, (str, bool, int, float)) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _yaml_scalar(value) -> str:
    value = _to_builtin(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _write_yaml(data: dict, path: Path) -> None:
    def emit(obj, indent=0):
        pad = " " * indent
        lines = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        elif isinstance(obj, list):
            for value in obj:
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(emit(value, indent + 2))
                else:
                    lines.append(f"{pad}- {_yaml_scalar(value)}")
        return lines
    path.write_text("\n".join(emit(_to_builtin(data))) + "\n")


def _write_json(data, path: Path) -> None:
    path.write_text(json.dumps(_to_builtin(data), indent=2))


# ── Stage 1: Data ─────────────────────────────────────────────────────────────

def stage_data(args, cutoff: str | None) -> dict:
    log.info("=" * 60)
    log.info("STAGE 1: Data Loading, Labelling & Partitioning")
    log.info("=" * 60)

    loaded: LoadedData = load_dataset(
        dataset=args.dataset, data_dir=DATA_DIR, test_size=TEST_SIZE,
        random_state=args.seed, split_mode=args.split_mode, cutoff=cutoff,
        test_window_months=args.test_window_months, horizon_months=args.horizon_months,
        prosper_include_pricing=args.prosper_include_pricing,
    )
    stats = imbalance_statistics(loaded.y_train)
    log.info("Imbalance stats (train): %s", json.dumps(stats))

    _write_json({
        "split_mode": args.split_mode, "cutoff": cutoff, "meta": loaded.meta,
        "train": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in loaded.X_train.index],
        "test": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in loaded.X_test.index],
    }, RES_DIR / "split_indices.json")

    plot_class_distribution(
        loaded.y_train, loaded.y_test, save_path=FIG_DIR / f"fig01_class_distribution.{FIGURE_EXT}",
    )
    return dict(
        _data_dict(loaded, args.split_mode),
        groups_train=loaded.groups_train.reset_index(drop=True),
        groups_test=loaded.groups_test.reset_index(drop=True),
        cutoff=cutoff, meta=loaded.meta, imbalance_stats=stats,
    )


# ── Stage 2: Imbalance-strategy ablation (OOF-selected) ───────────────────────

def stage_imbalance_ablation(data: dict, args) -> dict:
    log.info("=" * 60)

    checkpoint = MDL_DIR / "imbalance_selection.pkl"
    if args.resume or args.evaluate_only:
        cached = load_checkpoint(checkpoint, data["cache_key"])
        if cached is not None:
            log.info("Reusing imbalance-selection checkpoint.")
            return cached
    if args.evaluate_only:
        raise RuntimeError("No matching imbalance checkpoint. Run training with --resume first.")
    log.info("STAGE 2: Imbalance Strategy Ablation (selected on OOF AUROC)")
    log.info("=" * 60)

    probe_candidates = ["LightGBM", "XGBoost", "CatBoost", "Random Forest"]
    strategies = list(SMOTE_STRATEGIES)
    if args.fast:
        strategies = ["None", "ClassWeight", "SMOTE"]

    records, oof_auroc, probe_used = [], {}, None
    for strategy in strategies:
        order = ([probe_used] if probe_used else []) + [c for c in probe_candidates if c != probe_used]
        for probe in order:
            try:
                pipe = _make_pipeline(strategy, probe, args.seed, data)
                oof, mask = _oof(pipe, data, args)
                y_oof = data["y_train"].to_numpy()[mask]
                oof_m = compute_metrics(y_oof, oof[mask] >= 0.5, oof[mask], 0.5)
                pipe.fit(data["X_train"], data["y_train"])
                test_proba = pipe.predict_proba(data["X_test"])[:, 1]
                test_m = compute_metrics(data["y_test"].to_numpy(), test_proba >= 0.5, test_proba, 0.5)
                probe_used = probe
                break
            except Exception as exc:
                log.warning("Probe %s failed for strategy %s (%s: %s)", probe, strategy, type(exc).__name__, exc)
                oof_m = None
        if oof_m is None:
            log.warning("Skipping strategy '%s' — no probe model could be trained.", strategy)
            continue
        oof_auroc[strategy] = oof_m["AUROC"]
        records.append({
            "Model": probe_used, "Strategy": strategy,
            "OOF_AUROC": oof_m["AUROC"], "OOF_AUPRC": oof_m["AUPRC"], "OOF_Brier": oof_m["Brier"],
            "OOF_rows": int(mask.sum()), **test_m,
        })
        log.info("Ablation | %-16s | OOF AUROC=%.4f  test AUROC=%.4f", strategy, oof_m["AUROC"], test_m["AUROC"])

    if not records:
        raise RuntimeError("Imbalance ablation failed for every strategy.")

    ablation_df = pd.DataFrame(records)
    ablation_df.to_csv(RES_DIR / "smote_ablation.csv", index=False)
    best_strategy = max(oof_auroc, key=oof_auroc.get)
    summary = {
        "selection_metric": "OOF_AUROC_inner_cv",
        "inner_cv": "ObservableTimeSeriesSplit" if data["split_mode"] == "temporal" else "StratifiedKFold",
        "model_used": probe_used, "best_strategy": best_strategy,
        "best_oof_AUROC": oof_auroc[best_strategy], "none_oof_AUROC": oof_auroc.get("None"),
        "classweight_oof_AUROC": oof_auroc.get("ClassWeight"),
        "resampling_helped": bool(best_strategy not in {"None", "ClassWeight"}),
        "strategies_evaluated": list(oof_auroc),
    }
    log.info("Best imbalance strategy (OOF AUROC): %s → %.4f [probe: %s]",
             best_strategy, oof_auroc[best_strategy], probe_used)
    _write_json(summary, RES_DIR / "smote_ablation_summary.json")
    result = dict(ablation_df=ablation_df, best_strategy=best_strategy, smote_summary=summary)
    save_checkpoint(checkpoint, data["cache_key"], result)
    return result


# ── Stage 3: Training-only tuning ──────────────────────────────────────

def _legacy_record(path: Path, name: str) -> dict:
    """Read only the scalar model record emitted by our old YAML writer."""
    json_path = path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text())["models"].get(name, {})
    if not path.exists():
        return {}
    record, in_model, in_params = {}, False, False
    for line in path.read_text().splitlines():
        if line.startswith("  ") and not line.startswith("    "):
            if in_model:
                break
            in_model = line == f"  {name}:"
        elif in_model and line.startswith("    "):
            indent = len(line) - len(line.lstrip())
            key, _, raw = line.strip().partition(":")
            if indent == 4:
                in_params = key == "best_params"
                if in_params:
                    record["best_params"] = {}
                elif raw.strip():
                    record[key] = json.loads(raw.strip())
            elif indent == 6 and in_params and raw.strip():
                record["best_params"][key] = json.loads(raw.strip())
    return record


def _reuse_legacy_model(name: str, strategy: str, params: dict, data: dict, args):
    """Reuse a final classifier only after matching selection and original data.

    New inner selection and OOF fits still run. Reconstructing the full-training
    preprocessor is cheap; legacy files did not save that outer transformer.
    """
    if not args.reuse_models_from:
        return None
    run_name = "_".join([args.dataset, *sorted(set(args.external_datasets))])
    if args.split_mode == "temporal":
        run_name += "_temporal"
    run_id = f"seed_{args.seed}" if data["cutoff"] is None else f"cutoff_{data['cutoff']}_seed_{args.seed}"
    source = Path(args.reuse_models_from) / run_name / "runs" / run_id
    results = source / "results"
    if not (results / "run_manifest.json").exists():
        return None
    try:
        manifest = json.loads((results / "run_manifest.json").read_text())
        record = _legacy_record(results / "best_params.yaml", name)
        if record.get("imbalance_strategy") != strategy or record.get("best_params", {}) != params:
            return None
        if manifest.get("dataset") != args.dataset or manifest.get("seed") != args.seed or manifest.get("cutoff") != data["cutoff"]:
            return None
        if manifest.get("dataset_checksums", {}).get(args.dataset) != data["dataset_checksums"].get(args.dataset):
            return None
        indices = json.loads((results / "split_indices.json").read_text())
        if indices["train"] != data["X_train"].index.tolist() or indices["test"] != data["X_test"].index.tolist():
            return None
        if manifest["data_meta"]["nominal_categories"] != data["meta"]["nominal_categories"]:
            return None
        pipeline = _make_pipeline(strategy, name, args.seed, data, **params)
        pipeline.named_steps["prep"].fit(data["X_train"])
        model = pipeline.named_steps["model"]
        path = source / "models" / f"{name.replace(' ', '_').lower()}.pkl"
        if name == "CatBoost":
            from catboost import CatBoostClassifier
            model._model = CatBoostClassifier()
            model._model.load_model(str(path))
        else:
            model.load(Path(str(path) + ".zip") if name == "TabNet" else path)
        p = pipeline.predict_proba(data["X_test"])[:, 1]
        metrics = compute_metrics(data["y_test"].to_numpy(), p >= .5, p, .5)
        old = pd.read_csv(results / "test_metrics.csv", index_col=0).loc[name]
        if not all(np.isclose(metrics[m], float(old[m]), atol=5.1e-5, rtol=0) for m in ("AUROC", "AUPRC", "Brier")):
            log.warning("Legacy metric verification failed for %s; refitting.", name)
            return None
        log.info("Reusing verified legacy final model: %s", name)
        return pipeline
    except Exception as exc:
        log.warning("Legacy model %s could not be reused (%s); refitting.", name, exc)
        return None


def _tune_and_fit(name: str, strategy: str, data: dict, args) -> tuple[object, dict, np.ndarray, np.ndarray]:
    """RandomizedSearchCV over Pipeline(sampler, model), OOF predictions, final refit."""
    space = RANDOM_SEARCH_SPACES.get(name)
    n_iter = min(args.tune_iter, 4) if args.fast else args.tune_iter
    record = {
        "enabled": bool(args.tune and space), "model": name, "seed": args.seed,
        "imbalance_strategy": strategy, "search": "RandomizedSearchCV(imblearn Pipeline)",
        "scoring": RANDOM_SEARCH_SCORING, "n_iter_requested": n_iter,
        "inner_cv": type(_inner_cv(data, args)).__name__, "best_score": None,
        "best_params": {}, "status": "not_run",
    }
    best_params: dict = {}
    if args.tune and space:
        log.info("RandomizedSearchCV | model=%s strategy=%s n_iter=%d", name, strategy, n_iter)
        search = RandomizedSearchCV(
            estimator=_make_pipeline(strategy, name, args.seed, data),
            param_distributions={f"model__{k}": v for k, v in space.items()},
            n_iter=n_iter, scoring=RANDOM_SEARCH_SCORING, cv=_inner_cv(data, args),
            random_state=args.seed, n_jobs=1, refit=False, error_score=np.nan,
            return_train_score=True,
        )
        try:
            search.fit(data["X_train"], data["y_train"])
            best_params = {k.replace("model__", "", 1): v for k, v in search.best_params_.items()}
            pd.DataFrame(search.cv_results_).to_csv(
                RES_DIR / f"random_search_cv_results_{name.replace(' ', '_').lower()}.csv", index=False,
            )
            record.update({"status": "success", "best_score": float(search.best_score_),
                           "best_params": best_params,
                           "n_iter_actual": int(len(search.cv_results_["params"]))})
        except Exception as exc:
            log.warning("RandomizedSearchCV failed for %s (%s: %s). Using fixed params.",
                        name, type(exc).__name__, exc)
            record.update({"status": "failed_fallback_fixed", "error": f"{type(exc).__name__}: {exc}"})
    else:
        record["status"] = "disabled" if space else "skipped_no_space"

    pipeline = _make_pipeline(strategy, name, args.seed, data, **best_params)
    oof, mask = _oof(pipeline, data, args)
    reused = _reuse_legacy_model(name, strategy, best_params, data, args)
    if reused is None:
        pipeline.fit(data["X_train"], data["y_train"])
        record["final_fit"] = "new"
    else:
        pipeline = reused
        record["final_fit"] = "legacy_model_reused_after_split_parameter_and_metric_checks"
    record["oof_rows"] = int(mask.sum())
    return pipeline, record, oof, mask


def stage_training(data: dict, imb: dict, args) -> dict:
    log.info("=" * 60)
    log.info("STAGE 3: Model Training (training CV selection, OOF predictions, refit)")
    log.info("=" * 60)

    model_names = list(args.models or (FAST_MODELS if args.fast else ALL_MODELS))
    if not args.models and not args.fast and not args.skip_tabnet:
        model_names = model_names + ["TabNet"]
    strategies = imb["ablation_df"]["Strategy"].tolist()

    trained, records, oof_proba, oof_mask, strategy_choice = {}, {}, {}, {}, {}
    for name in model_names:
        checkpoint = MDL_DIR / f"{name.replace(' ', '_').lower()}_training.pkl"
        cached = load_checkpoint(checkpoint, data["cache_key"]) if args.resume or args.evaluate_only else None
        if cached is not None:
            trained[name], records[name] = cached["pipeline"], cached["record"]
            oof_proba[name], oof_mask[name] = cached["oof"], cached["mask"]
            strategy_choice[name] = cached["strategy_choice"]
            log.info("Reusing complete pipeline and OOF checkpoint: %s", name)
            continue
        if args.evaluate_only:
            raise RuntimeError(f"No matching training checkpoint for {name}; evaluation-only never fits models.")
        log.info("Training %s …", name)
        try:
            strategy = imb["best_strategy"]
            per_model = {}
            if args.strategy_selection == "per_model" and name != "TabNet":
                # TabNet inherits the stage-2 probe choice: K x |strategies| extra
                # TabNet fits per run would dominate the compute budget.
                # Strategy chosen for this model family on OOF AUROC with default
                # hyper-parameters, removing the LightGBM-probe confound.
                for candidate in strategies:
                    try:
                        oof_c, mask_c = _oof(_make_pipeline(candidate, name, args.seed, data), data, args)
                        y_c = data["y_train"].to_numpy()[mask_c]
                        per_model[candidate] = float(compute_metrics(y_c, oof_c[mask_c] >= 0.5, oof_c[mask_c], 0.5)["AUROC"])
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Per-model strategy %s failed for %s: %s", candidate, name, exc)
                if per_model:
                    strategy = max(per_model, key=per_model.get)
                log.info("%s | per-model strategy=%s | OOF AUROC by strategy=%s", name, strategy,
                         {k: round(v, 4) for k, v in per_model.items()})
            strategy_choice[name] = {"strategy": strategy, "selection": args.strategy_selection,
                                     "oof_auroc_by_strategy": per_model}
            pipeline, record, oof, mask = _tune_and_fit(name, strategy, data, args)
            record["strategy_selection"] = strategy_choice[name]
            trained[name] = pipeline
            records[name] = record
            oof_proba[name] = oof
            oof_mask[name] = mask
            save_checkpoint(checkpoint, data["cache_key"], {
                "pipeline": pipeline, "record": record, "oof": oof,
                "mask": mask, "strategy_choice": strategy_choice[name],
            })
            model = pipeline.named_steps["model"]
            if hasattr(model, "save"):
                try:
                    model.save(MDL_DIR / f"{name.replace(' ', '_').lower()}.pkl")
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not save %s: %s", name, exc)
        except Exception as exc:
            log.warning("Skipping %s: %s: %s", name, type(exc).__name__, exc)
            records[name] = {"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if not trained:
        raise RuntimeError("Every requested model failed; no benchmark can be reported.")
    _write_json({"seed": args.seed, "models": records}, RES_DIR / "best_params.json")
    _write_yaml({"seed": args.seed, "cutoff": data["cutoff"], "probe_imbalance_strategy": imb["best_strategy"],
                 "strategy_selection": args.strategy_selection,
                 "tuning_enabled": bool(args.tune), "models": records}, RES_DIR / "best_params.yaml")
    return dict(trained_models=trained, tuning_records=records, oof_proba=oof_proba, oof_mask=oof_mask,
                strategy_choice=strategy_choice)


# ── Stage 4: Evaluation ───────────────────────────────────────────────────────

def stage_evaluation(data: dict, train_out: dict, imb: dict, args) -> dict:
    log.info("=" * 60)
    log.info("STAGE 4: Evaluation (selection on OOF; held-out test reporting)")
    log.info("=" * 60)

    X_test, y_test = data["X_test"], data["y_test"]
    y_train = data["y_train"].to_numpy()
    results, oof_results, roc_data, thresholds, test_probas, test_preds = {}, {}, {}, {}, {}, {}

    for name, pipeline in train_out["trained_models"].items():
        oof, mask = train_out["oof_proba"][name], train_out["oof_mask"][name]
        opt_t, opt_score = _threshold_from_oof(y_train[mask], oof[mask], args)
        oof_m = compute_metrics(y_train[mask], oof[mask] >= opt_t, oof[mask], opt_t)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= opt_t).astype(int)
        m = compute_metrics(y_test.values, pred, proba, threshold=opt_t)
        oof_results[name], results[name] = oof_m, m
        roc_data[name] = {"y_true": y_test.values, "y_proba": proba}
        thresholds[name], test_probas[name], test_preds[name] = opt_t, proba, pred
        # Save row-aligned predictions so future threshold/CI updates need no fit.
        np.savez_compressed(
            RES_DIR / f"predictions_{name.replace(' ', '_').lower()}.npz",
            train_index=data["X_train"].index.to_numpy().astype(str),
            test_index=X_test.index.to_numpy().astype(str), y_train=y_train, y_test=y_test.to_numpy(),
            oof_proba=oof, oof_mask=mask, test_proba=proba, threshold=np.array(opt_t),
        )
        log.info("%s | OOF AUROC=%.4f  %s*=%.4f @ t=%.2f | test AUROC=%.4f  F1=%.4f  MCC=%.4f",
                 name, oof_m["AUROC"], args.threshold_criterion, opt_score, opt_t,
                 m["AUROC"], m["F1"], m["MCC"])

    results_df = build_results_table(results)
    results_df.to_csv(RES_DIR / "test_metrics.csv")
    build_results_table(oof_results).to_csv(RES_DIR / "oof_metrics.csv")
    _save_latex_table(results_df, RES_DIR / "table2_test_metrics.tex",
                      caption="Classification Performance on the Held-Out Test Partition",
                      label="tab:results")
    log.info("\n%s", results_df.to_string())

    plot_roc_curves(roc_data, save_path=FIG_DIR / f"fig02_roc_curves.{FIGURE_EXT}")
    plot_pr_curves(roc_data, save_path=FIG_DIR / f"fig03_pr_curves.{FIGURE_EXT}")

    # Champion chosen on OOF AUROC; the test partition is never consulted.
    best_name = max(oof_results, key=lambda n: oof_results[n]["AUROC"])
    best_pipeline = train_out["trained_models"][best_name]
    best_proba, best_opt_t = test_probas[best_name], thresholds[best_name]
    best_pred = test_preds[best_name]
    plot_confusion_matrix(
        y_test.values, best_pred, model_name=best_name,
        save_path=FIG_DIR / f"fig04_confusion_{best_name.lower().replace(' ', '_')}.{FIGURE_EXT}",
    )

    pairwise_df = pd.DataFrame()
    names = sorted(results, key=lambda n: results[n]["AUROC"], reverse=True)
    if len(names) >= 2:
        rows = []
        for i, m1 in enumerate(names):
            for m2 in names[i + 1:]:
                dl = delong_test(y_test.values, test_probas[m1], test_probas[m2])
                mn = mcnemar_test(y_test.values, test_preds[m1], test_preds[m2])
                rows.append({"model_a": m1, "model_b": m2, "auc_a": dl["auc_a"], "auc_b": dl["auc_b"],
                             "auc_diff": dl["auc_a"] - dl["auc_b"], "delong_z": dl["z"],
                             "delong_p": dl["p_value"], "mcnemar_chi2": mn["chi2"],
                             "mcnemar_p": mn["p_value"], "mcnemar_b": mn["b"], "mcnemar_c": mn["c"],
                             "threshold_a": thresholds[m1], "threshold_b": thresholds[m2]})
        pairwise_df = pd.DataFrame(rows)
        pairwise_df["delong_p_holm"] = holm_adjust(pairwise_df["delong_p"].to_numpy())
        pairwise_df["mcnemar_p_holm"] = holm_adjust(pairwise_df["mcnemar_p"].to_numpy())
        pairwise_df.to_csv(RES_DIR / "pairwise_model_tests.csv", index=False)

    bootstrap_ci, paired_ci = pd.DataFrame(), pd.DataFrame()
    if args.revision_analyses and args.bootstrap_reps > 0:
        bootstrap_ci = bootstrap_metric_intervals(y_test.values, test_probas, thresholds,
                                                  n_boot=args.bootstrap_reps, seed=args.seed)
        bootstrap_ci.to_csv(RES_DIR / "bootstrap_metric_confidence_intervals.csv", index=False)
        paired_ci = paired_bootstrap_differences(y_test.values, test_probas, thresholds,
                                                 n_boot=args.bootstrap_reps, seed=args.seed)
        paired_ci.to_csv(RES_DIR / "paired_bootstrap_model_differences.csv", index=False)

    # Fig 10: descriptive model x strategy grid (fixed params, fixed 0.5 threshold).
    ABLATION_TIMEOUT_S, TABNET_ABLATION_TIMEOUT_S = 360, 480
    ablation_all = []
    strategies = imb["ablation_df"]["Strategy"].tolist()
    for name in ([] if args.skip_ablation_grid else train_out["trained_models"]):
        if name == "TabNet" and not args.full_ablation:
            log.info("Skipping TabNet in Fig 10 ablation (pass --full-ablation to include).")
            continue
        for strat in strategies:
            try:
                if name == "TabNet":
                    from models.tabnet_model import fit_tabnet_isolated
                    X_res, y_res = apply_resampling(data["X_train"], data["y_train"], strat,
                                                    _smote_kwargs(strat, args.seed), data["nominal_columns"])
                    p = fit_tabnet_isolated(X_res, y_res, None, None, X_test,
                                            timeout_s=TABNET_ABLATION_TIMEOUT_S, seed=args.seed)
                    if p is None:
                        continue
                else:
                    with _time_limit(ABLATION_TIMEOUT_S, label=f"{name}/{strat}"):
                        pipe = _make_pipeline(strat, name, args.seed, data)
                        pipe.fit(data["X_train"], data["y_train"])
                        p = pipe.predict_proba(X_test)[:, 1]
                ev = compute_metrics(y_test.values, (p >= 0.5).astype(int), p)
                ablation_all.append({"Model": name, "Strategy": strat, **ev})
            except Exception as exc:
                log.warning("Fig 10 ablation skipped for %s/%s (%s: %s)", name, strat, type(exc).__name__, exc)
    if ablation_all:
        abl_df = pd.DataFrame(ablation_all)
        abl_df.to_csv(RES_DIR / "smote_model_ablation.csv", index=False)
        plot_smote_ablation(abl_df, metric="AUROC", save_path=FIG_DIR / f"fig10_smote_ablation.{FIGURE_EXT}")

    prep = best_pipeline.named_steps["prep"]
    Xt_train = prep.transform(data["X_train"])
    Xt_test = prep.transform(X_test)
    return dict(results=results, oof_results=oof_results, roc_data=roc_data, best_name=best_name,
                best_pipeline=best_pipeline, best_model=best_pipeline.named_steps["model"],
                Xt_train=Xt_train, Xt_test=Xt_test,
                best_proba=best_proba, best_pred=best_pred, best_opt_t=best_opt_t,
                thresholds=thresholds, test_probas=test_probas, pairwise_df=pairwise_df,
                bootstrap_ci=bootstrap_ci, paired_ci=paired_ci)


# ── Stage 5: SHAP ─────────────────────────────────────────────────────────────

def stage_shap(data: dict, eval_out: dict, args) -> dict:
    log.info("=" * 60)
    log.info("STAGE 5: SHAP Explanations")
    log.info("=" * 60)

    best_model, best_name = eval_out["best_model"], eval_out["best_name"]
    X_train, X_test, y_test = eval_out["Xt_train"], eval_out["Xt_test"], data["y_test"]
    feature_names = data["feature_names"]
    rng = np.random.default_rng(args.seed)

    bg_idx = rng.choice(len(X_train), size=min(SHAP_BACKGROUND_SAMPLES, len(X_train)), replace=False)
    X_bg = X_train.iloc[bg_idx]
    model_type = "tree" if best_name in TREE_MODELS else "kernel"
    explainer = get_shap_explainer(best_model, X_bg, model_type=model_type)

    # Random test subsample (chronological order would otherwise bias temporal runs).
    n_shap = min(2000 if model_type == "tree" else 300, len(X_test))
    xai_idx = np.sort(rng.choice(len(X_test), size=n_shap, replace=False))
    X_exp = X_test.iloc[xai_idx]
    sv = compute_shap_values(explainer, X_exp, model_type=model_type)

    ev = explainer.expected_value
    ev_c1 = ev[1] if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0 and len(ev) > 1 else ev
    proba = best_model.predict_proba(X_exp)[:, 1]
    additivity = shap_consistency_check(sv, proba, ev_c1)
    _write_json({"model": best_name, "output_space": getattr(explainer, "output_space", "unknown"),
                 "n_explained": int(n_shap), **additivity}, RES_DIR / "shap_manifest.json")

    plot_shap_summary(sv, X_exp, title=f"SHAP Summary — {best_name}",
                      save_path=FIG_DIR / f"fig05a_shap_summary_{best_name.replace(' ', '_')}.{FIGURE_EXT}")
    plot_shap_bar(sv, X_exp, title=f"Mean |SHAP| — {best_name}",
                  save_path=FIG_DIR / f"fig05b_shap_bar_{best_name.replace(' ', '_')}.{FIGURE_EXT}")
    top_feat = top_k_features(sv, feature_names, k=1)["feature"].iloc[0]
    plot_shap_dependence(sv, X_exp, feature=top_feat,
                         save_path=FIG_DIR / f"fig06_shap_dep_{top_feat}.{FIGURE_EXT}")
    top_df = top_k_features(sv, feature_names, k=20)
    top_df.to_csv(RES_DIR / "shap_feature_ranking.csv", index=False)
    _save_latex_table(top_df, RES_DIR / "table3_shap_ranking.tex",
                      caption="SHAP Feature Importance Ranking (Top 20)", label="tab:shap_ranking")

    fi = getattr(best_model, "feature_importances_", None)
    if fi is not None and len(fi) == len(feature_names):
        plot_feature_importance_comparison(
            pd.Series(np.abs(sv).mean(axis=0), index=feature_names), pd.Series(fi, index=feature_names),
            top_k=15, save_path=FIG_DIR / f"fig08_importance_comparison.{FIGURE_EXT}",
        )
    plot_threshold_sweep(y_test.values, eval_out["best_proba"], model_name=best_name,
                         save_path=FIG_DIR / f"fig09_threshold_sweep.{FIGURE_EXT}")
    return dict(explainer=explainer, shap_vals=sv, X_exp=X_exp, xai_idx=xai_idx, top_feature=top_feat, ev_c1=ev_c1)


# ── Stage 6: LIME ─────────────────────────────────────────────────────────────

def stage_lime(data: dict, eval_out: dict, shap_out: dict, args) -> dict:
    log.info("=" * 60)
    log.info("STAGE 6: LIME Explanations")
    log.info("=" * 60)

    best_model, best_pred, best_proba = eval_out["best_model"], eval_out["best_pred"], eval_out["best_proba"]
    X_test, y_test, feature_names = eval_out["Xt_test"], data["y_test"], data["feature_names"]
    cat_names = {idx: [str(c) for c in data["nominal_categories"][col]] + ["unknown"]
                 for idx, col in zip(data["nominal_idx"], data["nominal_columns"])}

    def _explainer(seed):
        return CreditLimeExplainer(
            X_train=eval_out["Xt_train"], feature_names=feature_names, predict_fn=best_model.predict_proba,
            num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES, random_state=seed,
            categorical_features=list(data["nominal_idx"]), categorical_names=cat_names,
        )
    lime_exp = _explainer(args.seed)

    instances = _select_representative_instances(X_test, y_test, best_pred, best_proba)
    log.info("Local instances selected: %s", instances)
    exps, exp_labels = [], []
    for kind, idx in instances.items():
        exp = lime_exp.explain_instance(X_test.iloc[idx].values, label=1)
        exps.append(exp)
        exp_labels.append(f"{kind} — idx {idx} | P(def)={best_proba[idx]:.3f}")
        plot_lime_explanation(exp, label=1, title=f"LIME ({kind}) — {eval_out['best_name']}",
                              save_path=FIG_DIR / f"fig07_{kind.lower()}_lime.{FIGURE_EXT}")
    plot_lime_multi_instance(exps, exp_labels, label=1,
                             save_path=FIG_DIR / f"fig07_lime_multi_instance.{FIGURE_EXT}")

    sv, X_exp = shap_out["shap_vals"], shap_out["X_exp"]
    n_agree = min(200, len(X_exp))
    lime_batch, lime_scores = lime_exp.batch_explain(X_exp.iloc[:n_agree], n=n_agree, return_scores=True)
    agree_df = compute_shap_lime_agreement(sv[:n_agree], lime_batch, feature_names)

    xai_df = pd.DataFrame()
    if args.revision_analyses:
        repeat_n = min(args.revision_analysis_n, n_agree)
        repeated_lime = []
        for i in range(repeat_n):
            row = []
            for repeat in range(3):
                rep = _explainer(args.seed + repeat + 1)
                row.append(rep.explanation_to_series(rep.explain_instance(X_exp.iloc[i].values, label=1), label=1))
            repeated_lime.append(row)
        xai_df = explanation_metrics(sv[:repeat_n], lime_batch[:repeat_n], feature_names, repeated_lime,
                                     lime_scores=lime_scores[:repeat_n])
        xai_df.to_csv(RES_DIR / "explanation_quality_metrics.csv", index=False)
        _save_latex_table(xai_df.describe().round(4), RES_DIR / "table5_explanation_quality.tex",
                          caption="Explanation parsimony, stability, and SHAP--LIME agreement",
                          label="tab:explanation_quality")
    agree_df.to_csv(RES_DIR / "shap_lime_agreement.csv", index=False)
    if not agree_df.empty:
        _save_latex_table(agree_df.describe().round(4), RES_DIR / "table4_shap_lime_agreement.tex",
                          caption="SHAP–LIME Spearman Rank Correlation Statistics", label="tab:agreement")
    return dict(lime_exp=lime_exp, exps=exps, agree_df=agree_df, xai_df=xai_df, shap_vals=sv, X_exp=X_exp)


# ── Stage 7: Revision analyses ────────────────────────────────────────────────

def stage_revision_analyses(data, imb, train_out, eval_out, lime_out, args) -> dict | None:
    if not args.revision_analyses:
        return None
    log.info("=" * 60)
    log.info("STAGE 7: Revision Analyses")
    log.info("=" * 60)

    y_train = data["y_train"].to_numpy()
    y_test = data["y_test"]
    nominal = data["nominal_columns"]
    groups_train, groups_test = data["groups_train"], data["groups_test"]
    have_groups = not groups_train.empty and not groups_test.empty
    best_name = eval_out["best_name"]

    # 1. Calibration, robustness (test), and OOF-based selection diagnostics.
    robustness_rows, selection_rows, calib_rows, calib_bins = [], [], [], []
    rng = np.random.default_rng(args.seed + 500)
    robust_idx = rng.choice(len(y_train), size=min(2000, len(y_train)), replace=False)
    for name, pipeline in train_out["trained_models"].items():
        t = eval_out["thresholds"][name]
        test_p = eval_out["test_probas"][name]
        summary, bins = calibration_diagnostics(y_test, test_p)
        calib_rows.append({"model": name, **summary})
        if not bins.empty:
            bins.insert(0, "model", name)
            bins.insert(1, "seed", args.seed)
            calib_bins.append(bins)
        frame = evaluate_robustness(pipeline, data["X_train"], data["X_test"], y_test, t, args.seed,
                                    nominal_columns=nominal)
        frame.insert(0, "model", name)
        robustness_rows.append(frame)

        # Selection-side robustness: final model on a training subsample (in-sample,
        # descriptive). Retention is a ratio so the in-sample bias largely cancels.
        train_frame = evaluate_robustness(
            pipeline, data["X_train"], data["X_train"].iloc[robust_idx], data["y_train"].iloc[robust_idx],
            t, args.seed + 1000, nominal_columns=nominal,
        )
        retention = float(train_frame.loc[train_frame["perturbation"] != "none", "AUROC_retention"].mean())
        oof, mask = train_out["oof_proba"][name], train_out["oof_mask"][name]
        oof_m = compute_metrics(y_train[mask], oof[mask] >= t, oof[mask], t)
        max_gap = float("nan")
        if have_groups:
            curve = fairness_utility_curve(y_train[mask], oof[mask], groups_train.iloc[mask].reset_index(drop=True),
                                           cost_fp=args.cost_fp, cost_fn=args.cost_fn)
            if not curve.empty:
                nearest = curve.iloc[(curve["threshold"] - t).abs().argsort()[:1]]
                max_gap = float(nearest["max_group_gap"].iloc[0])
        selection_rows.append({
            "model": name, "AUROC": oof_m["AUROC"], "Brier": oof_m["Brier"],
            "base_rate_brier": float(y_train[mask].mean() * (1 - y_train[mask].mean())),
            "robustness_retention": retention, "max_fairness_gap": max_gap,
            "selection_source": "oof_predictions; robustness on train subsample",
        })
    robustness = pd.concat(robustness_rows, ignore_index=True)
    robustness.to_csv(RES_DIR / "robustness_perturbations.csv", index=False)
    pd.DataFrame(calib_rows).to_csv(RES_DIR / "heldout_calibration_diagnostics.csv", index=False)
    if calib_bins:
        pd.concat(calib_bins, ignore_index=True).to_csv(RES_DIR / "heldout_calibration_bins.csv", index=False)

    reliability_selection = reliability_constrained_selection(
        pd.DataFrame(selection_rows), min_robustness=args.min_robustness, max_fairness_gap=args.max_fairness_gap,
    )
    reliability_selection.to_csv(RES_DIR / "reliability_selection.csv", index=False)
    feasible = reliability_selection[reliability_selection["selected"] & reliability_selection["feasible"]]
    reliable_name = str(feasible["model"].iloc[0]) if not feasible.empty else None

    # 2. Controlled ablation: strategy x threshold rule x calibration, champion family,
    #    tuned params, every choice on OOF predictions.
    best_params = train_out["tuning_records"].get(best_name, {}).get("best_params", {}) or {}
    champion_strategy = train_out["strategy_choice"].get(best_name, {}).get("strategy", imb["best_strategy"])
    strategy_runs = {}
    for strategy in ([] if args.skip_controlled_ablation else imb["ablation_df"]["Strategy"]):
        try:
            pipe = _make_pipeline(strategy, best_name, args.seed, data, **best_params)
            oof, mask = _oof(pipe, data, args)
            pipe.fit(data["X_train"], data["y_train"])
            strategy_runs[strategy] = {"oof_proba": oof[mask], "y_oof": y_train[mask],
                                       "test_proba": pipe.predict_proba(data["X_test"])[:, 1]}
        except Exception as exc:
            log.warning("Controlled ablation skipped for %s (%s: %s)", strategy, type(exc).__name__, exc)
    ablation = run_controlled_ablation(strategy_runs, y_test, threshold_criterion=args.threshold_criterion,
                                       cost_fp=args.cost_fp, cost_fn=args.cost_fn)
    ablation["ablation"] = "resampling_threshold_calibration"

    if not args.skip_controlled_ablation and args.dataset == "uci" and best_name != "TabNet":
        raw = load_dataset(dataset="uci", data_dir=DATA_DIR, test_size=TEST_SIZE, random_state=args.seed,
                           engineer=False, split_mode=args.split_mode)
        raw_data = _data_dict(raw, args.split_mode)
        pipe = _make_pipeline(champion_strategy, best_name, args.seed, raw_data, **best_params)
        oof, mask = _oof(pipe, raw_data, args)
        pipe.fit(raw.X_train, raw.y_train)
        raw_t, _ = _threshold_from_oof(raw.y_train.to_numpy()[mask], oof[mask], args)
        raw_test = pipe.predict_proba(raw.X_test)[:, 1]
        rows = [
            {"ablation": "feature_engineering", "strategy": "raw_features",
             "threshold_rule": f"oof_{args.threshold_criterion}",
             **compute_metrics(raw.y_test.to_numpy(), raw_test >= raw_t, raw_test, raw_t)},
            {"ablation": "feature_engineering", "strategy": "engineered_features",
             "threshold_rule": f"oof_{args.threshold_criterion}",
             **compute_metrics(y_test.to_numpy(), eval_out["best_proba"] >= eval_out["best_opt_t"],
                               eval_out["best_proba"], eval_out["best_opt_t"])},
        ]
        ablation = pd.concat([ablation, pd.DataFrame(rows)], ignore_index=True, sort=False)
    if not args.skip_controlled_ablation:
        ablation.insert(0, "model", best_name)
        ablation.to_csv(RES_DIR / "controlled_ablations.csv", index=False)

    # 3. Fairness--utility curves on test for every model.
    fairness = pd.DataFrame()
    if have_groups:
        curves = []
        for name in train_out["trained_models"]:
            curve = fairness_utility_curve(y_test, eval_out["test_probas"][name], groups_test,
                                           cost_fp=args.cost_fp, cost_fn=args.cost_fn)
            curve.insert(0, "model", name)
            curves.append(curve)
        fairness = pd.concat(curves, ignore_index=True)
        fairness.to_csv(RES_DIR / "fairness_utility_curve.csv", index=False)

    # 4. Equal-opportunity group thresholds fitted on OOF predictions, frozen, then
    #    evaluated once on the test partition.
    post_rows, post_ci = [], []
    if have_groups:
        for name in train_out["trained_models"]:
            oof, mask = train_out["oof_proba"][name], train_out["oof_mask"][name]
            g_oof = groups_train.iloc[mask].reset_index(drop=True)
            test_p = eval_out["test_probas"][name]
            for col in groups_train.columns:
                policy = fit_equal_opportunity_threshold_policy(
                    y_train[mask], oof[mask], g_oof, col, max_tpr_gap=args.max_fairness_gap,
                    cost_fp=args.cost_fp, cost_fn=args.cost_fn,
                )
                if policy["status"] != "fit":
                    post_rows.append({"model": name, "group_column": col, "status": policy["status"]})
                    continue
                heldout = evaluate_group_threshold_policy(y_test, test_p, groups_test, policy,
                                                         cost_fp=args.cost_fp, cost_fn=args.cost_fn)
                cand_pred = heldout.pop("test_predictions")
                ref_t = float(eval_out["thresholds"][name])
                ref_pred = test_p >= ref_t
                ref_row = fairness_utility_curve(y_test, test_p, groups_test[[col]], thresholds=[ref_t],
                                                 cost_fp=args.cost_fp, cost_fn=args.cost_fn).iloc[0]
                ref_acc = float((ref_pred == y_test.to_numpy()).mean())
                post_rows.append({
                    "model": name, "group_column": col, "status": "evaluated",
                    "selection_source": "oof_predictions",
                    "oof_target_tpr": policy["target_tpr"], "oof_tpr_gap": policy["validation_tpr_gap"],
                    "oof_utility": policy["validation_utility"], "oof_feasible": policy["feasible"],
                    "max_tpr_gap_constraint": policy["max_tpr_gap"],
                    "reference_threshold_rule": f"oof_{args.threshold_criterion}_global_threshold",
                    "reference_threshold": ref_t, "reference_test_accuracy": ref_acc,
                    "reference_test_utility": float(ref_row["expected_utility"]),
                    "reference_test_tpr_gap": float(ref_row[f"{col}_tpr_gap"]),
                    "reference_test_fpr_gap": float(ref_row[f"{col}_fpr_gap"]),
                    "reference_test_selection_gap": float(ref_row[f"{col}_selection_gap"]),
                    "test_accuracy_change": heldout["test_accuracy"] - ref_acc,
                    "test_utility_change": heldout["test_utility"] - float(ref_row["expected_utility"]),
                    "test_tpr_gap_change": heldout["test_tpr_gap"] - float(ref_row[f"{col}_tpr_gap"]),
                    "thresholds_by_group_json": json.dumps(policy["thresholds_by_group"], sort_keys=True),
                    **{k: v for k, v in heldout.items() if k not in {"per_group", "thresholds_by_group"}},
                })
                if args.bootstrap_reps > 0:
                    ci = bootstrap_fairness_policy_differences(
                        y_test, ref_pred, cand_pred, groups_test[[col]], n_boot=args.bootstrap_reps,
                        seed=args.seed + 2000 + len(post_rows), cost_fp=args.cost_fp, cost_fn=args.cost_fn,
                    )
                    ci.insert(0, "group_column", col)
                    ci.insert(0, "model", name)
                    ci.insert(0, "comparison", "oof_equal_opportunity_minus_global_oof_threshold")
                    post_ci.append(ci)
        pd.DataFrame(post_rows).to_csv(RES_DIR / "oof_fitted_fairness_postprocessing.csv", index=False)
    if post_ci:
        pd.concat(post_ci, ignore_index=True).to_csv(RES_DIR / "fairness_postprocessing_bootstrap_intervals.csv", index=False)

    # 5. Descriptive component-wise audit score for the champion.
    xai_df = lime_out.get("xai_df", pd.DataFrame()) if lime_out else pd.DataFrame()
    fidelity = pd.DataFrame()
    if lime_out and not xai_df.empty:
        n_fid = min(len(xai_df), len(lime_out["X_exp"]))
        fidelity = explanation_fidelity(eval_out["best_model"], lime_out["X_exp"].iloc[:n_fid], eval_out["Xt_train"],
                                        lime_out["shap_vals"][:n_fid], nominal_columns=nominal)
        fidelity.to_csv(RES_DIR / "explanation_fidelity_metrics.csv", index=False)
        xai_df = xai_df.merge(fidelity, on="instance", how="left")
        xai_df.to_csv(RES_DIR / "explanation_quality_metrics.csv", index=False)
    score_rows = []
    if not xai_df.empty:
        stability = (xai_df["lime_stability_rho"].mean() + 1.0) / 2.0
        parsimony = xai_df["shap_parsimony"].mean()
        comp = float(np.clip(xai_df.get("comprehensiveness", pd.Series([0.0])).mean(), 0, 1))
        lime_fid = float(np.clip(xai_df.get("lime_local_r2", pd.Series([0.0])).mean(), 0, 1))
        xai_component = float(np.clip((stability + parsimony + comp + lime_fid) / 4.0, 0, 1))
        metrics = eval_out["results"][best_name]
        rob = robustness[(robustness["model"] == best_name) & (robustness["perturbation"] != "none")]
        rob_component = float(np.clip(rob["AUROC_retention"].mean(), 0, 1)) if not rob.empty else 0.0
        brier_naive = float(y_test.mean() * (1 - y_test.mean()))
        brier_skill = 1.0 - float(metrics["Brier"]) / max(brier_naive, 1e-12)
        pq = float(np.clip(np.sqrt(max(metrics["AUROC"], 0) * max(brier_skill, 0)), 0, 1))
        inv_gap = None
        if not fairness.empty:
            row = fairness[fairness["model"] == best_name]
            row = row.iloc[(row["threshold"] - eval_out["thresholds"][best_name]).abs().argsort()[:1]]
            if not row.empty and np.isfinite(row["max_group_gap"].iloc[0]):
                inv_gap = float(np.clip(1.0 - row["max_group_gap"].iloc[0], 0, 1))
        score_rows.append({"model": best_name, **reliability_score(pq, rob_component, xai_component, inv_gap)})
    else:
        score_rows.append({"model": best_name, "status": "unavailable_without_xai"})
    pd.DataFrame(score_rows).to_csv(RES_DIR / "reliability_scores.csv", index=False)

    _write_json({
        "seed": args.seed, "cutoff": data["cutoff"], "protocol": PROTOCOL,
        "selection_data": "out-of-fold predictions inside the training partition",
        "inner_cv": type(_inner_cv(data, args)).__name__,
        "threshold_criterion": args.threshold_criterion,
        "robustness_levels": [0.05, 0.10, 0.20],
        "realistic_shift_families": ["missingness", "correlated_missingness", "numeric_noise", "covariate_shift"],
        "nominal_columns_protected_from_perturbation": nominal,
        "reliability_selected_model": reliable_name,
        "reliability_constraints": {"min_robustness": args.min_robustness, "max_fairness_gap": args.max_fairness_gap},
        "utility_costs": {"false_positive": args.cost_fp, "false_negative": args.cost_fn},
        "bootstrap_reps": args.bootstrap_reps,
        "multiple_testing_correction": "Holm step-down within each run; descriptive ranks across dependent runs",
        "fairness_postprocessing": "equal-opportunity group thresholds fitted on OOF predictions; test evaluated once",
        "reliability_aggregation": "descriptive_geometric_mean; unscored when group fairness data are absent",
    }, RES_DIR / "revision_analysis_manifest.json")
    return {"robustness": robustness, "ablations": ablation, "fairness": fairness,
            "reliability_selection": reliability_selection, "reliable_model": reliable_name, "fidelity": fidelity}


# ── External validation ───────────────────────────────────────────────────────

def stage_external_validation(args) -> pd.DataFrame:
    """Fixed-parameter, OOF-thresholded evaluation on additional datasets (random split)."""
    rows = []
    model_names = FAST_MODELS if args.fast else ALL_MODELS
    for dataset_name in args.external_datasets:
        if dataset_name == args.dataset:
            continue
        try:
            loaded = load_dataset(dataset=dataset_name, data_dir=DATA_DIR, test_size=TEST_SIZE,
                                  random_state=args.seed, split_mode="random")
            data = _data_dict(loaded, "random")
            for name in model_names:
                try:
                    pipe = _make_pipeline("None", name, args.seed, data)
                    oof, mask = _oof(pipe, data, args)
                    t, _ = _threshold_from_oof(loaded.y_train.to_numpy()[mask], oof[mask], args)
                    pipe.fit(loaded.X_train, loaded.y_train)
                    p = pipe.predict_proba(loaded.X_test)[:, 1]
                    rows.append({"dataset": dataset_name, "model": name, "seed": args.seed, "status": "ok",
                                 **compute_metrics(loaded.y_test.to_numpy(), p >= t, p, t)})
                except Exception as exc:
                    rows.append({"dataset": dataset_name, "model": name, "seed": args.seed,
                                 "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            rows.append({"dataset": dataset_name, "model": "__dataset__", "seed": args.seed,
                         "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(RES_DIR / "multi_dataset_validation.csv", index=False)
    return result


# ── Run orchestration ─────────────────────────────────────────────────────────

def _set_run_dirs(run_id: str) -> dict:
    global FIG_DIR, RES_DIR, MDL_DIR
    run_dir = RUNS_DIR / run_id
    FIG_DIR, RES_DIR, MDL_DIR = run_dir / "figures", run_dir / "results", run_dir / "models"
    for d in (FIG_DIR, RES_DIR, MDL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "fig_dir": FIG_DIR, "res_dir": RES_DIR, "mdl_dir": MDL_DIR}


def _run_one(args, seed: int, cutoff: str | None) -> dict:
    args.seed = seed
    run_id = f"seed_{seed}" if cutoff is None else f"cutoff_{cutoff}_seed_{seed}"
    dirs = _set_run_dirs(run_id)
    # A failed rerun must not leave an old completion eligible for aggregation.
    (RES_DIR / "run_manifest.json").unlink(missing_ok=True)
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  Credit Default Prediction with XAI (v3)     ║")
    log.info("║  Dataset: %-35s║", args.dataset)
    log.info("║  Run: %-39s║", run_id)
    log.info("╚══════════════════════════════════════════════╝")

    data_out = stage_data(args, cutoff)
    data_out["dataset_checksums"] = _dataset_checksums([args.dataset])
    data_out["cache_key"] = training_key(args, data_out)
    imb_out = stage_imbalance_ablation(data_out, args)
    train_out = stage_training(data_out, imb_out, args)
    eval_out = stage_evaluation(data_out, train_out, imb_out, args)
    shap_out = lime_out = None
    if not args.skip_xai:
        shap_out = stage_shap(data_out, eval_out, args)
        lime_out = stage_lime(data_out, eval_out, shap_out, args)
    revision_out = stage_revision_analyses(data_out, imb_out, train_out, eval_out, lime_out, args)
    external_out = stage_external_validation(args) if args.external_datasets else pd.DataFrame()

    manifest = {
        "run_id": run_id, "seed": seed, "cutoff": cutoff, "dataset": args.dataset,
        "split_mode": args.split_mode, "protocol": PROTOCOL,
        "data_meta": data_out["meta"], "fast": bool(args.fast), "tune": bool(args.tune),
        "tune_iter": int(args.tune_iter), "inner_folds": int(args.inner_folds),
        "inner_temporal_folds": int(args.inner_temporal_folds),
        "threshold_criterion": args.threshold_criterion, "skip_xai": bool(args.skip_xai),
        "cost_fp": args.cost_fp, "cost_fn": args.cost_fn,
        "horizon_months": args.horizon_months, "test_window_months": args.test_window_months,
        "prosper_include_pricing": args.prosper_include_pricing,
        "skip_tabnet": args.skip_tabnet,
        "best_model": eval_out["best_name"],
        "best_oof_AUROC": eval_out["oof_results"][eval_out["best_name"]]["AUROC"],
        "best_test_AUROC": eval_out["results"][eval_out["best_name"]]["AUROC"],
        "best_threshold": eval_out["best_opt_t"],
        "probe_imbalance_strategy": imb_out["best_strategy"],
        "champion_imbalance_strategy": train_out["strategy_choice"].get(eval_out["best_name"], {}).get("strategy"),
        "strategy_selection": args.strategy_selection,
        "resampling_helped": imb_out["smote_summary"]["resampling_helped"],
        "dataset_checksums": _dataset_checksums([args.dataset, *args.external_datasets]),
        "source_fingerprint": _source_fingerprint(), "package_versions": _package_versions(),
        "revision_analyses": bool(args.revision_analyses),
        "bootstrap_reps": args.bootstrap_reps,
        "cache_key": data_out["cache_key"],
        "requested_models": args.models,
        "core_only": args.core_only,
        "skip_ablation_grid": args.skip_ablation_grid,
        "skip_controlled_ablation": args.skip_controlled_ablation,
        "bootstrap_scope": "conditional_on_fitted_models_and_selected_thresholds",
        "oof_scope": "selection_diagnostics; hyperparameters selected on these training folds",
        "reliability_selected_model": revision_out.get("reliable_model") if revision_out else None,
        "results_dir": str(RES_DIR), "figures_dir": str(FIG_DIR),
    }
    _write_json({"results": eval_out["results"], "oof_results": eval_out["oof_results"],
                 "imbalance": imb_out["smote_summary"], "tuning_records": train_out["tuning_records"]},
                RES_DIR / "run_summary.json")
    # The completion manifest is written last; partial runs are not aggregated.
    _write_json(manifest, RES_DIR / "run_manifest.json")
    log.info("RUN %s COMPLETE | champion %s | OOF AUROC %.4f | test AUROC %.4f", run_id,
             eval_out["best_name"], manifest["best_oof_AUROC"], manifest["best_test_AUROC"])
    # Full fitted models are checkpointed on disk, not retained for all seeds.
    summary_eval = {key: eval_out[key] for key in ("results", "oof_results", "pairwise_df", "bootstrap_ci", "paired_ci")}
    return {"run_id": run_id, "seed": seed, "cutoff": cutoff, "dirs": dirs, "manifest": manifest,
            "imbalance": imb_out, "train": {"tuning_records": train_out["tuning_records"]}, "eval": summary_eval, "revision": revision_out,
            "external_validation": external_out}


def _write_multi_run_summary(runs: list[dict]) -> None:
    # A selective rerun must retain other completed runs from the same protocol.
    current = {run["run_id"]: run for run in runs}
    settings = ("dataset", "split_mode", "protocol", "fast", "tune", "tune_iter", "inner_folds",
                "inner_temporal_folds", "threshold_criterion", "strategy_selection", "requested_models",
                "cost_fp", "cost_fn", "horizon_months", "test_window_months", "prosper_include_pricing", "skip_tabnet",
                "source_fingerprint", "dataset_checksums", "package_versions")
    reference = runs[0]["manifest"]
    for path in RUNS_DIR.glob("*/results/run_manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest["run_id"] in current or not (path.parent / "run_summary.json").exists():
            continue
        if any(manifest.get(key) != reference.get(key) for key in settings):
            log.warning("Excluding incompatible completed run from summary: %s", manifest["run_id"])
            continue
        saved = json.loads((path.parent / "run_summary.json").read_text())
        evaluation = {"results": saved["results"], "oof_results": saved["oof_results"]}
        for key, filename in (("pairwise_df", "pairwise_model_tests.csv"),
                              ("bootstrap_ci", "bootstrap_metric_confidence_intervals.csv"),
                              ("paired_ci", "paired_bootstrap_model_differences.csv")):
            eligible = key == "pairwise_df" or (manifest.get("revision_analyses") and manifest.get("bootstrap_reps", 0) > 0)
            if eligible and (path.parent / filename).exists():
                evaluation[key] = pd.read_csv(path.parent / filename)
        current[manifest["run_id"]] = {
            "run_id": manifest["run_id"], "seed": manifest["seed"], "cutoff": manifest["cutoff"],
            "manifest": manifest, "eval": evaluation, "dirs": {"res_dir": path.parent},
            "imbalance": {"smote_summary": saved["imbalance"]},
            "train": {"tuning_records": saved["tuning_records"]},
        }
    runs = [current[key] for key in sorted(current)]
    summary_dir = RUNS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, oof_rows, best_rows, imb_rows = [], [], [], []
    pairwise, smote_model, boot, paired, external = [], [], [], [], []
    tuning = {}
    for run in runs:
        key = {"run_id": run["run_id"], "seed": run["seed"], "cutoff": run["cutoff"]}
        for name, m in run["eval"]["results"].items():
            metric_rows.append({**key, "model": name, **m})
        for name, m in run["eval"]["oof_results"].items():
            oof_rows.append({**key, "model": name, **m})
        best_rows.append(run["manifest"])
        imb_rows.append({**key, **run["imbalance"]["smote_summary"]})
        tuning[run["run_id"]] = run["train"].get("tuning_records", {})
        for frame, bucket in ((run["eval"].get("pairwise_df"), pairwise),
                              (run["eval"].get("bootstrap_ci"), boot),
                              (run["eval"].get("paired_ci"), paired),
                              (run.get("external_validation"), external)):
            if frame is not None and not frame.empty:
                tmp = frame.copy()
                for k, v in reversed(list(key.items())):
                    tmp.insert(0, k, v)
                bucket.append(tmp)
        path = Path(run["dirs"]["res_dir"]) / "smote_model_ablation.csv"
        if path.exists() and not run["manifest"].get("skip_ablation_grid", False):
            tmp = pd.read_csv(path)
            for k, v in reversed(list(key.items())):
                tmp.insert(0, k, v)
            smote_model.append(tmp)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(summary_dir / "all_run_test_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(summary_dir / "all_run_oof_metrics.csv", index=False)
    pd.DataFrame(best_rows).drop(columns=["data_meta", "package_versions"], errors="ignore") \
        .to_csv(summary_dir / "all_run_best_models.csv", index=False)
    pd.DataFrame(imb_rows).to_csv(summary_dir / "all_run_imbalance_summary.csv", index=False)
    _write_yaml(tuning, summary_dir / "all_run_best_params.yaml")

    metric_cols = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Specificity", "G-Mean", "MCC", "Brier"]
    agg = metrics_df.groupby("model")[metric_cols].agg(["mean", "std", "count"]).round(4)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_runs"] = agg["AUROC_count"].astype(int)
    agg = agg.drop(columns=[c for c in agg.columns if c.endswith("_count")])
    agg.reset_index().to_csv(summary_dir / "all_run_metric_summary.csv", index=False)

    _write_json(descriptive_ranks(metrics_df), summary_dir / "descriptive_model_ranks.json")

    for bucket, name in ((pairwise, "all_run_pairwise_model_tests.csv"),
                         (boot, "all_run_bootstrap_metric_confidence_intervals.csv"),
                         (paired, "all_run_paired_bootstrap_model_differences.csv"),
                         (external, "all_run_multi_dataset_validation.csv")):
        if bucket:
            pd.concat(bucket, ignore_index=True).to_csv(summary_dir / name, index=False)
        else:
            (summary_dir / name).unlink(missing_ok=True)
    if smote_model:
        sm = pd.concat(smote_model, ignore_index=True)
        sm.to_csv(summary_dir / "all_run_model_smote_ablation.csv", index=False)
        sm_agg = sm.groupby(["Model", "Strategy"])[metric_cols].agg(["mean", "std", "count"]).round(4)
        sm_agg.columns = [f"{m}_{s}" for m, s in sm_agg.columns]
        sm_agg["n_runs"] = sm_agg["AUROC_count"].astype(int)
        sm_agg = sm_agg.drop(columns=[c for c in sm_agg.columns if c.endswith("_count")])
        sm_agg.reset_index().to_csv(summary_dir / "all_run_model_smote_ablation_summary.csv", index=False)
    else:
        for name in ("all_run_model_smote_ablation.csv", "all_run_model_smote_ablation_summary.csv"):
            (summary_dir / name).unlink(missing_ok=True)

    _write_json({
        "dataset": best_rows[0]["dataset"], "split_mode": best_rows[0]["split_mode"],
        "protocol": PROTOCOL, "runs": [r["run_id"] for r in runs],
        "dataset_checksums": best_rows[0]["dataset_checksums"],
        "source_fingerprint": best_rows[0]["source_fingerprint"],
        "package_versions": best_rows[0]["package_versions"],
        "run_manifests": [str(Path(r["results_dir"]) / "run_manifest.json") for r in best_rows],
    }, summary_dir / "reproducibility_manifest.json")
    log.info("Multi-run summary -> %s", summary_dir)


def main():
    parser = argparse.ArgumentParser(description="Credit Default XAI Experiment Pipeline (protocol v3)")
    parser.add_argument("--dataset", default=DATASET, choices=["uci", "lending_club", "south_german", "german", "prosper"])
    parser.add_argument("--external-datasets", nargs="*", default=[],
                        choices=["uci", "lending_club", "south_german", "german", "prosper"])
    parser.add_argument("--split-mode", choices=["random", "temporal"], default="random")
    parser.add_argument("--cutoffs", nargs="*", default=None,
                        help="Rolling-origin cutoffs (YYYY-MM) for temporal mode. Default: Prosper list in config.")
    parser.add_argument("--horizon-months", type=int, default=PROSPER_HORIZON_MONTHS,
                        help="Default-within-H-months label horizon (prosper).")
    parser.add_argument("--test-window-months", type=int, default=PROSPER_TEST_WINDOW_MONTHS)
    parser.add_argument("--prosper-include-pricing", action="store_true",
                        help="Keep BorrowerRate/BorrowerAPR as features (excluded by default).")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--inner-folds", type=int, default=INNER_CV_FOLDS)
    parser.add_argument("--inner-temporal-folds", type=int, default=INNER_TEMPORAL_FOLDS)
    parser.add_argument("--threshold-criterion", default=THRESHOLD_CRITERION,
                        choices=["f1", "g_mean", "youden", "cost"])
    parser.add_argument("--strategy-selection", default="per_model", choices=["per_model", "probe"],
                        help="Imbalance strategy chosen per model family on OOF AUROC (default) or once "
                             "with the LightGBM probe from stage 2.")
    parser.add_argument("--fast", action="store_true", help="Reduced run (3 models, 3 strategies, no TabNet)")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS + ["TabNet"],
                        help="Optional model subset for targeted checks; use a separate output root.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR,
                        help="Active output root. Defaults to results/outputs; original runs are archived separately.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching imbalance and per-model checkpoints.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Require matching checkpoints; never train models or run fitting ablations.")
    parser.add_argument("--core-only", action="store_true",
                        help="Selection, training, test metrics only; skip XAI and fitting ablation grids.")
    parser.add_argument("--skip-ablation-grid", action="store_true", help="Skip the extra descriptive model x strategy fits.")
    parser.add_argument("--skip-controlled-ablation", action="store_true", help="Skip revision ablations requiring extra fits.")
    parser.add_argument("--reuse-models-from", type=Path,
                        default=ORIGINAL_OUTPUTS_DIR if ORIGINAL_OUTPUTS_DIR.exists() else OUTPUTS_DIR,
                        help="Legacy output root. Reuse final models only after new selection agrees and metrics reproduce.")
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.set_defaults(tune=True)
    parser.add_argument("--tune-iter", type=int, default=RANDOM_SEARCH_N_ITER)
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--skip-tabnet", action="store_true", help="Exclude TabNet (saves K+1 fits per run)")
    parser.add_argument("--full-ablation", action="store_true", help="Include TabNet in the Fig 10 grid")
    parser.add_argument("--revision-analyses", action="store_true")
    parser.add_argument("--revision-analysis-n", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--min-robustness", type=float, default=0.90)
    parser.add_argument("--max-fairness-gap", type=float, default=0.10)
    args = parser.parse_args()
    if args.core_only:
        if args.external_datasets:
            parser.error("--core-only excludes external validation fits.")
        args.skip_xai = True
        args.revision_analyses = False
        args.skip_ablation_grid = True
        args.skip_controlled_ablation = True
    if args.evaluate_only:
        args.skip_ablation_grid = True
        args.skip_controlled_ablation = True
        if args.external_datasets:
            parser.error("--evaluate-only cannot run external validation, which fits new models.")
    if args.tune_iter < 1 or args.inner_folds < 2 or args.inner_temporal_folds < 2:
        parser.error("Search iterations must be positive and CV folds must be at least two.")
    for archive in (ORIGINAL_OUTPUTS_DIR, OUTPUTS_DIR.parent / "outputs_v2_archive"):
        if args.out_dir.resolve() == archive.resolve() or archive.resolve() in args.out_dir.resolve().parents:
            parser.error("Choose an active --out-dir; original and v2 result archives are protected.")
    if args.dataset == "german":
        args.dataset = "south_german"

    if args.split_mode == "temporal":
        cutoffs = args.cutoffs or PROSPER_DEFAULT_CUTOFFS
    else:
        cutoffs = [None]
        if args.cutoffs:
            log.warning("--cutoffs ignored for random split mode.")

    run_name = "_".join([args.dataset, *sorted(set(args.external_datasets))])
    if args.split_mode == "temporal":
        run_name += "_temporal"
    global RUNS_DIR
    RUNS_DIR = args.out_dir / run_name / "runs"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for cutoff in cutoffs:
        for seed in args.seeds:
            runs.append(_run_one(args, seed, cutoff))
    _write_multi_run_summary(runs)
    log.info("ALL RUNS COMPLETE | runs=%s", [r["run_id"] for r in runs])


if __name__ == "__main__":
    main()
