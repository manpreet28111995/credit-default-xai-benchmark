"""Small, atomic checkpoints and descriptive ranks for incremental reruns."""

import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path

import pandas as pd

PROTOCOL = "v3_training_cv_oof_selection"
BOOTSTRAP_SCOPE = "conditional_on_fitted_models_and_selected_thresholds"


def training_key(args, data):
    """Exclude reporting options; include training inputs, code and runtime."""
    fields = ("dataset", "split_mode", "seed", "fast", "tune", "tune_iter",
              "inner_folds", "inner_temporal_folds", "strategy_selection")
    options = {field: getattr(args, field) for field in fields}
    options.update(cutoff=data["cutoff"], prep_spec=data["prep_spec"],
                   cat_dims=data["cat_dims"])
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for frame in (data["X_train"], data["X_test"], data["y_train"], data["y_test"], data.get("train_dates")):
        if frame is not None:
            digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
            digest.update(str(getattr(frame, "dtypes", frame.dtype if isinstance(frame, pd.Series) else "")).encode())
            digest.update(str(getattr(frame, "columns", [])).encode())
    root = Path(__file__).parent
    files = [root / "pipeline.py", root / "config.py", root / "data/data_loader.py",
             root / "evaluation/metrics.py", root / "evaluation/temporal_cv.py", root / "rerun_support.py"]
    files += sorted((root / "models").glob("*.py")) + sorted((root / "preprocessing").glob("*.py"))
    for path in files:
        digest.update(path.read_bytes())
    for name in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "scipy",
                 "xgboost", "lightgbm", "catboost", "torch", "pytorch-tabnet"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "absent"
        digest.update(f"{name}={version}".encode())
    return digest.hexdigest()


def save_checkpoint(path, key, value):
    """Only a complete atomic checkpoint can be resumed after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump({"key": key, "value": value}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, key):
    """Load only this project's trusted local checkpoints; reject stale inputs."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return payload["value"] if payload.get("key") == key else None


def descriptive_ranks(metrics):
    """Repeated splits are descriptive blocks, not independent datasets."""
    output = {"scope": "descriptive_only; runs share observations and training data", "metrics": {}}
    for metric, higher in (("AUROC", True), ("AUPRC", True), ("F1", True), ("MCC", True), ("Brier", False)):
        if metric not in metrics or metrics.empty:
            continue
        index = "cutoff" if "cutoff" in metrics and metrics["cutoff"].notna().all() else "run_id"
        # Seeds for one temporal cutoff share the test window: average first.
        wide = metrics.pivot_table(index=index, columns="model", values=metric).dropna()
        output["metrics"][metric] = {
            "n_blocks": len(wide), "block": index,
            "average_rank": wide.rank(axis=1, ascending=not higher).mean().sort_values().to_dict(),
        }
    return output
