"""
models/tabnet_model.py — TabNet (Arik & Pfister, 2021) wrapper.

TabNet is a sequential attention-based transformer for tabular data that
natively produces instance-level feature masks, providing a built-in
interpretability pathway complementary to post-hoc SHAP / LIME.

Reference:
  Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable
  tabular learning. AAAI-21.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
torch.set_num_threads(1)  # Also needed when restoring models without fit().

class _ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Compatibility shim for pytorch-tabnet + recent torch.

    pytorch-tabnet (<= 4.1) decides whether a scheduler needs the
    validation metric by checking ``hasattr(scheduler_fn, "is_better")``.
    Recent torch releases renamed that method to ``_is_better``, so
    TabNet called ``scheduler.step()`` without ``metrics`` and raised
    ``TypeError``. Re-exposing ``is_better`` restores the metric path.
    """

    def is_better(self, a, best):
        return self._is_better(a, best)


log = logging.getLogger(__name__)


def _import_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier
    except ImportError:
        raise ImportError(
            "Install pytorch-tabnet: pip install pytorch-tabnet"
        )


def _select_device() -> str:
    """
    Pick the fastest compute backend available, preferring CUDA, then
    Apple Silicon's MPS (Metal) backend, then CPU.

    pytorch-tabnet's own "auto" device logic only checks for CUDA, so on
    Apple Silicon Macs it silently falls back to CPU even when a GPU is
    sitting idle. MPS support for some of TabNet's ops (entmax/sparsemax
    masking) is newer and not universally stable across PyTorch versions,
    so we run a tiny smoke test before committing to it — if it fails for
    any reason, we transparently fall back to CPU instead of crashing
    mid-training.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            x = torch.ones(8, 8, device="mps")
            _ = (x @ x).sum().item()
            log.info("Apple MPS backend passed smoke test — using GPU acceleration.")
            return "mps"
        except Exception as exc:
            log.warning(
                "MPS backend available but failed smoke test (%s) — "
                "falling back to CPU.", exc,
            )

    return "cpu"


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper around pytorch-tabnet's TabNetClassifier.

    Key additions over the raw TabNet API:
      - Automatically converts DataFrames to float32 arrays.
      - Exposes `explain(X)` to retrieve per-instance feature masks
        (aggregated across all steps).
      - Implements `feature_importances_` property for SHAP TreeExplainer
        fallback compatibility.
    """

    # Keys consumed by the wrapper rather than forwarded to TabNetClassifier.
    WRAPPER_KEYS = ("cost_sensitive", "cat_idxs", "cat_dims", "nominal_indices",
                    "early_stopping_fraction")

    def __init__(self, **params):
        from config import TABNET_PARAMS
        self.params = {**TABNET_PARAMS, **params}
        self._model  = None
        self.classes_ = np.array([0, 1])

    def get_params(self, deep: bool = True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype("float32")
        return np.asarray(X, dtype="float32")

    def fit(
        self,
        X_train,
        y_train,
        X_val: Optional = None,
        y_val: Optional = None,
    ) -> "TabNetWrapper":
        TabNetClassifier = _import_tabnet()
        from config import EARLY_STOPPING_FRACTION

        kw = dict(self.params)
        cost_sensitive   = bool(kw.pop("cost_sensitive", False))
        cat_idxs         = list(kw.pop("cat_idxs", None) or kw.pop("nominal_indices", None) or [])
        kw.pop("nominal_indices", None)
        cat_dims         = list(kw.pop("cat_dims", None) or [])
        es_fraction      = float(kw.pop("early_stopping_fraction", EARLY_STOPPING_FRACTION))
        if cat_idxs and not cat_dims:
            arr_tmp = self._to_array(X_train)
            cat_dims = [int(arr_tmp[:, i].max()) + 2 for i in cat_idxs]
        max_epochs       = kw.pop("max_epochs", 200)
        patience         = kw.pop("patience", 15)
        batch_size       = kw.pop("batch_size", 1024)
        vbs              = kw.pop("virtual_batch_size", 128)
        num_workers      = kw.pop("num_workers", 0)
        drop_last        = kw.pop("drop_last", False)

        # scheduler_params needs to be a dict
        sched_params     = kw.pop("scheduler_params",
                                   dict(mode="min", patience=5,
                                        min_lr=1e-5, factor=0.9))
        opt_fn           = kw.pop("optimizer_fn", "Adam")
        opt_params       = kw.pop("optimizer_params",
                                   dict(lr=2e-2, weight_decay=1e-5))

        import torch
        # Single intra-op thread. This process also loads LightGBM/XGBoost,
        # which ship their own copy of libomp.dylib. With two OpenMP runtimes
        # loaded, torch's parallel ``randperm`` (DataLoader shuffle, used once
        # n_train exceeds torch's serial threshold, ~21k rows) forks OpenMP
        # workers and never returns from the join barrier: TabNet sat at
        # 0 epochs indefinitely on the 35k-row Lending Club split, on both
        # CPU and MPS. One thread avoids the parallel region entirely and was
        # also faster here (2.3 s/epoch vs 10 s/epoch with 8 threads).
        torch.set_num_threads(1)
        opt_map = {
            "Adam"   : torch.optim.Adam,
            "SGD"    : torch.optim.SGD,
            "AdamW"  : torch.optim.AdamW,
        }
        optimizer_fn = opt_map.get(opt_fn, torch.optim.Adam)

        device_name = kw.pop("device_name", None) or _select_device()
        log.info(
            "TabNet starting | device=%s  max_epochs=%d  patience=%d  "
            "batch_size=%d  n_train=%d",
            device_name, max_epochs, patience, batch_size, len(X_train),
        )

        X_tr = self._to_array(X_train)
        y_tr = self._to_array(y_train).astype(int).ravel()

        # Fold-local early stopping: carve a stratified slice from this fit's
        # own data when no eval set is supplied (see models.gradient_boosting).
        if X_val is None and es_fraction > 0 and len(y_tr) >= 50 and np.bincount(y_tr).min() >= 5:
            from sklearn.model_selection import train_test_split
            idx_fit, idx_es = train_test_split(
                np.arange(len(y_tr)), test_size=es_fraction, stratify=y_tr,
                random_state=int(kw.get("seed", 0)),
            )
            X_val, y_val = X_tr[idx_es], y_tr[idx_es]
            X_tr, y_tr = X_tr[idx_fit], y_tr[idx_fit]

        eval_set = []
        eval_name = []
        if X_val is not None:
            eval_set  = [(self._to_array(X_val),
                          self._to_array(y_val).astype(int).ravel())]
            eval_name = ["val"]

        import time
        t0 = time.time()

        def _build_and_fit(dev: str):
            model = TabNetClassifier(
                cat_idxs         = cat_idxs,
                cat_dims         = cat_dims,
                cat_emb_dim      = [min(8, max(1, int(round(d ** 0.5)))) for d in cat_dims] if cat_dims else 1,
                optimizer_fn     = optimizer_fn,
                optimizer_params = opt_params,
                scheduler_params = sched_params,
                scheduler_fn     = _ReduceLROnPlateau,
                device_name      = dev,
                verbose          = 1,
                **kw,
            )
            model.fit(
                X_train       = X_tr,
                y_train       = y_tr,
                eval_set      = eval_set,
                eval_name     = eval_name,
                eval_metric   = ["auc"],
                weights       = 1 if cost_sensitive else 0,
                max_epochs    = max_epochs,
                patience      = patience,
                batch_size    = batch_size,
                virtual_batch_size = vbs,
                num_workers   = num_workers,
                drop_last     = drop_last,
            )
            return model

        try:
            self._model = _build_and_fit(device_name)
        except Exception as exc:
            if device_name != "cpu":
                log.warning(
                    "TabNet training failed on device '%s' (%s: %s) — "
                    "retrying on CPU.", device_name, type(exc).__name__, exc,
                )
                device_name = "cpu"
                self._model = _build_and_fit(device_name)
            else:
                raise
        finally:
            # Release cached device memory and force a GC pass. TabNet
            # builds a fresh network + optimizer + scheduler on every fit;
            # without this, repeated in-process construction (e.g. several
            # TabNet fits run back-to-back) can accumulate fragmented
            # device memory, which manifests as progressively slower —
            # or in the worst case, stalled — subsequent fits.
            import gc
            if device_name == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif device_name == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - t0
        log.info(
            "TabNet trained | device=%s  best_epoch=%d  elapsed=%.1fs",
            device_name, self._model.best_epoch, elapsed,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(self._to_array(X))

    def predict(self, X) -> np.ndarray:
        return self._model.predict(self._to_array(X))

    # ── XAI: native attention masks ──────────────────────────────────────────

    def explain(self, X) -> Tuple[np.ndarray, list[np.ndarray]]:
        """
        Returns
        ───────
        M_agg  : (n_samples, n_features) aggregated attention masks
        M_step : list of per-step masks, each (n_samples, n_features)
        """
        arr            = self._to_array(X)
        M_agg, M_steps = self._model.explain(arr)
        return M_agg, M_steps

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Global feature importance as mean absolute attention weight across
        the training set explain call.  Available after calling explain().
        Falls back to uniform weights if explain was never called.
        """
        try:
            return self._mean_mask_
        except AttributeError:
            n_feat = self._model.input_dim
            return np.ones(n_feat) / n_feat

    def compute_global_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """Compute and cache global feature importance from attention masks."""
        M_agg, _ = self.explain(X_sample)
        self._mean_mask_ = M_agg.mean(axis=0)
        return self._mean_mask_

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path):
        path = Path(path)
        self._model.save_model(str(path))
        log.info("TabNet model saved to %s", path)

    def load(self, path: Path):
        TabNetClassifier = _import_tabnet()
        self._model = TabNetClassifier()
        self._model.load_model(str(path))
        return self


# ── Subprocess-isolated fit, for use in ablation/repeated-fit contexts ───────

def tabnet_subprocess_worker(X_train, y_train, X_val, y_val, X_test, seed, result_queue):
    """
    Fit a TabNetWrapper and return test-set predictions via a
    multiprocessing.Queue. Designed to run in its own OS process.

    Why this exists: repeatedly constructing and training TabNet in the
    *same* process (e.g. once per SMOTE strategy in an ablation loop) can
    accumulate fragmented MPS/CUDA device memory across fits, which has
    been observed in practice to manifest as a hang rather than a clean
    error. Running each fit in a brand-new process sidesteps this
    entirely — a fresh process means a fresh device context every time,
    with nothing left over from the previous fit. It also makes timeouts
    actually enforceable: the parent can kill this process at the OS
    level (SIGKILL) if it doesn't finish in time, which works even if the
    hang is inside native GPU code that would never yield control back to
    a Python-level signal handler.

    Must be a module-level function (not a method or closure) since
    macOS's multiprocessing "spawn" start method needs to import it by
    reference in the child process.

    Puts onto result_queue:
      - a 1-D numpy array of P(default) for X_test, on success
      - an Exception instance, on failure (never raised across the
        process boundary — the parent decides how to log/handle it)
    """
    try:
        model = TabNetWrapper(seed=seed)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        proba = model.predict_proba(X_test)[:, 1]
        result_queue.put(proba)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: report, don't crash
        result_queue.put(exc)


def fit_tabnet_isolated(
    X_train, y_train, X_val, y_val, X_test,
    timeout_s: int = 480,
    seed: int = 42,
):
    """
    Run tabnet_subprocess_worker() in an isolated child process with a
    hard wall-clock ceiling.

    Returns
    ───────
    np.ndarray of P(default) for X_test on success, or None if the fit
    failed or exceeded timeout_s (in which case the child is force-killed
    and the caller should treat this strategy/fit as skipped).
    """
    import multiprocessing as mp
    import queue as queue_mod

    ctx          = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target = tabnet_subprocess_worker,
        args   = (X_train, y_train, X_val, y_val, X_test, seed, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        log.warning(
            "TabNet subprocess exceeded %ds — force-killing (SIGKILL).",
            timeout_s,
        )
        proc.kill()       # SIGKILL: unconditional, works even on a true
        proc.join()       # native-code hang the process can't catch/ignore
        return None

    try:
        result = result_queue.get_nowait()
    except queue_mod.Empty:
        log.warning(
            "TabNet subprocess exited (code=%s) without returning a "
            "result — likely crashed silently.", proc.exitcode,
        )
        return None

    if isinstance(result, Exception):
        log.warning(
            "TabNet subprocess fit failed (%s: %s)",
            type(result).__name__, result,
        )
        return None

    return result
