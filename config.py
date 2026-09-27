"""
config.py — Centralised configuration for the XAI Credit Default Experiment.

Paper: Credit Default Prediction Using Explainable AI (XAI) on Imbalanced
       Financial Data
Target: Discover AI (Springer)
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "results" / "outputs"
FIG_DIR     = OUTPUTS_DIR / "figures"
RES_DIR     = OUTPUTS_DIR / "results"
MDL_DIR     = OUTPUTS_DIR / "models"
RUNS_DIR    = OUTPUTS_DIR / "runs"

for d in [DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET = "uci"            # "uci" | "lending_club" | "south_german"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
TARGET_COL   = "default_payment_next_month"
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10        # fraction of train set used for validation
RANDOM_STATE = 42
DEFAULT_SEEDS = [42, 99, 123, 326, 456, 515, 689, 777, 872, 999]
RANDOM_SEARCH_N_ITER = 12
RANDOM_SEARCH_CV_FOLDS = 3
RANDOM_SEARCH_SCORING = "roc_auc"

# ── Imbalance Handling ────────────────────────────────────────────────────────
SMOTE_STRATEGIES = {
    "SMOTE"            : dict(random_state=RANDOM_STATE, k_neighbors=5),
    "BorderlineSMOTE"  : dict(random_state=RANDOM_STATE, k_neighbors=5, kind="borderline-1"),
    "SVMSMOTE"         : dict(random_state=RANDOM_STATE, k_neighbors=5),
    "ADASYN"           : dict(random_state=RANDOM_STATE, n_neighbors=5),
    "None"             : None,    # baseline: no resampling
}

# ── Model Hyper-parameters ────────────────────────────────────────────────────
XGBOOST_PARAMS = dict(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    eval_metric       = "auc",
    early_stopping_rounds = 30,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
)

LGBM_PARAMS = dict(
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    num_leaves         = 63,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    min_child_samples  = 20,
    random_state       = RANDOM_STATE,
    n_jobs             = -1,
    verbosity          = -1,
)

CATBOOST_PARAMS = dict(
    iterations         = 500,
    depth              = 6,
    learning_rate      = 0.05,
    l2_leaf_reg        = 3,
    random_seed        = RANDOM_STATE,
    verbose            = 0,
    eval_metric        = "AUC",
    early_stopping_rounds = 30,
)

TABNET_PARAMS = dict(
    n_d             = 64,
    n_a             = 64,
    n_steps         = 5,
    gamma           = 1.5,
    n_independent   = 2,
    n_shared        = 2,
    momentum        = 0.02,
    mask_type       = "entmax",
    optimizer_fn    = "Adam",
    optimizer_params= dict(lr=2e-2, weight_decay=1e-5),
    scheduler_params= dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
    max_epochs      = 200,
    patience        = 15,
    batch_size      = 1024,
    virtual_batch_size = 128,
    num_workers     = 0,
    drop_last       = False,
    seed            = RANDOM_STATE,
)

# ── Hyper-parameter Search Spaces ────────────────────────────────────────────
RANDOM_SEARCH_SPACES = {
    "XGBoost": {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.05, 0.1, 0.2],
        "reg_alpha": [0, 0.05, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "LightGBM": {
        "n_estimators": [300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.05, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_samples": [10, 20, 40, 80],
    },
    "CatBoost": {
        "iterations": [300, 500, 800],
        "depth": [4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7, 10],
    },
    "Random Forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 6, 10, 14, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "Logistic Reg.": {
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
}

# ── Explainability ────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 500   # K-means background for TreeExplainer/DeepExplainer
LIME_NUM_FEATURES       = 15
LIME_NUM_SAMPLES        = 1000
LIME_TOP_LABELS         = 1

# Number of local instances to explain (selected: TP, FP, FN, TN)
N_LOCAL_EXPLANATIONS = 4

# ── Evaluation ───────────────────────────────────────────────────────────────
CV_FOLDS        = 5
SCORING_METRICS = [
    "roc_auc", "average_precision", "f1", "recall", "precision",
]
THRESHOLD_RANGE = (0.1, 0.9, 0.05)   # start, stop, step for threshold sweep

# ── Plotting ─────────────────────────────────────────────────────────────────
FIGURE_DPI  = 300
FIGURE_EXT  = "pdf"   # "pdf" for IEEE submission; "png" for preview
PALETTE     = {
    "XGBoost"        : "#1f77b4",
    "LightGBM"       : "#ff7f0e",
    "CatBoost"       : "#2ca02c",
    "TabNet"         : "#d62728",
    "Random Forest"  : "#9467bd",
    "Logistic Reg."  : "#8c564b",
}
