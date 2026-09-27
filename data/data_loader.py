"""
data/data_loader.py — Dataset acquisition, cleaning, and feature engineering.

Supports:
  - UCI Default of Credit Card Clients (Taiwan, 2005)
  - Lending Club loan data (subset)
  - Corrected South German Credit (UCI 573)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

log = logging.getLogger(__name__)


# ── UCI helper ────────────────────────────────────────────────────────────────

UCI_DTYPES: Dict[str, str] = {
    "LIMIT_BAL"  : "float32",
    "SEX"        : "int8",
    "EDUCATION"  : "int8",
    "MARRIAGE"   : "int8",
    "AGE"        : "int8",
    **{f"PAY_{i}" : "int8"  for i in [0, 2, 3, 4, 5, 6]},
    **{f"BILL_AMT{i}": "float32" for i in range(1, 7)},
    **{f"PAY_AMT{i}" : "float32" for i in range(1, 7)},
    "default_payment_next_month": "int8",
}

SOUTH_GERMAN_URL = (
    "https://archive.ics.uci.edu/static/public/573/"
    "south+german+credit+update.zip"
)
SOUTH_GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "amount", "savings",
    "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property", "age",
    "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_risk",
]


def _download_uci(cache_path: Path) -> pd.DataFrame:
    """Download or load cached UCI dataset."""
    from config import UCI_URL

    if cache_path.exists():
        log.info("Loading cached UCI dataset from %s", cache_path)
        return pd.read_csv(cache_path)

    log.info("Downloading UCI dataset …")
    try:
        r = requests.get(UCI_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), header=1)
        df.to_csv(cache_path, index=False)
        log.info("Saved to %s", cache_path)
        return df
    except Exception as exc:
        raise RuntimeError(
            "UCI Default of Credit Card Clients data are unavailable. "
            "Provide the verified source file at "
            f"{cache_path} or restore network access to {UCI_URL}. "
            "Synthetic fallback data are disabled to prevent accidental reporting "
            "of simulated observations as benchmark results."
        ) from exc



# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    winsor_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    return_winsor_bounds: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Domain-driven feature engineering for credit risk.

    New features
    ────────────
    util_rate_mean      Mean credit utilisation over 6 months
    pay_ratio_mean      Mean (payment / bill) ratio — proxy for repayment habit
    avg_pay_delay       Average repayment delay across 6 months
    max_pay_delay       Worst-case repayment delay
    bill_trend          Slope of bill amounts (debt accumulation)
    pay_trend           Slope of payment amounts
    limit_ratio         LIMIT_BAL normalised by mean bill
    delay_spike         1 if recent delay > historical mean
    """
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_stat_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    required_cols = ["LIMIT_BAL", *bill_cols, *pay_amt_cols, *pay_stat_cols]

    if all(col in df.columns for col in required_cols):
        bills = df[bill_cols].values.astype("float32")
        pays  = df[pay_amt_cols].values.astype("float32")
        stats = df[pay_stat_cols].values.astype("float32")

        # Utilisation
        avg_bill = bills.mean(axis=1)
        df["util_rate_mean"] = (avg_bill / df["LIMIT_BAL"].clip(lower=1)).astype("float32")

        # Repayment ratio — guard division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bills > 0, pays / bills, 0.0)
        df["pay_ratio_mean"] = ratio.mean(axis=1).astype("float32")

        # Delay statistics
        delay_pos = np.clip(stats, 0, 8)           # negative = paid early → 0
        df["avg_pay_delay"] = delay_pos.mean(axis=1).astype("float32")
        df["max_pay_delay"] = delay_pos.max(axis=1).astype("float32")

        # Trend features (positive = rising = bad)
        time_idx = np.arange(6, dtype="float32")
        def _slope(arr: np.ndarray) -> np.ndarray:
            """Vectorised OLS slope across 6 time steps."""
            t  = time_idx - time_idx.mean()
            v  = arr - arr.mean(axis=1, keepdims=True)
            return (v * t).sum(axis=1) / (t ** 2).sum()

        df["bill_trend"] = _slope(bills).astype("float32")
        df["pay_trend"]  = _slope(pays).astype("float32")

        # Limit ratio
        df["limit_ratio"] = (df["LIMIT_BAL"] / avg_bill.clip(1)).astype("float32")

        # Recency spike: recent delay > rolling mean of older months
        recent_delay = delay_pos[:, 0]
        hist_delay   = delay_pos[:, 1:].mean(axis=1)
        df["delay_spike"] = (recent_delay > hist_delay).astype("int8")
    else:
        log.info("Skipping UCI-specific engineered features; required columns not present.")

    # Clean extreme outliers (winsorise at 1st / 99th pct)
    num_cols = df.select_dtypes("number").columns.tolist()
    target   = "default_payment_next_month"
    if target in num_cols:
        num_cols.remove(target)

    fitted_bounds = dict(winsor_bounds or {})
    for col in num_cols:
        if col not in fitted_bounds:
            lo, hi = df[col].quantile([0.01, 0.99])
            fitted_bounds[col] = (float(lo), float(hi))
        else:
            lo, hi = fitted_bounds[col]
        df[col] = df[col].clip(lo, hi)

    if return_winsor_bounds:
        return df, fitted_bounds
    return df


def clean_uci(df: pd.DataFrame) -> pd.DataFrame:
    """UCI-specific cleaning steps (undocumented category levels, etc.)."""
    # Drop ID column if present
    df = df.drop(columns=["ID"], errors="ignore")
    # Rename target column for consistency
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default_payment_next_month"})

    # Remap undocumented EDUCATION values 0, 5, 6 → 4 (Other)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Remap undocumented MARRIAGE value 0 → 3 (Other)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df.astype({k: v for k, v in UCI_DTYPES.items() if k in df.columns})


def _load_south_german(data_dir: Path) -> pd.DataFrame:
    """Load the corrected South German Credit data, downloading it if needed."""
    candidates = [
        data_dir / "south_german_credit.csv",
        data_dir / "german_credit.csv",
        data_dir / "SouthGermanCredit.asc",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        try:
            log.info("South German data not found locally; downloading from %s", SOUTH_GERMAN_URL)
            response = requests.get(SOUTH_GERMAN_URL, timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                member = next(
                    name for name in archive.namelist()
                    if name.lower().endswith("southgermancredit.asc")
                )
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "SouthGermanCredit.asc").write_bytes(archive.read(member))
                source = data_dir / "SouthGermanCredit.asc"
            log.info("Saved South German data to %s", source)
        except Exception as exc:
            raise FileNotFoundError(
                "South German Credit data are unavailable. Place "
                "south_german_credit.csv or SouthGermanCredit.asc in "
                f"{data_dir}, or allow the UCI download ({SOUTH_GERMAN_URL})."
            ) from exc

    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        # The UCI archive includes a whitespace-separated header row.
        frame = pd.read_csv(source, sep=r"\s+", header=0)
    if frame.shape[1] != len(SOUTH_GERMAN_COLUMNS):
        raise ValueError(
            f"Expected 21 South German Credit columns, found {frame.shape[1]} in {source}."
        )
    frame.columns = SOUTH_GERMAN_COLUMNS

    raw_target = frame.pop("credit_risk")
    if raw_target.dtype == object:
        target = raw_target.astype(str).str.lower().isin({"bad", "default", "1"})
    else:
        values = pd.to_numeric(raw_target)
        unique = set(values.dropna().unique())
        if unique.issubset({1, 2}):
            target = values.eq(2)
        elif unique.issubset({0, 1}):
            # South German/UCI coding uses good=1 and bad=0 in common exports.
            target = values.eq(0)
        else:
            raise ValueError(f"Unsupported South German target coding: {sorted(unique)}")
    frame["default_payment_next_month"] = target.astype("int8")
    return frame


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(
    dataset: str = "uci",
    data_dir: Path | str = "data",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    engineer: bool = True,
    split_mode: str = "random",
    temporal_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,   pd.Series,   pd.Series,
           list[str],   StandardScaler]:
    """
    Returns
    ───────
    X_train, X_val, X_test : DataFrames
    y_train, y_val, y_test : Series
    feature_names          : list[str]
    scaler                 : fitted StandardScaler
    """
    data_dir = Path(data_dir)

    if dataset == "uci":
        raw = _download_uci(data_dir / "uci_credit.csv")
        df  = clean_uci(raw)
    elif dataset == "lending_club":
        df = _load_lending_club(data_dir)
    elif dataset in {"south_german", "german"}:
        df = _load_south_german(data_dir)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'uci', 'lending_club', or 'south_german'."
        )

    from config import TARGET_COL

    if split_mode not in {"random", "temporal"}:
        raise ValueError("split_mode must be 'random' or 'temporal'.")
    if split_mode == "temporal":
        if not temporal_column or temporal_column not in df.columns:
            raise ValueError(
                "Temporal validation requires a temporal_column present in the dataset."
            )
        order = pd.to_datetime(df[temporal_column], errors="coerce")
        if order.notna().all():
            order = order.astype("int64")
        else:
            order = pd.to_numeric(df[temporal_column], errors="coerce")
        if order.isna().any():
            raise ValueError(f"Temporal column '{temporal_column}' contains unparseable values.")
        ordered = df.assign(_split_order=order).sort_values("_split_order")
        ordered = ordered.drop(columns=["_split_order"])
        n_test = max(1, int(round(len(ordered) * test_size)))
        n_val = max(1, int(round(len(ordered) * val_size)))
        df_train = ordered.iloc[:len(ordered) - n_test - n_val].copy()
        df_val = ordered.iloc[len(ordered) - n_test - n_val:len(ordered) - n_test].copy()
        df_test = ordered.iloc[len(ordered) - n_test:].copy()
        # The ordering field is a split key, not a predictor.
        df_train = df_train.drop(columns=[temporal_column])
        df_val = df_val.drop(columns=[temporal_column])
        df_test = df_test.drop(columns=[temporal_column])
    else:
        # --- Train / Temp split
        df_train, df_test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state,
        )
        # --- Train / Val split (from train)
        val_frac = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_frac, stratify=df_train[TARGET_COL], random_state=random_state,
        )

    if engineer:
        df_train, winsor_bounds = engineer_features(df_train, return_winsor_bounds=True)
        df_val = engineer_features(df_val, winsor_bounds=winsor_bounds)
        df_test = engineer_features(df_test, winsor_bounds=winsor_bounds)

    X_train = df_train.drop(columns=[TARGET_COL])
    X_val   = df_val.drop(columns=[TARGET_COL])
    X_test  = df_test.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)
    y_val   = df_val[TARGET_COL].astype(int)
    y_test  = df_test[TARGET_COL].astype(int)

    # --- Scaling (fit on train only)
    num_cols = X_train.select_dtypes("number").columns.tolist()
    scaler   = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols]   = scaler.transform(X_val[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    feature_names = X_train.columns.tolist()

    log.info(
        "Dataset loaded | train=%d  val=%d  test=%d | "
        "default rate  train=%.2f%%  test=%.2f%%",
        len(y_train), len(y_val), len(y_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler


# ── Lending Club stub ─────────────────────────────────────────────────────────

def _load_lending_club(data_dir: Path) -> pd.DataFrame:
    """
    Load the packaged 50K sample or the full Kaggle Lending Club export.

    The checked-in sample is preferred so the temporal experiment is
    reproducible without the much larger raw Kaggle file.  The raw ``loan.csv``
    path remains supported for users who want to run on the full export.
    """
    sample_path = data_dir / "lending_club_subsample_50k_by_issue_date.csv"
    if sample_path.exists():
        log.info("Loading packaged Lending Club 50K sample from %s", sample_path)
        df = pd.read_csv(sample_path, low_memory=False)
        target = "default_payment_next_month"
        if target not in df.columns:
            raise ValueError(
                f"Packaged Lending Club sample is missing '{target}': {sample_path}"
            )

        date_column = "issue_date" if "issue_date" in df.columns else "issue_d"
        if "issue_time" not in df.columns:
            if date_column not in df.columns:
                raise ValueError(
                    "Packaged Lending Club sample must contain 'issue_date' or 'issue_d'."
                )
            date_format = "%b-%Y" if date_column == "issue_d" else None
            issue_date = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
            if issue_date.isna().any():
                raise ValueError(
                    f"Packaged Lending Club sample contains invalid values in '{date_column}'."
                )
            df["issue_time"] = issue_date.astype("int64") / 10**9

        for col in ["int_rate", "revol_util"]:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.rstrip("%").astype("float32")

        df = df.drop(
            columns=["loan_csv_row", "issue_date", "issue_d", "loan_status"],
            errors="ignore",
        )
        return df.dropna()

    path = data_dir / "loan.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Lending Club CSV not found. Download 'loan.csv' from Kaggle and "
            f"place it at {path}."
        )

    usecols = [
        "loan_amnt", "funded_amnt", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
        "fico_range_high", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "issue_d",
        "loan_status",
    ]
    available = set(pd.read_csv(path, nrows=0).columns)
    required = [column for column in usecols if column != "issue_d"]
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Lending Club file is missing required columns: {missing}")
    df = pd.read_csv(path, usecols=[column for column in usecols if column in available], low_memory=False)

    # Binary target: 1 = Charged Off / Default
    charged_off = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
    df["default_payment_next_month"] = df["loan_status"].isin(charged_off).astype(int)
    df = df.drop(columns=["loan_status"])

    # Parse percentage columns
    for col in ["int_rate", "revol_util"]:
        if df[col].dtype == object:
            df[col] = df[col].str.rstrip("%").astype("float32")

    if "issue_d" in df.columns:
        issue_date = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["issue_time"] = issue_date.astype("int64") / 10**9
        df = df.drop(columns=["issue_d"])

    df = df.dropna()
    return df
