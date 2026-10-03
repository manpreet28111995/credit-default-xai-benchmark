"""
data/data_loader.py — Dataset acquisition, cleaning, labelling, and partitioning.

Supports:
  - UCI Default of Credit Card Clients (Taiwan, 2005)            random split
  - Corrected South German Credit (UCI 573)                       random split
  - Lending Club packaged 50K sample                              random split only
  - Prosper Loan Data (Kaggle ``prosperLoanData.csv``)            random or
    rolling-origin out-of-time split with a fixed-horizon default label

Protocol v3 conventions
-----------------------
* ``load_dataset`` returns a :class:`LoadedData` record with a train and a test
  partition only. There is no held-out validation set: every data-dependent
  choice is made on out-of-fold predictions inside the training partition
  (see ``pipeline.py``).
* Nominal columns (``config.NOMINAL_COLUMNS``) are integer-coded with train-set
  categories; unseen/missing levels map to an extra "unknown" code. They are
  never scaled, winsorised, or treated as continuous downstream.
* Continuous columns are returned raw (NaNs kept). Imputation, missing
  indicators, winsorisation and standardisation live in
  ``preprocessing.tabular_preprocessor.TabularPreprocessor``, the first step of
  every model pipeline, so they are re-fitted inside each CV fold.
* Grouping variables used for fairness diagnostics are returned separately as
  raw categories (``groups_train`` / ``groups_test``).
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

TARGET_COL = "default_payment_next_month"
ORIGINATION_COL = "_origination"          # temporal key (datetime), dropped from X
OBSERVABLE_COL = "_label_observable_by"   # date by which the horizon label is known


class LoadedData(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    prep_spec: dict            # kwargs for TabularPreprocessor
    nominal_columns: list
    groups_train: pd.DataFrame
    groups_test: pd.DataFrame
    meta: dict
    train_dates: Optional[pd.DataFrame] = None  # split metadata, never a predictor


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
    TARGET_COL: "int8",
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
    skip_columns: Optional[list] = None,
    winsorise: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Domain-driven feature engineering for the UCI credit-card data, followed
    by winsorisation of continuous columns at the 1st/99th percentile.

    Winsor bounds are fitted on the frame passed in (the training partition)
    and re-applied to other partitions via ``winsor_bounds``. Columns listed in
    ``skip_columns`` (nominal codes, target, split keys) are left untouched.

    New features (UCI only)
    ───────────────────────
    util_rate_mean, pay_ratio_mean, avg_pay_delay, max_pay_delay,
    bill_trend, pay_trend, limit_ratio, delay_spike
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

        avg_bill = bills.mean(axis=1)
        df["util_rate_mean"] = (avg_bill / df["LIMIT_BAL"].clip(lower=1)).astype("float32")

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bills > 0, pays / bills, 0.0)
        df["pay_ratio_mean"] = ratio.mean(axis=1).astype("float32")

        delay_pos = np.clip(stats, 0, 8)
        df["avg_pay_delay"] = delay_pos.mean(axis=1).astype("float32")
        df["max_pay_delay"] = delay_pos.max(axis=1).astype("float32")

        time_idx = np.arange(6, dtype="float32")

        def _slope(arr: np.ndarray) -> np.ndarray:
            t = time_idx - time_idx.mean()
            v = arr - arr.mean(axis=1, keepdims=True)
            return (v * t).sum(axis=1) / (t ** 2).sum()

        df["bill_trend"] = _slope(bills).astype("float32")
        df["pay_trend"]  = _slope(pays).astype("float32")
        df["limit_ratio"] = (df["LIMIT_BAL"] / avg_bill.clip(1)).astype("float32")

        recent_delay = delay_pos[:, 0]
        hist_delay   = delay_pos[:, 1:].mean(axis=1)
        df["delay_spike"] = (recent_delay > hist_delay).astype("int8")
    else:
        log.info("Skipping UCI-specific engineered features; required columns not present.")

    skip = set(skip_columns or []) | {TARGET_COL, ORIGINATION_COL, OBSERVABLE_COL}
    num_cols = [c for c in df.select_dtypes("number").columns if c not in skip] if winsorise else []

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
    df = df.drop(columns=["ID"], errors="ignore")
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": TARGET_COL})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
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
            # South German coding: good = 1, bad = 0.
            target = values.eq(0)
        else:
            raise ValueError(f"Unsupported South German target coding: {sorted(unique)}")
    frame[TARGET_COL] = target.astype("int8")
    return frame


# ── Lending Club (packaged sample; random split only) ─────────────────────────

def _load_lending_club(data_dir: Path) -> pd.DataFrame:
    """
    Load the packaged 50K Lending Club sample.

    The sample keeps only resolved loans (Fully Paid / Charged Off) and carries
    no ``term`` or payment dates, so a fixed-horizon, censoring-aware default
    label cannot be built from it. It is therefore supported for random splits
    only; use ``prosper`` for out-of-time evaluation.
    """
    sample_path = data_dir / "lending_club_subsample_50k_by_issue_date.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Packaged Lending Club sample not found at {sample_path}."
        )
    log.info("Loading packaged Lending Club 50K sample from %s", sample_path)
    df = pd.read_csv(sample_path, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Packaged Lending Club sample is missing '{TARGET_COL}'.")
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip("%").astype("float32")
    df = df.drop(
        columns=["loan_csv_row", "issue_date", "issue_d", "loan_status", "issue_time"],
        errors="ignore",
    )
    return df.dropna()


# ── Prosper (Kaggle prosperLoanData.csv) ──────────────────────────────────────

PROSPER_DEFAULT_STATUSES = {"Defaulted", "Chargedoff"}
PROSPER_RESOLVED_GOOD_STATUSES = {"Completed", "FinalPaymentInProgress"}


def build_prosper_horizon_label(
    df: pd.DataFrame,
    horizon_months: int,
    snapshot: Optional[pd.Timestamp] = None,
) -> Tuple[pd.Series, pd.Timestamp]:
    """
    Fixed-horizon default label with explicit censoring.

    Returns a Series with values {1, 0, -1}:
      1  : loan entered Defaulted/Chargedoff status within ``horizon_months``
           of origination (``LoanFirstDefaultedCycleNumber`` <= H; falls back
           to months between origination and ``ClosedDate`` when the cycle is
           missing).
      0  : loan is known not to have defaulted within H months, either because
           it was fully repaid (Completed / FinalPaymentInProgress) or because
           it had been observed for at least H months at the snapshot date
           without a default inside the horizon (this includes loans that
           defaulted *after* H months).
     -1  : censored — still open at the snapshot with fewer than H months of
           history, or Cancelled. These rows are excluded from modelling.

    The snapshot is the latest origination date in the file unless given.
    """
    origination = pd.to_datetime(df["LoanOriginationDate"], format="ISO8601")
    closed_all = pd.to_datetime(df["ClosedDate"], format="ISO8601", errors="coerce")
    snapshot = snapshot or max(origination.max(), closed_all.max())
    status = df["LoanStatus"].astype(str)
    months_observed = (snapshot - origination).dt.days / 30.4375

    closed = pd.to_datetime(df["ClosedDate"], format="ISO8601", errors="coerce")
    months_to_close = (closed - origination).dt.days / 30.4375
    first_default_cycle = pd.to_numeric(df["LoanFirstDefaultedCycleNumber"], errors="coerce")
    months_to_default = first_default_cycle.where(first_default_cycle.notna(), months_to_close)

    is_default_status = status.isin(PROSPER_DEFAULT_STATUSES)
    # Defaults with an unknown or impossible default time cannot be placed
    # relative to the horizon; they stay censored rather than becoming negatives.
    unknown_default_time = is_default_status & (months_to_default.isna() | (months_to_default < 0))
    positive = is_default_status & (months_to_default <= horizon_months) & ~unknown_default_time
    resolved_good = status.isin(PROSPER_RESOLVED_GOOD_STATUSES)
    observed_full_horizon = months_observed >= horizon_months
    negative = (~positive & ~unknown_default_time & (resolved_good | observed_full_horizon)
                & ~status.eq("Cancelled"))

    label = pd.Series(-1, index=df.index, dtype="int8")
    label[negative] = 0
    label[positive] = 1
    return label, snapshot


def _load_prosper(
    data_dir: Path,
    horizon_months: int,
    include_pricing: bool = False,
) -> pd.DataFrame:
    """
    Load Prosper loans with a censoring-aware horizon label and
    origination-time-only features.
    """
    from config import PROSPER_FILE, PROSPER_EXCLUDED_COLUMNS, PROSPER_PRICING_COLUMNS

    path = data_dir / PROSPER_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Prosper data not found at {path}. Download 'prosperLoanData.csv' from "
            "Kaggle (Prosper Loan Data) and place it in the data directory."
        )
    log.info("Loading Prosper loans from %s", path)
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)
    # The Kaggle export contains exact duplicate loan rows.
    df = df.drop_duplicates(subset=["LoanKey"], keep="first").reset_index(drop=True)
    log.info("Prosper: %d rows, %d after LoanKey de-duplication", n_raw, len(df))

    label, snapshot = build_prosper_horizon_label(df, horizon_months)
    origination = pd.to_datetime(df["LoanOriginationDate"], format="ISO8601")
    keep = label.ge(0)
    log.info(
        "Prosper horizon label (H=%d months, snapshot %s): %d labelled, %d censored, "
        "default rate %.2f%%",
        horizon_months, snapshot.date(), int(keep.sum()), int((~keep).sum()),
        100 * label[keep].mean(),
    )
    df = df.loc[keep].copy()
    df.attrs["snapshot"] = str(snapshot.date())
    df.attrs["horizon_months"] = int(horizon_months)
    df[TARGET_COL] = label[keep].astype("int8")
    df[ORIGINATION_COL] = origination[keep]
    df[OBSERVABLE_COL] = origination[keep] + pd.DateOffset(months=horizon_months)

    # Credit-history length at origination replaces the raw date.
    first_line = pd.to_datetime(df["FirstRecordedCreditLine"], format="ISO8601", errors="coerce")
    df["CreditHistoryYears"] = ((df[ORIGINATION_COL] - first_line).dt.days / 365.25).astype("float32")
    df = df.drop(columns=["FirstRecordedCreditLine", "LoanOriginationDate"])

    # Prior-Prosper-loan fields are missing when the borrower had no prior loan.
    for col in [
        "TotalProsperLoans", "TotalProsperPaymentsBilled", "OnTimeProsperPayments",
        "ProsperPaymentsLessThanOneMonthLate", "ProsperPaymentsOneMonthPlusLate",
        "ProsperPrincipalBorrowed", "ProsperPrincipalOutstanding",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df = df.drop(columns=["ScorexChangeAtTimeOfListing"], errors="ignore")  # 83% missing

    df = df.rename(columns={
        "ListingCategory (numeric)": "ListingCategory",
        "TradesNeverDelinquent (percentage)": "TradesNeverDelinquentPct",
    })
    for col in ["IsBorrowerHomeowner", "CurrentlyInGroup", "IncomeVerifiable"]:
        df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype("int8")
    income_order = {
        "Not employed": 0, "$0": 0, "$1-24,999": 1, "$25,000-49,999": 2,
        "$50,000-74,999": 3, "$75,000-99,999": 4, "$100,000+": 5,
    }
    # Ordinal income band; "Not displayed" becomes missing and is median-imputed.
    df["IncomeRange"] = df["IncomeRange"].map(income_order).astype("float32")
    for col in ["BorrowerState", "Occupation", "EmploymentStatus"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    drop = [c for c in PROSPER_EXCLUDED_COLUMNS if c in df.columns]
    if not include_pricing:
        drop += [c for c in PROSPER_PRICING_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)
    return df


# ── Encoding / imputation helpers ─────────────────────────────────────────────

def _encode_nominals(
    frames: list[pd.DataFrame], nominal_columns: list[str]
) -> Tuple[list[pd.DataFrame], Dict[str, list]]:
    """Integer-code nominal columns using training categories; unknown -> len(categories)."""
    train = frames[0]
    categories: Dict[str, list] = {}
    for col in nominal_columns:
        cats = sorted(train[col].dropna().unique().tolist(), key=str)
        categories[col] = cats
    encoded = []
    for frame in frames:
        frame = frame.copy()
        for col in nominal_columns:
            cats = categories[col]
            mapping = {value: code for code, value in enumerate(cats)}
            frame[col] = frame[col].map(mapping).fillna(len(cats)).astype("int32")
        encoded.append(frame)
    return encoded, categories


def _group_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Raw grouping categories for fairness diagnostics (not model inputs)."""
    from config import GROUP_COLUMNS, SOUTH_GERMAN_YOUNG_AGE_CUTOFF

    groups = pd.DataFrame(index=df.index)
    for col in GROUP_COLUMNS.get(dataset, []):
        if col == "age_group" and "age" in df.columns:
            groups[col] = np.where(df["age"] <= SOUTH_GERMAN_YOUNG_AGE_CUTOFF, 1, 2).astype("int8")
        elif col in df.columns:
            groups[col] = df[col].to_numpy()
    return groups


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(
    dataset: str = "uci",
    data_dir: Path | str = "data",
    test_size: float = 0.20,
    random_state: int = 42,
    engineer: bool = True,
    split_mode: str = "random",
    cutoff: Optional[str] = None,
    test_window_months: Optional[int] = None,
    horizon_months: Optional[int] = None,
    prosper_include_pricing: bool = False,
) -> LoadedData:
    """
    Load, label, partition, and preprocess a dataset.

    split_mode="random"   : stratified train/test split (``test_size``) seeded by
                            ``random_state``.
    split_mode="temporal" : rolling-origin block. Training rows are loans whose
                            horizon label is fully observable at ``cutoff``
                            (origination + H <= cutoff); test rows are loans
                            originated in [cutoff, cutoff + test_window_months).
                            Requires a dataset with origination dates (prosper).
    """
    from config import (
        NOMINAL_COLUMNS, PROSPER_HORIZON_MONTHS, PROSPER_TEST_WINDOW_MONTHS,
    )

    data_dir = Path(data_dir)
    horizon_months = horizon_months or PROSPER_HORIZON_MONTHS
    test_window_months = test_window_months or PROSPER_TEST_WINDOW_MONTHS

    if dataset == "uci":
        df = clean_uci(_download_uci(data_dir / "uci_credit.csv"))
    elif dataset == "lending_club":
        df = _load_lending_club(data_dir)
    elif dataset in {"south_german", "german"}:
        df = _load_south_german(data_dir)
        dataset = "south_german"
    elif dataset == "prosper":
        df = _load_prosper(data_dir, horizon_months, include_pricing=prosper_include_pricing)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose 'uci', 'lending_club', 'south_german', or 'prosper'."
        )

    nominal_columns = [c for c in NOMINAL_COLUMNS.get(dataset, []) if c in df.columns]
    meta: dict = {"dataset": dataset, "split_mode": split_mode}

    if split_mode not in {"random", "temporal"}:
        raise ValueError("split_mode must be 'random' or 'temporal'.")

    if split_mode == "temporal":
        if ORIGINATION_COL not in df.columns:
            raise ValueError(
                f"Dataset '{dataset}' has no origination dates with a censoring-aware "
                "label; temporal evaluation is supported for 'prosper' only."
            )
        if not cutoff:
            raise ValueError("Temporal evaluation requires a cutoff (YYYY-MM).")
        cutoff_ts = pd.Timestamp(cutoff)
        test_end = cutoff_ts + pd.DateOffset(months=test_window_months)
        df_train = df[df[OBSERVABLE_COL] <= cutoff_ts].sort_values(ORIGINATION_COL, kind="stable").copy()
        df_test = df[(df[ORIGINATION_COL] >= cutoff_ts) & (df[ORIGINATION_COL] < test_end)].copy()
        if df_train.empty or df_test.empty:
            raise ValueError(
                f"Temporal cutoff {cutoff} yields train={len(df_train)} test={len(df_test)} rows."
            )
        snapshot = pd.Timestamp(df.attrs.get("snapshot")) if df.attrs.get("snapshot") else None
        if snapshot is not None and test_end + pd.DateOffset(months=horizon_months) > snapshot:
            raise ValueError(
                f"Test window {cutoff}..{test_end.date()} is not fully observable at the data "
                f"snapshot ({snapshot.date()}) for a {horizon_months}-month horizon; the surviving "
                "rows would be a censoring-biased sample. Choose an earlier cutoff or a shorter horizon."
            )
        meta.update({
            "cutoff": cutoff, "test_window_months": test_window_months,
            "horizon_months": horizon_months,
            "train_origination_range": [str(df_train[ORIGINATION_COL].min().date()),
                                        str(df_train[ORIGINATION_COL].max().date())],
            "test_origination_range": [str(df_test[ORIGINATION_COL].min().date()),
                                       str(df_test[ORIGINATION_COL].max().date())],
            "embargo_months": horizon_months,
            "label_snapshot": df.attrs.get("snapshot"),
        })
    else:
        df_train, df_test = train_test_split(
            df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state,
        )

    groups_train = _group_frame(df_train, dataset)
    groups_test = _group_frame(df_test, dataset)
    train_dates = (df_train[[ORIGINATION_COL, OBSERVABLE_COL]].copy()
                   if split_mode == "temporal" else None)

    for frame in (df_train, df_test):
        frame.drop(columns=[ORIGINATION_COL, OBSERVABLE_COL], errors="ignore", inplace=True)

    (df_train, df_test), categories = _encode_nominals([df_train, df_test], nominal_columns)

    if engineer:
        # Row-wise feature construction only; winsorisation is fold-local (TabularPreprocessor).
        df_train = engineer_features(df_train, skip_columns=nominal_columns, winsorise=False)
        df_test = engineer_features(df_test, skip_columns=nominal_columns, winsorise=False)

    X_train = df_train.drop(columns=[TARGET_COL])
    X_test = df_test.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)
    y_test = df_test[TARGET_COL].astype(int)

    continuous = [c for c in X_train.columns if c not in nominal_columns]
    for c in continuous:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype("float64")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype("float64")
    # Indicator set is fixed from the training partition so the preprocessor's
    # output layout (and hence nominal positions) is identical in every fold.
    indicator_cols = [c for c in continuous if X_train[c].isna().any()]

    ordered = continuous + nominal_columns
    X_train = X_train[ordered]
    X_test = X_test[ordered]
    prep_spec = {"continuous_columns": continuous, "indicator_columns": indicator_cols,
                 "nominal_columns": nominal_columns}
    feature_names = continuous + [f"{c}_missing" for c in indicator_cols] + nominal_columns

    meta.update({
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "default_rate_train": float(y_train.mean()), "default_rate_test": float(y_test.mean()),
        "nominal_columns": nominal_columns,
        "nominal_categories": {k: [str(v) for v in vals] for k, vals in categories.items()},
        "missing_indicator_columns": indicator_cols,
        "group_columns": groups_train.columns.tolist(),
    })

    log.info(
        "Dataset loaded | %s | train=%d  test=%d | default rate train=%.2f%% test=%.2f%% | "
        "%d nominal, %d continuous",
        dataset, len(y_train), len(y_test), 100 * y_train.mean(), 100 * y_test.mean(),
        len(nominal_columns), len(continuous),
    )
    return LoadedData(
        X_train, X_test, y_train, y_test, feature_names, prep_spec,
        nominal_columns, groups_train, groups_test, meta, train_dates,
    )
