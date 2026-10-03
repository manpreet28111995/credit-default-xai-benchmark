"""Month-disjoint forward validation with training-label availability checks."""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class ObservableTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, dates, n_splits=3, min_train_per_class=6):
        self.dates = dates
        self.n_splits = n_splits
        self.min_train_per_class = min_train_per_class

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if self.n_splits < 2 or len(X) != len(self.dates):
            raise ValueError("Temporal CV needs at least two folds and aligned dates.")
        if hasattr(X, "index") and not X.index.equals(self.dates.index):
            raise ValueError("Temporal CV dates are not aligned with feature rows.")
        origination = pd.to_datetime(self.dates["_origination"])
        observable = pd.to_datetime(self.dates["_label_observable_by"])
        if origination.isna().any() or observable.isna().any() or not origination.is_monotonic_increasing:
            raise ValueError("Temporal CV requires finite dates sorted by origination.")
        months = origination.dt.to_period("M")
        unique = pd.PeriodIndex(months.unique()).sort_values()
        y = np.asarray(y) if y is not None else None
        # Reserve the usual initial training block, extending it if labels have
        # not matured or an oversampler would have too few class neighbours.
        start = max(1, len(unique) // (self.n_splits + 1))
        while start < len(unique):
            boundary = unique[start].start_time
            train = np.flatnonzero((origination < boundary) & (observable <= boundary))
            counts = np.bincount(y[train].astype(int), minlength=2) if y is not None else [len(train)]
            if min(counts) >= self.min_train_per_class:
                break
            start += 1
        if len(unique) - start < self.n_splits:
            raise ValueError("Too few months with observable training labels for temporal CV.")
        folds = []
        for block in np.array_split(unique[start:], self.n_splits):
            boundary = block[0].start_time
            train = np.flatnonzero((origination < boundary) & (observable <= boundary))
            validation = np.flatnonzero(months.isin(block))
            if y is not None and len(np.unique(y[validation])) < 2:
                raise ValueError("A temporal validation block has fewer than two classes.")
            folds.append((train, validation))
        yield from folds
