"""Small stdlib-only smoke check for the manuscript-revision metrics."""

import numpy as np
import pandas as pd

from evaluation.revision_analysis import (
    explanation_metrics,
    explanation_parsimony,
    fairness_utility_curve,
    perturb_test_set,
    reliability_score,
    reliability_constrained_selection,
    holm_adjust,
)


def main() -> None:
    rng = np.random.default_rng(7)
    X_train = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "SEX"])
    X_test = pd.DataFrame(rng.normal(size=(8, 3)), columns=X_train.columns)
    assert perturb_test_set(X_train, X_test, "missingness", 0.2, 1).shape == X_test.shape
    assert perturb_test_set(X_train, X_test, "correlated_missingness", 0.2, 1).shape == X_test.shape
    assert perturb_test_set(X_train, X_test, "covariate_shift", 0.2, 1).shape == X_test.shape
    assert 0.0 <= explanation_parsimony(np.array([[1.0, 2.0, 0.0]]))["gini_mean"] <= 1.0
    assert len(explanation_metrics(
        np.array([[1.0, -2.0, 0.1]]),
        [pd.Series({"a <= 1": 0.4, "b > 0": -0.2})],
        list(X_train.columns),
    )) == 1
    assert len(fairness_utility_curve(
        [0, 1, 0, 1], np.array([0.1, 0.8, 0.6, 0.7]),
        pd.DataFrame({"SEX": [1, 1, 2, 2]}), [0.5],
    )) == 1
    assert np.isnan(reliability_score(0.8, 0.9, 0.7)["reliability_score"])
    assert 0.0 <= reliability_score(0.8, 0.9, 0.7, 0.8)["reliability_score"] <= 1.0
    selected = reliability_constrained_selection(pd.DataFrame([
        {"model": "a", "AUROC": 0.80, "Brier": 0.14, "base_rate_brier": 0.17,
         "robustness_retention": 0.95, "max_fairness_gap": 0.05},
        {"model": "b", "AUROC": 0.82, "Brier": 0.15, "base_rate_brier": 0.17,
         "robustness_retention": 0.80, "max_fairness_gap": 0.02},
    ]))
    assert selected.iloc[0]["model"] == "a"
    assert np.all((holm_adjust([0.01, 0.04, 0.2]) >= 0) & (holm_adjust([0.01, 0.04, 0.2]) <= 1))
    print("revision-analysis self-check: OK")


if __name__ == "__main__":
    main()
