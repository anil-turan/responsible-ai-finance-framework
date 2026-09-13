import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def synthetic_credit_data():
    """A small synthetic credit-scoring dataset with a built-in fairness gap:
    group 'B' has a systematically lower approval rate than group 'A', so
    fairness tests have something real to detect rather than asserting on
    near-zero noise."""
    rng = np.random.default_rng(42)
    n = 800

    income = rng.normal(35000, 12000, n).clip(5000, None)
    debt_ratio = rng.uniform(0, 1, n)
    age = rng.integers(18, 75, n)
    utilization = rng.uniform(0, 1, n)
    group = rng.choice(["A", "B"], size=n, p=[0.5, 0.5])

    # true default risk depends on income/debt/utilization, plus a group-correlated
    # income gap baked into the synthetic generator (not the label rule itself)
    # to simulate real-world historical bias entering via a proxy feature.
    group_bias = np.where(group == "B", -4000, 0)
    income_biased = income + group_bias

    logit = -0.2 + (-income_biased / 15000) + 2.5 * debt_ratio + 1.5 * utilization - 0.01 * age
    prob_default = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob_default).astype(int)

    X = pd.DataFrame(
        {"income": income, "debt_ratio": debt_ratio, "age": age, "utilization": utilization}
    )

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, y, group, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "X_test": X_test.reset_index(drop=True),
        "y_test": pd.Series(y_test).reset_index(drop=True),
        "group_test": pd.Series(group_test).reset_index(drop=True),
    }
