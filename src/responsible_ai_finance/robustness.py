"""Robustness checks: how much do predictions move under small, realistic
perturbations that a production model will actually see (noisy inputs,
a missing/zeroed feature)? These are deliberately simple, interpretable
stress tests rather than adversarial-example generation — the audience is
a model risk / governance reviewer, not a security researcher.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RobustnessResult:
    noise_flip_rate: float  # fraction of predictions that change class under small Gaussian noise
    feature_dropout_flip_rate: dict[str, float]  # per-feature: flip rate when that feature is zeroed
    most_fragile_feature: str | None

    def to_dict(self) -> dict:
        return {
            "noise_flip_rate": self.noise_flip_rate,
            "feature_dropout_flip_rate": self.feature_dropout_flip_rate,
            "most_fragile_feature": self.most_fragile_feature,
        }


def _get_predict_fn(model):
    if hasattr(model, "predict_proba"):
        return lambda X: (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    return model.predict


def evaluate_robustness(
    model,
    X: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    noise_std_fraction: float = 0.05,
    sample_size: int = 300,
    random_state: int = 42,
) -> RobustnessResult:
    """Stress-test `model` on `X` with two perturbations:

    1. Gaussian noise added to each numeric column, scaled to
       `noise_std_fraction` of that column's own std dev.
    2. Zeroing out one feature at a time (simulating a missing value that
       slipped past upstream validation).

    Reports the fraction of predictions whose predicted class flips.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    numeric_columns = numeric_columns or X_sample.select_dtypes(include=[np.number]).columns.tolist()
    predict_fn = _get_predict_fn(model)

    baseline = predict_fn(X_sample)

    noisy = X_sample.copy()
    for col in numeric_columns:
        std = X_sample[col].std()
        noise = rng.normal(0, std * noise_std_fraction, size=len(X_sample))
        noisy[col] = X_sample[col] + noise
    noisy_preds = predict_fn(noisy)
    noise_flip_rate = float(np.mean(noisy_preds != baseline))

    dropout_flip_rates: dict[str, float] = {}
    for col in numeric_columns:
        dropped = X_sample.copy()
        dropped[col] = 0
        dropped_preds = predict_fn(dropped)
        dropout_flip_rates[col] = float(np.mean(dropped_preds != baseline))

    most_fragile = max(dropout_flip_rates, key=dropout_flip_rates.get) if dropout_flip_rates else None

    return RobustnessResult(
        noise_flip_rate=noise_flip_rate,
        feature_dropout_flip_rate=dropout_flip_rates,
        most_fragile_feature=most_fragile,
    )
