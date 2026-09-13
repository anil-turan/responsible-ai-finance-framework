"""Explainability audit: global feature importance via SHAP, plus a simple
local-fidelity check comparing SHAP's additive explanation against the
model's actual output (a lightweight sanity check in the spirit of LIME's
local-fidelity idea, without adding a second heavy dependency).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass
class ExplainabilityResult:
    feature_importance: pd.Series  # mean |SHAP value| per feature, descending
    top_features: list[str]
    local_fidelity_mae: float  # mean abs error between SHAP reconstruction and model output

    def to_dict(self) -> dict:
        return {
            "feature_importance": self.feature_importance.to_dict(),
            "top_features": self.top_features,
            "local_fidelity_mae": self.local_fidelity_mae,
        }


def _get_predict_fn(model):
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]
    return model.predict


def evaluate_explainability(model, X: pd.DataFrame, top_n: int = 10, sample_size: int = 200) -> ExplainabilityResult:
    """Compute SHAP-based global feature importance for `model` on `X`.

    Uses `shap.Explainer`, which auto-selects TreeExplainer for tree models
    and falls back to a model-agnostic permutation explainer otherwise. `X`
    is subsampled to `sample_size` rows for tractability on larger datasets.
    """
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    predict_fn = _get_predict_fn(model)
    explainer = shap.Explainer(predict_fn, X_sample)
    shap_values = explainer(X_sample)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:  # some explainers return (n_samples, n_features, n_outputs)
        values = values[:, :, -1]

    importance = pd.Series(np.abs(values).mean(axis=0), index=X_sample.columns).sort_values(ascending=False)
    top_features = importance.head(top_n).index.tolist()

    base_values = np.asarray(shap_values.base_values).reshape(-1)
    if base_values.size == 1:
        base_values = np.full(len(values), base_values.item())
    reconstructed = values.sum(axis=1) + base_values
    actual = np.asarray(predict_fn(X_sample)).reshape(-1)
    local_fidelity_mae = float(np.mean(np.abs(reconstructed - actual)))

    return ExplainabilityResult(
        feature_importance=importance, top_features=top_features, local_fidelity_mae=local_fidelity_mae
    )
