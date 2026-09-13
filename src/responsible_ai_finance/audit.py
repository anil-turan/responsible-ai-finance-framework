"""The single public entry point: `audit(model, X, y_true, sensitive_features)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .explainability import evaluate_explainability
from .fairness import evaluate_fairness
from .report import AuditReport
from .robustness import evaluate_robustness


def audit(
    model,
    X: pd.DataFrame,
    y_true,
    sensitive_features,
    model_name: str = "model",
    top_n_features: int = 10,
    sample_size: int = 200,
) -> AuditReport:
    """Run a full fairness + explainability + robustness audit on `model`.

    Parameters
    ----------
    model : any object with `.predict` (and ideally `.predict_proba`)
    X : pd.DataFrame of features the model was scored on
    y_true : array-like of true labels (0/1)
    sensitive_features : array-like, one protected-attribute value per row
        (e.g. a gender or age-band column) — used for the fairness metrics.
    model_name : label used in the report header.
    """
    if hasattr(model, "predict_proba"):
        y_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    else:
        y_pred = np.asarray(model.predict(X))

    fairness_result = evaluate_fairness(y_true, y_pred, sensitive_features)
    explainability_result = evaluate_explainability(model, X, top_n=top_n_features, sample_size=sample_size)
    robustness_result = evaluate_robustness(model, X, sample_size=sample_size)

    return AuditReport(
        model_name=model_name,
        fairness=fairness_result,
        explainability=explainability_result,
        robustness=robustness_result,
    )
