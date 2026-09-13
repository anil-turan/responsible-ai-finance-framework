"""Group fairness metrics, implemented from the standard definitions rather
than wrapping a third-party fairness library, so the package stays
dependency-light (numpy + pandas only) and every formula is auditable in
~100 lines.

References: Barocas, Hardt & Narayanan, "Fairness and Machine Learning"
(fairmlbook.org); Feldman et al. 2015 (disparate impact / 80% rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GroupMetrics:
    group: str
    n: int
    selection_rate: float
    tpr: float | None = None
    fpr: float | None = None


@dataclass
class FairnessResult:
    demographic_parity_difference: float
    demographic_parity_ratio: float
    equalized_odds_tpr_difference: float | None
    equalized_odds_fpr_difference: float | None
    disparate_impact_ratio: float
    passes_80_percent_rule: bool
    group_metrics: list[GroupMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "demographic_parity_difference": self.demographic_parity_difference,
            "demographic_parity_ratio": self.demographic_parity_ratio,
            "equalized_odds_tpr_difference": self.equalized_odds_tpr_difference,
            "equalized_odds_fpr_difference": self.equalized_odds_fpr_difference,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "passes_80_percent_rule": self.passes_80_percent_rule,
            "group_metrics": [g.__dict__ for g in self.group_metrics],
        }


def _selection_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred)) if len(y_pred) else float("nan")


def _true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    positives = y_true == 1
    if positives.sum() == 0:
        return None
    return float(np.mean(y_pred[positives] == 1))


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    negatives = y_true == 0
    if negatives.sum() == 0:
        return None
    return float(np.mean(y_pred[negatives] == 1))


def evaluate_fairness(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: np.ndarray | pd.Series,
) -> FairnessResult:
    """Compute demographic parity, equalized odds and disparate impact across
    the groups defined by `sensitive_features`.

    All three metrics are computed directly from predictions/labels — no
    assumption about which group is "privileged"; disparate impact ratio is
    reported as min(group selection rate) / max(group selection rate), so a
    ratio below 0.8 flags a potential 80%-rule violation regardless of which
    group ends up on which side.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = pd.Series(np.asarray(sensitive_features), name="group")

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.values})

    group_metrics: list[GroupMetrics] = []
    selection_rates: dict[str, float] = {}
    tprs: dict[str, float] = {}
    fprs: dict[str, float] = {}

    for group_value, sub in df.groupby("group"):
        sr = _selection_rate(sub["y_pred"].to_numpy())
        tpr = _true_positive_rate(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        fpr = _false_positive_rate(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        group_metrics.append(
            GroupMetrics(group=str(group_value), n=len(sub), selection_rate=sr, tpr=tpr, fpr=fpr)
        )
        selection_rates[group_value] = sr
        if tpr is not None:
            tprs[group_value] = tpr
        if fpr is not None:
            fprs[group_value] = fpr

    rates = list(selection_rates.values())
    dp_diff = max(rates) - min(rates)
    # if every group has a 0% selection rate there is, by definition, no
    # disparity between them — report ratio 1.0 rather than an undefined 0/0
    dp_ratio = (min(rates) / max(rates)) if max(rates) > 0 else 1.0

    eo_tpr_diff = (max(tprs.values()) - min(tprs.values())) if len(tprs) >= 2 else None
    eo_fpr_diff = (max(fprs.values()) - min(fprs.values())) if len(fprs) >= 2 else None

    disparate_impact_ratio = dp_ratio
    passes_80_percent_rule = bool(disparate_impact_ratio >= 0.8) if not np.isnan(disparate_impact_ratio) else False

    return FairnessResult(
        demographic_parity_difference=dp_diff,
        demographic_parity_ratio=dp_ratio,
        equalized_odds_tpr_difference=eo_tpr_diff,
        equalized_odds_fpr_difference=eo_fpr_diff,
        disparate_impact_ratio=disparate_impact_ratio,
        passes_80_percent_rule=passes_80_percent_rule,
        group_metrics=group_metrics,
    )
