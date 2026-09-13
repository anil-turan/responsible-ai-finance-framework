"""Regenerates real matplotlib figures from the three case studies.

Uses the exact same synthetic-data generators and random seeds as
`_build_notebooks.py`, so the numbers plotted here match the numbers printed
in the executed notebooks. Run after `_build_notebooks.py` + nbconvert if the
case studies ever change.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from responsible_ai_finance.explainability import evaluate_explainability
from responsible_ai_finance.fairness import evaluate_fairness
from responsible_ai_finance.robustness import evaluate_robustness

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {"bar": "#2563eb", "bar2": "#f59e0b", "flag": "#dc2626", "ok": "#16a34a"}


def _credit_scoring_data():
    rng = np.random.default_rng(7)
    n = 1500
    income = rng.normal(34000, 11000, n).clip(6000, None)
    debt_ratio = rng.uniform(0, 1, n)
    age = rng.integers(18, 75, n)
    utilization = rng.uniform(0, 1, n)
    region = rng.choice(["Region_A", "Region_B"], size=n, p=[0.55, 0.45])
    income_adj = income - np.where(region == "Region_B", 3500, 0)
    logit = -0.2 - income_adj / 15000 + 2.3 * debt_ratio + 1.4 * utilization - 0.01 * age
    prob_default = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob_default).astype(int)
    X = pd.DataFrame({"income": income, "debt_ratio": debt_ratio, "age": age, "utilization": utilization})
    return X, y, region


def _insurance_pricing_data():
    rng = np.random.default_rng(11)
    n = 1200
    driving_experience_years = rng.integers(0, 40, n)
    annual_mileage = rng.normal(9000, 3000, n).clip(500, None)
    vehicle_value = rng.normal(15000, 6000, n).clip(1000, None)
    prior_claims = rng.poisson(0.3, n)
    age_band = np.where(driving_experience_years < 5, "Under_25", "25_Plus")
    logit = -1.5 + 0.8 * prior_claims - 0.02 * driving_experience_years + annual_mileage / 20000
    prob_claim = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob_claim).astype(int)
    X = pd.DataFrame(
        {
            "driving_experience_years": driving_experience_years,
            "annual_mileage": annual_mileage,
            "vehicle_value": vehicle_value,
            "prior_claims": prior_claims,
        }
    )
    return X, y, age_band


def _aml_screening_data():
    rng = np.random.default_rng(23)
    n = 1000
    transaction_amount = rng.lognormal(mean=7, sigma=1.2, size=n)
    velocity_24h = rng.poisson(2, n)
    new_beneficiary = rng.integers(0, 2, n)
    cross_border = rng.integers(0, 2, n)
    customer_segment = rng.choice(["Retail", "SME"], size=n, p=[0.7, 0.3])
    logit = (
        -3
        + np.log1p(transaction_amount) / 4
        + 0.6 * velocity_24h
        + 1.2 * new_beneficiary
        + 1.0 * cross_border
    )
    prob_escalate = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob_escalate).astype(int)
    X = pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "velocity_24h": velocity_24h,
            "new_beneficiary": new_beneficiary,
            "cross_border": cross_border,
        }
    )
    return X, y, customer_segment


CASE_STUDIES = {
    "credit_scoring": (_credit_scoring_data, "Credit Scoring", "region"),
    "insurance_pricing": (_insurance_pricing_data, "Insurance Pricing", "age band"),
    "aml_screening": (_aml_screening_data, "AML Screening", "customer segment"),
}


def make_figure(key: str, data_fn, title: str, group_label: str) -> Path:
    X, y, group = data_fn()
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, group, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    X_test = X_test.reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    g_test = pd.Series(g_test).reset_index(drop=True)
    y_pred = model.predict(X_test)

    fairness = evaluate_fairness(y_test, y_pred, g_test)
    explain = evaluate_explainability(model, X_test, sample_size=150)
    robust = evaluate_robustness(model, X_test, sample_size=150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"{title} — Responsible AI Audit", fontsize=13, fontweight="bold")

    # 1. Fairness: selection rate by group
    ax = axes[0]
    groups = [g.group for g in fairness.group_metrics]
    rates = [g.selection_rate for g in fairness.group_metrics]
    bars = ax.bar(groups, rates, color=COLORS["bar"])
    ax.axhline(
        max(rates) * 0.8 if max(rates) > 0 else 0,
        color=COLORS["flag"],
        linestyle="--",
        linewidth=1,
        label="80% rule threshold",
    )
    ax.set_title(f"Selection rate by {group_label}")
    ax.set_ylabel("Predicted positive rate")
    ax.legend(fontsize=8)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    # 2. Explainability: top SHAP features
    ax = axes[1]
    top = explain.feature_importance.head(6).sort_values()
    ax.barh(top.index, top.values, color=COLORS["bar2"])
    ax.set_title("SHAP feature importance")
    ax.set_xlabel("mean |SHAP value|")

    # 3. Robustness: feature dropout flip rate
    ax = axes[2]
    items = sorted(robust.feature_dropout_flip_rate.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLORS["flag"] if v > 0.10 else COLORS["ok"] for v in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0.10, color=COLORS["flag"], linestyle="--", linewidth=1, label="10% flag threshold")
    ax.set_title("Dropout flip rate by feature")
    ax.set_xlabel("fraction of predictions flipped")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIG_DIR / f"{key}_audit.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


def main() -> None:
    for key, (fn, title, label) in CASE_STUDIES.items():
        make_figure(key, fn, title, label)


if __name__ == "__main__":
    main()
