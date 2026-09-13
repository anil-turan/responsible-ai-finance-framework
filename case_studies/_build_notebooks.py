"""Generates the three case-study notebooks as real .ipynb files (not
hand-written JSON) using nbformat, then they are executed for real via
`jupyter nbconvert --execute` — see case_studies/README.md. This script is
a build tool, not part of the installable package.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT_DIR = Path(__file__).parent

COMMON_SETUP = '''\
import sys
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from responsible_ai_finance import audit

pd.set_option("display.precision", 3)
np.random.seed(42)
'''


def credit_scoring_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Case Study 1: Credit Scoring\n\n"
            "Audits a RandomForest credit-default model for fairness across a "
            "protected `region` attribute, explainability, and robustness. "
            "**Synthetic data**: no real applicant data is used anywhere in this "
            "repository. The income gap between regions is deliberately baked "
            "into the generator to simulate a historical-bias scenario a real "
            "audit would need to catch."
        ),
        nbf.v4.new_code_cell(COMMON_SETUP),
        nbf.v4.new_code_cell(
            '''\
rng = np.random.default_rng(7)
n = 1500

income = rng.normal(34000, 11000, n).clip(6000, None)
debt_ratio = rng.uniform(0, 1, n)
age = rng.integers(18, 75, n)
utilization = rng.uniform(0, 1, n)
region = rng.choice(["Region_A", "Region_B"], size=n, p=[0.55, 0.45])

# Simulated historical bias: Region_B applicants show a systematically lower
# recorded income for otherwise similar risk profiles (a common real-world
# proxy-discrimination pattern this audit should surface).
income_adj = income - np.where(region == "Region_B", 3500, 0)

logit = -0.2 - income_adj / 15000 + 2.3 * debt_ratio + 1.4 * utilization - 0.01 * age
prob_default = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, n) < prob_default).astype(int)

X = pd.DataFrame({"income": income, "debt_ratio": debt_ratio, "age": age, "utilization": utilization})
X_train, X_test, y_train, y_test, region_train, region_test = train_test_split(
    X, y, region, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)
print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")
print(f"Default rate in test set: {y_test.mean():.3f}")
'''
        ),
        nbf.v4.new_code_cell(
            '''\
report = audit(
    model,
    X_test.reset_index(drop=True),
    pd.Series(y_test).reset_index(drop=True),
    pd.Series(region_test).reset_index(drop=True),
    model_name="credit-scoring-rf",
)
print(report.to_markdown())
'''
        ),
        nbf.v4.new_markdown_cell(
            "## Key Takeaways\n\n"
            "- This notebook is executed end to end on every run — the numbers "
            "above are real outputs of this code, not hand-typed placeholders.\n"
            "- The governance flags section is the first thing a reviewer should "
            "read; the detailed tables below it exist to let them verify the flag.\n"
            "- Whether the disparate-impact flag fires depends on the specific "
            "random split and model fit — re-run the notebook to see this."
        ),
    ]
    return nb


def insurance_pricing_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Case Study 2: Insurance Pricing Risk Model\n\n"
            "Audits a claims-risk classifier used to inform premium loading, "
            "checking fairness across a synthetic `age_band` attribute — a "
            "protected characteristic under the FCA's fair pricing expectations "
            "for general insurance. Synthetic data only."
        ),
        nbf.v4.new_code_cell(COMMON_SETUP),
        nbf.v4.new_code_cell(
            '''\
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
X_train, X_test, y_train, y_test, band_train, band_test = train_test_split(
    X, y, age_band, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print(f"Claim rate in test set: {y_test.mean():.3f}")
'''
        ),
        nbf.v4.new_code_cell(
            '''\
report = audit(
    model,
    X_test.reset_index(drop=True),
    pd.Series(y_test).reset_index(drop=True),
    pd.Series(band_test).reset_index(drop=True),
    model_name="insurance-claim-risk-rf",
)
print(report.to_markdown())
'''
        ),
        nbf.v4.new_markdown_cell(
            "## Key Takeaways\n\n"
            "- `driving_experience_years` is used both as a legitimate risk "
            "feature and to derive the `age_band` protected attribute — a "
            "realistic proxy-variable tension that any fairness audit in "
            "insurance pricing has to reason about explicitly, not paper over.\n"
            "- This case study intentionally does **not** remove the proxy "
            "feature; the point of the audit is to surface the tension so a "
            "human reviewer can decide, not to silently 'fix' it."
        ),
    ]
    return nb


def aml_screening_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Case Study 3: AML Alert Screening Model\n\n"
            "Audits a transaction-alert classifier (predicting whether a flagged "
            "transaction should be escalated to a SAR) across a synthetic "
            "`customer_segment` attribute — checking that escalation rates "
            "aren't disproportionately driven by segment rather than genuine "
            "risk signal. Synthetic data only."
        ),
        nbf.v4.new_code_cell(COMMON_SETUP),
        nbf.v4.new_code_cell(
            '''\
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
X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
    X, y, customer_segment, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
print(f"Escalation rate in test set: {y_test.mean():.3f}")
'''
        ),
        nbf.v4.new_code_cell(
            '''\
report = audit(
    model,
    X_test.reset_index(drop=True),
    pd.Series(y_test).reset_index(drop=True),
    pd.Series(seg_test).reset_index(drop=True),
    model_name="aml-alert-screening-rf",
)
print(report.to_markdown())
'''
        ),
        nbf.v4.new_markdown_cell(
            "## Key Takeaways\n\n"
            "- Financial crime models are audited for fairness too: an AML "
            "model that escalates one customer segment at a materially higher "
            "rate than its true risk warrants creates both a fair-treatment "
            "problem and, ironically, alert-fatigue that can mask genuine "
            "risk elsewhere.\n"
            "- The robustness section matters operationally here: if zeroing "
            "`transaction_amount` (e.g. a missing/late-arriving field) flips a "
            "large share of predictions, that is a production data-quality "
            "dependency worth hardening before go-live."
        ),
    ]
    return nb


def main() -> None:
    notebooks = {
        "01_credit_scoring_case_study.ipynb": credit_scoring_notebook(),
        "02_insurance_pricing_case_study.ipynb": insurance_pricing_notebook(),
        "03_aml_screening_case_study.ipynb": aml_screening_notebook(),
    }
    for filename, nb in notebooks.items():
        path = OUT_DIR / filename
        nbf.write(nb, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
