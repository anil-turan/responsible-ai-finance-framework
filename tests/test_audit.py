from responsible_ai_finance import AuditReport, audit


def test_audit_returns_full_report(synthetic_credit_data):
    d = synthetic_credit_data
    report = audit(
        d["model"], d["X_test"], d["y_test"], d["group_test"], model_name="credit-rf-test", sample_size=80
    )
    assert isinstance(report, AuditReport)
    assert report.model_name == "credit-rf-test"
    assert report.fairness is not None
    assert report.explainability is not None
    assert report.robustness is not None


def test_report_to_dict_is_json_shaped(synthetic_credit_data):
    d = synthetic_credit_data
    report = audit(d["model"], d["X_test"], d["y_test"], d["group_test"], sample_size=80)
    payload = report.to_dict()
    assert set(payload) == {"model_name", "generated_at", "fairness", "explainability", "robustness"}


def test_report_to_markdown_contains_key_sections(synthetic_credit_data):
    d = synthetic_credit_data
    report = audit(d["model"], d["X_test"], d["y_test"], d["group_test"], sample_size=80)
    md = report.to_markdown()
    assert "# Responsible AI Audit" in md
    assert "## Fairness" in md
    assert "## Explainability" in md
    assert "## Robustness" in md
    assert "## Governance Flags" in md


def test_flags_include_disparate_impact_when_biased():
    import numpy as np

    from responsible_ai_finance.explainability import ExplainabilityResult
    from responsible_ai_finance.fairness import evaluate_fairness
    from responsible_ai_finance.report import AuditReport
    from responsible_ai_finance.robustness import RobustnessResult
    import pandas as pd

    groups = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.default_rng(1).integers(0, 2, 200)
    y_pred = np.array([1] * 90 + [0] * 10 + [1] * 20 + [0] * 80)
    fairness = evaluate_fairness(y_true, y_pred, groups)

    fake_explainability = ExplainabilityResult(
        feature_importance=pd.Series({"x": 1.0}), top_features=["x"], local_fidelity_mae=0.01
    )
    fake_robustness = RobustnessResult(noise_flip_rate=0.0, feature_dropout_flip_rate={"x": 0.0}, most_fragile_feature="x")

    report = AuditReport(
        model_name="biased-model", fairness=fairness, explainability=fake_explainability, robustness=fake_robustness
    )
    flags = report.flags()
    assert any("80%" in f or "0.80" in f for f in flags)
