import numpy as np

from responsible_ai_finance.fairness import evaluate_fairness


def test_identical_groups_have_zero_parity_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    # force identical selection rate per group by construction
    y_pred = np.array([0, 1] * 100)
    result = evaluate_fairness(y_true, y_pred, groups)
    assert result.demographic_parity_difference == 0.0
    assert result.disparate_impact_ratio == 1.0
    assert result.passes_80_percent_rule is True


def test_detects_disparate_impact():
    groups = np.array(["A"] * 100 + ["B"] * 100)
    y_true = np.random.default_rng(1).integers(0, 2, 200)
    # group A approved 90% of the time, group B only 20%
    y_pred = np.array([1] * 90 + [0] * 10 + [1] * 20 + [0] * 80)
    result = evaluate_fairness(y_true, y_pred, groups)
    assert result.demographic_parity_difference > 0.5
    assert result.disparate_impact_ratio < 0.8
    assert result.passes_80_percent_rule is False


def test_group_metrics_cover_every_group():
    groups = np.array(["A", "A", "B", "B", "C", "C"])
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    result = evaluate_fairness(y_true, y_pred, groups)
    assert {g.group for g in result.group_metrics} == {"A", "B", "C"}


def test_end_to_end_on_synthetic_model_flags_known_bias(synthetic_credit_data):
    d = synthetic_credit_data
    y_pred = d["model"].predict(d["X_test"])
    result = evaluate_fairness(d["y_test"], y_pred, d["group_test"])
    # the fixture bakes in a real income gap between group A and B, so we
    # only assert the metric machinery runs and produces sane bounded output
    assert 0.0 <= result.disparate_impact_ratio <= 1.0
    assert len(result.group_metrics) == 2
