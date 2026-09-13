from responsible_ai_finance.explainability import evaluate_explainability


def test_explainability_returns_all_features_ranked(synthetic_credit_data):
    d = synthetic_credit_data
    result = evaluate_explainability(d["model"], d["X_test"], sample_size=60)
    assert set(result.feature_importance.index) == set(d["X_test"].columns)
    assert list(result.feature_importance) == sorted(result.feature_importance, reverse=True)
    assert len(result.top_features) <= 10


def test_local_fidelity_is_reasonably_small(synthetic_credit_data):
    d = synthetic_credit_data
    result = evaluate_explainability(d["model"], d["X_test"], sample_size=60)
    # SHAP's additive decomposition should reconstruct the model's own output
    # closely for a tree ensemble — this is a sanity check, not a tight bound.
    assert result.local_fidelity_mae < 0.2
