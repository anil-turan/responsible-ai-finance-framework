from responsible_ai_finance.robustness import evaluate_robustness


def test_robustness_returns_bounded_flip_rates(synthetic_credit_data):
    d = synthetic_credit_data
    result = evaluate_robustness(d["model"], d["X_test"], sample_size=100)
    assert 0.0 <= result.noise_flip_rate <= 1.0
    assert set(result.feature_dropout_flip_rate) == set(d["X_test"].columns)
    assert all(0.0 <= v <= 1.0 for v in result.feature_dropout_flip_rate.values())
    assert result.most_fragile_feature in d["X_test"].columns


def test_zeroing_every_feature_is_evaluated(synthetic_credit_data):
    d = synthetic_credit_data
    result = evaluate_robustness(d["model"], d["X_test"], sample_size=50)
    assert len(result.feature_dropout_flip_rate) == d["X_test"].shape[1]
