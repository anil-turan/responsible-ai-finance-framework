"""responsible-ai-finance — a lightweight fairness, explainability and
robustness audit framework for ML models used in financial services.

    from responsible_ai_finance import audit
    report = audit(model, X_test, y_test, sensitive_features=X_test["gender"])
    print(report.to_markdown())
"""
from .audit import audit
from .report import AuditReport

__version__ = "0.1.0"
__all__ = ["audit", "AuditReport", "__version__"]
