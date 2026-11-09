"""Step 0.1: the `credit_risk` package is installed and importable."""

import importlib


def test_credit_risk_package_imports() -> None:
    module = importlib.import_module("credit_risk")
    assert module.__name__ == "credit_risk"
