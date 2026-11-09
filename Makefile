# Thin wrappers around `uv run python -m credit_risk <stage>`.
# On Windows without make, run the uv commands directly.

.PHONY: setup test test-all lint api app data features train evaluate explain fairness monitor report final export

setup:
	uv sync

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

api:
	uv run uvicorn api.main:app --reload

app:
	uv run streamlit run app/streamlit_app.py

data:
	uv run python -m credit_risk data

features:
	uv run python -m credit_risk features

train:
	uv run python -m credit_risk train

evaluate:
	uv run python -m credit_risk evaluate

explain:
	uv run python -m credit_risk explain

fairness:
	uv run python -m credit_risk fairness

monitor:
	uv run python -m credit_risk monitor

report:
	uv run python -m credit_risk report

# ONE-TIME out-of-time test run. Ask Kouki before running.
final:
	uv run python -m credit_risk final

export:
	uv run python -m credit_risk export
