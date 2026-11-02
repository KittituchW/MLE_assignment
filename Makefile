.PHONY: test lint run serve smoke load

test:
	pytest --cov=reco --cov-report=term-missing

lint:
	ruff check .
	mypy src/reco

run:
	python -m reco run-all

serve:
	python -m reco serve

smoke:
	python scripts/smoke.py

load:
	python scripts/loadtest.py
