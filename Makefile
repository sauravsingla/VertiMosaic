.PHONY: test coverage lint type format-check build security sbom demo benchmark

test:
	pytest -q
coverage:
	pytest -q --cov=vertimosaic --cov-report=term-missing
lint:
	ruff check .
type:
	mypy src/vertimosaic
format-check:
	ruff format --check .
build:
	python -m build && twine check dist/*
security:
	bandit -r src -q && pip-audit
sbom:
	cyclonedx-py environment --output-file sbom.json
demo:
	vertimosaic demo --rows 2000 --seed 42
benchmark:
	vertimosaic benchmark --sizes 10000,30000,50000,100000 --model logistic --seed 42
