# SPDX-License-Identifier: Apache-2.0
.PHONY: test coverage coverage-protocol lint type format-check build security sbom demo benchmark

test:
	pytest -q
coverage:
	pytest -q --cov=vertimosaic --cov-report=term-missing --cov-report=xml:coverage.xml --cov-report=json:coverage.json
coverage-protocol:
	python scripts/check_protocol_coverage.py --coverage coverage.json --threshold 90
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
	vertimosaic benchmark --sizes 10000,30000,50000,100000,250000 --model logistic --seed 42
