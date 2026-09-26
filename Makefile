.PHONY: install test lint typecheck security build demo sbom all
install:
	python -m pip install -e ".[data,dev]"
test:
	pytest -q --cov=vertimosaic --cov-report=term-missing
lint:
	ruff check .
	ruff format --check .
typecheck:
	mypy src/vertimosaic
security:
	bandit -r src -q
	pip-audit
build:
	python -m build
	twine check dist/*
demo:
	vertimosaic demo --rows 2000 --seed 42
sbom:
	cyclonedx-py environment -o sbom.json
all: lint typecheck test build
