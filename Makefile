.PHONY: test lint type format-check build security sbom demo

test:
	pytest -q
lint:
	ruff check .
type:
	mypy src/vertimosaic
format-check:
	ruff format --check .
build:
	python -m build && twine check dist/*
security:
	bandit -r src && pip-audit
sbom:
	cyclonedx-py environment -o sbom.json
demo:
	vertimosaic demo --rows 2000 --seed 42
