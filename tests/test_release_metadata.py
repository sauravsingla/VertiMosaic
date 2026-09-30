from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_release_metadata() -> tuple[dict, dict, dict]:
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return zenodo, citation, pyproject


def test_zenodo_metadata_matches_release_metadata() -> None:
    zenodo, citation, pyproject = _load_release_metadata()
    project = pyproject["project"]

    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"
    assert zenodo["version"] == citation["version"] == project["version"]
    assert zenodo["title"] == citation["title"]
    assert zenodo["license"] == citation["license"] == project["license"]["text"]
    assert zenodo["creators"]
    assert any(
        item.get("identifier") == "https://github.com/sauravsingla/VertiMosaic"
        for item in zenodo["related_identifiers"]
    )


def test_archival_doi_and_urls_are_consistent() -> None:
    _, citation, pyproject = _load_release_metadata()
    doi = "10.5281/zenodo.23021923"
    record_url = "https://zenodo.org/records/23021923"
    project_urls = pyproject["project"]["urls"]

    assert citation["doi"] == doi
    assert citation["url"] == record_url
    assert project_urls["Archive"] == record_url
    assert project_urls["DOI"] == f"https://doi.org/{doi}"
