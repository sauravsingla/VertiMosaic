# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from pathlib import Path

ACTION = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
FULL_SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def test_all_external_github_actions_are_pinned_to_full_commit_sha() -> None:
    workflow_root = Path(".github/workflows")
    violations: list[str] = []
    for workflow in sorted(workflow_root.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for match in ACTION.finditer(text):
            reference = match.group(1)
            if reference.startswith("./"):
                continue
            if not FULL_SHA.fullmatch(reference):
                violations.append(f"{workflow}:{reference}")
    assert not violations, "unpinned GitHub Actions: " + ", ".join(violations)
