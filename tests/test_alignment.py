# SPDX-License-Identifier: Apache-2.0
import pytest

from vertimosaic.alignment import EntityAligner


def test_pseudonymization_is_deterministic_and_salted() -> None:
    a = EntityAligner("alpha")
    b = EntityAligner("beta")
    assert a.pseudonymize("123") == a.pseudonymize("123")
    assert a.pseudonymize("123") != b.pseudonymize("123")
    assert a.pseudonymize("123") != "123"


def test_entity_aligner_requires_nonempty_salt() -> None:
    with pytest.raises(ValueError, match="salt must not be empty"):
        EntityAligner("")
