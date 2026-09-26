from vertimosaic.alignment import EntityAligner


def test_pseudonymization_is_deterministic_and_salted() -> None:
    a = EntityAligner("alpha"); b = EntityAligner("beta")
    assert a.pseudonymize("123") == a.pseudonymize("123")
    assert a.pseudonymize("123") != b.pseudonymize("123")
    assert a.pseudonymize("123") != "123"
