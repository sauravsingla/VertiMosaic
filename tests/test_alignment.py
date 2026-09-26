from vertimosaic.alignment import intersect_ids, pseudonymize


def test_pseudonymization_is_deterministic_and_salted() -> None:
    assert pseudonymize("abc", "s1") == pseudonymize("abc", "s1")
    assert pseudonymize("abc", "s1") != pseudonymize("abc", "s2")


def test_intersection() -> None:
    assert intersect_ids({"a": {"1", "2"}, "b": {"2", "3"}}) == {"2"}
