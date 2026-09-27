from vertimosaic.experiments.contribution import exact_shapley_party_utility


def test_exact_shapley_utility_sums_to_total_gain() -> None:
    weights = {"bank": 1.0, "telecom": 2.0, "insurance": 3.0, "retail": 4.0}

    def score(subset: tuple[str, ...]) -> float:
        return sum(weights[name] for name in subset)

    parties = tuple(weights)
    utility = exact_shapley_party_utility(parties, score)
    assert utility == weights
