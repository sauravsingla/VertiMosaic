import pytest

from vertimosaic.config import ExperimentConfig


def test_config_digest_is_deterministic() -> None:
    assert ExperimentConfig(seed=7).digest() == ExperimentConfig(seed=7).digest()


def test_config_rejects_invalid_split() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(train_fraction=0.8, validation_fraction=0.15, test_fraction=0.15)
