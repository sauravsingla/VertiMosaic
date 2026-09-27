# SPDX-License-Identifier: Apache-2.0
import io
import json
import logging

import numpy as np
import pytest

from vertimosaic.audit.logging import UnsafeLogFieldError, log_event


def _logger(stream: io.StringIO) -> logging.Logger:
    logger = logging.Logger("vertimosaic-test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def test_structured_log_emits_scalar_metadata_as_json() -> None:
    stream = io.StringIO()
    logger = _logger(stream)
    log_event(logger, "experiment_completed", mode="synthetic_scale", seed=42, rows=2000)
    payload = json.loads(stream.getvalue())
    assert payload == {
        "event": "experiment_completed",
        "mode": "synthetic_scale",
        "rows": 2000,
        "seed": 42,
    }


def test_structured_log_rejects_row_level_array_payload() -> None:
    stream = io.StringIO()
    logger = _logger(stream)
    with pytest.raises(UnsafeLogFieldError, match="row-level values are forbidden"):
        log_event(logger, "unsafe", raw_features=np.array([[1.0, 2.0]]))
    assert stream.getvalue() == ""


def test_structured_log_rejects_nested_structures() -> None:
    stream = io.StringIO()
    logger = _logger(stream)
    with pytest.raises(UnsafeLogFieldError):
        log_event(logger, "unsafe", private_row={"feature": 1.0})
    assert stream.getvalue() == ""


def test_structured_log_rejects_empty_event_and_invalid_field_name() -> None:
    stream = io.StringIO()
    logger = _logger(stream)
    with pytest.raises(UnsafeLogFieldError, match="event name"):
        log_event(logger, "")
    with pytest.raises(UnsafeLogFieldError, match="field names"):
        log_event(logger, "unsafe", **{"": 1})
    assert stream.getvalue() == ""
