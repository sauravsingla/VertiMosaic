import numpy as np
import pandas as pd

from vertimosaic.preprocessing import LocalTabularPreprocessor


def test_local_preprocessor_handles_unknown_category() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "cat": ["a", "b", "a", None]})
    test = pd.DataFrame({"x": [5.0], "cat": ["unseen"]})
    prep = LocalTabularPreprocessor(["x"], ["cat"], scaling="standard")
    transformed = prep.fit(train).transform(test)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
