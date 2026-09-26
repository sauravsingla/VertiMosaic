import pandas as pd

from vertimosaic.linkage import CopulaLinker


def test_target_blind_linkage_reproducible() -> None:
    anchor = pd.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(20)],
            "x": range(20),
            "target": [i % 2 for i in range(20)],
        }
    )
    donor = pd.DataFrame({"z": range(7), "w": range(7, 14)})
    linker = CopulaLinker(correlation=0.25, seed=4)
    a, ma = linker.link(anchor, {"telecom": donor}, target_column="target")
    b, mb = linker.link(anchor, {"telecom": donor}, target_column="target")
    pd.testing.assert_frame_equal(a["telecom"], b["telecom"])
    assert ma == mb
    assert ma.target_blind
    assert a["telecom"]["entity_id"].tolist() == anchor["entity_id"].tolist()
