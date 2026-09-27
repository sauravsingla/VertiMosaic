from vertimosaic.models import VFLHistGBDT


def test_split_gain_prefers_purer_partition() -> None:
    model = VFLHistGBDT(l2_leaf_reg=1.0)
    pure = {"g_left": -5.0, "h_left": 2.0, "g_right": 5.0, "h_right": 2.0}
    weak = {"g_left": -1.0, "h_left": 2.0, "g_right": 1.0, "h_right": 2.0}
    assert model._split_gain(pure) > model._split_gain(weak)
