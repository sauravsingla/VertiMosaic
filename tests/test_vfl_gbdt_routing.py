import numpy as np
import pytest

from vertimosaic.models import VFLHistGBDT
from vertimosaic.parties import (
    ActiveParty,
    HistogramRoutingState,
    OpaqueSplitReference,
    PassiveParty,
)


def test_party_routing_is_local_and_deterministic() -> None:
    x = np.array([[0.1], [0.7], [0.2], [0.9]])
    party = PassiveParty("telecom", x)
    left, right = party.route(np.arange(4), 0, 0.5)
    assert np.array_equal(left, np.array([0, 2]))
    assert np.array_equal(right, np.array([1, 3]))


def test_histogram_candidates_hide_numeric_threshold_and_route_by_opaque_reference() -> None:
    x = np.array([[0.1], [0.2], [0.7], [0.9]])
    party = PassiveParty("telecom", x)
    gradients = np.array([-1.0, -0.5, 0.5, 1.0])
    hessians = np.ones(4)
    candidates = party.candidate_histograms(
        gradients,
        hessians,
        np.arange(4),
        max_bins=4,
        min_samples_leaf=1,
    )
    assert candidates
    candidate = candidates[0]
    assert "threshold" not in candidate
    assert "feature" not in candidate
    assert "threshold_index" not in candidate
    split_ref = candidate["split_ref"]
    assert split_ref.feature_ref == 0
    assert isinstance(split_ref.bin_ref, int)
    assert not hasattr(split_ref, "_threshold")
    assert "threshold" not in repr(split_ref).lower()
    left, right = party.route_split(np.arange(4), split_ref)
    assert len(left) + len(right) == 4
    assert set(left).isdisjoint(set(right))


def test_histogram_routing_handle_contains_no_threshold_state() -> None:
    with pytest.raises(ValueError, match="max_bins"):
        HistogramRoutingState("telecom", "state", n_features=1, max_bins=1)
    with pytest.raises(ValueError, match="feature width"):
        HistogramRoutingState("telecom", "state", n_features=-1, max_bins=2)
    with pytest.raises(ValueError, match="identifiers"):
        HistogramRoutingState("", "state", n_features=1, max_bins=2)

    handle = HistogramRoutingState("telecom", "state", n_features=1, max_bins=2)
    assert set(vars(handle)) == {"party_name", "state_ref", "n_features", "max_bins"}
    assert not any(isinstance(value, np.ndarray) for value in vars(handle).values())
    assert "threshold" not in repr(handle).lower()


def test_party_local_histogram_guards_fail_closed() -> None:
    party = PassiveParty("telecom", np.array([[0.0], [1.0], [2.0], [3.0]]))
    with pytest.raises(RuntimeError, match="bins were not prepared"):
        party.export_histogram_routing_state()
    with pytest.raises(RuntimeError, match="routing handle is unavailable"):
        party.route_split(np.arange(4), OpaqueSplitReference(0, 0))
    with pytest.raises(ValueError, match="out of range"):
        party.route(np.arange(4), 3, 0.5)

    assert (
        party.candidate_histograms(
            np.ones(4),
            np.ones(4),
            np.array([0]),
            max_bins=2,
            min_samples_leaf=1,
        )
        == []
    )
    with pytest.raises(ValueError, match="out-of-range feature"):
        party.candidate_histograms(
            np.ones(4),
            np.ones(4),
            np.arange(4),
            max_bins=2,
            min_samples_leaf=1,
            feature_indices=np.array([4]),
        )


def test_party_local_importance_and_routing_handle_guards() -> None:
    party = PassiveParty("telecom", np.column_stack([np.arange(4), np.arange(4)]))
    with pytest.raises(ValueError, match="feature reference"):
        party.aggregate_local_split_importance([(OpaqueSplitReference(3, 0), 1.0)])

    party.prepare_histogram_bins(2)
    split_ref = OpaqueSplitReference(0, 0)
    wrong_width = HistogramRoutingState("telecom", "missing", n_features=1, max_bins=2)
    with pytest.raises(ValueError, match="feature width"):
        party.route_split(np.arange(4), split_ref, wrong_width)
    wrong_party = HistogramRoutingState("insurance", "missing", n_features=2, max_bins=2)
    with pytest.raises(ValueError, match="different party"):
        party.route_split(np.arange(4), split_ref, wrong_party)


def test_evaluation_routing_uses_training_party_histogram_state() -> None:
    train_party = PassiveParty("telecom", np.array([[0.0], [1.0], [2.0], [3.0]]))
    gradients = np.array([-1.0, -0.5, 0.5, 1.0])
    hessians = np.ones(4)
    candidates = train_party.candidate_histograms(
        gradients,
        hessians,
        np.arange(4),
        max_bins=2,
        min_samples_leaf=1,
    )
    split_ref = candidates[0]["split_ref"]
    routing_state = train_party.export_histogram_routing_state()

    evaluation_party = PassiveParty(
        "telecom",
        np.array([[-100.0], [1.0], [2.0], [100.0]]),
    )
    left, right = evaluation_party.route_split(
        np.arange(4),
        split_ref,
        routing_state,
    )
    assert np.array_equal(left, np.array([0, 1]))
    assert np.array_equal(right, np.array([2, 3]))
    assert not evaluation_party.histogram_bins_ready


def test_gbdt_model_retains_only_token_routing_handles() -> None:
    rng = np.random.default_rng(29)
    rows = 80
    bank = ActiveParty(
        "bank",
        rng.normal(size=(rows, 2)),
        np.tile(np.array([0.0, 1.0]), rows // 2),
    )
    telecom = PassiveParty("telecom", rng.normal(size=(rows, 2)))
    model = VFLHistGBDT(n_estimators=1, max_depth=1, min_samples_leaf=5, seed=29)
    model.fit(bank, [telecom])
    assert model._routing_states
    for handle in model._routing_states.values():
        assert set(vars(handle)) == {"party_name", "state_ref", "n_features", "max_bins"}
        assert not any(isinstance(value, np.ndarray) for value in vars(handle).values())
        assert "threshold" not in repr(handle).lower()


def test_gbdt_models_real_protocol_payloads_through_transport_messages() -> None:
    rng = np.random.default_rng(23)
    rows = 80
    bank = ActiveParty(
        "bank",
        rng.normal(size=(rows, 2)),
        np.tile(np.array([0.0, 1.0]), rows // 2),
    )
    telecom = PassiveParty("telecom", rng.normal(size=(rows, 2)))
    model = VFLHistGBDT(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=5,
        max_bins=8,
        seed=23,
    )
    model.fit(bank, [telecom])

    event_types = {event.message_type for event in model.transport.audit_log}
    assert {
        "gradients",
        "hessians",
        "node_membership",
        "feature_subsample_refs",
        "candidate_histogram_statistics",
        "split_selection",
        "partition_routing_indices",
    } <= event_types
    candidate_events = [
        event
        for event in model.transport.audit_log
        if event.message_type == "candidate_histogram_statistics"
    ]
    assert candidate_events
    assert all(event.scalar_count > 0 for event in candidate_events)
    assert all(event.estimated_bytes > 0 for event in candidate_events)
    assert all(not hasattr(event, "payload") for event in model.transport.audit_log)
    assert all(
        not any(isinstance(value, np.ndarray) for value in vars(event).values())
        for event in model.transport.audit_log
    )
