"""Explicit privacy guarantees, non-guarantees, and empirical leakage baselines."""

from vertimosaic.privacy.attacks import (
    MembershipInferenceResult,
    ResidualLabelInferenceResult,
    RoutingExposureResult,
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)
from vertimosaic.privacy.experiments import run_privacy_audit

IMPLEMENTED_PROPERTIES = (
    "raw feature locality",
    "party-local preprocessing",
    "protocol separation",
    "pseudonymous research identifiers",
    "empirical leakage measurement baselines",
)

NOT_GUARANTEED_PROPERTIES = (
    "cryptographic confidentiality",
    "malicious-party security",
    "collusion resistance",
    "private set intersection",
    "formal differential privacy",
)

__all__ = [
    "IMPLEMENTED_PROPERTIES",
    "NOT_GUARANTEED_PROPERTIES",
    "MembershipInferenceResult",
    "ResidualLabelInferenceResult",
    "RoutingExposureResult",
    "confidence_membership_inference",
    "residual_label_inference",
    "routing_membership_exposure",
    "run_privacy_audit",
]
