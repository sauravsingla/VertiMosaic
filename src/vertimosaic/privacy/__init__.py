"""Explicit privacy guarantees, non-guarantees, and empirical leakage research."""

from vertimosaic.privacy.attacks import (
    MembershipInferenceResult,
    ResidualLabelInferenceResult,
    RoutingExposureResult,
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)
from vertimosaic.privacy.experiments import run_privacy_audit
from vertimosaic.privacy.research import (
    PrivacyResearchConfig,
    privacy_research_summary_rows,
    run_privacy_research,
)

IMPLEMENTED_PROPERTIES = (
    "raw feature locality",
    "party-local preprocessing",
    "protocol separation",
    "pseudonymous research identifiers",
    "empirical leakage measurement baselines",
    "multi-seed privacy/utility research sweeps",
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
    "PrivacyResearchConfig",
    "ResidualLabelInferenceResult",
    "RoutingExposureResult",
    "confidence_membership_inference",
    "privacy_research_summary_rows",
    "residual_label_inference",
    "routing_membership_exposure",
    "run_privacy_audit",
    "run_privacy_research",
]
