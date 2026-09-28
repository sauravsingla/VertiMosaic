"""Explicit privacy guarantees, non-guarantees, and empirical leakage research."""

from vertimosaic.privacy.attacks import (
    MembershipInferenceResult,
    ResidualLabelInferenceResult,
    RoutingExposureResult,
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)
from vertimosaic.privacy.backends import (
    GaussianDPBackend,
    GaussianZCDPAccountant,
    PrivacyBackendRegistry,
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
    "optional Gaussian release mechanism with zCDP accounting",
)

NOT_GUARANTEED_PROPERTIES = (
    "cryptographic confidentiality",
    "malicious-party security",
    "collusion resistance",
    "private set intersection",
    "end-to-end formal differential privacy by default",
)

__all__ = [
    "IMPLEMENTED_PROPERTIES",
    "NOT_GUARANTEED_PROPERTIES",
    "GaussianDPBackend",
    "GaussianZCDPAccountant",
    "MembershipInferenceResult",
    "PrivacyBackendRegistry",
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
