"""Explicit privacy guarantees, non-guarantees, and empirical leakage research."""

from vertimosaic.privacy.advanced_attacks import (
    MessageFeatureReconstructionResult,
    gradient_label_inference,
    message_feature_reconstruction,
    privacy_attack_scenarios,
)
from vertimosaic.privacy.attacks import (
    MembershipInferenceResult,
    ResidualLabelInferenceResult,
    RoutingExposureResult,
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)
from vertimosaic.privacy.backends import (
    ClippedGaussianDPBackend,
    GaussianDPBackend,
    GaussianZCDPAccountant,
    PrivacyBackendRegistry,
)
from vertimosaic.privacy.crypto import (
    AdditiveSecretSharingSum,
    OpenMinedPSIBackend,
    PairwiseMaskSecureAggregation,
    PaillierHomomorphicSum,
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
    "empirical membership, label, gradient, routing and feature-reconstruction leakage baselines",
    "multi-seed privacy/utility research sweeps",
    "optional Gaussian release mechanism with zCDP accounting",
    "optional sensitivity-enforcing clipped Gaussian release mechanism",
    "reference pairwise-mask secure aggregation for integer sums",
    "reference additive secret-sharing sum primitive",
    "optional PSI and Paillier adapters when privacy-crypto dependencies are installed",
)

NOT_GUARANTEED_PROPERTIES = (
    "cryptographic confidentiality for default VFL protocols",
    "malicious-party security",
    "collusion resistance",
    "dropout-resilient secure aggregation",
    "general-purpose MPC",
    "homomorphically encrypted end-to-end VFL training",
    "end-to-end formal differential privacy by default",
)

__all__ = [
    "IMPLEMENTED_PROPERTIES",
    "NOT_GUARANTEED_PROPERTIES",
    "AdditiveSecretSharingSum",
    "ClippedGaussianDPBackend",
    "GaussianDPBackend",
    "GaussianZCDPAccountant",
    "MembershipInferenceResult",
    "MessageFeatureReconstructionResult",
    "OpenMinedPSIBackend",
    "PairwiseMaskSecureAggregation",
    "PaillierHomomorphicSum",
    "PrivacyBackendRegistry",
    "PrivacyResearchConfig",
    "ResidualLabelInferenceResult",
    "RoutingExposureResult",
    "confidence_membership_inference",
    "gradient_label_inference",
    "message_feature_reconstruction",
    "privacy_attack_scenarios",
    "privacy_research_summary_rows",
    "residual_label_inference",
    "routing_membership_exposure",
    "run_privacy_audit",
    "run_privacy_research",
]
