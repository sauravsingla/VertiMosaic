"""Explicit privacy guarantees and non-guarantees for the research simulator."""

IMPLEMENTED_PROPERTIES = (
    "raw feature locality",
    "party-local preprocessing",
    "protocol separation",
    "pseudonymous research identifiers",
)

NOT_GUARANTEED_PROPERTIES = (
    "cryptographic confidentiality",
    "malicious-party security",
    "collusion resistance",
    "private set intersection",
    "formal differential privacy",
)

__all__ = ["IMPLEMENTED_PROPERTIES", "NOT_GUARANTEED_PROPERTIES"]
