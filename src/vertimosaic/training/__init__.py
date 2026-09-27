"""Training-facing public API for VertiMosaic reference protocols.

The model classes own their protocol-specific ``fit`` implementations; this
package provides the stable training namespace required by the repository
layout without duplicating optimization logic.
"""

from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression

__all__ = ["VFLHistGBDT", "VFLLogisticRegression"]
