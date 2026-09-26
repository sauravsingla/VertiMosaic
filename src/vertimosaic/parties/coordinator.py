from __future__ import annotations

from dataclasses import dataclass, field

from vertimosaic.transport import InMemoryTransport


@dataclass
class Coordinator:
    """Metadata-only protocol coordinator.

    The coordinator knows party names and owns the simulated transport, but it
    intentionally has no field for party feature matrices or target vectors.
    """

    party_names: tuple[str, ...]
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)

    def __post_init__(self) -> None:
        if not self.party_names:
            raise ValueError("coordinator requires at least one party")
        if len(set(self.party_names)) != len(self.party_names):
            raise ValueError("coordinator party names must be unique")

    def has_party(self, name: str) -> bool:
        return name in self.party_names
