# Architecture

VertiMosaic models four feature-owning parties over aligned or overlapping entities. The active Bank party owns the binary target in the reference benchmark. Passive parties never need to transfer raw feature matrices to the coordinator.

The single-process `InMemoryTransport` exists to make message boundaries auditable. It records only message type, sender/receiver roles, scalar counts, and estimated bytes; it deliberately does not retain payload values in the audit history.

The implementation is designed so a real network transport and secure computation layer could later replace the in-memory simulator without redefining vertical feature ownership.
