# Architecture

VertiMosaic models column-partitioned learning. Bank is the active party and owns the binary target; Telecom, Insurance, and Retail are passive parties that own complementary feature columns for aligned entities. Federated model code requests local computations from party objects rather than concatenating passive raw matrices.

`InMemoryTransport` records only message metadata: type, sender/receiver roles, shapes, scalar counts, and estimated bytes. It never retains transmitted array values. This provides a reproducible communication simulator, not real network isolation.
