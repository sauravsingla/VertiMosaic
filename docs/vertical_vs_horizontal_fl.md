# Vertical versus horizontal federated learning

- Horizontal FL: parties typically hold different rows/entities with comparable columns.
- Vertical FL: parties hold different feature columns over the same or overlapping entities.

VertiMosaic implements vertical partitioning. Synthetic-scale data uses identical aligned `entity_id` values across parties. External four-industry experiments must construct and disclose synthetic linkage because the independent public datasets do not share real identities.
