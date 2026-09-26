# Privacy boundaries

The default simulator enforces architectural boundaries:

- raw feature matrices are owned by party objects;
- coordinators exchange local logits, residual signals, histogram summaries, and routing signals instead of raw features;
- audit events retain metadata rather than payload values;
- source identifiers need not appear in modelling messages.

These boundaries reduce unnecessary raw-data movement. They are not substitutes for secure computation against a malicious or colluding adversary.
