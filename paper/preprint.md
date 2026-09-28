# VertiMosaic: CPU-First Vertical Federated Learning for Heterogeneous Tabular Data

**Working technical report / preprint**  
Saurav Singla  
Version aligned with the VertiMosaic v0.3 research branch

## Abstract

Vertical federated learning (VFL) addresses settings in which multiple parties hold complementary feature columns for overlapping entities and wish to train a joint predictive model without pooling all raw feature tables at one location. Practical evaluation of VFL systems is difficult because public datasets with authoritative cross-source entity linkage are scarce, security properties vary substantially by protocol, communication measurements are often reported at incompatible layers, and experimental code may conflate federated and centralized baselines.

VertiMosaic is a CPU-first open-source research framework for reproducible VFL experiments on heterogeneous tabular data. The framework provides first-principles vertical logistic regression and histogram gradient boosting reference protocols, explicit active/passive party abstractions, auditable message transports, reproducibility bundles, statistical evaluation, provenance tracking, controlled overlap/dropout/drift studies, public exact-linked and semi-synthetic benchmark modes, empirical privacy attacks, and narrowly scoped privacy-enhancing mechanisms. The project deliberately separates raw-feature locality from cryptographic or differential-privacy guarantees. v0.3 strengthens protocol correctness with order-sensitive entity-alignment validation, makes missing-party behavior explicit, adds a protected logistic research path combining PSI-based intersection with clipped-Gaussian residual releases, defines a framework-neutral comparison contract for FATE/SecretFlow, and adds a real exact-NPI multi-source public benchmark builder.

All numeric claims in release versions of this report are intended to be generated from measured artifacts rather than manually entered tables. The repository's release-evidence manifest records hashes for canonical result files, environment metadata, and reproduction outputs.

## 1. Introduction

Organizations frequently observe different aspects of the same real-world entities. A financial institution may observe payment or credit behavior, a telecommunications provider may observe service usage, an insurer may observe claims or risk signals, and a retailer may observe purchasing behavior. Pooling such data into one centralized training table can be operationally, contractually, legally, or strategically infeasible. Vertical federated learning instead partitions the feature space across parties while aligning common entities and coordinating model training through protocol messages.

The phrase "federated learning" alone does not specify a security guarantee. A protocol may keep raw feature matrices local while still exposing gradients, logits, Hessians, routing information, model parameters, entity membership, or aggregate statistics. A system may use TLS without providing malicious-party security. A private-set-intersection step may protect entity matching without protecting subsequent model-training messages. Likewise, a differential-privacy mechanism applied to one message family does not establish end-to-end differential privacy for an entire training procedure.

VertiMosaic is designed around this distinction. The framework treats raw-data locality, protocol behavior, privacy mechanisms, empirical leakage, communication accounting, reproducibility, and benchmark provenance as separate properties that should be measured and documented explicitly.

The principal contributions of the framework are:

1. a CPU-friendly vertical logistic regression protocol implemented from first principles;
2. a vertical histogram-gradient-boosting reference implementation that keeps numeric split thresholds with the feature-owning party;
3. explicit message/transport abstractions and communication accounting;
4. deterministic run bundles containing configuration, predictions, metrics, environment metadata, provenance, hashes, and Git state;
5. benchmark modes that distinguish exact public linkage, exact-row vertical partitioning, authorized local linkage, and semi-synthetic cross-domain linkage;
6. statistical evaluation with bootstrap confidence intervals and paired metric differences;
7. empirical privacy attacks and mitigation experiments with explicit non-guarantees;
8. order-sensitive entity-alignment validation to prevent silent equal-length row-order errors;
9. explicit missing-party policies rather than silent omission of trained parties;
10. a protected logistic research path combining PSI intersection with clipped-Gaussian residual releases and zCDP accounting;
11. a reproducible external-comparator contract for established VFL frameworks; and
12. an exact-NPI multi-source public benchmark builder linking real provider entities across public data products.

## 2. Problem setting

Consider parties \(P_0, P_1, ..., P_k\) that hold different feature subsets for a common or partially overlapping set of entities. The active party \(P_0\) owns the target \(y\). Each party owns a local feature matrix \(X_j\). For the exact-overlap case, rows must be aligned such that row \(i\) in every \(X_j\) refers to the same entity.

VertiMosaic's default research goal is not to hide all intermediate information. It is to avoid pooling passive-party raw feature matrices at the active party while making the messages required by the reference algorithm observable, measurable, and auditable. Stronger mechanisms are opt-in and must be named individually.

### 2.1 Entity alignment is part of correctness

Equal matrix lengths do not imply entity alignment. If one party's rows are permuted, a VFL implementation can train without a shape error while associating unrelated features with the active party's labels. This produces a correctness failure that can be difficult to detect from aggregate metrics alone.

VertiMosaic v0.3 therefore supports explicit ordered entity identifiers. `bind_entity_ids` attaches immutable ordered IDs plus an order-sensitive SHA-256 digest to a party. `validate_exact_entity_alignment` verifies that all participating parties contain the same identifiers in the same order. Official research/reproduction entry points bind entity IDs and enable strict checks. The low-level API retains a compatibility mode for older callers, which is documented as relying on caller-supplied positional alignment.

### 2.2 Missing parties are a model condition, not a convenience

A model trained with multiple vertical parties generally changes when one contribution disappears. VertiMosaic logistic inference therefore defaults to `missing_party_policy="error"`. An explicit `zero_contribution` mode exists for availability experiments, but it must be evaluated as a separate degraded configuration. The active party cannot be omitted.

## 3. Reference protocols

### 3.1 Vertical logistic regression

For a set of aligned parties, the global logit can be expressed as

\[
z_i = b + \sum_j x_{ij}^{\top} w_j.
\]

Each party computes its local contribution from its own feature matrix and local weights. Passive parties send local logits through the transport abstraction. The active party owns labels and computes the loss and residual-related signal. Residual messages are returned to passive parties, which compute local gradients using local features. The reference implementation supports L1/L2 regularization, weighted classes, deterministic mini-batches, learning-rate schedules, gradient clipping, early stopping, and warm starts.

The implementation is intentionally transparent. Passive logits and residual signals are legitimate protocol messages and can leak information. Keeping \(X_j\) local does not make \(z_j\), residuals, or learned parameters private.

### 3.2 Vertical histogram gradient boosting

The VFL histogram GBDT implementation keeps train-derived numeric quantile thresholds local to each party. The coordinator receives aggregate candidate statistics and opaque feature/bin references. When a split is selected, the owning party resolves the private threshold and performs routing locally. The coordinator retains token-only routing handles rather than numeric threshold arrays.

This design avoids unnecessary threshold exposure but still reveals target-derived gradients/Hessians to passive parties and reveals node membership, aggregate candidate statistics, opaque references, and routed indices as required by the protocol. These signals are included in the threat model and privacy experiments.

## 4. Transport and communication accounting

`InMemoryTransport` provides deterministic protocol simulation and an auditable message log. `RemoteHTTPTransport` provides a serialized separate-process path with bounded requests/responses, authentication/authorization controls, replay and idempotency controls, rate limiting, optional compressed NumPy transport, signed-message support, and HTTPS/mTLS-capable deployment patterns.

VertiMosaic distinguishes logical protocol-payload accounting from network-stack measurement. In-process byte counts are not packet captures. The remote benchmark measures request/response behavior of the reference serialized path but does not claim TLS-record or full WAN byte accounting when the underlying client does not expose it.

## 5. Benchmark taxonomy

A central design goal is to avoid presenting synthetic linkage as real cross-organization linkage.

### 5.1 MovieLens 1M: exact public multi-table linkage

MovieLens provides observed user, rating, and movie identifiers that can be joined exactly. VertiMosaic derives temporally separated user behavior features and targets. This is genuine linkage across public tables describing the same service users, but it remains a single-service dataset.

### 5.2 UCI Credit: exact-row vertical partition

The UCI Credit benchmark partitions disjoint source columns while preserving exact rows. It is useful for correctness and utility sanity checks but is not evidence of entity resolution between independent organizations.

### 5.3 IEEE-CIS: authorized local exact linkage

When an authorized user supplies the transaction and identity files, VertiMosaic joins them on the observed `TransactionID` intersection and records file hashes and provenance. The source files are not redistributed by the repository.

### 5.4 Four-industry public-source benchmark: semi-synthetic linkage

The Bank/Telecom/Insurance/Retail benchmark uses real public source-domain datasets that do not describe the same individuals. Cross-source linkage is therefore explicitly semi-synthetic. The mode is useful for controlled cross-domain feature-complementarity research but must not be used as evidence of real cross-company customer matching.

### 5.5 Exact-NPI multi-source public benchmark

v0.3 adds a builder for a real-entity public benchmark using the 10-digit US National Provider Identifier (NPI) as the authoritative linkage key. Earlier-period Open Payments aggregates form active-party features, a later Open Payments period provides the target, NPPES supplies one passive feature set, and CMS Care Compare/provider-data supplies another passive feature set.

The same NPI identifies the same provider across the public sources, so linkage is exact rather than synthetic. The benchmark is nevertheless public-government-data research and is not evidence of confidential private-company federation.

## 6. Evaluation methodology

Standard entity-level splits use train/validation/test partitions. Threshold selection uses validation predictions only. Numeric preprocessing, imputers, category vocabularies, and transformations that learn from data should be fitted on training entities only.

Reported predictive metrics include ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, calibration error, and confusion counts. Deterministic bootstrap intervals quantify sampling uncertainty. Paired bootstrap differences are used when comparing predictions on the same test entities.

Resource measurements include training/inference wall-clock time, process resident memory measurements, logical protocol messages/scalars/bytes, and separate remote-transport measurements where applicable.

## 7. Robustness studies

VertiMosaic includes controlled studies for partial entity overlap, missing parties, feature drift, model family, party contribution, and scale.

Partial-overlap experiments separate intersection coverage from predictive utility. Missing-party experiments distinguish training-time availability strategies from inference-time fallback. Drift experiments perturb means, variance, missingness, and categorical frequencies in controlled ways. These experiments provide stress tests, not universal guarantees about production distribution shift.

## 8. Privacy and security research

### 8.1 Threat model

The default setting is primarily honest-but-curious. Parties are expected to execute the protocol but may inspect messages they are legitimately allowed to observe. Malicious-party deviations, collusion resistance, Byzantine robustness, and poisoning defenses are not generally solved by the default protocols.

### 8.2 Empirical attacks

The repository includes reproducible attack measurements covering confidence-based membership inference, residual-label inference, gradient/feature reconstruction scenarios, and routing/entity exposure for tree protocols. Attack results are configuration-specific empirical evidence rather than formal privacy proofs.

### 8.3 Clipped Gaussian residual release

`ClippedGaussianDPBackend` L2-clips each protected vector, derives the corresponding sensitivity bound under the configured adjacency relation, adds Gaussian noise, and accounts repeated releases using zCDP. v0.3 wires this backend directly into active-to-passive logistic residual messages.

The model can return a privacy report containing release count, clipping norm, adjacency, noise multiplier, rho, epsilon, and delta. The report also states that other model messages remain outside the mechanism.

### 8.4 PSI plus protected logistic path

The `fit_protected_logistic` research path performs PSI-based entity intersection, canonical ordering, strict entity binding, and VFL logistic training with clipped-Gaussian residual releases. The optional packaged PSI implementation is the OpenMined PSI adapter.

The guarantee is compositional and narrow: PSI protects set intersection according to the selected backend, and DP accounting covers the residual releases passed through the clipped-Gaussian backend. Logits, learned parameters, timing, message sizes, and other signals are not automatically protected.

### 8.5 Other research primitives

VertiMosaic also includes reference pairwise-mask secure aggregation for integer sums, additive secret-sharing sum experiments, and optional Paillier homomorphic-sum support. These components are deliberately not presented as a complete malicious-secure MPC runtime or encrypted end-to-end tree/logistic training system.

## 9. Comparison with established VFL frameworks

Internal centralized and single-party baselines validate objectives and utility, but they do not establish competitiveness with established VFL software. v0.3 therefore defines a stable external-comparator contract for FATE, SecretFlow, or another VFL framework.

An external runner receives a frozen benchmark manifest and must emit normalized utility/runtime fields. The wrapper records hashes, environment details, framework identity, and an interpretation boundary. A paper comparison row is considered complete only when a measured normalized comparator result exists in the released evidence bundle. Missing external results are reported as pending rather than manually estimated.

This design is intentionally framework-neutral because FATE and SecretFlow differ from VertiMosaic in deployment topology, cryptographic options, optimization implementation, and measurement layers. Utility/resource comparison and security-property comparison should therefore be presented separately.

## 10. Reproducibility and canonical evidence

Every standard VertiMosaic run can record configuration, dataset/linkage provenance, predictions, metrics, training history, communication metadata, seed, dependency/environment versions, timestamps, Git state, and hashes.

Release evidence is generated in a pinned Python environment. `scripts/build_release_evidence.py` builds a complete artifact manifest and a `canonical-results.json` index. Numeric claims should trace to measured files listed in this index or to normalized external-comparator records. The policy prohibits manually typing a metric into a release table as a substitute for a missing measured artifact.

Maintainer-run CI is not independent validation. The repository includes a separate `reproductions/` submission format in which unaffiliated reproducers can publish affiliation/relationship attestations plus the untouched evidence bundle generated from a released package.

## 11. Software quality and release engineering

The project uses a `src/` package layout, typed interfaces where practical, Ruff lint/format checks, mypy, pytest, branch coverage, protocol-critical coverage gates, Bandit, dependency auditing, CodeQL, package builds, Twine validation, pinned GitHub Actions, dependency automation, SBOM/security workflows, portability smoke tests, release metadata, citation metadata, and release-evidence generation.

These controls improve software quality and supply-chain visibility but do not substitute for protocol-security proofs or independent scientific replication.

## 12. Limitations

The most important limitations are summarized here; the complete discussion lives in `docs/limitations.md`.

1. The default VFL protocols are not malicious-secure and do not provide generic collusion resistance.
2. Raw-feature locality does not prevent leakage from gradients, Hessians, logits, residuals, routing information, aggregate statistics, or learned parameters.
3. The four-industry benchmark is semi-synthetic at the cross-domain linkage layer.
4. Exact public linked benchmarks do not establish the operational/legal feasibility of private cross-company federation.
5. In-process communication bytes are protocol-level estimates rather than packet captures.
6. CPU benchmark results do not establish internet-scale or accelerator-scale performance.
7. Fairness across demographic or other protected groups requires separate analysis.
8. Synthetic drift studies do not exhaust real production distribution shifts.
9. Optional privacy mechanisms protect only the steps/messages to which they are explicitly applied.
10. No repository feature constitutes a legal or regulatory compliance certification.
11. Independent external reproduction and established-framework comparator rows remain separate evidence requirements; maintainer-generated artifacts do not satisfy them by themselves.

## 13. Research questions and artifact policy

The experiment manifest defines research questions covering complementary utility, centralized-objective agreement, nonlinear versus linear VFL, predictive party utility, partial entity overlap, unavailable parties, industry-specific drift, CPU scaling/communication, and sensitivity to benchmark mode.

Figures and tables are generated from result files. If a required measured input is absent, artifact generation should fail or mark the evidence as incomplete. The repository should never replace an unavailable experiment with a plausible hand-entered value.

## 14. Conclusion

VertiMosaic is intended as a transparent research substrate for vertical federated learning rather than a black-box claim of privacy. Its primary engineering principle is that raw-data locality, entity correctness, model utility, communication cost, reproducibility, privacy leakage, and cryptographic/privacy mechanisms should be separately inspectable.

The v0.3 hardening work closes two protocol-level correctness gaps—silent row-order mismatch and implicit missing-party inference—and connects previously isolated privacy mechanisms to an executable protected logistic path. It also broadens evidence through a real exact-linked public multi-source benchmark, a formal external-framework comparator contract, canonical release-result indexing, and a structured path for genuinely independent reproduction.

The remaining scientific milestones are measured external FATE/SecretFlow comparison on a frozen benchmark and at least one unaffiliated reproduction of a released evidence bundle. Those are intentionally treated as external evidence rather than something the maintainer can self-certify.

## Reproducibility statement

The repository contains the experiment manifest, commands, pinned reproduction dependencies, test suite, result-generation scripts, release-evidence builder, privacy experiments, comparator contract, and independent-reproduction format needed to audit the claims in this report. Release numeric results should be cited from the corresponding immutable evidence bundle and its SHA-256 manifest.
