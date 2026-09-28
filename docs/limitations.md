# Limitations

VertiMosaic is a CPU-first research implementation for studying vertical federated learning on tabular data. It is designed to make assumptions, measurements, and non-guarantees explicit. It should not be interpreted as a production security product or as evidence that a particular deployment is compliant, private, fair, or operationally safe.

## External validity

The four-industry Bank/Telecom/Insurance/Retail benchmark uses real public source-domain data, but those source files do not describe the same real people. Its cross-domain linkage is therefore semi-synthetic and is not evidence of deployment performance on real cross-industry customers.

MovieLens provides genuine exact linkage across public tables for the same service users, but it remains a single-service dataset. UCI Credit exact-row partitioning is useful for correctness and controlled vertical experiments, but both sides originate from one source dataset. IEEE-CIS can provide an exact transaction/identity join when an authorized user supplies the local files, but it is still a two-table competition dataset.

The optional NPI benchmark provides genuine exact linkage across multiple public provider data products. It describes real providers, but the source data are public US government/provider datasets and therefore do not demonstrate confidential private-company collaboration.

No public benchmark in the repository should be generalized automatically to banks, insurers, telecom operators, payment networks, hospitals, or other organizations without separate domain validation.

## Entity alignment assumptions

Vertical learning is correct only when every party's row position refers to the same entity. Equal row counts are insufficient. v0.3 provides order-sensitive entity identifiers/digests and strict alignment validation for official research/reproduction paths.

Low-level API callers can still operate in a backwards-compatible mode without bound IDs. Such usage relies on positional alignment supplied by the caller and cannot detect a pre-construction row permutation. Production-like experiments should bind explicit pseudonymous entity identifiers and enable `require_entity_ids=True`.

The default digest-based alignment check verifies equality/order after identifiers are available; it is not private set intersection. The optional OpenMined PSI adapter protects the set-intersection operation according to its upstream threat model, but does not protect the remainder of VFL training traffic.

## Honest-but-curious assumption

The default research protocols primarily model honest-but-curious participants: parties execute the prescribed protocol but may inspect legitimate messages. The project does not provide a general malicious-party proof or runtime capable of preventing arbitrary protocol deviation.

A malicious party may send malformed gradients, Hessians, logits, histograms, routing information, model updates, or metadata. Application-layer validation, transport authentication, replay controls, and bounded messages reduce some operational risks but do not constitute malicious-secure multiparty computation.

## Collusion

The default protocols do not provide collusion resistance. Two or more participants may combine observations in ways that reveal more information than any one participant can infer independently. Optional reference primitives such as pairwise-mask secure sums and additive secret sharing have explicitly narrower assumptions and must not be promoted to a blanket collusion-resistance claim.

## Gradient, Hessian, residual, and logit leakage

Raw passive feature matrices remain local in the reference VFL protocols, but derived signals can be sensitive. Depending on the model and observer, gradients, Hessians, residuals, logits, aggregate split statistics, model parameters, and repeated message trajectories may reveal label, feature, membership, or distribution information.

The repository includes empirical attacks and mitigations for specific configurations. A reduced empirical attack score is not a privacy proof and may not transfer to a stronger attacker, a different dataset, or a larger number of rounds.

## Routing and tree leakage

Vertical histogram boosting intentionally keeps numeric thresholds with the feature-owning party, but node membership, opaque feature/bin references, aggregate candidate statistics, and routed entity indices are legitimate protocol outputs. Those messages can leak structural or membership information.

Opaque references hide literal feature names/thresholds from the coordinator; they do not make the surrounding protocol information-free.

## Differential privacy scope

The clipped-Gaussian backend enforces a message-level L2 clipping/sensitivity contract and records zCDP composition for releases routed through it. v0.3 can attach this backend directly to active-to-passive logistic residual messages.

That accounting covers those releases only. End-to-end differential privacy requires a precise neighboring-dataset definition plus accounting for every sensitive data-dependent release that influences the observer. VertiMosaic therefore does not claim end-to-end DP for the default VFL protocol.

## PSI, secure aggregation, MPC, and homomorphic encryption

Optional PSI, pairwise-mask secure sum, additive secret sharing, and Paillier sum adapters are research mechanisms with narrow scopes. They are not a complete end-to-end encrypted VFL runtime, do not automatically protect unrelated messages, and do not imply malicious-party security, key-management maturity, dropout recovery, or collusion resistance.

The protected logistic research path combines PSI-based entity intersection with clipped-Gaussian residual releases, but its guarantee is intentionally compositional and partial: PSI protects the set-intersection step according to the selected backend, and the DP mechanism protects the configured residual releases. Other messages remain outside those guarantees.

## Poisoning and integrity

VertiMosaic does not solve feature poisoning, label poisoning, backdoor attacks, Byzantine participants, adversarial model updates, or coordinated data manipulation. Transport signatures/authentication help establish message provenance; they do not establish that a participant's local computation or source data are truthful.

## Communication measurement

In-process communication numbers are logical protocol-payload estimates. They are not packet captures and do not include the full network stack, TLS records, retransmissions, kernel buffering, serialization overhead not represented by the accounting layer, proxies, service meshes, or cloud load balancers.

The separate-process remote benchmark measures request/response behavior for the reference serialized HTTP path, but it is still not a substitute for production network instrumentation across real organizations and geographic links.

## Runtime and scalability

VertiMosaic is deliberately CPU-first and prioritizes transparent reference implementations over maximum throughput. Benchmarks on one CPU, operating system, or memory hierarchy should not be treated as universal performance results.

The project does not currently establish internet-scale federation, very wide sparse feature spaces, billions of rows, high-latency WAN convergence, elastic cluster scheduling, accelerator efficiency, or fault tolerance comparable to mature distributed ML platforms.

## Missing parties and availability

VFL models can become unavailable or change behavior when one trained party is absent. Logistic inference now rejects missing parties by default. The explicit `zero_contribution` policy exists for controlled availability research only and should be used only when the resulting degradation has been evaluated for the intended deployment.

Histogram GBDT does not silently omit a trained party. Missing-party strategies require separately defined training/inference configurations and measurements.

## Fairness and subgroup performance

The repository reports predictive utility and calibration metrics, but this does not establish fairness across demographic, geographic, economic, clinical, or other subgroups. Public benchmark categories may be incomplete, historically biased, or inappropriate proxies.

Any deployment-sensitive use requires a separately designed fairness evaluation, representative population analysis, and review of which attributes are ethically and legally appropriate to use.

## Distribution shift

Synthetic drift experiments provide controlled stress tests; they do not exhaust real-world covariate shift, label shift, concept drift, policy changes, fraud adaptation, participant entry/exit, or measurement changes. Robustness conclusions apply only to the tested perturbations and ranges.

## Data quality and licensing

External dataset availability, schemas, provider metadata, terms of use, and licensing can change. The repository records provenance where practical, but users remain responsible for verifying source licenses/terms at retrieval time and for determining whether their local use/redistribution is permitted.

No restricted dataset should be committed merely to make a benchmark easier to reproduce.

## Deployment boundary

The remote HTTP transport is a research/reference transport with authentication, replay/idempotency controls, bounds, rate limiting, optional signed messages, and HTTPS/mTLS-capable deployment patterns. It is not by itself a production service mesh, secrets platform, HSM/KMS integration, certificate lifecycle system, disaster-recovery design, or audited security boundary.

The Docker Compose tutorial is a reproducible deployment example, not a production architecture recommendation.

## Compliance and legal non-claims

VertiMosaic does not certify compliance with GDPR, UK GDPR, DPDP Act, HIPAA, PCI DSS, GLBA, banking regulations, model-risk standards, data-localization rules, contractual restrictions, or any other legal/regulatory framework.

Keeping raw features local may be relevant to some governance designs, but legal obligations depend on the complete data flow, identifiers, derived signals, purpose, parties, jurisdiction, contracts, controls, retention, and organizational processes.

## Research evidence and independence

Maintainer-run CI, maintainer-generated release artifacts, and deterministic re-runs are valuable reproducibility evidence but are not independent validation. A stronger claim requires an unaffiliated person or organization to install a released package from a public source and publish the complete evidence bundle plus relationship/affiliation disclosure.

External FATE/SecretFlow comparison rows likewise remain incomplete until measured results from the frozen comparator contract are present. Missing external evidence must be reported as pending rather than replaced by manually entered or estimated values.
