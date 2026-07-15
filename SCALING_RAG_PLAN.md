# DocARAG improvement plan: maintainability, usability, scale, latency, and accuracy

- Status: compatibility foundation, a narrow dependency-inverted document-extraction application use case, routed ingestion seam, strict vendor-service clients, immutable local parser CAS, explicit structural and bounded semantic-subgraph preservation for Docling/MinerU/Paddle, canonical selected-material identifiers (not yet exact per-invocation model-use receipts), immutable evidence-graph contracts/builder/publication, a dependency-neutral retrieval-candidate contract, Qdrant as the sole vector backend, and a checksummed PostgreSQL metadata/work/trace/index ledger are implemented and regression-tested. The compatibility CLI now requires PostgreSQL and records document admission plus dispatch events, while full adoption by every legacy BookRAG stage, live KG/candidate-contract wiring, real-model qualification, supported routed indexing, and clean deployment profiles remain staged.
- Audit date: 2026-07-15
- Repository baseline: `b6e37d3`
- Scope: code/model/API maintainability, setup and operations, indexing, multimodal retrieval, graph/tree retrieval, serving, evaluation, and migration

## 1. Executive decision

Do not choose one universal parser. Make the **page router** the primary parser and use three specialized routes:

1. **Docling fast/native route** for clean, digitally generated reports and pages with a reliable text layer.
2. **MinerU technical route** for formula-heavy, table-heavy, and structurally complex technical pages.
3. **PaddleOCR-VL visual route** for scans, slides, screenshots, diagrams, workflow pages, image-based tables, and difficult OCR.

This parser router does **not** replace BookRAG. BookRAG remains DocARAG's reasoning and retrieval architecture. The router improves the multimodal blocks from which BookRAG constructs its `BookIndex`:

```text
BookIndex B = (T, G, M)

T = document-native tree of sections and multimodal evidence nodes
G = entity/relation knowledge graph extracted from those nodes
M = Graph-Tree Links (GT-Links) from graph entities/relations back to evidence nodes
```

The target system preserves BookRAG's agent planning, `Select_by_Entity`, `Select_by_Section`, modality/range filters, graph and text reasoning, Skyline ranking, and Map/Reduce synthesis. The work in this plan makes `T`, `G`, and `M` durable, versioned, multimodal, multi-document capable, and traceable; it does not reduce DocARAG to ordinary vector RAG.

Operational simplicity is a product requirement. The default installation must be useful without a local GPU, the full-GPU stack must start through one supported profile, and adding or replacing a model must not change BookRAG domain code. Optimize the system for the simplest immutable build policy and request-time query budget that meet the declared requirements; expensive parsers are build-gated, while graph/VLM/global reasoning is also capability- and query-gated.

The benchmark evidence already shows why:

| Test document | Best observed route | Evidence |
|---|---|---|
| `ai_report.pdf`, clean native report | Docling | Best inspected hierarchy and Markdown tables; 11.886 s versus MinerU 46.347 s and PaddleOCR-VL 12.259 s |
| `312_LG_E_L3.pdf`, formula-heavy worksheet | MinerU | Best inspected formulas, technical reading order, and structured extraction, although slower |
| `noetics_pitch_deck_v28.pdf`, visual slide deck | PaddleOCR-VL | Best inspected visual OCR, screenshot tables, and diagram labels; 45.597 s versus MinerU 117.848 s and Docling 146.952 s |

Character, word, heading, and Markdown-marker counts in the existing benchmark are smoke metrics, not quality scores. The qualitative conclusions above came from comparing the PDF and parser artifacts. They must be converted into a repeatable annotated evaluation before the router is trained or thresholds are fixed.

For RAG, ship a cheap **base index** first and make expensive intelligence progressive:

- Base tier: canonical blocks, provenance, deterministic structure, dense + BM25 retrieval.
- Precision tier: text reranking and visual retrieval for eligible queries.
- Global-analysis tier: cached section summaries and bounded graph exploration only when the query needs them.

This reverses the current failure mode in which all documents pay for eager node summaries and a knowledge graph before becoming searchable. RAPTOR supports hierarchical retrieval, but it does not imply that every summary must be generated in one eager transaction. Microsoft’s LazyGraphRAG results are a strong reason to defer LLM graph work and expose an explicit request-time query budget ([RAPTOR](https://arxiv.org/abs/2401.18059), [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)).

The maintainable product shape is:

| Concern | Decision |
|---|---|
| Retrieval and reasoning core | Keep BookRAG `B=(T,G,M)` and its planner/operators; a backend-neutral candidate port preserves domain isolation while Qdrant is the sole/default vector database. Qdrant never replaces the tree, graph, GT-Links, or PostgreSQL authority. |
| Parser/model replacement | Use small capability ports and one canonical evidence contract. Vendor SDK objects never cross into BookRAG or public APIs. |
| Rich document fidelity | Retain text, OCR words, table cells/spans, formula markup, crops, figures, diagram elements/edges, reading order, hierarchy, and every parser assertion as separate evidence-bearing representations. |
| Conceptual graph | Publish an evidence-grounded directed multigraph sidecar; derive the current BookRAG graph and mapping views as checked compatibility projections. |
| Source of truth | Immutable document bytes -> parser observations -> canonical evidence revisions -> versioned `T/G/M`; Markdown, vectors, summaries, caches, and UI views are rebuildable derivatives. |
| Retrieval default | Dense + BM25 candidates fused by ordinary RRF, then a bounded reranker and BookRAG narrow-then-reason; visual/table/formula legs activate only for matching intent or failed text quality. |
| Code organization | Modular-monolith controller plus isolated resource-conflicting workers. CLI, API, SDK, Web UI, and tests call the same application use cases. |
| Setup | Keep the implemented PostgreSQL/Qdrant `storage up/down/migrate/status` lifecycle, then add one whole-product CLI with `doctor`, explicit model pull, health-gated parser/model-service `up/down`, and curated `lite`/`workstation`/`production` profiles; no source edits or surprise model downloads. |
| Latency and resource control | Stream pages, use bounded queues/microbatches/candidate sets, checkpoint expensive work, and admit GPU stages through an explicit capacity lease. |
| Default eligibility | A component must clear modality accuracy, p95 latency, memory, clean-host setup, restart/resume, diagnostics, and upgrade-compatibility gates; benchmark accuracy alone is insufficient. |

This table describes the target contract. The checkpoint below says which parts
are actually implemented in the current worktree.

### 1.1 Implementation checkpoint (2026-07-15 worktree)

This plan distinguishes the compatibility foundation that now exists from the
target architecture. Passing a compatibility test is not evidence that the
canonical, scalable design is complete.

| Area | Implemented now | Still required before the target claim is true |
|---|---|---|
| Supported entry point | Repository/editable-install `docarag` CLI with `init`, `settings`, read-only `setup-plan`, `doctor`, implemented `storage up/down/migrate/status`, routed-profile `validate/show`, parser-worker `validate/serve`, `index`, `status`, `verify`, `ask`, and loopback `web`; the installed entry point plus help, settings, and storage command parsing pass a clean-wheel boundary test without checkout-owned `Core`, PyYAML, or model-framework imports | Standalone complete `lite` application services and packaged compatibility-run resources: `status` execution, named-run verification, `index`, `ask`, and `web` still use a configured checkout. Also required: model management, curated deployment profiles, stable `/v1` API and SDK |
| Reproducible indexing | PDF/config/code-derived revision ID, content-addressed source snapshot, attempt-local state/logs, runtime fingerprint, checksummed artifact manifest, deterministic MinerU observation/canonical artifacts, derived-block lineage, quick/deep verification of run-local artifacts, a fixture-only runtime compatibility manifest inspected by `doctor`, an atomic source/tree/config/embedding-bound Qdrant readiness attestation plus scoped live count, and PostgreSQL document/dispatch trace adoption | Certified package/model/container pins and served-weight digests in revision identity; remote Qdrant content-digest/outbox reconciliation able to detect same-count mutations; adoption of PostgreSQL work leases, artifact receipts, index revisions, and dead-letter recovery by every legacy stage; distributed coordination and full durable `T/G/M` publication |
| Default storage | Loopback-only Compose starts PostgreSQL 18 and Qdrant with durable named volumes. PostgreSQL has explicit checksummed migrations and APIs for immutable document versions/artifact receipts, guarded index revisions, append-only events, idempotent work, `FOR UPDATE SKIP LOCKED` leases, attempts and fencing; the compatibility CLI currently adopts only document admission and dispatch events. Qdrant is the only vector provider and uses deterministic IDs, scoped payload filters, idempotent upserts/deletes, and readiness attestations checked against exact scoped counts. | Backup/restore drills, outbox integration, access-policy enforcement, full legacy-stage PostgreSQL ledger adoption, remote Qdrant content reconciliation, Qdrant sparse/visual/multivector qualification, and clustered production operation |
| Routed parsing contract | Dependency-light Docling, MinerU-normalized, and PaddleOCR-VL translators; source-bound lazy PyMuPDF page probe; explicit capability registry; deterministic router; bounded fallback/adjudication; versioned generic parser-worker HTTP client/server; strict clients for Docling Serve async export, MinerU 3.x task results, and PaddleOCR-VL's complete `/layout-parsing` pipeline; one-vendor foreground worker composition with strict secret-safe TOML profiles; fail-closed fusion and canonical provider; typed MinerU effort; contiguous sparse-page batching; verified source-byte handoff; immutable local-CAS publication for raw exports and parser assets; exact Paddle response retention with request-wide artifact bounds; result-scoped MinerU middle-artifact receipts; explicit bounded parent/caption/footnote/reference/reading-order/table-continuation assertions; and typed table-cell, bounded MathML-symbol, explicit diagram/flow subgraphs preserved through reconciliation and BookRAG tree metadata | Pinned worker/service images and owned supervision; supported routed `docarag index` composition through the application service; shared/object artifact transport for multi-host workers; CAS quota/garbage collection; curated deployment profiles; measured routing thresholds; live qualification of MinerU/Paddle extension fields; real-model qualification/version pins; and the clean-host deployment gate |
| Evidence graph | Immutable dependency-light contracts can represent exact representation/asset/char-span/table-cell/bbox/diagram anchors, extractor-revision-bound entity mentions, directed parallel relation assertions and candidate GT-Links. The current bounded builder consumes encoded source-bound legacy BookRAG KG checkpoints only after byte limits. It verifies exact deterministic length-framed canonical-input and tree-evidence-descriptor snapshots plus selected CAS assets, then incrementally guards release materialization. The descriptor snapshot includes `tree_revision_id`, but does not bind full `T` topology, order, summaries, or node content. Original PDF bytes, parser-native artifacts, the full tree and legacy BookRAG pickle remain separately traceable. Checkpoint assertion evidence is coarse whole-node selection, not an assertion-specific model-consumption receipt; releases produced by this path are candidate/audit artifacts and are **not authority-eligible**. Checkpoint publication/read primitives exist; adoption must fail closed without an external ledger authorization. The legacy `G`/`M` projection is inspection-only. Canonical-to-BookRAG image paths default to an active verified CAS lease; the sole unverified mutable-path enum is explicitly legacy-inspection-only and marks every output row accordingly. | Publish and adopt a full-tree manifest binding root, topology, sibling order, complete relevant node content, source/canonical revisions, tree revision and descriptor digest. Invoke exact-receipt graph construction from the live KG stage, preserve pre-repair extractor assertions separately, verify exact consumed materials plus each assertion's evidence subset and semantic support, then adopt only through the fenced durable ledger. Add persisted alias decisions/entity revisions, relation-aware bounded retrieval, citation hydration, and sharded/streamed publication above the whole-release compatibility envelope |
| Provider/runtime boundary | Backend-specific LLM/VLM/embedding/reranker validation and request plumbing; finite 300-second default per-attempt remote LLM/VLM request timeout; 30-second default remote-rerank request timeout; SDK retry suppression plus an explicit `1..10` attempt limit; finite Ollama `keep_alive="5m"` default and opt-in unload; explicit owned/borrowed `RAGSession`; context-local token/log scopes propagated through bounded worker threads; process-local reference-counted GME ownership with one-call GPU admission; guarded remote-only import tests exclude local ML stacks | Overall wall-clock deadlines/cancellation, cross-process GPU/model ownership leases, safe shared-server unload/down, warm model retention, package lock and clean-host real-service qualification |
| Failure behavior | Required-stage failures return nonzero; detached stages stop on invalid output; typed LLM/VLM/KG failures replace fake content; summary and KG work checkpoints independently; bounded provider/KG admission and explicit model/store cleanup are implemented; missing/invalid structural filters, oversized plans, and nonempty-but-unusable retrieval fail closed | Capability-aware GPU leases, uniform retry classification across every legacy stage, and item-level recovery after process/host loss |
| Current BookRAG answers | Canonical catalog/tree hydration validates document/source/tree/canonical lineage and emits citation IDs. Parser-service assets can now be atomically stored as digest-addressed local CAS bytes and reverified without following symlinks. Directed structural relations plus bounded table/formula/diagram subgraphs reach BookRAG node metadata with assertion lineage; full graphs are stored once in the tree and vector rows carry compact receipts, but these subgraphs are not yet used as the semantic KG or element-level retrieval units. Assets remain withheld from the active VLM path until hydration owns the verified byte snapshot. Simple/global/complex answer markers are structurally validated against supplied canonical bindings with collision-free scopes, while traces explicitly say semantic entailment is unverified. | Verified asset snapshot-to-model integration, a shared/object artifact contract for remote workers, typed claim-bearing `AnswerResult`, semantic per-claim support validation, evidence-sidecar-backed directed multigraph retrieval, element-level table/formula/diagram candidate legs, and durable published GT-Links |
| Query/index complexity | Parser selection is linear in pages and registered adapters; routed targets are split into maximal contiguous runs before bounded batches so sparse pages do not force parsing of every intervening page; reconciliation has explicit plausible-candidate/raw-probe caps; lossless tree construction is `O(B + T)`; the **local** lexical/structural preselector scans a chosen subtree once and retains at most `M` reranker inputs in `O(text bytes + N_S log M)` time and `O(M)` retained-candidate memory; section partitioning is `O(S + L)` and the two-score Skyline is deterministic `O(M log M)` and hard-capped. The current end-to-end query still materializes the selected subtree and an induced graph, runs PageRank over that graph, and sorts graph candidates, so its time is approximately `O(N_S + L_S + I_S(V_S+E_S) + N_G log N_G + text bytes + N_S log M)` and memory is `O(N_S+V_S+E_S)` plus an optional guarded `O(V_S^2)` semantic-link matrix. The dependency-light source/canonical/tree/index-bound candidate contract performs ordinary RRF/deduplication with a bounded top-`K` heap in `O(H + U log K)` time and `O(H)` working memory for `H` hits and `U` unique evidence identities; material unions are deferred to the selected `K` outputs. It records exact plan/leg outcomes, permits base legs without KG and gates graph legs on eligible ledger-adopted releases; no live BookRAG/vector adapter invokes it. Its candidate-to-material associations remain provisional references until the hydrator revalidates scope, revision and bytes; they are not source evidence by themselves. Model/KG work and planner/entity/filter/prompt fan-out have item caps, but not yet one propagated wall-clock/token/visual budget. | Wire current TreeVDB/entity/section adapters through the candidate contract, run scoped dense+sparse/entity legs in parallel, add bounded graph expansion and offline ANN semantic links, measure candidate policies, bound parallel independent complex sub-queries, stream every non-BookRAG baseline, use asynchronous hierarchical global reduction, and enforce end-to-end latency/token/pixel budgets |
| Batch evaluation | Stable SHA-256 document sharding is independent of corpus order for a fixed shard count; query directories and reuse are bound to exact question/artifact/config/model/code identity; private atomic result/commit/final publication; JSONL/NDJSON, generators, and DataFrame rows stream in `O(1)` extra result-count memory | Stream `main.py` input/grouping instead of eager pandas; cross-process output-directory writer lease; cache retention/garbage collection; immutable served-weight identity |
| Tests | CLI/setup-plan/diagnostics, storage lifecycle/configuration, checksummed PostgreSQL migrations and ledger semantics, Qdrant provider/readiness, index-driver, all three parser adapter fixtures, page-probe/source-material binding, generic worker transport, simulated official vendor-service protocols, adversarial ZIP/CAS publication, fixture-only compatibility checks, routed provider reuse/failure behavior, router/orchestrator contracts, adversarial reconciliation, lossless tree construction, bounded retrieval/planning, lineage, projection, checkpoints, concurrent token/log/session/GME ownership lifecycle, provider request/timeout truth, remote-only imports, stable sharding, and adversarial query-cache streaming | Live vendor-service/model integration, certified compatibility matrix, golden parser/RAG accuracy, live database backup/restore, load/chaos and clean-host setup tests |

The routing/adaptation/fusion **application contract**, source-bound page probe,
generic worker transport, strict vendor-service clients, strict foreground
worker profiles/composition, immutable local parser CAS, run-local durable
provider, opt-in composition root, structural-relation DTOs and evidence-graph
release contracts now exist. They are
not connected through the supported `docarag index`, and simulated protocol
tests are not real-model qualification. Pinned service images, supervision,
certified real-model pins, shared/object artifact delivery for multi-host
workers, and clean-host deployment remain open. Other major capabilities remain
unimplemented: full database-backed canonical and `T/G/M` adoption, wiring the
implemented resumable item-level PostgreSQL workflow into every stage, Qdrant
sparse/visual hybrid indexing, immutable asset-byte-to-model
materialization and semantic claim-support verification, multi-document corpus indexing, a
capability-aware GPU lease manager, curated service profiles, and an authenticated
service API. The current opt-in routed TOML profile is configuration input, not a
qualified deployment profile. The current
canonical DTOs and run-local JSON ready pair are a compatibility authority for
one run, not the transactional corpus store described below.

### 1.2 Immediate implementation order

The highest-value next work first makes the package and application boundary
real, then proves one traceable BookRAG source-to-search journey. New extraction,
graph, model and retrieval behavior must enter through that boundary rather than
adding more lifecycle and policy logic to the compatibility scripts:

1. **`APP-1` — narrow extraction seam implemented; complete the package
   boundary:** the dependency-light routed document-extraction use case now
   validates a bounded request, exact source/document/configuration identity,
   accepted canonical blocks, and both the canonical artifact and replayable
   routed audit's actual schema, contents and bytes before returning compact
   companion receipts. Move the remaining
   `index/search/query/status/evidence` workflows into
   shared application use cases, package required resources or declare a
   separately versioned compatibility plugin, add locked dependency groups,
   and gate a wheel installed into a clean environment. CLI, future `/v1` API,
   Web UI and tests must call those same use cases.
2. **`LEDGER-1` (foundation implemented; full stage adoption depends on
   `APP-1`) — complete PostgreSQL adoption.** PostgreSQL is now the required
   default durable ledger. Its deterministic migration provides tenant/corpus/
   document versions, immutable artifact receipts, guarded index revisions,
   append-only stage events, and idempotent leased work with attempts and
   fencing. The compatibility CLI records document admission and dispatch
   events. Next, make every stage use those records, adopt verified artifacts
   by hash rather than copying payloads, bind exact `T/G/M` selectors and a
   canonical-catalog root, and add backup/restore plus outbox recovery tests.
3. **`APP-1A` (depends on `APP-1` and `LEDGER-1`) — prove traceable
   BookRAG-lite search before live graph/retrieval expansion:** install the real
   wheel and run one qualified CPU-native parser through canonical evidence,
   BookRAG `T`, BM25 plus Qdrant dense candidates, the candidate contract and
   the hydrator. The CLI
   and minimal `/v1` document/job/search/evidence handlers call the same use
   cases. Every hit resolves to immutable source bytes plus page/bbox and one
   durable operator trace; `G/M`, pixels and answering remain typed unavailable
   capabilities. Repeated and kill/resume runs produce the same IDs and lose at
   most one active microbatch. Query scope binds a constant-size catalog root
   per selected document rather than copying complete canonical material-ID
   rosters. Public search/query JSON is capped at 1 MiB per request and 8 MiB per
   response; internal candidate-worker JSON is capped at 16 MiB per direction,
   while crops/raw artifacts use authorized streaming/CAS endpoints.
4. **`MODEL-PORT-1` (depends on `APP-1`) — make model replacement executable:**
   introduce versioned `LanguageModel`, `VisionModel`, `EmbeddingModel` and
   `Reranker` request/result ports plus shared timeout, retry, cancellation,
   admission, lifecycle, trace and material-use decorators. Ship a fake
   conformance adapter and one qualified remote adapter through
   `docarag.models.v1`; an independently built replacement wheel must install,
   pass `doctor`/contract/golden replay, and uninstall without changing BookRAG
   or application code. Live KG/model calls and optional `ask` use only these
   ports.
5. **`EXTRACT-1` (depends on `APP-1`) — contract and compatibility handoff
   implemented; live qualification open:** Docling, MinerU, and Paddle explicit
   parent/child, caption/figure, footnote, references, adjacent reading order,
   and parser-attested table continuation now survive into BookRAG tree metadata
   with exact assertion lineage. Typed table-cell, bounded MathML-symbol and
   directed explicit diagram/flow assertions now survive parser observations,
   conservative source-neutral comparison, reconciliation, BookRAG tree
   storage, hydration, and compact vector receipts. Keep the exact raw artifact, representation, asset,
   page/bbox and source-observation anchors. Missing or failed visual structure
   is a typed degradation, never an empty success.
6. **`RETRIEVE-1` (base expansion depends on `APP-1A` and `EXTRACT-1`; graph
   legs depend on adopted `GRAPH-ADOPT-1`) — candidate/RRF contract implemented;
   live non-lite adapters open:** put the dependency-neutral candidate boundary
   in front of the current dense, entity, TreeVDB and section stores. Reject
   low-confidence entity seeds, apply exact access/version/readiness filters to
   every leg, run independent legs in parallel, deduplicate by evidence
   identity, and record every contribution, rejection and latency under one
   request budget. Candidate material links are provisional until hydration
   revalidates the exact scope, revisions and bytes. Scope construction is
   `O(selected documents)` and never enumerates all canonical material IDs; a
   catalog-root membership lookup or proof validates each returned hit.
7. **`GRAPH-BUILD-1` (depends on `APP-1`, `MODEL-PORT-1` and `EXTRACT-1`) —
   contract, bounded legacy builder, and immutable publication implemented;
   exact live invocation open:** keep BookRAG `B=(T,G,M)` as the serving core and
   add its evidence-grounded directed-multigraph sidecar. Preserve pre-repair
   extractor assertions and publish immutable mentions, directed relation
   assertions and candidate GT-Links without inventing aliases. The current
   legacy-checkpoint builder has only coarse whole-node selections; its releases
   are useful for audit and shadow comparison but are not authority-eligible.
   Add an exact per-invocation `MaterialUseReceipt` plus an assertion-specific
   `AssertionEvidenceReceipt` that binds each assertion digest to the exact
   consumed span/cell/region/asset subset and a semantic-support verdict. Reuse
   the builder's verified CAS materialization receipts and prove which assets
   actually reached the model before marking a release eligible.
8. **`LITE-1` (depends on `APP-1A`) — graduate the first honest `lite` profile:** CPU
   Docling, PostgreSQL and single-node Qdrant, with explicit
   degradation for scans and complex visual pages. Endpoint-free lite must pass
   offline-preflight -> explicit-model-pull -> up -> live-doctor -> index ->
   search without NVIDIA, source edits or manual environment surgery. The two
   loopback storage services start through the supported Compose command and
   retain named volumes across `down`.
   `ask` is capability-gated: it is available only after `doctor` qualifies an
   explicitly configured answer endpoint, otherwise it returns typed
   `capability_unavailable` and never pretends retrieval is generation.
   Complete the small `/v1` API and make the Web UI a client of it; do not build
   a second indexing/query path. Optional `ask` also depends on
   `MODEL-PORT-1` and a qualified answer endpoint.
9. **`INGEST-EXT-1` (depends on `APP-1` and `EXTRACT-1`) — add non-PDF source
   fidelity without forking BookRAG:** translate DOCX, PPTX, XLSX, HTML, images,
   and email plus attachments into the same canonical observation contract.
   Preserve native part/slide/shape/sheet/cell/DOM/message and attachment IDs,
   source byte digests, coordinates and relationships. Contract and golden tests
   must resolve every canonical object and citation back to the immutable native
   source; format adapters stop at the canonical boundary and do not implement a
   second tree, graph or retrieval path.
10. **`GRAPH-ADOPT-1` (depends on `GRAPH-BUILD-1` and `LEDGER-1`) — make graph
   authority explicit:** validate release identity, exact invocation/material
   and assertion-evidence receipts, semantic verdicts, and an adopted full-tree
   manifest binding root, topology, sibling order, relevant node content,
   source/canonical revisions, tree revision and descriptor digest. Validate
   integrity counts and adoption
   fencing, then atomically record the eligible `G/M` release in the revision
   ledger. A coarse legacy release is rejected; retrieval graph/entity legs may
   open only for the adopted digest.
11. **`CLAIM-VERIFY-1` (depends on `APP-1`, `MODEL-PORT-1` and live
   `RETRIEVE-1`; graph-derived
   claims also depend on `GRAPH-ADOPT-1`) — add immutable transitive claim
   lineage and semantic verification:** Map claims bind exact hydrated evidence
   and model-use receipts; Reduce claims bind the ordered immutable digests of
   the child claim artifacts actually supplied to that Reduce call. Verify
   semantic support at every Map-evidence and Reduce-child edge and publish a
   claim as supported only when its complete transitive closure reaches the
   original immutable source. Consumption receipts alone never establish
   entailment.
12. **`PARSER-QUAL-1` (depends on `EXTRACT-1`) — qualify the implemented service
   clients and worker profiles** against pinned
   Docling, MinerU and complete PaddleOCR-VL images, then ship the digest-pinned
   qualification records. Run
   the modality-labelled golden corpus and record package/model/image identity,
   page limits, latency, accuracy, setup effort and peak/idle VRAM.
13. **`WORKSTATION-1` (depends on `APP-1A`, `MODEL-PORT-1`, `LEDGER-1`, live
   `RETRIEVE-1` and `PARSER-QUAL-1`) — ship
   the managed workstation profile** with owned supervision, explicit service
   ownership and a GPU capacity lease.
14. **`VECTOR-SCALE-1` (depends on live `RETRIEVE-1` and `LEDGER-1`) — qualify
   and scale the existing Qdrant backend.** Qdrant is already the sole/default
   vector database with one unnamed dense vector. After measured recall,
   latency and corpus-size gates, add the PostgreSQL outbox, sparse and named
   visual vectors, dense+BM25 RRF, bounded reranking, quantization and, only
   after single-node saturation, sharding/replication. Tune fusion and model
   size only from held-out labels.

These backlog IDs and dependencies are authoritative. The thematic phases and
historical PR notes below must reference them; they may split an item into
smaller reversible releases but may not reorder or silently add a prerequisite.

Defer trained routing, distributed Qdrant, 8B visual defaults, Kafka and
Kubernetes until these gates show a measured need. This order improves setup,
recoverability and evaluation truth before adding more infrastructure.

## 2. What the current implementation tells us

### 2.1 Confirmed remaining scale and reliability defects

- Tree summaries and KG extraction now commit source/model/prompt-bound checkpoints, but `Core/Index/Tree.py` still publishes the complete mutable object graph only at the end. A crash can therefore repeat non-checkpointed tree/refinement work even though completed model calls are reusable.
- The MinerU positional merge has been replaced by source-bound, page-local reconciliation with bounded candidate/probe work, explicit disagreement status, stable prior-revision identity, and cross-parser lineage. The generic routed orchestration path can fuse all three adapter results, the lazy page probe can supply source-bound routing features, and the provider can persist/reuse a staged, rollback-protected ready pair. Strict service clients now cover all three documented vendor surfaces and publish verified local artifacts, but the supported compatibility index command still uses the legacy MinerU runtime because routed settings/index composition, service supervision, and live qualification are not shipped.
- MinerU now retains each task's validated `middle.json` as one result-scoped,
  digest-bound receipt (including an explicit `not_emitted` outcome), so an
  empty legacy content list no longer loses it or a large list duplicate it.
  The middle payload is still decoded eagerly under a 128 MiB default bound.
  Paddle retains the exact HTTP response and decoded crops under one
  request-wide artifact budget and returns one exact raw-response receipt for
  every successful page, including a page with no recognized blocks. This
  intentionally accounts for both Base64 response bytes and decoded images,
  and response JSON decoding remains eager under byte/depth/node limits. A
  shared 4,096-receipt ceiling now bounds parser results and MinerU task count.
  If a later service/sink operation fails after earlier CAS writes, those
  earlier objects still need transactional publication or orphan collection.
  Streamed native-artifact translation remains a scale follow-up.
- The application extraction contract now verifies stable, bounded canonical
  and routed-audit files, compares their exact bytes and semantic digests, and
  returns resolvable receipts for both. A capped reader prevents concurrent
  growth from making JSON parsing unbounded, but both envelopes are still
  eagerly decoded under a 512 MiB default limit. Replace the compatibility
  envelopes with a streamed manifest plus sharded/JSONL blocks before claiming
  predictable peak memory for very large documents.
- All three adapters now emit only explicit, bounded structural relations and semantic subgraphs, and `CanonicalBlock` preserves them through reconciliation and the BookRAG tree handoff. MinerU/Paddle unresolved, ambiguous, invalid, self, container-only and disallowed-cross-page refs remain audited instead of guessed; reading order is adjacent and costs `O(B log B + R)`, never all-pairs. Exact assertion digests remain distinct from conservative source-neutral comparison digests; equivalent vendor tables can share one representative without laundering the other parser's selectors, while changed content/topology remains review-gated. Full graphs are stored once in the tree; vector metadata contains bounded ID/digest receipts. Live MinerU/Paddle rich-field qualification and live evidence-sidecar construction/invocation remain open; immutable checkpoint publication itself is implemented. Do not compensate by asking the knowledge-graph LLM to reconstruct lost layout from flattened text.
- `pdf_refiner.py` still treats some text/table judgment failures as a logged fallback. These quality degradations need typed stage issues and policy-controlled retry/review rather than an implicit best-effort result.
- The BookRAG tree vector path is now microbatched, but some baseline/RAPTOR and offline multimodal-evaluation paths still materialize whole chunk/embedding collections.
- `Core/rag/gbc_utils.py::enhance_graph_with_semantic_links` still constructs a dense pairwise cosine-similarity matrix below its new 512-node guard. The guard prevents runaway scopes, but the allowed path remains quadratic and should move to offline ANN links.
- `Core/rag/gbc_retrieval.py` now scans the chosen subtree once, combines positive lexical matches with deterministic structural coverage, and caps **that leg's retained reranker input** in `O(text bytes + N_S log M)` time and `O(M)` retained-candidate memory. This does not make the complete query `O(M)`: `gbc_rag.py` first materializes the selected subtree, `gbc_utils.py` copies the induced graph, PageRank visits that selected graph, and graph candidates are sorted before fusion. The current overall bound is therefore approximately `O(N_S + L_S + I_S(V_S+E_S) + N_G log N_G + text bytes + N_S log M)` time and `O(N_S+V_S+E_S)` memory, with a guarded optional `O(V_S^2)` semantic-link matrix. The two-score graph/text Skyline itself is deterministic, hard-capped, and `O(M log M)`; candidate generation is still tree/graph scoped rather than hybrid retrieval.
- With `candidate_pool_size=512` and remote rerank batches of four, the current sequential path can issue 128 rerank calls for one sub-query. At the configured 30-second per-attempt timeout that is a theoretical 64-minute timeout envelope before any outer decomposition, not an acceptable request deadline. Item caps must be supplemented by a single propagated request deadline, call budget and token/pixel budget.
- `Core/provider/vdb.py` now uses deterministic representation IDs,
  source-bound strict IDs for tree indexing, `upsert`, exact
  payload-scope-filtered deletion, and mandatory tenant/corpus/document/source/
  index filters. The tree-vector builder reconciles stale rows only inside that
  bound scope in the shared named collection, publishes a run-local exact-tree/
  config/embedding Qdrant readiness attestation after cleanup, and MMR holds a
  shared read lease that excludes rebuild. The attestation plus exact scoped
  count does not detect same-count remote vector/payload mutation. Incremental
  transactional outbox/diffing, remote content reconciliation, immutable visual
  byte handoff, served-weight attestation, and multi-node scale remain open.
- Provider calls now have process-wide admission and lifecycle cleanup. Compatible GME handles share a model/device-keyed, reference-counted runtime, and one forward pass is admitted at a time so concurrent requests cannot multiply its transient GPU memory. The active configuration can still place MinerU, Ollama LLM/VLM, a local reranker, and GME on one GPU without a capacity-aware cross-process lease. The observed process mix reached roughly 43.3 GiB of a 48.9 GiB GPU while idle.
- `Core/Index/Graph.py` uses `networkx.Graph`, so relation direction and multiple distinct relations between the same entity pair cannot be represented robustly.
- The new evidence-release builder preserves every row and ordinal it receives,
  but `KGExtractor.res_repair` currently aggregates same-name/type entities and
  collapses relations on `(source, target, relation_name)` before the checkpoint
  is written. Directed/parallel DTOs prevent further loss; they cannot recover
  assertions already removed upstream. Publish both the immutable pre-repair
  extractor output and a separately identified repaired view before claiming
  raw assertion fidelity.
- Graph `source_ids` still contain document-local integer tree IDs. The referenced tree nodes now carry canonical/derived source lineage, but integer IDs still collide across documents and are not sufficient as the durable corpus-level GT-Link authority.
- Relation edges are now serialized into a bounded answer-facing graph context, but the underlying undirected single-edge graph still collapses direction and multiple relation assertions.
- The multimodal reranker in `Core/rag/gbc_retrieval.py` is commented out. Images must first survive entity/section selection and text reranking based mostly on captions.
- Long evidence is split for reranking and only the first chunk is scored. This
  creates a systematic tail blind spot for long sections/tables. Replace it
  with query-aware bounded chunk candidates (plus heading/path context), retain
  the winning chunk/span in the trace, and evaluate long-evidence recall before
  changing the reranker model.
- Tree-vector image text currently relies on caption/footnote-style fields and
  does not consistently index canonical OCR, visual descriptions or diagram
  labels. Project those exact selected textual representations into separate
  candidate fields/legs before enabling the visual reranker; otherwise image
  recall depends on whether a caption happened to contain the query term.
- `CanonicalEvidenceHydrator` validates retrieved tree records against a bounded source-bound canonical artifact and emits citation IDs. Asset metadata remains `reference_only`; strict VLM use withholds it. A separate descriptor-safe local/CAS resolver verifies bounded bytes, but it is deliberately not wired into generation because reopening a mutable pathname would reintroduce TOCTOU. Final answers now have fail-closed structural binding validation, not semantic claim-support verification.
- `CanonicalEvidenceHydrator` still reads and indexes the complete bounded
  canonical JSON artifact, and each CLI `ask` constructs/loads then closes the
  BookRAG index and model stack. For hot-query latency, add an indexed/lazy
  evidence catalog and a long-lived application query service with explicit
  owned/borrowed caches before tuning model kernels. CLI and Web UI become
  clients of that service rather than lifecycle owners per question.

### 2.2 Version and integration risk

The repository pins `mineru==2.1.11` and imports MinerU internal modules directly. The benchmark environment currently reports MinerU `3.4.4`, Docling `2.112.0`, and PaddleOCR `3.7.0`. MinerU has changed backends and intermediate/output schemas across major releases; its official changelog explicitly records incompatible VLM and `middle.json`/`content_list.json` changes ([MinerU changelog](https://opendatalab.github.io/MinerU/reference/changelog/)).

Therefore:

- No DocARAG domain code may import parser internals.
- Each adapter must consume a documented CLI/API output and translate it into a versioned DocARAG schema.
- Every parse record must store parser name, package/model version, configuration hash, and raw-artifact URI.
- Parser upgrades require replaying the parser conformance suite before production rollout.

### 2.3 Maintainability and setup defects

The current repository is a research implementation, not yet an easy-to-operate product:

- The legacy `requirements.txt` installs Torch, Transformers, MinerU, vision utilities, NLP packages, and evaluation dependencies into one environment. This maximizes resolver, CUDA, import-time, and upgrade coupling. The dependency-light controller now keeps PostgreSQL/Qdrant clients separate from parser/model runtimes, but the remaining compatibility environment still needs decomposition and locking.
- The dependency-light domain/port/application layer no longer imports parser vendors. Docling, MinerU-normalized, and PaddleOCR-VL outputs translate through the same source-bound contract. Standard-library clients now target Docling Serve async conversion, MinerU's task API, and PaddleOCR-VL's complete pipeline; a versioned parser-worker transport and immutable local CAS provide the same-host isolation boundary. The supported compatibility parser runtime still invokes MinerU internals in-process because routed `docarag index`, owned service lifecycle, pinned images, and live qualification are open.
- The supported CLI no longer applies the old `main.py::process_resource` URL/GPU overrides, but legacy YAMLs and helper scripts still contain fixed GPU indices, endpoints, Conda names, run IDs, and developer paths. They must not be mistaken for portable presets.
- `docarag index` now hides generated dataset JSON/YAML for a single PDF, but the compatibility runtime still depends on the legacy system/dataset configuration shapes and overlapping shell entry points remain.
- The Web UI loads storage and RAG implementation objects directly. There is no versioned public API, idempotency contract, consistent error envelope, or capability discovery endpoint.
- LLM/VLM providers now have typed requests, bounded admission/retries, streaming, and cleanup, but still combine configuration, client creation, lifecycle and vendor invocation. Model adapters and shared runtime policies remain to be separated behind ports.
- The compatibility driver now establishes source/config/code revision identity, immutable source and config snapshots, `run-manifest.json`, attempt-local state/logs/runtime fingerprints, atomic latest state, a role-aware artifact manifest, selective query-loaded-artifact hashing, deep source/output verification, global indexing admission, and signal forwarding. All discoverable installed distributions, `CUDA_VISIBLE_DEVICES`, and GPU/driver identity are attempt provenance rather than revision identity; complete environment locks and remote model/container digests remain unavailable. Inside the BookIndex, pickle and graph JSON remain mutable compatibility formats; Qdrant points are rebuildable derivatives with a scope-bound readiness attestation and exact count, not a remote content proof or durable canonical `T/G/M` repository.
- A dependency-light `pyproject.toml`, checksummed PostgreSQL migration policy, parser contract suites for all three adapters, deterministic router/orchestrator/reconciliation/projector fixtures, strict configuration round trips, and architecture-boundary tests now exist. There is still no complete package lock, clean-host real-worker suite, database backup/restore gate, or supported model/parser compatibility matrix.
- The wheel includes `docarag*`, and its entry point plus help, settings, and
  storage command parsing pass a clean-wheel test with all checkout-owned
  `Core` and PyYAML imports blocked. Compatibility `status` execution,
  named-run verification, `index`, `ask`, and `web` still discover `Core`,
  `Scripts/index_pdf.sh`, `Scripts/web.py`, or other checkout resources at
  runtime. The release must either package those application services and
  resources cleanly or make a separately versioned BookRAG compatibility
  checkout an explicit, validated dependency with an actionable error.
- There is no supported CPU/lite golden path. `settings` has only the
  compatibility profile and `doctor` probes the heavyweight BookRAG package
  set, so a future lite install cannot honestly pass today. Profile-specific
  capability manifests and doctor requirements must land before claiming easy
  setup.

The 2026-07-15 maintainability audit drove an implemented compatibility slice.
The current truth, including the remaining release blockers, is:

- LLM/VLM/embedding/reranker settings are now backend-specific and validated
  before allocation. Endpoint, API key, sampling/output, device, timeout,
  keep-alive, and reranker flash-attention policy reach their selected backend;
  unsupported combinations fail explicitly. Remote LLM/VLM attempts pass a
  finite 300-second request timeout, the remote reranker passes 30 seconds,
  OpenAI-compatible SDK retries are suppressed, and DocARAG's explicit attempt
  budget is constrained to `1..10`. These are per-attempt client timeouts, not
  overall wall-clock deadlines. Cancellation and retry classification are not
  yet uniform across every legacy parser/index stage.
- Ollama no longer requests indefinite residency by default. LLM/VLM requests
  default to `keep_alive="5m"`, and `unload_on_close=false` leaves shared server
  state alone. Explicit opt-in unload sends `keep_alive=0`, but no cross-process
  model-ownership lease proves that another user is not sharing that model.
  Project-owned parser/model-service `up`/`down`, vLLM/Paddle sleep, and
  capacity-aware GPU lifecycle remain open; PostgreSQL/Qdrant `storage
  up/down` is already implemented.
- The in-process Qwen VLM uses the loaded model's execution device, honors the
  configured output/sampling limits, and implements the structured operation.
  Local-only ML imports are lazy. A guarded subprocess suite exercises injected
  remote embedding/reranker/LLM/VLM paths without importing Torch,
  Transformers, ModelScope, or FlashAttention. This is an import-boundary test,
  not a clean-host install or real-model qualification result.
- `TokenUsageScope` and `RunLogScope` use context variables for request/run
  isolation; the bounded executor copies the context into each worker thread.
  File handlers attach only to named DocARAG loggers, overlapping parent/child
  logger namespaces are deduplicated, and concurrent tests prove one event and
  token ledger do not leak into another request. Unscoped legacy calls retain a
  locked global token fallback, so supported service entry points must continue
  establishing scopes.
- `RAGSession` makes dependency ownership explicit: a caller either transfers
  an owned agent or borrows it, and construction/query/close failures are tested
  for exactly-once cleanup without hiding the active error.
- Planner outputs have hard query, entity, filter, sub-question, value, and
  prompt limits. Section membership is `O(S + L)`; every linked section is
  retained or the request fails visibly, while supplementary top-`M` candidates
  combine lexical/path relevance with source-stratified coverage in
  `O(S * (D + log M))` time. Independent complex sub-questions still execute
  sequentially; bounded parallel execution and complete deadline/queue traces
  remain open.
- Batch reuse is no longer ordinal. A SHA-derived query directory is reusable
  only when exact UTF-8 question bytes, verified published artifact-set,
  secret-free resolved config/model, and current first-party query-code
  identities match a digest-bound commit marker. Corrupt or absent bindings
  recompute. JSONL/NDJSON and generators stream, DataFrame rows avoid
  `to_dict(orient="records")`, and the final JSON array is privately and
  atomically published without retaining a result list. Legacy JSON-array input
  and the `main.py` pandas dataset remain eager; pre-identity caches are not
  migrated; caches have no GC and assume one writer per output directory.
  Neither quick nor deep verification scrolls and re-hashes remote Qdrant
  points: the run-local attestation plus exact scoped count cannot detect
  same-count vector/payload tampering. A per-scope digest, outbox reconciliation,
  or explicit remote scroll audit is still required, and mutable remote weights
  can also hide behind an unchanged configured model identity.
- Document assignment now uses length-prefixed SHA-256 over document UUID/path,
  so reorder or corpus extension does not move a document for a fixed shard
  count. `main.py` still reads and groups the entire dataset in pandas, so this
  is stable assignment rather than a durable streaming work queue.
- The operational revision/attempt/publication state machine remains largely in
  the compatibility shell driver. It still needs one typed, migration-tested
  application service while the script becomes a thin launcher.

The system should remain a **modular monolith controller with isolated
resource-conflicting workers**. BookRAG domain, provenance, routing policy, and
orchestration stay together; only CUDA/Torch/Paddle/Transformers/vLLM or
model-lifecycle conflicts cross the versioned transport. A microservice per
stage would add deployment and distributed-debugging cost without improving the
BookRAG algorithm. The current generic worker transport and strict service
clients implement this boundary without claiming that the unpinned, simulated
deployments are production-qualified. Local `cas://` and source-material
bindings require the worker and controller to share explicitly configured
filesystem mounts; they are not a multi-host artifact protocol.

### 2.4 What should remain

- BookRAG's complete `B = (T, G, M)` BookIndex is the core retrieval structure, not a legacy format.
- The document tree is the document-native hierarchy and the scope selector for contiguous evidence patches.
- The knowledge graph supplies information scent and multi-hop relations; GT-Links ground graph results back in source nodes.
- The Formulator, Selector, Reasoner, and Synthesizer operator families remain the public retrieval semantics.
- Existing QASPER, MMLongBench, and M3DocRAG evaluation entry points are useful foundations.
- Qwen embeddings/rerankers are credible candidates, but model size and vector dimensions must be chosen from DocARAG measurements rather than leaderboard position alone. The Qwen3 family provides 0.6B/4B/8B embedding and reranking variants ([Qwen3 Embedding report](https://arxiv.org/abs/2506.05176)).

### 2.5 How BookRAG is used in the target system

The local [BookRAG paper](BOOKRAG_VLDB_2026_full.pdf) defines two construction stages and an operator-driven query path. Each is retained and extended:

| BookRAG concept | Current DocARAG implementation | Target implementation |
|---|---|---|
| Layout parsing into typed blocks | MinerU internals in `extract_pdf_info.py` | Routed Docling/MinerU/Paddle observations reconciled into canonical multimodal blocks |
| Section filtering and tree construction | `outline_extractor.py`, `doc_tree_builder.py`, `Tree.py`; lossless source-order construction is now `O(B + T)` and invalid outline parents fail/fall back explicitly without dropping blocks | Versioned tree nodes/edges with parser and LLM decisions recorded as lineage |
| Modality-dependent KG extraction | `kg_extractor.py` uses LLM for text/table/formula and VLM for image | Typed extractors for text, table, formula, figure, diagram, flowchart, sketch, scan OCR, and text-in-image |
| Gradient-based entity resolution | `kg_refiner.py` vector blocking + reranking + LLM adjudication | Preserve algorithm, but keep mention entities and versioned non-destructive canonicalization decisions |
| GT-Link `M` | entity `source_ids` and `tree2kg` | Explicit many-to-many `gt_links` rows with document/node/revision/provenance |
| Agent planning | `gbc_plan.py` simple/complex/global plans | Preserve and add document selection plus evidence-modality requirements |
| Selector operators | entity and section selection; global type/range filters | Preserve; execute over durable tree indexes and add hybrid/document selection |
| Graph/Text reasoning + Skyline | `gbc_retrieval.py`; full-subtree lexical/structural preselection with bounded reranker input and deterministic hard-capped `O(M log M)` two-score Skyline | Preserve scores, replace query-time all-pairs graph expansion with offline ANN, and add measured visual/table/formula fusion |
| Map/Reduce synthesis | `gbc_answer.py` | Preserve, but hydrate and cite original evidence objects, including crops and structured table/formula data |

BookRAG readiness is a distinct state. A document can be `BASE_SEARCHABLE`
after canonical blocks and the deployment profile's declared candidate indexes
are committed (the build specification declares the required indexes; an
endpoint-free lite build may be BM25-only with an explicit degraded reason),
then become `BOOKRAG_READY`
only after its tree, KG, entity resolution, and adopted GT-Links pass integrity
checks. This provides low time-to-search without pretending that base search is
full BookRAG.

Visual availability is not a later linear readiness state. An optional
`visual_capability_revision` is an orthogonal, immutable attachment to either a
`BASE_SEARCHABLE` or `BOOKRAG_READY` index revision. It binds the exact visual
assets, vector/extractor schema, model/config digests and integrity result. A
request that needs pixels may use only that exact qualified attachment; when it
is absent, the planner returns a typed capability degradation or fails the
evidence requirement. It never infers “visual ready” merely from the main
readiness label.

`INDEXING_PARTIAL` is a separate, opt-in preview state, never an active document
version. It carries an immutable indexed-page watermark, an incompleteness
warning and the exact provisional revision. Preview search is restricted to
pages at or below that watermark and may not return an “unanswerable” or
whole-document conclusion. Normal `search`/`ask` resolves only an atomically
committed `BASE_SEARCHABLE` or `BOOKRAG_READY` version; the active-version
pointer never moves page by page.

## 3. Target architecture

```text
                         CONTROL PLANE (CPU, no ML imports)
 upload -> fingerprint -> document version -> stage DAG -> leased work items
                                      |                 |
                                      |                 +-> status/events/metrics
                                      v
                         DATA PLANE (bounded workers)
 PDF -> page probe -> route -> parser adapter(s) -> parser block observations
                                      |
                         reconcile by page/bbox/stable ID
                                      |
                       canonical multimodal evidence
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
              v                       v                       v
       Tree T=(N, E_T)       Knowledge Graph G=(V, E_G)   candidate index port
   sections + typed leaves      entities + relations      text/BM25/visual
              |                       |
              +---------- GT-Link M --+
                                      |
                          BOOKINDEX B = (T, G, M)
                                      |
 query -> BookRAG planner -> Formulator -> Selector -> Reasoner -> Synthesizer
              |                 |            |             |
              +-> hybrid/document/entity/section/modal/range selection
                                -> graph/text/visual/table/formula scores -> Skyline
                                -> hydrate original evidence -> Map/Reduce -> cited answer

 metadata/work/trace/index authority <-> PostgreSQL
 raw/canonical artifacts <-> object storage
 candidate vectors <-> Qdrant
```

### 3.1 Storage roles

Use different systems for different consistency and access patterns:

| Data | Local/single-host mode | Horizontally scaled mode | Rule |
|---|---|---|---|
| Documents, versions, artifact receipts, index revisions, jobs, stage events | PostgreSQL | PostgreSQL HA/managed service | Required system of record; transactions, append-only trace, leases, attempts and fencing |
| PDFs, page renders, crops, parser JSON, Markdown | Local content-addressed files | S3/MinIO-compatible object store | Immutable, addressed by hash |
| Dense, sparse, and visual candidates | Single-node Qdrant | Qdrant cluster after single-node load-test saturation | Sole vector database and rebuildable derivative; the current implementation uses one unnamed dense vector, while sparse/named visual vectors remain measured extensions |
| Logs, traces, metrics | JSONL + Prometheus locally | OpenTelemetry-compatible backend | Never infer job success only from logs |

PostgreSQL is required in every supported profile; there is no alternative
relational fallback. The implemented migration and repository already cover
tenant/corpus/document versions, immutable artifact receipts, guarded index
revisions, append-only events and idempotent leased work. Current CLI indexing
uses the document and dispatch-event subset; full page/block and `T/G/M`
adoption remains work. The local summary checkpoint database is only a
non-authoritative recomputation cache and can be deleted and rebuilt.

The first migration must be an **adoption ledger**, not a second canonical
payload store. It registers and re-hashes the existing immutable source,
canonical, audit and BookIndex artifacts; records their selectors and component
digests; and keeps `document_version_id`, `build_spec_id`, and
`bookindex_revision_id` distinct. It binds current `T` to the verified tree
artifacts, `G` to the graph selector, and `M` to the `tree2kg` selector even
when `G` and `M` share one physical JSON artifact. It also publishes one
`CanonicalCatalogRevision` per document version: a deterministic root digest,
record count, schema revision and artifact/store selector over the canonical
block, representation and asset identities. Query scopes carry that constant-
size root binding, never the complete material-ID roster. A local authoritative
store proves membership with a revision-bound point read; a remote/untrusted
catalog additionally returns a bounded Merkle proof. Hydration still re-reads
and hashes the selected bytes. Normalized block/node rows can be introduced
transactionally after migration/read requirements justify them. Until then,
SQL stores immutable references and integrity metadata rather than copying
`canonical_blocks.json` and creating two authorities.

Represent lineage with the same conceptual separation as W3C PROV: immutable
source/canonical/model artifacts are entities, parser/index/query executions
are activities, runtimes/operators are agents, and every derived record has an
explicit derivation edge. This is a mapping discipline, not a requirement to
store RDF, and it keeps the durable ledger understandable across adapters
([W3C PROV-DM](https://www.w3.org/TR/prov-dm/)). OpenTelemetry trace IDs may
join those records, but baggage must carry only opaque validated identifiers;
it has no built-in integrity guarantee and can propagate to downstream
services ([OpenTelemetry baggage](https://opentelemetry.io/docs/concepts/signals/baggage/)).

### 3.2 Durable stage model

Make indexing a versioned DAG rather than three monolithic commands:

```text
fingerprint
  -> render_page -> probe_page -> parse_page -> reconcile_page
  -> build_structure -> chunk_block -> index_text
  -> [describe_visual] -> [index_visual]
  -> [summarize_node] -> [extract_entities] -> [merge_entities]
```

Each work item must include:

```text
stage, document_version_id, item_key,
input_hash, code_version, config_hash, model_hash, prompt_hash,
status, priority, attempt_count,
lease_owner, lease_expires_at, next_attempt_at,
result_uri, error_class, error_message, created_at, finished_at
```

Required semantics:

- Atomic claim with a renewable lease.
- At-least-once execution and idempotent writes.
- Deterministic output IDs derived from document version + stage + item key + configuration hash.
- Exponential backoff with jitter for retryable failures.
- Permanent failure and dead-letter states; never convert errors into content.
- Dependency counters so a parent summary is eligible only after all required child results are committed.
- A transactional outbox for Qdrant upsert/delete operations.
- Stage invalidation by hash: a prompt change reruns summaries, not parsing; a parser change invalidates blocks and all descendants.

### 3.3 Canonical document schema

The canonical schema is the most important parser abstraction. At minimum:

```text
DocumentVersion(id, document_id, content_sha256, mime_type, page_count, metadata)
Page(id, document_version_id, number, width, height, rotation, image_uri, native_stats)
BlockRevision(id, block_id, page_id, parent_block_id, type, semantic_subtype,
              bbox, polygon, reading_order, confidence, provenance)
Representation(id, block_revision_id, kind, content_or_artifact_uri,
               producer_revision, confidence, provenance)
Asset(id, block_revision_id, digest_uri, sha256, media_type, geometry, provenance)
CanonicalCatalogRevision(id, document_version_id, root_sha256, record_count,
                         schema_revision, artifact_or_store_selector)
StructuralRelation(id, source_block_revision_id, target_block_revision_id,
                   type, direction, confidence, assertion_provenance)
MaterialSelection(id, consumer_revision_id, block_revision_id,
                  selected_representation_ids, selected_asset_ids, purpose)
Chunk(id, document_version_id, block_ids, section_path, text, token_count, context)
Node(id, document_version_id, parent_id, type, title, depth, source_block_ids)
```

`BlockRevision` never has singular authoritative `text`, `latex`, `html`, or
`asset_uri` columns. Those are flattened compatibility/export views over the
normalized representation and asset rows. `MaterialSelection` records what a
specific tree/KG/vector/answer consumer actually used, separately from all
material available on the block.

`Block.type` should cover at least title, paragraph, list item, table, equation, figure, chart, diagram, code, caption, footnote, header, footer, and unknown. Keep parser-native raw JSON in object storage, not in the normalized row.

Provenance must include parser/model version, source page, bbox or polygon, parser-native ID, raw artifact reference, and transformations applied. Merge only by a stable native ID or page-local geometric/text matching. Never repair a count mismatch by globally pairing arrays by index.

Docling already represents texts, tables, pictures, groups, body/furniture, and reading order in a unified document structure ([Docling document model](https://docling-project.github.io/docling/concepts/docling_document/)). MinerU and PaddleOCR-VL expose typed page/block output with bboxes and content ([MinerU output format](https://opendatalab.github.io/MinerU/reference/output_files/), [PaddleOCR-VL pipeline](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)). Preserve those common capabilities instead of reducing each parser to Markdown.

### 3.4 One format for all parsers

The parsers can work together only if DocARAG distinguishes **parser observations** from **canonical evidence**. A parser never writes a BookRAG tree node directly.

```text
ParserBlockObservation
  observation_id
  document_version_id, page_id
  parser_name, parser_version, model_version, config_hash
  parser_native_id
  type, semantic_subtype
  bbox/polygon in normalized page coordinates
  reading_order
  representations[]
  assets[]
  relations[]
  parser_confidence
  raw_artifact_uri

CanonicalBlock
  block_id, revision
  selected/fused type and geometry
  representations[]
  assets[]
  source_observation_ids[]
  reconciliation_method, quality_status
```

The BookRAG compatibility projection records both
`canonical_{representation,asset}_ids` (everything retained) and
`selected_{representation,asset}_ids` (the exact material used to construct the
node view). Refinement unions these source-scoped selections instead of silently
leaving the first block's IDs on merged content. Retrieval hydration validates
both sets against the canonical artifact before returning evidence.

All content types use the same envelope but carry modality-specific representations:

| Content | Required representations retained together |
|---|---|
| Native text | Unicode text, spans/words with bbox, reading order, language |
| Text in image / scan | OCR text and word boxes, page/region crop, OCR language/confidence |
| Table | Cell grid with row/column spans, HTML, Markdown/CSV serialization, caption/footnote, table crop |
| Formula | LaTeX or MathML, normalized text, formula crop, surrounding text |
| Figure/image | Original extracted asset or crop, caption, nearby references, OCR text, visual description |
| Chart | Chart crop, title/legend/axis OCR, optional structured series/data table, description |
| Diagram/workflow/flowchart | Crop, OCR labels, visual description, optional diagram nodes and directed edges |
| Sketch | Crop, OCR labels if present, description, detected components/relations with confidence |

Use a broad BookRAG node type such as `IMAGE` for compatibility and a `semantic_subtype` for `figure`, `chart`, `diagram`, `workflow`, `flowchart`, `sketch`, `screenshot`, or `scan_region`. New reasoner operators can use the subtype without breaking the original BookIndex definition.

### 3.5 How multiple parser results are combined

The router selects the cheapest likely parser, but fusion is region-aware:

1. Docling may provide the page's native text, headings, reading order, and native tables.
2. MinerU may replace or add a formula/table observation for difficult technical regions.
3. PaddleOCR-VL may add OCR, chart, diagram, and image-table observations for visual regions.
4. Reconciliation clusters observations only within the same page using native IDs where available, otherwise polygon IoU + type compatibility + normalized text similarity.
5. The immutable build policy selects or fuses representations. It never deletes the losing raw observation.
6. The resulting canonical block revision is the only object from which a BookRAG tree leaf is built.

Examples:

- A native table recognized well by Docling keeps Docling's cell grid and the source page crop. If Paddle finds additional text inside a rasterized cell, that OCR becomes another representation of the same block.
- A formula emitted as plain text by Docling and LaTeX by MinerU keeps both representations and the crop. Formula retrieval searches the normalized text/LaTeX but the answer stage can inspect the crop.
- A pitch-deck workflow page can retain Docling text, Paddle OCR labels, a Paddle visual description, and a VLM-produced directed diagram graph under one canonical visual node.

Conflicting observations produce a `NEEDS_REVIEW` or targeted reparse item when invariant checks cannot choose safely. There is no global positional merge.

### 3.6 End-to-end stage traceability

Every indexing and query action emits a trace span and a durable stage record.
Logs are supporting detail, not the source of state.

| Stage | Traceable input | Persisted output and required metrics |
|---|---|---|
| Ingest/fingerprint | Source URI/upload | document/version hash, MIME, size, page count |
| Render/probe | document version + page | page image hash, native-text stats, probe features, duration/RSS |
| Route | probe revision | chosen parser(s), page/regions, rule/model scores, reason, estimated cost |
| Parse | route + parser config | raw parser artifact, observations, model/package version, page latency, GPU peak, warnings |
| Reconcile | observation IDs | canonical block revision, selected/fused sources, quality checks, rejected candidates |
| Tree structure | canonical block revisions | tree node/edge revisions, heading decisions, orphan/cycle/order checks |
| Visual/table/formula enrichment | block revision | OCR/description/cells/LaTeX/diagram JSON, crop IDs, model invocation and confidence |
| Node summary | node + child revision set | summary revision, prompt/model hash, token/cost/latency |
| KG extraction | node revision + exact material selection | mention entities, relation assertions, evidence spans, extractor invocation plus request/material-use receipt; legacy rows are marked coarse whole-node |
| Entity resolution | mention/entity revision | candidates/scores, selected canonical entity, adjudication, alias decision |
| GT-Link | entity/relation + node revisions | explicit link rows, link role/confidence, integrity result |
| Vectorization | evidence revision + embedding config | deterministic point ID, vector kind/dimension/hash, outbox status |
| Readiness | all required checks | optional `INDEXING_PARTIAL` preview + page watermark, or committed `BASE_SEARCHABLE`/`BOOKRAG_READY`; independent `visual_capability_revision` and any typed degraded reason |
| Candidate retrieval | authorized index revision + query budget + plan | provisional refs, exact per-leg rank/status/latency/budget and rejection reasons; no evidence promotion |
| Hydration | provisional refs + evidence needs | verified evidence revisions/bytes plus hydration receipt, or typed integrity/capability failure |
| Map claim | exact hydrated materials + request | immutable claim artifact, direct material-use receipts, model/prompt/response digests and semantic edge verdicts |
| Reduce/verify | ordered child claim digests + request | immutable parent claim, child-use receipt, bounded transitive source closure and final semantic-support state |

Core trace tables:

```text
pipeline_runs, stage_runs, work_items, model_invocations,
artifacts, artifact_lineage, quality_checks, state_transitions,
node_revisions, entity_revisions, relation_revisions, gt_link_revisions,
vector_outbox, retrieval_traces, hydration_receipts, material_use_receipts,
claim_artifacts, claim_edges, semantic_support_verdicts
```

Every row carries `run_id`, `trace_id`, `document_version_id`, timestamps, code/config/model hashes, status, attempt, and error classification. `artifact_lineage(parent_artifact_id, child_artifact_id, transformation)` answers questions such as:

- Which parser observation created tree node 812?
- Which table crop and cell grid produced entity `Revenue FY2025`?
- Which entity-resolution decision merged two mentions?
- Which GT-Link caused a graph score to promote an image node?
- Which exact table/formula/image revision supported a sentence in the answer?

The Web UI should show a document/page/stage timeline, live completed/pending/retry/permanent counts, parser route overlays, tree revisions, entity/relation/GT-Link diffs, vector outbox lag, and per-query BookRAG operator traces. All views read the durable records rather than scraping log text.

### 3.7 Maintainable code and extension boundaries

Organize DocARAG as a modular monolith using dependency inversion. The domain layer must run its unit tests without importing Torch, Paddle, Transformers, Qdrant, FastAPI, or a parser package.

```text
src/docarag/
  domain/          canonical evidence, BookIndex T/G/M, policies, errors
  application/     ingest/index/query use cases and stage orchestration
  ports/           parser, model, store, vector, artifact, event protocols
  adapters/        docling, mineru, paddle, qdrant, postgres, s3
  api/             versioned HTTP schemas and handlers
  cli/             human-oriented commands calling application services
  runtime/         worker bootstrap, capability registry, health checks
tests/
  unit/ contract/ golden/ integration/ migration/ load/
```

Dependency direction is one-way: `api/cli/adapters -> application -> domain/ports`. Domain and application code never imports an adapter. Enforce this in CI with an import-boundary test.

The first vertical seam is implemented under `docarag/domain`, `docarag/ports`,
`docarag/adapters`, and the dependency-light routing/reconciliation/orchestration
modules. Its DTOs validate source hashes, page identity, coordinate spaces,
parser/model/config provenance, representations, assets, capabilities, typed
issues, and deterministic IDs without importing an ML or storage runtime. The
Docling, MinerU-normalized, and PaddleOCR-VL adapters emit the same observation
contract; the composition layer projects reconciled canonical blocks into the
legacy `pdf_list` consumed by BookRAG. Import-boundary, fake-adapter,
conformance, adversarial complexity, and projection tests are release guards.

A first application-layer use case now sits above that seam. It accepts an
already admitted local PDF snapshot, invokes a replaceable routed backend, and
returns a compact receipt only after the document/configuration identity,
accepted canonical block digest, canonical artifact schema, replay-audit
schema, exact file bytes, and mutual audit/canonical bindings all agree. The
receipt exposes resolvable URI/SHA-256/size tuples for both artifacts. This is
deliberately not presented as completed `APP-1`:
the supported CLI, Web UI and future API still need to call shared
`index/search/query/status/evidence` services, and snapshot admission plus a
durable work ledger remain separate work.

The stable boundary also contains immutable evidence-graph DTOs, a bounded
source/tree/material-bound builder, content-addressed publication, and a
loss-aware legacy `G/M` projection. These modules are dependency-light and
covered by the same import-boundary guard; the live KG stage has not adopted
them yet.

The seam now also includes a source-bound lazy PyMuPDF probe, a versioned
standard-library HTTP parser-worker client/server, a fixture-only compatibility
manifest with offline/online doctor validation, a fail-closed provider that
stages, fsyncs, validates, and rollback-protects replacement or reuse of
matching audit/canonical artifacts, plus an opt-in compatibility composition
root whose legacy `balanced` name is deprecated in favor of build policy
`routed`.
Core accepts that provider through constructor
injection while retaining its legacy MinerU default. This proves the modular
boundary, not the vendor deployment: no supported settings/CLI path constructs
it, and no qualified Docling/MinerU/Paddle loader, pinned image, or supervisor
is shipped yet.

Every extension implements a small, versioned port and declares capabilities rather than being selected through `if backend == ...` chains:

- `ParserAdapter`: supported MIME/page types, modalities, page/region parsing, health and output schema version.
- `LanguageModel` and `VisionModel`: structured output, streaming, token/image limits, batching and cancellation.
- `EmbeddingModel` and `Reranker`: vector kind/dimension, maximum batch, normalization and score semantics.
- `VectorIndex`, `BookIndexRepository`, `ArtifactStore`, and `TraceSink`: storage-independent application contracts.
- `Ingestor`: PDF, DOCX, PPTX, XLSX, HTML, image and email conversion into source-preserving observations.

Adapters are registered through explicit configuration or versioned Python
entry-point groups: `docarag.parsers.v1`, `docarag.ingestors.v1`,
`docarag.models.v1`, and `docarag.candidate_indexes.v1`. Discovery reads only
package metadata and a dependency-light manifest; loading a capability is a
separate explicit action. Adding a parser should require an independently built
adapter wheel, a capability manifest, conformance fixtures, and one preset
entry—not edits throughout BookRAG.
Adapter output validates against versioned JSON Schema plus dependency-light
domain DTOs; Pydantic may validate HTTP boundary models but is not a transitive
domain requirement. Schema changes use additive fields first, explicit
migrations, a deprecation window, and golden replay.

Keep provider APIs capability-based even when several services expose an OpenAI-compatible HTTP surface. OpenAI compatibility standardizes transport, not image limits, structured-output behavior, embeddings, lifecycle controls, or error semantics. The adapter normalizes those differences into typed DocARAG results.

#### 3.7.1 Supported surface and extension policy

Maintainability requires a deliberately small support surface:

- **Public:** versioned domain DTOs, ports, the application composition facade,
  CLI/API request and result schemas, named presets, artifact formats, and
  documented configuration keys.
- **Internal:** vendor payload translators, BookRAG compatibility projection,
  worker supervision, persistence implementation, caches, and UI state.
- **Never public:** parser SDK objects, model client objects, pickle internals,
  Qdrant response types, process IDs, list offsets, or mutable global
  singletons.

The CLI, future HTTP API, Web UI, tests, and workers must call the same
application use cases. They may render results differently, but must not each
implement routing, fallback, source binding, indexing order, or retrieval
policy. Composition happens once at startup through explicit registries and
constructor injection; importing a module must not download a model, connect to
a service, allocate a GPU, or mutate configuration.

Every parser/model/store extension must ship:

1. a capability manifest with stable component/model/schema versions and hard
   limits;
2. a contract adapter that never leaks vendor classes across its port;
3. source-binding, malformed-output, timeout/cancellation, and deterministic
   replay fixtures;
4. a compatibility-matrix entry with tested package/image/CUDA versions;
5. golden accuracy, latency, memory, and clean-setup evidence;
6. additive schema evolution or an explicit migration and deprecation window.

The extension exit test installs an independently built fixture wheel, discovers
it through the named group, runs `doctor` plus the shared conformance suite,
indexes a fixture, then uninstalls it and proves core behavior is unchanged. An
in-tree fake alone cannot satisfy this gate.

Semantic versioning applies to the supported DTO/API/config surface. Stored
artifacts carry their schema and producer versions; readers either migrate them
explicitly or reject them with remediation. No code guesses which historical
shape it received. CI runs the dependency-free unit/contract suite, parser
goldens, migration replay, wheel/import smoke test, and selected real-worker
integration tests as separate gates so contributors do not need every GPU stack
to change domain code.

Every supported release also reads and migrates N-1 config/artifact/API fixtures,
rejects unsupported older/newer schemas with remediation, and passes a failed-
upgrade rollback test. Production adapters declare a rolling compatibility
window so N and N-1 controllers/workers can coexist during an upgrade without
writing a shape the older reader can misinterpret.

#### 3.7.2 Operational-cost gate for model selection

A model is not eligible for a default preset merely because it wins an accuracy
metric. Record a release scorecard for each candidate:

```text
quality:       per-modality extraction/retrieval/answer metrics and calibration
latency:       cold start, p50/p95 page or query time, timeout/fallback rate
resources:     download, disk, peak CPU RAM/VRAM, steady idle VRAM, unload time
operability:   clean-host setup time, service count, diagnostics coverage,
               restart/resume success, supported upgrade combinations
cost:          GPU-seconds or API cost per accepted page/query
```

Selection is deterministic and is made separately for each declared page or
query class. First apply hard eligibility constraints from the versioned release
protocol: every applicable modality-quality floor, maximum p95 latency and
failure rate, RAM/VRAM ceiling, clean-host setup/download/service ceiling,
restart/resume gate and source-fidelity gate must pass. Among eligible
components, minimize measured expected cost per **accepted** page/query,
including retries, fallbacks and API or GPU-seconds. Costs within 5% are a tie;
break ties by lower p95 latency, then lower clean-host setup time, peak VRAM,
download bytes and finally lexical component/version ID. This order is recorded
in the selection trace and cannot be changed after seeing a workload result.

There is no opaque weighted “best model” score and no manual choice from an
unordered Pareto frontier. A more complex model is admitted only when it clears
the same hard constraints and wins that deterministic order for the declared
class. Request budgets may permit more work, but never weaken source binding,
validation, provenance or failure semantics.

### 3.8 Source of truth and rebuild hierarchy

There must be one unambiguous authority chain:

1. **Original source:** immutable uploaded document bytes, source metadata, content hash, and access policy.
2. **Authoritative interpretation:** versioned page renders, parser observations, canonical evidence revisions, and human corrections with complete lineage.
3. **Authoritative BookIndex:** active versioned `T`, `G`, and `M` rows referencing canonical evidence revisions.
4. **Rebuildable derivatives:** chunks, summaries, embeddings, Qdrant points, caches, reports, Markdown exports, and UI projections.

The compatibility foundation implements immutable source identity plus a
run-local shadow canonical interpretation, canonical/derived lineage, and
bounded hydration back through the active tree. It also snapshots active
configuration, fingerprints controller code, writes an immutable run manifest,
and publishes atomic tree/graph state plus a run-local Qdrant readiness
attestation and verified artifact metadata.
The immutable evidence-graph checkpoint and checked legacy projection are now
available as dependency-light publication primitives, but the live KG stage
does not yet populate or adopt them. The current legacy builder's coarse
whole-node selections do not contain exact per-assertion model-use receipts, so
its releases are not authority-eligible even if written to the checkpoint
store. These JSON artifacts are not yet a transactional corpus authority or durable
`T/G/M` repository; `tree.pkl` and graph JSON remain active BookRAG
compatibility read formats, while Qdrant is the active derived vector read path.
Query traces still do not prove complete entity/relation/GT-Link lineage or
final claim support.

Never edit generated Markdown or a vector point as the correction mechanism. A correction creates a new canonical evidence revision, invalidates affected descendants through the lineage graph, and rebuilds only the impacted tree, graph, GT-Link, chunk, and vector records. The target typed `AnswerResult` must pin `document_version_id`, `evidence_revision_id`, retrieval-policy version, and model/prompt versions so it can be reproduced after the active document changes.

Deletion and access revocation start at the authoritative document version. Derived indexes are removed through a durable outbox and are considered unavailable immediately through authoritative metadata filters, even if physical vector deletion is still pending.

## 4. Page router design

### 4.1 Routing is deterministic first

Render every page at a low probe resolution and collect cheap features without a large VLM:

- native characters and words per page;
- replacement/control character ratio and duplicated-text ratio;
- text coverage versus page area;
- raster-image coverage and connected-component density;
- font count, vector-path density, rotation, and skew;
- candidate table regions, equations, figures, and multi-column layout;
- disagreement between native text and a small sampled OCR pass;
- neighboring-page route and document-level prior.

Start with rules and logged reasons. Train a small classifier only after the golden corpus contains enough labeled pages. A learned router must output calibrated probabilities and retain a deterministic safety policy.

### 4.2 Initial routing policy

| Page evidence | Primary adapter | Escalation |
|---|---|---|
| Strong native text, ordinary report/book layout | Docling | MinerU if structure invariants fail |
| Native technical page with many formulas/tables or complex columns | MinerU hybrid/pipeline | PaddleOCR-VL for image-only regions |
| Scan, slide, screenshot, diagram, chart, photographed/warped page | PaddleOCR-VL 1.6 | MinerU for formula/table disagreement |
| Ambiguous or low-confidence page | Cheapest likely adapter first | Parse only disputed regions or run a second adapter |

PaddleOCR-VL 1.6 is an appropriate visual candidate: its current report targets text, formula, table, and chart recognition with a compact coarse-to-fine design ([PaddleOCR-VL paper](https://arxiv.org/abs/2606.03264)). MinerU’s coarse-to-fine model is similarly motivated by preserving high-resolution detail without processing every pixel at full cost ([MinerU2.5 paper](https://arxiv.org/abs/2509.22186)). DocARAG should apply the same principle at system level: cheap page analysis first, expensive region recognition second.

Use supported vendor service surfaces rather than importing their internal
pipelines. Current Docling exposes versioned synchronous/asynchronous REST,
polling/WebSocket status, an official client, bounded requests, containers and
Redis-backed workers through `docling-serve` ([Docling REST API](https://docling-project.github.io/docling/usage/api_server/rest_api/), [Docling deployment](https://docling-project.github.io/docling/usage/api_server/deployment/)).
MinerU 3.x exposes `mineru-api` and `mineru-router` for asynchronous and
multi-service/GPU execution, but its documented task state is process-local;
DocARAG work items must therefore remain the durable authority
([MinerU quick usage](https://opendatalab.github.io/MinerU/usage/quick_usage/)).
Record MinerU's `effort` (`medium` or `high`) in parser revision identity:
`medium` omits image/chart analysis, so `high` is a distinct visual-capability
route rather than a transparent performance setting
([MinerU CLI](https://opendatalab.github.io/MinerU/usage/cli_tools/)). A Paddle
worker must wrap the complete layout-localization, region-recognition and
post-processing pipeline; a raw VLM/vLLM endpoint is not an equivalent parser
([PaddleOCR-VL pipeline](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)).

### 4.3 Confidence, reconciliation, and fallback

Do not trust a parser’s confidence field as a universal score. Compute adapter-independent quality checks:

- nonempty block and reading-order coverage;
- bbox bounds and overlap sanity;
- OCR/native-text agreement on sampled spans;
- balanced table dimensions and valid HTML/Markdown;
- formula parseability and suspicious Unicode rate;
- heading hierarchy consistency;
- caption-to-figure proximity;
- repeated header/footer detection;
- page-to-page continuity for tables and paragraphs.

If a page fails checks, create a targeted retry work item. Where two parsers disagree, retain both raw candidates, choose at block/region level, and record the arbitration reason. Use adjacent-page windows only for structure reconciliation such as continued tables; do not make overlapping parser windows create duplicate canonical blocks.

### 4.4 Router quality metric

Measure **routing regret**:

```text
regret(page) = Q_page(best eligible parser) - Q_page(router output)
```

`Q_page`, applicability, weights, eligibility, failures, paired stratified
bootstrap and pass/fail confidence bounds are defined once in Section 9.1; this
section does not invent another “quality” score. Report mean/p95 regret as
diagnostics plus every co-primary metric, heavy-parser invocation rate,
dual-parse rate, accepted-page cost and failure rate by document class. The goal
is not maximum Docling/MinerU/Paddle use; it is statistically near-oracle
quality at materially lower heavy-model volume.

## 5. Index construction for accuracy without runaway cost

### 5.1 Structure-aware chunks

Build chunks from canonical structure, not from exported Markdown character windows:

- inherit document title, section path, page, and block type;
- keep a table with its headers; split very large tables by row groups while repeating headers;
- keep display equations with the preceding definition and following explanation;
- keep figures/diagrams with captions and nearby referring paragraphs;
- preserve list hierarchy and code blocks;
- use modest prose overlap only at semantic boundaries;
- store block IDs so every answer span can resolve to page and bbox.

Start with 400–800 text tokens per prose chunk and tune on retrieval evaluation. Do not apply one chunk size to tables, equations, and images.

Add a deterministic context prefix such as document title + section path + page range before embedding. This captures much of the value of contextual retrieval without an LLM call per chunk. Evaluate LLM-generated contextual prefixes and late chunking as controlled alternatives. Both contextual retrieval and late chunking address the loss of document context caused by independent chunk embedding ([Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval), [Late Chunking](https://arxiv.org/abs/2409.04701)).

### 5.2 Progressive index tiers

#### Tier A: mandatory backend-neutral base index

- canonical text/table/formula serialization;
- deterministic structural tree and relations;
- BM25 sparse candidates in every profile;
- a text dense vector when the selected profile has a qualified local or remote
  embedding capability (Qwen3 is the workstation candidate, not a Tier A
  implementation dependency);
- payload filters for tenant, document version, page, block type, language, and access policy.

The document becomes queryable when the profile's declared Tier A commits.
Endpoint-free `lite` may therefore be BM25-only and reports `BASE_SEARCHABLE`
with an explicit degraded capability/reason until a dense embedding capability is configured;
it must never pretend to meet the dense+BM25 production recall gate. Visual
descriptions, summaries, and KG construction cannot block this state.

#### Tier B: precision enrichment

- page/region visual vectors on visual pages only;
- figure/diagram descriptions stored separately from source text;
- Qwen3 text reranking over a bounded candidate set;
- optional table row/cell representations for table-centric corpora.

Visual document retrieval research shows that page-image embeddings retain layout and visual signals lost by text-only pipelines ([ColPali/ViDoRe](https://arxiv.org/abs/2407.01449)). Qwen3-VL-Embedding and its paired reranker are current unified multimodal candidates with 2B and 8B variants ([Qwen3-VL retrieval report](https://arxiv.org/abs/2601.04720)). Benchmark the 2B model against the existing GME model first; do not pay 8B latency or multivector storage without a measured gain.

Do not collapse text and vision into one opaque score. Recent co-modality work
reports gains from retrieving text and page-image evidence in separate channels
and explicitly normalizing their score distributions before fusion; it also
shows that extra context can hurt on unanswerable or slide questions. Treat that
as a hypothesis to validate on DocARAG's golden set: use RRF as the safe initial
fusion, calibrate modality scores only from held-out data, and let the query
router omit an irrelevant leg ([CMRAG](https://arxiv.org/abs/2509.02123)). A
training-free multi-granularity study likewise supports page, region and OCR
candidate legs followed by bounded reranking/verification, rather than sending
the whole document to a VLM ([multi-granularity multimodal retrieval](https://arxiv.org/abs/2505.01457)).

#### Tier C: full BookRAG index

- cached summaries for high-value structural nodes;
- modality-dependent entity/relation extraction from every eligible BookRAG node;
- gradient-based entity resolution and explicit GT-Links;
- graph/text reasoning scores and BookRAG operator readiness;
- graph communities or topic aggregates for global queries;
- on-demand summary generation cached by input/model/prompt hash.

The semantic KG is required for `BOOKRAG_READY`, because BookRAG's entity scent, graph reasoning, and GT-Link mapping depend on it. It remains optional only for the earlier `BASE_SEARCHABLE` state. Always keep deterministic structural relations too: document/section/page/block hierarchy, next/previous reading order, caption-of, contains, table-continues, and explicit citation/reference edges.

Do not create semantic links with all-pairs cosine similarity. Candidate neighbors must come from the ANN index and be capped per node. Query-time expansion must be restricted to retrieved seeds, ancestors, and a small `k`-hop/edge budget.

### 5.3 Robust BookIndex tree, graph, and GT-Links

#### Tree `T`

Store the tree as versioned adjacency rows, not Python parent pointers as the authority:

```text
tree_nodes(node_id, document_version_id, revision, parent_node_id,
           node_type, semantic_subtype, canonical_block_id,
           order_key, depth, section_path, status)
tree_edges(parent_node_id, child_node_id, edge_type, revision)
```

Tree invariants checked transactionally or at readiness:

- exactly one root per document version;
- every non-root node has one active parent;
- no cycles; stored depth/path agrees with the parent chain;
- sibling order is deterministic and consistent with page/reading order;
- every leaf resolves to an active canonical block revision;
- section ranges contain their descendants' page/order ranges;
- no orphan leaves after title reclassification;
- changes create a new revision and invalidate only affected ancestor summaries/KG work.

Use stable IDs derived from document version + canonical block/section identity. Never expose list offsets as durable node identities.

The BookRAG paper focuses on documents with a logical narrative hierarchy and explicitly excludes slide categories without that structure. DocARAG must use document-type tree profiles while keeping the same `T` interface:

| Document type | Tree profile |
|---|---|
| Book/report/paper | document -> chapter/section/subsection -> multimodal block |
| Slide deck | deck -> inferred topic/slide group -> slide -> text/table/figure/diagram region |
| Scanned document | document -> detected section or page -> OCR paragraph/form/table/image region |
| Spreadsheet | workbook -> sheet -> named table/region -> row group/cell evidence |
| Form | document -> page/section -> field group -> label/value/signature/table region |
| HTML/email | collection/thread -> message/heading -> content/attachment block |

If a trustworthy higher hierarchy cannot be inferred, page/slide/sheet nodes form a valid shallow tree. The planner then relies more heavily on hybrid candidates, entity scent, modality/range filters, and graph reasoning instead of inventing section headings.

#### Knowledge graph `G`

The authoritative graph must be a **directed multigraph**, represented by relational rows and hydrated into a bounded in-memory graph for BookRAG reasoning:

```text
entity_mentions(mention_id, node_id, span/bbox, surface_form, type,
                description, extractor_revision, confidence)
canonical_entities(entity_id, tenant/corpus scope, canonical_name, type, revision)
entity_aliases(mention_id, entity_id, er_decision_id, confidence)
relation_assertions(relation_id, source_entity_id, target_entity_id,
                    relation_type, direction, weight, evidence_node_id,
                    evidence_span, extractor_revision, confidence)
```

This preserves multiple relation types and directions between the same entities. Entity resolution is non-destructive: mentions remain immutable; gradient-based ER creates or revises the canonical mapping. Low-confidence merges become `SAME_AS_CANDIDATE` decisions instead of silently collapsing evidence.

Graph invariants:

- relation endpoints exist and are active;
- every extracted entity/relation has at least one evidence node;
- every canonical entity has at least one active mention;
- self-loops are retained only for explicitly allowed relation types;
- merge/split operations preserve all evidence and alias history;
- graph statistics and disconnected-component changes are checked against thresholds;
- readiness fails or becomes degraded if extraction/ER coverage falls below the configured level.

#### GT-Link `M`

Replace integer `source_ids` as the authority with explicit, versioned links:

```text
gt_links(link_id, document_version_id,
         entity_or_relation_id, tree_node_id,
         role, evidence_span_or_bbox,
         extractor_revision, confidence, active)
```

Roles include `MENTIONED_IN`, `DERIVED_FROM`, `DEPICTS`, `TABLE_CONTAINS`, `FORMULA_DEFINES`, and `RELATION_EVIDENCE`. Both directions are indexed. An entity merge unions links logically through aliases without rewriting or losing the original mention links.

Before `BOOKRAG_READY`, run BookIndex integrity checks over `T`, `G`, and `M`, generate counts/checksums, and store them in the readiness record. Query traces record exactly which GT-Links converted graph scores into selected tree nodes.

### 5.4 Qdrant candidate-index layout and scale gate

The application keeps a backend-neutral candidate-index port so Qdrant SDK
types never leak into BookRAG domain code, but the deployment decision is
settled: Qdrant is the sole/default vector database in every profile. The
implemented provider currently uses one unnamed dense vector, deterministic
UUID point IDs and scope-filtered payloads. Sparse and named visual vectors are
future schema revisions, not claims about the current collection. Their target
layout is one shared collection per embedding-schema version, not one collection
per document:

```text
named vectors:
  text_dense       Qwen3-Embedding candidate
  text_sparse      BM25 sparse vector
  visual_dense     GME or Qwen3-VL-Embedding candidate

payload:
  tenant_id, document_id, document_version_id,
  chunk_id, page_ids, block_ids, block_types,
  language, access_tags, content_hash, indexed_at
```

Use deterministic point IDs and outbox-driven upserts. Retire a document version with a payload-filtered delete or tombstone, never global `reset()`.

Qdrant is the fixed DocARAG vector-store choice. Its payload filtering,
deterministic upserts, local deployment and distributed sharding/replication fit
the required evolution path. Its Query API also supports hybrid fusion
([Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)).
Start with the implemented unnamed dense vector and ordinary application-level
RRF. Add sparse/named visual vectors or weighted fusion only under a new
versioned schema after a held-out evidence-labelled set demonstrates a stable
gain.

Selection rule:

| Scale/workload | Decision |
|---|---|
| Local development and CPU `lite` | Loopback-only single-node Qdrant with a durable named volume |
| Current workstation | Single-node Qdrant with snapshots; benchmark RAM/disk before enabling quantization |
| Production below measured single-node saturation | Managed or self-hosted single-node Qdrant with tested backup/restore and PostgreSQL outbox replay |
| Production HA or measured single-node saturation | Qdrant 3+ nodes, replication factor at least 2, and shard count planned from tenant size/query locality |

`VECTOR-SCALE-1` is executable and numerically bounded. Freeze one million
version/access-scoped candidate records, their text/visual/query-class
distribution, raw encoded vector-plus-payload byte count and labelled query set.
For each adapter run five independent repetitions on the recorded workstation:

1. build from an empty store in at most 30 minutes;
2. warm with 10 concurrent clients for 10 minutes, then keep 10 clients active
   until **both** 15 measured minutes and 50,000 completed measured queries have
   been reached, using the frozen query mix;
3. require cgroup peak working-set memory no greater than 16 GiB, active
   index-plus-WAL disk no greater than `3.0x` the frozen raw encoded input, and
   active index plus one tested snapshot no greater than `4.5x` that input;
4. require the one-sided stratified-bootstrap 95% **upper** confidence bound for
   warm p95 candidate latency to be at most 300 ms, and the paired 95% **lower**
   confidence bound for Recall@10 difference from the frozen exact-search
   reference on the labelled sample to be at least -1.0 percentage point;
5. reproduce deterministic rebuild/delete counts and hashes, return zero
   cross-tenant/access-filter violations, and restore the tested snapshot to a
   query-ready store in at most 10 minutes.

Bootstrap resamples query IDs within query class and repetition so repeated
measurements are not treated as independent labels. Resource/build/restore
ceilings must pass in every repetition. Qdrant is already required; the gate
decides whether a new sparse/visual schema, quantization setting, or clustered
topology is eligible. Failure keeps the last qualified Qdrant schema/topology
and blocks the scale or feature change rather than selecting another database.

Qdrant's official distributed guide documents sharding, replication,
consistency choices, and three-node failure tolerance; it also makes clear that
point updates are not distributed ACID transactions, which is why PostgreSQL
plus the outbox remains the authority
([Qdrant distributed deployment](https://qdrant.tech/documentation/distributed_deployment/)).

Do not create a collection per document or tenant. Use an embedding-schema-version collection, indexed payload fields, and document/tenant filters. At scale, choose partitioning/shard keys from measured tenant size and query locality.

### 5.5 Model recommendation for the current 48 GB machine

Initial deployment candidates, subject to the gates in Section 9:

| Function | Default experiment | Why |
|---|---|---|
| Text embedding | Qwen3-Embedding-4B, test MRL-reduced dimensions | Better accuracy/cost balance than making 8B the default |
| Low-latency text baseline | Qwen3-Embedding-0.6B | Establish whether 4B actually earns its latency/VRAM |
| Text reranker | Qwen3-Reranker-4B on top 30–50 | Bounded high-precision second stage |
| Visual embedding | Qwen3-VL-Embedding-2B versus current GME | Current, unified visual/text candidate; index visual pages only |
| Visual extraction | PaddleOCR-VL 1.6 | Strong result on the project’s slide-deck test |
| Formula/table parser | MinerU current stable adapter | Strong result on the technical worksheet test |
| Native parser | Docling current pinned adapter | Strong result on clean native report and rich canonical structure |

Do not use leaderboard scores to select 8B automatically. Measure Recall@k, nDCG, reranked answer accuracy, p95 latency, index bytes/chunk, and peak VRAM at 0.6B/4B/8B and multiple Matryoshka dimensions.

## 6. Query path

### 6.1 Query router

BookRAG's planner remains the query router. It first classifies the original BookRAG categories—single-hop, multi-hop, and global aggregation—and also emits document scope plus one or more evidence needs:

- exact/local fact;
- semantic local explanation;
- table/numeric comparison;
- formula/derivation;
- visual/diagram/slide;
- multi-hop entity relation;
- global summary/theme/comparison;
- unanswerable or access-restricted.

Use deterministic signals first—quoted terms, identifiers, page references, words such as “diagram,” “table,” or “overall”—plus a small classifier when needed. Log chosen legs and scores for offline evaluation.

The target operator library is an additive extension of the paper and current `gbc_*` code:

| Family | Preserved BookRAG operators | Additions for scale/multimodality |
|---|---|---|
| Formulator | `Extract`, `Decompose` | `Classify_Evidence_Need`, `Resolve_Document_Scope` |
| Selector | `Filter_Modal`, `Filter_Range`, `Select_by_Entity`, `Select_by_Section` | `Select_Documents`, `Hybrid_Candidates`, `Select_by_Visual` |
| Reasoner | `Graph_Reasoning`, `Text_Reasoning`, `Skyline_Ranker` | `Visual_Reasoning`, `Table_Reasoning`, `Formula_Reasoning`, `Evidence_Consistency` |
| Synthesizer | `Map`, `Reduce` | `Hydrate_Evidence`, `Cite_and_Verify` |

Every operator execution creates a `retrieval_trace_step` with input IDs,
resolved scope and revisions, parameters/budgets, per-leg ranks and scores,
candidate counts before/after, visited graph/tree IDs, selected/rejected IDs with
reason codes, queue/remote/operator latency, fusion contribution, fallback
reason and deadline state. The UI can therefore reproduce the exact path from
question to answer without treating telemetry as the source of truth.

### 6.2 Candidate-first retrieval

BookRAG remains **narrow-then-reason**. The candidate-index port is a document/
section scent generator backed by Qdrant for vector search; Qdrant does not
replace `T`, `G`, or `M`. `VECTOR-SCALE-1` controls schema/topology upgrades,
not whether Qdrant is used.

Default single-hop path:

```text
Extract(query entities)
  -> Select_Documents when corpus scope is not fixed
  -> run scoped Select_by_Entity and dense + sparse candidates concurrently
     (do not force a nearest entity below the configured confidence threshold)
  -> ordinary RRF + deduplicate by canonical evidence identity
  -> group hits into sections and add only bounded direct neighbors from T/M
  -> Graph_Reasoning + Text_Reasoning
     + modality reasoner when required
  -> Skyline_Ranker
  -> Hydrate_Evidence
  -> Map/Reduce -> Cite_and_Verify
```

Default candidate budget inside that path:

1. Resolve tenant, access, active document version, readiness and caller-selected
   document scope before fan-out; bind each document's immutable canonical-
   catalog root/count rather than enumerating its material IDs, and apply the
   same filters to every retrieval leg.
2. Run entity, dense text top 60 and BM25 top 60 independently and concurrently.
   A missing/low-confidence entity match contributes no entity hits; it never
   force-selects the numerically nearest entity.
3. Apply ordinary RRF and deduplicate by canonical evidence identity to about 60 chunks.
4. If visual intent is present, retrieve visual top 30 and fuse by rank.
5. Convert entity hits through exact GT-Links to tree nodes; group candidates into document/section subtrees.
6. Expand only direct structural neighbors needed for coherence, capped at 2–3
   neighbors per seed and by total vertices, edges, hops and deadline.
7. Rerank/reason over 30–50 candidates under both item and input-token/image-byte budgets.
8. Apply Skyline and an absolute safety cap; hydrate 8–15 final evidence units.

Multi-hop queries run the single-hop path for independent decomposed questions, retain each evidence set and partial answer, then Reduce. Global queries preserve BookRAG's `Filter_Modal`/`Filter_Range` + Map/Reduce path; corpus-global queries first map per selected document and reduce across documents. This prevents local queries from paying global-graph cost.

### 6.3 Evidence contract and answering

Retrieval returns stable evidence references first. `Hydrate_Evidence` then loads the exact source material the reasoner/generator needs; descriptions are never treated as a substitute for the original table, formula, or image.

```text
evidence_id, document title/version, immutable source_sha256, page ID/number/bbox,
canonical catalog revision/root and membership receipt,
tree_node_id/revision, canonical block/revision, section path, block type/subtype,
all available representation/asset IDs, exact selected representation/asset IDs,
raw parser observation/artifact IDs and verified asset/crop digest URIs,
supporting entities/relations/GT-Links,
retrieval operators and graph/text/visual/table/formula scores
```

Canonical material availability and query/build selection are deliberately
separate. A candidate's catalog membership is checked by revision-bound point
lookup or bounded proof and then its selected bytes are re-read and hashed; a
query request never copies the document's complete canonical-ID catalog.
Hydration is also not proof of model consumption. Every model call
persists a `MaterialUseReceipt` containing the exact canonical revision and
span/cell IDs, SHA-256 of the serialized text actually placed in the request,
asset-byte digests for every supplied crop, the ordered material-selection
digest, request/prompt digest, model identity and response digest. An answer may
cite only material present in both its hydration receipt and the producing model
call's use receipt; listing an image/table in the canonical block—or merely
loading it into application memory—is not proof that the model saw it. Every citation resolves
backward to the canonical revision, parser observation/raw artifact, page locus
and original document hash, while summaries, vectors and Markdown remain
rebuildable derivatives.

Map/Reduce keeps that proof transitive through immutable `ClaimArtifact`s; it
must not flatten intermediate prose and lose the source chain:

- A Map claim records its normalized claim/text digest, producer invocation and
  the exact hydration plus `MaterialUseReceipt` IDs for the source evidence that
  was actually supplied. It cannot cite material outside those receipts.
- A Reduce claim records the ordered child-claim IDs and immutable digests, the
  SHA-256 of the exact serialized child claims placed in the Reduce request, the
  Reduce invocation/model/prompt/response digests and its output claim digest.
  Direct source material enters a hierarchy through a Map claim, so every
  Reduce leaf has the same evidence contract.
- Claim IDs bind the producer, content and ordered parents. The claim graph is a
  bounded acyclic DAG; a Reduce citation set must be a subset of the transitive
  union of its child Map evidence. Replaying any edge with a changed child,
  prompt, response, canonical revision or asset digest produces a different ID.
- A calibrated semantic verifier evaluates support at each Map-to-evidence and
  Reduce-to-child edge and stores its versioned verdict. The final claim is
  `supported` only if every edge in its transitive closure is supported and all
  leaves resolve through hydration to immutable original source bytes. A use
  receipt proves consumption, not entailment; a missing or failed verdict yields
  `insufficient_evidence`, never an authoritative answer.

Hydration by evidence type:

| Retrieved node | Material supplied to reasoning/generation |
|---|---|
| Text | exact canonical text plus adjacent heading/path and word/page provenance |
| Scanned/text-in-image | OCR text with word boxes **and** the matching crop/page image |
| Table | cell grid/HTML, headers, caption/footnote, relevant row slices, and original table crop |
| Formula | LaTeX/MathML, surrounding definition/explanation, and formula crop |
| Image/figure | original asset/crop, caption, nearby references, OCR, and visual description |
| Chart | crop plus legend/axis OCR and structured series when available |
| Diagram/workflow/flowchart/sketch | crop plus OCR labels, description, and structured diagram nodes/edges when available |

The query's evidence need chooses the answer backend. Text-only evidence can use the text LLM. Any answer dependent on pixel/layout interpretation uses the VLM with the actual crops. Tables and formulas can use the text LLM only when the structured representation passes quality checks; otherwise the crop is included in VLM reasoning.

Current compatibility hydration verifies canonical text/representation lineage
and emits citation IDs, but asset bytes remain reference-only. The material
guarantees below are target behavior.

The generator must cite `evidence_id`s. Each answer claim is linked to one or more evidence IDs in the retrieval trace. A post-check verifies that citations exist, cited nodes were actually hydrated **and occur in the model-use receipt**, numeric units match, table cells/formulas/images are from the active document version, and the complete claim DAG passes semantic support verification. Return “insufficient evidence” when the evidence threshold is not met. Legacy KG checkpoint assertions retain explicit `coarse_whole_node` provenance until an extractor emits assertion-specific selections; current coarse releases are inspection/shadow artifacts, are not authority-eligible, and cannot open production graph retrieval or prove a fine-grained claim locus.

### 6.4 Algorithmic complexity and latency budgets

Use bounded candidate algorithms throughout. Let `D` be encoded source bytes,
`P` pages, `X` total decoded pixels actually processed, `O` parser observations,
`B` canonical blocks, `E` entity mentions, `N` corpus candidate records, `H`
the sum of hits returned by all retrieval legs, `U` unique candidate evidence
identities, `K` fused output candidates, and `S`
the selected graph/tree scope. Page count alone is not a memory bound: raster
dimensions, decompression size, observation density and table-cell count all
have independent hard limits.

| Operation | Rejected behavior | Target behavior |
|---|---|---|
| Page parsing | whole-document materialization or unbounded raster decode | `O(D + X + O)` consumed work; per-page encoded bytes, decoded pixels, observations/cells and microbatch size are hard-capped, so working memory is `O(microbatch limits)`, not `O(P)` |
| Block reconciliation | compare every block pair, `O(B^2)` | page-local spatial index/sweep, approximately `O(B log B + C)` where `C` is nearby candidates |
| Tree construction | repeated full-tree scans | implemented source-order/indexed construction, `O(B + T)`, preserving every source block |
| Entity resolution | all mention pairs, `O(E^2)` | normalized exact blocking + ANN top-`k` + rerank/adjudicate, approximately `O(E log E + Ekr)` |
| Vector candidates | full corpus scan | Qdrant ANN retrieval with authorization/version payload filters pushed down; sparse and visual legs use versioned Qdrant schema extensions only after their accuracy/latency gates |
| Query scope binding | copy every canonical revision/representation/asset ID into every request | one constant-size catalog-root binding per selected document plus point membership checks/proofs only for returned candidates, `O(selected documents + returned material IDs)` |
| Candidate fusion | score-scale mixing or unbounded duplicate lists | implemented ordinary RRF with source/revision-bound deduplication and a bounded top-`K` heap in `O(H + U log K)` time and `O(H)` working memory; material unions are deferred to the selected `K`, with hard per-leg and total-hit caps |
| Reranking | rerank every chunk | the implemented local lexical/structural leg scans the chosen subtree and retains at most `M` inputs; target hybrid fused `K` has hard item, token, image-byte, call and deadline budgets |
| Graph reasoning | query-time all-pairs similarity, `O(S^2)` memory | bounded adjacency/beam expansion, `O(V_S + E_S)` for visited scope |
| Tree expansion | traverse the whole tree | indexed subtree/range lookup plus capped neighbors, `O(log B + output)` |
| Skyline fusion | add unbounded score dimensions | current two-score skyline is `O(M log M)`; extra modality scores use scalarized preselection or a fixed `d` and hard-capped `M`, with the explicit `O(d M^2)` fallback included in the request budget |
| Global answer | one enormous context | per-section/document Map with bounded contexts, then hierarchical Reduce |

The implemented routed parser contract already enforces the same principle.
For `P` pages and `A` registered adapters, deterministic routing examines a
bounded capability set per page (approximately `O(P * A log A)` with current
candidate ordering; `A=3` in the opt-in compatibility routed profile, whose
legacy `balanced` build-side name is deprecated). Parser calls are split into
configured microbatches. Reconciliation indexes observations page-locally and
has separate hard limits for plausible matches `K` and raw probes `R`; an
observation exceeding either bound produces no arbitrary match and is marked
for review. Canonical-to-BookRAG projection sorts once, `O(B log B)`, and then
converts each block once. Tree construction then indexes the `B` source blocks
and `T` outline entries in `O(B + T)` time without dropping titleless or
non-text blocks. Text candidate selection scans the full chosen subtree once,
combines lexical matches and structural coverage in
`O(text bytes + N_S log M)` time with `O(M)` retained-candidate memory, and
exposes at most the validated `candidate_pool_size` (512 by default) to the
reranker. Before that local cap, however, the compatibility query materializes
the selected subtree and induced graph, runs PageRank, and sorts all graph
candidates. With `I_S` PageRank iterations, its current overall cost is
approximately
`O(N_S + L_S + I_S(V_S+E_S) + N_G log N_G + text bytes + N_S log M)` time and
`O(N_S+V_S+E_S)` memory, plus the guarded optional `O(V_S^2)` semantic-link
matrix. Current graph/text Skyline fusion then sorts its bounded input once,
returns at most `topk`, and is `O(M log M)`. These local guards are useful but
do not yet establish candidate-first end-to-end complexity. Load tests must
still measure parser memory, large tables, adversarial page density, recall
under the cap, and end-to-end latency.

For the target candidate-first query, authorization/readiness and immutable
catalog-root binding are `O(selected documents)`; membership work is
`O(returned material IDs)` rather than `O(all canonical material IDs)`. Fusion
is `O(H + U log K)` time/`O(H)` working memory,
bounded tree/graph work
is `O(V_S + E_S)`, reranking is linear in its capped serialized token/image
input, and hydration is linear in the exact verified bytes/pixels returned.
Thus the non-model path is
`O(H + U log K + V_S + E_S + hydration_bytes + hydration_pixels)` time and
`O(H + V_S + E_S + bounded hydration batch)` memory. ANN backend costs remain
adapter- and index-dependent and are reported empirically; the public contract
does not claim an exact HNSW asymptotic guarantee.

The compatibility implementation now meets the bounded-memory target for
BookRAG tree vector writes to Qdrant: one stable tree scan with `O(batch)` text/
image working memory. `vdb.index_batch_size` is validated and defaults to 256.
The transactional PostgreSQL outbox is not yet wired, and the remaining
baseline/evaluation materializers must be migrated separately.

`C`, per-leg hits, fused `K`, graph vertices/edges/hops, rerank items/input
tokens/calls, selected documents, hydrated evidence/pixels, Map calls,
generation input/output tokens and the overall deadline are policy limits
recorded in the retrieval trace. One request budget is shared across
decomposed sub-questions; a stage cannot reset it. The planner chooses among
three named budgets:

- `fast`: deterministic intent routing, hybrid search, small rerank, no query-time LLM planning unless ambiguity remains;
- `balanced`: BookRAG entity/section planning, bounded graph and modality reasoning;
- `deep`: decomposition and Map/Reduce with a declared cost/time ceiling.

Make those names executable. The first qualification run starts with these
**safety ceilings**, not achieved-SLO claims; lower them only when the golden
set preserves recall and raise them only through a reviewed profile revision:

| Policy | Overall deadline | Candidate input | Graph expansion | Rerank budget | Final hydration | Generation/Map budget |
|---|---:|---:|---:|---:|---:|---:|
| `fast` | 15 s | 3 legs x 40 hits; fuse 40 | 128 vertices / 512 edges / 1 hop | 24 items, 2 remote calls, 24k input tokens | 8 evidence units, 8 MP pixels | 1 model call, 24k input / 2k output tokens |
| `balanced` | 60 s | 4 legs x 60 hits; fuse 60 | 512 vertices / 2,048 edges / 2 hops | 40 items, 4 calls, 64k input tokens | 12 units, 32 MP | 5 total model/Map calls, 96k input / 4k output tokens |
| `deep` | 180 s | 6 legs x 100 hits; fuse 100 | 2,000 vertices / 8,000 edges / 3 hops | 50 items, 8 calls, 128k input tokens | 15 units, 64 MP | 17 total calls, 256k input / 8k output tokens |

These totals cover the complete request, including decomposed sub-questions and
retries; they are not reset per stage. A deployment whose qualified model
cannot operate inside a policy must mark that policy unavailable rather than
silently extending its deadline.

The service returns partial diagnostics or `budget_exhausted` rather than silently exceeding the caller's deadline. Cancellation and deadlines propagate through API, scheduler, model calls, retrieval, and hydration.

Latency optimization order is: remove unnecessary work, run independent dense/sparse/visual legs concurrently, cache only version-pinned deterministic results, batch within a small wait window, then tune models. Do not trade provenance checks for latency.

Track latency as separate spans so model time is not confused with system overhead:

```text
admission -> intent/plan -> document selection -> candidate retrieval
-> graph/tree expansion -> rerank -> hydrate -> generate -> citation verify
```

Easy text questions should not invoke a VLM or global graph. Visual questions should not rely on text-only captions merely to meet a latency target. Quality gates and confidence-triggered escalation control the speed/accuracy trade-off.

## 7. GPU isolation and throughput

### 7.1 Separate controller from model processes

The controller must not import Torch, PaddlePaddle, Transformers, or parser
internals. Isolation is chosen by dependency/resource conflict, not by pipeline
stage. Compatible capabilities may share one deployment unit; incompatible
CUDA/Paddle/Torch stacks and independently scaled bottlenecks do not.

| Deployment unit | Capabilities it may host | Split condition |
|---|---|---|
| Controller/application | routing, reconciliation, policy, API/CLI/UI use cases, PostgreSQL coordination and Qdrant port orchestration; no ML imports | never merged with a GPU/vendor runtime |
| CPU document worker | Docling/native conversion, probes and deterministic transforms | split only for independent scaling or an incompatible native library |
| Parser GPU worker pool | one qualified MinerU **or** Paddle runtime per worker image; the pool exposes the common parser protocol | separate images/environments when Paddle/Torch/CUDA dependencies conflict |
| Model endpoint pool | text generation, VLM enrichment, embedding and reranking capabilities | share an existing endpoint when model/runtime/lifecycle are compatible; split for VRAM, latency isolation or incompatible engines |

`lite` starts the controller, PostgreSQL, Qdrant, CPU document capability and
configured remote model dependencies. `workstation` starts the required qualified units
through profiles and one GPU lease; it does not start every model. Production
may scale a measured bottleneck independently without changing the application
port. Each worker advertises capability, model/version hash, estimated VRAM,
maximum batch and health state, and the scheduler issues work only within its
lease/budget.

### 7.2 Target single-GPU policy

The current compatibility layer implements a filesystem admission lock for
supported indexing drivers, in-process LLM/VLM/KG semaphores, bounded batches,
strict provider cleanup, a finite Ollama five-minute residency default, and an
explicit opt-in unload request. It does not yet inspect VRAM, coordinate
query/external model processes, prove exclusive ownership before server-wide
unload, control vLLM/Paddle sleep, or enforce cross-process token-aware
concurrency. The following is the target scheduler policy:

- Reserve at least 6 GiB headroom; initial hard admission limit: 42 GiB total process allocation.
- Allow one heavy GPU model family at a time during indexing.
- Stop or sleep PaddleOCR-vLLM immediately after visual parsing drains.
- Choose finite stage-scoped Ollama retention; request `keep_alive=0` only when
  a cross-process ownership lease proves that the session may evict the model.
- Bound request concurrency by both sequence count and estimated prompt/image tokens.
- Move completed tensors to CPU/disk per microbatch; never concatenate a document’s full embedding set on GPU or RAM.
- Measure the persistent model memory separately from peak request memory.

vLLM sleep mode can release model weights and KV cache between workloads without destroying the container, and its current documentation reports reclaiming most GPU memory ([vLLM sleep mode](https://docs.vllm.ai/en/stable/features/sleep_mode/)). Keep its management endpoint private.

For production concurrency, separate online-query GPUs from batch-indexing GPUs. Query latency should never depend on a 1,000-page parser stage yielding the only GPU.

### 7.3 Backpressure

Use bounded queues at every expensive boundary:

```text
page render queue -> parse queue -> canonical-write queue -> embedding queue -> outbox
```

The downstream queue high-water mark pauses upstream claims. Store page images and raw results immediately, then release RAM. Microbatch size is a configuration discovered by load testing, not the document or tree level size.

### 7.4 Easy setup and deployment profiles

The supported path must not require users to manually create three environments, edit multiple YAML files, discover GPU IDs, start unrelated servers, or know which stage comes next.

Operational effort is part of model quality. A parser/model cannot become a
default solely because it wins an accuracy benchmark: its clean-host setup time,
download size, service count, cold start, steady and peak RAM/VRAM, failure
diagnostics, unload behavior and upgrade compatibility must pass the same release
matrix. A smaller model that stays within the latency/accuracy envelope and is
reliably provisioned is the better default; higher-cost parsing remains an
explicit build policy and higher query work remains an explicit request budget.

The compatibility CLI currently implements `init`, `settings`, read-only
`setup-plan`, `doctor`, `storage up/down/migrate/status`, `index`, `status`,
`verify`, `ask`, and `web`. The implemented storage journey starts loopback-only
PostgreSQL and Qdrant from `compose.yaml`, waits for health, applies checksummed
PostgreSQL migrations, and preserves the `docarag-postgres` and
`docarag-qdrant` named volumes across `storage down`:

```bash
docarag storage up                 # start both databases and migrate PostgreSQL
docarag storage migrate --json     # safe, checksummed, repeatable upgrade
docarag storage status --json      # PostgreSQL schema + Qdrant health
docarag storage down               # preserve both named volumes
```

Occupied ports are handled by putting matching service and client settings in
the project `.env` (or exporting them, which takes precedence):
`DOCARAG_POSTGRES_PORT` with `DOCARAG_DATABASE_URL`, and
`DOCARAG_QDRANT_HTTP_PORT`/`DOCARAG_QDRANT_GRPC_PORT` with
`DOCARAG_QDRANT_URL`. `DOCARAG_POSTGRES_PASSWORD` must match the password in the
database URL. `docarag.toml` stores only the `DOCARAG_DATABASE_URL` environment-
variable name; settings, health and exception output expose only a credential-
free endpoint. The CLI parses only an allowlist of storage assignments from
`.env` and never executes it. `.env.example` is a loopback development template,
not a production secret store. Shared deployments require a secret manager,
PostgreSQL TLS/authentication and `QDRANT_API_KEY`. Named-volume deletion is a
separate explicit destructive operation, never part of `storage down`.

`setup-plan` emits secret-safe argv arrays, static blockers, component ownership,
GPU-contention policy, and manifest-derived qualification gaps without writes,
subprocesses, network probes, or GPU state changes. The following is the target
whole-product command journey. Model lifecycle `up/down`, profiles, job IDs,
streaming progress, and document IDs remain roadmap items; they do not diminish
the implemented storage commands above:

```bash
docarag init --profile lite --build-policy native  # persists deployment + build-spec selectors
docarag doctor --offline --strict    # static config/runtime/disk/port preflight; no service dependency
docarag models pull --build-spec <id> # explicit, resumable build-model downloads only
docarag up --profile lite             # starts owned services and waits for health
docarag doctor --strict              # live capabilities, exact models and service health
docarag index /data/report.pdf --build-spec <id>  # returns a job ID/index revision
docarag search --index-revision <id> --budget fast "..." # source-linked retrieval
docarag ask --index-revision <id> --budget balanced "..." # capability-gated answer
docarag status [job-id]              # stage/page progress and actionable errors
docarag down                         # graceful model unload and service shutdown
```

`doctor` must produce a red/amber/green capability report, not a Python traceback. It should detect NVIDIA/container-toolkit availability, GPU capacity, incompatible ports, unreachable APIs, missing write permission, unsupported parser/model versions, and expected download/disk sizes. Every error includes the failing component, likely cause, exact remediation, and trace ID.

Support three curated profiles:

| Profile | Dependencies | Intended experience |
|---|---|---|
| `lite` | CPU controller + CPU Docling + PostgreSQL + single-node Qdrant; optional explicitly configured answer endpoint | Lowest-friction native-document indexing and `search`; no NVIDIA requirement. `ask` is advertised only when doctor qualifies the endpoint. Scanned/complex visual pages report explicit degradation unless an OCR/VLM endpoint is configured |
| `workstation` | Docker Compose with PostgreSQL and single-node Qdrant, isolated parser/model containers, one NVIDIA GPU | Full routed multimodal indexing after the routing and GPU-lease gates; storage remains the same default topology |
| `production` | API/controller, PostgreSQL, object store, Qdrant, independent workers/model endpoints | HA, autoscaling, tenant isolation and rolling upgrades; clustered Qdrant is enabled only after measured single-node saturation and backup/restore qualification |

Deployment, index construction and request-time reasoning are three separate
choices. A workstation should not load every expensive model merely because it
can, `production` does not imply the most expensive build, and a `deep` query
must not retroactively change an index.

Every service binding also declares ownership: `owned` means `docarag up/down`
starts and stops it, `borrowed` means one request/session may close only its
client, and `shared` means DocARAG must never unload or terminate the server.
Health checks and capability discovery apply to all three; lifecycle actions
apply only to `owned` resources. This makes local setup convenient without
turning `down` into a destructive operation against Ollama, vLLM, Qdrant, or a
remote API used by another project.

| Immutable build/index policy | Parse and enrichment behavior |
|---|---|
| `native` | Docling/native extraction for capable digital documents; OCR is used only when native extraction is unusable. Missing scan/visual/table/formula capability is recorded as typed degradation; there is no speculative dual parse |
| `routed` (default multimodal) | Deterministic page router selects Docling native, MinerU technical or Paddle visual and permits at most one quality-triggered fallback under the build budget |
| `routed-adjudicated` | `routed` plus a second-parser/adjudication pass only for golden-set-defined high-risk or failed-quality regions; every extra observation and selection decision remains immutable |

The request-time `QueryBudget` names remain `fast`, `balanced` and `deep` and
have the ceilings in Section 6.4. They control planner fan-out, retrieval legs,
graph expansion, reranking, hydration and synthesis for one request; they never
select parsers or mutate canonical/index artifacts.

Valid and degraded combinations are explicit:

| Published index capability | Query budget behavior |
|---|---|
| Any `native`, `routed` or `routed-adjudicated` index | `fast`, `balanced` and `deep` are valid resource ceilings; a larger budget may do more reasoning but cannot create evidence that was not built |
| `BASE_SEARCHABLE` without adopted graph | Text/sparse/dense legs may run; a graph-requiring plan is rewritten only when the declared query policy permits a traced degraded plan, otherwise it returns `graph_capability_unavailable` |
| Index without `visual_capability_revision` | Text-supported queries remain valid; a pixel/layout-dependent evidence requirement returns `visual_capability_unavailable`, never a caption-only answer presented as visual proof |
| Index with an orthogonal visual capability | Visual retrieval/hydration may run within any budget's limits; `fast` may decline it when the question does not require pixels |
| No qualified answer endpoint | `/search` remains valid; `/query`/`ask` returns `answer_capability_unavailable` rather than treating retrieved text as a generated answer |

The deployment profile controls *where* components run, the build policy fixes
*what evidence is constructed*, and the query budget caps *how that published
evidence is searched and reasoned over*. All three resolved identities and every
effective limit are stored in the appropriate build or query trace. This avoids
hidden behavior such as turning on Paddle, a VLM or global graph reasoning simply
because a GPU is visible.

Use a locked core Python project and optional dependency groups such as `dev`, `docling`, and `api`; keep CUDA/Paddle/MinerU runtimes in their own locked images. `uv.lock` gives application-level reproducible resolution while dependency groups avoid installing every development/model package into the core ([uv project documentation](https://docs.astral.sh/uv/concepts/projects/)). Docker Compose profiles make heavy services opt-in, and health-gated dependencies prevent the controller from racing unready services ([Docker Compose profiles](https://docs.docker.com/compose/how-tos/profiles/), [Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)). Pin images by version and production digest; never depend on `latest` for a reproducible run.

The repository should contain:

```text
pyproject.toml, uv.lock
compose.yaml
deploy/profiles/{lite,workstation,production}.yaml
config/build/{native,routed,routed-adjudicated}.yaml
config/query/{fast,balanced,deep}.yaml
.env.example
scripts/install-nvidia-container-toolkit.md
compatibility/models.yaml
```

Configuration precedence is predictable within each disjoint namespace: schema
defaults < immutable deployment/build/query policy < `docarag.toml` < explicit
CLI flag. Deployment, build and query files may not own the same field;
conflicting duplicate ownership is an error. Environment variables do not
broadly override typed configuration:
secrets enter only through explicit `*_env` or secret-file references and are
never copied into the resolved snapshot. Store the fully resolved,
secret-redacted configuration and exact source-file digests with every run.
Remove code-level GPU/URL overrides and absolute developer paths. Validate
configuration before allocating a model.

`init` resolves one deployment profile and creates an immutable
`BuildSpecification` from exactly one build policy, parser/index component and
model/config/schema digests plus all construction limits. `index` accepts only
that specification ID and publishes it in the index revision. `models pull` and
`up` use the declared deployment/capability set, but a deployment-only change
does not change semantic index identity unless an effective build component or
configuration changes.

`search` and `ask` do **not** consume a build-policy selector. They accept an
exact `index_revision_id` plus a request-scoped `query_budget` (`fast`,
`balanced` or `deep`) and optional query-capability preferences. The resolved
budget and any typed degradation are written to the retrieval trace, not the
index revision. This separation prevents a request from silently reparsing a
document or choosing a different model set.

In the target deployment, model downloads are explicit and resumable through
`docarag models pull`; `up` reports a missing model instead of unexpectedly
downloading tens of gigabytes. Its certified compatibility entries record the
tested parser/model version, image digest, CUDA/Paddle/Torch requirements,
minimum VRAM, schema version, and golden-suite result. The manifest currently
checked into this worktree validates the generic protocol and contract only;
its parser entries are fixture-only and intentionally certify none of those
package/model/image combinations yet.

### 7.5 Stable public API and operational observability

Expose a small `/v1` API; keep internal stage and storage shapes private:

```text
POST   /v1/documents                  upload/register, idempotency key supported
GET    /v1/documents/{id}
POST   /v1/documents/{id}/index       build_specification_id -> job/index revision
GET    /v1/jobs/{id}
GET    /v1/jobs/{id}/events           SSE progress stream
POST   /v1/search                     index_revision_id + query_budget -> evidence refs
POST   /v1/query                      index_revision_id + query_budget -> capability-gated answer
GET    /v1/evidence/{id}              authorized source/crop/structured evidence
GET    /v1/traces/{id}                authorized stage or retrieval explanation
GET    /v1/capabilities
GET    /health/live
GET    /health/ready
```

`lite` and `workstation` bind loopback by default. Any non-loopback binding and
every production deployment require TLS plus an explicitly configured
authentication provider; authorization is checked again at document, index,
job, evidence and trace access. Tenant/access filters are derived from the
authenticated principal, never accepted as trusted client payload. Signed
artifact URLs are short lived and scope bound, and deletion/revocation closes
metadata access immediately even while derivative cleanup is pending.

OpenAPI schemas, typed error envelopes, idempotency behavior and retention,
pagination, rate limits, cancellation, and version/deprecation policy are part
of the public contract. `/v1/search` and `/v1/query` accept at most 1 MiB JSON
and return at most 8 MiB JSON; internal candidate-worker requests and responses
are at most 16 MiB each. Large uploads, source bytes, crops and raw parser
artifacts use bounded streaming or authorized CAS/object references and are
never base64-inlined into query JSON. Oversize content fails before DTO
materialization. SSE disconnect propagates cancellation to queued and active
work, bounded cleanup runs, and no partial answer is published as success.
The index request cannot contain a query budget, and search/query cannot contain
a build-policy override. CLI, Python SDK, Web UI, and tests use this API or the
same application service; none reimplement indexing/retrieval decisions. Long
indexing is always asynchronous. Query supports server-sent streaming with final
evidence, immutable claim-lineage and trace metadata.

Use W3C trace context/OpenTelemetry across API, work items, model calls, stores, and retrieval so a single trace ID joins distributed spans, logs, and metrics ([OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/)). High-cardinality evidence/model details belong in trace events or durable trace tables, not Prometheus labels. Redact source text, prompts, credentials, signed URLs, and tenant data by policy.

API and worker health are separate: liveness means the process can run; readiness means required stores and the advertised capability are usable. A GPU worker that cannot load its declared model must withdraw the capability so the scheduler can fail fast or use an allowed fallback.

## 8. Scale model

### 8.1 Multi-document BookRAG

Do not build one enormous tree. Preserve one BookIndex per document version:

```text
B_d = (T_d, G_d, M_d)

CorpusBookIndex C = ({B_d}, G_cross, I)

G_cross = versioned cross-document canonical entities/aliases/relations
I       = backend-neutral document/section/block/entity/visual candidate-index port
```

This keeps each document's physical/logical hierarchy intact while allowing corpus retrieval:

1. `Select_Documents` uses document/section dense + BM25 candidates and access filters.
2. BookRAG runs entity/section selection inside the top document BookIndexes.
3. Per-document `G_d` prevents unrelated documents from contaminating local entity resolution.
4. `G_cross` connects canonical entities across documents through non-destructive alias/mention mappings.
5. Cross-document questions run Map within each selected document and Reduce with document/evidence citations.

Node, entity, relation, and GT-Link IDs always include tenant/corpus and document-version scope. Updating one document creates a new `B_d`; the active-version pointer switches only after the required readiness state, and the previous version remains available for rollback.

The same architecture covers multiple input types. A format-specific ingestor first converts PDF, DOCX, PPTX, XLSX, HTML, images, and email/attachments into pages/slides/sheets plus canonical observations. The BookRAG construction path after reconciliation is identical. Native objects such as spreadsheet cells, PowerPoint shapes, notes, and DOCX headings are preserved as representations instead of being flattened prematurely.

### 8.2 Single workstation

- PostgreSQL is the metadata/work/trace/index authority and provides
  transactional leasing/fencing; Qdrant is the single-node vector database.
- CPU page probe and Docling workers can run in parallel within measured RAM limits.
- One GPU lease serializes MinerU/Paddle/VLM/embedding/reranking stages.
- The supported Compose lifecycle keeps PostgreSQL and Qdrant in durable named
  volumes and binds both services to loopback by default.

### 8.3 Multi-worker, one host

- Multiple CPU workers claim page/block jobs through PostgreSQL leases using
  `FOR UPDATE SKIP LOCKED`, attempt budgets and fencing tokens.
- A single GPU coordinator owns model lifecycle.
- Batch event/artifact writes and pool sizing are bounded from measured
  PostgreSQL contention; no process-local database is coordination authority.

### 8.4 Multi-host production

- PostgreSQL remains the metadata/workflow source of truth and moves to an HA
  or managed deployment with tested migrations and backup/restore.
- S3/MinIO stores immutable source and derived artifacts.
- Workers are partitioned by capability and autoscale from queue depth/age.
- Separate interactive and batch priority queues.
- Qdrant uses replicated/sharded collections only after load testing shows a single node is insufficient.
- Keep the stage/store interfaces stable so a durable workflow engine can replace the database queue later without changing parser or index domain logic.

Do not introduce Kafka, Kubernetes, or a workflow platform merely to fix the current single-host crash. First make work idempotent, leased, observable, and portable; horizontal orchestration becomes a deployment choice after that.

## 9. Evaluation and release gates

### 9.1 Build a parser golden set

Use the three benchmark PDFs plus the 1,871-page stress document. Add representative PDFs for scans, multi-column papers, financial tables, forms, handwriting if in scope, diagrams/workflows, multilingual content, rotated pages, cross-page tables, and malformed PDFs.

Annotate 50–100 stratified pages initially for **calibration and schema
development**, including every difficult content type. Qualification uses a
separate document-held-out set with at least 10 independent documents and 30
applicable pages for every declared critical modality; a modality with fewer
independent documents is reported but cannot pass a default-selection gate.
No page, near-duplicate document, template family, or question from the held-out
set may tune router thresholds, prompts, normalizers or model choice. Store
page/block ground truth, expected relations and the immutable split manifest in
a versioned evaluation dataset.

Parser metrics:

- text CER/WER and normalized edit similarity;
- reading-order accuracy;
- block type F1 and bbox IoU;
- heading hierarchy accuracy;
- table cell/row/column F1 and structure similarity;
- formula exact/normalized edit score;
- figure/chart asset recall and caption association F1;
- cross-page continuation accuracy;
- provenance validity rate;
- page failure, retry, and fallback rates;
- seconds/page, CPU RSS, GPU peak, and output bytes/page.

The benchmark report must show per-content-type results, not only one overall
average. The parser release gate uses six co-primary, page-level quality metrics
on `[0,1]` (resource and provenance gates remain separate):

```text
q_text     = clamp(1 - word_error_rate, 0, 1)
q_order    = pairwise reading-order accuracy over annotated comparable blocks
q_block    = macro block-type F1 with the predeclared geometry-match rule
q_table    = table-structure F1 over cells, row/column adjacency and spans
q_formula  = exact-match accuracy after the frozen LaTeX/MathML normalizer
q_caption  = figure/chart-to-caption association F1
```

Each metric is applicable only when its annotation exists. No aggregate can
hide a failed co-primary metric or provenance gate. For router-regret analysis,
the precommitted protocol additionally defines
`Q_page = sum(w_m * q_m) / sum(w_m)` over applicable metrics with weights
`text=.20`, `order=.15`, `block=.15`, `table=.20`, `formula=.15`, and
`caption=.15`; content strata, not raw page counts, receive equal weight in the
reported `Q_router`. Changing normalization, geometry matching, applicability or
weights creates a new protocol revision.

An eligible parser has the required modality capability and passes the hard
source-fidelity/resource ceilings. The per-page oracle is the eligible parser
with the highest `Q_page` (ties use the deterministic model-selection order in
Section 3.7.2); the best single parser is the one eligible parser with highest
stratified score when selected as the sole parser for every evaluation page;
unsupported or failed applicable content scores zero. Report both comparators.

All router differences are paired by page, but pages from one document are not
treated as independent. Use 10,000 stratified hierarchical bootstrap resamples:
resample held-out documents within content stratum, then pages within each
sampled document, keeping every parser/router result for a page together.
`PARSER-QUAL-1` passes only when the 95% lower confidence bound of
`Q_router - Q_oracle` is at least -0.02, the lower bound of
`Q_router - Q_best_single` is at least -0.01, and every predeclared critical
co-primary modality metric has a lower bound at least -0.01 versus its best
eligible single parser. On the declared native-document subset, define heavy
work as any MinerU/Paddle or second-parser/adjudicator invocation; the paired
bootstrap 95% lower bound for heavy-invocation **relative reduction** versus the
frozen always-heavy baseline must be at least 0.50. Failed pages score zero for
applicable metrics and may not disappear from either the quality or volume
denominator.

BookIndex construction metrics:

- tree parent/level accuracy, orphan/cycle/order violations, and section coverage;
- entity mention precision/recall and canonicalization pairwise F1;
- relation extraction precision/recall by type and direction;
- GT-Link precision/recall from entity/relation to evidence node;
- percentage of graph assertions with resolvable source evidence;
- graph connected components, density, degree distribution, merge/split churn, and change from the prior version;
- stage completeness, retries, permanent failures, lineage coverage, and time from ingest to each readiness state.

### 9.2 End-to-end RAG set

Retain QASPER, MMLongBench, and M3DocRAG, then add at least 150 DocARAG questions split across local text, table/numeric, formula, visual, multi-hop, global, and unanswerable queries. Every answer needs expected page/block evidence. Calibration and qualification are split by source document/template family, not by question alone; retrieval weights, prompts, escalation thresholds and verifier calibration are frozen before the held-out document set is opened. Report per-stratum sample counts, and do not issue a modality accuracy claim when its predeclared minimum independent-document count is unmet.

Retrieval metrics:

- Recall@5/10/20, MRR, and nDCG@10;
- evidence-page and evidence-block recall;
- routing recall/regret by query type;
- document-selection recall for multi-document questions;
- modality evidence recall separately for text, scanned OCR, tables, formulas, figures, charts, diagrams, and flowcharts;
- entity-to-GT-Link-to-gold-node recall;
- candidate count and latency by retrieval leg.

Answer metrics:

- exact match/F1/ANLS as appropriate;
- numeric and unit accuracy;
- citation precision and recall;
- claim faithfulness and unsupported-claim rate;
- percentage of table/formula/visual claims for which the original structured artifact or crop was hydrated;
- answer completeness and unanswerable accuracy.

Build a human-labelled, modality- and Map/Reduce-edge-stratified semantic-support
set with supported, contradicted and insufficient-evidence edges. Structurally
missing receipt/lineage cases are deterministic failures, not judge examples.
On the semantic set, `CLAIM-VERIFY-1` requires the stratified-bootstrap 95%
lower confidence bound for precision of the `supported` verdict to be at least
0.98 and for recall of contradicted/insufficient edges as **not supported** to be
at least 0.95. Abstention is safe but is reported and cannot be removed from the
answer-completeness denominator. The verifier's model/prompt/calibration digest
and human adjudication protocol are versioned with every verdict.

Use LLM judges only as diagnostics calibrated against human labels. RAGChecker is useful because it separates retriever claim recall/context precision from generation utilization, noise sensitivity, hallucination, and faithfulness ([RAGChecker](https://arxiv.org/abs/2408.08067)).

### 9.3 Required ablations

For every model or architecture decision, compare:

- router versus Docling-only, MinerU-only, Paddle-only, and oracle best-parser-per-page;
- current DocARAG/BookRAG implementation versus durable BookIndex with identical models/prompts;
- dense-only versus BM25-only versus hybrid;
- no reranker versus Qwen reranker;
- text-only versus visual leg on visual questions;
- deterministic context versus LLM contextualization versus late chunking;
- vector-only base versus full BookRAG `T+G+M`;
- text+graph Skyline versus text+graph+requested-modality Skyline;
- destructive entity merge versus retained mentions + canonical mappings;
- per-document BookIndex versus corpus-global graph retrieval on multi-document questions;
- eager versus lazy summaries for global questions;
- Qwen3 embedding 0.6B/4B/8B and chosen vector dimensions;
- GME versus Qwen3-VL-Embedding 2B, with storage and latency included.

### 9.4 Initial service-level targets

Phase 0 first commits `evaluation-protocol.yaml`: hardware/driver identity,
corpus count and content-type distribution, warm/cold cache state, concurrency,
five fixed repetition seeds, immutable calibration/held-out document and
template-family splits, minimum independent-document counts, hierarchical
bootstrap method, and raw-result artifact hashes. Quality comparisons use
document-cluster-aware stratified 95% bootstrap confidence intervals. The
initial non-inferiority margin is -1.0 percentage point for Recall@10, citation
precision and answer F1, and -2.0 points for any predeclared difficult modality.
An “improvement” requires the lower confidence bound to exceed +2.0 points on a
quality metric, or at least 20% p95-latency reduction while every quality metric
is non-inferior. Threshold changes require a versioned protocol revision before
running the experiment, never after seeing results.

These are executable engineering starting targets to recalibrate only through
that protocol:

| Dimension | Initial gate |
|---|---|
| Failure semantics | Any failed required stage returns nonzero; no empty/error string persisted as content |
| Clean setup | On a documented clean Linux host, endpoint-free `lite` reaches `search` through one installer plus the documented DocARAG commands in <=30 minutes excluding separately reported network transfer; <=5 GiB declared download, zero source/config edits, zero manual environments. A separate gate qualifies optional remote `ask` |
| Workstation setup | On the declared clean Linux/NVIDIA host, one installer plus `init -> models pull -> up -> doctor -> routed index -> search/ask -> down` completes on a real two-page multimodal fixture in <=45 minutes excluding separately reported network transfer; zero source edits/manual Python environments, all services have explicit ownership, `down` leaves borrowed/shared services untouched, and owned GPU allocation returns to within 1 GiB of the pre-`up` baseline within 120 seconds |
| Diagnostics | `docarag doctor` catches every fixture for missing service/model, incompatible version, port, disk and GPU capacity with remediation |
| Dependency isolation | Core/API tests run without Torch, Paddle, Transformers, MinerU or Docling installed |
| Extension contract | An independently built adapter wheel is installed/discovered through a named v1 entry-point group, passes doctor/conformance/index replay, is uninstalled without core edits, and N-1 migration/rollback fixtures pass |
| API compatibility | `/v1` OpenAPI contract and supported SDK compatibility tests pass; breaking changes require a new API/schema version |
| API security and resource bounds | Non-loopback and production bindings require TLS/authentication; authorization-negative tests across document/index/job/evidence/trace endpoints and adversarial tenant filters return zero cross-tenant disclosures. Search/query JSON is rejected before DTO construction above 1 MiB request or 8 MiB response; candidate-worker JSON is capped at 16 MiB each way; streamed disconnect/cancel, rate-limit and idempotency-expiry fixtures leave no false success or leaked work |
| Resume | `kill -9` loses no committed items and at most one microbatch of active compute |
| Provenance | 100% canonical blocks resolve to document version + page; geometric blocks have valid bbox/polygon |
| Catalog-bound query scope | Query authorization carries one immutable catalog root/count per selected document and no full material-ID roster; constructing the maximum selected-document scope is `O(selected documents)`, remains within the 1 MiB public request limit, and every returned material passes a revision-bound membership lookup/proof before byte hydration |
| Source fidelity | 100% answer citations resolve to immutable source bytes and occur in both the exact hydration receipt and producing Map call's `MaterialUseReceipt`; every Reduce claim binds ordered child-claim digests, and the final claim's acyclic transitive closure resolves to those Map leaves. Pixel/layout claims include the asset-byte digest; receipts without semantic-support verdicts are not authoritative |
| BookIndex integrity | Zero active tree cycles/orphans; 100% entity/relation assertions and GT-Links resolve to active revisions |
| Graph authority | 100% adopted assertions bind an exact model invocation, consumed-material receipt, assertion-specific evidence subset and supported semantic verdict; zero `coarse_whole_node` releases are adoptable |
| Trace completeness | 100% required stage/model/operator executions have durable trace records and input/output lineage |
| Controller memory | RSS is bounded by configured encoded-byte, decoded-pixel, observation/cell and microbatch ceilings; doubling pages at fixed per-page limits changes controller peak RSS by <=10% |
| GPU safety | Peak below 42 GiB on the current workstation; VRAM drops after each model stage |
| Long native parse | `test.pdf` baseline parse target under 30 minutes before optional enrichments, versus observed ~3 h 43 m heavy parse |
| Partial preview | Optional `INDEXING_PARTIAL` search exposes an immutable page watermark within 60 seconds, always warns incomplete, and cannot answer unanswerable/whole-document questions |
| Time to searchable | Normal `search` switches atomically to a complete `BASE_SEARCHABLE` version; no partially indexed version is active |
| Parser router quality | The Section 9.1 co-primary metrics and paired stratified bootstrap pass: lower CI router-oracle `>= -0.02`, router-best-single and each critical modality `>= -0.01`, and native-subset heavy-parser reduction `>= 0.50` |
| Retrieval latency | Warm p95 candidate retrieval under 300 ms on the one-million-record, 10-concurrent-client qualification corpus and recorded workstation; profile-specific smaller/larger gates are explicit |
| Reranked retrieval | Warm p95 retrieval + rerank under 2 seconds; cold-start reported separately |
| End-to-end query latency | At the declared concurrency, warm `/search` p95 is <=2.5 s/5 s/15 s for `fast`/`balanced`/`deep`; qualified `/query` p95 is <=80% of the corresponding 15 s/60 s/180 s deadline, admission+queue p95 is <=10% of that deadline, and timeout plus `budget_exhausted` is <=1% on the predeclared answerable set. Cold start, throughput and abstention are reported separately; no unbounded graph/rerank/context path |
| Accuracy release | Deterministic BookRAG operator fixtures match exactly; statistical Recall@10/citation/answer metrics meet the declared non-inferiority margins. An accuracy claim additionally meets the +2 point lower-CI rule; no difficult modality crosses -2 points. Supported-answer claims also pass the versioned semantic gate at every Map-evidence and Reduce-child edge |
| Semantic claim gate | 100% malformed/missing claim lineage fails deterministically; on the human-labelled edge set, lower 95% CI for supported-verdict precision is `>=0.98` and for contradicted/insufficient detection recall is `>=0.95`; abstentions remain in completeness metrics |

## 10. Delivery roadmap

Section 1.2 is the authoritative dependency-ordered backlog. The phases below
are release gates and the PR section is a status/history mapping; neither may
silently introduce a competing order. Each numbered increment below is a
separate vertical change with its own rollback and exit test.

### Phase 0: freeze and measure the baseline

Deliverables:

- record an architecture decision for the modular-monolith/domain/port/adapter boundaries;
- freeze the current CLI, configuration, Web UI, API-less workflow, environment creation, model downloads, and first successful query as a timed usability baseline;
- capture package/model/config/GPU versions in every benchmark result;
- convert the three current parser runs into a checked-in benchmark manifest, excluding large artifacts;
- create the initial parser and RAG golden-set formats;
- add RSS, GPU, per-stage duration, queue, and model-load telemetry;
- record current QASPER/MMLongBench/M3DocRAG results where reproducible.
- commit `evaluation-protocol.yaml` before comparison runs, including hardware,
  corpus size/type distribution, concurrency, warm/cold state, five seeds,
  confidence intervals, non-inferiority/improvement margins, plus setup-time,
  manual-step, service-count and download ceilings;

Exit gate: one command produces a versioned baseline report with parser quality,
RAG quality, latency, memory, setup time, manual steps, download size, failure
diagnostics and the exact precommitted protocol; repeated runs reproduce every
deterministic operator trace and fall within the declared statistical interval.

### Phase 1: stop silent corruption and resource leaks

Status: **compatibility hardening substantially implemented; product operations
remain in progress**. The CLI/diagnostics, nonzero failure result, revision
driver, strict stage chaining, attempt history, source/artifact/runtime
verification, global indexing admission, process-group shutdown, bounded typed
LLM/VLM/KG execution, strict provider request timeouts/retries/lifecycle, scoped
runtime accounting/logging, explicit RAG ownership, summary/KG checkpoints,
structural answer citation validation, exact bounded global counts,
identity-bound streaming batch results, provenance-safe
MinerU reconciliation, bounded tree/vector/retrieval paths, and fixture-only
runtime compatibility checks are present. `down`, vendor service supervision,
locked qualified worker environments, capacity-aware GPU leases, and durable
page/block work-item recovery remain open.

Implemented:

- dependency-light `docarag` commands for `init`, `settings`, `doctor`, `index`,
  `status`, `verify`, `ask`, and `web` compatibility operation;
- validated compatibility settings, immutable source/config/code run identity,
  secret-redacted runtime fingerprints, strict detached stage chaining, and
  verified publication;
- typed and bounded LLM/VLM/KG execution, explicit failure classification,
  strict finite per-attempt remote request timeouts and retry caps, reliable
  local cleanup, lazy VLM allocation for only visual strategies, and source-bound summary/KG
  checkpoints;
- owned/borrowed `RAGSession`, context-scoped token and logging state propagated
  through bounded workers, and guarded remote-only provider imports;
- page-local MinerU reconciliation plus the dependency-light routed contracts,
  strict routed TOML loader, fail-closed provider, and Core injection seam;
- bounded lossless tree construction, vector microbatching, and retrieval
  candidate/Skyline selection;
- an atomic run-local exact-revision Qdrant readiness attestation, scoped
  stale-row reconciliation, and a query-lifetime read lease excluding
  concurrent rebuild;
- exact bounded global `COUNT`, fail-closed filters/empty evidence, and
  structural citation validation across simple/global/complex answers;
- hard planner/entity/filter/sub-question/prompt caps plus deterministic bounded
  source-stratified section selection;
- stable document SHA sharding and identity-bound atomic query caching with
  streaming JSONL/generator/DataFrame result publication.

Remaining:

- whole-product parser/model-service `up`/`down`, service logs, GPU capacity
  leases, and project-owned worker supervision (database `storage up/down` is
  implemented);
- safe cross-process Ollama/vLLM/Paddle ownership, unload/sleep coordination,
  and fully streaming main-driver dataset ingestion;
- separately locked and certified Docling, MinerU, and PaddleOCR-VL worker
  environments/images with real-model golden tests;
- one resolved routed configuration bound into immutable run identity and wired
  through the supported index command;
- a clean-host setup/first-query gate, cache retention/GC, bounded parallel
  complex sub-queries, and durable page/block work-item recovery.

Exit gate: `docarag doctor` diagnoses a clean/lite/workstation fixture; forced failures stop the chain; VRAM is released; the long run cannot lose successful microbatches or persist fake content.

### Phase 2: canonical schema and durable local workflow

Status: **shadow evidence schema, routed composition seam, and PostgreSQL
ledger foundation implemented; full workflow adoption open**. Typed source/page/geometry/representation/asset/parser-
observation/canonical-block contracts, a registry, three parser translators,
deterministic routing and reconciliation, canonical projection, run-local
artifacts, canonical/derived lineage through BookRAG retrieval, and a fail-closed
routed provider now exist. The provider stages, fsyncs, validates, and
rollback-protects replacement of the matching routed audit plus canonical
evidence and can safely reuse that pair. A failed forced attempt does not
become ready evidence. PostgreSQL now provides deterministic migrations,
immutable document/artifact truth, index revision records, append-only events,
and fenced work leases. The compatibility CLI uses document admission and
dispatch events; canonical rows, all page/block work and full `T/G/M` are not
yet adopted transactionally.

Add in independently releasable increments:

1. **`APP-1` application/package slice:** introduce shared `index`, `search`,
   `query`, `status` and `evidence` use cases behind the existing
   dependency-light domain/ports;
   package required resources or validate an explicit versioned BookRAG
   compatibility plugin; pass a clean-wheel smoke test.
2. **`LEDGER-1` PostgreSQL adoption slice:** the migration-tested ledger,
   content-addressed artifact receipts, index revisions, append-only events and
   leased/fenced work primitives are implemented. Wire page/block stages to
   them, add dependency hashes and retry/dead-letter policy, then publish
   normalized canonical plus versioned `T/G/M` rows and an immutable canonical-
   catalog root. Adopt existing verified artifacts; do not copy payloads into a
   second authority.
3. **`APP-1A` traceable BookRAG-lite slice:** run a qualified CPU-native parser
   through canonical evidence, BookRAG `T`, BM25 plus Qdrant dense retrieval, bounded candidate
   fusion and source-byte hydration through the shared application use cases.
   Add the minimal document/job/search/evidence API with the 1 MiB/8 MiB public
   and 16 MiB internal candidate transport caps; CLI and API traces must match.
4. **`MODEL-PORT-1` replaceable-model slice:** add versioned generation, vision,
   embedding and reranking ports plus shared runtime decorators, fake
   conformance and one independently packaged qualified remote adapter.
5. **`LITE-1` product slice:** graduate APP-1A into the curated CPU Docling +
   PostgreSQL + Qdrant profile, complete `/v1/query` only when a qualified answer
   endpoint is configured, and make CLI/Web thin clients.
Adapt `DocumentTree` into a repository/view over stored nodes rather than the system of record. Keep `tree.pkl` as a compatibility export only.

After `APP-1` and `EXTRACT-1`, deliver `INGEST-EXT-1` as independent format
adapter slices. DOCX, PPTX, XLSX, HTML, image and email/attachment fixtures must
enter the same canonical repository and resolve citations to their immutable
native part/shape/cell/DOM/message or attachment locus; no format-specific
BookRAG fork is allowed.

Exit gate: a clean host can run the endpoint-free `lite` index-to-search journey
without source edits; repeated kill/resume tests produce the same canonical rows
and IDs as a clean run, with bounded RSS, catalog-root membership for every hit,
and identical CLI/API operator traces. Optional `ask` has its own
`MODEL-PORT-1` qualified-endpoint gate and is never required for endpoint-free
lite.

### Phase 3: parser adapters and page router

Status: **dependency-light probe, transport, provider, and composition seam
implemented; vendor execution and evaluation open**.

Implemented under `docarag`:

- source-bound parser DTOs, capability registry, and fake adapter contract;
- Docling export, MinerU-normalized, and PaddleOCR-VL page translators;
- deterministic capability/cost/latency-aware page routing;
- bounded primary/fallback/adjudicator orchestration and page-local fusion;
- loss-aware canonical-to-BookRAG projection;
- source-bound lazy PyMuPDF page feature probing;
- versioned generic parser-worker HTTP client/server with bounded payloads,
  timeouts, identity checks, health and capability endpoints;
- fixture-only runtime compatibility manifest plus doctor validation;
- fail-closed routed provider, staged rollback-protected ready-pair reuse,
  opt-in compatibility runtime composition (legacy build-side name
  `balanced`, deprecated in favor of `routed`), and Core provider injection;
- adapter, probe, transport, provider, router, orchestration, adversarial
  reconciliation, and import-boundary fixtures.

Still add:

- pinned deployment images/configuration and live conformance/accuracy gates
  for the **existing strict clients** for Docling async conversion, MinerU task
  results and PaddleOCR-VL's complete page pipeline; do not add another adapter
  layer or return to vendor-private Python internals;
- pinned images and exact package/model/config identities, including MinerU
  `effort`, for each tested parser route;
- worker process/container supervision, cancellation and lifecycle control;
- routed parser settings and supported `docarag index` composition;
- durable raw-result, observation, route, and canonical revision repositories;
- clean-host real-model, golden accuracy, load, cancellation, and restart tests;
- certified package/model/image pins and measured routing thresholds.

After `PARSER-QUAL-1` passes, deliver `WORKSTATION-1`: digest-pinned Compose,
owned/borrowed/shared lifecycle enforcement, health-gated startup, capacity-
aware GPU leases, cancellation, and a clean-host routed index/query test. It
depends on `APP-1A`, `MODEL-PORT-1`, `LEDGER-1`, live `RETRIEVE-1` and
`PARSER-QUAL-1`; it must pass the timed setup/owned-resource teardown gate in
Section 9.4. A merely configured but unqualified worker cannot satisfy the
profile.

Adapters run out-of-process behind the same contract when their dependency/GPU stack conflicts; the core never installs their SDK merely to discover a capability. Add one reference fake parser so contributors can run all contract and routing tests without models.

Run the router in shadow mode first: parse with the current production parser while logging the proposed route. Then dual-parse the golden pages, set initial thresholds, and enable routing by document class behind a feature flag.

Exit gate for `PARSER-QUAL-1`: use the exact co-primary metrics, normalized
router score, eligibility rule, immutable held-out document split, minimum
independent-document counts, paired 10,000-resample hierarchical bootstrap and
failure denominators in Section 9.1. The lower bounds must satisfy
router-minus-oracle `>= -0.02`, router-minus-best-single `>= -0.01`, every
critical modality `>= -0.01`, and native-subset heavy-invocation relative
reduction `>= 0.50`. Point estimates alone cannot pass the gate.

### Phase 4: durable BookIndex and hybrid candidate index

Port BookRAG construction and retrieval without changing its intended semantics:

- `Tree.py` becomes a view/repository over versioned `T` rows;
- `Graph.py` becomes a bounded directed-multigraph view over `G` rows;
- `source_ids/tree2kg` become explicit `M` GT-Links;
- `kg_extractor.py` writes mention entities and evidence-backed relation assertions incrementally;
- `kg_refiner.py` retains gradient-based ER but records non-destructive mention-to-canonical decisions;
- `gbc_plan.py`, `gbc_retrieval.py`, and `gbc_answer.py` execute traced BookRAG operators over the durable BookIndex.

Complete `GRAPH-BUILD-1` by preserving pre-repair assertions and emitting exact
per-invocation material receipts plus assertion-specific evidence subsets and
semantic verdicts. Then complete `GRAPH-ADOPT-1`: the ledger may adopt only an
eligible release whose source/canonical/tree identity, assertion receipt
closure, semantic support and integrity checks pass. The current coarse legacy-
builder release remains shadow/inspection-only. `RETRIEVE-1` may ship base legs
earlier, but its entity/graph legs remain closed until they bind the exact
adopted digest.

Extend the APP-1A backend-neutral BM25 slice with structural chunks, dense and
entity/section legs, RRF, bounded reranking and citation-bearing evidence
references. Every scope binds immutable catalog roots and every returned
material passes membership validation; no corpus-scale request contains a full
canonical-ID roster. Extend the current unnamed dense Qdrant schema with sparse
or named visual vectors and a PostgreSQL outbox consumer only under
`VECTOR-SCALE-1` after its measured gate.

Remove query-time all-pairs graph similarity. Keep Qdrant scope filtering,
deterministic rebuild/delete behavior and readiness binding under frozen fixture
regression; schema upgrades publish a new revision and retain feature-flag
rollback until Section 9.4 non-inferiority passes.

Exit gate: Tier A becomes `BASE_SEARCHABLE`; after `GRAPH-ADOPT-1`, the durable
`T+G+M` reaches `BOOKRAG_READY`; frozen deterministic BookRAG operator traces match exactly, and
Recall@10, citation precision and answer F1 meet the Section 9.4 -1.0 point
non-inferiority margin before accuracy improvements are enabled.

### Phase 5: multimodal and multi-document BookRAG

Add:

- visual page/region indexing only for router-eligible pages;
- GME versus Qwen3-VL-Embedding experiment;
- visual/table/formula reasoners and Skyline score dimensions;
- original crop/cell-grid/LaTeX hydration into Map/Reduce;
- cached on-demand summaries;
- corpus document selection and per-document BookIndex execution;
- cross-document mention/canonical-entity links without destructive global merges;
- bounded local/global query policies and configurable relevance budgets;
- `CLAIM-VERIFY-1` immutable Map/Reduce claim artifacts, transitive child-digest
  receipts and calibrated semantic-support verification.

Exit gate: the lower 95% confidence bound improves Recall@10 by at least 3.0
points on the predeclared table/formula/visual subset, every generated claim on
those modalities has an immutable acyclic claim artifact whose complete
Map/Reduce closure reaches matching source/hydration/model-use receipts, and
every Map-evidence plus Reduce-child edge passes the versioned semantic-support
gate and the verifier itself passes the Section 9.2 precision/unsupported-recall
confidence bounds. Receipts alone do not pass. Cross-document citation precision plus
single-document BookRAG metrics must also meet the Section 9.4 non-inferiority
margins.

### Phase 6: production scale-out

Take the implemented PostgreSQL ledger to full live adoption and HA: add
canonical/`T/G/M` repositories, the transactional Qdrant outbox, object storage,
capability queues, separate batch/interactive pools, access-control filters,
backup/restore automation and operational dashboards. Scale Qdrant from the
default single node only after the measured saturation gate.

Exit gate: a 24-hour multi-worker test kills 10% of active workers at seeded
intervals, loses zero committed items, creates zero duplicate authoritative
revisions, recovers each lease within two lease TTLs, rebuilds the selected
candidate adapter from the outbox, and returns zero cross-tenant hits across the
adversarial authorization suite.

## 11. Migration without a big-bang rewrite

The Qdrant cutover is a hard vector-readiness boundary. Historical runs that do
not publish `docarag-qdrant-index.ready.json` are view/audit-only; a legacy local
vector directory is never adopted or treated as ready. Re-run indexing from the
immutable source and verified configuration to create a new revision and
rebuild the scoped points in Qdrant before serving retrieval. PostgreSQL import
or tree/graph compatibility adoption does not waive that reindex requirement.

1. **Compatibility foundation implemented:** immutable run manifests, revision-scoped state, verified artifacts, strict failure behavior, Qdrant vector storage, and PostgreSQL-required dispatch tracing now wrap the current pipeline.
2. **Partially implemented:** deterministic canonical observations/blocks are written alongside existing MinerU JSON/tree output; the routed provider gives one run an atomic reusable ready pair; PostgreSQL work/receipt/revision primitives exist but page/block stages have not adopted them.
3. Write durable `T`, `G`, and `M` in shadow mode while current `tree.pkl`/graph JSON remain the BookRAG read path.
4. Keep the existing Qdrant unnamed-dense read path while building any sparse,
   visual, quantized or clustered schema as a new shadow index revision. Promote
   it only if `VECTOR-SCALE-1` passes; otherwise retain the last qualified
   Qdrant revision.
5. Compare current and durable BookRAG operator traces, retrieval nodes, graph paths, and answers on the evaluation set.
6. Switch reads per document/index version behind a feature flag.
7. Import existing `tree.pkl` and graph JSON into compatibility document versions where provenance is trustworthy.
8. Reparse runs affected by the positional MinerU merge; do not migrate potentially shifted provenance as authoritative data.
9. Retire pickle/graph-JSON compatibility writes only after PostgreSQL `T/G/M`
   adoption, Qdrant rebuild, rollback and backup/restore procedures pass.

## 12. Proposed interfaces

```python
class ParserAdapter(Protocol):
    def manifest(self) -> ParserCapabilityManifest: ...
    def parse(self, request: ParseRequest) -> ParseResult: ...


class PageProbe(Protocol):
    def probe(self, document: DocumentRef) -> Iterator[PageFeatures]: ...


class CapabilityProvider(Protocol):
    def manifest(self) -> CapabilityManifest: ...
    def health(self) -> CapabilityHealth: ...


class LanguageModel(CapabilityProvider, Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def stream(self, request: GenerationRequest) -> Iterator[GenerationEvent]: ...


class EmbeddingModel(CapabilityProvider, Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...


class Reranker(CapabilityProvider, Protocol):
    def rerank(self, request: RerankRequest) -> RerankResult: ...


class IndexStore(Protocol):
    def claim(self, worker: str, capabilities: set[str], limit: int) -> list[WorkItem]: ...
    def heartbeat(self, work_id: str, lease_token: str) -> None: ...
    def commit_result(self, work_id: str, lease_token: str, result: StageResult) -> None: ...
    def fail(self, work_id: str, lease_token: str, error: StageError) -> None: ...


class BookIndexRepository(Protocol):
    def tree(self, scope: AuthorizedQueryScope, budget: RequestBudget) -> TreeView: ...
    def graph(self, scope: AuthorizedQueryScope, graph_scope: GraphScope,
              budget: RequestBudget) -> GraphView: ...
    def gt_links_for_entities(self, scope: AuthorizedQueryScope,
                              entity_ids: Sequence[str],
                              budget: RequestBudget) -> tuple[GTLinkRef, ...]: ...
    def entities_for_nodes(self, scope: AuthorizedQueryScope,
                           node_ids: Sequence[str],
                           budget: RequestBudget) -> tuple[GTLinkRef, ...]: ...
    def verify(self, scope: AuthorizedQueryScope) -> BookIndexIntegrityReport: ...


class Retriever(Protocol):
    def retrieve(self, query: Query, scope: AuthorizedQueryScope,
                 plan: RetrievalPlan, budget: RequestBudget,
                 deadline: Deadline) -> CandidateFusionResult: ...


class EvidenceHydrator(Protocol):
    def hydrate(
        self, references: Sequence[EvidenceRef], scope: AuthorizedQueryScope,
        needs: set[EvidenceNeed], budget: RequestBudget, deadline: Deadline
    ) -> tuple[Evidence, ...]: ...
```

`AuthorizedQueryScope` is immutable and binds tenant/corpus/access policy,
allowed document versions, one constant-size canonical-catalog root/count per
document, tree/index revisions and readiness. It never embeds complete
canonical material-ID rosters. Candidate adapters return the selected IDs and
the hydrator validates catalog membership by revision-bound point lookup or
bounded proof before re-reading exact bytes. An
eligible, ledger-adopted evidence-graph release is optional at
`BASE_SEARCHABLE` and mandatory for entity/graph legs. Repository and candidate
adapters return stable bounded references plus trace identity. Those candidate
material associations are explicitly **provisional**: a matching ID or rank is
not evidence that the referenced bytes exist, are in scope or support a claim.
Only the hydrator may promote a reference to evidence after revalidating tenant,
document/source/canonical/tree/index revision, selected material identity and
exact bytes. No storage port returns already-hydrated evidence or accepts an
unscoped entity/node ID.

Requests and results above are versioned domain DTOs, not vendor SDK classes. `CapabilityManifest` declares component/model/version, supported operations/modalities, schema version, limits, estimated resources, deterministic/batch/stream support, and health endpoint. A provider may wrap an in-process CPU library, local subprocess, container, or remote API without changing BookRAG. `MODEL-PORT-1` applies this same contract and conformance-wheel gate to generation, vision, embedding and reranking rather than leaving those providers as compatibility-only backend branches.

Stage implementations return immutable result references. They do not decide the next stage, mutate global vector collections, load the full tree, or control GPU lifecycle themselves. Retry, timeout, idempotency, tracing, and admission policies live in application/runtime decorators shared by all adapters; vendor adapters only translate and invoke.

## 13. Implementation slices and current status

Avoid a large foundation rewrite. Deliver reviewable vertical pull requests
while keeping the current BookRAG output path working.

### PR 1: one supported entry point and truthful diagnostics — foundation implemented; real-model gate open

1. **Implemented:** packaged `docarag` CLI wrapper with `init`, `settings`,
   read-only `setup-plan`, `doctor`, `index`, `status`, `verify`, `ask`, and
   `web` compatibility commands. `setup-plan` emits bounded argv, ownership,
   GPU-contention and qualification information without mutating the host.
2. **Partial:** one `docarag.toml` selects project/config/run/runtime settings and the supported path no longer mutates GPU/URLs in Python. Portable build policies, request query budgets, and cleanup of legacy scripts/configs remain.
3. **Implemented foundation:** red/amber/green, JSON, offline and strict diagnostics with path/command/endpoint/model/GPU checks. A packaged runtime compatibility manifest and worker health/capability checks fail explicitly, but all parser entries are fixture-only; certified real package/model/image pins and the clean-host gate remain open.
4. **Implemented:** dataset-mode failure is nonzero; the detached driver stops on failed commands or artifact verification, mirrors every transition to attempt history, serializes default GPU admission, forwards termination, and publishes source/runtime/artifact identities.
5. **Open exit-gate test:** index a real two-page fixture through the supported command and resolve one answer citation to source evidence. Existing mocked driver/readiness tests do not prove parser/model integration or answer accuracy.

Acceptance: a new user follows one README path without editing Python or shell
files; a broken dependency produces a remediation message and nonzero status.
The dependency-free controller half passes a fresh-wheel smoke test. Full
acceptance remains open until a clean host provisions the model runtime, indexes
a real fixture, and answers from it through the supported commands.

### PR 2: bounded execution and reproducible trace — compatibility slice implemented

1. **Implemented:** string/empty provider failures were replaced by typed
   request/result/error handling; remote LLM/VLM attempts pass finite request
   timeouts (not overall wall-clock deadlines),
   hidden SDK retries are suppressed, and explicit retries are bounded to at
   most ten attempts.
2. **Implemented:** summaries, KG extraction/refinement and provider admission use bounded iterators/semaphores rather than document-sized future queues.
3. **Implemented:** successful summaries use the deletable, non-authoritative
   local recomputation cache `summary-checkpoints.sqlite3`, keyed by prompt/
   model/image-byte identity; successful KG extraction uses atomic, source/
   model/prompt/payload-bound checkpoints.
4. **Implemented:** the positional MinerU merge was replaced by page-local reconciliation with strict and review modes.
5. **Partial:** local providers and BookRAG stages now close resources reliably;
   Ollama defaults to finite five-minute residency with explicit opt-in unload,
   while safe cross-process Ollama/vLLM/Paddle sleep/unload control and GPU
   memory spans remain.
6. **Partial:** run/stage artifacts, canonical observations, derived lineage and query provenance exist; durable `pipeline_run`/`work_item`/model-invocation records remain.
7. **Implemented local bounded retrieval slice:** subtree lexical/structural
   preselection caps its reranker input in `O(text bytes + N_S log M)` time and
   `O(M)` retained-candidate memory, graph/text Skyline selection is
   deterministic and hard-capped in `O(M log M)`, and lossless tree
   construction is `O(B + T)`. The complete query still materializes the
   selected subtree/graph and lacks a propagated request deadline; the
   candidate-first port and end-to-end bound remain open.
8. **Partial:** structural citation bindings and validation are now persisted in answer traces. Still freeze an end-to-end BookRAG fixture covering planning, mappings, graph/text scores, Skyline selection, canonical hydrated citation IDs, immutable verified asset bytes, and semantic claim support.

Acceptance status: forced provider/KG failures no longer publish fake content,
and committed summary/KG calls are reusable after interruption. Full `kill -9`
page-workflow recovery, immutable asset delivery to the actual model, and
semantic claim support resolving through the exact hydrated artifact to source
hash/page/bbox remain open.

### PR 3: routed parser-to-BookRAG seam — dependency-light slice implemented

1. **Implemented for parsing:** dependency-light domain DTOs, parser capabilities, `ParserAdapter`, registry and fake-adapter tests.
2. **Implemented:** Docling, MinerU-normalized, and PaddleOCR-VL translators produce the same canonical observation contract.
3. **Implemented:** deterministic capability-aware routing, bounded conditional fallback/adjudication, fail-closed reconciliation, stable canonical revisions, and a loss-aware BookRAG projection.
4. **Implemented:** source-bound lazy PyMuPDF page probe and versioned generic parser-worker HTTP client/server with bounded transport and pinned worker identity.
5. **Implemented:** fail-closed routed canonical provider with staged, fsynced,
   rollback-protected audit/canonical ready-pair replacement and reuse, opt-in
   compatibility runtime composition (legacy build-side `balanced` alias,
   deprecated in favor of `routed`), and injected Core parsing-provider plumbing.
   A narrow shared application use case now verifies that the backend's exact
   canonical JSON artifact contains the same accepted DTOs, binds it to the
   exact replayable routed audit, and issues compact byte receipts for both;
   the supported CLI is not wired to it yet.
6. **Implemented foundation:** fixture-only compatibility manifest/doctor checks plus adapter, probe, transport, provider, router, orchestration, projection, adversarial complexity, official-protocol service-client fixtures, verified source handoff, immutable local-CAS publication, and import-boundary suites run without vendor model packages.
7. **Partial:** strict Docling Serve, MinerU task-API, and complete PaddleOCR-VL service clients exist with strict one-profile/one-worker foreground composition, read-only validation, exact source/CAS root bindings, and packaged example profiles. Pinned images, owned supervision, routed settings/index wiring, shared/object artifact delivery, CAS quota/garbage collection, a certified version matrix, the clean-host real-parser gate, and a minimal `lite` deployment profile remain open.
8. **Partial:** explicit Docling, MinerU, and Paddle parent/child, caption,
   footnote, reference, adjacent reading-order, and attested table-continuation
   assertions retain exact observation lineage through canonical reconciliation
   and the BookRAG tree projection. Typed table cells, bounded MathML elements,
   and explicit directed diagram/flow graphs now retain exact selectors and
   separate source-neutral comparison digests through the BookRAG tree and
   hydration; compact vector metadata carries IDs/digests rather than full
   graphs. Only Docling native cells are currently proven by the live service
   path; MinerU/Paddle rich extension fields remain fixture-qualified. Immutable evidence-graph
   DTOs define representation/asset/span/cell/bbox/diagram anchors, extractor-
   revision-bound mentions, directed parallel assertions, candidate GT-Links,
   and candidate releases bound to source/canonical and tree-descriptor
   identities. The current graph/mapping fields are deterministic semantic
   digests inside the whole-release envelope, not separately published CAS
   artifacts, and the descriptor is not a full adopted `T`. Legacy checkpoint
   anchors are explicitly coarse whole-node evidence and releases built from
   them are not authority-eligible without exact invocation receipts. A
   bounded fail-closed checkpoint-to-release builder, content-addressed
   publisher, and checked read-only legacy `G/M` projection are implemented.
   Full-tree-manifest adoption, sharded CAS publication of `G/M`, live
   `GRAPH-BUILD-1` exact-receipt invocation, `GRAPH-ADOPT-1`, and persisted
   entity resolution remain open.
9. **Open:** stable public HTTP API/SDK, full PostgreSQL-backed canonical and
   `T/G/M` persistence/adoption, the Qdrant outbox and qualified sparse/visual
   schema, immutable verified asset-byte hydration into model calls, a typed
   public answer result, and `CLAIM-VERIFY-1` transitive semantic claim-support
   validation. Structural answer citation binding is implemented in the
   compatibility path.

Acceptance status: the domain/port/application contracts can swap registered
adapters without changing BookRAG or importing model runtimes; the generic
transport and provider prove the isolated handoff and fail-closed publication
semantics. Full acceptance remains open until the supported index command uses
qualified real isolated workers, the golden router gate passes, and a clean
host completes setup without source edits.

### PR 4: provider truth, request-scoped runtime, and bounded planning — compatibility slice substantially implemented

This maintainability slice has landed in the compatibility path, with product
operations and several end-to-end exit gates still open:

1. **Substantially implemented:** provider configuration is strict and
   backend-specific. Accepted endpoint/key/sampling/output/device/timeout/
   keep-alive fields reach typed backend requests, invalid combinations fail
   before allocation, remote LLM/VLM attempts pass a 300-second default request
   timeout, and the remote reranker passes 30 seconds. Hidden OpenAI-compatible
   SDK retries are disabled, and explicit retries are limited to `1..10`.
   These client timeouts are not overall wall-clock deadlines. Local model
   imports are lazy; a guarded subprocess proves injected remote provider paths
   import no Torch, Transformers, ModelScope, or FlashAttention. Full backend
   port separation, `doctor` coverage for every capability, package locks, and
   a clean-host real-service test remain open.
2. **Partial:** Ollama LLM/VLM requests default to `keep_alive="5m"` and
   `unload_on_close=false`; explicit unload is tested and sends `keep_alive=0`.
   Process-local GME users now have distinct reference-counted owner handles;
   final close waits for admitted work, and one forward pass per shared runtime
   bounds transient GPU memory. This deliberately serializes same-runtime GME
   inference and has no cross-process or warm-retention policy.
   There is no cross-process ownership lease, so shared-server-safe `down`,
   vLLM/Paddle lifecycle control, and GPU capacity admission remain open.
3. **Implemented compatibility path:** `TokenUsageScope` and `RunLogScope` bind
   counters and named-logger files to context-local run/query identity; bounded
   worker threads receive copied context. Overlapping logger namespaces are
   deduplicated, concurrent isolation is adversarially tested, and library code
   no longer clears the root logger. The locked global token fallback remains
   only for legacy unscoped callers.
4. **Implemented:** context-managed `RAGSession` has explicit owned/borrowed
   semantics and exactly-once cleanup tests across construction, execution,
   active-failure, and close-failure paths.
5. **Partial:** query/entity/filter/sub-question/value/prompt limits and
   fail-visible model-output validation are implemented. Linked sections are
   never silently dropped; bounded supplementary selection uses relevance plus
   source-stratified coverage with persisted policy/status/IDs. Independent
   complex sub-queries are still sequential, and complete admission/queue/
   deadline trace events remain open.
6. **Implemented compatibility path:** batch reuse is bound to exact question
   bytes, verified artifact set, secret-free resolved config/model, and current
   first-party query-code identity. SHA-derived directories, digest-bound atomic
   query commits, atomic streaming final JSON, JSONL/NDJSON and generator input,
   and row-wise DataFrame iteration are adversarially tested. Missing/corrupt
   identities disable reuse. Legacy JSON arrays and the main pandas dataset are
   still eager; one output-directory writer, cache GC, and immutable served
   model-weight identity remain open.
7. **Implemented narrow slice:** document-to-worker assignment uses stable
   length-prefixed SHA-256 identity and no longer depends on corpus order.
   `main.py` still eagerly loads/groups pandas data; a durable leased streaming
   queue is open.
8. **Open:** move revision/attempt/state/publication transitions out of
   `Scripts/index_pdf.sh` into a typed, migration-tested application service;
   retain the script only as a thin compatibility launcher.

Implemented exit-gate evidence covers strict settings, finite retries/request
timeouts, remote-only import boundaries, concurrent trace/token isolation,
owned-resource failure paths, adversarial plan caps, and reordered/corrupt
batch-cache cases.
The PR is not product-complete until a clean remote-only/lite host provisions
and queries a real service, cross-process ownership makes unload safe, the main
input path is streaming, and the typed application service replaces shell-owned
workflow state.

Continue vertically in the authoritative Section 1.2 order: extend the landed
APP-1 extraction seam into the complete shared application surface and add the
minimal `LEDGER-1`, then prove `APP-1A` from qualified CPU-native parsing
through BookRAG `T`, BM25 candidates, catalog membership and source hydration
using the same CLI/API use cases. Land `MODEL-PORT-1` before any live KG or
answer-model integration. Only then qualify the implemented parser-native
semantic structures in live graph/retrieval adapters and expose routed workers after
their held-out real-document gate passes. This protects BookRAG behavior,
extraction accuracy, setup simplicity and source traceability without growing a
second compatibility framework.
