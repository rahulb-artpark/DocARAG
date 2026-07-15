<h1 align="center">BookRAG</h1>

<p align="center">
  <strong>BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.03413">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A%202512.03413-b31b1b?style=flat-square" alt="Paper">
  </a>
  <a href="./BOOKRAG_VLDB_2026_full.pdf">
    <img src="https://img.shields.io/badge/PDF-Paper%20PDF-7c3aed?style=flat-square" alt="Paper PDF">
  </a>
  <a href="https://github.com/sam234990/BookRAG">
    <img src="https://img.shields.io/badge/Code-BookRAG-111827?style=flat-square" alt="Code">
  </a>
</p>

> **BookRAG** is the official repository for **"BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents"**, accepted by **VLDB 2026**.

## Current DocARAG quick start

The runnable path today is the **BookRAG compatibility runtime**, not yet the
promised standalone `lite` product. If the existing `gbc-rag` environment and
its configured model services are already installed, the shortest supported
journey is:

```bash
cd /home/rahul/DocARAG
conda activate gbc-rag
python -m pip install -e .
docarag init --project-root "$PWD"
docarag storage up
docarag storage status
docarag doctor --offline
docarag doctor --strict
docarag index /absolute/path/to/document.pdf
docarag status --watch
docarag ask "What evidence supports the conclusion?"
docarag web --host 127.0.0.1 --port 8765
```

`docarag storage up` starts the loopback-only PostgreSQL and Qdrant services,
waits for them, and applies the checksummed PostgreSQL migrations. `Ctrl-C`
stops `status --watch` or the foreground Web UI; use `docarag storage down` to
stop the storage containers. The editable install installs the lightweight
PostgreSQL and Qdrant clients but does not provision BookRAG, parser, CUDA, or
model dependencies. Routed
Docling/MinerU/Paddle parsing is not yet selected by `docarag index`. See the
[compatibility CLI details](#supported-docarag-cli-compatibility-foundation)
below for verification, worker commands, limitations, and remediation. The
[scaling plan](./SCALING_RAG_PLAN.md#12-immediate-implementation-order) defines
the clean-host `lite` and managed-workstation replacement for this transitional
setup.

### Default storage: PostgreSQL plus Qdrant

DocARAG has one supported storage topology. **Qdrant is the sole and default
vector database. PostgreSQL is the required and default durable ledger for
tenant/corpus/document versions, immutable artifact receipts, index revisions,
append-only stage events, and leased work items.** BookRAG's `B=(T,G,M)` remains
the retrieval and reasoning core; Qdrant stores rebuildable vector candidates,
not the source of truth.

The current compatibility `docarag index` path requires a current PostgreSQL
schema. It records document-version admission and dispatch plus append-only
`parse`, `tree`, `summary`, `graph`, `entity`, and `vector` stage events with
per-stage artifact receipts and index-revision publication; a verified live run
completes tree -> graph -> vector and publishes `artifact-manifest.json`. Its
tree-vector read/write path uses Qdrant with tenant/corpus/document/source/index
payload scoping. The PostgreSQL schema and transactional lease/index APIs are
implemented, and the pipeline stages now emit ledger events and receipts, but
item-level resumable leased work is not yet adopted by every stage. Routed
parsing also remains opt-in and is not selected by `docarag index`.

Use the supported storage lifecycle commands:

```bash
# Starts both services and migrates PostgreSQL.
docarag storage up

# Safe to repeat; useful after upgrading DocARAG.
docarag storage migrate
docarag storage status

# Stops/removes the Compose containers and network, preserving named volumes.
docarag storage down
```

`compose.yaml` persists data in the `docarag-postgres` and `docarag-qdrant`
named volumes. `storage down` does not delete them. Deleting them requires the
separate, destructive `docker compose down --volumes` operation and permanently
removes local database state.

The defaults bind both services to loopback. If ports `5432`, `6333`, or `6334`
are occupied, export matching service and client values before running the
storage commands:

```bash
export DOCARAG_POSTGRES_PORT=15432
export DOCARAG_DATABASE_URL='postgresql://docarag:choose-a-password@127.0.0.1:15432/docarag'
export DOCARAG_POSTGRES_PASSWORD='choose-a-password'

export DOCARAG_QDRANT_HTTP_PORT=16333
export DOCARAG_QDRANT_GRPC_PORT=16334
export DOCARAG_QDRANT_URL='http://127.0.0.1:16333'

docarag storage up
docarag storage status --json
```

The PostgreSQL password in `DOCARAG_DATABASE_URL` must match
`DOCARAG_POSTGRES_PASSWORD`. For repeatable local setup, copy `.env.example` to
`.env` and edit it. Docker Compose and the DocARAG CLI both read the same
allowlisted storage values from this file; an already-exported process value
wins. The CLI parses assignments as data and never executes `.env` as shell
code. `docarag.toml` stores only the
environment-variable name, never a DSN or password, and `docarag settings`,
health output, and errors redact credentials and query parameters. The checked-
in local password is for loopback development only. Use a secret manager,
TLS/authentication, and `QDRANT_API_KEY` for any shared deployment.

> **DocARAG development status:** this worktree extends the BookRAG research
> implementation into DocARAG. BookRAG's `B = (T, G, M)` BookIndex, query
> planning, structural selection, graph/text reasoning, Skyline ranking, and
> Map/Reduce remain the retrieval core. The compatibility path now adds typed
> Docling, MinerU, and PaddleOCR-VL parser contracts, deterministic page
> routing, bounded fallback/adjudication and reconciliation, loss-aware BookRAG
> projection, canonical/derived evidence lineage, bounded model/KG/vector work,
> restart-safe expensive-call checkpoints, a source-bound lazy PDF page probe,
> a versioned generic parser-worker HTTP transport, and a fail-closed routed
> canonical provider. It also includes a standalone bounded asset-byte verifier,
> fail-closed structural citation validation for simple/global/complex answers,
> and a run-local, revision-bound Qdrant readiness attestation plus a
> query-lifetime read lease for the scoped BookRAG tree-vector points. Query
> execution now has explicit `RAGSession` ownership, context-scoped token
> accounting and run logging, strict provider
> configuration/lifecycle contracts, hard planner/entity/filter/prompt bounds,
> and identity-bound atomic batch-result reuse with streaming JSONL/output
> support. The routed provider stages, fsyncs, validates, and
> rollback-protects a run-local audit/canonical ready pair, but it is not yet
> selected by the supported CLI. Strict foreground parser-worker composition,
> verified source-byte handoff, official-protocol service clients, and immutable
> local-CAS publication now exist. The application handoff now issues separate
> byte receipts for both canonical evidence and its replayable routed audit and
> rejects missing, unstable, arbitrary, or mutually inconsistent companions.
> MinerU task intermediates and exact Paddle page responses are carried as
> bounded result-scoped parser artifact receipts, including successful empty
> pages/results. Explicit Docling, MinerU, and PaddleOCR-VL
> parent/caption/footnote/reference/reading-order relations now survive
> reconciliation into BookRAG tree metadata; table-continuation links are kept
> only when parser-native IDs attest them. Bounded table-cell, MathML-symbol,
> and explicit diagram/flow graphs now use one parser-neutral semantic contract;
> exact vendor selectors remain source assertions while a separate conservative
> comparison digest prevents equivalent cross-parser tables from becoming false
> conflicts. Full semantic graphs are retained once in BookRAG tree metadata;
> vector records carry only bounded ID/digest receipts. Immutable evidence-graph contracts can
> represent entity mentions, directed relation assertions, and GT-Links bound to
> exact text spans, table cells, page regions, diagram elements,
> representations, and assets.
> A verified immutable checkpoint store and a checked read-only legacy `G/M`
> projection now exist, and tree lineage distinguishes all available canonical
> material from the representations/assets selected by the compatibility
> projection; that selection does not by itself prove model consumption. Building
> a bounded release from strict, source-bound BookRAG KG checkpoints is now
> implemented. The builder accepts encoded, byte-bounded checkpoints, re-reads
> canonical-input and tree-evidence-descriptor references as exact deterministic
> length-framed snapshots, and rejects a release if either snapshot's bytes or
> size differ; the descriptor snapshot includes `tree_revision_id` but does
> **not** contain full tree topology, ordering, summaries, or node content.
> Original PDF bytes, parser-native files, the complete BookRAG tree, and the
> legacy BookRAG pickle remain separately traceable inputs. Selected CAS assets
> are verified through exact materialization receipts. The legacy-checkpoint
> builder accepts whole selected materials only and rejects unproven
> span/cell/bbox/diagram precision. That coarse whole-node output is a
> candidate/audit artifact and is **not authority-eligible** without exact
> per-invocation material-use plus assertion-specific evidence/support receipts;
> invoking exact-receipt construction from the live KG stage and recording a
> fenced durable adoption are not implemented. Checkpoint adoption is
> fail-closed without an injected external authority, and its authorization
> must match the exact release ID, release-content SHA, checkpoint SHA, and
> source/canonical plus declared tree/graph/map selector binding SHA; this
> verifies an external decision but does not prove a complete `T/G/M` manifest
> and is not itself the durable adoption ledger.
> A bounded CAS-to-local image materialization lease
> now preserves the canonical asset ID while supplying verified bytes to legacy
> BookRAG KG extraction. Canonical-to-BookRAG projection withholds mutable local
> `img_path` values by default and accepts image paths only from an active
> verified CAS lease. The sole escape hatch is
> `BookRAGImageMaterialPolicy.LEGACY_INSPECTION_ALLOW_UNVERIFIED_MUTABLE_LOCAL_PATHS`;
> it marks every row `legacy_inspection_only` and must never feed indexing or
> evidence authority. Live composition must hold a verified lease through the
> model call/checkpoint. A backend-neutral candidate/RRF contract is source/canonical/
> tree/index bound, records the exact leg roster/status/latency, permits base
> BM25/dense retrieval before KG adoption, and requires an eligible,
> ledger-adopted graph release for entity/graph legs. Candidate material
> associations are provisional until
> the hydrator revalidates their full scope/revision and exact bytes; the
> candidate contract is not source evidence by itself and is not yet wired into
> live retrieval. Pinned service
> images, owned supervision, real-model version
> qualification, full live adoption of the PostgreSQL ledger by every legacy
> BookRAG stage, curated
> `lite`/`workstation`/`production` service profiles, immutable asset snapshots
> wired through visual embedding/VLM calls, transitive immutable Map/Reduce
> claim lineage and semantic claim-support validation,
> and the `/v1` API remain target architecture. See
> [`SCALING_RAG_PLAN.md`](./SCALING_RAG_PLAN.md) for the staged migration.

BookRAG targets RAG over long, highly structured documents such as books, manuals, handbooks, reports, and technical guidebooks. Instead of flattening documents into isolated text chunks, BookRAG builds a structure-aware **BookIndex** that combines document hierarchies, entity relations, and fine-grained evidence mapping for more effective retrieval and generation.

## News

- **VLDB 2026**: BookRAG has been accepted to VLDB 2026.
- **arXiv**: The preprint is available at [arXiv:2512.03413](https://arxiv.org/abs/2512.03413).
- **Environment**: Installation files are provided through [`requirements.txt`](./requirements.txt) and [`environment.yml`](./environment.yml).

## Why BookRAG?

Complex documents are rarely just bags of chunks. They usually contain explicit logical structures, nested chapters, cross-section references, tables, figures, and entity-level dependencies. Conventional RAG pipelines often lose these signals during parsing and indexing.

BookRAG is designed around three ideas:

- **Hierarchy matters**: document trees provide natural navigation paths from coarse chapters to fine evidence.
- **Relations matter**: entity-level graphs capture cross-section dependencies that pure chunk retrieval can miss.
- **Query strategy matters**: different questions require different retrieval workflows, from local lookup to global reasoning.

<p align="center">
  <img src="./assets/alpha.png" alt="Comparison of existing RAG methods and BookRAG" width="92%">
</p>

<p align="center"><em>Comparison of existing methods and BookRAG for complex document QA.</em></p>

## Method Overview

BookRAG has two stages:

1. **Offline index construction** builds a BookIndex from each document.
2. **Online retrieval and generation** uses the BookIndex to retrieve relevant evidence and answer user questions.

The current implementation supports multiple retrieval configurations and baselines through YAML configuration files under [`config`](./config) and dataset configuration files under [`Scripts/cfg`](./Scripts/cfg).

### BookIndex Construction

BookRAG first constructs a document-native BookIndex by combining a hierarchical document tree, an entity-relation graph, and mappings between entities and document tree nodes.

<p align="center">
  <img src="./assets/index.png" alt="BookIndex construction process" width="96%">
</p>

<p align="center"><em>The BookIndex construction process.</em></p>

### Agent-based Retrieval

Given a user question, BookRAG performs query classification and planning, then dynamically composes retrieval operators over the BookIndex to locate relevant evidence and generate the final answer.

<p align="center">
  <img src="./assets/online.png" alt="Agent-based retrieval workflow in BookRAG" width="96%">
</p>

<p align="center"><em>The general workflow of agent-based retrieval in BookRAG.</em></p>

## Supported DocARAG CLI (compatibility foundation)

The repository now includes a dependency-light `docarag` controller command. It
provides one supported entry point while preserving the current BookRAG index and
retrieval implementation. Model-heavy code is loaded only by indexing, querying,
or Web UI commands; `help`, `init`, `settings`, `status`, and offline diagnostics
do not import Torch, MinerU, PaddlePaddle, or Transformers. PostgreSQL and Qdrant
clients are loaded only by storage-aware commands and adapters.

The current/target boundary is intentionally explicit:

| Area | Implemented compatibility foundation | Not implemented yet |
| --- | --- | --- |
| Retrieval core | Existing BookRAG tree, knowledge graph, graph-to-tree mapping, planner, and retrieval/generation path; lossless `O(B + T)` source-block/tree-outline construction; the local subtree lexical/structural leg retains at most `M` reranker inputs in `O(text bytes + N_S log M)` time and `O(M)` retained-candidate memory; deterministic `O(M log M)` two-score Skyline selection; exact bounded global `COUNT`; hard query/sub-question/filter/entity/prompt limits; source-stratified bounded top-`M` section candidates; low-confidence entity matches are no longer force-selected; invalid structural filters, oversized plans, and empty/unusable evidence fail closed. A dependency-light candidate contract binds tenant/corpus/document/source/canonical/tree/index identity, exact query/config/plan leg roster and per-leg outcome, applies early nested bounds, and fuses/deduplicates refs with deterministic ordinary RRF in `O(H + U log K)` time/`O(H)` working memory for `H` leg hits, `U` unique evidence identities and at most `K` outputs; material unions are deferred to those `K` outputs. BM25/dense/sparse work at `BASE_SEARCHABLE`; entity/graph legs require an eligible adopted evidence-graph release. Candidate material links remain provisional until hydration revalidates scope, revision and exact bytes. The full current query does not invoke that contract yet: it still materializes the selected subtree and induced graph, runs PageRank, and lacks a propagated wall-clock/token/pixel budget. | Wire current entity/TreeVDB/section legs through the candidate contract, run scoped entity+dense+BM25 legs in parallel, add bounded graph/tree expansion, offline ANN graph links, durable directed-multigraph/corpus BookRAG, bounded parallel independent sub-questions, asynchronous hierarchical global reduction, and immutable transitive Map/Reduce claim artifacts with semantic support validation |
| Ingestion | One PDF through the supported MinerU compatibility runtime; common Docling/MinerU/Paddle contracts; source-bound lazy PyMuPDF probe; deterministic routing/fusion; versioned generic parser-worker HTTP client/server; strict official-protocol service clients; verified source-byte handoff; immutable local-CAS assets; result-scoped MinerU/Paddle raw-artifact receipts; one-vendor foreground worker composition with strict TOML profiles; fail-closed routed provider with staged, rollback-protected ready-pair replacement/reuse; injected Core provider plumbing; an opt-in Python composition root; a bounded application-level routed extraction receipt that binds both canonical and replay-audit bytes; and explicit structural plus table/formula/diagram subgraphs preserved through canonical reconciliation into BookRAG tree metadata | Transactional parser-artifact publication/orphan GC; streaming large parser/audit envelopes; pinned service images and owned supervision; supported routed `docarag index` composition through the application use case; live qualification of MinerU/Paddle rich semantic fields; real-model version pins; multi-host object/artifact transport; durable database-backed canonical repository; non-PDF ingestion |
| Reproducibility | Content-addressed PDF snapshot, immutable input identity, attempt history, runtime fingerprint, checksummed manifest, canonical/derived lineage, source-bound summary/KG checkpoints, deep verification of run-local artifacts, a fixture-only runtime compatibility manifest checked by `doctor`, and an atomic source/tree/config/embedding-bound readiness attestation for the scoped Qdrant points | Certified package/model/container pins, served-weight fingerprints in revision identity, durable page/block/entity/relation/GT-Link repositories, remote Qdrant content reconciliation, and leased resumable work items |
| Retrieval evidence | A bounded canonical catalog/tree hydrator validates document version, source hash, tree lineage, canonical block/revision, all available and actually selected representation/asset IDs, page/geometry/structure, and emits citation IDs. A standalone local/CAS byte resolver verifies assets without following symlinks. Structural citation markers are validated against exact supplied bindings, including collision-free complex-query scopes. Immutable DTOs define exact representation/asset/span/cell/bbox/diagram anchors, directed parallel relation assertions, candidate GT-Links, and revision-bound evidence-graph releases. The bounded builder consumes encoded checkpoints only after byte limits; requires canonical-input and tree-evidence-descriptor references to exact deterministic length-framed snapshots; verifies their bytes/size and selected CAS assets through exact materialization receipts; and incrementally guards release size. The descriptor snapshot includes `tree_revision_id` but is not a full `T` manifest. Original PDF/parser-native artifacts, full tree, and legacy pickle remain separately traceable. Current legacy checkpoint evidence is coarse whole-node selection, so its releases are publishable audit artifacts, not authority-eligible graph evidence. The legacy `G/M` projection is a checked compatibility view. Verified/CAS-leased image material is the default; the explicitly named unverified mutable-local-path enum marks the complete output `legacy_inspection_only`. | An adopted full-tree manifest binding root, topology, sibling order, node content, source/canonical revision and descriptor digest; exact-receipt live KG construction plus a fenced durable adoption ledger; preservation of pre-`res_repair` extractor assertions (the current checkpoint has already aggregated/collapsed some rows); exact per-model material-use receipts and assertion-specific evidence/support receipts; immutable verified asset snapshot-to-answer-model integration; structured table/formula verification; a typed public answer result; immutable Map claims with direct source receipts, Reduce claims bound to exact child-claim digests, and semantic support validation over the complete source-reaching claim DAG |
| Operations | Repository/editable-install CLI, read-only `setup-plan`, diagnostics, implemented PostgreSQL/Qdrant `storage up/down/migrate/status`, detached indexing, status, verification, ask, loopback Web UI, secret-safe routed-profile inspection, and read-only parser-worker profile validation plus foreground `serve`; one shared, dependency-inverted routed document-extraction application use case now returns compact content-bound canonical and replay-audit receipts, but the legacy CLI does not call it yet; explicit owned/borrowed `RAGSession`; context-scoped token ledgers and named-logger files propagated through bounded worker threads; strict backend-specific provider validation, finite per-attempt remote request timeouts and bounded retries; stable SHA document sharding; identity-bound atomic query cache and streaming JSONL/generator/DataFrame result publication. The installed wheel's entry point plus help, settings, and storage command family are tested without checkout-owned `Core` or PyYAML imports. Commands that operate the BookRAG compatibility run (`status` execution, named-run selection/verification, `index`, `ask`, and `web`) still require configured checkout resources; `index`/`ask`/`web` additionally require the heavyweight BookRAG runtime. There is no supported CPU-lite golden path yet. | Complete shared `index/search/query/status/evidence` application services, a standalone complete `lite` distribution and packaged compatibility-run services, locked dependency groups, whole-product parser/model-service `up`/`down`, explicit model pulls, `lite`/`workstation`/`production` deployment profiles, separate immutable `native`/`routed`/`routed-adjudicated` build policies and request-scoped `fast`/`balanced`/`deep` query budgets, capacity-aware cross-process GPU/model leases, CAS quota/garbage collection, fully streaming main dataset ingestion, clean-host runtime qualification, authenticated public API |

The maintainability decision is a **modular monolith controller with isolated
resource-conflicting workers**. BookRAG and the routing, provenance, and policy
code stay in one dependency-light application boundary. Only runtimes that
conflict over CUDA, Torch, PaddlePaddle, Transformers, vLLM, or model lifecycle
cross the versioned worker transport. This avoids both a single unmaintainable
GPU environment and a microservice per pipeline stage.

The target configuration deliberately separates three axes. Deployment profiles
(`lite`, `workstation`, `production`) say where components run; immutable build
policies (`native`, `routed`, `routed-adjudicated`) say what evidence an index
constructs; request query budgets (`fast`, `balanced`, `deep`) cap retrieval and
reasoning over an already published index. `/v1/documents/{id}/index` will accept
a `build_specification_id`, while `/v1/search` and `/v1/query` will accept an
exact `index_revision_id` plus `query_budget`. A deep query cannot silently
reparse a native index: missing graph, visual or answer capability is returned as
a typed degradation. These APIs and profiles are plan targets, not current CLI
claims.

The compatibility-only names `RoutedBookRAGFacade.balanced()` and
`config/routed-balanced.example.toml` are deprecated because `balanced` belongs
to the request-time query-budget namespace. They currently mean the `routed`
**build** behavior and never select a query budget. New public configuration and
APIs use `routed`; the aliases remain documented only until the next breaking
schema/API revision.

### Opt-in routed runtime profile

[`config/routed-balanced.example.toml`](./config/routed-balanced.example.toml)
is a loadable template for the three isolated parser workers. It fails on
unknown fields, validates the complete immutable composition DTO, keeps bearer
tokens outside the file by accepting only environment-variable names, and
places hard bounds on time, request/response bytes, pages, audit artifacts,
batch size, and reconciliation candidates. Its adapter versions and zero
manifest digests are placeholders—not qualified model pins—and must be replaced
before connecting to workers. Arbitrary `[parse_options].extra` values are
forbidden in profile files because an untyped extension could smuggle a secret
into normalized diagnostics; add an explicit typed field with a `*_env`
reference instead.

The profile is intentionally opt-in and is not selected by `docarag index`:

```python
from docarag.routed_profile import load_routed_runtime_profile

profile = load_routed_runtime_profile("config/routed-balanced.example.toml")
runtime_config = profile.config
revision_fields = dict(profile.revision_binding())
exact_profile_snapshot = profile.source_bytes
```

Before a deployment wires `runtime_config` into indexing, it must store
`exact_profile_snapshot` as an immutable run input and include all
`revision_fields` in the run identity. The exact-file SHA-256 captures the
operator-authored source; the normalized configuration SHA-256 captures runtime
semantics independently of TOML ordering and comments. The current compatibility
driver hashes only the PDF, BookRAG YAML, and controller code, so silently
selecting this profile through an environment variable would create an
untraceable index and is deliberately unsupported.

Inspect the profile without enabling it:

```bash
# The checked-in template intentionally exits 1 because its worker pins are
# placeholders. Structural errors exit 2 through the normal CLI error path.
docarag routed-profile validate config/routed-balanced.example.toml --json
docarag routed-profile show config/routed-balanced.example.toml

# Offline checks static pins/token-variable presence; omit --offline to probe
# worker health and capability manifests. This still does not enable indexing.
docarag doctor --routed-profile config/routed-balanced.example.toml --offline --strict
```

The dependency-light controller and parser-worker package supports Python
3.11--3.13. The existing BookRAG compatibility runtime remains pinned to
Python 3.12, so install the command there for `index`, `ask`, and `web`:

```bash
conda activate gbc-rag
python -m pip install --no-deps -e .
docarag init --project-root "$PWD"
docarag setup-plan --json  # read-only commands, ownership and qualification gaps
docarag doctor --offline  # local files, runtime, packages, tools and GPU
```

`pip install --no-deps -e .` only registers the controller in the already
complete `gbc-rag` environment; it does not install BookRAG, parser or model
dependencies. Start the model services required by the selected compatibility
YAML (for example, the configured Ollama models), then run the live check:

```bash
docarag doctor --strict
```

The controller wheel itself has a tested checkout-independent entry point:
help, settings, and storage command parsing do not import `Core` or PyYAML.
This is not yet the promised standalone **complete** `lite` distribution,
because `status` execution, named-run verification, `index`, `ask`, and `web`
still resolve compatibility resources from a configured BookRAG checkout. The
target release sequence is static offline preflight, explicit resumable model
pull, owned service startup, live strict capability check, then indexing.

### Isolated parser-worker commands

Three packaged example profiles compose exactly one Docling, MinerU, or
PaddleOCR-VL adapter per process. Copy the relevant profile, replace its exact
parser/model labels and local paths, and point it at a separately managed vendor
service. The examples remain explicitly `uncertified` until a live golden gate
records the installed package, model, and service-image identities.

```bash
# This is read-only: it does not resolve secrets, create roots, import PyMuPDF,
# contact a service, or allocate a model/GPU.
docarag parser-worker validate \
  docarag/profiles/docling.example.toml --json

# Run one worker in its isolated environment. Ctrl-C stops it and closes owned
# local resources. Use the corresponding MinerU or Paddle profile/environment
# for the other workers.
./.venv-docling/bin/python -m docarag.cli parser-worker serve \
  /absolute/path/to/docling.toml
```

`serve` resolves only the named secret environment variables, verifies source
snapshots against the configured allowlisted roots, and publishes parser
artifacts under `cas://sha256/<digest>`. Docling's canonical raw export and
requested embedded images are retained with exact digest/fragment lineage.
The current local handoff is deliberately same-host: controller and worker must
see source and CAS roots at the same absolute paths (or through identical
container mounts). Cross-host workers require the planned bounded object/artifact
transport; changing a logical `source_uri` never authorizes a worker to fetch it.

These commands run the DocARAG adapter process, not the underlying vendor
service. They also do not yet connect routed parsing to `docarag index`; pinned
parser/model-service startup, supervision, live-model qualification, and owned
whole-product `up`/`down` remain release gates. The PostgreSQL/Qdrant
`docarag storage up/down` lifecycle described above is already implemented.

`docarag init` creates `docarag.toml`. Relative system configuration and run
paths are resolved against its `project_root`, so commands do not depend on the
terminal's current directory. Review the generated settings and the selected
BookRAG YAML before starting a model-heavy workload.

The current compatibility commands are:

```bash
# Start detached tree -> graph -> vector indexing. The driver stops on the first
# failed command or invalid/missing stage artifact.
docarag index /absolute/path/to/document.pdf

# Inspect once or watch continuously.
docarag status
docarag status --watch

# Verify the exact source and published index. Deep hashing is the default for
# run-local artifacts; --quick still hashes artifacts parsed/deserialized by a
# query. Both validate remote Qdrant through its scoped readiness attestation
# and exact live point count; neither scrolls and hashes every remote point.
docarag verify --run /absolute/path/to/run
docarag verify --run /absolute/path/to/run --quick --json

# Query the latest searchable run, or pass --run explicitly.
docarag ask "What evidence supports the conclusion?"
docarag ask --run /absolute/path/to/run "What is shown in the table?"

# Run the existing Web UI in the foreground; Ctrl-C stops it.
docarag web --host 127.0.0.1 --port 8765
```

### Maintainability and bounded-work controls

The supported configuration is strict: unknown root keys, misspelled fields,
mixed legacy/canonical RAG shapes, and invalid concurrency/batch values fail
before model allocation. The checked-in runnable configurations round-trip
through load and save; [`config/default.yaml`](./config/default.yaml) is an
explicit template and must be rendered before use. Query-agent construction
also validates required indexes first and does not allocate a VLM for the
text-only Vanilla or GBC-Vanilla strategies.

Operators can bound the compatibility runtime without editing code:

```yaml
llm:
  max_workers: 4
  max_in_flight: 4
vlm:
  max_workers: 1
  max_in_flight: 1
graph:
  extraction_max_workers: 4
  extraction_max_in_flight: 4
  extraction_batch_size: 32
  refinement_max_workers: 4
  refinement_max_in_flight: 4
vdb:
  index_batch_size: 256
rag:
  strategy_config:
    candidate_pool_size: 512
    global_evidence_limit: 32
    planner_max_query_chars: 4096
    planner_max_subquestions: 8
    planner_max_subquestion_chars: 1024
    planner_max_filters: 8
    planner_max_filter_value_chars: 256
    query_entity_limit: 16
    query_entity_candidate_limit: 5
    section_selection_candidate_limit: 48
    section_selection_supplementary_limit: 8
    section_selection_prompt_chars: 65536
```

These are safety bounds **and** recall/latency/cost knobs: lowering them can
increase abstention or miss relevant evidence, while raising them increases
memory and latency. Calibrate them jointly on the labelled retrieval/answer
golden set and load tests; never scale them implicitly with document length.
Model-produced plans
or entity lists that exceed them are rejected before retrieval fan-out; they
are never answered from a silently truncated plan. Section selection always
represents every graph-linked section or fails visibly if those source-derived
links alone exceed the prompt budget. Remaining headings use deterministic
lexical/path relevance plus source-stratified coverage. Its top-`M` selection
is `O(S * (D + log M))` for `S` sections and maximum path depth `D`; the
stratification pass is `O(S log M)`. Represented IDs, omitted count, prompt
size, policy name, and status are persisted in `retrieval_res.json`. Successful
tree summaries and KG extractions are checkpointed by their source, prompt, model and
asset identities, so changing evidence or a behavior-relevant setting causes
recomputation instead of unsafe reuse. A node whose KG extraction fails after
its retries is logged and skipped with an empty result rather than aborting the
whole document; its checkpoint is not committed, so a later run re-attempts it.

LLM, VLM, embedding, and reranker configuration is backend-specific and strict:
unsupported combinations fail before allocation, while endpoint, credentials,
sampling/output limits, device, and timeout settings reach their corresponding
backend. Remote LLM/VLM attempts pass a finite configurable transport/request
timeout (300 seconds by default), OpenAI-compatible SDK retries are disabled so
they cannot multiply DocARAG's retry loop, and the explicit attempt count is
limited to `1..10`. The remote reranker passes its own finite request timeout
(30 seconds by default). These are per-attempt client timeouts, not an overall
wall-clock deadline or process-kill guarantee. Guarded subprocess tests prove
that the injected OpenAI/Ollama/vLLM remote provider paths do not import Torch,
Transformers, ModelScope, or
FlashAttention. This is a tested import boundary, not a qualified clean-host
installation or real-service compatibility matrix.

Query strategies close owned stores, embedders, local VLM weights, and HTTP
clients on success, failure, and partial construction. Ollama requests now use
a finite `keep_alive="5m"` default. `unload_on_close` is disabled by default;
when explicitly enabled, close requests server-wide eviction with
`keep_alive=0`. There is no cross-process ownership lease, so that option is
unsafe when unrelated users share the same Ollama model. Independently managed
vLLM/Paddle services and capacity-aware GPU admission remain deployment
responsibilities.

Compatible GME users now hold separate owner handles over one process-local,
model/device-keyed runtime. Closing one handle cannot unload another user's
weights; the final close waits for admitted work and then releases CUDA memory.
One forward pass is admitted per shared runtime by default, which favors
predictable peak VRAM and avoids cross-request model races. It intentionally
limits in-process throughput; qualified parallel scale-out belongs in isolated
workers/GPUs, and the pool has no cross-process lease or warm-retention grace.

Supported query and batch paths use an explicit owned-or-borrowed `RAGSession`,
a context-local `TokenUsageScope`, and `RunLogScope` handlers attached only to
DocARAG logger namespaces. The bounded executor copies context into worker
threads; overlapping parent/child logger namespaces are deduplicated so one
event is written once, and concurrency tests verify that token deltas and log
files do not cross requests. Calls outside a scope retain a locked legacy
token-accounting fallback, so new service code must always establish a request
scope.

Batch inference directories are derived from SHA-256 identity rather than row
ordinal. Reuse requires exact UTF-8 question bytes, the verified published
artifact set, secret-free resolved configuration/model identity, and current
first-party query-code identity; missing, mismatched, or corrupt bindings cause
recomputation. `result.json`, its digest-bound commit marker, token cost, and
`final_results.json` are private atomic publications. JSONL/NDJSON, generators,
and DataFrame rows stream without a document-sized result list; legacy JSON
arrays and `main.py` dataset loading remain eager. The compatibility driver
assigns documents to stable SHA shards independent of input order, but it still
loads and groups the full pandas dataset. Cache entries are not garbage
collected, pre-identity ordinal caches are not migrated, one result is loaded at
a time, duplicate questions share one last-seen per-query `result.json` while
the aggregate keeps current metadata for every row, and one writer per output
directory is assumed. Online artifact validation is intentionally bounded.
Neither quick nor deep verification scrolls and re-hashes remote Qdrant vectors
and payloads. Both validate the run-local readiness attestation and exact scoped
live point count, so same-count remote vector or payload tampering is not
detected. Detecting it requires a per-scope content digest, outbox
reconciliation, or an explicit bounded/full remote scroll audit. A configured
remote model name/URL likewise cannot detect weights silently replaced behind
that identity.

Parser extensions follow one rule: vendor SDK execution stays behind injected
loaders or isolated workers; the dependency-light DTOs in `docarag/domain`,
interfaces in `docarag/ports`, and routing/orchestration code never import
model/parser/storage frameworks. A new adapter declares a capability manifest,
translates into the same validated observation result, and passes the shared
contract suite. The deterministic page router selects registered adapters
without changing BookRAG's tree/graph/retrieval code. The current compatibility
facade's deprecated `balanced` routing preset prefers Docling for reliable native pages,
MinerU for technical table/formula pages, and PaddleOCR-VL for scanned or visual
pages; it invokes at most one fallback or adjudicator. In the target public
contract this maps to immutable build policy `routed`; it is not the request-time
`balanced` query budget.

The supported application composition boundary is `RoutedBookRAGFacade`:

```python
from pathlib import Path

from docarag.composition import RoutedBookRAGFacade
from docarag.ports import ParserRegistry

# Each adapter is constructed with an injected file/RPC loader. The loader owns
# the vendor environment; the controller receives JSON-compatible output only.
registry = ParserRegistry((docling_adapter, mineru_adapter, paddle_adapter))
# This selects routed build behavior, not the request-time balanced query budget.
facade = RoutedBookRAGFacade.routed_default(registry=registry)

# request is a source-bound ParseRequest; page_features are emitted lazily by
# the implemented source-bound PDF probe. Adapters may be HTTP worker clients.
result = facade.run(request, page_features, prior_blocks=prior_blocks)
Path("routed-parse-audit.json").write_text(result.to_json(), encoding="utf-8")
pdf_list = result.to_bookrag_pdf_list()  # raises unless every page is ready
```

`result.parsing` contains every route, request/result or typed failure,
observation ID, canonical revision, and review reason. Failed, partial,
unresolved, and review results cannot enter the normal BookRAG handoff. An
explicit `to_audit_partial_bookrag_pdf_list()` exists only for diagnostics.
Compatibility rows are projected lazily from canonical blocks and are not a
second source of truth.

This is a tested controller/application API, not yet a supported routed-index
CLI path. The source-bound lazy page probe, generic versioned HTTP
client/server, strict service clients, one-profile/one-worker foreground
composition, local CAS, opt-in runtime composition, and run-local fail-closed
provider exist, and Core accepts that provider through injection. The current
`docarag index` command still executes the existing MinerU compatibility runtime. It has
no supported settings that construct the routed runtime. The repository does
not yet ship pinned vendor-service bootstraps, owned supervision, certified
version pins, cross-host artifact delivery, or a clean-host real-model gate.

Verification is an integrity operation, not a parser-quality score. Quick
verification is O(number of manifest entries + bytes in query-load-bearing
artifacts); deep verification is O(total run-local source and artifact bytes)
because it recomputes every local hash. It checks run-local containment and
manifest self-consistency. For remote Qdrant it verifies the bound attestation
and exact scoped point count, not every point's vector and payload bytes; a
same-count remote mutation can therefore pass both modes. Verification does not
prove semantic extraction accuracy, provide cryptographic attestation, or
defend against an authorized writer replacing both data and its manifests.

Each indexing invocation identifies the immutable revision from the resolved
source path plus the SHA-256 hashes of the PDF bytes, active system
configuration, and current controller/code. Its run directory is:

```text
runs/<document-id>/<revision-id>/
```

`run-manifest.json` records that identity and the immutable source snapshot;
`index-state.json` records the active PID and tree/graph/vector stage states. The
generated dataset JSON/YAML and system-config snapshot live under the same run.
Each actual driver attempt has a private `attempts/<attempt-id>/` directory.
It records live and terminal state; stage logs and a runtime fingerprint are
present when the attempt progresses far enough to create them. An already
verified duplicate exits without creating another attempt. A successful attempt
atomically publishes `artifact-manifest.json`,
which binds the tree, selected graph variant, selected entity store, selected
retrieval store, canonical parser evidence when produced, source snapshot, and
runtime fingerprint by SHA-256. Query result/execution traces copy these
identities so a query execution can be traced to its document and published
artifact set. This is run-level provenance, not proof that every generated
claim is semantically supported. When canonical evidence hydration is active,
the answer path now requires every emitted citation marker to resolve to an
exact supplied canonical citation ID and fails closed on missing, unknown,
compound, malformed, or citation-only output. The trace explicitly records
that semantic entailment has not yet been verified.

Before an `ask` loads code or models, its readiness gate re-hashes the selected
tree pickle, tree JSON, graph JSON, configuration snapshot, and BM25 pickle (if
used). Hash results are cached by device/inode/size/mtime/ctime identity, so
unchanged repeated Web questions do not re-read those files. Vector readiness
is checked against the live Qdrant collection and the exact tenant, corpus,
document-version, source-hash, and index-kind payload scope, together with the
run-local Qdrant readiness attestation; it is never inferred from a local vector
directory or database file.

Qdrant writes use deterministic logical record IDs mapped to stable UUID point
IDs, idempotent upserts, cosine-distance collections, indexed scope payloads,
and exact filtered counts. A rebuild deletes only the bound scope; an unscoped
collection reset fails unless an explicit destructive capability was granted.
Queries automatically include the store scope, preventing records from another
document or tenant from entering a result. The BookRAG tree-vector builder
invalidates its run-local Qdrant readiness attestation before mutation,
reconciles stale points in that scope, verifies the exact filtered count, and
publishes `docarag-qdrant-index.ready.json` only after store/model cleanup. The
attestation directory contains no vector database; vector data remains in
Qdrant. Startup binds
the attestation to the saved tree, source, document revision, vector
configuration, embedding identity, Qdrant collection, and scope, then holds a
shared read lease through the query lifetime. Normal retrieval remains an ANN
query plus Qdrant payload filtering. Incremental outbox-based diffs, remote
served-weight attestation, and immutable image-byte-to-embedding handoff remain
open. Legacy visual paths can still change between hashing and model decode, so
the readiness record must not be interpreted as verified visual material.

Runs produced by an earlier DocARAG development build that already contain an
`artifact-manifest.json` but lack terminal attempt/manifest bindings are treated
as untrusted. Historical and manifest-free BookRAG runs without a published
`docarag-qdrant-index.ready.json` are view/audit-only after the Qdrant cutover;
legacy local vector directories are never accepted as readiness. Re-run
`docarag index` for the same PDF to create a verified new code revision and
rebuild its vectors in Qdrant before using the supported `ask` path.

Changing document bytes, configuration, or indexing code therefore creates a
different revision instead of silently reusing an old tree. Re-submitting an
already verified successful revision is idempotent; the deduplication check
performs deep artifact verification first. A damaged artifact starts a new
repair attempt in the same input revision. The controller quarantines the
damaged stage and its dependent outputs under that new attempt's `recovery/`
directory, so BookRAG cannot load and re-bless them; an intact upstream stage
can still be reused. It also requires `tree.pkl` and `tree.json` to describe the
same topology and immutable source identity before reuse or publication. Prior
attempt state, logs, manifests, and quarantined bytes remain available for
audit. `canonical_blocks.json` and the matching parser-observation JSON are
immutable, source-bound shadow artifacts for the current run. BookRAG nodes
point either to one unchanged canonical block or, after refinement/merging, to a
deterministic derived evidence ID with every contributing
block/observation/page/geometry reference. Compatibility tree/graph exports and
Qdrant remain the active read formats; the JSON artifacts are not yet the
future transactional canonical evidence repository described in the scaling
plan.

The main audit/resume artifacts inside a run are:

```text
canonical_blocks.json                         immutable canonical evidence shadow
<mineru-method>/<pdf>_parser_observations.json parser contract result and provenance
summary-checkpoints.sqlite3                    local recomputation cache, not system of record
kg_extractor_res/kg_extractor_res_<node>.json  source/model/prompt-bound KG result
tree.json / tree.pkl                           active BookRAG tree views
graph_data*.json                                active BookRAG graph compatibility view
<vector-attestation>/docarag-qdrant-index.ready.json run-local scoped Qdrant attestation; no vector data
artifact-manifest.json                         published role-to-hash binding
```

Deleting a checkpoint only causes recomputation. Editing one causes its digest
or identity check to fail and the item to be recomputed; it is never accepted
as a different source node merely because the integer node ID matches.

The compatibility driver admits only one supported indexing run at a time by
default across documents that use the same configured `runs_dir`. This reduces
GPU contention among those driver invocations, but it is not memory-aware and
does not coordinate `ask`, Web, manually launched scripts, external
Ollama/vLLM/Paddle services, or drivers using another runs directory. Set
`DOCARAG_INDEX_ADMISSION=none` only for CPU-only execution or when an external
scheduler already owns capacity. Before sending `TERM` to the PID in
`index-state.json`, confirm `overall_status` is `running` and verify that PID is
the matching indexing driver; terminal state retains historical PIDs and PID
reuse makes a blind `kill` unsafe. A graceful `TERM` is forwarded to managed
work as a process group, escalates to `KILL` after the bounded
`DOCARAG_TERM_GRACE_SECONDS` interval (three seconds by default), records the
interrupted attempt, and releases locks. If a driver is lost to `SIGKILL` or a
host failure, the next driver marks its stale `running` attempt interrupted
only after acquiring the revision lock; it does not resume the interrupted
item. Item-level resume and cross-host leases remain recovery gaps in this
compatibility layer.

The runtime fingerprint records Python/platform identity, every Python
distribution discoverable in the indexing interpreter, `CUDA_VISIBLE_DEVICES`,
and the GPU/driver inventory returned by `nvidia-smi`. It does not record a
byte-for-byte environment lock, CUDA toolkit/runtime libraries, active local or
remote model-weight identity, or Ollama/vLLM container-image digests. These
remain explicit reproducibility gaps rather than being inferred. The packaged
runtime compatibility manifest and `doctor` checks validate protocol and
declared worker identities without importing vendor SDKs, but its current
parser entries are fixture-only and deliberately contain no certified
package/model/image matrix.

Run directories can contain source-derived text, prompts, retrieval results,
and an owner-readable unredacted configuration snapshot. Treat the runs
directory as sensitive data. The compatibility Web UI is deliberately
loopback-only and has no authentication; do not expose it directly to a network.

Use `docarag --settings /path/to/docarag.toml ...` when maintaining more than one
checkout/profile. `docarag doctor --json` is suitable for automation. A non-strict
doctor run reports all red/amber/green checks without blocking exploratory use;
`--strict` returns nonzero when a required check is red.

These are compatibility-hardening slices, not yet the final isolated-parser
deployment. The existing `main.py` and scripts remain available for backward
compatibility. The generic isolation boundary and foreground worker composition
are implemented; future slices must pin and qualify the vendor services, add
owned supervision, wire routed parsing into the supported settings/CLI, and add
the `lite`, `workstation`, and `production` deployment
profiles described in
[`SCALING_RAG_PLAN.md`](./SCALING_RAG_PLAN.md).

## Legacy BookRAG compatibility environment

The supported `docarag index` path currently uses BookRAG's single MinerU
environment. This heavyweight environment is not the isolated three-worker
routed deployment.

BookRAG requires Python 3.12. The recommended setup is to create a fresh conda environment from [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
conda activate gbc-rag
```

Alternatively, install the Python dependencies directly:

```bash
pip install -r requirements.txt
```

BookRAG uses [MinerU](https://github.com/opendatalab/MinerU) for PDF parsing and document information extraction. The compatibility preset selected by a new `docarag.toml`, [`config/ollama-gbc.yaml`](./config/ollama-gbc.yaml), currently uses the local `vlm-transformers` MinerU backend; its `server_url` is therefore unused and the parser consumes local RAM/VRAM. Other checked-in configurations may use `vlm-sglang-client` and require the corresponding service. For graph extraction, the default local NLP model is `en_core_web_sm`, which is included in [`requirements.txt`](./requirements.txt).

The Ollama names in that preset (`gpt-oss-bookrag`, `qwen3-vl-bookrag`, and
`qwen3-embedding:0.6b`) are workstation model tags, not models provisioned by
this repository. Confirm them with `ollama list`, replace the `model_name`
values with your installed tags (or create the intended aliases), and run the
online `docarag doctor --strict` check before indexing. Automatic model
provisioning belongs to the future deployment profiles and is not implied by
`docarag init`.

Use the controller diagnostics to check the installation without importing
models or parsing documents:

```bash
docarag doctor --offline
```

The runtime probe checks the modules required by `requirements.txt` and runs
`pip check` inside the configured BookRAG interpreter, so missing packages and
incompatible installed requirements are reported before a long indexing job.

The legacy `python main.py --help` path imports the research runtime and is not a
lightweight setup diagnostic; use it only inside the fully installed model
environment.

If MinerU prints a warning that `sglang` is not installed, it can be ignored
when the selected backend does not embed an SGLang engine. Install the local
SGLang runtime only for an in-process SGLang backend; a client backend instead
requires the matching external service.

## Legacy BookRAG research workflow

Use this path to reproduce the original multi-dataset BookRAG experiments. New
single-document DocARAG users should use the supported `docarag` commands above.

Before running BookRAG, update the following configuration files:

- **System configuration**: model, retriever, embedding, VLM, and runtime settings, for example [`config/default.yaml`](./config/default.yaml)
- **Dataset configuration**: input path, working directory, dataset split, and metadata settings, for example [`Scripts/cfg/example-m3docVQA.yaml`](./Scripts/cfg/example-m3docVQA.yaml)

### Offline Index Construction

Use the example script to construct the BookIndex:

```bash
bash Scripts/example-index.sh
```

### Online Retrieval

Run BookRAG on a configured dataset:

```bash
bash Scripts/example-rag.sh
```

### Evaluation

We use a strong LLM as an answer extractor for evaluating model responses. Please configure the API file before evaluation, for example [`config/api.txt`](./config/api.txt).

```bash
bash Scripts/example-eval.sh
```

## Supported Datasets

BookRAG is evaluated on widely used complex document QA benchmarks:

- **MMLongBench-Doc**: [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc)
- **m3docvqa**: [M3DocRAG](https://github.com/bloomberg/m3docrag)
- **Qasper**: [Qasper](https://huggingface.co/datasets/allenai/qasper)

This repository releases the final filtered QA files used in our experiments. The original documents, raw QA files, and dataset-specific metadata should still be obtained from the official dataset repositories above.

| Dataset | Released QA file | Original source |
| --- | --- | --- |
| MMLongBench-Doc | [`datasets/MMLongBench-Doc.json`](./datasets/MMLongBench-Doc.json) | [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc) |
| m3docvqa | [`datasets/m3docvqa.json`](./datasets/m3docvqa.json) | [M3DocRAG](https://github.com/bloomberg/m3docrag) |
| Qasper | [`datasets/Qasper.json`](./datasets/Qasper.json) | [Qasper](https://huggingface.co/datasets/allenai/qasper) |

Each released QA file follows the unified JSON format below:

```json
[
  {
    "question": "THE FIRST QUESTION",
    "answer": "THE ANSWER OF FIRST QUESTION",
    "doc_uuid": "UUID OF THE DOCUMENT PDF",
    "doc_path": "PATH_TO_DIR/DOCUMENT.pdf"
  },
  {
    "question": "THE SECOND QUESTION",
    "answer": "THE ANSWER OF SECOND QUESTION",
    "doc_uuid": "UUID OF THE DOCUMENT PDF",
    "doc_path": "PATH_TO_DIR/DOCUMENT.pdf"
  }
]
```

Example preprocessing notebooks and scripts are available in [`Scripts/preprocess`](./Scripts/preprocess).

## Repository Layout

```text
DocARAG/
+-- Core/                 # current BookRAG indexing, retrieval, generation, and providers
+-- docarag/               # controller, canonical domain/ports, provenance, and adapters
+-- Eval/                  # evaluation scripts and answer extraction utilities
+-- Scripts/               # compatibility drivers, Web UI, examples, and preprocessing
+-- tests/                 # contracts, lineage, checkpoints, lifecycle, retrieval, and driver tests
+-- assets/                # figures and static assets used in the README
+-- config/                # legacy/current system and method configurations
+-- datasets/              # filtered QA files used in the paper experiments
+-- BOOKRAG_VLDB_2026_full.pdf
+-- SCALING_RAG_PLAN.md     # target architecture, gates, and migration sequence
+-- pyproject.toml          # dependency-light controller package metadata
+-- main.py                 # current BookRAG research/compatibility entry point
`-- README.md
```

## Links

- `arXiv`: [2512.03413](https://arxiv.org/abs/2512.03413)
- `Paper PDF`: [BOOKRAG_VLDB_2026_full.pdf](./BOOKRAG_VLDB_2026_full.pdf)
- `Code`: [sam234990/BookRAG](https://github.com/sam234990/BookRAG)

## Citation

If you find BookRAG useful, please cite our paper:

```bibtex
@article{wang2025bookrag,
  title   = {BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents},
  author  = {Wang, Shu and Zhou, Yingli and Fang, Yixiang},
  journal = {arXiv preprint arXiv:2512.03413},
  year    = {2025},
  eprint  = {2512.03413},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2512.03413}
}
```
