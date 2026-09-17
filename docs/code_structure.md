# Code structure and naming

The two source packages are named by responsibility:

- `document_services/` contains the controller, application services, persistence,
  and isolated parser integrations. It replaces the `docarag/` source folder.
- `retrieval_engine/` contains model-backed indexing and retrieval algorithms.
  It replaces the implementation previously under `Core/`.

The lightweight controller wheel includes `document_services` only. Engine
commands run in the configured BookRAG interpreter from this checkout.

## Document services

```text
document_services/
├── cli.py                    # Public console entry point
├── __main__.py               # python -m document_services
├── domain/                   # Evidence, parsing, and knowledge contracts
├── ports/                    # Parser and page-probe interfaces
├── application/              # Document extraction use cases
├── parsing/
│   ├── bookrag_handoff.py    # Canonical parser output → BookRAG input
│   ├── document_sharding.py
│   ├── orchestrator.py
│   ├── page_probe.py
│   ├── reconciliation.py
│   ├── routed_ingestion.py
│   ├── routing.py
│   └── routing_presets.py
├── parser_adapters/          # Normalize vendor output into domain contracts
├── parser_clients/           # Docling, MinerU, and PaddleOCR HTTP clients
├── worker_transport/         # Isolated parser-worker protocol, client, server
├── enrichment/               # Recover and transcribe text, tables, formulas, figures
├── evidence/                 # Identity, assets, provenance, hydration, integrity
│   └── projections/          # BookRAG, graph, and persistent record projections
├── retrieval/                # Candidate fusion and cited-answer validation
├── indexing/
│   ├── admission.py
│   ├── document_linker.py    # Cross-document entity linking
│   ├── editing.py
│   ├── grouping_maintenance.py
│   ├── linking_cli.py        # Command-line adapter for document linking
│   ├── pipeline.py
│   └── recovery.py
├── persistence/              # Ledger, artifact/corpus/document/tree stores
│   ├── evidence_writer.py
│   ├── vector_tree_source.py
│   └── migrations/           # SQL upgrades for existing databases
├── runtime/
│   ├── bookrag_bridge.py     # CLI integration with the model interpreter
│   ├── compatibility_manifest.py
│   ├── parser_composition.py
│   ├── parser_worker.py
│   ├── routing_profile.py
│   ├── settings.py
│   └── runtime_compatibility.toml
├── operations/               # Diagnostics, setup, logging, GPU/process supervision
├── profiles/                 # Packaged parser-worker configuration examples
└── evaluation/               # Golden-document acceptance metrics
```

`parsing` selects and combines parsers; `parser_adapters` normalize their output;
`parser_clients` call vendor APIs; `worker_transport` connects the controller to
isolated workers. Keep these responsibilities separate when adding a parser.

## Retrieval engine

```text
retrieval_engine/
├── configuration/           # Validated runtime, provider, and index settings
│   └── strategies/          # Per-strategy retrieval settings
├── conversation/            # Message and memory models
├── providers/
│   ├── language_model.py
│   ├── vision_model.py
│   ├── embeddings.py
│   ├── reranking.py
│   ├── structured_output.py
│   ├── token_tracker.py
│   └── qwen3_vl_embedding.py # Vendored model implementation
├── parsing/
│   ├── mineru_parser.py
│   ├── mineru_reconciliation.py
│   ├── page_sharding.py
│   ├── pdf_refinement.py
│   └── table_conversion.py
├── indexes/                 # Tree, graph, GBC, BM25, vector store and manifests
├── indexing/
│   ├── pipeline.py
│   ├── statistics.py
│   ├── tree/                # Hierarchy, outline, summaries, checkpoints
│   ├── graph/               # Extraction, refinement, entity facets
│   └── vector/              # Vector indexing and RAPTOR
├── rag/
│   ├── base_strategy.py
│   ├── factory.py
│   ├── session.py
│   ├── query_cache.py
│   ├── grounding.py
│   ├── stored_indexes.py
│   ├── gbc/                 # agent.py, planner.py, retrieval.py, answer.py, context.py
│   └── strategies/          # Graph, multimodal, vanilla, GBC vanilla, traversal
├── runtime/                 # inference.py and resource_loader.py
├── prompts/                 # gbc, knowledge_graph, outline, refinement, summaries, traversal
└── utils/                   # Shared text, bounded execution, timing, safe paths
```

## Naming conventions

Use lowercase `snake_case` for packages and modules. Prefer descriptive nouns
for modules: `language_model`, `figure_description`, `compatibility_manifest`.
Keep established algorithm and vendor names such as `gbc`, `bm25`, `qwen3`, and
`mineru` where they identify an actual implementation.

Package context supplies the role: use `configuration/language_model.py` and
`prompts/knowledge_graph.py` rather than repeating `_config` and `_prompt` in
every filename. Use clear qualifiers when nearby modules have distinct roles,
such as `runtime/bookrag_bridge.py` and `runtime/compatibility_manifest.py`.

Keep `domain`, `ports`, and application contracts free of model/vendor imports.
Keep optional dependencies lazy so CLI help and diagnostics work without ML
frameworks. Add vendor integrations behind ports, persistence code in
`persistence`, and model-backed algorithms in `retrieval_engine`. Keep package
initializers small and use explicit imports.

## Migration examples

| Previous import | Current import |
| --- | --- |
| `docarag.cli` | `document_services.cli` |
| `docarag.adapters.docling` | `document_services.parser_adapters.docling` |
| `docarag.services.mineru` | `document_services.parser_clients.mineru` |
| `docarag.transport.client` | `document_services.worker_transport.client` |
| `docarag.runtime.compat` | `document_services.runtime.bookrag_bridge` |
| `docarag.runtime.compatibility` | `document_services.runtime.compatibility_manifest` |
| `docarag.runtime.composition` | `document_services.runtime.parser_composition` |
| `docarag.indexing.cross_doc_linker` | `document_services.indexing.document_linker` |
| `docarag.persistence.evidence` | `document_services.persistence.evidence_writer` |
| `Core.configs.system_config` | `retrieval_engine.configuration.system` |
| `Core.providers.llm` | `retrieval_engine.providers.language_model` |
| `Core.providers.vlm` | `retrieval_engine.providers.vision_model` |
| `Core.prompts.kg_prompt` | `retrieval_engine.prompts.knowledge_graph` |
| `Core.rag.gbc_plan` / `Core.rag.gbc.planner` | `retrieval_engine.rag.gbc.planner` |
| `Core.runtime.resources` | `retrieval_engine.runtime.resource_loader` |

The public console command is still `docarag`. Module execution now uses
`python -m document_services`; external Python callers must update their imports.
After pulling the rename, refresh an existing editable installation:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

Examples of internal module commands:

```bash
python -m document_services --help
python -m document_services.indexing.pipeline --help
python -m document_services.indexing.linking_cli --help
python -m document_services.persistence.evidence_writer --help
```

## Compatibility and generated data

`Core/` now contains only saved-index compatibility exports. Historical pickle
artifacts reference `Core.Index.Tree`, `Core.indexes.tree`, and the old BM25
locations; these exports deserialize them into the current engine classes.
All new code imports `retrieval_engine`. The controller wheel excludes both
the engine and these compatibility exports.

The installed distribution name, console command, `docarag.toml`, database
schema, environment variables, and persisted schema identifiers remain stable.
This source rename does not migrate or rename stored document data.

`config/`, `Scripts/`, `Eval/`, and `docs/` remain repository support directories.
`runs/`, `tmp/`, caches, build output, and package metadata are generated local
state. New batch failure reports go to `runs/_logs/`.

## Validation

```bash
python -m unittest discover -s tests -v
python -m document_services --help
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/controller-wheel .
```

`test_package_layout.py` checks local import targets, checkout discovery,
stable schema identity, and loading saved tree/BM25 indexes from previous module
paths. Architecture and controller boundary tests guard the separation from
model dependencies.
