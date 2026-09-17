"""Cross-link documents' entities: grouping nodes and cross-docs relations.

Chosen by how many ``--run`` directories are given, and ``--mode`` for a
single ``--run``:

Auto (one ``--run``, default mode): cross-link that just-persisted document
against the whole corpus. This is what ``document_services/indexing/pipeline.py`` shells out to
automatically when ``graph_config.cross_doc_link_auto_trigger`` is on.

Mesh (one ``--run``, ``--mode mesh``): cross-link that document only against
its existing mesh -- its direct matches, plus whatever those documents are
already cross-linked to -- skipping documents with no existing connection.

Direct (two or more ``--run``): cross-link the first ``--run`` (the focal
document -- the one just ingested, or the one "Link to..." was clicked on)
against the rest, ignoring the rest of the corpus -- the user-selects-
documents or user-selects-group path. One-directional: only the focal
document's entities are scanned, restricted to the given targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

from retrieval_engine.configuration.system import SystemConfig


def _load_run(run: Path) -> tuple[dict, SystemConfig]:
    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    config = SystemConfig(**yaml.safe_load((run / "run_config.yaml").read_text()))
    return manifest, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-link document entities: groupings and cross-docs relations"
    )
    parser.add_argument(
        "--run", type=Path, action="append", required=True,
        help="one run directory (auto mode) or two-or-more (direct mode)",
    )
    parser.add_argument("--database-url")
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--relatedness-floor", type=float, default=None)
    parser.add_argument(
        "--mode", choices=("auto", "mesh"), default="auto",
        help="only meaningful with a single --run; mesh restricts linking to "
        "documents already reachable via an existing cross-doc connection",
    )
    args = parser.parse_args(argv)
    if args.mode == "mesh" and len(args.run) != 1:
        print("cross_doc_link error: --mode mesh requires exactly one --run", file=sys.stderr)
        return 1

    database_url = args.database_url or os.environ.get("DOCARAG_DATABASE_URL")
    if not database_url:
        print("cross_doc_link error: DOCARAG_DATABASE_URL is not set", file=sys.stderr)
        return 1

    from document_services.indexing.document_linker import CrossDocLinker
    from document_services.persistence.contracts import DocumentScope
    from document_services.persistence.postgres import PostgresLedger

    runs = [r.expanduser().resolve() for r in args.run]
    try:
        manifests_configs = [_load_run(run) for run in runs]
    except (OSError, ValueError) as exc:
        print(f"cross_doc_link error: cannot read run manifest/config: {exc}", file=sys.stderr)
        return 1

    tenant_ids = {str(m["tenant_id"]) for m, _ in manifests_configs}
    corpus_ids = {str(m["corpus_id"]) for m, _ in manifests_configs}
    if len(tenant_ids) != 1 or len(corpus_ids) != 1:
        print("cross_doc_link error: all --run documents must share one tenant/corpus", file=sys.stderr)
        return 1
    tenant_id, corpus_id = next(iter(tenant_ids)), next(iter(corpus_ids))

    try:
        ledger = PostgresLedger(database_url)
        revision_keys = [
            ledger.revision_key(
                DocumentScope(
                    tenant_id=str(m["tenant_id"]), corpus_id=str(m["corpus_id"]),
                    document_id=str(m["document_id"]), document_version_id=str(m["document_version_id"]),
                ),
                str(m["revision_id"]),
            )
            for m, _ in manifests_configs
        ]
        _, first_config = manifests_configs[0]
        linker = CrossDocLinker.open(
            tenant_id=tenant_id, corpus_id=corpus_id, system_config=first_config,
            database_url=database_url,
            merge_threshold=args.merge_threshold, relatedness_floor=args.relatedness_floor,
        )
        try:
            if len(revision_keys) == 1:
                stats = (
                    linker.link_document_expanding(revision_keys[0])
                    if args.mode == "mesh"
                    else linker.link_document(revision_keys[0])
                )
            else:
                stats = linker.link_documents(revision_keys)
        finally:
            linker.close()
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a clean message
        print(f"cross_doc_link error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    mode = args.mode if len(revision_keys) == 1 else "direct"
    print(f"[cross_doc_link:{mode}] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
