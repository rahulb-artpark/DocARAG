"""Persist a completed run's canonical evidence into PostgreSQL and the asset CAS.

Reads the immutable run manifest for scope/identity, projects the run's JSON
artifacts into the evidence store, and writes them under one index revision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from docarag.evidence_projection import build_document_evidence


def _default_cas_root(run: Path) -> Path:
    # runs/<document-id>/<revision-id>/ -> runs/_assets
    return run.parent.parent / "_assets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist run evidence to PostgreSQL")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database-url")
    parser.add_argument("--asset-cas-root", type=Path)
    args = parser.parse_args(argv)

    run = args.run.expanduser().resolve()
    try:
        manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"evidence persistence error: cannot read run manifest: {exc}", file=sys.stderr)
        return 1
    source = manifest.get("source")
    if not isinstance(source, dict) or not source.get("sha256"):
        print("evidence persistence error: run manifest has no source identity", file=sys.stderr)
        return 1

    database_url = args.database_url or os.environ.get("DOCARAG_DATABASE_URL")
    if not database_url:
        print("evidence persistence error: DOCARAG_DATABASE_URL is not set", file=sys.stderr)
        return 1

    from docarag.domain.evidence import DocumentRef
    from docarag.persistence.contracts import CorpusScope
    from docarag.persistence.postgres import PostgresLedger

    revision_id = str(manifest["revision_id"])
    document = DocumentRef.identified(
        content_sha256=source["sha256"],
        mime_type="application/pdf",
        source_uri=source.get("uri") or Path(source.get("path", "")).as_uri(),
        display_name=Path(source.get("path", "document.pdf")).name,
    )
    cas_root = args.asset_cas_root or _default_cas_root(run)

    try:
        ledger = PostgresLedger(database_url)
        scope = ledger.record_document(
            CorpusScope(str(manifest["tenant_id"]), str(manifest["corpus_id"])),
            document,
        )
        evidence = build_document_evidence(run, revision_id, asset_cas_root=cas_root)
        ledger.persist_document_evidence(scope, evidence)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a clean message
        print(f"evidence persistence error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        "Persisted evidence for revision {rev}: {b} blocks, {t} tree nodes, "
        "{e} entities, {r} relations, {g} gt-links, {a} assets".format(
            rev=revision_id,
            b=len(evidence.blocks),
            t=len(evidence.tree_nodes),
            e=len(evidence.entities),
            r=len(evidence.relations),
            g=len(evidence.gt_links),
            a=len(evidence.assets),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
