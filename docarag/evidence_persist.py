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


def ensure_revision_admitted(run: Path, *, database_url: str | None = None) -> int:
    """Admit a run's revision to the ledger if it is not there already.

    Every stage's Postgres bookkeeping (stage_state, revision_seal) is keyed
    by revision_key, which only exists once the ledger has a row -- so this
    runs before the very first stage of a freshly admitted run, not just
    before persistence. Safe to call repeatedly: admission is a no-op once
    the row exists.
    """

    from docarag.persistence.contracts import CorpusScope, IndexRevisionSpec
    from docarag.persistence.postgres import PostgresLedger

    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source")
    if not isinstance(source, dict) or not source.get("sha256"):
        raise RuntimeError("run manifest has no source identity")
    from docarag.domain.evidence import DocumentRef

    document = DocumentRef.identified(
        content_sha256=source["sha256"],
        mime_type="application/pdf",
        source_uri=source.get("uri") or Path(source.get("path", "")).as_uri(),
        display_name=Path(source.get("path", "document.pdf")).name,
    )
    url = database_url or os.environ.get("DOCARAG_DATABASE_URL")
    if not url:
        raise RuntimeError("DOCARAG_DATABASE_URL is not set")
    ledger = PostgresLedger(url)
    scope = ledger.record_document(
        CorpusScope(str(manifest["tenant_id"]), str(manifest["corpus_id"])), document
    )
    _admit_revision(ledger, scope, manifest, run, IndexRevisionSpec)
    return ledger.revision_key(scope, str(manifest["revision_id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist run evidence to PostgreSQL")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database-url")
    parser.add_argument("--asset-cas-root", type=Path)
    parser.add_argument("--part", choices=("tree", "graph", "all"), default="all")
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
    from docarag.persistence.contracts import CorpusScope, IndexRevisionSpec
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
        # Evidence is keyed by the revision, so the revision has to exist in
        # the ledger before anything derived from it can be written.
        _admit_revision(ledger, scope, manifest, run, IndexRevisionSpec)
        parts = (
            frozenset({"tree", "graph"})
            if args.part == "all"
            else frozenset({args.part})
        )
        evidence = build_document_evidence(
            run, revision_id, asset_cas_root=cas_root, parts=parts
        )
        written = ledger.persist_document_evidence(scope, evidence, parts=parts)
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a clean message
        print(f"evidence persistence error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _mirror_stage_state(ledger, scope, revision_id, run)
    print(
        f"Persisted revision {revision_id}: "
        + ", ".join(f"{count} {name}" for name, count in sorted(written.items()) if count)
    )
    return 0


def _mirror_stage_state(ledger, scope, revision_id: str, run: Path) -> None:
    """Mirror the driver's stage outcomes into the store.

    A fresh run is driven from the run directory's state file; persistence is
    the first store write it performs, so this is where the pipeline's history
    becomes queryable alongside its evidence.  Failures here must not undo a
    completed persist.
    """

    try:
        state = json.loads((run / "index-state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return
    try:
        revision_key = ledger.revision_key(scope, revision_id)
        with ledger._connection() as connection:
            for name, entry in stages.items():
                if name not in ("tree", "graph", "vdb"):
                    continue
                status = (entry or {}).get("status")
                if status not in ("succeeded", "failed", "running"):
                    continue
                connection.execute(
                    "INSERT INTO docarag.stage_state "
                    "(revision_key, stage, status, started_at, ended_at, error) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (revision_key, stage) DO UPDATE SET "
                    " status = EXCLUDED.status, "
                    " started_at = EXCLUDED.started_at, "
                    " ended_at = EXCLUDED.ended_at, "
                    " error = EXCLUDED.error",
                    (
                        revision_key,
                        name,
                        status,
                        (entry or {}).get("started_at"),
                        (entry or {}).get("ended_at"),
                        (entry or {}).get("error"),
                    ),
                )
            connection.execute(
                "INSERT INTO docarag.stage_state (revision_key, stage, status) "
                "VALUES (%s, 'persist', 'succeeded') "
                "ON CONFLICT (revision_key, stage) DO UPDATE SET "
                " status = 'succeeded', ended_at = clock_timestamp(), error = NULL",
                (revision_key,),
            )
    except Exception:  # noqa: BLE001 - history mirroring is best effort
        return


def _admit_revision(ledger, scope, manifest, run, spec_type) -> None:
    """Record the revision this run published, if it is not already known.

    A revision is admitted once and never changed, so re-running persistence
    for the same run finds it already present and proceeds.
    """

    import hashlib

    try:
        ledger.revision_key(scope, str(manifest["revision_id"]))
        return
    except Exception:
        pass

    config_sha = str((manifest.get("system_config") or {}).get("sha256") or "")
    canonical = run / "canonical_blocks.json"
    canonical_sha = (
        hashlib.sha256(canonical.read_bytes()).hexdigest()
        if canonical.is_file()
        else "0" * 64
    )
    collection = "TreeVDB"
    config_path = run / "run_config.yaml"
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("collection_name:"):
                collection = line.split(":", 1)[1].strip().strip('"\'')
                break
    ledger.create_index_revision(
        spec_type(
            scope=scope,
            configuration_sha256=config_sha if len(config_sha) == 64 else "0" * 64,
            canonical_artifact_sha256=canonical_sha,
            qdrant_collection=collection,
            revision_id=str(manifest["revision_id"]),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
