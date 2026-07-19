"""Run and re-run pipeline stages against one revision, state in PostgreSQL.

A stage is a store transformation: it reads what earlier stages wrote and
writes its own output for one revision.  Its record lives in stage_state, so
any stage can be re-run alone -- after an edit, a model upgrade or a failure --
without running the pipeline around it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

STAGES = ("tree", "graph", "vdb", "persist")


def _connect():
    import psycopg
    from psycopg.rows import dict_row

    url = os.environ.get("DOCARAG_DATABASE_URL")
    if not url:
        raise RuntimeError("DOCARAG_DATABASE_URL is not set")
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


def _revision_key(connection, run: Path) -> int:
    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    row = connection.execute(
        "SELECT revision_key FROM docarag.index_revisions WHERE revision_id = %s",
        (str(manifest["revision_id"]),),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "revision is not admitted to the ledger; run evidence persistence "
            "once, or docarag index"
        )
    return int(row["revision_key"])


def _record(connection, revision_key: int, stage: str, status: str, error: str | None = None) -> None:
    connection.execute(
        "INSERT INTO docarag.stage_state (revision_key, stage, status, started_at, ended_at, error) "
        "VALUES (%s, %s, %s, "
        " CASE WHEN %s = 'running' THEN clock_timestamp() END, "
        " CASE WHEN %s IN ('succeeded','failed') THEN clock_timestamp() END, %s) "
        "ON CONFLICT (revision_key, stage) DO UPDATE SET "
        " status = EXCLUDED.status, "
        " started_at = COALESCE(EXCLUDED.started_at, docarag.stage_state.started_at), "
        " ended_at = EXCLUDED.ended_at, "
        " error = EXCLUDED.error",
        (revision_key, stage, status, status, status, error),
    )


def stage_status(run: Path) -> list[dict]:
    with _connect() as connection:
        key = _revision_key(connection, run)
        rows = connection.execute(
            "SELECT stage, status, started_at, ended_at, error "
            "FROM docarag.stage_state WHERE revision_key = %s",
            (key,),
        ).fetchall()
    by_stage = {row["stage"]: row for row in rows}
    return [by_stage.get(name, {"stage": name, "status": "pending"}) for name in STAGES]


def run_stage(run: Path, stage: str) -> int:
    """Execute one stage for one revision and record the outcome."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}; expected one of {STAGES}")
    run = run.expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent

    if stage == "persist":
        command = [sys.executable, "-m", "docarag.evidence_persist", "--run", str(run)]
    else:
        config = run / "run_config.yaml"
        if not config.is_file():
            raise RuntimeError(f"run has no frozen configuration: {config}")
        datasets = sorted((run / "inputs").glob("dataset-*.yaml"))
        if not datasets:
            raise RuntimeError(f"run has no dataset snapshot under {run}/inputs")
        dataset = str(datasets[0])
        command = [
            sys.executable, "main.py",
            "-c", str(config), "-d", dataset,
            "--num", "1", "--nsplit", "1",
            "index", "--stage", stage,
        ]

    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.update(
        {
            "DOCARAG_TENANT_ID": str(manifest["tenant_id"]),
            "DOCARAG_CORPUS_ID": str(manifest["corpus_id"]),
            "DOCARAG_DOCUMENT_ID": str(manifest["document_id"]),
            "DOCARAG_DOCUMENT_VERSION_ID": str(manifest["document_version_id"]),
            "DOCARAG_INDEX_REVISION_ID": str(manifest["revision_id"]),
            "DOCARAG_SYSTEM_CONFIG_SHA256": str(
                (manifest.get("system_config") or {}).get("sha256") or ""
            ),
            "DOCARAG_RUNS_DIR": str(run.parent.parent),
        }
    )
    with _connect() as connection:
        key = _revision_key(connection, run)
        _record(connection, key, stage, "running")
    completed = subprocess.run(command, cwd=str(repo), env=env)
    status = "succeeded" if completed.returncode == 0 else "failed"
    error = None if completed.returncode == 0 else f"exit {completed.returncode}"
    with _connect() as connection:
        _record(connection, key, stage, status, error)
        if status == "succeeded" and stage == "tree":
            # New tree rows: graph bindings and vectors no longer describe them.
            _record(connection, key, "graph", "stale")
            _record(connection, key, "vdb", "stale")
    if status == "succeeded" and stage in ("tree", "graph", "vdb"):
        # A re-run stage rewrote published artifacts; the manifest and the
        # sealed run state must describe what is now on disk.
        republish = _republish(run, repo, env)
        if republish != 0:
            with _connect() as connection:
                _record(connection, key, stage, "failed", "artifact republication failed")
            print(f"[stage] {stage}: failed (republication)")
            return republish
    if status == "succeeded" and stage in ("tree", "graph"):
        # A stage's output belongs to the store the moment the stage ends.
        part = subprocess.run(
            [
                sys.executable, "-m", "docarag.evidence_persist",
                "--run", str(run), "--part", stage,
            ],
            cwd=str(repo),
        )
        with _connect() as connection:
            _record(
                connection, key, "persist",
                "succeeded" if part.returncode == 0 else "failed",
                None if part.returncode == 0 else f"exit {part.returncode}",
            )
        if part.returncode != 0:
            status = "failed"
    print(f"[stage] {stage}: {status}")
    return completed.returncode


def _republish(run: Path, repo: Path, env: dict) -> int:
    """Refresh the artifact manifest and reseal run state after a stage re-run."""

    manifest = json.loads((run / "artifact-manifest.json").read_text(encoding="utf-8"))
    state = json.loads((run / "index-state.json").read_text(encoding="utf-8"))
    roles = manifest.get("roles") or {}
    retrieval = run / str(roles["retrieval_attestation"])
    command = [
        sys.executable, "-m", "docarag.provenance", "publish",
        "--run", str(run),
        "--revision-id", str(manifest["revision_id"]),
        "--attempt-id", str(state.get("attempt_id") or manifest["attempt_id"]),
        "--retrieval-attestation", str(retrieval),
        "--graph-artifact", str(run / str(roles["graph"])),
        "--entity-attestation", str(run / str(roles["entity_attestation"])),
    ]
    published = subprocess.run(command, cwd=str(repo), env=env)
    if published.returncode != 0:
        return published.returncode
    finalized = subprocess.run(
        ["bash", str(repo / "Scripts" / "index_pdf.sh"), "--finalize-run", str(run)],
        cwd=str(repo),
        env=env,
    )
    return finalized.returncode


def run_all(run: Path) -> int:
    """Re-derive the whole pipeline in order, stopping at the first failure."""

    for stage in ("tree", "graph", "vdb"):
        code = run_stage(run, stage)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline stage from the store's state")
    parser.add_argument("stage", choices=STAGES + ("status", "all"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "status":
        for row in stage_status(args.run):
            line = f"{row['stage']:8} {row['status']}"
            if row.get("error"):
                line += f"  ({row['error']})"
            print(line)
        return 0
    if args.stage == "all":
        return run_all(args.run)
    return run_stage(args.run, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
