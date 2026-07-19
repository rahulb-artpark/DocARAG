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
            raise RuntimeError(f"run has no configuration snapshot: {config}")
        dataset = os.environ.get("DOCARAG_DATASET_CONFIG", "config/dataset.yaml")
        command = [
            sys.executable, "main.py",
            "-c", str(config), "-d", dataset,
            "--num", "1", "--nsplit", "1",
            "index", "--stage", stage,
        ]

    with _connect() as connection:
        key = _revision_key(connection, run)
        _record(connection, key, stage, "running")
    completed = subprocess.run(command, cwd=str(repo))
    status = "succeeded" if completed.returncode == 0 else "failed"
    error = None if completed.returncode == 0 else f"exit {completed.returncode}"
    with _connect() as connection:
        _record(connection, key, stage, status, error)
        if status == "succeeded" and stage in ("tree", "graph"):
            # Whatever this stage rewrote is no longer what the store holds.
            _record(connection, key, "persist", "stale")
        if status == "succeeded" and stage == "tree":
            _record(connection, key, "vdb", "stale")
    print(f"[stage] {stage}: {status}")
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline stage from the store's state")
    parser.add_argument("stage", choices=STAGES + ("status",))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "status":
        for row in stage_status(args.run):
            line = f"{row['stage']:8} {row['status']}"
            if row.get("error"):
                line += f"  ({row['error']})"
            print(line)
        return 0
    return run_stage(args.run, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
