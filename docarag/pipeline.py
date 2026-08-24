"""Run and re-run pipeline stages against one revision, state in PostgreSQL.

A stage is a store transformation: it reads what earlier stages wrote and
writes its own output for one revision.  Its record lives in stage_state, so
any stage can be re-run alone -- after an edit, a model upgrade or a failure --
without running the pipeline around it.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

STAGES = ("tree", "graph", "vdb", "persist")


class StageLockedError(RuntimeError):
    """Another process already holds this revision's stage lock."""


def _lock_key(revision_key: int, stage: str) -> tuple[int, int]:
    """Two signed int4 keys for pg_advisory_lock, deterministic per stage."""

    digest = hashlib.sha256(f"docarag-stage:{revision_key}:{stage}".encode()).digest()
    key1 = int.from_bytes(digest[:4], "big", signed=True)
    key2 = int.from_bytes(digest[4:8], "big", signed=True)
    return key1, key2


@contextmanager
def _stage_lock(connection, revision_key: int, stage: str):
    """Hold a session-scoped advisory lock for one stage of one revision.

    Two stages can run concurrently; the same stage of the same revision
    cannot. Session-scoped means the lock releases if the connection dies,
    so a killed process can never strand a run permanently locked.
    """

    key1, key2 = _lock_key(revision_key, stage)
    acquired = connection.execute(
        "SELECT pg_try_advisory_lock(%s, %s)", (key1, key2)
    ).fetchone()["pg_try_advisory_lock"]
    if not acquired:
        raise StageLockedError(
            f"stage '{stage}' is already running for this revision -- "
            "another docarag stage invocation holds it"
        )
    try:
        yield
    finally:
        connection.execute("SELECT pg_advisory_unlock(%s, %s)", (key1, key2))


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


def _cross_doc_link_auto_trigger_enabled(config_path: Path) -> bool:
    """Whether this run's graph config opts into auto-triggered cross-doc linking."""

    import yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return False
    return bool((data.get("graph") or {}).get("cross_doc_link_auto_trigger"))


def run_cross_doc_link_stage(run: Path) -> int:
    """Manually (re-)run cross-doc linking for one run, outside the driven sequence."""

    run = run.expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.update({"DOCARAG_RUNS_DIR": str(run.parent.parent)})

    with _connect() as connection:
        key = _revision_key(connection, run)
        with _stage_lock(connection, key, "cross_doc_link"):
            _record(connection, key, "cross_doc_link", "running")
            completed = subprocess.run(
                [sys.executable, "-m", "docarag.cross_doc_link", "--run", str(run)],
                cwd=str(repo), env=env,
            )
            status = "succeeded" if completed.returncode == 0 else "failed"
            error = None if completed.returncode == 0 else f"exit {completed.returncode}"
            _record(connection, key, "cross_doc_link", status, error)
    print(f"[stage] cross_doc_link: {status}")
    return completed.returncode


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
            # No stage has ever run against this revision, so main.py has not
            # frozen a snapshot yet -- it does that itself, the first time,
            # from the admitted config snapshot.
            snapshots = sorted((run / "inputs").glob("system-config-*.yaml"))
            if not snapshots:
                raise RuntimeError(
                    f"run has neither a frozen configuration nor an admitted "
                    f"snapshot under {run}/inputs"
                )
            config = snapshots[0]
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

    from docarag.admission import STAGE_NAMES, update_index_state
    from docarag.evidence_persist import ensure_revision_admitted

    revision_id = str(manifest["revision_id"])
    state_path = run / "index-state.json"
    file_state = json.loads(state_path.read_text(encoding="utf-8"))
    attempt_id = file_state.get("attempt_id")
    pid = os.getpid()

    # The ledger row is admission's job, but stage_state/revision_seal are
    # keyed by revision_key -- so it must exist before the very first stage
    # of a freshly admitted run, not just before persistence.
    ensure_revision_admitted(run)

    with _connect() as connection:
        key = _revision_key(connection, run)
        with _stage_lock(connection, key, stage):
            _record(connection, key, stage, "running")
            if stage in STAGE_NAMES:
                update_index_state(
                    state_path, revision_id, pid, "stage_started",
                    stage=stage, attempt_id=attempt_id,
                )
            completed = subprocess.run(command, cwd=str(repo), env=env)
            status = "succeeded" if completed.returncode == 0 else "failed"
            error = None if completed.returncode == 0 else f"exit {completed.returncode}"
            _record(connection, key, stage, status, error)
            if stage in STAGE_NAMES:
                update_index_state(
                    state_path, revision_id, pid,
                    "stage_succeeded" if status == "succeeded" else "stage_failed",
                    stage=stage, exit_code=completed.returncode, error=error,
                    attempt_id=attempt_id,
                )
            if status == "succeeded" and stage == "tree":
                # New tree rows invalidate whatever graph/vdb previously
                # described the old ones -- but a stage that has never run
                # is "pending", not stale; there is nothing to invalidate.
                previously_run = {
                    row["stage"]
                    for row in connection.execute(
                        "SELECT stage FROM docarag.stage_state "
                        "WHERE revision_key = %s AND stage IN ('graph', 'vdb') "
                        "AND status = 'succeeded'",
                        (key,),
                    )
                }
                for downstream in ("graph", "vdb"):
                    if downstream in previously_run:
                        _record(connection, key, downstream, "stale")
            if status == "succeeded" and stage in ("tree", "graph"):
                # A stage's output belongs to the store the moment the
                # stage ends.
                part = subprocess.run(
                    [
                        sys.executable, "-m", "docarag.evidence_persist",
                        "--run", str(run), "--part", stage,
                    ],
                    cwd=str(repo),
                )
                _record(
                    connection, key, "persist",
                    "succeeded" if part.returncode == 0 else "failed",
                    None if part.returncode == 0 else f"exit {part.returncode}",
                )
                if part.returncode != 0:
                    status = "failed"
                if (
                    stage == "graph"
                    and part.returncode == 0
                    and _cross_doc_link_auto_trigger_enabled(config)
                ):
                    # Best effort: this document's own tree/graph/vdb/persist
                    # outcome must never depend on the corpus having anything
                    # to link against yet, so a failure here is recorded under
                    # its own stage and never turns `status` into "failed".
                    _record(connection, key, "cross_doc_link", "running")
                    link = subprocess.run(
                        [sys.executable, "-m", "docarag.cross_doc_link", "--run", str(run)],
                        cwd=str(repo), env=env,
                    )
                    _record(
                        connection, key, "cross_doc_link",
                        "succeeded" if link.returncode == 0 else "failed",
                        None if link.returncode == 0 else f"exit {link.returncode}",
                    )
            if status == "succeeded" and stage in ("tree", "graph", "vdb"):
                # Only a run whose file state now shows every stage
                # succeeded is publishable -- mirroring the bash driver,
                # which only finalizes after its whole stage loop completes.
                current = json.loads(state_path.read_text(encoding="utf-8"))
                all_succeeded = all(
                    (current["stages"].get(name) or {}).get("status") == "succeeded"
                    for name in STAGE_NAMES
                )
                if all_succeeded:
                    republish = _republish(run, repo, env)
                    if republish != 0:
                        _record(
                            connection, key, stage,
                            "failed", "artifact republication failed",
                        )
                        print(f"[stage] {stage}: failed (republication)")
                        return republish
    print(f"[stage] {stage}: {status}")
    return completed.returncode


def republish_run_artifacts(run: Path) -> int:
    """Re-seal the artifact manifest around artifacts that changed in place.

    The stage writes an attestation; publication is what records its digest in
    the manifest. Anything that rewrites an attestation without republishing
    leaves readiness failing closed on "artifact digest differs for
    qdrant-tree-attestation/..." -- correct, since it cannot tell a rebuild
    from a corrupted file, and unhelpful when the rebuild was intended.

    This is publication only. ``_republish`` additionally runs the bash
    finalizer, which re-persists canonical evidence -- a whole-run operation an
    edit has no business triggering, and one that would rewrite a revision from
    an artifact the store has deliberately diverged from.
    """

    repo = Path(__file__).resolve().parent.parent
    artifact_manifest = run / "artifact-manifest.json"
    manifest = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    state_path = run / "index-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    roles = manifest.get("roles") or {}
    published = subprocess.run(
        [
            sys.executable, "-m", "docarag.provenance", "publish",
            "--run", str(run),
            "--revision-id", str(manifest["revision_id"]),
            "--attempt-id", str(state.get("attempt_id") or manifest["attempt_id"]),
            "--retrieval-attestation", str(run / str(roles["retrieval_attestation"])),
            "--graph-artifact", str(run / str(roles["graph"])),
            "--entity-attestation", str(run / str(roles["entity_attestation"])),
        ],
        cwd=str(repo),
        env=dict(os.environ),
    )
    if published.returncode != 0:
        return published.returncode

    republished = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    digests = {
        "artifact_manifest_sha256": hashlib.sha256(
            artifact_manifest.read_bytes()
        ).hexdigest(),
        "artifact_set_sha256": republished["artifact_set_sha256"],
    }
    state.update(digests)
    state_path.write_text(
        json.dumps(state, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    )
    # The archived attempt records the same digests and is sealed read-only.
    attempt = run / "attempts" / str(state["attempt_id"]) / "state.json"
    if attempt.is_file():
        mode = attempt.stat().st_mode & 0o777
        attempt.chmod(0o600)
        archived = json.loads(attempt.read_text(encoding="utf-8"))
        archived.update(digests)
        attempt.write_text(
            json.dumps(archived, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
        )
        attempt.chmod(mode)
    return 0


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


_OVERVIEW_SQL = """
SELECT
    doc.display_name,
    rev.revision_id,
    seal.tree_sha256 IS NOT NULL AS sealed,
    stage.stage,
    stage.status,
    stage.started_at
FROM docarag.index_revisions rev
JOIN docarag.documents doc
    ON doc.tenant_id = rev.tenant_id
   AND doc.corpus_id = rev.corpus_id
   AND doc.document_id = rev.document_id
LEFT JOIN docarag.revision_seal seal ON seal.revision_key = rev.revision_key
LEFT JOIN docarag.stage_state stage ON stage.revision_key = rev.revision_key
ORDER BY doc.display_name, rev.created_at, stage.stage
"""


def overview() -> int:
    """One line per revision: what every document's pipeline looks like now."""

    import datetime

    with _connect() as connection:
        rows = connection.execute(_OVERVIEW_SQL).fetchall()
    if not rows:
        print("no revisions in the store")
        return 0
    revisions: dict[str, dict] = {}
    for row in rows:
        entry = revisions.setdefault(
            row["revision_id"],
            {
                "name": row["display_name"] or "?",
                "sealed": bool(row["sealed"]),
                "stages": {},
            },
        )
        if row["stage"]:
            entry["stages"][row["stage"]] = row
    for revision_id, entry in revisions.items():
        cells = []
        running = False
        for name in STAGES:
            stage = entry["stages"].get(name)
            status = (stage or {}).get("status", "-")
            if status == "running" and stage and stage.get("started_at"):
                elapsed = (
                    datetime.datetime.now(datetime.timezone.utc)
                    - stage["started_at"]
                ).total_seconds()
                status = f"running {int(elapsed)}s"
                running = True
            elif status == "stale":
                status = "STALE"
            cells.append(f"{name}:{status}")
        seal = "sealed" if entry["sealed"] else "unsealed"
        marker = ">" if running else " "
        print(
            f"{marker} {entry['name'][:32]:32} {revision_id[:12]}  "
            f"{seal:8}  " + "  ".join(cells)
        )
    return 0


def watch(interval: float = 5.0) -> int:
    """Refresh the overview until nothing is running."""

    import time

    while True:
        print("\033[2J\033[H", end="")
        print(f"pipeline overview  ({time.strftime('%H:%M:%S')})\n")
        overview()
        with _connect() as connection:
            active = connection.execute(
                "SELECT count(*) AS n FROM docarag.stage_state "
                "WHERE status = 'running'"
            ).fetchone()["n"]
        if not active:
            print("\nnothing running")
            return 0
        time.sleep(interval)


def run_all(run: Path) -> int:
    """Re-derive the whole pipeline in order, stopping at the first failure."""

    for stage in ("tree", "graph", "vdb"):
        code = run_stage(run, stage)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pipeline stage from the store's state")
    parser.add_argument(
        "stage", choices=STAGES + ("cross_doc_link", "status", "all", "overview", "watch")
    )
    parser.add_argument("--run", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "overview":
        return overview()
    if args.stage == "watch":
        return watch()
    if args.run is None:
        parser.error(f"--run is required for '{args.stage}'")
    if args.stage == "status":
        for row in stage_status(args.run):
            line = f"{row['stage']:8} {row['status']}"
            if row.get("error"):
                line += f"  ({row['error']})"
            print(line)
        return 0
    try:
        if args.stage == "all":
            return run_all(args.run)
        if args.stage == "cross_doc_link":
            return run_cross_doc_link_stage(args.run)
        return run_stage(args.run, args.stage)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
