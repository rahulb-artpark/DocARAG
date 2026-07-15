"""Durable, content-addressed checkpoints for expensive node summaries.

The BookRAG tree is still published as one compatibility artifact, but summary
generation can take hours on a large document.  This small store commits each
successful model result independently so a process/host interruption does not
force the next attempt to repeat already completed calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_cache_identity(
    *,
    kind: str,
    prompt: str,
    model: Mapping[str, Any],
    image_paths: Sequence[str | os.PathLike[str]] = (),
) -> tuple[str, str, tuple[dict[str, str], ...]]:
    """Return ``(cache_key, prompt_hash, image_identities)`` for one request.

    Image bytes, not paths or mtimes, participate in the identity.  A parser
    repair that recreates a crop at the same path therefore cannot reuse a
    summary of different pixels.
    """

    normalized_kind = str(kind).strip()
    if not normalized_kind:
        raise ValueError("summary kind must not be blank")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("summary prompt must not be blank")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("summary model identity must not be empty")

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    image_identities: list[dict[str, str]] = []
    for raw_path in image_paths:
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"summary image is not a file: {path}")
        image_identities.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": normalized_kind,
        "prompt_sha256": prompt_sha256,
        "model": dict(model),
        # Paths are retained in the checkpoint metadata for audit, but only
        # bytes participate in reuse. Moving an immutable run does not cause
        # every visual summary to be regenerated.
        "images": [{"sha256": item["sha256"]} for item in image_identities],
    }
    cache_key = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return cache_key, prompt_sha256, tuple(image_identities)


class SummaryCheckpointStore:
    """SQLite-backed, process-safe store of successful summary results."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(self.path, timeout=30.0)
        try:
            self._database.execute("PRAGMA journal_mode=DELETE")
            self._database.execute("PRAGMA synchronous=FULL")
            self._database.execute("PRAGMA busy_timeout=30000")
            self._database.execute(
                """
                CREATE TABLE IF NOT EXISTS summary_checkpoints (
                    cache_key TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    model_identity_json TEXT NOT NULL,
                    image_identities_json TEXT NOT NULL,
                    summary TEXT NOT NULL CHECK(length(trim(summary)) > 0),
                    created_at TEXT NOT NULL
                )
                """
            )
            self._database.commit()
            os.chmod(self.path, 0o600)
        except BaseException:
            self._database.close()
            raise

    def __enter__(self) -> "SummaryCheckpointStore":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._database is not None:
            self._database.close()
            self._database = None

    def get(self, cache_key: str) -> str | None:
        if self._database is None:
            raise RuntimeError("summary checkpoint store is closed")
        row = self._database.execute(
            "SELECT summary FROM summary_checkpoints WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return str(row[0]) if row else None

    def put(
        self,
        *,
        cache_key: str,
        kind: str,
        prompt_sha256: str,
        model: Mapping[str, Any],
        image_identities: Sequence[Mapping[str, str]],
        summary: str,
    ) -> str:
        """Commit one nonempty result and return the authoritative value.

        First-writer-wins makes retries idempotent even for nondeterministic
        model backends.  Failed/empty calls never reach this method.
        """

        if self._database is None:
            raise RuntimeError("summary checkpoint store is closed")
        normalized = summary.strip() if isinstance(summary, str) else ""
        if not normalized:
            raise ValueError("refusing to checkpoint an empty summary")
        if len(cache_key) != 64 or len(prompt_sha256) != 64:
            raise ValueError("summary checkpoint identity is invalid")

        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._database:
            self._database.execute(
                """
                INSERT OR IGNORE INTO summary_checkpoints (
                    cache_key, schema_version, kind, prompt_sha256,
                    model_identity_json, image_identities_json, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    SCHEMA_VERSION,
                    kind,
                    prompt_sha256,
                    _canonical_json(dict(model)),
                    _canonical_json([dict(item) for item in image_identities]),
                    normalized,
                    created_at,
                ),
            )
        authoritative = self.get(cache_key)
        if authoritative is None:
            raise RuntimeError("summary checkpoint commit did not become visible")
        return authoritative

    def count(self) -> int:
        if self._database is None:
            raise RuntimeError("summary checkpoint store is closed")
        return int(
            self._database.execute(
                "SELECT COUNT(*) FROM summary_checkpoints"
            ).fetchone()[0]
        )
