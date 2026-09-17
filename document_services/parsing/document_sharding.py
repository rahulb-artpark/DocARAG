"""Deterministic, dependency-light assignment of documents to workers."""

from __future__ import annotations

import hashlib
import os


def document_shard_number(doc_uuid: object, doc_path: object, nsplit: int) -> int:
    """Return a stable one-indexed shard from immutable document-key fields.

    Assignment depends only on the document key and shard count, never on
    corpus order or size. Adding or reordering other documents therefore does
    not move an existing document while ``nsplit`` is unchanged.
    """

    if isinstance(nsplit, bool) or not isinstance(nsplit, int) or nsplit < 1:
        raise ValueError("nsplit must be a positive integer")
    if doc_uuid is None or doc_path is None:
        raise ValueError("document shard identity requires doc_uuid and doc_path")
    uuid_text = str(doc_uuid)
    path_text = (
        os.fspath(doc_path)
        if isinstance(doc_path, os.PathLike)
        else str(doc_path)
    )
    if not uuid_text or not path_text:
        raise ValueError("document shard identity requires doc_uuid and doc_path")

    digest = hashlib.sha256()
    for value in (uuid_text, path_text):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], byteorder="big") % nsplit + 1


__all__ = ["document_shard_number"]
