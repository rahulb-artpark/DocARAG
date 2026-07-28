"""How an entity is represented in the vector store, and read back as one row.

Entity search used to embed the node name alone -- ``Name: x\\nType: y`` -- with
the description sitting in the payload unembedded.  Retrieval compared surface
forms and the reranker carried every semantic judgement, which is why "LLM"
never retrieved "Large Language Model" on its own.

An entity is now one Qdrant point carrying three vectors that fail in different
directions:

``name``     dense, today's exact behaviour, kept so this adds evidence rather
             than replacing the only signal ever measured against the quality bar
``context``  dense over name + type + description, for semantic equivalence with
             no shared tokens ("LLM" = "Large Language Model")
``lexical``  sparse with server-side IDF, for shared-token aliases that dense
             embeddings are worst at ("Qwen3-Reranker-4B" = "Qwen3 Reranker 4B")

Merges never rewrite vectors.  When A and B merge into C, their points stay as
written and only the ``canonical_id`` payload field is repointed, so a later
entity matching A's original phrasing matches A's own vector and resolves to C
rather than to a blurred centroid.

That leaves several points resolving to one entity, which BookRAG 4.3.2 must
never see: adjacent near-identical scores read as a smooth run where a sharp
drop belongs, so ``case_a`` fires and the duplicate is created instead of
merged.  On the query path it is worse, because PageRank double-weights the
entity.  Qdrant collapses this server-side by grouping on ``canonical_id``;
``collapse_entity_hits`` is the client-side net for ungrouped reads and for a
``canonical_id`` that has not caught up with the alias map yet.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import re


NAME_VECTOR = "name"
CONTEXT_VECTOR = "context"
LEXICAL_VECTOR = "lexical"

DENSE_VECTORS = (NAME_VECTOR, CONTEXT_VECTOR)

#: Payload field Qdrant groups on. A derived cache of ``entity_alias_map``,
#: which stays the source of truth and can rebuild it at any point.
CANONICAL_FIELD = "canonical_id"

# The cap exists so both producers of these records -- the refiner and the
# query-side rebuild -- cut at exactly the same place. Disagreeing would give
# one entity different vectors from different code paths, and the entity
# attestation compares them.
#
# The value is not tuned, and measurement says it does not need to be: swept at
# 60, 120, 300 and uncapped over 64 real entities whose descriptions run to a
# 214-word median, alias recall stayed 4/4 and the fused ranking never changed
# (mean margin 0.4458 at every setting above 60). Retrieval is indifferent
# here, so this bounds embedding input rather than buying quality -- do not
# read it as a quality knob.
#
# Set generously at 500 for that reason: since quality does not depend on it,
# the cap should keep as much of a description as it can while still bounding
# the input. 500 words is roughly 650 tokens, well inside the embedder's 2048,
# and past the 731-word maximum seen in the sampled corpus for all but the
# longest entities.
_CONTEXT_DESCRIPTION_WORDS = 500

_TOKEN = re.compile(r"[a-z0-9]+")


def context_text(node_name: str, description: str) -> str:
    """What the ``context`` vector embeds for one entity."""

    words = (description or "").split()
    if len(words) > _CONTEXT_DESCRIPTION_WORDS:
        words = words[:_CONTEXT_DESCRIPTION_WORDS]
    summary = " ".join(words)
    return f"{node_name}\nDescription: {summary}" if summary else node_name


def lexical_terms(text: str) -> dict[int, float]:
    """Term frequencies keyed by a stable token id.

    Qdrant applies IDF from collection statistics, so the client sends raw
    counts and no vocabulary has to be persisted or kept in sync.  Token ids
    come from a fixed hash rather than ``hash()``, whose per-process seed would
    make a rebuilt index disagree with the one it replaced.
    """

    counts: dict[int, float] = {}
    for token in _TOKEN.findall((text or "").lower()):
        token_id = _stable_token_id(token)
        counts[token_id] = counts.get(token_id, 0.0) + 1.0
    return counts


def _stable_token_id(token: str) -> int:
    # FNV-1a, 32-bit, masked to stay a non-negative Qdrant index.
    value = 0x811C9DC5
    for byte in token.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value & 0x7FFFFFFF


def lexical_text(node_name: str, description: str) -> str:
    """What the ``lexical`` sparse vector is built from."""

    return f"{node_name} {description or ''}"


def collapse_entity_hits(
    hits: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    resolve: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Reduce hits to one row per entity, keeping each entity's best score.

    The safety net behind Qdrant's own grouping.  It runs on ungrouped reads,
    and it still earns its place on grouped ones: ``canonical_id`` is a cache
    updated after a merge commits, so a search racing that update can see two
    points whose payloads disagree, and ``resolve`` -- backed by the in-memory
    alias map -- is authoritative where the payload is merely current.

    Scores are comparable across vectors because fused ranks and cosine
    distances are each internally consistent, so taking the best is the
    entity's strongest evidence rather than an average that dilutes it.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be at least 1")

    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for hit in hits:
        if not isinstance(hit, Mapping):
            continue
        metadata = hit.get("metadata") or {}
        identity = metadata.get(CANONICAL_FIELD) or metadata.get("node_id")
        if not isinstance(identity, str) or not identity:
            continue
        if resolve is not None:
            resolved = resolve(identity)
            if isinstance(resolved, str) and resolved:
                identity = resolved
        distance = hit.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            continue
        incumbent = best.get(identity)
        if incumbent is None:
            best[identity] = dict(hit)
            order.append(identity)
        elif distance < incumbent["distance"]:
            # Distance, so smaller is better.
            best[identity] = dict(hit)

    collapsed = [best[identity] for identity in order]
    collapsed.sort(key=lambda hit: hit["distance"])
    return collapsed[:limit]
