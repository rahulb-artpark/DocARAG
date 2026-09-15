"""Keep grouping hubs consistent after a member entity stops existing.

cross_doc_linker.py only ever grows a grouping (write_grouping_membership,
update_grouping_centroid) -- nothing shrinks one back down when a document
that contributed a member gets deleted. This is that other half: given the
groupings a just-deleted document's orphaned entities belonged to, recompute
each one's centroid/member_count from whoever actually still belongs, or
remove the grouping outright once nobody does.
"""

from __future__ import annotations

from collections.abc import Iterable

from Core.provider.embedding import TextEmbeddingProvider
from docarag.corpus_store import CorpusStore
from docarag.persistence.contracts import CorpusScope
from docarag.persistence.postgres import PostgresLedger


def recalculate_affected_groupings(
    ledger: PostgresLedger,
    connection,
    scope: CorpusScope,
    grouping_keys: Iterable[int],
    embedder: TextEmbeddingProvider,
) -> dict[str, int]:
    """Recompute or remove every grouping in ``grouping_keys``.

    A grouping with surviving members gets a fresh centroid, re-embedded from
    exactly those members' canonical names (the per-member vectors that fed
    the old centroid were never stored, only the running average) and a
    corrected member_count. A grouping with none left is deleted.
    """

    store = CorpusStore(lambda: connection, scope.tenant_id, scope.corpus_id)
    recalculated = 0
    deleted = 0
    for grouping_key in {int(key) for key in grouping_keys}:
        members = store.members_for_grouping(grouping_key)
        if not members:
            ledger.delete_grouping(connection, scope, grouping_key=grouping_key)
            deleted += 1
            continue
        vectors = embedder.embed_texts([member.canonical for member in members])
        centroid = [
            sum(float(vector[i]) for vector in vectors) / len(vectors)
            for i in range(len(vectors[0]))
        ]
        ledger.update_grouping_centroid(
            connection, scope,
            grouping_key=grouping_key, centroid=centroid, member_count=len(members),
        )
        recalculated += 1
    return {"recalculated": recalculated, "deleted": deleted}
