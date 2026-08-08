"""Small protocol fakes for Qdrant provider tests (no network or local mode)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any


class FakeModels:
    class Distance:
        COSINE = "Cosine"

    class PayloadSchemaType:
        KEYWORD = "keyword"

    @dataclass
    class VectorParams:
        size: int
        distance: str

    class Modifier:
        IDF = "idf"

    @dataclass
    class SparseIndexParams:
        on_disk: bool | None = None

    @dataclass
    class SparseVectorParams:
        index: Any | None = None
        modifier: str | None = None

    @dataclass
    class SparseVector:
        indices: list[int]
        values: list[float]

    @dataclass
    class PointStruct:
        id: str
        # A named-vector point carries a mapping of vector name to vector, and
        # may carry only a subset of the collection's names -- which is the
        # property a one-point-per-node schema depends on.
        vector: list[float] | dict[str, Any]
        payload: dict[str, Any]

    @dataclass
    class MatchValue:
        value: Any

    @dataclass
    class Range:
        gte: float | None = None
        lte: float | None = None

    @dataclass
    class FieldCondition:
        key: str
        match: Any | None = None
        range: Any | None = None

    @dataclass
    class HasIdCondition:
        has_id: list[str]

    @dataclass
    class Filter:
        must: list[Any]

    @dataclass
    class FilterSelector:
        filter: Any

    @dataclass
    class PointIdsList:
        points: list[str]

    @dataclass
    class QueryRequest:
        query: Any
        filter: Any | None = None
        limit: int = 10
        with_payload: bool = True
        with_vectors: bool = False
        using: str | None = None


def _named_vectors(point: Any) -> dict[str | None, Any]:
    """One point's vectors keyed by name; an unnamed point keys on None."""

    vector = point.vector
    return dict(vector) if isinstance(vector, dict) else {None: vector}


def _matches(record_id: str, payload: dict[str, Any], filter_: Any | None) -> bool:
    if filter_ is None:
        return True
    for condition in filter_.must:
        if isinstance(condition, FakeModels.HasIdCondition):
            if record_id not in condition.has_id:
                return False
            continue
        value = payload.get(condition.key)
        if condition.match is not None and value != condition.match.value:
            return False
        if condition.range is not None:
            if value is None:
                return False
            if condition.range.gte is not None and value < condition.range.gte:
                return False
            if condition.range.lte is not None and value > condition.range.lte:
                return False
    return True


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.batch_query_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.set_payload_calls: list[dict[str, Any]] = []
        self.closed = False

    def collection_exists(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: Any | None = None,
    ) -> None:
        if collection_name in self.collections:
            error = RuntimeError("already exists")
            error.status_code = 409
            raise error
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_vectors_config,
            "payload_schema": {},
            "records": {},
        }

    def delete_collection(self, *, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def get_collection(self, collection_name: str):
        collection = self.collections[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=collection["vectors_config"],
                    sparse_vectors=collection.get("sparse_vectors_config"),
                )
            ),
            payload_schema=dict(collection["payload_schema"]),
        )

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: Any,
        wait: bool,
    ) -> None:
        assert wait is True
        self.collections[collection_name]["payload_schema"][field_name] = field_schema

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[Any],
        wait: bool,
    ) -> None:
        assert wait is True
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": list(points),
                "wait": wait,
                "ids": [point.payload["_docarag_record_id"] for point in points],
                "embeddings": [point.vector for point in points],
                # The same vectors keyed by name, so a named-vector test can
                # assert which slots a point carries without unpacking
                # PointStruct. An unnamed point reports {None: vector}.
                "vectors": [_named_vectors(point) for point in points],
                "documents": [point.payload["_docarag_content"] for point in points],
                "metadatas": [
                    {
                        key: value
                        for key, value in point.payload.items()
                        if not key.startswith("_docarag_")
                    }
                    for point in points
                ],
            }
        )
        records = self.collections[collection_name]["records"]
        for point in points:
            records[point.id] = point

    def count(
        self,
        *,
        collection_name: str,
        count_filter: Any | None,
        exact: bool,
    ):
        assert exact is True
        records = self.collections[collection_name]["records"]
        return SimpleNamespace(
            count=sum(
                _matches(record_id, point.payload, count_filter)
                for record_id, point in records.items()
            )
        )

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: Any,
        wait: bool,
    ) -> None:
        assert wait is True
        self.delete_calls.append(
            {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
            }
        )
        records = self.collections[collection_name]["records"]
        if isinstance(points_selector, FakeModels.PointIdsList):
            selected = set(points_selector.points)
            for record_id in tuple(records):
                if record_id in selected:
                    del records[record_id]
            return
        filter_ = points_selector.filter
        for record_id, point in tuple(records.items()):
            if _matches(record_id, point.payload, filter_):
                del records[record_id]

    def set_payload(
        self,
        *,
        collection_name: str,
        payload: dict[str, Any],
        points: Any,
        wait: bool,
    ) -> None:
        assert wait is True
        self.set_payload_calls.append(
            {
                "collection_name": collection_name,
                "payload": dict(payload),
                "points": points,
                "wait": wait,
            }
        )
        records = self.collections[collection_name]["records"]
        if isinstance(points, list):
            selected = set(points)
            predicate = lambda record_id, _point: record_id in selected
        else:
            predicate = lambda record_id, point: _matches(
                record_id, point.payload, points
            )
        for record_id, point in records.items():
            if predicate(record_id, point):
                point.payload.update(payload)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
            sum(b * b for b in right)
        )
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _sparse_dot(left: Any, right: Any) -> float:
        """Term-frequency overlap, standing in for Qdrant's IDF scoring.

        The server weights each term by its collection-wide rarity; a fake has
        no corpus to compute that from, so this scores on shared terms alone.
        Enough to assert that a keyword query reaches the point holding the
        term and not the ones that do not.
        """

        weights = dict(zip(right.indices, right.values))
        return float(
            sum(value * weights.get(index, 0.0)
                for index, value in zip(left.indices, left.values))
        )

    @staticmethod
    def _vector_for(point: Any, using: str | None) -> Any:
        """The vector a query targets, or None when the point does not carry it.

        Returning None -- and skipping the point -- is what makes "a point may
        carry a subset of the collection's vectors" a testable property.
        """

        vector = point.vector
        if isinstance(vector, dict):
            return vector.get(using)
        return vector if using is None else None

    def _score(self, query: Any, vector: Any) -> float:
        if isinstance(query, FakeModels.SparseVector):
            if not isinstance(vector, FakeModels.SparseVector):
                return 0.0
            return self._sparse_dot(query, vector)
        if isinstance(vector, FakeModels.SparseVector):
            return 0.0
        return self._cosine(query, vector)

    def query_points(
        self,
        *,
        collection_name: str,
        query: Any,
        query_filter: Any | None,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        using: str | None = None,
    ):
        call = {
            "collection_name": collection_name,
            "query": query,
            "query_filter": query_filter,
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
            "using": using,
        }
        self.query_calls.append(call)
        records = self.collections[collection_name]["records"]
        hits = []
        for record_id, point in records.items():
            if not _matches(record_id, point.payload, query_filter):
                continue
            vector = self._vector_for(point, using)
            if vector is None:
                continue
            hits.append(
                SimpleNamespace(
                    id=record_id,
                    score=self._score(query, vector),
                    payload=dict(point.payload),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return SimpleNamespace(points=hits[:limit])

    def query_batch_points(self, *, collection_name: str, requests: list[Any]):
        self.batch_query_calls.append(
            {"collection_name": collection_name, "requests": list(requests)}
        )
        return [
            self.query_points(
                collection_name=collection_name,
                query=request.query,
                query_filter=request.filter,
                limit=request.limit,
                with_payload=request.with_payload,
                with_vectors=request.with_vectors,
                using=getattr(request, "using", None),
            )
            for request in requests
        ]

    def close(self) -> None:
        self.closed = True

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        logical: dict[str, dict[str, Any]] = {}
        for collection in self.collections.values():
            for point in collection["records"].values():
                record_id = point.payload["_docarag_record_id"]
                logical[record_id] = {
                    "embedding": point.vector,
                    "vectors": _named_vectors(point),
                    "document": point.payload["_docarag_content"],
                    "metadata": {
                        key: value
                        for key, value in point.payload.items()
                        if not key.startswith("_docarag_")
                    },
                }
        return logical


def qdrant_store(embedding_model, *, scope_filter=None, collection_name="test"):
    from Core.provider.vdb import VectorStore

    client = FakeQdrantClient()
    store = VectorStore(
        embedding_model,
        collection_name=collection_name,
        scope_filter=scope_filter,
        client=client,
        models=FakeModels,
    )
    return store, client
