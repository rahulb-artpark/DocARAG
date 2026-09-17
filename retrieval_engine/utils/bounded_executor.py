"""Small, dependency-free primitives for bounded concurrent map operations.

``ThreadPoolExecutor.map`` and the common ``submit``-everything pattern retain
one future per input.  That is an avoidable memory spike when an ingestion run
contains hundreds of thousands of nodes.  This module keeps only a configured
number of submitted items alive and yields typed outcomes in input order.

The implementation is intentionally provider-agnostic so extraction, graph,
and embedding stages can adopt the same backpressure semantics.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from threading import Event
from typing import Callable, Generic, Iterable, Iterator, TypeVar


ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class BatchItemResult(Generic[ItemT, ResultT]):
    """The success or failure of one input item.

    ``error is None`` is the success discriminator.  Keeping the original item
    makes the primitive useful for keyed requests without a separate, unbounded
    index-to-item list.
    """

    index: int
    item: ItemT
    value: ResultT | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> ResultT:
        """Return the value or re-raise the original per-item exception."""

        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]


def iter_bounded_ordered(
    items: Iterable[ItemT],
    worker: Callable[[ItemT], ResultT],
    *,
    max_workers: int,
    max_in_flight: int,
    cancel_event: Event | None = None,
    thread_name_prefix: str = "bounded-work",
) -> Iterator[BatchItemResult[ItemT, ResultT]]:
    """Apply ``worker`` with bounded submission and ordered typed outcomes.

    At most ``max_in_flight`` inputs have been pulled from ``items`` but not yet
    yielded.  Results are yielded in input order, while up to ``max_workers``
    calls execute concurrently.  Closing the iterator or setting
    ``cancel_event`` stops input consumption, cancels work that has not started,
    and waits for already-running calls before releasing their threads.

    Ordinary ``Exception`` instances are represented in the result stream.
    Process-level exceptions such as ``KeyboardInterrupt`` and ``SystemExit``
    are deliberately not converted into item failures.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")

    source = iter(items)
    pending: deque[tuple[int, ItemT, Future[ResultT]]] = deque()
    executor = ThreadPoolExecutor(
        max_workers=min(max_workers, max_in_flight),
        thread_name_prefix=thread_name_prefix,
    )
    next_index = 0
    source_exhausted = False

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def fill() -> None:
        nonlocal next_index, source_exhausted
        while (
            not source_exhausted
            and not cancelled()
            and len(pending) < max_in_flight
        ):
            try:
                item = next(source)
            except StopIteration:
                source_exhausted = True
                break
            # ``ThreadPoolExecutor`` does not propagate ContextVars.  Capture a
            # distinct context for every submission so request-scoped token
            # ledgers, trace IDs, and cancellation metadata remain attached to
            # the logical run instead of falling back to process-global state.
            context = copy_context()
            future = executor.submit(context.run, worker, item)
            pending.append((next_index, item, future))
            next_index += 1

    try:
        fill()
        while pending and not cancelled():
            index, item, future = pending.popleft()
            try:
                value = future.result()
            except Exception as exc:
                yield BatchItemResult(index=index, item=item, error=exc)
            else:
                yield BatchItemResult(index=index, item=item, value=value)
            fill()
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
