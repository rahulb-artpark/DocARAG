"""Accumulate how long each named operation in a stage takes.

Counters answer how often something happens; they do not answer where the time
goes.  Entity resolution reranked 11,201 times in a 1,871-page run, which is
85.4% of resolutions, but the duration of a rerank was never measured -- so that
share said nothing about its cost.  This module exists so optimisation targets
are chosen from seconds rather than from frequency.

Durations are summed per call, so under concurrency the total for a phase can
exceed the stage's wall clock.  ``wall_seconds`` is reported alongside for that
reason: the ratio between them is the achieved parallelism, not an error.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Iterator
import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PhaseTotal:
    """What one named operation cost across a stage."""

    calls: int
    seconds: float

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0


class PhaseTimings:
    """Thread-safe duration accumulator keyed by operation name.

    Phases are created on first use rather than declared, so instrumenting a
    new call site is a one-line change and no registry falls out of date.
    """

    def __init__(self, stage: str):
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        self.stage = stage.strip()
        self._lock = Lock()
        self._calls: dict[str, int] = {}
        self._seconds: dict[str, float] = {}
        self._started = perf_counter()

    @contextmanager
    def time(self, phase: str) -> Iterator[None]:
        """Attribute the enclosed block to ``phase``, including on failure.

        A call that raises still consumed time, and excluding it would make a
        stage that fails slowly look fast.
        """

        start = perf_counter()
        try:
            yield
        finally:
            self.add(phase, perf_counter() - start)

    def add(self, phase: str, seconds: float, *, calls: int = 1) -> None:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a non-empty string")
        if seconds < 0:
            raise ValueError("seconds may not be negative")
        phase = phase.strip()
        with self._lock:
            self._calls[phase] = self._calls.get(phase, 0) + calls
            self._seconds[phase] = self._seconds.get(phase, 0.0) + seconds

    @property
    def wall_seconds(self) -> float:
        return perf_counter() - self._started

    def totals(self) -> dict[str, PhaseTotal]:
        """Phases ordered by cost, which is the order they matter in."""

        with self._lock:
            items = [
                (phase, PhaseTotal(calls=self._calls[phase], seconds=seconds))
                for phase, seconds in self._seconds.items()
            ]
        items.sort(key=lambda item: item[1].seconds, reverse=True)
        return dict(items)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            phase: {
                "calls": total.calls,
                "seconds": round(total.seconds, 3),
                "mean_seconds": round(total.mean_seconds, 4),
            }
            for phase, total in self.totals().items()
        }

    def log_summary(self) -> None:
        totals = self.totals()
        wall = self.wall_seconds
        if not totals:
            log.info("%s timings: wall=%.1fs, nothing instrumented ran", self.stage, wall)
            return
        parts = [
            "%s=%.1fs/%d (%.1f%% wall, mean %.3fs)"
            % (
                phase,
                total.seconds,
                total.calls,
                (100.0 * total.seconds / wall) if wall > 0 else 0.0,
                total.mean_seconds,
            )
            for phase, total in totals.items()
        ]
        log.info("%s timings: wall=%.1fs  %s", self.stage, wall, "  ".join(parts))
