"""Bounded low-cardinality Prometheus metrics without an SDK dependency."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
import math
import re


_METRIC = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class MetricsRegistry:
    def __init__(self, *, allowed_label_values: dict[str, frozenset[str]] | None = None):
        self.allowed_label_values = dict(allowed_label_values or {})
        if any(_LABEL.fullmatch(name) is None for name in self.allowed_label_values):
            raise ValueError("metric label names are invalid")
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._help: dict[str, str] = {}
        self._lock = RLock()

    def increment(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
        help_text: str = "DocARAG operational metric",
    ) -> None:
        if _METRIC.fullmatch(name) is None:
            raise ValueError("metric name is invalid")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("counter increment must be finite and non-negative")
        normalized = self._normalize_labels(labels or {})
        if "\n" in help_text or not help_text.strip():
            raise ValueError("metric help text is invalid")
        with self._lock:
            previous = self._help.setdefault(name, help_text)
            if previous != help_text:
                raise ValueError("metric help text is immutable")
            self._values[(name, normalized)] += amount

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name in sorted(self._help):
                lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} counter")
                for (metric_name, labels), value in sorted(self._values.items()):
                    if metric_name != name:
                        continue
                    rendered_labels = ""
                    if labels:
                        inner = ",".join(
                            f'{key}="{value}"' for key, value in labels
                        )
                        rendered_labels = "{" + inner + "}"
                    lines.append(f"{name}{rendered_labels} {value:g}")
            return "\n".join(lines) + ("\n" if lines else "")

    def _normalize_labels(self, labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
        if set(labels).difference(self.allowed_label_values):
            raise ValueError("metric contains an undeclared high-cardinality label")
        for name, value in labels.items():
            if value not in self.allowed_label_values[name]:
                raise ValueError(f"metric label {name} has an undeclared value")
        return tuple(sorted(labels.items()))


__all__ = ["MetricsRegistry"]
