"""Dependency-light structural validation for source-cited answers.

This module proves that citation markers in a generated answer resolve to
hydrated canonical evidence supplied to that generation request.  It does not
claim semantic entailment; a later claim-support verifier must evaluate that
separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


ANSWER_CITATION_VALIDATION_SCHEMA_VERSION = "docarag.answer-citations.v2"
_VALID_LABEL = re.compile(r"(?:Q[1-9][0-9]*-)?E[1-9][0-9]*")
_BRACKET_GROUP = re.compile(r"\[[^\[\]\r\n]*\]")
_CITATION_LIKE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:Q[0-9]+-)?E[0-9]+(?![A-Za-z0-9_])"
)
_CITATION_ID = re.compile(r"citation_[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class AnswerCitationValidation:
    """Machine-readable structural citation outcome for one final answer."""

    required: bool
    status: str
    cited_labels: tuple[str, ...]
    resolved_citation_ids: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    unbound_labels: tuple[str, ...]
    malformed_markers: tuple[str, ...]
    citation_only: bool
    allowed_bindings: tuple[tuple[str, str], ...]
    message: str

    @property
    def valid(self) -> bool:
        return self.status in {"verified", "not_required"}

    @property
    def unknown_evidence_labels(self) -> tuple[str, ...]:
        """Backward-compatible bracketed form used by answer traces."""

        return tuple(f"[{label}]" for label in self.unknown_labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ANSWER_CITATION_VALIDATION_SCHEMA_VERSION,
            "required": self.required,
            "status": self.status,
            "valid": self.valid,
            "cited_labels": list(self.cited_labels),
            "resolved_citation_ids": list(self.resolved_citation_ids),
            "unknown_labels": list(self.unknown_labels),
            "unknown_evidence_labels": list(self.unknown_evidence_labels),
            "unbound_labels": list(self.unbound_labels),
            "malformed_markers": list(self.malformed_markers),
            "citation_only": self.citation_only,
            "allowed_bindings": [
                {"evidence_label": label, "citation_id": citation_id}
                for label, citation_id in self.allowed_bindings
            ],
            "message": self.message,
        }


def _ordered_unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _normalize_bindings(
    evidence_bindings: Mapping[str, str | None],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(evidence_bindings, Mapping):
        raise TypeError("evidence_bindings must be a mapping")
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_label, raw_citation_id in evidence_bindings.items():
        if not isinstance(raw_label, str):
            raise TypeError("evidence binding labels must be strings")
        label = (
            raw_label[1:-1]
            if raw_label.startswith("[") and raw_label.endswith("]")
            else raw_label
        )
        if _VALID_LABEL.fullmatch(label) is None:
            raise ValueError(f"invalid evidence label: {raw_label!r}")
        if label in seen:
            raise ValueError(f"duplicate evidence label: {label}")
        seen.add(label)
        citation_id = raw_citation_id if isinstance(raw_citation_id, str) else ""
        normalized.append((label, citation_id))
    return tuple(normalized)


def _parse_answer_markers(
    answer: str,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Parse the deliberately narrow answer-citation grammar.

    Only one label inside one pair of square brackets is supported. Compound
    groups, ranges, zero-valued labels, and raw canonical citation IDs are
    reported as malformed instead of being silently ignored. Ordinary bracketed
    prose such as ``[Table 2]`` remains outside the citation grammar.
    """

    cited: list[str] = []
    malformed: list[str] = []
    bracket_spans: list[tuple[int, int]] = []
    valid_marker_spans: list[tuple[int, int]] = []
    for match in _BRACKET_GROUP.finditer(answer):
        bracket_spans.append(match.span())
        marker = match.group(0)
        content = marker[1:-1]
        if _VALID_LABEL.fullmatch(content) is not None:
            cited.append(content)
            valid_marker_spans.append(match.span())
        elif (
            _CITATION_LIKE_TOKEN.search(content) is not None
            or _CITATION_ID.search(content) is not None
        ):
            malformed.append(marker)

    # Canonical citation IDs are trace identities, not an answer marker format.
    # Catch a model printing one directly unless it was already captured as a
    # malformed bracket group.
    bracket_index = 0
    for match in _CITATION_ID.finditer(answer):
        while (
            bracket_index < len(bracket_spans)
            and bracket_spans[bracket_index][1] <= match.start()
        ):
            bracket_index += 1
        inside_brackets = (
            bracket_index < len(bracket_spans)
            and bracket_spans[bracket_index][0] <= match.start()
            and match.end() <= bracket_spans[bracket_index][1]
        )
        if not inside_brackets:
            malformed.append(match.group(0))

    cited_tuple = _ordered_unique(cited)
    malformed_tuple = _ordered_unique(malformed)
    residual_parts: list[str] = []
    cursor = 0
    for start, end in valid_marker_spans:
        residual_parts.append(answer[cursor:start])
        cursor = end
    residual_parts.append(answer[cursor:])
    residual = "".join(residual_parts)
    citation_only = bool(cited_tuple) and not re.sub(r"[\W_]+", "", residual)
    return cited_tuple, malformed_tuple, citation_only


def validate_answer_citations(
    answer: str,
    evidence_bindings: Mapping[str, str | None] | Sequence[str],
    *,
    required: bool | None = None,
    citations_required: bool | None = None,
) -> AnswerCitationValidation:
    """Validate that every answer marker resolves to supplied canonical evidence."""

    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    if required is not None and not isinstance(required, bool):
        raise TypeError("required must be a boolean or None")
    if citations_required is not None and not isinstance(citations_required, bool):
        raise TypeError("citations_required must be a boolean or None")
    if (
        required is not None
        and citations_required is not None
        and required != citations_required
    ):
        raise ValueError("required and citations_required disagree")
    effective_required = (
        required
        if required is not None
        else citations_required
        if citations_required is not None
        else False
    )

    require_canonical_binding = isinstance(evidence_bindings, Mapping)
    if require_canonical_binding:
        bindings = _normalize_bindings(evidence_bindings)
    elif isinstance(evidence_bindings, Sequence) and not isinstance(
        evidence_bindings, (str, bytes, bytearray)
    ):
        bindings = _normalize_bindings(
            {
                label: ""
                for label in evidence_bindings
                if isinstance(label, str)
            }
        )
    else:
        raise TypeError("evidence_bindings must be a mapping or label sequence")
    binding_by_label = dict(bindings)
    cited, malformed, citation_only = _parse_answer_markers(answer)
    unknown = tuple(label for label in cited if label not in binding_by_label)
    unbound = tuple(
        label
        for label in cited
        if label in binding_by_label
        and require_canonical_binding
        and _CITATION_ID.fullmatch(binding_by_label[label]) is None
    )
    resolved = _ordered_unique(
        binding_by_label[label]
        for label in cited
        if label in binding_by_label
        and _CITATION_ID.fullmatch(binding_by_label[label]) is not None
    )

    if not effective_required:
        status = "not_required"
        message = "Canonical answer citation enforcement was not requested."
    elif malformed:
        status = "malformed_citations"
        message = (
            "The generated answer contains unsupported or malformed citation markers."
        )
    elif not bindings:
        status = "no_evidence_bindings"
        message = "No canonical evidence bindings were supplied to generation."
    elif not cited:
        status = "missing_citations"
        message = "The generated answer contains no evidence citation markers."
    elif unknown:
        status = "unknown_citations"
        message = "The generated answer cites evidence that was not supplied."
    elif unbound:
        status = "unbound_citations"
        message = "The generated answer cites evidence without a canonical citation ID."
    elif citation_only:
        status = "citation_only_answer"
        message = "The generated output contains citations but no substantive answer."
    else:
        status = "verified"
        message = "Every answer citation marker resolves to supplied canonical evidence."

    return AnswerCitationValidation(
        required=effective_required,
        status=status,
        cited_labels=cited,
        resolved_citation_ids=resolved,
        unknown_labels=unknown,
        unbound_labels=unbound,
        malformed_markers=malformed,
        citation_only=citation_only,
        allowed_bindings=bindings,
        message=message,
    )


__all__ = [
    "ANSWER_CITATION_VALIDATION_SCHEMA_VERSION",
    "AnswerCitationValidation",
    "validate_answer_citations",
]
