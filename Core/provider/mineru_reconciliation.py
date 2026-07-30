"""Reconcile MinerU's two page-content representations without positional drift.

MinerU emits both ``middle_json.pdf_info[].para_blocks`` (layout/provenance) and
``content_list`` (the normalized content consumed by BookRAG).  The two streams
are not guaranteed to have identical cardinality or ordering.  This module
joins them using page-local evidence and never falls back to a document-global
zip.
"""

from __future__ import annotations

import bisect
import copy
import difflib
import hashlib
import html
import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable


log = logging.getLogger(__name__)

_IDENTITY_KEYS = (
    "id",
    "uid",
    "uuid",
    "block_id",
    "content_id",
    "element_id",
    "para_id",
    "paragraph_id",
    "block_idx",
    "para_idx",
    "index",
)
_GENERIC_ID_KEYS = frozenset(_IDENTITY_KEYS) - {"index", "block_idx", "para_idx"}
_ASSET_KEYS = frozenset({"img_path", "image_path"})
_PAGE_KEYS = ("page_idx", "page_index", "page_no", "page_number")
_SPACE_RE = re.compile(r"\s+")
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+.!~$<>|\-])")
#: A list marker at the start of a line or after whitespace. Anchored so a
#: hyphen inside a word ("Near-Sightedness") is never touched.
_LIST_MARKER_RE = re.compile(r"(?:^|(?<=\s))[•·▪◦‣∙*\-–—]+(?=\s)", re.MULTILINE)


class MinerUReconciliationError(ValueError):
    """Raised when strict reconciliation cannot account for every observation."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ReconciliationResult:
    items: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Observation:
    source_index: int
    page_index: Any
    page_position: int
    value: dict[str, Any]
    family: str
    identities: frozenset[str]
    asset: str | None
    text_key: str | None
    format_text: str
    format_key: str | None
    empty_text: bool
    text_preview: str
    bbox: tuple[float, float, float, float] | None


def _json_scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _normal_page_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        return stripped
    return value


def _family(raw_type: Any) -> str:
    value = str(raw_type or "unknown").strip().casefold().replace("-", "_")
    if value in {
        "text",
        "title",
        "list",
        "reference",
        "paragraph",
        "code",
        "index",
    }:
        return "text"
    if value in {"image", "figure", "chart"}:
        return "image"
    if value in {"table", "table_body"}:
        return "table"
    if value in {
        "equation",
        "formula",
        "interline_equation",
        "inline_equation",
        "isolated_equation",
    }:
        return "equation"
    return value


def _identity_tokens(value: dict[str, Any]) -> frozenset[str]:
    tokens: set[str] = set()
    for key in _IDENTITY_KEYS:
        scalar = _json_scalar(value.get(key))
        if scalar is None:
            continue
        tokens.add(f"{key}:{scalar}")
        if key in _GENERIC_ID_KEYS:
            tokens.add(f"native-id:{scalar}")
    return frozenset(tokens)


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first_string(value: Any, keys: frozenset[str]) -> str | None:
    for node in _walk_dicts(value):
        for key in keys:
            candidate = node.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _asset_key(value: dict[str, Any]) -> str | None:
    raw = _first_string(value, _ASSET_KEYS)
    if not raw:
        return None
    # MinerU uses ``images/foo`` in content_list and often just ``foo`` in a
    # middle-json span.  The page-local basename is their stable common part.
    return PurePosixPath(raw.replace("\\", "/")).name.casefold()


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for child in value:
            result.extend(_flatten_strings(child))
        return result
    return []


def _semantic_text(value: dict[str, Any], family: str, *, middle: bool) -> str:
    if not middle:
        if family == "table":
            values = _flatten_strings(value.get("table_body"))
        elif family == "image":
            values = []
            for key in ("image_caption", "image_footnote", "text"):
                values.extend(_flatten_strings(value.get(key)))
        else:
            values = _flatten_strings(value.get("text"))
        return " ".join(part for part in values if part.strip())

    if family == "table":
        table = _first_string(value, frozenset({"html", "table_body"}))
        return table or ""

    values: list[str] = []
    for node in _walk_dicts(value):
        candidate = node.get("content")
        if isinstance(candidate, str) and candidate.strip():
            values.append(candidate)
    return " ".join(values)


def _normalize_text(value: str) -> str:
    return _SPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    ).strip()


def _format_insensitive_text(value: str) -> str:
    """Canonicalize renderer-only markup while retaining semantic characters."""

    normalized = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    normalized = _MARKDOWN_ESCAPE_RE.sub(r"\1", normalized)
    normalized = normalized.replace("\u00ad", "")
    # MinerU occasionally preserves an OCR line-break hyphen in middle JSON
    # while its content renderer joins the word (``on- prem`` -> ``onprem``).
    normalized = re.sub(r"(?<=\w)[\-‐‑]\s+(?=\w)", "", normalized)
    # Dollar signs are renderer-added math delimiters in one representation and
    # Markdown-escaped currency markers in the other.  The surrounding formula
    # or amount remains intact; all other operators and punctuation are kept.
    normalized = normalized.replace("$", "")
    # MinerU 3.x writes the same list item as "• item" in middle JSON and
    # "- item" through its content renderer.  Unmatched observations are
    # retained rather than dropped, so an unfolded marker does not lose the
    # list -- it emits the list twice, once per representation, at two
    # different coordinate scales.
    normalized = _LIST_MARKER_RE.sub("", normalized)
    return "".join(character for character in normalized if not character.isspace())


def _text_identity(value: str) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{len(normalized)}:{digest}"


def _page_size(value: dict[str, Any]) -> tuple[float, float] | None:
    raw = value.get("page_size")
    if isinstance(raw, dict):
        raw = (raw.get("width", raw.get("w")), raw.get("height", raw.get("h")))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            width, height = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0 and math.isfinite(width) and math.isfinite(height):
            return width, height
    return None


def _raw_bbox(value: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox")
    if isinstance(raw, dict):
        raw = (
            raw.get("x0", raw.get("left")),
            raw.get("y0", raw.get("top")),
            raw.get("x1", raw.get("right")),
            raw.get("y1", raw.get("bottom")),
        )
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(raw[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox(
    value: dict[str, Any], page_size: tuple[float, float] | None
) -> tuple[float, float, float, float] | None:
    result = _raw_bbox(value)
    if result is None:
        return None
    if page_size and max(abs(item) for item in result) > 1.0:
        width, height = page_size
        result = (
            result[0] / width,
            result[1] / height,
            result[2] / width,
            result[3] / height,
        )
    return tuple(round(item, 6) for item in result)


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _observation(
    value: dict[str, Any],
    *,
    source_index: int,
    page_index: Any,
    page_position: int,
    page_size: tuple[float, float] | None,
    middle: bool,
) -> _Observation:
    family = _family(value.get("type"))
    text = _semantic_text(value, family, middle=middle)
    format_text = _format_insensitive_text(text)
    return _Observation(
        source_index=source_index,
        page_index=page_index,
        page_position=page_position,
        value=value,
        family=family,
        identities=_identity_tokens(value),
        asset=_asset_key(value),
        text_key=_text_identity(text),
        format_text=format_text,
        format_key=_text_identity(format_text),
        empty_text=family in {"text", "image", "table", "equation"}
        and not _normalize_text(text),
        text_preview=_SPACE_RE.sub(" ", text).strip()[:160],
        bbox=_bbox(value, page_size),
    )


def _content_page(
    content: dict[str, Any], known_pages: set[Any]
) -> tuple[Any | None, str | None]:
    for key in _PAGE_KEYS:
        if key not in content:
            continue
        page = _normal_page_value(content[key])
        # Explicit page-number fields are commonly one-based.  Only apply the
        # conversion when it identifies an existing page; page-index fields
        # remain zero-based.
        if key in {"page_no", "page_number"} and isinstance(page, int):
            zero_based = page - 1
            if zero_based in known_pages:
                return zero_based, key
        try:
            if page in known_pages:
                return page, key
        except TypeError:
            return None, key
        return None, key
    if len(known_pages) == 1:
        return next(iter(known_pages)), None
    return None, None


def _pair_by_tokens(
    middle: list[_Observation],
    content: list[_Observation],
    unmatched_middle: set[int],
    unmatched_content: set[int],
    token_getter: Callable[[_Observation], Iterable[str]],
) -> list[tuple[int, int]]:
    """Return only pairs uniquely supported by a shared evidence token."""

    accepted: list[tuple[int, int]] = []
    while True:
        middle_tokens: dict[str, set[int]] = defaultdict(set)
        content_tokens: dict[str, set[int]] = defaultdict(set)
        for index in unmatched_middle:
            for token in token_getter(middle[index]):
                middle_tokens[token].add(index)
        for index in unmatched_content:
            for token in token_getter(content[index]):
                content_tokens[token].add(index)

        proposals: dict[int, set[int]] = defaultdict(set)
        reverse: dict[int, set[int]] = defaultdict(set)
        for token in middle_tokens.keys() & content_tokens.keys():
            left, right = middle_tokens[token], content_tokens[token]
            if len(left) != 1 or len(right) != 1:
                continue
            middle_index, content_index = next(iter(left)), next(iter(right))
            if middle[middle_index].family != content[content_index].family:
                continue
            proposals[middle_index].add(content_index)
            reverse[content_index].add(middle_index)

        current = [
            (middle_index, next(iter(candidates)))
            for middle_index, candidates in proposals.items()
            if len(candidates) == 1
            and len(reverse[next(iter(candidates))]) == 1
        ]
        if not current:
            return accepted
        for middle_index, content_index in current:
            unmatched_middle.remove(middle_index)
            unmatched_content.remove(content_index)
            accepted.append((middle_index, content_index))


def _pair_by_geometry(
    middle: list[_Observation],
    content: list[_Observation],
    unmatched_middle: set[int],
    unmatched_content: set[int],
    threshold: float,
) -> list[tuple[int, int]]:
    candidates: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)
    for middle_index in unmatched_middle:
        middle_item = middle[middle_index]
        if middle_item.bbox is None:
            continue
        for content_index in unmatched_content:
            content_item = content[content_index]
            if (
                middle_item.family != content_item.family
                or content_item.bbox is None
                or _iou(middle_item.bbox, content_item.bbox) < threshold
            ):
                continue
            candidates[middle_index].add(content_index)
            reverse[content_index].add(middle_index)

    accepted = [
        (middle_index, next(iter(options)))
        for middle_index, options in candidates.items()
        if len(options) == 1 and len(reverse[next(iter(options))]) == 1
    ]
    for middle_index, content_index in accepted:
        unmatched_middle.remove(middle_index)
        unmatched_content.remove(content_index)
    return accepted


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shorter, longer = sorted((len(left), len(right)))
    if shorter / longer < 0.85:
        return 0.0
    # Layout paragraphs can be very long.  A bounded head/tail representation
    # keeps this fallback predictable while exact hash stages still use all
    # content.
    if longer > 4096:
        left = left[:2048] + left[-2048:]
        right = right[:2048] + right[-2048:]
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _pair_by_similarity(
    middle: list[_Observation],
    content: list[_Observation],
    unmatched_middle: set[int],
    unmatched_content: set[int],
    *,
    threshold: float = 0.97,
    ambiguity_margin: float = 0.02,
) -> list[tuple[int, int]]:
    middle_rankings: dict[int, list[tuple[float, int]]] = defaultdict(list)
    content_rankings: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for middle_index in unmatched_middle:
        middle_item = middle[middle_index]
        if not middle_item.format_text:
            continue
        for content_index in unmatched_content:
            content_item = content[content_index]
            if (
                middle_item.family != content_item.family
                or not content_item.format_text
            ):
                continue
            score = _similarity(middle_item.format_text, content_item.format_text)
            if score < threshold:
                continue
            middle_rankings[middle_index].append((score, content_index))
            content_rankings[content_index].append((score, middle_index))

    def unique_best(ranking: list[tuple[float, int]]) -> int | None:
        ranking.sort(reverse=True)
        if not ranking:
            return None
        if len(ranking) > 1 and ranking[0][0] - ranking[1][0] < ambiguity_margin:
            return None
        return ranking[0][1]

    middle_best = {
        index: candidate
        for index, ranking in middle_rankings.items()
        if (candidate := unique_best(ranking)) is not None
    }
    content_best = {
        index: candidate
        for index, ranking in content_rankings.items()
        if (candidate := unique_best(ranking)) is not None
    }
    accepted = [
        (middle_index, content_index)
        for middle_index, content_index in middle_best.items()
        if content_best.get(content_index) == middle_index
    ]
    for middle_index, content_index in accepted:
        unmatched_middle.remove(middle_index)
        unmatched_content.remove(content_index)
    return accepted


def _pair_anchored_duplicate_tokens(
    middle: list[_Observation],
    content: list[_Observation],
    unmatched_middle: set[int],
    unmatched_content: set[int],
    matches: dict[int, tuple[int, str]],
    token_getter: Callable[[_Observation], str | None],
) -> list[tuple[int, int]]:
    """Pair duplicate groups under anchors or identical page-local ordinals."""

    middle_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    content_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in unmatched_middle:
        token = token_getter(middle[index])
        if token:
            middle_groups[(middle[index].family, token)].append(index)
    for index in unmatched_content:
        token = token_getter(content[index])
        if token:
            content_groups[(content[index].family, token)].append(index)

    accepted: list[tuple[int, int]] = []
    matched_content_to_middle = {
        content_index: middle_index
        for middle_index, (content_index, _method) in matches.items()
    }
    for token in sorted(middle_groups.keys() & content_groups.keys()):
        middle_group = sorted(middle_groups[token])
        content_group = sorted(content_groups[token])
        if len(middle_group) < 2 or len(middle_group) != len(content_group):
            continue

        previous_middle = max(
            (index for index in matches if index < middle_group[0]), default=None
        )
        following_middle = min(
            (index for index in matches if index > middle_group[-1]), default=None
        )
        previous_content = max(
            (
                index
                for index in matched_content_to_middle
                if index < content_group[0]
            ),
            default=None,
        )
        following_content = min(
            (
                index
                for index in matched_content_to_middle
                if index > content_group[-1]
            ),
            default=None,
        )
        has_anchor = previous_middle is not None or following_middle is not None
        previous_consistent = (
            previous_middle is None
            and previous_content is None
            or previous_middle is not None
            and previous_content is not None
            and matches[previous_middle][0] == previous_content
        )
        following_consistent = (
            following_middle is None
            and following_content is None
            or following_middle is not None
            and following_content is not None
            and matches[following_middle][0] == following_content
        )
        same_page_ordinals = (
            len(middle) == len(content) and middle_group == content_group
        )
        if not same_page_ordinals and (
            not has_anchor or not previous_consistent or not following_consistent
        ):
            continue
        pairs = list(zip(middle_group, content_group))
        accepted.extend(pairs)
        for middle_index, content_index in pairs:
            unmatched_middle.remove(middle_index)
            unmatched_content.remove(content_index)
            matches[middle_index] = (content_index, "anchored_duplicate_text")
            matched_content_to_middle[content_index] = middle_index
    return accepted


def _evidence_tokens(
    observation: _Observation, attribute: str, prefix: str
) -> Iterable[str]:
    value = getattr(observation, attribute)
    if value is None:
        return ()
    if isinstance(value, (set, frozenset, list)):
        return (f"{prefix}:{item}" for item in value)
    return (f"{prefix}:{value}",)


def _match_page(
    middle: list[_Observation], content: list[_Observation]
) -> tuple[dict[int, tuple[int, str]], set[int], set[int], Counter[str]]:
    unmatched_middle = set(range(len(middle)))
    unmatched_content = set(range(len(content)))
    matches: dict[int, tuple[int, str]] = {}
    methods: Counter[str] = Counter()

    stages: tuple[tuple[str, Callable[[_Observation], Iterable[str]]], ...] = (
        ("native_identity", lambda item: item.identities),
        ("asset_identity", lambda item: _evidence_tokens(item, "asset", "asset")),
        (
            "exact_geometry",
            lambda item: _evidence_tokens(item, "bbox", "bbox"),
        ),
        ("exact_text", lambda item: _evidence_tokens(item, "text_key", "text")),
        (
            "format_insensitive_text",
            lambda item: _evidence_tokens(item, "format_key", "format-text"),
        ),
        (
            "empty_semantic_text",
            lambda item: ("empty-semantic-text",) if item.empty_text else (),
        ),
    )
    for method, getter in stages:
        for middle_index, content_index in _pair_by_tokens(
            middle, content, unmatched_middle, unmatched_content, getter
        ):
            matches[middle_index] = (content_index, method)
            methods[method] += 1

    # Exact bbox tokens use rounded coordinates.  A high-overlap pass handles
    # harmless coordinate rounding/model-version changes without accepting a
    # merely nearby block.
    for middle_index, content_index in _pair_by_geometry(
        middle, content, unmatched_middle, unmatched_content, threshold=0.90
    ):
        matches[middle_index] = (content_index, "overlap_geometry")
        methods["overlap_geometry"] += 1

    for middle_index, content_index in _pair_by_similarity(
        middle, content, unmatched_middle, unmatched_content
    ):
        matches[middle_index] = (content_index, "normalized_similarity")
        methods["normalized_similarity"] += 1

    for middle_index, content_index in _pair_anchored_duplicate_tokens(
        middle,
        content,
        unmatched_middle,
        unmatched_content,
        matches,
        lambda item: item.format_key
        or ("empty-semantic-text" if item.empty_text else None),
    ):
        methods["anchored_duplicate_text"] += 1

    return matches, unmatched_middle, unmatched_content, methods


def _describe(observation: _Observation, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "source_index": observation.source_index,
        "page_idx": observation.page_index,
        "page_position": observation.page_position,
        "type": observation.value.get("type", "unknown"),
        "native_index": observation.value.get("index"),
        "bbox": list(observation.bbox) if observation.bbox else None,
        "asset": observation.asset,
        "text_preview": observation.text_preview,
    }


def _ambiguous_groups(
    middle: list[_Observation],
    content: list[_Observation],
    unmatched_middle: set[int],
    unmatched_content: set[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    evidence_getters = {
        "native_identity": lambda item: item.identities,
        "asset_identity": lambda item: ({item.asset} if item.asset else set()),
        "exact_geometry": lambda item: ({item.bbox} if item.bbox else set()),
        "exact_text": lambda item: ({item.text_key} if item.text_key else set()),
    }
    for method, getter in evidence_getters.items():
        middle_groups: dict[Any, list[int]] = defaultdict(list)
        content_groups: dict[Any, list[int]] = defaultdict(list)
        for index in unmatched_middle:
            for token in getter(middle[index]):
                middle_groups[(middle[index].family, token)].append(index)
        for index in unmatched_content:
            for token in getter(content[index]):
                content_groups[(content[index].family, token)].append(index)
        for token in middle_groups.keys() & content_groups.keys():
            left, right = middle_groups[token], content_groups[token]
            if len(left) > 1 or len(right) > 1:
                result.append(
                    {
                        "evidence": method,
                        "type_family": token[0],
                        "middle_positions": [middle[index].page_position for index in left],
                        "content_positions": [content[index].source_index for index in right],
                    }
                )
    return result


def _resolved_image_path(raw: str, parse_dir: str | os.PathLike[str]) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    parse_path = Path(parse_dir)
    direct = parse_path / path
    if direct.exists() or len(path.parts) > 1:
        return str(direct)
    return str(parse_path / "images" / path)


def _prepare_content_item(
    observation: _Observation,
    middle: dict[str, Any] | None,
    parse_dir: str | os.PathLike[str],
    marker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(observation.value)
    result.setdefault("page_idx", observation.page_index)
    result["middle_json"] = copy.deepcopy(middle) if middle is not None else {}
    if marker is not None:
        result["_reconciliation"] = marker
    raw_path = result.get("img_path")
    if isinstance(raw_path, str) and raw_path:
        result["img_path"] = _resolved_image_path(raw_path, parse_dir)
        if not os.path.exists(result["img_path"]):
            log.error("Image path does not exist: %s", result["img_path"])
    return result


def _content_placement_buckets(
    matches: dict[int, tuple[int, str]],
    unmatched_content: set[int],
    middle_count: int,
) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Place unmatched content between non-conflicting matched anchors.

    The anchors come from independently established matches, not from a zip.
    If content ordering conflicts with middle ordering, the observation is
    appended to the page instead of guessing which layout position is right.
    """

    content_to_middle = {
        content_index: middle_index
        for middle_index, (content_index, _method) in matches.items()
    }
    matched_content = sorted(content_to_middle)
    buckets: dict[int, list[int]] = defaultdict(list)
    reasons: dict[int, str] = {}
    for content_index in sorted(unmatched_content):
        insertion = bisect.bisect_left(matched_content, content_index)
        previous = matched_content[insertion - 1] if insertion else None
        following = (
            matched_content[insertion] if insertion < len(matched_content) else None
        )
        previous_middle = (
            content_to_middle[previous] if previous is not None else None
        )
        following_middle = (
            content_to_middle[following] if following is not None else None
        )
        if previous_middle is None and following_middle is not None:
            slot, reason = following_middle, "before_next_matched_anchor"
        elif previous_middle is not None and following_middle is None:
            slot, reason = previous_middle + 1, "after_previous_matched_anchor"
        elif (
            previous_middle is not None
            and following_middle is not None
            and previous_middle < following_middle
        ):
            slot, reason = following_middle, "between_matched_anchors"
        else:
            slot, reason = middle_count, "page_end_no_consistent_anchors"
        buckets[min(max(slot, 0), middle_count)].append(content_index)
        reasons[content_index] = reason
    return buckets, reasons


def _synthetic_content_item(
    observation: _Observation, parse_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    block = observation.value
    family = observation.family
    item: dict[str, Any] = {
        "type": family if family in {"text", "image", "table", "equation"} else "text",
        "page_idx": observation.page_index,
    }
    text = _semantic_text(block, family, middle=True)
    if family in {"text", "equation"}:
        item["text"] = text
    if str(block.get("type", "")).casefold() == "title":
        item["text_level"] = 1
    if family == "table":
        item.update(
            table_body=text,
            table_caption=[],
            table_footnote=[],
        )
    elif family == "image":
        item.update(image_caption=[], image_footnote=[])

    raw_asset = _first_string(block, _ASSET_KEYS)
    if raw_asset and family in {"image", "table", "equation"}:
        item["img_path"] = _resolved_image_path(raw_asset, parse_dir)
        if not os.path.exists(item["img_path"]):
            log.error("Image path does not exist: %s", item["img_path"])
    item["middle_json"] = copy.deepcopy(block)
    item["_reconciliation"] = {
        "status": "unmatched_middle",
        "reason": "no_confident_content_match",
        "middle_page_position": observation.page_position,
    }
    return item


def _validate_inputs(
    middle_json: dict[str, Any], content_list: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    if policy not in {"retain", "error"}:
        raise ValueError("mismatch_policy must be 'retain' or 'error'")
    if not isinstance(middle_json, dict) or not isinstance(middle_json.get("pdf_info"), list):
        raise ValueError("middle_json must contain a list at 'pdf_info'")
    if not isinstance(content_list, list) or not all(
        isinstance(item, dict) for item in content_list
    ):
        raise ValueError("content_list must be a list of objects")
    pages = middle_json["pdf_info"]
    if not all(isinstance(page, dict) for page in pages):
        raise ValueError("middle_json 'pdf_info' entries must be objects")
    return pages


def reconcile_middle_content(
    middle_json: dict[str, Any],
    content_list: list[dict[str, Any]],
    parse_dir: str | os.PathLike[str],
    *,
    mismatch_policy: str = "retain",
) -> ReconciliationResult:
    """Join MinerU observations page-by-page using conservative evidence.

    ``retain`` keeps unmatched middle and content observations as separately
    marked output items.  ``error`` raises :class:`MinerUReconciliationError`
    with the same structured diagnostics.  Neither mode attaches a later
    content item to an earlier layout block merely because their positions line
    up after a gap.
    """

    pages = _validate_inputs(middle_json, content_list, mismatch_policy)
    page_order: list[Any] = []
    page_sizes: dict[Any, tuple[float, float] | None] = {}
    middle_by_page: dict[Any, list[_Observation]] = {}
    seen_pages: set[Any] = set()
    middle_source_index = 0

    for page_position, page in enumerate(pages):
        page_index = _normal_page_value(page.get("page_idx", page_position))
        try:
            duplicate_page = page_index in seen_pages
        except TypeError as exc:
            raise ValueError(
                f"middle-json page_idx must be a scalar, got {page_index!r}"
            ) from exc
        if duplicate_page:
            raise ValueError(f"duplicate middle-json page_idx: {page_index!r}")
        seen_pages.add(page_index)
        page_order.append(page_index)
        size = _page_size(page)
        page_sizes[page_index] = size
        blocks = page.get("para_blocks", [])
        if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
            raise ValueError(f"page {page_index!r} para_blocks must be a list of objects")
        observations: list[_Observation] = []
        for page_block_position, block in enumerate(blocks):
            observations.append(
                _observation(
                    block,
                    source_index=middle_source_index,
                    page_index=page_index,
                    page_position=page_block_position,
                    page_size=size,
                    middle=True,
                )
            )
            middle_source_index += 1
        middle_by_page[page_index] = observations

    content_by_page: dict[Any, list[_Observation]] = defaultdict(list)
    orphan_content: list[_Observation] = []
    for content_index, content in enumerate(content_list):
        page_index, _page_key = _content_page(content, seen_pages)
        if page_index is None:
            orphan_content.append(
                _observation(
                    content,
                    source_index=content_index,
                    page_index=None,
                    page_position=len(orphan_content),
                    page_size=None,
                    middle=False,
                )
            )
            continue
        page_content = content_by_page[page_index]
        page_content.append(
            _observation(
                content,
                source_index=content_index,
                page_index=page_index,
                page_position=len(page_content),
                page_size=page_sizes[page_index],
                middle=False,
            )
        )

    items: list[dict[str, Any]] = []
    page_diagnostics: list[dict[str, Any]] = []
    unmatched_middle_descriptions: list[dict[str, Any]] = []
    unmatched_content_descriptions: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    total_methods: Counter[str] = Counter()

    for page_index in page_order:
        middle = middle_by_page[page_index]
        content = content_by_page.get(page_index, [])
        matches, missing_middle, missing_content, methods = _match_page(middle, content)
        total_methods.update(methods)
        page_ambiguous = _ambiguous_groups(
            middle, content, missing_middle, missing_content
        )
        for entry in page_ambiguous:
            entry["page_idx"] = page_index
        ambiguous.extend(page_ambiguous)

        page_output_start = len(items)
        placement_buckets, placement_reasons = _content_placement_buckets(
            matches, missing_content, len(middle)
        )
        if mismatch_policy == "retain":
            for middle_index in range(len(middle) + 1):
                for content_index in placement_buckets.get(middle_index, []):
                    observation = content[content_index]
                    description = _describe(observation, "content_list")
                    description["output_index"] = len(items)
                    description["placement"] = placement_reasons[content_index]
                    unmatched_content_descriptions.append(description)
                    items.append(
                        _prepare_content_item(
                            observation,
                            None,
                            parse_dir,
                            marker={
                                "status": "unmatched_content",
                                "reason": "no_confident_middle_match",
                                "placement": placement_reasons[content_index],
                                "content_source_index": observation.source_index,
                            },
                        )
                    )
                if middle_index == len(middle):
                    continue
                middle_observation = middle[middle_index]
                if middle_index in matches:
                    content_index, _method = matches[middle_index]
                    items.append(
                        _prepare_content_item(
                            content[content_index], middle_observation.value, parse_dir
                        )
                    )
                else:
                    description = _describe(middle_observation, "middle_json")
                    description["output_index"] = len(items)
                    unmatched_middle_descriptions.append(description)
                    items.append(_synthetic_content_item(middle_observation, parse_dir))

        # In error mode the descriptions are still required even though no
        # output item is constructed.
        if mismatch_policy == "error":
            unmatched_middle_descriptions.extend(
                _describe(middle[index], "middle_json") for index in sorted(missing_middle)
            )
            for content_index in sorted(missing_content):
                description = _describe(content[content_index], "content_list")
                description["placement"] = placement_reasons[content_index]
                unmatched_content_descriptions.append(description)

        page_diagnostics.append(
            {
                "page_idx": page_index,
                "middle_count": len(middle),
                "content_count": len(content),
                "matched_count": len(matches),
                "unmatched_middle_count": len(missing_middle),
                "unmatched_content_count": len(missing_content),
                "output_start": page_output_start if mismatch_policy == "retain" else None,
                "match_methods": dict(sorted(methods.items())),
            }
        )

    for observation in orphan_content:
        description = _describe(observation, "content_list")
        description["reason"] = "missing_or_unknown_page_identity"
        description["output_index"] = len(items) if mismatch_policy == "retain" else None
        unmatched_content_descriptions.append(description)
        if mismatch_policy == "retain":
            items.append(
                _prepare_content_item(
                    observation,
                    None,
                    parse_dir,
                    marker={
                        "status": "unmatched_content",
                        "reason": "missing_or_unknown_page_identity",
                        "content_source_index": observation.source_index,
                    },
                )
            )

    mismatch_count = len(unmatched_middle_descriptions) + len(
        unmatched_content_descriptions
    )
    if mismatch_count == 0:
        status = "matched"
    elif mismatch_policy == "retain":
        status = "retained_mismatch"
    else:
        status = "failed_mismatch"
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "policy": mismatch_policy,
        "status": status,
        "summary": {
            "middle_count": sum(len(value) for value in middle_by_page.values()),
            "content_count": len(content_list),
            "matched_count": sum(total_methods.values()),
            "unmatched_middle_count": len(unmatched_middle_descriptions),
            "unmatched_content_count": len(unmatched_content_descriptions),
            "orphan_content_count": len(orphan_content),
            "output_count": len(items) if mismatch_policy == "retain" else None,
            "match_methods": dict(sorted(total_methods.items())),
        },
        "pages": page_diagnostics,
        "unmatched_middle": unmatched_middle_descriptions,
        "unmatched_content": unmatched_content_descriptions,
        "ambiguous_groups": ambiguous,
    }

    if mismatch_count and mismatch_policy == "error":
        summary = diagnostics["summary"]
        raise MinerUReconciliationError(
            "MinerU reconciliation failed: "
            f"{summary['unmatched_middle_count']} middle-json and "
            f"{summary['unmatched_content_count']} content-list observations "
            "lack a unique page-local match",
            diagnostics,
        )

    return ReconciliationResult(items=items, diagnostics=diagnostics)
