from copy import deepcopy
from typing import Optional, List, Dict

from Core.provider.llm import LLM
from Core.prompts.refiner_prompt import (
    TABLE_MERGE_PROMPT,
    MergeJudgmentsResponse,
    TEXT_MERGE_PROMPT,
    LABEL_MERGE_PROMPT,
    StitchingJudgmentsResponse,
)
from Core.utils.utils import num_tokens, get_json_content, enumerate_pdf_list
from Core.utils.bounded_executor import iter_bounded_ordered
import json
import re
import logging
from bs4 import BeautifulSoup

from statistics import median

from docarag.adapters.mineru_compat import unwrap_presentation
from docarag.derived_identity import (
    derived_evidence_id,
    derived_identity_payload,
    derived_output_identity,
)

log = logging.getLogger(__name__)

# Pairs packed into one judgment call, on top of the token budget. Raising it
# cuts the number of calls proportionally but changes what each prompt asks, so
# measure the response-length-mismatch rate before moving it.
MAX_PAIRS_PER_JUDGMENT = 3


_DIRECT_EVIDENCE_FIELDS = (
    "canonical_block_id",
    "canonical_revision_id",
    "observation_id",
)


def _ordered_unique(values) -> list[str]:
    """Return non-empty strings in first-seen order."""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _source_evidence_refs(content: dict) -> list[dict]:
    """Return the ordered immutable-source references supporting ``content``."""

    existing = content.get("source_evidence_refs")
    content_semantic_refs = _semantic_substructure_references(
        content.get("semantic_substructures")
    )
    refs: list[dict] = []
    if isinstance(existing, list):
        for raw_ref in existing:
            if not isinstance(raw_ref, dict):
                continue
            ref = deepcopy(raw_ref)
            # Migrate early semantic projections away from duplicating the
            # complete graph in every source reference.
            legacy_structures = ref.pop("semantic_substructures", None)
            if "semantic_substructure_refs" not in ref:
                semantic_refs = _semantic_substructure_references(
                    legacy_structures
                )
                if semantic_refs:
                    ref["semantic_substructure_refs"] = semantic_refs
            elif content_semantic_refs is not None:
                upgraded = _upgrade_legacy_semantic_receipts(
                    ref.get("semantic_substructure_refs"),
                    content_semantic_refs,
                )
                if upgraded is not None:
                    ref["semantic_substructure_refs"] = upgraded
            refs.append(ref)
    observation_ids = _ordered_unique(
        [
            content.get("observation_id"),
            *(content.get("source_observation_ids") or []),
        ]
    )
    has_direct_identity = any(
        (
            content.get("canonical_block_id"),
            content.get("canonical_revision_id"),
            content.get("observation_id"),
        )
    )
    if refs and not has_direct_identity:
        return refs
    if not has_direct_identity and not any(
        (
            content.get("page_id"),
            content.get("geometry", content.get("source_geometry")),
        )
    ):
        return refs

    direct_ref = {
        "canonical_block_id": content.get("canonical_block_id"),
        "canonical_revision_id": content.get("canonical_revision_id"),
        "observation_id": content.get("observation_id"),
        "source_observation_ids": observation_ids,
        "document_version_id": content.get("document_version_id"),
        "source_sha256": content.get("source_sha256"),
        "page_id": content.get("page_id"),
        "page_idx": content.get("page_idx"),
        "geometry": deepcopy(
            content.get("geometry", content.get("source_geometry"))
        ),
        "representation_ids": deepcopy(
            content.get("canonical_representation_ids")
        ),
        "asset_ids": deepcopy(content.get("canonical_asset_ids")),
        "selected_representation_ids": deepcopy(
            content.get("selected_representation_ids")
        ),
        "selected_asset_ids": deepcopy(content.get("selected_asset_ids")),
        "canonical_relations": deepcopy(content.get("canonical_relations")),
    }
    if content_semantic_refs:
        direct_ref["semantic_substructure_refs"] = content_semantic_refs
    direct_key = _ref_identity(direct_ref)
    if direct_key not in {_ref_identity(ref) for ref in refs}:
        refs.append(direct_ref)
    return refs


def _ref_identity(ref: dict) -> str:
    """Create a stable equality key without changing the reference payload."""

    return json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validated_semantic_substructures(raw_structures: object):
    """Parse a bounded full-graph array exactly once."""

    from docarag.domain import (
        MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
        SemanticSubstructure,
    )

    if raw_structures is None:
        return []
    if not isinstance(raw_structures, list) or len(
        raw_structures
    ) > MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK:
        return None
    by_id = {}
    for raw_structure in raw_structures:
        if not isinstance(raw_structure, dict):
            return None
        try:
            structure = SemanticSubstructure.from_dict(raw_structure)
        except (TypeError, ValueError):
            return None
        if not structure.is_source_bound:
            return None
        previous = by_id.get(structure.substructure_id)
        if previous is not None and previous != structure:
            return None
        by_id[structure.substructure_id] = structure
    return [by_id[substructure_id] for substructure_id in sorted(by_id)]


def _semantic_substructure_references(
    raw_structures: object,
) -> list[dict] | None:
    """Validate full graph DTOs and reduce them to bounded lineage receipts."""

    structures = _validated_semantic_substructures(raw_structures)
    if structures is None:
        return None
    receipts: list[dict] = []
    for structure in structures:
        receipt = {
            "substructure_id": structure.substructure_id,
            "semantic_sha256": structure.semantic_sha256,
            "comparison_sha256": structure.comparison_sha256,
            "source_observation_ids": list(structure.source_observation_ids),
            "source_substructure_ids": list(structure.source_substructure_ids),
        }
        receipts.append(receipt)
    return receipts


def _upgrade_legacy_semantic_receipts(
    raw_receipts: object,
    authoritative_receipts: list[dict],
) -> list[dict] | None:
    """Add comparison digests to the explicit transitional receipt shape."""

    if not isinstance(raw_receipts, list):
        return None
    comparison_by_id = {
        receipt.get("substructure_id"): receipt.get("comparison_sha256")
        for receipt in authoritative_receipts
    }
    upgraded: list[dict] = []
    for raw_receipt in raw_receipts:
        if not isinstance(raw_receipt, dict):
            return None
        receipt = deepcopy(raw_receipt)
        if "comparison_sha256" not in receipt:
            comparison_sha256 = comparison_by_id.get(
                receipt.get("substructure_id")
            )
            if not isinstance(comparison_sha256, str):
                return None
            receipt["comparison_sha256"] = comparison_sha256
        upgraded.append(receipt)
    return upgraded


def _normalized_semantic_receipts(
    raw_receipts: object,
    *,
    source_observation_ids: object,
    legacy_comparison_by_id: dict[str, str],
) -> list[dict] | None:
    """Preflight a source reference's small semantic lineage receipts."""

    from docarag.domain import (
        MAX_SEMANTIC_SOURCE_IDS,
        MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK,
    )

    if raw_receipts is None:
        return []
    if not isinstance(raw_receipts, list) or len(
        raw_receipts
    ) > MAX_SEMANTIC_SUBSTRUCTURES_PER_BLOCK:
        return None
    if not isinstance(source_observation_ids, list) or len(
        source_observation_ids
    ) > MAX_SEMANTIC_SOURCE_IDS:
        return None
    owner_observations = {
        value
        for value in source_observation_ids
        if isinstance(value, str) and 0 < len(value) <= 256
    }
    if len(owner_observations) != len(source_observation_ids):
        return None

    expected_fields = {
        "substructure_id",
        "semantic_sha256",
        "comparison_sha256",
        "source_observation_ids",
        "source_substructure_ids",
    }
    legacy_fields = expected_fields.difference({"comparison_sha256"})
    by_id: dict[str, dict] = {}
    for raw_receipt in raw_receipts:
        if not isinstance(raw_receipt, dict) or set(raw_receipt) not in (
            expected_fields,
            legacy_fields,
        ):
            return None
        substructure_id = raw_receipt.get("substructure_id")
        semantic_sha256 = raw_receipt.get("semantic_sha256")
        comparison_sha256 = raw_receipt.get("comparison_sha256")
        observations = raw_receipt.get("source_observation_ids")
        raw_ids = raw_receipt.get("source_substructure_ids")
        if (
            not isinstance(substructure_id, str)
            or not 0 < len(substructure_id) <= 256
            or not isinstance(semantic_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", semantic_sha256)
            or not isinstance(observations, list)
            or not observations
            or len(observations) > MAX_SEMANTIC_SOURCE_IDS
            or not isinstance(raw_ids, list)
            or len(raw_ids) != 1
        ):
            return None
        if comparison_sha256 is None:
            # Explicit compatibility for receipts emitted by the short-lived
            # pre-comparison schema. Never guess: the full exact graph must be
            # present at top level so its computed comparison digest can bind
            # this receipt by substructure ID.
            comparison_sha256 = legacy_comparison_by_id.get(substructure_id)
        if not isinstance(comparison_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", comparison_sha256
        ):
            return None
        if any(
            not isinstance(value, str) or not 0 < len(value) <= 256
            for value in [*observations, *raw_ids]
        ):
            return None
        if not set(observations).issubset(owner_observations):
            return None
        receipt = {
            "substructure_id": substructure_id,
            "semantic_sha256": semantic_sha256,
            "comparison_sha256": comparison_sha256,
            "source_observation_ids": observations,
            "source_substructure_ids": raw_ids,
        }
        previous = by_id.get(substructure_id)
        if previous is not None and previous != receipt:
            return None
        by_id[substructure_id] = receipt
    return [by_id[substructure_id] for substructure_id in sorted(by_id)]


def _semantic_substructure_union(
    inputs: list[dict], refs: list[dict]
) -> list[dict] | None:
    """Union full source graphs only when their lineage receipts match exactly."""

    structures_by_id: dict[str, dict] = {}
    expected_receipts: dict[str, dict] = {}
    for source in inputs:
        raw_structures = source.get("semantic_substructures")
        if raw_structures is None:
            continue
        structures = _validated_semantic_substructures(raw_structures)
        if structures is None:
            return None
        for structure in structures:
            payload = structure.to_dict()
            previous = structures_by_id.get(structure.substructure_id)
            if previous is not None and previous != payload:
                return None
            structures_by_id[structure.substructure_id] = payload
            expected_receipts[structure.substructure_id] = {
                "substructure_id": structure.substructure_id,
                "semantic_sha256": structure.semantic_sha256,
                "comparison_sha256": structure.comparison_sha256,
                "source_observation_ids": list(
                    structure.source_observation_ids
                ),
                "source_substructure_ids": list(
                    structure.source_substructure_ids
                ),
            }

    declared_receipts: dict[str, dict] = {}
    for ref in refs:
        receipts = _normalized_semantic_receipts(
            ref.get("semantic_substructure_refs"),
            source_observation_ids=ref.get("source_observation_ids"),
            legacy_comparison_by_id={
                substructure_id: receipt["comparison_sha256"]
                for substructure_id, receipt in expected_receipts.items()
            },
        )
        if receipts is None:
            return None
        for receipt in receipts:
            substructure_id = receipt["substructure_id"]
            previous = declared_receipts.get(substructure_id)
            if previous is not None and previous != receipt:
                return None
            declared_receipts[substructure_id] = receipt

    if expected_receipts != declared_receipts:
        return None
    return [
        structures_by_id[substructure_id]
        for substructure_id in sorted(structures_by_id)
    ]


def _derived_output_content(content: dict) -> dict:
    """Select output fields whose value determines derived evidence identity."""

    return derived_output_identity(
        content_type=content.get("type", "unknown"),
        text=content.get("text", ""),
        table_body=content.get("table_body", ""),
        table_caption=content.get("table_caption", []),
        table_footnote=content.get("table_footnote", []),
    )


def _mark_derived_evidence(
    content: dict,
    contributors: list[dict],
    transformation: str,
) -> None:
    """Bind changed content to ordered canonical inputs instead of one direct block.

    ``content`` must already contain its transformed output. The canonical parser
    artifact remains untouched; this helper only annotates the mutable legacy
    block consumed by BookRAG.
    """

    inputs = [content, *contributors]
    refs: list[dict] = []
    ref_keys: set[str] = set()
    for source in inputs:
        for ref in _source_evidence_refs(source):
            key = _ref_identity(ref)
            if key not in ref_keys:
                ref_keys.add(key)
                refs.append(ref)

    canonical_block_ids = _ordered_unique(
        [
            *(
                value
                for source in inputs
                for value in [
                    *(source.get("source_canonical_block_ids") or []),
                    source.get("canonical_block_id"),
                ]
            ),
            *(ref.get("canonical_block_id") for ref in refs),
        ]
    )
    canonical_revision_ids = _ordered_unique(
        [
            *(
                value
                for source in inputs
                for value in [
                    *(source.get("source_canonical_revision_ids") or []),
                    source.get("canonical_revision_id"),
                ]
            ),
            *(ref.get("canonical_revision_id") for ref in refs),
        ]
    )
    observation_ids = _ordered_unique(
        [
            *(
                value
                for source in inputs
                for value in [
                    *(source.get("source_observation_ids") or []),
                    source.get("observation_id"),
                ]
            ),
            *(
                value
                for ref in refs
                for value in [
                    *(ref.get("source_observation_ids") or []),
                    ref.get("observation_id"),
                ]
            ),
        ]
    )
    transformations = _ordered_unique(
        [
            *(
                item
                for source in inputs
                for item in (source.get("transformations") or [])
            ),
            transformation,
        ]
    )
    representation_ids = _ordered_unique(
        value
        for ref in refs
        for value in (ref.get("representation_ids") or [])
    )
    asset_ids = _ordered_unique(
        value for ref in refs for value in (ref.get("asset_ids") or [])
    )
    selected_representation_ids = _ordered_unique(
        value
        for ref in refs
        for value in (ref.get("selected_representation_ids") or [])
    )
    selected_asset_ids = _ordered_unique(
        value
        for ref in refs
        for value in (ref.get("selected_asset_ids") or [])
    )
    semantic_substructures = _semantic_substructure_union(inputs, refs)

    identity_payload = derived_identity_payload(
        output=_derived_output_content(content),
        source_revision_ids=canonical_revision_ids,
        # Some legacy/adaptor inputs predate canonical revision IDs.  Binding
        # their document/block/observation/reference identities prevents the
        # same transformed text on two source pages from colliding.
        source_document_version_ids=_ordered_unique(
            [
                *(source.get("document_version_id") for source in inputs),
                *(ref.get("document_version_id") for ref in refs),
            ]
        ),
        source_block_ids=canonical_block_ids,
        source_observation_ids=observation_ids,
        source_references=refs,
        transformations=transformations,
    )

    for field in _DIRECT_EVIDENCE_FIELDS:
        content[field] = None
    content["source_evidence_refs"] = refs
    content["source_canonical_block_ids"] = canonical_block_ids
    content["source_canonical_revision_ids"] = canonical_revision_ids
    content["source_observation_ids"] = observation_ids
    content["canonical_representation_ids"] = representation_ids
    content["canonical_asset_ids"] = asset_ids
    content["selected_representation_ids"] = selected_representation_ids
    content["selected_asset_ids"] = selected_asset_ids
    content["selected_representation_id"] = None
    content["selected_asset_id"] = None
    if semantic_substructures:
        content["semantic_substructures"] = semantic_substructures
    else:
        # No top-level graph is more accurate than a malformed or ambiguous
        # aggregate. Receipts remain available in ``refs`` for audit/debugging.
        content.pop("semantic_substructures", None)
    content["transformations"] = transformations
    content["derived_evidence_id"] = derived_evidence_id(identity_payload)


def is_likely_incomplete_paragraph(text: str) -> bool:
    """
    Determine if an English paragraph is likely incomplete (truncated due to page/column breaks).

    :param text: input text to check
    :return: bool, True if the paragraph is likely incomplete, False otherwise
    e.g. "He said, "This method is the best." -> False (complete)
    e.g. "The quick brown fox jumps over the lazy dog and" -> True (incomplete)
    """
    if not text:
        return False  # 空文本不是我们关心的“不完整段落”

    text = text.strip()

    # Rule 1: Filter out very short strings. They are likely standalone titles/captions, not paragraphs to be merged.
    if len(text.split()) < 5 or len(text) < 25:
        return False

    # --- From here, we look for clear signals of INCOMPLETENESS ---

    # Rule 2: Ending with a hyphen is a very strong signal of incompleteness (a word was split).
    if text.endswith("-"):
        return True

    # Handles cases like "said he," or "he said."
    cleaned_text = re.sub(r"['\"]+$", "", text)

    # Rule 3: Ending with a comma, colon, or semicolon is also a strong signal.
    if cleaned_text.endswith((",", ":", ";")):
        return True

    # Rule 4: Not ending with a standard terminal punctuation mark. This is the most common case.
    if not re.search(r"[.!?]$", cleaned_text):
        return True

    # Rule 5: Ending with a common connector word (even if mistakenly followed by a period).
    # e.g., "The quick brown fox jumps over the lazy dog and."
    incomplete_endings = {
        "and",
        "or",
        "but",
        "because",
        "although",
        "however",
        "if",
        "while",
        "when",
        "to",
        "for",
        "in",
        "of",
        "with",
        "on",
        "as",
        "at",
        "by",
        "from",
        "such",
        "the",
        "a",
        "an",
    }
    last_word_match = re.findall(r"\b\w+\b", cleaned_text)
    if last_word_match and last_word_match[-1].lower() in incomplete_endings:
        return True

    # If no "incomplete" signals were triggered, we assume it's complete.
    return False


def is_first_word_acronym(text: str) -> bool:
    """
    Check if the first word of the text is an acronym (all uppercase letters).
    :param text: The input text to check
    :return: True if the first word is an acronym, False otherwise
    e.g. "LLM is a powerful tool." -> True
    e.g. "This is a test." -> False
    """
    if not text:
        return False
    parts = text.split()
    if not parts:
        return False
    first_word = parts[0]
    # Check if the first word is all uppercase and longer than 1 character
    # This is a simple heuristic for acronyms
    return first_word.isupper() and len(first_word) > 1


def search_continuation_candidates(
    cur_content: dict,
    pdf_list: list[dict],
    cur_idx: int,
    end_idx: int,
    max_page_gap: int = 2,
) -> list[dict]:
    """
    Hint for searching continuation candidates in a PDF list.
    :param cur_content: current incomplete paragraph content
    :param pdf_list: all PDF paragraphs
    :param cur_idx: current paragraph index
    :param end_idx: end index for searching (inclusive)
    :param max_page_gap: maximum number of pages to skip backward
    :return: a group of candidate paragraphs (for splicing)
    """
    candidates = []
    cur_page_idx = cur_content["page_idx"]

    # For Rule #6: counts the number of "normal" capitalized paragraphs encountered.
    normal_uppercase_count = 0

    i = cur_idx + 1
    while i <= end_idx and i < len(pdf_list):
        next_content = pdf_list[i]
        next_page_idx = next_content.get("page_idx", -1)

        # Rule #1: If the next paragraph is too far away, stop searching.
        if next_page_idx > cur_page_idx + max_page_gap:
            # If the next paragraph is too far away, stop searching
            break

        # Rule #2: Skip non-body content like tables, images, or titles.
        if (
            next_content.get("type") in ["table", "image"]
            or next_content.get("text_level", -1) >= 0
        ):
            i += 1
            continue

        # Rule #3: If it's an equation, it might belong to the current paragraph. Add and continue.
        if next_content.get("type") == "equation":
            candidates.append(next_content)
            i += 1
            continue

        next_text = next_content.get("text", "").strip()

        # Skip empty or too short texts
        if not next_text or len(next_text) < 3:
            i += 1
            continue

        # Rule #4 & #5: If the next paragraph starts with a lowercase letter or an acronym,
        # it's very likely part of the current paragraph. Add and continue searching.
        if next_text[0].islower() or is_first_word_acronym(next_text):
            candidates.append(next_content)
            i += 1
            continue

        # Rule #6: Handle paragraphs starting with a "normal" capital letter.
        # This usually signals a new sentence, but could be a parsing error.
        # Strategy: include the first occurrence and when not found candidate, but stop at the second.
        if next_text[0].isupper():
            normal_uppercase_count += 1
            if normal_uppercase_count == 1 and len(candidates) == 0:
                candidates.append(next_content)
                i += 1
                continue
            else:
                break

        # If we reach here, it means the next paragraph is not a continuation
        break

    return candidates


def is_likely_content_label(text: str) -> bool:
    """
    A short, non-title text block that might be a label introducing the
    content immediately following it (e.g. "Answer to Check Yourself",
    "Solution", "Example 3" -- in any wording or language) rather than a
    self-contained caption or title that happens to also be short.

    is_likely_incomplete_paragraph's own Rule 1 excludes exactly this same
    short-text population from continuation-merging, because a genuine
    short caption/title must not swallow whatever unrelated content happens
    to follow it. This targets that same population from the other
    direction, for a different question ("does it introduce what follows",
    not "is it a sentence cut short") -- so the two passes never propose the
    same block for both judgments.

    Deliberately no keyword list: which labels actually get merged is
    decided by the LLM judgment this only proposes candidates for (the same
    judgment that already knows to reject a genuine caption/title). Staying
    permissive here and letting that real decision happen downstream
    generalizes across wording and language, instead of only the specific
    vocabulary anyone thought to list ahead of time.
    """
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    words = text.split()
    # The exact population Rule 1 (above) excludes.
    if not (len(words) < 5 or len(text) < 25):
        return False
    # A sentence ending in terminal punctuation is describing something
    # complete in itself, not introducing what follows -- language-agnostic
    # enough (shared across most punctuation-using scripts), unlike a
    # vocabulary list.
    if re.search(r"[.!?]$", text):
        return False
    # Excludes page numbers, bullets, and other non-linguistic short
    # fragments that could never be a label in any language -- a structural
    # check (does it contain a letter), not a vocabulary one.
    if not any(char.isalpha() for char in text):
        return False
    return True


def search_label_content_candidate(
    cur_content: dict,
    pdf_list: list[dict],
    cur_idx: int,
    end_idx: int,
    max_page_gap: int = 2,
) -> Optional[dict]:
    """
    Find the single block that immediately follows a content label, for the
    LLM to judge. Unlike search_continuation_candidates (tuned for "does
    this look like a mid-sentence continuation" -- lowercase start, acronym,
    bounded uppercase-restart count), a label can be followed by content in
    any case or form ("1(A) 2(A)..." starts with a digit, "The Second Law
    states..." starts with a normal capital) -- so this returns the very
    next non-empty, non-title, non-table, non-image block without those
    sentence-continuation-specific shape checks, still bounded by the same
    max_page_gap the rest of this module's search already uses (matching
    the verified cross-shard carry-over window from sharded_parsing.py).
    """
    i = cur_idx + 1
    cur_page_idx = cur_content["page_idx"]
    while i <= end_idx and i < len(pdf_list):
        next_content = pdf_list[i]
        next_page_idx = next_content.get("page_idx", -1)
        if next_page_idx > cur_page_idx + max_page_gap:
            break
        if (
            next_content.get("type") in ("table", "image")
            or next_content.get("text_level", -1) >= 0
        ):
            i += 1
            continue
        next_text = next_content.get("text", "").strip()
        if not next_text:
            i += 1
            continue
        return next_content
    return None


def get_json_str_text(text_candidates, max_tokens) -> List[str]:
    """
    Convert the text candidates into a list of JSON strings for LLM input.
    Each JSON string does not exceed the max_tokens limit.
    """
    pairs_lists = []
    number_of_tokens = 0
    number_of_pairs = 0
    str_list = []
    # Prepare the JSON structure for each text candidate
    columns = ["incomplete_text", "candidate_list"]
    for prev_content, candidates in text_candidates:
        incomplete_text = prev_content.get("text", "")
        candidate_list = [
            {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")}
            for c in candidates
        ]
        current_json = {
            "incomplete_text": incomplete_text,
            "candidate_list": candidate_list,
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)

        if (
            number_of_tokens + current_tokens < max_tokens
            and number_of_pairs < MAX_PAIRS_PER_JUDGMENT
        ):
            pairs_lists.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            # If adding this candidate exceeds the limit, save the current list and start a new one
            str_list.append((number_of_pairs, get_json_content(
                pairs_lists, selected_columns=columns)))
            pairs_lists = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if pairs_lists:
        # If there are remaining candidates, add them to the list
        str_list.append((number_of_pairs, get_json_content(
            pairs_lists, selected_columns=columns)))

    return str_list


def found_remove_text(text_candidate_pairs: list[tuple[dict, list[dict]]], error_json_str: str):
    """
    Find and remove text candidates from the list based on the error JSON string.
    :param text_candidate_pairs: List of text candidates to search in
    :param error_json_str: The JSON string that caused the error
    """
    columns = ["incomplete_text", "candidate_list"]
    for i, (prev_content, candidates) in enumerate(text_candidate_pairs):
        current_json = {
            "incomplete_text": prev_content.get("text", ""),
            "candidate_list": [
                {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")} for c in candidates
            ],
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"Found and removed text candidate at index {i}: {cur_json_str}")
            # If the error_json_str contains the current json_str, remove the pair
            text_candidate_pairs.pop(i)


def _judge_batches(
    json_str_list, llm: LLM, prompt_template: str, schema, *, log_label: str,
    format_key: str,
):
    """Judge independent batches concurrently, in batch order.

    Batches share no state and are merged in a later pass, so only the model
    calls overlap. Returns the ordered judgments and the batches that never
    produced a usable answer.
    """

    def judge(entry):
        number, json_str = entry
        prompt = prompt_template.format(**{format_key: json_str})
        log.info(f"Processing {number} {log_label} pairs with LLM judgment.")
        log.info(f"number of tokens in prompt: {num_tokens(prompt)}")
        last_error = None
        for _attempt in range(2):
            try:
                response = llm.get_json_completion(prompt=prompt, schema=schema)
            except Exception as error:
                log.error(f"LLM error: {error}")
                log.error(f"Prompt: {prompt}")
                last_error = error
                continue
            judgments = response.judgments
            if len(judgments) != number:
                log.error(
                    f"LLM response length mismatch: {len(judgments)} vs {number}"
                )
                last_error = ValueError(
                    f"expected {number} judgments, received {len(judgments)}"
                )
                continue
            return judgments
        raise last_error

    judgments = []
    unresolved = []
    for outcome in iter_bounded_ordered(
        json_str_list,
        judge,
        max_workers=getattr(llm, "max_workers", 1),
        max_in_flight=getattr(llm, "max_in_flight", 1),
        thread_name_prefix="pdf-refiner-judge",
    ):
        number, json_str = outcome.item
        if outcome.ok:
            judgments.extend(outcome.unwrap())
        else:
            log.error(
                f"Failed to process {number} {log_label} pairs with LLM judgment.")
            unresolved.append(json_str)
    return judgments, unresolved


def _llm_stitching_judge(
    text_candidate_pairs, llm: LLM, prompt_template: str, *, log_label: str
) -> int:
    """
    Shared retry/parse/merge core for both llm_text_judge (continuation of an
    incomplete sentence) and llm_label_judge (a short label merged with the
    content it introduces). The two differ only in prompt wording and in
    which blocks get proposed as pairs beforehand; the stitching decision
    format, retry behavior, and merge mechanics are identical.
    """
    if not text_candidate_pairs:
        log.info(f"No {log_label} candidates found for LLM judgment.")
        return 0
    json_str_list = get_json_str_text(
        text_candidate_pairs, llm.config.max_tokens -
        num_tokens(prompt_template) - 400
    )
    llm_infer_results, unresolved = _judge_batches(
        json_str_list,
        llm,
        prompt_template,
        StitchingJudgmentsResponse,
        log_label=log_label,
        format_key="json_text",
    )
    # Deferred: this pops from the list the batches were built from, which
    # cannot happen while results are still arriving.
    for json_str in unresolved:
        found_remove_text(text_candidate_pairs, json_str)

    if len(llm_infer_results) != len(text_candidate_pairs):
        log.error(
            f"LLM inference results length mismatch: {len(llm_infer_results)} vs {len(text_candidate_pairs)}"
        )
        return 0
    # Merge the results reversely
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        stitched_pdf_ids = llm_res.stitched_pdf_ids

        if len(stitched_pdf_ids) == 1 and stitched_pdf_ids[0] == -1:
            # If the text should not be merged, skip it
            i -= 1
            continue

        prev_content, candidates = text_candidate_pairs[i]
        selected_candidates = [
            c for c in candidates if c.get("pdf_id", -1) in stitched_pdf_ids
        ]
        if not selected_candidates:
            log.warning(
                f"No candidates found for merging with PDF IDs: {stitched_pdf_ids} at index {i}"
            )
            i -= 1
            continue
        # Merge the text content of selected candidates into prev_content
        merge_text_and_mark_invalid(prev_content, selected_candidates)
        merged_cnt += 1
        i -= 1

    return merged_cnt


def llm_text_judge(text_candidate_pairs, llm: LLM):
    """
    Judge the text stitching task using the LLM.
    """
    merged_cnt = _llm_stitching_judge(
        text_candidate_pairs, llm, TEXT_MERGE_PROMPT, log_label="text"
    )
    log.info(f"LLM text judgment completed. Merged {merged_cnt} texts.")


def llm_label_judge(label_candidate_pairs, llm: LLM):
    """
    Judge whether each short label introduces its single following
    candidate block, using the LLM.
    """
    merged_cnt = _llm_stitching_judge(
        label_candidate_pairs, llm, LABEL_MERGE_PROMPT, log_label="label"
    )
    log.info(f"LLM label judgment completed. Merged {merged_cnt} labels.")


def merge_text_and_mark_invalid(prev_content: dict, merged_list: list[dict]):
    """
    Merge the text content of merged_list into prev_content and mark the merged items as invalid.
    Not mark equation as invalid, since it is not a text content.
    :param prev_content: Previous content to merge into
    :param merged_list: List of content to merge
    """
    if not merged_list:
        return

    prev_text = prev_content.get("text", "")
    merged_text = [prev_text]
    for content in merged_list:
        merged_text.append(content.get("text", ""))
        if content.get("type") == "text":
            content["invalid"] = True

    merged_str = ""
    for text in merged_text:
        # if text is end with "-", remove the last character and not add a space
        if text.endswith("-"):
            merged_str += text[:-1]
        else:
            merged_str += text + " "
    prev_content["text"] = merged_str.strip()
    _mark_derived_evidence(
        prev_content,
        merged_list,
        "bookrag.text_merge.v1",
    )

    print(f"Merged text in page: {prev_content.get('page_idx', -1) + 1}")
    print(f"Index in Page: {prev_content['middle_json'].get("index", -1) + 1}")
    print(f"{prev_content['text']}")  # Print first 100 chars for debug


def text_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    incomplete_paragraphs = []
    # for循环的逻辑可以更清晰地组织
    for content in pdf_list:
        if (
            content is None
            or content.get("type") != "text"
            or content.get("text_level", -1) >= 0
        ):  # text_level >= 0
            continue

        # The logic is now direct: "if the paragraph is likely incomplete, add it."
        text = content.get("text", "")
        if is_likely_incomplete_paragraph(text):
            incomplete_paragraphs.append(content)

    if not incomplete_paragraphs:
        log.info("No incomplete paragraphs found.")
        return pdf_list

    log.info(f"Found {len(incomplete_paragraphs)} incomplete paragraphs.")
    llm_infer_candidates = []
    for i in range(len(incomplete_paragraphs)):
        cur_content = incomplete_paragraphs[i]
        start_idx = pdf_list.index(cur_content)
        end_serch_id = (
            pdf_list.index(incomplete_paragraphs[i + 1])
            if i + 1 < len(incomplete_paragraphs)
            else len(pdf_list) - 1
        )
        candidates = search_continuation_candidates(
            cur_content, pdf_list, start_idx, end_serch_id
        )
        if len(candidates) == 0:
            # If no candidates found, skip this paragraph
            continue
        llm_infer_candidates.append((cur_content, candidates))

    if not llm_infer_candidates:
        log.info("No candidates found for merging incomplete paragraphs.")
        return pdf_list
    log.info(
        f"Found {len(llm_infer_candidates)} candidates for merging incomplete paragraphs."
    )

    llm_text_judge(llm_infer_candidates, llm)
    log.info("LLM text judgment completed.")

    return pdf_list


def label_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    """
    Merge a short section label (e.g. "Answer to Check Yourself", "Solution")
    with the single content block that immediately follows it, when the
    parser split them into two separate blocks -- mirrors text_merger's
    structure exactly, using is_likely_content_label/search_label_content_candidate
    instead of is_likely_incomplete_paragraph/search_continuation_candidates.
    """
    labels = []
    for content in pdf_list:
        if (
            content is None
            or content.get("type") != "text"
            or content.get("text_level", -1) >= 0
            or content.get("invalid")
        ):
            continue
        text = content.get("text", "")
        if is_likely_content_label(text):
            labels.append(content)

    if not labels:
        log.info("No content labels found.")
        return pdf_list

    log.info(f"Found {len(labels)} content labels.")
    llm_infer_candidates = []
    for i in range(len(labels)):
        cur_content = labels[i]
        start_idx = pdf_list.index(cur_content)
        end_search_id = (
            pdf_list.index(labels[i + 1])
            if i + 1 < len(labels)
            else len(pdf_list) - 1
        )
        candidate = search_label_content_candidate(
            cur_content, pdf_list, start_idx, end_search_id
        )
        if candidate is None:
            continue
        llm_infer_candidates.append((cur_content, [candidate]))

    if not llm_infer_candidates:
        log.info("No candidates found for merging content labels.")
        return pdf_list
    log.info(
        f"Found {len(llm_infer_candidates)} candidates for merging content labels."
    )

    llm_label_judge(llm_infer_candidates, llm)

    return pdf_list


def get_table_col_count(table_html: str) -> int:
    """
    Return the maximum number of columns in a table HTML string.
    :param table_html: HTML string of the table
    :return: maximum number of columns in the table
    """
    soup = BeautifulSoup(table_html, "html.parser")
    max_cols = 0
    for row in soup.find_all("tr"):
        cols = 0
        for cell in row.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            cols += colspan
        if cols > max_cols:
            max_cols = cols
    return max_cols


def search_previous_table(
    cur_content: dict, pdf_list: list[dict], cur_idx: int
) -> Optional[dict]:
    """
    Search for the previous table content in the PDF list.
    :param cur_content: current incomplete paragraph content
    :param pdf_list: all PDF paragraphs
    :param cur_idx: current paragraph index
    :return: the previous table content if found, otherwise None
    """
    cur_page_idx = cur_content.get("page_idx", -1)
    if cur_page_idx <= 0:
        return None

    i = cur_idx - 1
    while i >= 0:
        prev_content = pdf_list[i]
        prev_page_idx = prev_content.get("page_idx", -1)
        prev_type = prev_content.get("type", "unknown")

        if prev_page_idx < 0 or prev_page_idx < cur_page_idx - 1:
            # Invalid page index or too far back, stop searching
            break

        if prev_type == "table":
            if prev_page_idx == cur_page_idx:
                # Same page, return None immediately
                return None
            cur_table_html = cur_content.get("table_body", "")
            prev_table_html = prev_content.get("table_body", "")
            if cur_table_html and prev_table_html:
                cur_col_count = get_table_col_count(cur_table_html)
                prev_col_count = get_table_col_count(prev_table_html)
                if cur_col_count == prev_col_count and cur_col_count > 0:
                    return prev_content  # Found matching table, return immediately

            break  # Only matching current table
        else:
            i -= 1

    # If no matching table found in previous page, return None
    return None


def merge_tables_and_mark_invalid(prev_content: dict, cur_content: dict):
    """
    Merge two table HTML contents and mark the current content as invalid.
    :param prev_content: Previous table content (to be merged into)
    :param cur_content: Current table content (to be merged)
    """
    prev_html = prev_content.get("table_body", "")
    cur_html = cur_content.get("table_body", "")
    if not prev_html or not cur_html:
        return

    prev_soup = BeautifulSoup(prev_html, "html.parser")
    cur_soup = BeautifulSoup(cur_html, "html.parser")

    prev_table = prev_soup.find("table")
    prev_tbody = prev_table.find("tbody") if prev_table else None

    cur_trs = cur_soup.find_all("tr")

    if prev_tbody:
        for tr in cur_trs:
            prev_tbody.append(tr)
    elif prev_table:
        for tr in cur_trs:
            prev_table.append(tr)
    else:
        # fallback: 直接拼接字符串
        prev_content["table_body"] += cur_html
        cur_content["invalid"] = True
        _mark_derived_evidence(
            prev_content,
            [cur_content],
            "bookrag.table_merge.v1",
        )
        return

    # 更新prev_content的table_body
    prev_content["table_body"] = str(prev_table)
    # 标记cur_content为无效
    cur_content["invalid"] = True
    _mark_derived_evidence(
        prev_content,
        [cur_content],
        "bookrag.table_merge.v1",
    )


def get_json_str_table_pairs(
    table_pairs: list[tuple[dict, dict]], max_tokens: int = 4096
):
    """
    return a list of JSON strings of table pairs for LLM input.
    each JSON string does not exceed the max_tokens limit.
    """
    json_pairs = []
    str_list = []
    number_of_tokens = 0
    number_of_pairs = 0
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for table_1, table_2 in table_pairs:
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)
        if (
            number_of_tokens + current_tokens < max_tokens
            and number_of_pairs < MAX_PAIRS_PER_JUDGMENT
        ):
            json_pairs.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            str_list.append((number_of_pairs, get_json_content(
                json_pairs, selected_columns=columns)))
            json_pairs = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if json_pairs:
        # If there are remaining pairs, add them to the list
        str_list.append((number_of_pairs, get_json_content(
            json_pairs, selected_columns=columns)))
    return str_list


def found_remove_table(cur_table_pairs: list[tuple[dict, dict]], error_json_str: str):
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for i, pairs in enumerate(cur_table_pairs):
        table_1, table_2 = pairs
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"Found and removed table pair at index {i}: {cur_json_str}")
            # If the error_json_str contains the current json_str, remove the pair
            cur_table_pairs.pop(i)


def llm_table_judger(table_pairs: list[tuple[dict, dict]], llm: LLM):
    """
    Use LLM to judge a list of table pairs whether they can be merged.
    :param table_pairs: a list of table pairs, each pair is a dict with keys
    """

    json_str_list = get_json_str_table_pairs(
        table_pairs, llm.config.max_tokens -
        num_tokens(TABLE_MERGE_PROMPT) - 500
    )
    llm_infer_results, unresolved = _judge_batches(
        json_str_list,
        llm,
        TABLE_MERGE_PROMPT,
        MergeJudgmentsResponse,
        log_label="table",
        format_key="json_pairs",
    )
    for json_str in unresolved:
        found_remove_table(table_pairs, json_str)

    if len(llm_infer_results) != len(table_pairs):
        log.error(
            f"LLM inference results length mismatch: {len(llm_infer_results)} vs {len(table_pairs)}"
        )
        return

    # Merge the results reversely
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        merged_id = llm_res.merged_id
        if merged_id == -1:
            # If the table should not be merged, skip it
            i -= 1
            continue
        prev_content, cur_content = table_pairs[i]
        if prev_content.get("pdf_id", -1) == merged_id:
            merge_tables_and_mark_invalid(prev_content, cur_content)
            merged_cnt += 1
        i -= 1

    log.info(f"LLM table judgment completed. Merged {merged_cnt} tables.")


def table_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    possible_tables = []
    for content in pdf_list:
        content_type = content.get("type", "unknown")
        if content_type == "table":
            # If the content is a table, we need to check if it should be merged or
            # if the table does not have footnotes or captions.
            footnote = content.get("table_footnote", [])
            caption = content.get("table_caption", [])
            if len(footnote) == 0 and len(caption) == 0:
                possible_tables.append(content)
    if not possible_tables:
        log.info("No tables found to merge.")
        return pdf_list
    log.info(f"Found {len(possible_tables)} tables to merge.")

    candidate_merge_table_pairs: list[tuple[dict, dict]] = []
    for i in range(len(possible_tables)):
        cur_content = possible_tables[i]
        cur_idx = pdf_list.index(cur_content)

        # Search for the previous table
        prev_table = search_previous_table(cur_content, pdf_list, cur_idx)
        if prev_table is not None:
            candidate_merge_table_pairs.append((prev_table, cur_content))

    if not candidate_merge_table_pairs:
        log.info("No candidate table pairs found for merging.")
        return pdf_list
    log.info(
        f"Found {len(candidate_merge_table_pairs)} candidate table pairs for merging."
    )

    llm_table_judger(candidate_merge_table_pairs, llm)
    log.info("LLM table judgment completed.")
    return pdf_list


# A pseudocode listing reaches us in pieces. MinerU labels only the part it is
# sure about -- a code block carrying sub_type "algorithm" -- and emits the
# surrounding step lines as ordinary text blocks, so "Algorithm 6.2: Binary
# Search" arrived as a 114-character fragment starting mid-sentence at
# 'PRINT "Element found at position"' with Steps 1-5 scattered beside it as
# prose. Two documents shred it two different ways (one splits on the lines
# carrying maths, the other on indentation depth), so the run is rebuilt from
# layout rather than from what the step labels look like: keying on "Step N:" or
# "N.M. [annotation]" would fit these two books and break on the next one.
#
# The pass only ever grows outward from a block MinerU itself called an
# algorithm. That is what keeps it away from real code: a Python or C++ listing
# has no algorithm anchor, arrives whole with its indentation already intact,
# and is never a candidate here.
#
# The pass runs after bind_canonical_evidence, which overwrites each block's
# "type" with the canonical projection's node type. An anchor therefore arrives
# typed "algorithm", not "code" carrying sub_type "algorithm" -- the shape it
# has in MinerU's own content list, and the shape the first version of this pass
# looked for, which made it a silent no-op on every real document. Both shapes
# are accepted so the pass cannot be broken again by where it sits in the order.
_LISTING_MERGEABLE_TYPES = frozenset({"text", "code", "algorithm"})
# A neighbour further from the run than this is a new block of content rather
# than the listing's next line. Expressed as a multiple of the run's own line
# pitch so it scales with the page, with a floor for a run too short to have
# established one.
_LISTING_GAP_MULTIPLE = 2.0
_LISTING_GAP_FLOOR = 14


def _listing_bbox(content: dict) -> Optional[tuple[float, float, float, float]]:
    bbox = content.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    return left, top, right, bottom


def _visual_line_count(content: dict) -> int:
    """How many lines on the page this block occupies, as MinerU counted them."""

    middle = content.get("middle_json")
    lines = middle.get("lines") if isinstance(middle, dict) else None
    return len(lines) if isinstance(lines, list) else 0


def _listing_reject_reason(content: Optional[dict]) -> Optional[str]:
    """Why this block is not part of a listing, or None if it could be."""

    if content is None or content.get("invalid"):
        return "already merged"
    if content.get("type") not in _LISTING_MERGEABLE_TYPES:
        return f"type {content.get('type')!r}"
    if content.get("text_level") is not None:
        # A levelled heading both ends the listing and stays its own node: it is
        # the caption ("Algorithm 6.2: Binary Search") or the next section
        # ("2.3 Format Convention of algorithm"), never a step.
        return "heading"
    if content.get("type") == "text" and _visual_line_count(content) > 1:
        # A step is one line on the page; a paragraph wraps. This is the only
        # thing separating the listing from the prose that follows it when both
        # sit at the same left margin, which is how one of the two documents
        # lays the page out.
        return f"wrapped over {_visual_line_count(content)} lines"
    if _listing_bbox(content) is None:
        return "no bbox"
    return None


def _is_listing_anchor(content: dict) -> bool:
    """A block MinerU itself called an algorithm, in either shape."""

    return (
        content.get("sub_type") == "algorithm" or content.get("type") == "algorithm"
    )


def _listing_continues(run: list[dict], candidate: dict) -> bool:
    """Whether a block sits close enough, and in the same column, to belong."""

    boxes = [box for box in (_listing_bbox(item) for item in run) if box is not None]
    box = _listing_bbox(candidate)
    if not boxes or box is None:
        return False
    left = min(item[0] for item in boxes)
    right = max(item[2] for item in boxes)
    # A neighbouring column is next to the listing, not inside it.
    if min(right, box[2]) - max(left, box[0]) <= 0:
        return False
    tops = sorted(boxes, key=lambda item: item[1])
    pitches = [
        max(0.0, later[1] - earlier[3]) for earlier, later in zip(tops, tops[1:])
    ]
    if box[1] >= tops[-1][3]:
        gap = box[1] - tops[-1][3]
    elif box[3] <= tops[0][1]:
        gap = tops[0][1] - box[3]
    else:
        gap = 0.0
    limit = (
        max(_LISTING_GAP_MULTIPLE * median(pitches), _LISTING_GAP_FLOOR)
        if pitches
        else float(_LISTING_GAP_FLOOR)
    )
    return gap <= limit


def _listing_run_indices(page: list[dict], anchor: int, taken: set[int]) -> list[int]:
    """Grow a listing outward from an anchor, in both directions, while it holds."""

    run = [anchor]
    for step in (-1, 1):
        index = anchor + step
        while 0 <= index < len(page) and index not in taken:
            candidate = page[index]
            if _listing_reject_reason(candidate) is not None:
                break
            if not _listing_continues([page[i] for i in run], candidate):
                break
            run.append(index)
            index += step
    return sorted(run)


# MinerU also splits a real program, one line per block, with no algorithm
# anchor anywhere to grow from: 016.0 SORTING.pdf emits the twelve lines of one
# insertion sort as twelve code blocks, so no node held the program and the
# answer correctly reported the document did not contain it. Earlier work here
# assumed a program always arrives whole. That held for one book and not for
# this one.
#
# A fragment is a code block holding a single line. A code block that carries
# its own line breaks is already a complete listing and is never merged -- which
# is what keeps a program and the console transcript printed under it, adjacent
# on the page, from collapsing into one node.
def _listing_body(content: dict) -> str:
    for key in ("text", "code_body"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return unwrap_presentation(value)
    return ""


def _is_code_fragment(content: Optional[dict]) -> bool:
    if content is None or content.get("invalid"):
        return False
    if content.get("type") not in {"code", "algorithm"}:
        return False
    if content.get("text_level") is not None:
        return False
    if _listing_bbox(content) is None:
        return False
    return "\n" not in _listing_body(content).strip()


# A fragment was cropped on its own, so its leading whitespace is gone and its
# indentation survives only in its left edge. In Python that indentation is the
# control flow, so it is rebuilt: the distinct left edges are clustered into
# levels and each level re-emitted as four spaces. On the insertion sort above,
# left edges 52/56, 152/155/156 and 255/256 become levels 0, 1 and 2.
_INDENT_CLUSTER_PX = 24
_INDENT_WIDTH = 4


def _indent_prefixes(boxes: list[tuple[float, float, float, float]]) -> list[str]:
    starts: list[float] = []
    for left in sorted({box[0] for box in boxes}):
        if not starts or left - starts[-1] > _INDENT_CLUSTER_PX:
            starts.append(left)
    prefixes: list[str] = []
    for box in boxes:
        level = 0
        for index, start in enumerate(starts):
            if box[0] >= start - _INDENT_CLUSTER_PX / 2:
                level = index
            else:
                break
        prefixes.append(" " * (_INDENT_WIDTH * level))
    return prefixes


def merge_listing_and_mark_invalid(
    anchor: dict, contributors: list[dict], *, reindent: bool = False
) -> None:
    """Rebuild one listing from its pieces, keeping every line break.

    Unlike ``merge_text_and_mark_invalid`` the pieces are joined with newlines
    and every contributor is invalidated, not only the text ones: a listing's
    line breaks and indentation are its meaning, and the code pieces here are
    listing lines rather than equations to be preserved separately.

    ``reindent`` rebuilds each piece's leading whitespace from its left edge, for
    a run of separately-cropped fragments that no longer carry any. It must stay
    off for pieces that kept their own indentation, which would otherwise be
    indented twice.
    """

    if not contributors:
        return
    ordered = sorted([anchor, *contributors], key=lambda item: item.get("_listing_seq", 0))
    prefixes: list[str] = [""] * len(ordered)
    if reindent:
        boxes = [_listing_bbox(content) for content in ordered]
        if all(box is not None for box in boxes):
            prefixes = _indent_prefixes([box for box in boxes if box is not None])
    parts: list[str] = []
    for content, prefix in zip(ordered, prefixes):
        raw = content.get("text")
        if not isinstance(raw, str) or not raw.strip():
            raw = content.get("code_body")
        if not isinstance(raw, str):
            continue
        piece = unwrap_presentation(raw).strip("\n")
        if piece.strip():
            parts.append(prefix + piece if prefix else piece)
    if not parts:
        return
    for content in contributors:
        content["invalid"] = True
    if reindent:
        # MinerU guesses a language per block, and on a single line it has too
        # little to go on: one Python program's twelve fragments were tagged
        # javascript, lua, txt and python. Only a language every piece agrees on
        # is claimed for the whole.
        languages = {content.get("code_language") for content in ordered}
        languages.discard(None)
        if len(languages) == 1:
            anchor["code_language"] = languages.pop()
        else:
            anchor.pop("code_language", None)

    anchor["text"] = "\n".join(parts)
    # The listing now lives in "text" like any other block's content. Leaving a
    # stale wrapped code_body behind would give the block two readings, and
    # every reader prefers "text", so the losing one is removed rather than
    # left as a trap.
    anchor.pop("code_body", None)
    boxes = [_listing_bbox(content) for content in ordered]
    boxes = [box for box in boxes if box is not None]
    if boxes:
        anchor["bbox"] = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    _mark_derived_evidence(anchor, contributors, "bookrag.listing_merge.v1")
    log.info(
        "Merged a %d-piece listing on page %d",
        len(ordered),
        anchor.get("page_idx", -1) + 1,
    )


def listing_merger(pdf_list: list[Optional[dict]]) -> list[Optional[dict]]:
    """Rejoin a listing that arrived split across blocks.

    Two passes, because a split listing arrives in two forms. An algorithm MinerU
    labelled is grown outward from that label, absorbing the step lines it emitted
    as prose. A program with no label anywhere is recovered from its own
    fragments: a run of code blocks each holding a single line.

    The labelled pass runs first, so a block it claims is not reconsidered.
    """

    pages: Dict[object, list[dict]] = {}
    for content in pdf_list:
        if content is None:
            continue
        pages.setdefault(content.get("page_idx"), []).append(content)

    for page in pages.values():
        for position, content in enumerate(page):
            content["_listing_seq"] = position
        taken: set[int] = set()
        for position, content in enumerate(page):
            if position in taken or not _is_listing_anchor(content):
                continue
            if _listing_reject_reason(content) is not None:
                continue
            run = _listing_run_indices(page, position, taken)
            taken.update(run)
            if len(run) < 2:
                continue
            merge_listing_and_mark_invalid(
                content, [page[i] for i in run if i != position]
            )

        position = 0
        while position < len(page):
            if position in taken or not _is_code_fragment(page[position]):
                position += 1
                continue
            run = [position]
            index = position + 1
            while (
                index < len(page)
                and index not in taken
                and _is_code_fragment(page[index])
                and _listing_continues([page[i] for i in run], page[index])
            ):
                run.append(index)
                index += 1
            taken.update(run)
            if len(run) >= 2:
                merge_listing_and_mark_invalid(
                    page[run[0]], [page[i] for i in run[1:]], reindent=True
                )
            position = max(index, position + 1)

        for content in page:
            content.pop("_listing_seq", None)
    return pdf_list


def dash_line_refiner(pdf_list: list[Optional[str]]):
    for content in pdf_list:
        if content is None:
            continue
        content_type = content.get("type", "unknown")
        if content_type == "text":
            # If the content is a paragraph, we need to check for dash line errors
            text = content.get("text", "")
            # If a space follows a dash line, it is likely a dash line error
            # We need to remove the space after the dash line
            refined_text = re.sub(r'-\s+', '-', text)
            if refined_text != text:
                content["text"] = refined_text
                _mark_derived_evidence(
                    content,
                    [],
                    "bookrag.dash_line_normalization.v1",
                )
    return pdf_list


def truncate_ocr_error_refiner(
    pdf_list: List[Optional[Dict]],
    window_size: int = 15,
    single_char_ratio_threshold: float = 0.9,
) -> List[Optional[Dict]]:
    """
    Identifies text with trailing OCR garbage and truncates it, preserving the valid part.

    This function uses a sliding window approach to find the starting point of
    OCR errors (like 't t t t') and cuts the text off just before it.

    Args:
        pdf_list: A list of dictionaries representing content blocks.
        window_size: The number of words to check in each sliding window.
        single_char_ratio_threshold: The ratio of single-character words within the
                                     window to trigger a truncation.

    Returns:
        The modified list with garbage text truncated.
    """
    if not pdf_list:
        return []

    for content in pdf_list:
        if content and isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
            if not text or not isinstance(text, str):
                continue

            words = text.strip().split()

            # Only process text long enough to analyze
            if len(words) < window_size:
                continue

            garbage_start_index = -1

            # --- Sliding Window Logic ---
            # Iterate through the text in windows to find where the garbage begins.
            for i in range(len(words) - window_size + 1):
                window = words[i: i + window_size]

                # Count how many words in the current window are single characters
                single_char_count = sum(1 for word in window if len(word) == 1)

                # If the ratio is high, we've found the start of the garbage
                if single_char_count / window_size >= single_char_ratio_threshold:
                    garbage_start_index = i
                    break

            # If we found a garbage section, truncate the text
            if garbage_start_index != -1:
                # Keep the text *before* the garbage started
                refined_text = " ".join(words[:garbage_start_index])
                # Add a log or print statement to know when truncation happens
                log.info(
                    f"Truncated OCR garbage. Original length: {len(text)}, New length: {len(refined_text)}")
                content["text"] = refined_text
                _mark_derived_evidence(
                    content,
                    [],
                    "bookrag.ocr_garbage_truncation.v1",
                )

    return pdf_list


def pdf_info_refiner(
    pdf_list: list[Optional[str]], llm: LLM, *, start_pdf_id: int = 1
) -> list[Optional[str]]:
    # Heuristic refiner for "-" error in OCR
    pdf_list = dash_line_refiner(pdf_list)
    # Heuristic refiner for OCR Error
    pdf_list = truncate_ocr_error_refiner(pdf_list)

    # we first enumerate the pdf_list to ensure each content has a unique index
    pdf_list = enumerate_pdf_list(pdf_list, start=start_pdf_id)

    # Rejoin a split pseudocode listing before any prose-oriented pass sees it.
    # Its step lines are single-line text blocks, which is exactly what
    # text_merger and label_merger hunt for -- run after them and the listing
    # has already been absorbed into the paragraph beside it, space-joined and
    # stripped of the line breaks that make it a listing.
    pdf_list = listing_merger(pdf_list)

    # Then we refine the PDF information by merging incomplete paragraphs and tables
    pdf_list = text_merger(pdf_list, llm)
    # A short section label ("Answer to Check Yourself", "Solution") split
    # from the content it introduces -- a different pattern from a sentence
    # cut short, so a separate pass after text_merger, before invalid items
    # from that pass are excluded below.
    pdf_list = label_merger(pdf_list, llm)
    pdf_list = table_merger(pdf_list, llm)

    # After merging, we need to re-enumerate the pdf_list. A sharded caller
    # passes the same start_pdf_id here as above: merging only ever removes
    # items (two continuation pieces become one), never adds them, so the
    # id sequence re-numbers within the same range, still globally unique.
    pdf_list = enumerate_pdf_list(pdf_list, start=start_pdf_id)
    log.info("PDF information refinement completed.")
    # Return the refined pdf_list
    return pdf_list


if __name__ == "__main__":

    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    print(
        is_likely_incomplete_paragraph('He said, "This method is the best."')
    )  # ✅ True

    # Example usage
    tmp_save_path = "/mnt/data/wangshu/mmrag/m3docrag/index/63a6b3f4-ebee-5024-b87b-84a9bcc26a63/vlm/8513db80c11ea439ab11eba406ec00d9_merged_content.json"
    # tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index/vlm/double_paper_merged_content.json"
    with open(tmp_save_path, "rb") as f:
        pdf_list = json.load(f)
    print(f"Loaded content from {tmp_save_path}")
    from Core.configs.llm_config import LLMConfig

    llm = LLM(LLMConfig())
    pdf_list = pdf_info_refiner(pdf_list, llm)
