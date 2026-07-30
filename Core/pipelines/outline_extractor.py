from typing import Optional, List

from Core.provider.llm import LLM
from Core.prompts.outline_prompt import OUTLINE_EXTRACTION_PROMPT, OutlineExtraction
from Core.utils.utils import get_json_content, num_tokens, enumerate_pdf_list
import logging

log = logging.getLogger(__name__)
import json
import math
import re

SELECT_COLS = ["pdf_id", "text", "page_idx", "height"]

# The outline pass is a second opinion on the parser's headings, and the
# parser's judgement stands unless the pass explicitly overrules it. Silence --
# a dropped chunk, an unparseable response, a pdf_id the model failed to echo
# back -- must never change a block's kind, in either direction.
#
# Which way the burden of proof runs depends on what the parser said:
#   parser labelled a heading -> stays a heading unless the pass returns -1
#   parser labelled body text -> stays body text unless the pass returns a level
#
# Only heading-shaped body blocks are offered for promotion. Sending every
# paragraph would cost chunks of prompt for candidates no reader would mistake
# for a heading, and give the model that many more chances to promote a
# sentence. A block missed by the parser looks like a heading to the eye: it is
# short, it names a topic, it starts like a name, and it is not a sentence.
#
# The shape rules below are each here because a looser version of this filter
# offered a block the model then promoted, and the promotion was wrong:
#   "Power of accomodation of eye decreases. ↓ Causes"  -> a sentence plus a
#       label, which headed a section holding the next two blocks
#   "Conditions for its formation →"  -> a label pointing at the list beneath
#       it, which the model made a top-level section owning half the chapter
_HEADING_SHAPE_MAX_WORDS = 12
_SENTENCE_TERMINATORS = (".", "?", "!")
# A period that ends a sentence is followed by a space; one inside "2.1" or
# "Fig.3" is not, so section numbers survive this.
_MID_SENTENCE_BREAK = re.compile(r"[.?!]\s")
# In handwritten notes an arrow points from a label at the content it labels.
# A heading names its section instead of pointing at it.
_ANNOTATION_ARROWS = ("→", "←", "↓", "↑", "⇒", "⇐", "=>", "->")


def heading_shaped(text: object) -> bool:
    """Whether a body block is worth offering to the outline pass as a heading."""

    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # "www.notesd" and "owrof th Sun at Sunvise" are OCR debris, not headings;
    # a heading starts with a capital or a section number.
    first = stripped[0]
    if not (first.isdigit() or (first.isalpha() and first.isupper())):
        return False
    if stripped.endswith(_SENTENCE_TERMINATORS):
        return False
    if _MID_SENTENCE_BREAK.search(stripped):
        return False
    if stripped.startswith(_ANNOTATION_ARROWS) or stripped.endswith(
        _ANNOTATION_ARROWS
    ):
        return False
    return 1 <= len(stripped.split()) <= _HEADING_SHAPE_MAX_WORDS


def outline_refine(outline_list: List[Optional[str]]) -> List[Optional[str]]:
    # 1.check the outline_list contains any one entry with text_level == 0
    # if not, assign the first entry with text_level == 0
    if not any(entry["text_level"] == 0 for entry in outline_list):
        if outline_list:
            outline_list[0]["text_level"] = 0
            log.info("Assigned text_level 0 to the first outline entry.")
        else:
            log.warning("Outline list is empty, no entries to assign text_level 0.")
            return outline_list

    # 2. check the parent_id of each entry
    # If the text_level is not 0 but the parent_id is 0
    # Assign the parent_id to the pdf_id of the entry with text_level 0

    # get the pdf_id of the entry with text_level 0
    pdf_id_level_0 = None
    for entry in outline_list:
        if entry["text_level"] == 0:
            pdf_id_level_0 = entry["pdf_id"]
            break

    for entry in outline_list:
        if entry["text_level"] != 0 and entry["parent_id"] == 0:
            entry["parent_id"] = pdf_id_level_0
            log.info(
                f"Assigned parent_id {pdf_id_level_0} to entry with pdf_id {entry['pdf_id']}"
            )
    return outline_list


def extract_pdf_outline(pdf_list: List[Optional[str]], llm: LLM) -> List[Optional[str]]:
    """Extract the outline from the PDF content."""

    pdf_length = len(pdf_list)
    title_list = []
    original_title_outline = []
    pdf_list = enumerate_pdf_list(pdf_list)

    stack = []
    for content in pdf_list:
        if "text" in content and "text_level" in content:
            level = content["text_level"]
            pdf_id = content.get("pdf_id", -1)

            # find the parent id
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None

            content_copy = content.copy()
            content_copy["parent_id"] = parent_id

            bbox = content.get("middle_json", {}).get("bbox")
            height_calculation_input = {"text": content.get("text", ""), "bbox": bbox}
            effective_height = calculate_effective_height(height_calculation_input)
            content_copy["height"] = effective_height

            original_title_outline.append(content_copy)
            stack.append((level, pdf_id))

            title_list.append(content_copy)

    json_format_title = get_json_content(title_list, selected_columns=SELECT_COLS)

    prompt = OUTLINE_EXTRACTION_PROMPT.format(json_title=json_format_title)
    log.info(f"number of token in prompt: {num_tokens(prompt)}")
    response: OutlineExtraction = llm.get_json_completion(prompt, OutlineExtraction)
    outline_list = []
    try:
        # parse the response
        outline = response.model_dump()
        if "outline" in outline:
            # check the length of the outline equal to the original title outline or not
            if len(outline["outline"]) != len(original_title_outline):
                log.warning(
                    f"Outline length mismatch: {len(outline['outline'])} vs {len(original_title_outline)}"
                )
            # merge the outline with the original title outline into outline_list
            for i, item in enumerate(outline["outline"]):
                if item["pdf_id"] != original_title_outline[i]["pdf_id"]:
                    log.warning(
                        f"PDF ID mismatch at index {i}: {item['pdf_id']} vs {original_title_outline[i]['pdf_id']}"
                    )
                tmp_outline = original_title_outline[i].copy()
                tmp_outline["text_level"] = item.get("level", -1)
                tmp_outline["parent_id"] = item.get("parent_id", -1)

                if tmp_outline["text_level"] != -1:
                    outline_list.append(tmp_outline)
                else:
                    # invalid entry should be skipped
                    log.info(
                        f"Skipping invalid outline entry at {tmp_outline['pdf_id']}"
                    )
        else:
            log.error("Outline not found in the response.")
            log.error(f"Response: {response}")
            log.error(f"Use original title outline: {original_title_outline}")
            outline_list = original_title_outline

    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON response: {e}")
        log.error(f"Response: {response}")
        log.error(f"Use original title outline: {original_title_outline}")
        outline_list = original_title_outline

    outline_list = outline_refine(outline_list=outline_list)

    # generate the outline scope of each section in the outlist
    max_level = 0
    for i, outline in enumerate(outline_list):
        if i != len(outline_list) - 1:
            end_id = outline_list[i + 1]["pdf_id"]
        else:
            end_id = pdf_length + 1
        outline["end_id"] = end_id
        max_level = max(max_level, outline["text_level"])

    log.info("Outline extraction completed.")
    log.info(f"Total {len(outline_list)} outline entries extracted.")
    log.info(f"Max level in outline: {max_level}")
    return outline_list


def calculate_effective_height(entry: dict) -> float:
    """
    Calculates the effective single-line height of a text block to better
    represent font size, accounting for multi-line text.

    Args:
        entry: A dictionary containing 'text' and 'bbox' keys.
               'bbox' is expected to be a list [x0, y0, x1, y1].

    Returns:
        A float representing the estimated single-line height.
    """
    bbox = entry.get("bbox")
    text = entry.get("text", "")

    if not bbox or len(bbox) != 4:
        return 0.0

    # 1. Calculate basic dimensions from bbox
    width = bbox[2] - bbox[0]
    total_height = bbox[3] - bbox[1]
    num_chars = len(text)

    # Handle edge cases to prevent division by zero or invalid calculations
    if width <= 0 or total_height <= 0 or num_chars == 0:
        return total_height if total_height > 0 else 0.0

    # 2. Heuristic for Estimating Line Count
    # This core heuristic is based on the idea that the total area occupied by
    # characters (num_chars * avg_char_area) is related to the bbox area (width * height).
    # We assume an average character's width is about half its height (a common typographic ratio).
    # So, avg_char_area ≈ (0.5 * line_height) * line_height = 0.5 * line_height^2
    # num_lines = total_height / line_height
    # After substitution and simplification, we get a formula to estimate the number of lines.

    # A calibration factor. Values between 0.4 and 0.6 often work well.
    # It accounts for the average character width-to-height ratio and spacing.
    ESTIMATION_FACTOR = 0.5

    # This ratio helps determine if the text is "cramped" enough to require multiple lines.
    # A higher value suggests more characters are packed into a tall, narrow space.
    line_estimation_ratio = (num_chars * total_height) / width

    # The number of lines is at least 1, and is related to the square root of the ratio.
    estimated_lines = round(
        max(1.0, math.sqrt(line_estimation_ratio * ESTIMATION_FACTOR))
    )

    # 3. Final Sanity Check using Aspect Ratio
    # If the box is extremely wide and short, it's almost certainly a single line,
    # regardless of the calculation above.
    aspect_ratio = width / total_height
    if aspect_ratio > 15:  # A very high aspect ratio strongly implies a single line
        estimated_lines = 1

    # 4. Calculate the effective height
    effective_height = total_height / estimated_lines

    return effective_height


def extract_pdf_outline_in_chunks(
    pdf_list: List[Optional[str]], llm: LLM
) -> tuple[List[Optional[str]], frozenset[int]]:
    """Extract the outline in chunks, with stateful context between chunks.

    Returns the hierarchy and the pdf_ids the pass explicitly rejected as
    non-headings.  The rejections are returned rather than dropped because the
    tree seeds its titles from the parser: a rejection that is merely absent
    from the hierarchy is indistinguishable from a heading the pass never
    looked at, and the block opens a section anyway.  That is how "↓ Correction"
    and "atmospheric refraction." came to head sections of a scanned chapter
    after this pass had correctly marked both invalid.
    """
    # 1. More precise token budget calculation (Your Point 1 & 4)
    prompt_template_tokens = num_tokens(OUTLINE_EXTRACTION_PROMPT.format(json_title=""))
    # Leave a 400-token buffer for the LLM's response generation and other overhead
    available_tokens_for_titles = llm.config.max_tokens - prompt_template_tokens - 500
    available_tokens_for_titles = min(2000, available_tokens_for_titles)
    log.info(
        f"LLM max_tokens: {llm.config.max_tokens}. Available for titles: {available_tokens_for_titles}"
    )

    # Pre-processing to get an initial, naive outline structure
    pdf_length = len(pdf_list)
    original_title_outline = []
    pdf_list_enumerated = enumerate_pdf_list(pdf_list)
    stack = []
    # pdf_ids the parser itself labelled headings. Everything else in the
    # candidate list is a promotion offer the parser did not make.
    parser_title_ids = set()
    for content in pdf_list_enumerated:
        if "text" not in content:
            continue
        level = content.get("text_level")
        is_parser_title = isinstance(level, int) and not isinstance(level, bool)
        if not is_parser_title:
            # Only prose is offered for promotion. A list item, a page header
            # and a footer are all title-capable once promoted, and none of
            # them is a section heading worth the risk of offering.
            if content.get("type") != "text":
                continue
            if not heading_shaped(content.get("text")):
                continue
        pdf_id = content.get("pdf_id", -1)

        if is_parser_title:
            while stack and stack[-1][0] >= level:
                stack.pop()
        parent_id = stack[-1][1] if stack else 0

        content_copy = content.copy()
        content_copy["parent_id"] = parent_id

        bbox = content.get("middle_json", {}).get("bbox")
        height_calculation_input = {"text": content.get("text", ""), "bbox": bbox}
        effective_height = calculate_effective_height(height_calculation_input)
        content_copy["height"] = effective_height

        original_title_outline.append(content_copy)
        # A candidate the parser did not level owns no depth, so it cannot
        # reparent the headings that follow it if the pass declines to promote
        # it.
        if is_parser_title:
            parser_title_ids.add(pdf_id)
            stack.append((level, pdf_id))

    # --- Main processing loop ---
    final_outline = []
    rejected_title_ids: set[int] = set()
    processed_titles_count = 0

    while processed_titles_count < len(original_title_outline):
        log.info(
            f"--- Processing new chunk starting at index {processed_titles_count} ---"
        )

        # 2. Smart Context Building (Your Point 2 & 3)
        context_titles = []
        if final_outline:  # Context is only built after the first chunk time
            # 2.1. High-level context from already processed outline
            level_0_title = [t for t in final_outline if t.get("text_level") == 0]
            level_1_titles = [t for t in final_outline if t.get("text_level") == 1]

            first_3_level_1 = level_1_titles[:3]
            last_5_level_1 = level_1_titles[-5:]

            # 2.2. Tail context: from the last processed item back to the nearest level 1
            tail_context_titles = []
            for item in reversed(final_outline):
                tail_context_titles.append(item)
                if item.get("text_level") == 1:
                    break
            tail_context_titles.reverse()  # Restore correct order
            if len(tail_context_titles) > 5:
                tail_context_titles = tail_context_titles[-5:]

            # 2.3. Combine and deduplicate context parts
            combined_context = (
                level_0_title + first_3_level_1 + last_5_level_1 + tail_context_titles
            )
            seen_ids = set()
            context_titles = [
                d
                for d in combined_context
                if d["pdf_id"] not in seen_ids and not seen_ids.add(d["pdf_id"])
            ]

        # 3. Dynamically select new titles for the current chunk
        new_titles_for_chunk = []
        remaining_titles = original_title_outline[processed_titles_count:]

        for new_title in remaining_titles:
            # Estimate token size of the potential prompt payload
            potential_payload = context_titles + new_titles_for_chunk + [new_title]
            json_str = get_json_content(potential_payload, SELECT_COLS)

            if num_tokens(json_str) > available_tokens_for_titles:
                # We can't add this new title, so the chunk is full.
                log.info(
                    f"Token limit reached. This chunk will process {len(new_titles_for_chunk)} new titles."
                )
                break

            new_titles_for_chunk.append(new_title)
            if len(new_titles_for_chunk) > 50:
                # At most 50 titles can be processed in a single chunk
                break

        if not new_titles_for_chunk and remaining_titles:
            log.error(
                f"A single title is too large to process, skipping. Title: {remaining_titles[0]}"
            )
            processed_titles_count += 1
            continue

        if not new_titles_for_chunk:  # All titles have been processed
            break

        # 4. Call LLM with the constructed prompt
        prompt_payload = context_titles + new_titles_for_chunk
        json_format_title = get_json_content(prompt_payload, SELECT_COLS)
        prompt = OUTLINE_EXTRACTION_PROMPT.format(json_title=json_format_title)
        log.info(f"Number of tokens in prompt: {num_tokens(prompt)}")

        try:
            response: OutlineExtraction = llm.get_json_completion(
                prompt, OutlineExtraction
            )
            llm_outline = response.model_dump().get("outline", [])

            if not llm_outline:
                raise ValueError("LLM response did not contain 'outline' field.")

            # 5. Incrementally merge results for NEW titles only
            new_titles_ids = {t["pdf_id"] for t in new_titles_for_chunk}
            newly_processed_items = []

            for llm_item in llm_outline:
                pdf_id = llm_item.get("pdf_id")
                if pdf_id in new_titles_ids:
                    original_item = next(
                        (t for t in original_title_outline if t["pdf_id"] == pdf_id),
                        None,
                    )
                    if original_item:
                        tmp_outline = original_item.copy()
                        tmp_outline["text_level"] = llm_item.get("level", -1)
                        tmp_outline["parent_id"] = llm_item.get("parent_id", -1)

                        if tmp_outline["text_level"] != -1:
                            newly_processed_items.append(tmp_outline)
                        elif pdf_id in parser_title_ids:
                            # An explicit verdict on a heading the parser found.
                            # Carried out of this function so the tree can stop
                            # seeding a section from it.
                            rejected_title_ids.add(pdf_id)
                            log.info(
                                f"Outline pass rejected parser heading pdf_id {pdf_id}"
                            )
                        else:
                            log.debug(
                                f"Outline pass declined to promote pdf_id {pdf_id}"
                            )

            final_outline.extend(newly_processed_items)

        except Exception as e:
            log.error(
                f"Error processing chunk: {e}. Falling back to original outline for this chunk."
            )
            # A failed chunk is silence, not a verdict: the parser's headings
            # keep their levels and nothing gets promoted off the back of a
            # response that never arrived.
            final_outline.extend(
                title
                for title in new_titles_for_chunk
                if title["pdf_id"] in parser_title_ids
            )

        # 6. Run outline_refine IN-LOOP to ensure context for the next iteration is valid (Your Point 5)
        final_outline = outline_refine(outline_list=final_outline)
        processed_titles_count += len(new_titles_for_chunk)

    log.info(
        f"--- All {len(original_title_outline)} candidates processed in chunks "
        f"({len(parser_title_ids)} parser headings, "
        f"{len(original_title_outline) - len(parser_title_ids)} promotion offers) ---"
    )

    # 7. Final post-processing for end_id calculation
    max_level = 0
    for i, outline in enumerate(final_outline):
        next_item_pdf_id = (
            final_outline[i + 1]["pdf_id"]
            if i < len(final_outline) - 1
            else pdf_length + 1
        )
        outline["end_id"] = next_item_pdf_id
        max_level = max(max_level, outline["text_level"])

    promoted = sum(
        1 for entry in final_outline if entry["pdf_id"] not in parser_title_ids
    )
    log.info("Outline extraction completed.")
    log.info(f"Total {len(final_outline)} outline entries extracted.")
    log.info(f"Max level in outline: {max_level}")
    log.info(
        f"Promoted {promoted} block(s) the parser missed; "
        f"rejected {len(rejected_title_ids)} parser heading(s)."
    )

    return final_outline, frozenset(rejected_title_ids)
