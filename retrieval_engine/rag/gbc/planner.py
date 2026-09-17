import re

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal
from retrieval_engine.configuration.gbc_limits import (
    ABSOLUTE_MAX_FILTERS,
    ABSOLUTE_MAX_FILTER_VALUE_CHARS,
    ABSOLUTE_MAX_QUERY_CHARS,
    ABSOLUTE_MAX_REQUESTED_NODE_TYPES,
    ABSOLUTE_MAX_SUBQUESTION_CHARS,
    ABSOLUTE_MAX_SUBQUESTIONS,
)
from retrieval_engine.providers.language_model import LLM


class PlanningBudgetExceeded(ValueError):
    """An untrusted query plan exceeded a configured synchronous budget."""


MAX_PROMPT_SECTION_TITLES = 200
MAX_PROMPT_SECTION_TITLE_CHARS = 120


def bounded_section_titles(titles: object) -> tuple[str, ...]:
    """Return the document's distinct section titles, bounded for one prompt."""

    if not isinstance(titles, (list, tuple)):
        return ()
    seen: dict[str, None] = {}
    for title in titles:
        if not isinstance(title, str):
            continue
        cleaned = " ".join(title.split())[:MAX_PROMPT_SECTION_TITLE_CHARS]
        if cleaned:
            seen.setdefault(cleaned, None)
        if len(seen) >= MAX_PROMPT_SECTION_TITLES:
            break
    return tuple(seen)


def _validate_bounded_text(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > max_chars:
        raise PlanningBudgetExceeded(
            f"{field} contains {len(value)} characters, exceeding the "
            f"configured limit of {max_chars}"
        )
    return value


#: What a question can ask to be shown, as the four families a reader actually
#: distinguishes rather than as the eight node types the tree happens to use.
#:
#: Someone asking for "the image" means any picture, and whether the extractor
#: labelled that picture an image, a chart or a diagram is a fact about the
#: extractor, not about the question. Likewise a listing and an algorithm are
#: one thing to a reader, while a table, a list and a paragraph are the same
#: content in three formats. An equation is genuinely its own kind.
#:
#: Naming the families rather than the types also gives the planner four
#: choices instead of eight, and none of them require it to know how MinerU
#: classifies a figure.
RequestedNodeType = Literal["image", "table", "equation", "code"]

#: Each family expanded to the ``node_type`` payload values it covers. Node type
#: strings rather than the enum so this stays usable where retrieval reads
#: payloads back out of Qdrant.
REQUESTED_NODE_TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "image": ("image", "chart", "diagram"),
    "table": ("table", "text", "list"),
    "equation": ("equation",),
    "code": ("code", "algorithm"),
}

#: The family that is already what retrieval returns by default. Scoping to it
#: selects most of the document, so it cannot lift anything -- and issuing a
#: second leg over the same population would double those nodes' weight in a
#: rank-based fusion, biasing the answer toward prose for a question that only
#: asked to see a table. So the request is accepted and recorded, and no
#: scoped leg is built for it.
UNSCOPED_NODE_TYPE_FAMILIES = frozenset({"table"})


def requested_node_type_filter(families: object) -> tuple[str, ...]:
    """The node types a scoped retrieval leg should match, for these families.

    Empty when nothing was requested, and empty when everything requested is
    already the default population -- in both cases the caller issues no scoped
    leg and retrieval behaves exactly as it did before.
    """

    if not isinstance(families, (list, tuple)):
        return ()
    types: dict[str, None] = {}
    for family in families:
        if family in UNSCOPED_NODE_TYPE_FAMILIES:
            continue
        for node_type in REQUESTED_NODE_TYPE_FAMILIES.get(family, ()):
            types.setdefault(node_type, None)
    return tuple(types)


# Step 1 Output Model
class QueryTypeResult(BaseModel):
    query_type: Literal["simple", "complex", "global"]
    #: Asked for in the same call that classifies the query, because a simple
    #: plan makes no second call and a modality request is worth no extra
    #: round trip. Only used for a simple plan: a complex one carries it per
    #: sub-question and a global one says it through ``filters``.
    requested_node_types: Optional[List[RequestedNodeType]] = Field(
        default=None,
        max_length=ABSOLUTE_MAX_REQUESTED_NODE_TYPES,
    )


class LocalQueryTypeResult(QueryTypeResult):
    """Classification schema used while global classification is disabled."""

    query_type: Literal["simple", "complex"]


# Step 2 'complex' Output Model
class SubQuestion(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=ABSOLUTE_MAX_SUBQUESTION_CHARS,
    )
    type: Literal["retrieval", "synthesis"]
    requested_node_types: Optional[List[RequestedNodeType]] = Field(
        default=None,
        max_length=ABSOLUTE_MAX_REQUESTED_NODE_TYPES,
    )

    @model_validator(mode="after")
    def validate_question_text(self):
        if not self.question.strip():
            raise ValueError("sub-question must not contain only whitespace")
        if self.requested_node_types is not None:
            # A synthesis step does no lookup, so a modality request on one
            # would scope a retrieval that never happens.
            if self.type != "retrieval":
                raise ValueError(
                    "only a retrieval sub-question may request node types"
                )
            deduplicated = list(dict.fromkeys(self.requested_node_types))
            if deduplicated != self.requested_node_types:
                raise ValueError("requested_node_types must not repeat a type")
            if not deduplicated:
                raise ValueError(
                    "requested_node_types must be omitted rather than empty"
                )
        return self


class ComplexResult(BaseModel):
    sub_questions: List[SubQuestion] = Field(
        min_length=1,
        max_length=ABSOLUTE_MAX_SUBQUESTIONS,
    )


# Step 2 'global' Output Models (统一为Filter列表)
_PAGE_FILTER_VALUE = re.compile(r"([1-9][0-9]*)(?:\s*-\s*([1-9][0-9]*))?")


def parse_page_filter_value(value: object) -> tuple[int, int]:
    """Parse one-based page/range syntax without silently widening scope."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("page filter_value must be a page number or page range")
    match = _PAGE_FILTER_VALUE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "page filter_value must use one-based syntax such as '3' or '3-7'"
        )
    start_page = int(match.group(1))
    end_page = int(match.group(2) or match.group(1))
    if end_page < start_page:
        raise ValueError("page filter range end must be greater than or equal to start")
    return start_page, end_page


class Filter(BaseModel):
    filter_type: Literal[
        "document",
        "section",
        "title",
        "image",
        "table",
        "equation",
        "page",
    ]
    filter_value: Optional[str] = Field(
        default=None,
        max_length=ABSOLUTE_MAX_FILTER_VALUE_CHARS,
    )

    @model_validator(mode="after")
    def check_filter_value(self):
        if self.filter_type in {"document", "title", "image", "table", "equation"}:
            if self.filter_value is not None:
                raise ValueError(
                    "filter_value can only be set for a named 'section' or 'page', "
                    f"not for '{self.filter_type}'."
                )
            return self
        if self.filter_type == "section":
            if not isinstance(self.filter_value, str) or not self.filter_value.strip():
                raise ValueError("section filter_value must be a non-empty section name")
            return self
        if self.filter_type == "page":
            parse_page_filter_value(self.filter_value)
            return self
        # ``filter_type`` is a Literal, so this is a defensive future-proofing
        # branch rather than a reachable compatibility fallback.
        if self.filter_value is not None:
            raise ValueError(
                f"unsupported filter type: {self.filter_type}"
            )
        return self


class GlobalResult(BaseModel):
    filters: List[Filter] = Field(
        min_length=1,
        max_length=ABSOLUTE_MAX_FILTERS,
    )
    operation: Literal["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]


# Final combined result model (for returning to the user)
class PlanResult(BaseModel):
    query_type: Literal["simple", "complex", "global"]
    original_query: str = Field(
        min_length=1,
        max_length=ABSOLUTE_MAX_QUERY_CHARS,
    )
    sub_questions: Optional[List[SubQuestion]] = Field(
        default=None,
        max_length=ABSOLUTE_MAX_SUBQUESTIONS,
    )
    filters: Optional[List[Filter]] = Field(  # 原scope字段已改为filters
        default=None,
        max_length=ABSOLUTE_MAX_FILTERS,
    )
    operation: Optional[Literal["COUNT", "LIST", "SUMMARIZE", "ANALYZE"]] = None
    #: What kind of node the question asks to be shown, when it asks at all.
    #: Carried here for a simple query and on each sub-question for a complex
    #: one. A global query already expresses this through ``filters``, and two
    #: mechanisms that could disagree is worse than one.
    requested_node_types: Optional[List[RequestedNodeType]] = Field(
        default=None,
        max_length=ABSOLUTE_MAX_REQUESTED_NODE_TYPES,
    )

    @model_validator(mode="after")
    def check_fields_for_type(self):
        if not self.original_query.strip():
            raise ValueError("original_query must not contain only whitespace")
        if self.query_type == "complex" and not self.sub_questions:
            raise ValueError(
                "For 'complex' query_type, 'sub_questions' must be provided."
            )
        if self.query_type == "complex" and (
            self.filters is not None or self.operation is not None
        ):
            raise ValueError(
                "For 'complex' query_type, filters and operation must be null."
            )
        if self.query_type == "global" and (not self.filters or not self.operation):
            raise ValueError(
                "For 'global' query_type, 'filters' and 'operation' must be provided."
            )
        if self.query_type == "global" and self.sub_questions is not None:
            raise ValueError(
                "For 'global' query_type, sub_questions must be null."
            )
        # A complex plan carries the request per sub-question, because its
        # sub-questions ask for different things: "show me the image" and "what
        # is the equation" are one query and two modalities. A global plan says
        # it through ``filters`` already. Either one also setting it here would
        # be a second, disagreeing answer to the same question.
        if self.query_type in ("complex", "global") and (
            self.requested_node_types is not None
        ):
            raise ValueError(
                f"For '{self.query_type}' query_type, requested_node_types must "
                "be null; it belongs on each sub-question"
                if self.query_type == "complex"
                else f"For '{self.query_type}' query_type, requested_node_types "
                "must be null; use filters"
            )
        if self.query_type == "simple" and (
            self.sub_questions is not None
            or self.filters is not None
            or self.operation is not None
        ):
            raise ValueError(
                "For 'simple' query_type, all other specific fields must be null."
            )
        return self


# --- The Refactored Planner Class ---
class TaskPlanner:
    """
    A two-step Planner for user queries.
    Step 1: Classify the query type.
    Step 2: Process the query based on its type.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        enable_global_classification: bool = False,
        max_query_chars: int = 4_096,
        max_subquestions: int = 16,
        max_subquestion_chars: int = 1_024,
        max_filters: int = 8,
        max_filter_value_chars: int = 256,
    ):
        self.llm = llm
        self.enable_global_classification = enable_global_classification
        limits = {
            "max_query_chars": (max_query_chars, ABSOLUTE_MAX_QUERY_CHARS),
            "max_subquestions": (max_subquestions, ABSOLUTE_MAX_SUBQUESTIONS),
            "max_subquestion_chars": (
                max_subquestion_chars,
                ABSOLUTE_MAX_SUBQUESTION_CHARS,
            ),
            "max_filters": (max_filters, ABSOLUTE_MAX_FILTERS),
            "max_filter_value_chars": (
                max_filter_value_chars,
                ABSOLUTE_MAX_FILTER_VALUE_CHARS,
            ),
        }
        for name, (value, ceiling) in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            if value > ceiling:
                raise ValueError(f"{name} cannot exceed {ceiling}")
            setattr(self, name, value)

    def limits_trace(self) -> dict[str, int]:
        """Return the effective limits persisted with the query trace."""

        return {
            "max_query_chars": self.max_query_chars,
            "max_subquestions": self.max_subquestions,
            "max_subquestion_chars": self.max_subquestion_chars,
            "max_filters": self.max_filters,
            "max_filter_value_chars": self.max_filter_value_chars,
        }

    def validate_query(self, query: object) -> str:
        return _validate_bounded_text(
            query,
            field="query",
            max_chars=self.max_query_chars,
        )

    def validate_plan(self, plan: PlanResult) -> PlanResult:
        """Validate model-controlled fan-out before retrieval begins.

        Values are rejected rather than truncated.  Truncation would turn an
        apparently successful complex/global answer into one based on an
        undisclosed subset of the model's requested evidence.
        """

        if not isinstance(plan, PlanResult):
            raise TypeError("plan must be a PlanResult")
        # Re-run the structural schema from serialized data so callers cannot
        # bypass it with ``model_construct`` or a stale unsafe cache object.
        plan = PlanResult.model_validate(plan.model_dump())
        self.validate_query(plan.original_query)

        if plan.query_type == "complex":
            sub_questions = plan.sub_questions or []
            if len(sub_questions) > self.max_subquestions:
                raise PlanningBudgetExceeded(
                    "complex plan contains "
                    f"{len(sub_questions)} sub-questions, exceeding the "
                    f"configured limit of {self.max_subquestions}"
                )
            for index, sub_question in enumerate(sub_questions, start=1):
                _validate_bounded_text(
                    sub_question.question,
                    field=f"sub-question {index}",
                    max_chars=self.max_subquestion_chars,
                )

        if plan.query_type == "global":
            filters = plan.filters or []
            if len(filters) > self.max_filters:
                raise PlanningBudgetExceeded(
                    f"global plan contains {len(filters)} filters, exceeding "
                    f"the configured limit of {self.max_filters}"
                )
            for index, filter_spec in enumerate(filters, start=1):
                value = filter_spec.filter_value
                if value is not None and len(value) > self.max_filter_value_chars:
                    raise PlanningBudgetExceeded(
                        f"filter {index} contains {len(value)} characters, "
                        "exceeding the configured limit of "
                        f"{self.max_filter_value_chars}"
                    )
        return plan

    def _get_classification_prompt(self, query: str) -> str:
        categories = (
            'three categories: "simple", "complex", or "global"'
            if self.enable_global_classification
            else 'two categories: "simple" or "complex"'
        )
        global_category = ""
        if self.enable_global_classification:
            global_category = """3.  **global**: The question requires an **aggregation operation** (e.g., counting, listing, summarizing) over a set of items that are identified by a **clear structural filter**. The available filters are by layout type (such as `table`, `image`, `section`) or by a document range (such as `page` or `section`). This operation must be performed across all items that match the specified filter(s).
    - Example: "How many tables are in the document?" -> This is GLOBAL because the process is to **filter** for all items of type 'table' across the document, and then **aggregate** by counting them.
    - Example: "Summarize the discussion about 'data augmentation' in the 'Methodology' section." -> This is GLOBAL because the process is to first apply a **structural filter** (retrieve all content from the 'Methodology' section), and then perform an **aggregation** (summarize that content) with a focus on 'data augmentation'."""
        else:
            global_category = (
                "Questions requiring counting, listing, or summarizing across "
                "multiple locations must be classified as complex and decomposed "
                "into retrieval sub-questions. If all required information is in "
                "one contiguous location, classify as simple."
            )
        return f"""
You are an expert query analyzer. Your *only* task is to classify the user's question into one of {categories}. Respond only with the specified JSON object.

Category Definitions:
1.  **simple**: The question can be fully answered by retrieving information from a **SINGLE, contiguous location** in the document (e.g., one specific paragraph, one complete table, or one figure).
    - This includes questions that require reasoning or comparison, **as long as all the necessary data is present within that single retrieved location.**
    - Example: "What is the title of Figure 2?"
    - Example: "How do 5% of the Latinos see economic upward mobility for their children?" -> This is SIMPLE because the answer can be found by looking at a **single chart or paragraph**. The system just needs to find that one location, find the '5%' data point, and extract the corresponding label.

2.  **complex**: The question requires decomposition into multiple simple sub-questions, where **each sub-question must be answered by a separate retrieval action**.
    - It often contains a **nested or indirect constraint** that requires a preliminary step to resolve before the main question can be answered.
    - The key is that **no single location contains all the necessary information** to answer the original query in one go.
    - Example: "What is the color of the personality vector in the soft-labled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?" -> This is COMPLEX because it requires two **separate retrieval actions**: 1) A retrieval to find the vector with the 'highest score', and 2) a separate retrieval to find the 'color' associated with that vector.

{global_category}

{self._requested_node_types_rules()}

User Query: "{query}"
"""

    def _requested_node_types_rules(self) -> str:
        """The one instruction block both prompts share, written once.

        The hard case is not recognising a request; it is refusing one. In a
        technical document the words are also the subject matter: this corpus
        uses "image" 3913 times, at least 1622 of them in "image surface",
        "image space", "image height" -- optics, not a request to be shown a
        picture. A word-spotting rule fires on all of them, which is why this
        asks for the reader's *intent* and spells out the negative cases.
        """

        return f"""
### REQUESTED NODE TYPES ###

Set `requested_node_types` ONLY when the user is asking to be **shown a
particular kind of content**. Choose from: "image" (any picture, figure, chart,
plot or diagram), "table" (tabular or itemised data), "equation" (a formula),
"code" (a listing, file format or algorithm). At most
{ABSOLUTE_MAX_REQUESTED_NODE_TYPES}, no repeats. Omit the field entirely -- do
not send an empty list -- when the user is not asking to see a kind of content.

The words themselves are NOT the signal; the request is. These words are often
ordinary technical vocabulary of the subject, and in that use they must NOT set
this field.

SET IT:
- "show me the image of the Wolter Surface"        -> ["image"]
- "what does the Zernike surface look like?"       -> ["image"]
- "show me the figure and the equation for it"     -> ["image", "equation"]
- "give me the code for the INT Zernike format"    -> ["code"]

DO NOT SET IT:
- "what is the paraxial image plane?"              -> omit (optics term)
- "how is image height computed?"                  -> omit (optics term)
- "how do I configure the merit function?"         -> omit (no request)
- "what does the table say about tolerances?"      -> omit (asks the content,
  not to be shown the table)
- "how many images are in the document?"           -> omit (this is a global
  aggregation; it is expressed through filters)
"""

    def _get_complex_prompt(self, query: str) -> str:
        # Prompt for Step 2 - 增加了对输出字段的明确说明
        return f"""
You are a query decomposition expert. You have been given a "complex" question. Your task is to break it down into a series of simple, atomic sub-questions and classify each one by type.

**Crucial Instructions:**
1.  Each `retrieval` sub-question MUST be a direct information retrieval task that can be answered independently by looking up a specific fact, number, or value in the document.
2.  **`retrieval` sub-questions MUST NOT depend on the answer of another sub-question.** They should be parallelizable. All logic for combining their results must be placed in a final `synthesis` question.
3.  A `synthesis` question requires comparing, calculating, or combining the answers of the previous `retrieval` questions. It does **NOT** require a new lookup in the document.
4.  Return at most {self.max_subquestions} sub-questions, and keep each `question` at or below {self.max_subquestion_chars} characters. If the request cannot be represented within this budget, do not invent a partial decomposition.

You MUST provide your response in a JSON object with a single key 'sub_questions', which contains a list of objects. Each object must have a 'question' (string) and a 'type' (string: "retrieval" or "synthesis"), and may have 'requested_node_types' (a list of strings) on a `retrieval` sub-question only.

{self._requested_node_types_rules()}

--- EXAMPLE 1 (Correct Decomposition with Independent Lookups) ---
Complex Query: "What is the color of the personality vector in the soft-labled personality embedding matrix that with the highest Receptiviti score for User A2GBIFL43U1LKJ?"

Expected JSON Output:
{{
  "sub_questions": [
    {{
      "question": "What are all the Receptiviti scores for each personality vector for User A2GBIFL43U1LKJ?",
      "type": "retrieval"
    }},
    {{
      "question": "What is the mapping of personality vectors to their colors in the soft-labled personality embedding matrix?",
      "type": "retrieval"
    }},
    {{
      "question": "From the gathered scores, identify the personality vector with the highest score, and then find its corresponding color from the vector-to-color mapping.",
      "type": "synthesis"
    }}
  ]
}}
--- END EXAMPLE 1 ---

--- EXAMPLE 2 (Decomposition with retrieval and synthesis steps) ---
Complex Query: "According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?"

Expected JSON Output:
{{
  "sub_questions": [
    {{
      "question": "According to the report, what is the population of foreign born Latinos in the survey?",
      "type": "retrieval"
    }},
    {{
      "question": "According to the report, what is the population of Latinos interviewed by cellphone in the survey?",
      "type": "retrieval"
    }},
    {{
      "question": "Which of the two population counts is greater?",
      "type": "synthesis"
    }}
  ]
}}
--- END EXAMPLE 2 ---

Now, perform the decomposition for the following query.

Complex Query: "{query}"
"""

    def _get_global_prompt(self, query: str, section_titles: tuple[str, ...]) -> str:
        if section_titles:
            section_block = "\n".join(f"- {title}" for title in section_titles)
            section_rule = (
                "Use ONLY when the query explicitly refers to one of the "
                "sections listed under SECTIONS IN THIS DOCUMENT, and copy "
                "that title exactly. A topic the query merely asks about is "
                "NOT a section. If the query names no section from that list, "
                "do NOT emit a section filter."
            )
        else:
            section_block = "(this document has no detected sections)"
            section_rule = (
                "This document has no detected sections, so you MUST NOT emit "
                "a section filter."
            )
        return f"""
You are a highly specialized AI assistant. Your ONLY function is to analyze a "Global Query" and return a single, valid JSON object that specifies **both** the filtering steps and the final aggregation operation. You MUST NOT output any other text or explanation.

### SECTIONS IN THIS DOCUMENT ###

{section_block}

### INSTRUCTIONS & DEFINITIONS ###

1.  **Filters**: You MUST determine the list of `filters` to apply. Even if the filter is for the whole document (e.g., all tables), the `filters` list must be present.
    - `filter_type`: One of ["document", "section", "title", "image", "table", "equation", "page"].
        - `document`: Use for the whole document, such as a full-document summary. This does not mean a layout item count.
        - `section`: {section_rule}
        - `title`: Use for all structural headings when counting or listing chapters/sections.
        - `image`: Use for visual elements like figures, images, pictures, or plots.
        - `table`: Use for tabular data.
        - `equation`: Use for formulas or equations.
        - `page`: Use for specific page numbers or ranges.
    - `filter_value`: Required for "section" (e.g., a section title) and "page" (e.g., '3-10' or '5'). For every other filter type it MUST be null.

2.  **Operation**: Determine the final aggregation operation.
    - `operation`: One of ["COUNT", "LIST", "SUMMARIZE", "ANALYZE"].

3.  **Budget**: Return at most {self.max_filters} filters. Each `filter_value` must be at most {self.max_filter_value_chars} characters. Do not silently omit a required filter to fit this budget.

### EXAMPLES OF YOUR TASK ###

User: "How many figures are in this paper from Page 3 to Page 10?"
Assistant: {{"filters": [{{"filter_type": "page", "filter_value": "3-10"}}, {{"filter_type": "image"}}], "operation": "COUNT"}}

User: "Summarize the discussion about 'data augmentation' in the 'Methodology' section."
Assistant: {{"filters": [{{"filter_type": "section", "filter_value": "Methodology"}}], "operation": "SUMMARIZE"}}
(Only correct when "Methodology" is listed above. If it is not listed, emit
{{"filters": [{{"filter_type": "document"}}], "operation": "SUMMARIZE"}} instead.)

User: "Analyze what drives the margin gap."
Assistant: {{"filters": [{{"filter_type": "document"}}], "operation": "ANALYZE"}}
(The query names a topic, not a section, so no section filter is emitted.)

User: "How many chapter or section headings are in the document?"
Assistant: {{"filters": [{{"filter_type": "title"}}], "operation": "COUNT"}}

User: "Summarize the whole document."
Assistant: {{"filters": [{{"filter_type": "document"}}], "operation": "SUMMARIZE"}}

### YOUR CURRENT TASK ###

User: "{query}"
Assistant:
"""

    def _classify_query_type(self, query: str) -> QueryTypeResult:
        prompt = self._get_classification_prompt(query)
        schema = (
            QueryTypeResult
            if self.enable_global_classification
            else LocalQueryTypeResult
        )
        result = self.llm.get_json_completion(prompt, schema=schema)
        # Validate again for providers that return an already constructed model.
        return schema.model_validate(result.model_dump())

    def _process_complex_query(self, query: str) -> List[SubQuestion]:
        prompt = self._get_complex_prompt(query)
        result = self.llm.get_json_completion(prompt, schema=ComplexResult)
        return result.sub_questions

    def _process_global_query(
        self, query: str, section_titles: tuple[str, ...]
    ) -> GlobalResult:
        prompt = self._get_global_prompt(query, section_titles)
        result = self.llm.get_json_completion(prompt, schema=GlobalResult)
        return result

    def analyze(
        self, query: str, *, section_titles: object = ()
    ) -> PlanResult:
        """Plan one query against the sections the document actually has.

        Without the real titles a global plan invents a plausible section name
        from the query's topic, and a section filter that matches no heading
        silently reduces the whole retrieval scope to nothing.
        """

        self.validate_query(query)
        known_sections = bounded_section_titles(section_titles)
        classification = self._classify_query_type(query)
        query_type = classification.query_type

        if query_type == "simple":
            plan = PlanResult(
                query_type=query_type,
                original_query=query,
                # A complex plan carries this per sub-question and a global one
                # through its filters, so the classifier's answer is only ever
                # read here. Dropping it elsewhere keeps one source of truth.
                requested_node_types=classification.requested_node_types,
            )

        elif query_type == "complex":
            sub_questions = self._process_complex_query(query)
            plan = PlanResult(
                query_type=query_type, original_query=query, sub_questions=sub_questions
            )

        elif query_type == "global":
            global_result = self._process_global_query(query, known_sections)
            plan = PlanResult(
                query_type=query_type,
                original_query=query,
                filters=global_result.filters,
                operation=global_result.operation,
            )
        else:  # pragma: no cover - the Literal schema makes this unreachable.
            raise ValueError(f"unsupported query type: {query_type}")

        return self.validate_plan(plan)
