"""Process-safety ceilings shared by BookRAG config and runtime validation.

Normal limits live in :class:`GBCRAGConfig`.  These ceilings prevent a config
from turning one synchronous query into an effectively unbounded request while
keeping the configuration layer independent from runtime/provider imports.
"""

ABSOLUTE_MAX_QUERY_CHARS = 16_384
ABSOLUTE_MAX_SUBQUESTIONS = 32
ABSOLUTE_MAX_SUBQUESTION_CHARS = 4_096
ABSOLUTE_MAX_FILTERS = 16
ABSOLUTE_MAX_FILTER_VALUE_CHARS = 1_024
#: A question can ask for a few kinds of node at once -- "the image and the
#: equation" -- but each requested kind costs a retrieval leg, so the count is
#: bounded like every other model-controlled fan-out here.
ABSOLUTE_MAX_REQUESTED_NODE_TYPES = 4

