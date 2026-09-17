"""Shared fail-closed responses for retrieval-grounded strategies."""

NO_RETRIEVED_EVIDENCE_ANSWER = (
    "I could not find usable evidence in the indexed document for this question, "
    "so no grounded answer is returned."
)

__all__ = ["NO_RETRIEVED_EVIDENCE_ANSWER"]
