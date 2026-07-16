"""Validate a model's structured reply however it chose to serialize it.

Ollama does not enforce a JSON schema as a decoding grammar for every model.
An unconstrained model answers the same schema in the shape it prefers: a
fenced ``json`` block, or YAML. Both carry the requested fields, so the reply
is recovered rather than discarded, and the schema still decides what is valid.
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_FENCE = re.compile(
    r"\A\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)\n?\s*```\s*\Z",
    re.DOTALL,
)


def strip_code_fence(payload: str) -> str:
    match = _FENCE.match(payload)
    return match.group("body") if match else payload


def parse_structured(payload: str, schema: type[SchemaT]) -> SchemaT:
    """Return ``payload`` validated against ``schema``."""

    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("model returned an empty structured response")
    candidates = [payload]
    unfenced = strip_code_fence(payload)
    if unfenced != payload:
        candidates.append(unfenced)
    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except ValidationError:
            continue
    import yaml

    for candidate in candidates:
        try:
            loaded = yaml.safe_load(candidate)
        except yaml.YAMLError:
            continue
        if isinstance(loaded, dict):
            return schema.model_validate(loaded)
    return schema.model_validate_json(candidates[-1])


__all__ = ["parse_structured", "strip_code_fence"]
