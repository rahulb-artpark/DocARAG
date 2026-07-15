"""Curated, dependency-light application presets.

Presets are ordinary versioned domain objects, not hidden global state.  A
deployment can start with the supported default, serialize the resulting
policy for audit, and replace it with an explicitly configured policy without
changing routing or BookRAG code.
"""

from __future__ import annotations

from collections.abc import Mapping

from docarag.domain.base import ContractValidationError
from docarag.routing import AdapterRoutingProfile, PageRoutingPolicy


BALANCED_PDF_PRESET = "balanced-pdf"

DOCLING_ADAPTER = "docling-export"
MINERU_ADAPTER = "mineru-normalized"
PADDLEOCR_VL_ADAPTER = "paddleocr-vl-page"

_DEFAULT_LATENCY_MS = {
    DOCLING_ADAPTER: 1_000,
    MINERU_ADAPTER: 30_000,
    PADDLEOCR_VL_ADAPTER: 45_000,
}
_DEFAULT_COST_MICROUSD = {
    DOCLING_ADAPTER: 0,
    MINERU_ADAPTER: 0,
    PADDLEOCR_VL_ADAPTER: 0,
}


def _non_negative_overrides(
    defaults: Mapping[str, int],
    overrides: Mapping[str, int] | None,
    *,
    field_name: str,
) -> dict[str, int]:
    result = dict(defaults)
    if overrides is None:
        return result
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ContractValidationError(
            f"unknown {field_name} adapter(s): {', '.join(unknown)}"
        )
    for adapter_name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractValidationError(
                f"{field_name}[{adapter_name!r}] must be a non-negative integer"
            )
        result[adapter_name] = value
    return result


def balanced_pdf_routing_policy(
    *,
    estimated_latency_ms: Mapping[str, int] | None = None,
    estimated_cost_microusd: Mapping[str, int] | None = None,
    max_primary_latency_ms: int | None = None,
    max_total_latency_ms: int | None = None,
    max_primary_cost_microusd: int | None = None,
    max_total_cost_microusd: int | None = None,
) -> PageRoutingPolicy:
    """Return DocARAG's conservative default for the three shipped adapters.

    Native pages prefer Docling, technical pages prefer MinerU, and visual
    pages prefer PaddleOCR-VL.  Ambiguous pages plan one independent
    adjudicator.  The latency values are initial routing estimates, not an SLA;
    production deployments should replace them with measured p50 values for
    their page corpus and hardware.  Local adapters have zero monetary cost by
    default, while deployments using paid endpoints can provide explicit
    micro-USD estimates.
    """

    latency = _non_negative_overrides(
        _DEFAULT_LATENCY_MS,
        estimated_latency_ms,
        field_name="estimated_latency_ms",
    )
    cost = _non_negative_overrides(
        _DEFAULT_COST_MICROUSD,
        estimated_cost_microusd,
        field_name="estimated_cost_microusd",
    )
    profiles = (
        AdapterRoutingProfile(
            adapter_name=DOCLING_ADAPTER,
            native_text_rank=0,
            technical_rank=1,
            visual_rank=2,
            ambiguous_rank=1,
            estimated_latency_ms=latency[DOCLING_ADAPTER],
            estimated_cost_microusd=cost[DOCLING_ADAPTER],
            heavy=False,
        ),
        AdapterRoutingProfile(
            adapter_name=MINERU_ADAPTER,
            native_text_rank=1,
            technical_rank=0,
            visual_rank=1,
            ambiguous_rank=2,
            estimated_latency_ms=latency[MINERU_ADAPTER],
            estimated_cost_microusd=cost[MINERU_ADAPTER],
            heavy=True,
        ),
        AdapterRoutingProfile(
            adapter_name=PADDLEOCR_VL_ADAPTER,
            native_text_rank=2,
            technical_rank=2,
            visual_rank=0,
            ambiguous_rank=0,
            estimated_latency_ms=latency[PADDLEOCR_VL_ADAPTER],
            estimated_cost_microusd=cost[PADDLEOCR_VL_ADAPTER],
            heavy=True,
        ),
    )
    return PageRoutingPolicy(
        policy_name=BALANCED_PDF_PRESET,
        policy_version="1.0.0",
        adapter_profiles=profiles,
        max_primary_latency_ms=max_primary_latency_ms,
        max_total_latency_ms=max_total_latency_ms,
        max_primary_cost_microusd=max_primary_cost_microusd,
        max_total_cost_microusd=max_total_cost_microusd,
        allow_fallback=True,
        allow_adjudicator=True,
        prefer_non_heavy=True,
    )


def routing_policy_preset(name: str = BALANCED_PDF_PRESET, **kwargs: object) -> PageRoutingPolicy:
    """Resolve a named routing preset without import-time plugin discovery."""

    if name != BALANCED_PDF_PRESET:
        raise ContractValidationError(
            f"unsupported routing preset {name!r}; supported: {BALANCED_PDF_PRESET}"
        )
    return balanced_pdf_routing_policy(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "BALANCED_PDF_PRESET",
    "DOCLING_ADAPTER",
    "MINERU_ADAPTER",
    "PADDLEOCR_VL_ADAPTER",
    "balanced_pdf_routing_policy",
    "routing_policy_preset",
]
