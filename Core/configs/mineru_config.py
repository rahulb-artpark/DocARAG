from dataclasses import dataclass

# MinerU 3.x backend names. The 2.x names (vlm-transformers,
# vlm-sglang-engine, vlm-sglang-client) no longer exist.
#
# hybrid-engine is the default because it is the only backend measured to read
# handwriting correctly *and* keep the prose as text. On a cursive scan the
# pipeline backend scored 44.9% lexicon hit rate (printed documents score
# 93-98%), while MinerU 2.x's VLM backend read the same page at 97.4% but
# classified nearly every handwritten paragraph as a figure -- 39 words out of
# two pages. hybrid-engine scores 96.7% and emits 75 text blocks and 14 titles.
BACKENDS = (
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
    "vlm-http-client",
    "hybrid-http-client",
)

# effort applies only to the hybrid-* backends. "high" enables image/chart
# analysis; "medium" is faster and disables it.
EFFORTS = ("medium", "high")


@dataclass
class MinerU:
    backend: str
    method: str
    lang: str
    server_url: str = "http://127.0.0.1:30000"
    # Pages parsed per MinerU call. A whole large document parsed in one call
    # renders/OCRs/layout-analyzes every page at once, which is what drives
    # peak memory for big PDFs; bounding this bounds that peak instead.
    # Tune down for tighter memory budgets, up for fewer (faster) MinerU calls.
    shard_pages: int = 10
    effort: str = "medium"

    def __post_init__(self):
        if self.shard_pages < 1:
            raise ValueError(f"shard_pages must be positive: {self.shard_pages}")
        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Expected one of {', '.join(BACKENDS)}"
            )
        if self.effort not in EFFORTS:
            raise ValueError(
                f"Unsupported effort: {self.effort}. "
                f"Expected one of {', '.join(EFFORTS)}"
            )
        self.method = "auto" if self.backend == "pipeline" else "vlm"

    @property
    def is_hybrid(self) -> bool:
        return self.backend.startswith("hybrid-")
