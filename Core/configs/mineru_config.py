from dataclasses import dataclass

@dataclass
class MinerU:
    backend: str
    method: str
    lang:str
    server_url: str = "http://127.0.0.1:30000"
    # Pages parsed per MinerU call. A whole large document parsed in one call
    # renders/OCRs/layout-analyzes every page at once, which is what drives
    # peak memory for big PDFs; bounding this bounds that peak instead.
    # Tune down for tighter memory budgets, up for fewer (faster) MinerU calls.
    shard_pages: int = 10

    def __post_init__(self):
        if self.shard_pages < 1:
            raise ValueError(f"shard_pages must be positive: {self.shard_pages}")
        if self.backend not in [
            "vlm-sglang-client",
            "vlm-transformers",
            "vlm-sglang-engine",
            "pipeline",
        ]:
            raise ValueError(f"Unsupported backend: {self.backend}")
        self.method = "auto" if self.backend == "pipeline" else "vlm"
