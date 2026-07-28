"""Can a smaller model adjudicate entity resolution, and how much faster?

The graph stage is 9,005 adjudications at ~2.9 s. Batching cuts the count;
a cheaper call cuts every one of them. ER selection is a constrained task --
pick a candidate id or -1 -- so it is the kind of step a small model may do
as well, which is worth knowing before building anything more elaborate.

Cases are built from the completed run's own graph: an entity plus the nine
others sharing the most tokens with it, which is roughly the shape retrieval
hands the adjudicator.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/rahul/DocARAG")

from Core.configs.llm_config import LLMConfig, indexing_llm_config
from Core.prompts.kg_prompt import ENTITY_RESOLUATION_PROMPT, ERExtractSel
from Core.provider.llm import LLM
from Core.utils.utils import truncate_description

GRAPH = Path(
    "/home/rahul/DocARAG/runs/doc_df198aeb160bb6240f5843b978d6fbcc/"
    "ee3d1f8d2bbfb5aef8ee657346835d15264b666647c4a757954f29e36f2de245/graph_data.json"
)
CASES = 40
CANDIDATES = 10
BASELINE = "qwen3.6:35b"
CHALLENGERS = ["gemma4:e2b", "gpt-oss:20b"]


def build_cases() -> list[dict]:
    nodes = json.loads(GRAPH.read_text())["graph"]["nodes"]
    nodes = [n for n in nodes if n.get("entity_name") and n.get("description")]
    by_token: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for token in set(re.findall(r"[a-z0-9]+", node["entity_name"].lower())):
            by_token[token].append(index)

    cases = []
    # Entities sharing a token with many others are exactly the ones retrieval
    # returns a crowded candidate list for, so they are the adjudicated cases.
    crowded = sorted(by_token.items(), key=lambda kv: -len(kv[1]))
    for _token, indexes in crowded:
        if len(indexes) <= CANDIDATES or len(cases) >= CASES:
            continue
        probe = nodes[indexes[0]]
        candidates = [nodes[i] for i in indexes[1 : CANDIDATES + 1]]
        cases.append({"probe": probe, "candidates": candidates})
        if len(cases) >= CASES:
            break
    return cases


def prompt_for(case: dict) -> str:
    def shrink(node: dict, index: int | None = None) -> dict:
        payload = {
            "entity_name": node["entity_name"],
            "entity_type": node["entity_type"],
            "description": truncate_description(node["description"]),
        }
        if index is not None:
            payload["id"] = index
        return payload

    input_data = {
        "new_entity": shrink(case["probe"]),
        "candidate_entities": [
            shrink(node, i) for i, node in enumerate(case["candidates"])
        ],
    }
    return ENTITY_RESOLUATION_PROMPT.format(
        input_json=json.dumps(input_data, indent=2, ensure_ascii=False)
    )


def run(model: str, prompts: list[str]) -> tuple[list[int], list[float]]:
    config = indexing_llm_config(
        LLMConfig(
            model_name=model,
            api_key="ollama",
            api_base="http://localhost:11434",
            backend="ollama",
            max_tokens=5000,
            temperature=0.1,
            num_ctx=16384,
        )
    )
    llm = LLM(config)
    decisions: list[int] = []
    durations: list[float] = []
    try:
        for prompt in prompts:
            start = time.perf_counter()
            try:
                res: ERExtractSel = llm.get_json_completion(
                    prompt=prompt, schema=ERExtractSel
                )
                decisions.append(int(res.select_id))
            except Exception:
                decisions.append(-999)  # a failure is not a "no match"
            durations.append(time.perf_counter() - start)
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return decisions, durations


def main() -> None:
    cases = build_cases()
    prompts = [prompt_for(case) for case in cases]
    print(f"cases: {len(cases)}  candidates each: {CANDIDATES}")
    print(f"prompt chars: median={statistics.median(len(p) for p in prompts):.0f}")
    print()

    base_decisions, base_durations = run(BASELINE, prompts)
    base_mean = statistics.mean(base_durations)
    print(
        f"{BASELINE:14} mean={base_mean:6.2f}s  median={statistics.median(base_durations):6.2f}s"
        f"  failures={base_decisions.count(-999)}"
    )
    print(f"{'':14} projected over 9,005 calls: {9005 * base_mean / 60:6.1f} min")
    print()

    for model in CHALLENGERS:
        decisions, durations = run(model, prompts)
        mean = statistics.mean(durations)
        agree = sum(a == b for a, b in zip(base_decisions, decisions))
        # A disagreement that merges where the baseline did not is the costly
        # direction: a wrong merge silently redirects every edge.
        false_merge = sum(
            b != -1 and a == -1 for a, b in zip(base_decisions, decisions)
        )
        missed_merge = sum(
            b == -1 and a != -1 for a, b in zip(base_decisions, decisions)
        )
        print(
            f"{model:14} mean={mean:6.2f}s  median={statistics.median(durations):6.2f}s"
            f"  failures={decisions.count(-999)}"
        )
        print(f"{'':14} projected over 9,005 calls: {9005 * mean / 60:6.1f} min"
              f"  ({base_mean / mean:.1f}x faster)")
        print(f"{'':14} agreement with {BASELINE}: {agree}/{len(cases)}"
              f"  ({100.0 * agree / len(cases):.0f}%)")
        print(f"{'':14} extra merges={false_merge}  missed merges={missed_merge}")
        print()


if __name__ == "__main__":
    main()
