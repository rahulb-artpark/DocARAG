"""Does the Ollama server overlap generations, or run them one at a time?

Run before and after Scripts/enable_ollama_parallelism.sh. Identical requests
are used deliberately: if the server overlaps them, N concurrent finish in about
the time of one, and if it serialises them, in about N times that. Nothing about
the application can beat a server that is serialising -- refine_entities already
threads 8-way and gained nothing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL = "qwen3.6:35b"
BODY = {
    "model": MODEL,
    "prompt": "Count slowly from 1 to 60, one number per line.",
    "stream": False,
    "options": {"num_predict": 180, "temperature": 0.1},
    "think": False,
}


def call(_: int) -> float:
    start = time.perf_counter()
    requests.post("http://localhost:11434/api/generate", json=BODY, timeout=600)
    return time.perf_counter() - start


def main() -> None:
    call(0)  # load the model so the first timing is not a cold start

    start = time.perf_counter()
    call(0)
    solo = time.perf_counter() - start
    print(f"1 generation alone : {solo:5.2f}s")

    for workers in (2, 4, 8):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            start = time.perf_counter()
            list(pool.map(call, range(workers)))
            elapsed = time.perf_counter() - start
        print(
            f"{workers} concurrent      : {elapsed:5.2f}s"
            f"   (serial {workers * solo:5.2f}s, parallel ~{solo:.2f}s)"
            f"   throughput {workers * solo / elapsed:4.2f}x"
        )

    print()
    print("throughput ~1.0x at every level means the server is serialising.")


if __name__ == "__main__":
    main()
