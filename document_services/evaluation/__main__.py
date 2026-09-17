from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_services.evaluation.golden import evaluate_golden_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a DocARAG golden prediction envelope")
    parser.add_argument("manifest")
    parser.add_argument("predictions")
    parser.add_argument("--output")
    parser.add_argument("--no-verify-sources", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_golden_run(
        args.manifest, args.predictions, verify_sources=not args.no_verify_sources
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
