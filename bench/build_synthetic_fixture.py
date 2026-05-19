"""Build a synthetic N-doc JSONL fixture from the 10-post seed.

Usage:
    python build_synthetic_fixture.py --count 100000 --output synthetic_100000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures_seed" / "meltwater_posts.jsonl",
    )
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    base = [json.loads(line) for line in args.seed.read_text().splitlines() if line.strip()]
    if not base:
        raise SystemExit(f"empty seed: {args.seed}")

    n_groups = (args.count + len(base) - 1) // len(base)
    written = 0
    with args.output.open("w") as fh:
        for i in range(n_groups):
            for j, post in enumerate(base):
                if written >= args.count:
                    break
                content = post.get("content") or {}
                title = content.get("title", "") or ""
                body = content.get("body", "") or ""
                fh.write(
                    json.dumps({"document": f"{title}. {body} (variant {i}.{j})"})
                    + "\n"
                )
                written += 1
            if written >= args.count:
                break
    print(f"wrote {written} docs to {args.output}")


if __name__ == "__main__":
    main()
