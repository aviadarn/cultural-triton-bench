"""Run a benchmark sweep over multiple configs and aggregate results.

Each config in the YAML file produces one bench row. CPU baseline runs once
on the largest corpus (or as configured) and is reused as the parity anchor
for every Triton config. Sweep writes:

- ``<output>.json`` — raw report with one entry per config
- ``<output>.csv`` — flat table for spreadsheet review

Usage:

    python sweep.py \\
        --configs configs/sweep_default.yml \\
        --fixture /path/to/cmi_documents.parquet \\
        --output sweep_report

YAML schema:

    cpu_baseline:
      limit: 50000              # docs in the parity anchor run
      embedding_batch_size: 256
      min_topic_size: null      # null = use agent heuristic

    triton_runs:
      - name: ensemble-fp16-batch128
        triton_model: ensemble-topics
        limit: 50000
        max_length: 256
      - name: ensemble-fp16-batch64
        triton_model: ensemble-topics
        limit: 50000
        max_length: 256
      - name: ensemble-fp16-100k
        triton_model: ensemble-topics
        limit: 100000
        max_length: 256

Notes:
- Parity NMI is computed *only* when a Triton run's ``limit`` matches the
  CPU baseline's ``limit``. Otherwise the corpora differ and NMI would be
  meaningless — the report marks parity as ``null``.
- The CPU baseline runs once and is cached for the rest of the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from bench import _load_fixture, parity_nmi, run_cpu, run_triton


def _cpu_row(timings, labels: np.ndarray, num_docs: int) -> Dict[str, Any]:
    return {
        "wall_seconds": sum(t.seconds for t in timings),
        "timings": [t.to_dict() for t in timings],
        "num_docs": num_docs,
        "topic_count": int(
            len(set(labels.tolist())) - (1 if -1 in labels else 0)
        ),
        "outlier_share": float((labels == -1).mean()),
    }


def _triton_row(timings, labels: np.ndarray, num_docs: int) -> Dict[str, Any]:
    return {
        "wall_seconds": sum(t.seconds for t in timings),
        "timings": [t.to_dict() for t in timings],
        "num_docs": num_docs,
        "topic_count": int(
            len(set(labels.tolist())) - (1 if -1 in labels else 0)
        ),
        "outlier_share": float((labels == -1).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--triton-url", default="http://localhost:8000")
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "model_repository"
        / "minilm-onnx"
        / "tokenizer",
    )
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p.add_argument("--output", type=Path, default=Path("sweep_report"))
    p.add_argument("--parity-threshold", type=float, default=0.95)
    p.add_argument("--skip-cpu", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.configs.read_text())
    cpu_cfg = cfg.get("cpu_baseline", {}) or {}
    triton_cfgs: List[Dict[str, Any]] = cfg.get("triton_runs", []) or []

    report: Dict[str, Any] = {
        "fixture": str(args.fixture),
        "configs": str(args.configs),
        "cpu": None,
        "triton": [],
    }

    cpu_labels: np.ndarray | None = None
    cpu_limit: int | None = None
    if not args.skip_cpu and cpu_cfg:
        cpu_limit = cpu_cfg.get("limit")
        docs = _load_fixture(args.fixture, cpu_limit)
        print(f"\n── CPU baseline ({len(docs):,} docs) ──")
        _, cpu_labels, timings = run_cpu(
            docs,
            args.model_name,
            embedding_batch_size=cpu_cfg.get("embedding_batch_size", 256),
            min_topic_size=cpu_cfg.get("min_topic_size"),
        )
        report["cpu"] = _cpu_row(timings, cpu_labels, len(docs))
        report["cpu"]["config"] = cpu_cfg
        print(json.dumps(report["cpu"], indent=2, default=str))

    for tcfg in triton_cfgs:
        name = tcfg.get("name", tcfg.get("triton_model", "unnamed"))
        limit = tcfg.get("limit")
        docs = _load_fixture(args.fixture, limit)
        print(f"\n── Triton run {name!r} ({len(docs):,} docs) ──")
        labels, timings = run_triton(
            docs,
            args.triton_url,
            args.model_name,
            args.tokenizer_dir,
            triton_model=tcfg.get("triton_model", "ensemble-topics"),
            max_length=tcfg.get("max_length", 256),
        )
        row = _triton_row(timings, labels, len(docs))
        row["name"] = name
        row["config"] = tcfg
        if cpu_labels is not None and limit == cpu_limit:
            nmi = parity_nmi(cpu_labels, labels)
            row["parity_nmi"] = nmi
            row["parity_passed"] = bool(nmi >= args.parity_threshold) if nmi == nmi else False
        else:
            row["parity_nmi"] = None
            row["parity_passed"] = None
        if report["cpu"]:
            cpu_wall = report["cpu"]["wall_seconds"]
            row["speedup_vs_cpu"] = (
                cpu_wall / row["wall_seconds"] if row["wall_seconds"] else None
            )
        else:
            row["speedup_vs_cpu"] = None
        report["triton"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "timings"}, indent=2))

    json_path = args.output.with_suffix(".json")
    csv_path = args.output.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, default=str))

    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "name",
                "num_docs",
                "triton_model",
                "wall_seconds",
                "speedup_vs_cpu",
                "topic_count",
                "outlier_share",
                "parity_nmi",
                "parity_passed",
            ]
        )
        if report["cpu"]:
            w.writerow(
                [
                    "cpu_baseline",
                    report["cpu"]["num_docs"],
                    "—",
                    round(report["cpu"]["wall_seconds"], 2),
                    "—",
                    report["cpu"]["topic_count"],
                    round(report["cpu"]["outlier_share"], 3),
                    "—",
                    "—",
                ]
            )
        for row in report["triton"]:
            w.writerow(
                [
                    row["name"],
                    row["num_docs"],
                    row["config"].get("triton_model", "ensemble-topics"),
                    round(row["wall_seconds"], 2),
                    round(row["speedup_vs_cpu"], 2) if row["speedup_vs_cpu"] else "—",
                    row["topic_count"],
                    round(row["outlier_share"], 3),
                    round(row["parity_nmi"], 4) if row["parity_nmi"] else "—",
                    row["parity_passed"] if row["parity_passed"] is not None else "—",
                ]
            )

    print(f"\nWrote {json_path} and {csv_path}")
    any_parity_miss = any(
        r.get("parity_passed") is False for r in report["triton"]
    )
    return 2 if any_parity_miss else 0


if __name__ == "__main__":
    sys.exit(main())
