"""CPU baseline vs Triton GPU ensemble — side-by-side bench + NMI parity.

Runs the full cultural-agent inner loop (embed + UMAP + HDBSCAN) twice on the
same fixture corpus: once with the current CPU stack, once against a running
Triton server. Emits a JSON report with per-stage timings and the topic-label
NMI between the two runs (parity gate = NMI >= 0.95).

Usage:
    python bench.py \
        --fixture /path/to/cmi_documents.parquet \
        --triton-url http://localhost:8000 \
        --output bench_report.json

Notes:
- The Triton path uses the `ensemble-topics` model so one HTTP call returns
  topic labels — this is the shape the worker will use in Phase 2.
- The CPU path mirrors `agent.py:999-1034` exactly (same hyperparameters,
  same models). If those drift, update both sides together.
- NMI is computed on overlapping labels only; outliers (label = -1) are
  excluded from the parity calculation since cuML/sklearn HDBSCAN classify
  borderline points differently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class StageTiming:
    name: str
    seconds: float
    docs: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["docs_per_sec"] = self.docs / self.seconds if self.seconds else 0.0
        return d


def _load_fixture(path: Path, limit: int | None) -> List[str]:
    """Load a parquet/jsonl fixture of cultural-agent input docs."""
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        col = (
            "document"
            if "document" in df.columns
            else "text"
            if "text" in df.columns
            else df.columns[0]
        )
        docs = df[col].dropna().astype(str).tolist()
    elif path.suffix in {".jsonl", ".ndjson"}:
        docs = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("document") or obj.get("text") or ""
                if text:
                    docs.append(text)
    else:
        raise ValueError(f"Unsupported fixture format: {path.suffix}")
    if limit is not None:
        docs = docs[:limit]
    if not docs:
        raise ValueError(f"Empty fixture at {path}")
    return docs


# ── CPU baseline ────────────────────────────────────────────────────────────


def run_cpu(
    docs: List[str],
    model_name: str,
    *,
    embedding_batch_size: int = 256,
    min_topic_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[StageTiming]]:
    """Mirror of agent.py:999-1034 CPU path. Hyperparameters overridable for sweeps."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    num_docs = len(docs)
    if min_topic_size is None:
        min_topic_size = max(10, min(int(num_docs * 0.015), 120))

    timings: List[StageTiming] = []

    t0 = time.perf_counter()
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        docs,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    timings.append(StageTiming("embedding", time.perf_counter() - t0, num_docs))

    umap = UMAP(
        n_neighbors=min(15, max(5, int(num_docs * 0.05))),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=max(5, int(min_topic_size * 0.3)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap,
        hdbscan_model=hdbscan,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    t1 = time.perf_counter()
    topics, _ = topic_model.fit_transform(docs, embeddings)
    timings.append(StageTiming("bertopic_fit", time.perf_counter() - t1, num_docs))

    return embeddings, np.asarray(topics, dtype=np.int32), timings


# ── Triton path ─────────────────────────────────────────────────────────────


def run_triton(
    docs: List[str],
    triton_url: str,
    model_name: str,
    tokenizer_dir: Path,
    *,
    triton_model: str = "ensemble-topics",
    max_length: int = 256,
) -> Tuple[np.ndarray, List[StageTiming]]:
    """One InvokeEndpoint-equivalent call to the `ensemble-topics` model.

    ``triton_model`` lets sweeps target alternate model versions in the
    repo (e.g. an FP32 build at ``ensemble-topics-fp32``) without touching
    the bench harness.
    """
    import tritonclient.http as httpclient
    from transformers import AutoTokenizer

    timings: List[StageTiming] = []
    num_docs = len(docs)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    encoded = tokenizer(
        docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    timings.append(StageTiming("tokenize", time.perf_counter() - t0, num_docs))

    client = httpclient.InferenceServerClient(url=triton_url.replace("http://", ""))

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    outputs = [httpclient.InferRequestedOutput("topic_labels")]

    t1 = time.perf_counter()
    result = client.infer(triton_model, inputs, outputs=outputs)
    timings.append(StageTiming("triton_ensemble", time.perf_counter() - t1, num_docs))

    labels = result.as_numpy("topic_labels").astype(np.int32)
    return labels, timings


# ── Parity ──────────────────────────────────────────────────────────────────


def parity_nmi(cpu_labels: np.ndarray, gpu_labels: np.ndarray) -> float:
    """NMI excluding outliers from both sides (label == -1)."""
    from sklearn.metrics import normalized_mutual_info_score

    mask = (cpu_labels != -1) & (gpu_labels != -1)
    if mask.sum() < 2:
        return float("nan")
    return float(
        normalized_mutual_info_score(cpu_labels[mask], gpu_labels[mask])
    )


# ── Entrypoint ──────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
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
        help="HF model id for the CPU baseline (must match the exported ONNX).",
    )
    p.add_argument("--output", type=Path, default=Path("bench_report.json"))
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--skip-triton", action="store_true")
    p.add_argument("--parity-threshold", type=float, default=0.95)
    args = p.parse_args()

    docs = _load_fixture(args.fixture, args.limit)
    print(f"Loaded {len(docs):,} docs from {args.fixture}")

    report: Dict[str, Any] = {
        "fixture": str(args.fixture),
        "num_docs": len(docs),
        "model_name": args.model_name,
    }

    cpu_labels: np.ndarray | None = None
    if not args.skip_cpu:
        print("\n── CPU baseline ──")
        _, cpu_labels, cpu_timings = run_cpu(docs, args.model_name)
        report["cpu"] = {
            "timings": [t.to_dict() for t in cpu_timings],
            "wall_seconds": sum(t.seconds for t in cpu_timings),
            "topic_count": int(len(set(cpu_labels.tolist())) - (1 if -1 in cpu_labels else 0)),
            "outlier_share": float((cpu_labels == -1).mean()),
        }
        print(json.dumps(report["cpu"], indent=2, default=str))

    gpu_labels: np.ndarray | None = None
    if not args.skip_triton:
        print("\n── Triton GPU ──")
        gpu_labels, gpu_timings = run_triton(
            docs, args.triton_url, args.model_name, args.tokenizer_dir
        )
        report["triton"] = {
            "timings": [t.to_dict() for t in gpu_timings],
            "wall_seconds": sum(t.seconds for t in gpu_timings),
            "topic_count": int(len(set(gpu_labels.tolist())) - (1 if -1 in gpu_labels else 0)),
            "outlier_share": float((gpu_labels == -1).mean()),
        }
        print(json.dumps(report["triton"], indent=2, default=str))

    if cpu_labels is not None and gpu_labels is not None:
        nmi = parity_nmi(cpu_labels, gpu_labels)
        report["parity"] = {
            "nmi": nmi,
            "threshold": args.parity_threshold,
            "passed": bool(nmi >= args.parity_threshold) if nmi == nmi else False,
        }
        print(f"\nParity NMI = {nmi:.4f} (threshold {args.parity_threshold})")

    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {args.output}")

    if "parity" in report and not report["parity"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
