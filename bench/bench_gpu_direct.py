"""Direct GPU bench — no Triton, no ONNX.

Phase 1 measurement: prove the GPU win on cultural-agent's inner loop.
- CPU baseline: SentenceTransformer (cpu) + sklearn-equivalent BERTopic
- GPU run: SentenceTransformer (cuda) + cuML UMAP + cuML HDBSCAN

Triton serving (TensorRT FP16, dynamic batching) is a Phase 4 optimisation
detail that doesn't change the conclusion: GPU > CPU for embed + BERTopic.

Usage:
    python bench_gpu_direct.py --fixture <jsonl|parquet> --limit N --output report.json
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

from bench import _load_fixture, parity_nmi


@dataclass
class StageTiming:
    name: str
    seconds: float
    docs: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["docs_per_sec"] = self.docs / self.seconds if self.seconds else 0.0
        return d


def run_cpu(
    docs: List[str], model_name: str, *, embedding_batch_size: int = 256
) -> Tuple[np.ndarray, List[StageTiming]]:
    """CPU baseline — mirror of agent.py:999-1034."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    num_docs = len(docs)
    min_topic_size = max(10, min(int(num_docs * 0.015), 120))
    timings: List[StageTiming] = []

    t0 = time.perf_counter()
    embedder = SentenceTransformer(model_name, device="cpu")
    embeddings = embedder.encode(
        docs,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    timings.append(StageTiming("cpu_embed", time.perf_counter() - t0, num_docs))

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
    timings.append(StageTiming("cpu_bertopic", time.perf_counter() - t1, num_docs))
    return np.asarray(topics, dtype=np.int32), timings


def run_gpu(
    docs: List[str], model_name: str, *, embedding_batch_size: int = 256
) -> Tuple[np.ndarray, List[StageTiming]]:
    """GPU path — sentence-transformers cuda + cuML UMAP + cuML HDBSCAN."""
    import torch
    from sentence_transformers import SentenceTransformer
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.manifold import UMAP as cuUMAP

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available for GPU bench")

    num_docs = len(docs)
    min_topic_size = max(10, min(int(num_docs * 0.015), 120))
    timings: List[StageTiming] = []

    t0 = time.perf_counter()
    embedder = SentenceTransformer(model_name, device="cuda")
    embeddings = embedder.encode(
        docs,
        batch_size=embedding_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    timings.append(StageTiming("gpu_embed", time.perf_counter() - t0, num_docs))

    t1 = time.perf_counter()
    umap = cuUMAP(
        n_neighbors=min(15, max(5, int(num_docs * 0.05))),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    reduced = umap.fit_transform(embeddings)
    timings.append(StageTiming("gpu_umap", time.perf_counter() - t1, num_docs))

    t2 = time.perf_counter()
    hdbscan = cuHDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=max(5, int(min_topic_size * 0.3)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = hdbscan.fit_predict(reduced)
    timings.append(StageTiming("gpu_hdbscan", time.perf_counter() - t2, num_docs))

    return np.asarray(labels, dtype=np.int32), timings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--output", type=Path, default=Path("bench_gpu_direct.json"))
    p.add_argument("--parity-threshold", type=float, default=0.95)
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    docs = _load_fixture(args.fixture, args.limit)
    print(f"Loaded {len(docs):,} docs from {args.fixture}")

    report: Dict[str, Any] = {
        "fixture": str(args.fixture),
        "num_docs": len(docs),
        "model_name": args.model_name,
    }

    cpu_labels = gpu_labels = None
    if not args.skip_cpu:
        print("\n── CPU baseline ──")
        cpu_labels, cpu_timings = run_cpu(docs, args.model_name)
        report["cpu"] = {
            "timings": [t.to_dict() for t in cpu_timings],
            "wall_seconds": sum(t.seconds for t in cpu_timings),
            "topic_count": int(len(set(cpu_labels.tolist())) - (1 if -1 in cpu_labels else 0)),
            "outlier_share": float((cpu_labels == -1).mean()),
        }
        print(json.dumps(report["cpu"], indent=2, default=str))

    if not args.skip_gpu:
        print("\n── GPU direct ──")
        gpu_labels, gpu_timings = run_gpu(docs, args.model_name)
        report["gpu"] = {
            "timings": [t.to_dict() for t in gpu_timings],
            "wall_seconds": sum(t.seconds for t in gpu_timings),
            "topic_count": int(len(set(gpu_labels.tolist())) - (1 if -1 in gpu_labels else 0)),
            "outlier_share": float((gpu_labels == -1).mean()),
        }
        print(json.dumps(report["gpu"], indent=2, default=str))

    if cpu_labels is not None and gpu_labels is not None:
        nmi = parity_nmi(cpu_labels, gpu_labels)
        report["parity"] = {
            "nmi": nmi,
            "threshold": args.parity_threshold,
            "passed": bool(nmi >= args.parity_threshold) if nmi == nmi else False,
        }
        cpu_wall = report["cpu"]["wall_seconds"]
        gpu_wall = report["gpu"]["wall_seconds"]
        report["speedup"] = cpu_wall / gpu_wall if gpu_wall else None
        print(f"\nNMI = {nmi:.4f} | speedup = {report['speedup']:.2f}x")

    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {args.output}")
    if "parity" in report and not report["parity"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
