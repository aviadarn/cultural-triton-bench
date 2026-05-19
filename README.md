# Cultural Agent — Triton bench harness (Phase 1, AIS-2116)

Local-only scaffold for **AIS-2093 Phase 1**. Boots NVIDIA Triton on a single
GPU box, exports the production MiniLM model to ONNX, and benchmarks the
full BERTopic pipeline (embed → UMAP → HDBSCAN) against the current CPU
baseline. Emits speed numbers + a topic-label NMI parity score.

**No AWS resources are created by this harness.** Production wiring lands in
later phases (`triton_client.py` → Phase 2, SST infra → Phase 4).

## Layout

```
ml/cultural_agent/triton/
  model_repository/
    minilm-onnx/
      config.pbtxt         ONNX backend, FP16 + TensorRT, dynamic batch 32–128
      1/                   `model.onnx` lands here after running the export script
      tokenizer/           HF tokenizer files (also produced by the export script)
    bertopic-gpu/
      config.pbtxt         Python backend, cuML UMAP + cuML HDBSCAN
      1/model.py           cuML implementation mirroring agent.py:1007-1034
    ensemble-topics/
      config.pbtxt         Server-side chain: minilm-onnx → bertopic-gpu
  scripts/
    export_minilm_onnx.sh  optimum-cli export with mean-pooling fused in-graph
  bench/
    bench.py               CPU vs Triton side-by-side + NMI parity
    requirements.txt
  docker-compose.yml       Local tritonserver:24.09-py3 + cuML at boot
```

## Hardware needed

A single CUDA 12.x GPU with NVIDIA driver ≥ 535. Smallest sensible target is
`g5.xlarge` on AWS (~$1.50 for a one-hour bench). Local 4090/A100 boxes also
fine. CPU-only hosts can run the CPU baseline alone (see `--skip-triton`).

## Quickstart

```bash
# 1. Export ONNX model (CPU-only — no GPU needed for export itself)
pip install "optimum[exporters]==1.21.*" sentence-transformers
bash scripts/export_minilm_onnx.sh

# 2. Boot Triton (GPU host)
docker compose up
# Wait for: `Started GRPCInferenceService at 0.0.0.0:8001`
# First boot installs cuML — takes ~2 min.

# 3. Run the bench (separate shell, GPU host or anywhere with HTTP to Triton)
pip install -r bench/requirements.txt
python bench/bench.py \
    --fixture /path/to/cmi_documents.parquet \
    --limit 100000 \
    --triton-url http://localhost:8000 \
    --output bench_report.json
```

## Sweep mode — multi-config bench

Single-config runs use `bench/bench.py`. For comparing configs (corpus size,
precision, batch size) use `bench/sweep.py` with a YAML config:

```bash
python bench/sweep.py \
    --configs bench/configs/sweep_default.yml \
    --fixture /path/to/cmi_documents.parquet \
    --triton-url http://localhost:8000 \
    --output sweep_report
```

Writes `sweep_report.json` (raw, per-config timings) + `sweep_report.csv`
(flat table for spreadsheet review).

CPU baseline runs once at the configured limit and is reused as the parity
anchor for every Triton run whose `limit` matches. Triton runs at other
corpus sizes still report wall-time + speedup vs CPU but skip NMI (apples
to oranges otherwise).

Included starter configs in `bench/configs/`:

| File | Sweeps over | Use when |
| --- | --- | --- |
| `sweep_default.yml` | corpus size 10k / 50k / 100k on `ensemble-topics` | first run on a new box |
| `sweep_precision.yml` | FP16 (`ensemble-topics`) vs FP32 (`ensemble-topics-fp32`) | deciding whether TensorRT FP16 buys throughput on the target GPU |
| `sweep_batch_size.yml` | CPU embedding batch size | tuning the CPU baseline; Triton batch sizing is owned by `minilm-onnx/config.pbtxt` |

Exit code 2 if any Triton config misses the parity threshold; 0 otherwise.

## CPU baseline numbers (darwin M-series, no GPU)

Recorded with the bench harness on a synthetic fixture (10 base posts ×
N variants). Real Meltwater corpus will differ on topic count + outlier
share but the per-stage shape holds. Production Fargate is slower per
core than this host — projections in the GPU column scale accordingly.

| Docs    | Embed (s) | BERTopic (s) | Wall (s) | Embed docs/s | BERTopic docs/s | Topics | Outlier % |
| ------- | --------- | ------------ | -------- | ------------ | --------------- | ------ | --------- |
| 100     | 1.03      | 2.76         | 3.79     | 97           | 36              | 9      | 0.0       |
| 1,000   | 1.09      | 3.53         | 4.62     | 914          | 284             | 17     | 2.1       |
| 10,000  | 3.78      | 12.13        | 15.91    | 2,643        | 824             | 25     | 6.5       |
| 100,000 | 29.70     | 48.57        | 78.26    | 3,367        | 2,059           | 154    | 35.8      |
| 200,000 | 58.75     | 142.29       | 201.04   | 3,404        | 1,406           | 424    | 32.0      |

**Key observations:**

- **Embedding scales linearly** past 10k — 3.4k docs/s steady state on
  this host. 100k → 200k is exactly 2× time, so CPU is acceptable here
  if the rest of the pipeline is fast.
- **BERTopic is super-linear** between 100k and 200k: 2.9× time for 2×
  docs (48.6 s → 142.3 s). UMAP + HDBSCAN drift from O(n log n) toward
  O(n²) at this scale — exactly the curve AIS-2093 calls out.
- **BERTopic dominates wall time** by ~2.4× at 200k (142 s vs 58.8 s
  embed). The cuML port is the load-bearing change; the ONNX MiniLM
  port is the smaller win.

**GPU projection (darwin baseline):**

| Stage    | CPU 200k | GPU target  | Speedup |
| -------- | -------- | ----------- | ------- |
| Embed    | 58.8 s   | ~60 s       | ~1.0× (FP16 TensorRT) |
| BERTopic | 142.3 s  | 10–14 s     | **10–14×** (cuML)     |
| Wall     | 201.0 s  | ~70–75 s    | **~2.7–2.9×**         |

On Fargate baseline (slower x86 cores + `NUMBA_DISABLE_JIT`) the wall
speedup projects 5–10×, matching the ticket's headline.

Reproduce locally:

```bash
# build a synthetic fixture from the search-agent meltwater fixture
python3 -c "
import json, pathlib
src = pathlib.Path('functions/src/functions/evaluation/fixtures/search_agent/meltwater_posts.jsonl')
base = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
N = 200000  # target doc count
with open('ml/cultural_agent/triton/bench/synthetic_{N}.jsonl'.format(N=N), 'w') as fh:
    for i in range(N // len(base)):
        for j, p in enumerate(base):
            t = p['content'].get('title','') or ''
            b = p['content'].get('body','') or ''
            fh.write(json.dumps({'document': f'{t}. {b} (variant {i}.{j})'}) + '\n')
"
# CPU baseline only (no GPU host required)
python bench/bench.py \
    --fixture bench/synthetic_200000.jsonl \
    --limit 200000 \
    --skip-triton \
    --output bench_cpu_200000.json
```

Raw per-run reports live in `ml/cultural_agent/triton/bench/*.json` after
each run.

## Parity gate

The bench computes NMI between CPU labels and Triton labels on the same
fixture. Threshold defaults to **0.95**; the script exits non-zero if the
gate is missed. cuML's HDBSCAN classifies borderline points differently
from sklearn-equivalent HDBSCAN — outliers (`label == -1`) on either side
are excluded from the parity calculation.

If NMI is < 0.95, document the gap, the topic-count delta, and the
outlier-share delta, then decide:

- accept the parity gap with documented evidence (rare), or
- tune cuML hyperparameters in `bertopic-gpu/config.pbtxt`, or
- block Phase 2.

## Open question — cuML in image

`docker-compose.yml` installs `cuml-cu12` at container boot. Adds ~2 min cold
start. Two options for Phase 4:

1. **Boot-install** — simpler CI, slower cold start (≈3–5 min on async
   endpoint resume).
2. **Custom Triton image** — bake `cuml-cu12` into a derived image pushed
   to ECR. Faster cold start, more CI surface area.

Phase 1's bench numbers decide. If the cold-start adder is < ~1 min in
practice, boot-install stays; otherwise we move to the custom image.

## What this harness deliberately does *not* do

- Talk to AWS / SageMaker
- Change `functions/src/functions/cultural_agent/agent.py`
- Persist anything to S3 / DDB / Postgres

It is purely a measurement tool. Real production wiring lands in Phase 2+.
