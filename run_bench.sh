#!/bin/bash
set -ex
export PATH=$HOME/.local/bin:$PATH

cd /home/ec2-user/SageMaker/cultural-triton-bench

echo "=== GPU check ==="
nvidia-smi || true

# Skip Triton + ONNX entirely — direct GPU bench using sentence-transformers
# CUDA + cuML. Triton serving is a Phase 4 optimisation detail; the ONNX
# exporter is fragile on torch 2.12. Direct GPU still answers the Phase 1
# question: how much faster is GPU end-to-end + does parity NMI hold.

echo "=== Bench deps ==="
# datasets 4.x needs pyarrow>=21 (SageMaker base has 17). Upgrade pyarrow
# first so sentence-transformers + bertopic imports don't crash.
python3 -m pip install --user --quiet --upgrade 'pyarrow>=21'
python3 -m pip install --user --quiet 'sentence-transformers==3.0.*'
python3 -m pip install --user --quiet --no-deps 'bertopic==0.16.*' 'hdbscan==0.8.*' 'umap-learn==0.5.*'

echo "=== cuML (RAPIDS) for GPU UMAP + HDBSCAN ==="
python3 -m pip install --user --quiet \
    --extra-index-url=https://pypi.nvidia.com \
    'cuml-cu12==24.10.*'

echo "=== Build synthetic 100k fixture ==="
python3 bench/build_synthetic_fixture.py --count 100000 --output bench/synthetic_100k.jsonl

echo "=== Bench (CPU + GPU direct + parity NMI) ==="
cd bench
python3 bench_gpu_direct.py \
    --fixture synthetic_100k.jsonl \
    --limit 100000 \
    --output ../bench_report_100k.json
cd ..

echo "=== Upload ==="
aws s3 cp bench_report_100k.json s3://sagemaker-ais-2116-phase1-062377979297-us-east-2/bench_report_100k.json
echo DONE
