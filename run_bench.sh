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
# Earlier optimum install dragged datasets 4.x into --user site-packages,
# which then imports against the env's pyarrow 17 and crashes on pa.json_().
# Cleanest fix: downgrade datasets to the 2.x line that's compatible with
# pyarrow 17, and uninstall the broken --user copy first.
python3 -m pip uninstall -y datasets pyarrow 2>/dev/null || true
python3 -m pip install --user --quiet 'datasets==2.21.*' 'pyarrow>=14,<18'
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
