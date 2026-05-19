#!/bin/bash
set -ex
export PATH=$HOME/.local/bin:$PATH

cd /home/ec2-user/SageMaker/cultural-triton-bench

echo "=== GPU check ==="
nvidia-smi || true

echo "=== ONNX export deps ==="
python3 -m pip install --user --quiet "optimum[exporters]==1.21.*" "sentence-transformers==3.0.*"

echo "=== ONNX export ==="
bash scripts/export_minilm_onnx.sh

echo "=== Triton up ==="
docker compose up -d
i=0
until curl -fsS localhost:8000/v2/health/ready >/dev/null 2>&1; do
  i=$((i+1))
  if [ $i -ge 120 ]; then
    echo "triton never came up"
    docker compose logs --tail=80
    exit 1
  fi
  sleep 5
done
echo "triton ready"

echo "=== Bench deps (skip conflicting requirements file) ==="
python3 -m pip install --user --quiet 'tritonclient[http]==2.49.*' pyyaml
python3 -m pip install --user --quiet --no-deps 'bertopic==0.16.*' 'hdbscan==0.8.*' 'umap-learn==0.5.*'

echo "=== Build synthetic 100k fixture ==="
python3 bench/build_synthetic_fixture.py --count 100000 --output bench/synthetic_100k.jsonl

echo "=== Bench (CPU + Triton + parity NMI) ==="
python3 bench/bench.py \
    --fixture bench/synthetic_100k.jsonl \
    --limit 100000 \
    --triton-url http://localhost:8000 \
    --output bench_report_100k.json

echo "=== Upload ==="
aws s3 cp bench_report_100k.json s3://sagemaker-ais-2116-phase1-062377979297-us-east-2/bench_report_100k.json
echo DONE
