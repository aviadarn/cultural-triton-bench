# SageMaker Notebook runbook — Phase 1 bench

End-to-end steps for running the GPU side of the cultural-agent Triton bench
on a SageMaker `ml.g5.xlarge` notebook instance. Targets AWS account
`UnileverSandbox` (062377979297, us-east-2).

## Prereqs
- `aws sso login --profile UnileverSandbox` already run on your laptop
- Notebook instance `ais-2116-phase1-bench` created and `InService`
- SageMaker exec role attached (S3 + pip + docker hub access)

## 1. Open the notebook terminal

```bash
aws --profile UnileverSandbox sagemaker create-presigned-notebook-instance-url \
    --notebook-instance-name ais-2116-phase1-bench
```

Open the URL → New → Terminal.

## 2. Clone the mirror + boot Triton

```bash
cd /home/ec2-user/SageMaker
git clone https://github.com/aviadarn/cultural-triton-bench.git
cd cultural-triton-bench

# ONNX export (CPU, ~1 min)
python3 -m pip install --user "optimum[exporters]==1.21.*" sentence-transformers
bash scripts/export_minilm_onnx.sh

# Triton + cuML (first boot ~2 min installing cuml-cu12)
docker compose up -d
until curl -fsS localhost:8000/v2/health/ready; do sleep 3; done
echo "Triton ready"
```

## 3. Run the bench

```bash
python3 -m pip install --user -r bench/requirements.txt

# Build the synthetic fixture
python3 bench/build_synthetic_fixture.py --count 100000 --output bench/synthetic_100k.jsonl

# Single-config bench (full CPU vs GPU + parity NMI)
python3 bench/bench.py \
    --fixture bench/synthetic_100k.jsonl \
    --limit 100000 \
    --triton-url http://localhost:8000 \
    --output bench_report_100k.json

# Or a sweep (writes .json + .csv)
python3 bench/sweep.py \
    --configs bench/configs/sweep_default.yml \
    --fixture bench/synthetic_100k.jsonl \
    --output sweep_report
```

## 4. Pull results back

From your laptop:

```bash
aws --profile UnileverSandbox s3 cp bench_report_100k.json s3://<scratch-bucket>/aviad/phase1/
# or just download from the Jupyter file browser
```

## 5. Tear down

```bash
aws --profile UnileverSandbox sagemaker stop-notebook-instance \
    --notebook-instance-name ais-2116-phase1-bench
# and optionally:
aws --profile UnileverSandbox sagemaker delete-notebook-instance \
    --notebook-instance-name ais-2116-phase1-bench
```

`stop` halts billing; `delete` removes the instance + EBS. Stop is enough
between sessions; delete when Phase 1 is closed.
