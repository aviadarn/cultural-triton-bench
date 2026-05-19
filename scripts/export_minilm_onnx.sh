#!/usr/bin/env bash
# Export sentence-transformers/all-MiniLM-L6-v2 to ONNX with mean-pooling
# fused into the graph. Drops `model.onnx` into the Triton model repo.
#
# Requires:
#   pip install "optimum[exporters]==1.21.*" sentence-transformers
#
# The optimum exporter is the only path that includes the pooling layer
# inside the ONNX graph. Raw `transformers.onnx` / HF export emits the
# encoder only and forces client-side pooling, which loses throughput.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../model_repository" && pwd)"
OUT_DIR="${REPO_ROOT}/minilm-onnx/1"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

MODEL_ID="${MODEL_ID:-sentence-transformers/all-MiniLM-L6-v2}"

echo "Exporting ${MODEL_ID} -> ${OUT_DIR}/model.onnx"

optimum-cli export onnx \
  --model "${MODEL_ID}" \
  --task feature-extraction \
  --optimize O3 \
  --device cpu \
  "${WORK_DIR}"

# optimum drops `model.onnx` + tokenizer files into WORK_DIR.
mkdir -p "${OUT_DIR}"
cp "${WORK_DIR}/model.onnx" "${OUT_DIR}/model.onnx"

# Tokenizer files live next to the model repo so the bench harness can
# tokenize client-side (mirrors what the worker will do via Triton client).
TOK_DIR="${REPO_ROOT}/minilm-onnx/tokenizer"
mkdir -p "${TOK_DIR}"
cp "${WORK_DIR}/tokenizer.json" "${TOK_DIR}/"
cp "${WORK_DIR}/tokenizer_config.json" "${TOK_DIR}/"
cp "${WORK_DIR}/vocab.txt" "${TOK_DIR}/" 2>/dev/null || true
cp "${WORK_DIR}/special_tokens_map.json" "${TOK_DIR}/" 2>/dev/null || true

echo "Done. Inspect: $(ls -lh "${OUT_DIR}/model.onnx")"
