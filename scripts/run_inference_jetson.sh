#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/zhanghongyu/fsas/models/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

unset PYTORCH_CUDA_ALLOC_CONF

PYTHON_BIN="${PYTHON_BIN:-jetson_venv/bin/python}"
SAMPLE_PATH="${ALPAMAYO_SAMPLE_PATH:-.codex/alpamayo_sample.pt}"
LOG_PATH="${ALPAMAYO_LOG_PATH:-.codex/run_inference_jetson_repo.log}"

exec "${PYTHON_BIN}" src/alpamayo_r1/test_inference.py \
  --sample-path "${SAMPLE_PATH}" \
  --attn-implementation sdpa \
  --jetson-compat \
  --low-cpu-mem-usage \
  --device-map-cuda \
  --log-path "${LOG_PATH}"
