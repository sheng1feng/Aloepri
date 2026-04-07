#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nss-d/sf/Aloepri}"
MODEL_DIR="${MODEL_DIR:-/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B}"
CONDA_ENV="${CONDA_ENV:-qwen-transformers}"
DTYPE="${DTYPE:-bfloat16}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
SEED="${SEED:-20260323}"

cd "$REPO_DIR"

echo "[1/4] run real Llama noise calibration"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_stage_j_llama_real_noise_calibration.py \
  --model-dir "$MODEL_DIR" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_j_llama/real_noise_calibration.json

echo "[2/4] export tiny_a real checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_j_llama_real_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_j_llama_real_full_square_tiny_a \
  --dtype "$DTYPE" \
  --device cpu \
  --seed "$SEED" \
  --alpha-e 0.02 \
  --alpha-h 0.01

echo "[3/4] validate tiny_a real checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_remote_validation.py \
  --baseline-model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_j_llama_real_full_square_tiny_a/server \
  --client-secret artifacts/stage_j_llama_real_full_square_tiny_a/client/client_secret.pt \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_j_llama/real_tiny_a_remote_validation.json

echo "[4/4] export Llama Stage-K release"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_k_llama_release.py \
  --export-dir artifacts/stage_k_llama_release

echo "Done. Outputs are under:"
echo "  $REPO_DIR/outputs/stage_j_llama/"
echo "Artifacts are under:"
echo "  $REPO_DIR/artifacts/stage_j_llama_real_full_square_tiny_a"
echo "  $REPO_DIR/artifacts/stage_k_llama_release"
