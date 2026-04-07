#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nss-d/sf/Aloepri}"
MODEL_DIR="${MODEL_DIR:-/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B}"
CONDA_ENV="${CONDA_ENV:-qwen-transformers}"
DTYPE="${DTYPE:-bfloat16}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
SEED="${SEED:-20260323}"

cd "$REPO_DIR"

echo "[1/6] baseline smoke"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_baseline_smoke.py \
  --model-dir "$MODEL_DIR" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --output-path outputs/llama_baseline_smoke.json

echo "[2/6] export Stage I real checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_i_llama_real_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_i_llama_real \
  --dtype "$DTYPE" \
  --device "$EXPORT_DEVICE" \
  --seed "$SEED"

echo "[3/6] Stage I artifact sanity"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_stage_i_artifact_sanity.py \
  --model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_i_llama_real/server \
  --client-secret artifacts/stage_i_llama_real/client/client_secret.pt \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_i_llama/real_artifact_sanity.json

echo "[4/6] Stage I remote validation"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_remote_validation.py \
  --baseline-model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_i_llama_real/server \
  --client-secret artifacts/stage_i_llama_real/client/client_secret.pt \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_i_llama/real_remote_validation.json

echo "[5/6] export Stage J real checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_j_llama_real_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_j_llama_real_full_square \
  --dtype "$DTYPE" \
  --device "$EXPORT_DEVICE" \
  --seed "$SEED"

echo "[6/6] Stage J remote validation"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_remote_validation.py \
  --baseline-model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_j_llama_real_full_square/server \
  --client-secret artifacts/stage_j_llama_real_full_square/client/client_secret.pt \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_j_llama/real_remote_validation.json

echo "Done. Outputs are under:"
echo "  $REPO_DIR/outputs/"
echo "Artifacts are under:"
echo "  $REPO_DIR/artifacts/stage_i_llama_real"
echo "  $REPO_DIR/artifacts/stage_j_llama_real_full_square"
