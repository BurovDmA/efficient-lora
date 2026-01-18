#!/usr/bin/env bash
set -euo pipefail

# Start MLflow Tracking Server with sqlite backend store.
# Creates:
# - ./database.db  (runs/metrics/params)
# - ./mlruns/      (artifacts)
#
# Usage (from repo root `efficient-lora/`):
#   bash scripts/mlflow_server.sh
#
# Then in another terminal run training with:
#   python src/train.py logger=mlflow ...

HOST="${MLFLOW_HOST:-127.0.0.1}"
PORT="${MLFLOW_PORT:-5000}"

mlflow server \
  --backend-store-uri "sqlite:///database.db" \
  --default-artifact-root "./mlruns" \
  --host "${HOST}" \
  --port "${PORT}"


