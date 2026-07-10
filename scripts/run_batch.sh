#!/usr/bin/env bash
set -u

PROJECT_DIR=/home/opc/AI_Cloud
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/batch.log"
LOCK_FILE="$LOG_DIR/batch.lock"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

{
  echo "[RUN_BATCH START] time=$(date '+%Y-%m-%d %H:%M:%S')"
  flock -n "$LOCK_FILE" "$PYTHON_BIN" batch_collect.py
  exit_code=$?
  echo "[RUN_BATCH END] time=$(date '+%Y-%m-%d %H:%M:%S') exit_code=$exit_code"
  exit "$exit_code"
} >> "$LOG_FILE" 2>&1
