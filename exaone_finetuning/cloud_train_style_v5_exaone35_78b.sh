#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/cloud_venv/bin/python"
DATA="$ROOT/data/processed/gyeongsang/style_v5"
OUTPUT="$ROOT/outputs/exaone35-78b-gyeongsang-style-v5"
MODEL="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION="0ff6b5ec7c13b049b253a16a889aa269e6b79a94"

"$PYTHON" "$ROOT/validate_gyeongsang_style_v5.py" --data-dir "$DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$PYTHON" "$ROOT/train.py" \
  --model-id "$MODEL" \
  --revision "$REVISION" \
  --train-file "$DATA/train.jsonl" \
  --validation-file "$DATA/validation.jsonl" \
  --output-dir "$OUTPUT" \
  --epochs 2 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 512
