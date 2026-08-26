#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/cloud_venv/bin/python}"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION="0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
DATA_DIR="$ROOT/data/processed/gyeongsang/style_v4"
OUTPUT_DIR="$ROOT/outputs/exaone35-78b-gyeongsang-style-v4"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

"$PYTHON" "$ROOT/validate_gyeongsang_style_v4.py" --data-dir "$DATA_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$PYTHON" "$ROOT/train.py" \
  --model-id "$MODEL_ID" \
  --revision "$REVISION" \
  --train-file "$DATA_DIR/train.jsonl" \
  --validation-file "$DATA_DIR/validation.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 2 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 512
