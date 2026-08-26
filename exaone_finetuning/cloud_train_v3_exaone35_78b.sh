#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x cloud_venv/bin/python ]]; then
    echo "cloud_venv가 없습니다. 기존 VM 가상환경을 확인하세요." >&2
    exit 1
fi

PYTHON="$(pwd)/cloud_venv/bin/python"
MODEL_ID="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
MODEL_REVISION="0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
OUTPUT_DIR="outputs/exaone35-78b-gyeongsang-lora-v3-pilot"

"$PYTHON" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
"$PYTHON" build_gyeongsang_lora_v3.py
"$PYTHON" train.py \
    --model-id "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --train-file data/processed/gyeongsang/lora_v3/train.jsonl \
    --validation-file data/processed/gyeongsang/lora_v3/validation.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --learning-rate 2e-5 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-length 1024
