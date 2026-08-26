#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="$(pwd)/cloud_venv/bin/python"
ADAPTER="outputs/exaone35-78b-gyeongsang-lora-v3-pilot"

if [[ ! -x "$PYTHON" ]]; then
    echo "cloud_venv가 없습니다." >&2
    exit 1
fi
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "학습 어댑터를 찾을 수 없습니다: $ADAPTER" >&2
    exit 1
fi

"$PYTHON" compare_adapters.py \
    --model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --revision 0ff6b5ec7c13b049b253a16a889aa269e6b79a94 \
    --adapter "v3=$ADAPTER" \
    --output outputs/exaone35-78b-v3-comparison.jsonl \
    --max-new-tokens 256
