#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x cloud_venv/bin/python ]]; then
    echo "cloud_venv가 없습니다. 먼저 bash cloud_setup.sh를 실행하세요." >&2
    exit 1
fi

source cloud_venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python compare_adapters.py \
    --model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --output outputs/exaone35-78b-baseline.jsonl \
    --max-new-tokens 256
