#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x cloud_venv/bin/python ]]; then
    echo "cloud_venv가 없습니다. 먼저 bash cloud_setup.sh를 실행하세요." >&2
    exit 1
fi

source cloud_venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python build_gyeongsang_lora_v2.py
python train.py \
    --train-file data/processed/gyeongsang/lora_v2/train.jsonl \
    --validation-file data/processed/gyeongsang/lora_v2/validation.jsonl \
    --output-dir outputs/exaone-gyeongsang-lora-v2-pilot \
    --epochs 5 \
    --learning-rate 3e-5
