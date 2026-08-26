#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x cloud_venv/bin/python ]]; then
    echo "cloud_venv가 없습니다. 먼저 bash cloud_setup.sh를 실행하세요." >&2
    exit 1
fi

source cloud_venv/bin/activate
EVAL_PACKAGES="$PWD/eval_transformers_454"
if [[ ! -f "$EVAL_PACKAGES/transformers/__init__.py" ]]; then
    PIP_NO_CACHE_DIR=1 python -m pip install \
        --target "$EVAL_PACKAGES" \
        --no-deps \
        "transformers==4.54.1" \
        "tokenizers==0.21.4" \
        "huggingface-hub==0.36.0"
fi
export PYTHONPATH="$EVAL_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -c "import transformers; assert transformers.__version__ == '4.54.1'; print(transformers.__version__)"
python compare_adapters.py \
    --model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --output outputs/exaone35-78b-baseline.jsonl \
    --max-new-tokens 256
