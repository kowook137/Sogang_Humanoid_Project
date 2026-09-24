#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential
python3 -m venv --system-site-packages cloud_venv
source cloud_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python train.py
