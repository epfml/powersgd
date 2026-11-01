#!/usr/bin/env python3
"""Download datasets on a LOGIN node (compute nodes have no internet).  Usage: python tools/prepare_data.py [DATA_DIR]"""
import os
import sys
import urllib.request
from pathlib import Path

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA_DIR", Path(os.environ.get("SCRATCH", ".")) / "powersgd" / "data"))
data_dir.mkdir(parents=True, exist_ok=True)

import torchvision  # noqa: E402

torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

shk = data_dir / "shakespeare_char"
shk.mkdir(exist_ok=True)
if not (shk / "input.txt").exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", shk / "input.txt")
print("data ready in", data_dir)
