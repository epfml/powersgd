#!/bin/bash
# One-time setup on a Perlmutter login node: checks the Python stack and downloads
# CIFAR-10 and WikiText-2 into $DATA/data (compute jobs then never touch the internet).
#   bash perlmutter/setup.sh
set -eo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$DATA/data" "$RESULTS"

python - <<'EOF'
import importlib
for name in ["torch", "torchvision", "numpy", "matplotlib"]:
    try:
        print(f"{name:12s} {importlib.import_module(name).__version__}")
    except ImportError:
        print(f"{name:12s} MISSING -> run: pip install --user {name}")
import torch
print(f"{'CUDA':12s} {torch.version.cuda}, NCCL {torch.cuda.nccl.version()}")
EOF

cd "$REPO_DIR/paper-code"
python -c "import train; train.download_cifar(); train.download_wikitext2()"
echo
echo "Data in $DATA/data:"
ls "$DATA/data"
echo "Results will go to $RESULTS"
