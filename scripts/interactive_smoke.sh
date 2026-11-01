#!/bin/bash
# First thing to run on Perlmutter. Inside an interactive allocation, e.g.:
#   salloc -N 1 -C gpu -G 4 -q interactive -t 00:30:00 -A m4431_g     # (or -q debug; check `sacctmgr show assoc`)
#   bash scripts/interactive_smoke.sh
# Verifies: torch/NCCL/float8 versions, DDP + PowerSGD + fp16/bf16/fp8 hooks on 4 GPUs, and measures real
# step times so the Tier 1 job can be costed. Uses --synthetic (no dataset needed), ~5 minutes.
# Results go to a throwaway runs file so they never mix with real results.
set -u
export PROJ=${PROJ:-$SCRATCH/powersgd}
source "$PROJ/scripts/common.sh"
export RUNS_JSONL=$PROJ/results/smoke_runs.jsonl

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version(), "gpus", torch.cuda.device_count())
print("float8:", hasattr(torch, "float8_e4m3fn"), hasattr(torch, "float8_e5m2"), "| bf16 ok:", torch.cuda.is_bf16_supported())
EOF

for cfg in "ddp fp32" "powersgd fp32" "powersgd fp16" "powersgd bf16" "ddp bf16" "ddp fp8_e4m3" "powersgd fp8_e4m3"; do
  set -- $cfg
  echo "=== resnet18 mode=$1 precision=$2"
  launch --model resnet18 --synthetic --epochs 3 --batch-size 128 --start-iter 5 --rank 2 \
         --mode "$1" --precision "$2" --run-name "smoke_${1}_${2}" --out-dir "$PROJ/results/smoke" 2>&1 | grep -E '"type": "(eval|done)"|Error' | tail -2
done
python - <<'EOF'
import json, os
for l in open(os.environ["RUNS_JSONL"]):
    r = json.loads(l); print(f'{r["run_id"]:28s} {r["status"]:8s} median_step={r["median_step_time_s"]*1000:.1f}ms  bytes/step={r["comm_bytes_per_step"]:.3e}')
EOF
