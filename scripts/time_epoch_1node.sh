#!/bin/bash
# Step 1b: measure seconds/epoch on REAL CIFAR-10 (the smoke test uses synthetic data, so it can't tell you this).
# Run inside an interactive allocation, after tools/prepare_data.py and scripts/interactive_smoke.sh:
#   salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A m4431_g
#   bash scripts/time_epoch_1node.sh
# 2 epochs each for plain DDP and PowerSGD R=2 (~a few minutes). Results are tagged timing_* in results/runs.jsonl.
# Epoch 1 includes dataloader/cudnn warmup; the median step time ignores the first steps.
set -u
export PROJ=${PROJ:-$SCRATCH/powersgd}
source "$PROJ/scripts/common.sh"
start_gpu_monitor
for cfg in "ddp 0" "powersgd 2"; do
  set -- $cfg
  launch --model resnet18 --epochs 2 --batch-size 128 --start-iter 10 --mode "$1" --rank "$2" \
         --run-name "timing_${1}_r$2" 2>&1 | grep -E '"type": "(eval|done)"|Error'
done
stop_gpu_monitor
python - <<'EOF'
import json, os
rows = [json.loads(l) for l in open(os.path.join(os.environ["PROJ"], "results", "runs.jsonl")) if '"timing_' in l]
for r in rows:
    spe = r["planned_steps"] // 2
    print(f'{r["run_id"]}: median step {r["median_step_time_s"]*1000:.1f} ms, {spe} steps/epoch '
          f'=> ~{r["median_step_time_s"]*spe:.1f} s/epoch (train only), wall {r["wall_time_s"]:.0f}s for 2 epochs')
print("Rough 30-epoch cost per config: 15x the 2-epoch wall time; node-hours = that in hours x 1 node.")
EOF
