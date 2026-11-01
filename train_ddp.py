#!/usr/bin/env python3
"""Config-driven DDP / PowerSGD trainer for the midterm (ResNet-18/CIFAR-10, NanoGPT/Shakespeare-char).

Launch with torchrun (1 or more nodes), e.g.
    torchrun --nproc_per_node=4 train_ddp.py --model resnet18 --mode powersgd --rank 2 --epochs 30

--mode       ddp       plain all-reduce (via PyTorch's allreduce_hook, so bytes are counted the same way)
             powersgd  torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook (error feedback + warm start ON)
--precision  gradient *communication* precision (compute stays fp32 / TF32 as PyTorch default):
             fp32 | fp16 | bf16 | fp8_e4m3 | fp8_e5m2
             fp8 is EMULATED (A100 has no FP8 hardware): each bucket is scaled per-tensor, cast to float8 and back
             to fp32 *before* the all-reduce, so numerics are tested but bytes on the wire and speed are NOT FP8.
--rank       PowerSGD matrix_approximation_rank R (named --rank to match the brief; NOT the dist rank).
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook as powerSGD
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.run_logger import ROOT, log_run  # noqa: E402
from models import GPT, GPTConfig, ResNet18  # noqa: E402


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["resnet18", "nanogpt"], required=True)
    p.add_argument("--mode", choices=["ddp", "powersgd"], default="ddp")
    p.add_argument("--rank", type=int, default=2, help="PowerSGD rank R")
    p.add_argument("--start-iter", type=int, default=10, help="PowerSGD start_powerSGD_iter (vanilla allreduce before)")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"], default="fp32")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None, help="per GPU (default 128 resnet / 64 nanogpt)")
    p.add_argument("--epochs", type=int, default=30, help="resnet18 only")
    p.add_argument("--max-steps", type=int, default=None, help="nanogpt total steps (default 2000); also caps resnet")
    p.add_argument("--eval-interval", type=int, default=250, help="nanogpt eval every N steps")
    p.add_argument("--lr", type=float, default=None, help="LR at --lr-ref-batch global batch; scaled linearly")
    p.add_argument("--lr-ref-batch", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "runs"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--synthetic", action="store_true", help="random data, for pipeline smoke tests only")
    p.add_argument("--profile-step", type=int, default=0, help=">0: torch.profiler trace of that single step")
    p.add_argument("--timing-warmup", type=int, default=None, help="steps ignored in step-time stats")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    return p.parse_args()


# ----------------------------------------------------------------------------- comm hooks
class CommCounter:
    """Counts logical payload bytes passed to dist.all_reduce (per rank, not ring wire traffic)."""

    def __init__(self):
        self.bytes, self.enabled = 0, True
        orig = dist.all_reduce

        def counted(tensor, *a, **k):
            if self.enabled:
                self.bytes += tensor.numel() * tensor.element_size()
            return orig(tensor, *a, **k)

        dist.all_reduce = counted


def fp8_wrapper(hook, fp8_dtype):
    """Emulated FP8: per-bucket scale -> cast to float8 -> back to the bucket dtype, then run `hook`."""
    fmax = torch.finfo(fp8_dtype).max

    def wrapped(state, bucket):
        buf = bucket.buffer()
        scale = fmax / buf.abs().max().clamp(min=1e-12)
        buf.copy_(((buf * scale).to(fp8_dtype).to(buf.dtype)) / scale)
        return hook(state, bucket)

    return wrapped


def build_hook(args):
    if args.mode == "powersgd":
        state = powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=args.rank,
            start_powerSGD_iter=args.start_iter,
            use_error_feedback=True,
            warm_start=True,
            random_seed=args.seed,
        )
        hook = powerSGD.powerSGD_hook
    else:
        state, hook = None, default_hooks.allreduce_hook
    if args.precision == "fp16":
        hook = default_hooks.fp16_compress_wrapper(hook)
    elif args.precision == "bf16":
        hook = default_hooks.bf16_compress_wrapper(hook)
    elif args.precision.startswith("fp8"):
        hook = fp8_wrapper(hook, torch.float8_e4m3fn if args.precision == "fp8_e4m3" else torch.float8_e5m2)
    return state, hook


# ----------------------------------------------------------------------------- tasks
def cosine_lr(step, total, base, warmup, floor=0.0):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * t))


class ResNetTask:
    name, dataset = "resnet18", "cifar10"

    def __init__(self, a, ctx):
        self.a, self.ctx = a, ctx
        self.bs = a.batch_size or 128
        self.dev = ctx["device"]
        if a.synthetic:
            g = torch.Generator().manual_seed(0)
            mk = lambda n: torch.utils.data.TensorDataset(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g))
            self.train_ds, self.test_ds = mk(4096), mk(512)
        else:
            import torchvision
            import torchvision.transforms as T

            norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            self.train_ds = torchvision.datasets.CIFAR10(
                a.data_dir, train=True, download=False,
                transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm]))
            self.test_ds = torchvision.datasets.CIFAR10(a.data_dir, train=False, download=False, transform=T.Compose([T.ToTensor(), norm]))
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_ds, shuffle=True, seed=a.seed, drop_last=True)
        self.loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.bs, sampler=self.sampler, drop_last=True,
            num_workers=0 if a.synthetic else a.workers, pin_memory=self.dev.type == "cuda", persistent_workers=not a.synthetic and a.workers > 0)
        self.steps_per_epoch = len(self.loader)
        self.total_steps = a.epochs * self.steps_per_epoch
        if a.max_steps:
            self.total_steps = min(self.total_steps, a.max_steps)
        self._it, self._epoch = None, -1

    def build_model(self):
        return ResNet18()

    def make_optimizer(self, model):
        decay = [p for p in model.parameters() if p.ndim > 1]
        nodecay = [p for p in model.parameters() if p.ndim <= 1]
        return torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": nodecay, "weight_decay": 0.0}],
                               lr=0.1, momentum=0.9, nesterov=True)

    def base_lr(self):
        return self.a.lr if self.a.lr is not None else 0.1

    def warmup(self):
        return self.a.warmup_steps if self.a.warmup_steps is not None else self.steps_per_epoch  # 1 epoch

    def lr_at(self, step, lr_eff):
        return cosine_lr(step, self.total_steps, lr_eff, self.warmup())

    def next_batch(self, step):
        if step % self.steps_per_epoch == 0 or self._it is None:
            self._epoch = step // self.steps_per_epoch
            self.sampler.set_epoch(self._epoch)
            self._it = iter(self.loader)
        x, y = next(self._it)
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)

    def loss(self, model, batch):
        x, y = batch
        return F.cross_entropy(model(x), y)

    def is_eval_step(self, step):
        return (step + 1) % self.steps_per_epoch == 0 or step + 1 == self.total_steps

    @torch.no_grad()
    def evaluate(self, model):  # rank 0 only, full test set
        model.eval()
        n = correct = 0
        tot = 0.0
        for x, y in torch.utils.data.DataLoader(self.test_ds, batch_size=500):
            x, y = x.to(self.dev), y.to(self.dev)
            out = model(x)
            tot += F.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        model.train()
        return {"test_acc": correct / n, "test_loss": tot / n}

    def final_metric(self, ev):
        return {"test_acc": ev["test_acc"], "test_loss": ev["test_loss"]}

    def samples_per_step(self, ws):
        return self.bs * ws


class GPTTask:
    name, dataset = "nanogpt", "shakespeare_char"

    def __init__(self, a, ctx):
        self.a, self.ctx, self.dev = a, ctx, ctx["device"]
        self.bs = a.batch_size or 64
        if a.synthetic:
            data = np.random.RandomState(0).randint(0, 65, 200_000).astype(np.int64)
            self.vocab = 65
        else:
            text = open(Path(a.data_dir) / "shakespeare_char" / "input.txt").read()
            chars = sorted(set(text))
            stoi = {c: i for i, c in enumerate(chars)}
            data = np.array([stoi[c] for c in text], dtype=np.int64)
            self.vocab = len(chars)
        n = int(0.9 * len(data))
        self.train, self.val = torch.from_numpy(data[:n]), torch.from_numpy(data[n:])
        self.total_steps = a.max_steps or 2000
        # per-rank generator: different data per rank, deterministic given seed
        self.gen = torch.Generator().manual_seed(a.seed * 1000 + ctx["rank"])
        self.steps_per_epoch = self.total_steps  # no epochs; keeps trainer generic

    def build_model(self):
        a = self.a
        return GPT(GPTConfig(vocab_size=self.vocab, block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd))

    def make_optimizer(self, model):
        decay = [p for p in model.parameters() if p.ndim >= 2]
        nodecay = [p for p in model.parameters() if p.ndim < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                                 lr=1e-3, betas=(0.9, 0.99))

    def base_lr(self):
        return self.a.lr if self.a.lr is not None else 1e-3

    def warmup(self):
        return self.a.warmup_steps if self.a.warmup_steps is not None else max(1, min(100, self.total_steps // 10))

    def lr_at(self, step, lr_eff):
        return cosine_lr(step, self.total_steps, lr_eff, self.warmup(), floor=lr_eff * 0.1)

    def _batch(self, data, gen):
        T = self.a.block_size
        ix = torch.randint(len(data) - T - 1, (self.bs,), generator=gen)
        x = torch.stack([data[i:i + T] for i in ix])
        y = torch.stack([data[i + 1:i + T + 1] for i in ix])
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)

    def next_batch(self, step):
        return self._batch(self.train, self.gen)

    def loss(self, model, batch):
        return model(*batch)[1]

    def is_eval_step(self, step):
        return (step + 1) % self.a.eval_interval == 0 or step + 1 == self.total_steps

    @torch.no_grad()
    def evaluate(self, model):  # rank 0 only, fixed eval batches
        model.eval()
        g = torch.Generator().manual_seed(12345)
        losses = [model(*self._batch(self.val, g))[1].item() for _ in range(40)]
        model.train()
        v = float(np.mean(losses))
        return {"val_loss": v, "val_ppl": math.exp(v)}

    def final_metric(self, ev):
        return ev

    def samples_per_step(self, ws):
        return self.bs * ws * self.a.block_size  # tokens


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    dist.init_process_group("nccl" if cuda else "gloo")
    is0 = rank == 0
    # same seed on all ranks -> identical init (DDP also broadcasts rank 0's weights)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ctx = {"device": dev, "rank": rank}
    task = ResNetTask(a, ctx) if a.model == "resnet18" else GPTTask(a, ctx)

    rid = [a.run_name or f"{a.model}_{a.mode}{a.rank if a.mode == 'powersgd' else ''}_{a.precision}_w{ws}_s{a.seed}_{time.strftime('%m%d-%H%M%S')}"]
    dist.broadcast_object_list(rid, src=0)
    run_id = rid[0]
    run_dir = Path(a.out_dir) / run_id
    if is0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(a) | {"world_size": ws, "run_id": run_id}, indent=1))
    metrics_f = open(run_dir / "metrics.jsonl", "a") if is0 else None

    def emit(rec):
        if is0:
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(json.dumps(rec), flush=True)

    model = task.build_model().to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    model = DDP(model, device_ids=[local] if cuda else None)
    counter = CommCounter()
    state, hook = build_hook(a)
    model.register_comm_hook(state, hook)
    opt = task.make_optimizer(model)

    global_batch = task.bs * ws if a.model == "resnet18" else task.bs * ws
    lr_eff = task.base_lr() * global_batch / a.lr_ref_batch if a.model == "resnet18" else task.base_lr()
    if is0:  # config.json holds CLI args (lr/batch_size may be None = default); record what was actually used
        (run_dir / "config.json").write_text(json.dumps(
            vars(a) | {"world_size": ws, "run_id": run_id,
                       "resolved": {"batch_size_per_gpu": task.bs, "global_batch": global_batch, "lr_eff": lr_eff}}, indent=1))
    warm_ignore = a.timing_warmup if a.timing_warmup is not None else max(20, a.start_iter + 10)
    if is0:
        emit({"type": "start", "run_id": run_id, "params": nparams, "world_size": ws, "total_steps": task.total_steps,
              "lr_eff": lr_eff, "global_batch": global_batch, "device": torch.cuda.get_device_name(local) if cuda else "cpu",
              "torch": torch.__version__, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32})

    def sync():
        if cuda:
            torch.cuda.synchronize()

    step_times, data_times, bytes_per_step = [], [], []
    run_loss, run_n = 0.0, 0
    best, best_step, last_ev = None, None, None
    status = "ok"
    dist.barrier()
    sync()
    t_start = time.perf_counter()
    train_time = 0.0
    try:
        for step in range(task.total_steps):
            for g in opt.param_groups:
                g["lr"] = task.lr_at(step, lr_eff)
            td0 = time.perf_counter()
            batch = task.next_batch(step)
            sync()
            td1 = time.perf_counter()

            prof = None
            if a.profile_step and step == a.profile_step and is0:
                acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
                prof = torch.profiler.profile(activities=acts)
                prof.__enter__()
            b0 = counter.bytes
            opt.zero_grad(set_to_none=True)
            loss = task.loss(model, batch)
            loss.backward()
            opt.step()
            sync()
            td2 = time.perf_counter()
            if prof is not None:
                prof.__exit__(None, None, None)
                (run_dir / "profile_table.txt").write_text(
                    prof.key_averages().table(sort_by="cuda_time_total" if cuda else "cpu_time_total", row_limit=30))
                prof.export_chrome_trace(str(run_dir / "trace_rank0.json"))

            lv = loss.item()
            if not math.isfinite(lv):
                status = "diverged"
                emit({"type": "diverged", "step": step})
                break
            step_times.append(td2 - td1)
            data_times.append(td1 - td0)
            bytes_per_step.append(counter.bytes - b0)
            train_time += td2 - td0
            run_loss += lv
            run_n += 1

            if task.is_eval_step(step):
                # eval on rank 0 only; others wait. Excluded from train_time, included in wall time.
                counter.enabled = False
                if is0:
                    ev = task.evaluate(model.module)
                    last_ev = ev
                    key = "test_acc" if "test_acc" in ev else "val_loss"
                    if best is None or (ev[key] > best[key] if key == "test_acc" else ev[key] < best[key]):
                        best, best_step = ev, step + 1
                    steady = step_times[warm_ignore:] or step_times
                    emit({"type": "eval", "step": step + 1, "epoch": (step + 1) / task.steps_per_epoch if a.model == "resnet18" else None,
                          "train_loss": run_loss / run_n, "train_time_s": train_time, "elapsed_s": time.perf_counter() - t_start,
                          "median_step_s": float(np.median(steady)), **ev})
                run_loss, run_n = 0.0, 0
                dist.barrier()
                counter.enabled = True
    except Exception as e:  # log failure then re-raise
        status = f"error: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        sync()
        wall = time.perf_counter() - t_start
        if is0:
            steady = step_times[warm_ignore:] if len(step_times) > warm_ignore else step_times
            # steady-state bytes: after PowerSGD kicks in
            sb = bytes_per_step[a.start_iter + 2:] or bytes_per_step
            sps = task.samples_per_step(ws)
            np.savetxt(run_dir / "step_times.csv", np.c_[np.arange(len(step_times)), step_times, data_times, bytes_per_step],
                       delimiter=",", header="step,step_time_s,data_time_s,comm_bytes", comments="", fmt=["%d", "%.6f", "%.6f", "%d"])
            rec = {
                "run_id": run_id, "model": a.model, "dataset": task.dataset, "world_size": ws, "nodes": ws // local_ws,
                "gpus_used": ws if cuda else 0, "rank_R": a.rank if a.mode == "powersgd" else None, "mode": a.mode,
                "precision": a.precision, "precision_note": "fp8 emulated (A100 has no FP8)" if a.precision.startswith("fp8") else None,
                "seed": a.seed, "batch_size_per_gpu": task.bs, "global_batch": global_batch,
                "steps": len(step_times), "planned_steps": task.total_steps, "epochs": a.epochs if a.model == "resnet18" else None,
                "start_powerSGD_iter": a.start_iter if a.mode == "powersgd" else None,
                "final_metric": task.final_metric(last_ev) if last_ev else None, "best_metric": best, "best_step": best_step, "lr_eff": lr_eff,
                "wall_time_s": wall, "train_time_s": train_time,
                "median_step_time_s": float(np.median(steady)) if len(steady) else None,
                "mean_step_time_s": float(np.mean(steady)) if len(steady) else None,
                "throughput_per_s": sps / float(np.mean(steady)) if len(steady) else None,
                "throughput_unit": "images/s" if a.model == "resnet18" else "tokens/s",
                "comm_bytes_per_step": float(np.mean(sb)) if len(sb) else None,
                "comm_bytes_note": "logical payload per rank passed to dist.all_reduce, steady state; not wire bytes",
                "status": status, "run_dir": str(run_dir), "params": nparams, "torch": torch.__version__,
            }
            if not cuda:
                rec["note"] = "CPU/gloo run: timing not representative"
            if a.synthetic:
                rec["note"] = "SYNTHETIC DATA smoke test - not a result"
            log_run(rec)
            emit({"type": "done", **{k: rec[k] for k in ("wall_time_s", "median_step_time_s", "comm_bytes_per_step", "status")}})
        dist.barrier() if status == "ok" else None
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
