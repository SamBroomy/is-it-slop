"""Torch-MPS mirror of the Rust step_bench (crates/is-it-slop-training/examples/step_bench.rs).

Raw loop over one resident synthetic batch, same protocol as the Rust side: frozen recipe config
(v16384-d64-g2, wd 0.01), batch [128, 8, 192], train mode, AdamW with rank-gated decay groups,
per-iteration loss.item() sync, 50 warmup + 300 timed (STEP_BENCH_WARMUP/ITERS/ONLY override).
Never run the two sides concurrently — they contend for the GPU and inflate each other.
"""

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hierarchical_train import Config, HierarchicalSlopClassifier, losses

WARMUP = int(os.environ.get("STEP_BENCH_WARMUP", "50"))
ITERS = int(os.environ.get("STEP_BENCH_ITERS", "300"))

cfg = Config(vocab_size=16384, embed_dim=64, group=2, weight_decay=0.01)
device = torch.device("mps")
torch.manual_seed(42)

model = HierarchicalSlopClassifier(cfg).to(device)
model.train()
params = sum(t.numel() for t in model.parameters())
print(f"backend=torch-mps v{cfg.vocab_size}-d{cfg.embed_dim} batch={cfg.batch_size} params={params:,}")

x_cpu = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_chunks, cfg.chunk_size))
x = x_cpu.to(device)
y = torch.tensor([i % 2 for i in range(cfg.batch_size)], dtype=torch.float32, device=device)
y_chunk = y.unsqueeze(1).expand(-1, cfg.max_chunks)

decay = [p for p in model.parameters() if p.ndim >= 2]
rest = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": rest, "weight_decay": 0.0}], lr=cfg.lr
)


def bench(label: str, fn: Callable[[], None]) -> None:
    for _ in range(WARMUP):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    per_iter = (time.perf_counter() - t0) * 1000 / ITERS
    print(f"{label:<10} {per_iter:>8.2f} ms/iter  ({per_iter:.1f} s/1000)")


def fwd() -> None:
    loss = losses(*model(x), y, y_chunk, cfg)
    loss.item()


def fwd_bwd() -> None:
    opt.zero_grad()
    loss = losses(*model(x), y, y_chunk, cfg)
    loss.backward()
    loss.item()


def step() -> None:
    opt.zero_grad()
    loss = losses(*model(x), y, y_chunk, cfg)
    loss.backward()
    opt.step()
    loss.item()


def step_pull() -> None:
    opt.zero_grad()
    doc_logits, chunk_logits, chunk_real = model(x)
    loss = losses(doc_logits, chunk_logits, chunk_real, y, y_chunk, cfg)
    loss.backward()
    opt.step()
    doc_logits.detach().cpu()
    y.cpu()
    loss.item()


def step_h2d() -> None:
    opt.zero_grad()
    fresh = x_cpu.to(device)
    loss = losses(*model(fresh), y, y_chunk, cfg)
    loss.backward()
    opt.step()
    loss.item()


def fwd_inner() -> None:
    with torch.no_grad():
        doc_logits, _, _ = model(x)
        doc_logits.mean().item()


if os.environ.get("STEP_BENCH_ONLY") == "fwd_inner":
    model.eval()
    bench("fwd_inner", fwd_inner)
    sys.exit(0)

bench("fwd", fwd)
bench("fwd_bwd", fwd_bwd)
bench("step", step)
bench("step+pull", step_pull)
bench("step+h2d", step_h2d)
