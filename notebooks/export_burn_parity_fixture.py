#!/usr/bin/env python
"""Export a trained bundle's weights and a golden-vector fixture for the Burn parity test.

The Burn port of the model (`crates/is-it-slop-training/src/model.rs`) is asserted against the
PyTorch model it replaces: same weights, same inputs, same probabilities. This script produces the
two safetensors files that test consumes, from the smallest campaign bundle so the checked-in
fixture stays under 2 MB.

    uv run --with torch --with safetensors python notebooks/export_burn_parity_fixture.py

Re-run only when the reference bundle or the fixture documents change; the outputs are committed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).parent.parent.resolve()
# Trained bundles and caches live in the main checkout; a worktree can point back at it.
ARTIFACTS = Path(os.environ.get("ARTIFACTS_ROOT", ROOT)).resolve()
BUNDLE = ARTIFACTS / "model_artifacts" / "hierarchical" / "c192-k8-v8192-d48-g2-wd0.01-p6-cos"
CACHE = ARTIFACTS / "data" / "3.0.0" / "hierarchical_cache" / "test-c192-k8-v8192.npy"
OUT = ROOT / "crates" / "is-it-slop-training" / "tests" / "fixtures"

N_DOCS = 16
MAX_CHUNKS = 8
CHUNK_SIZE = 192
PAD_ID = 8193  # vocab 8192: oov row 8192, pad row 8193


def main() -> None:
    from hierarchical_train import Config, HierarchicalSlopClassifier

    cfg = Config(vocab_size=8192, embed_dim=48, group=2)
    model = HierarchicalSlopClassifier(cfg)
    state = torch.load(BUNDLE / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Spread over the split so lengths vary; the first document is truncated to one real chunk so
    # the single-chunk path is covered.
    feats = np.memmap(CACHE, dtype=np.uint16, mode="r").reshape(-1, MAX_CHUNKS, CHUNK_SIZE)
    idx = np.linspace(0, len(feats) - 1, N_DOCS, dtype=np.int64)
    ids = np.asarray(feats[idx], dtype=np.int64).copy()
    ids[0, 1:, :] = PAD_ID

    with torch.no_grad():
        doc_logits, chunk_logits, _ = model(torch.from_numpy(ids))
        probs = torch.sigmoid(doc_logits)

    OUT.mkdir(parents=True, exist_ok=True)
    save_file(state, OUT / "weights.safetensors")
    save_file(
        {
            "ids": torch.from_numpy(ids),
            # Raw logits are the sharp comparison: sigmoid saturates near 0/1 and would hide
            # drift the probabilities can't show.
            "doc_logits": doc_logits.float(),
            "doc_probs": probs.float(),
            "chunk_logits": chunk_logits.float(),
        },
        OUT / "golden.safetensors",
    )
    print(f"weights -> {OUT / 'weights.safetensors'} ({(OUT / 'weights.safetensors').stat().st_size / 1e6:.2f} MB)")
    print(f"golden  -> {OUT / 'golden.safetensors'} ({(OUT / 'golden.safetensors').stat().st_size / 1e6:.2f} MB)")
    print("doc_probs:", [round(float(p), 6) for p in probs])


if __name__ == "__main__":
    main()
