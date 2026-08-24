#!/usr/bin/env python
"""Chunked hierarchical slop classifier: Magika's trunk, applied per chunk, pooled per document.

This is the synthesis arm. It keeps what the production pipeline gets right (chunk the whole
document, aggregate) and what Magika's architecture gets right (fixed-size padded input, learned
features, global max pool), and drops what only makes sense for file-format detection.

    production      whole doc -> ragged chunks -> 2-4gram TF-IDF -> linear ensemble -> weighted mean
    magika (flat)   one fixed begin/middle/end window -> learned trunk -> one prediction
    this file       whole doc -> constant padded chunks -> shared trunk -> attention pool -> both

Why each choice, and what the evidence was:

  - Constant 192-token chunks, no overlap. Production's chunker solves for a tiling, so chunk size
    varies with document length (164 distinct sizes observed, 115-164 tokens) and it emits a
    redundant trailing chunk on 31% of multi-chunk documents. Constant size removes both. Overlap
    existed to protect n-grams spanning a boundary; the document head sees neighbouring chunks
    anyway, so it starts at zero. 192 is divisible by the reshape group.

  - Cap of 8 chunks. 8 x 192 = 1536 tokens, the same coverage as the flat token arm, so a comparison
    against it isolates chunking and hierarchy from context length. Raise it to see more document.

  - Pad the final chunk rather than back-filling it from earlier text. Back-filling duplicates the
    tail (a 316-token document would repeat 134 of the last chunk's 192 tokens), and duplication is
    measurably harmful: production scores worse on documents whose chunker emitted a redundant tail,
    in every length band. Padding is safe as long as the model can see it, hence the masking below.

  - Frequency-ranked vocabulary. Folding ids modulo a bound makes two *common* tokens share one
    embedding row at random. Ranking by frequency confines collisions to the rare tail.

  - Two heads. The production aggregate is a hand-written mean weighted by |p - threshold|. Here the
    document head learns its own weighting, and keeping the chunk head means disagreement between
    the two levels is available as a signal rather than being averaged away.

Run (torch is not a project dependency; it is injected for this experiment only):

    DEPS="--with torch --with onnx --with onnxscript"
    uv run $DEPS python notebooks/hierarchical_train.py --help
    uv run $DEPS python notebooks/hierarchical_train.py --limit 20000    # smoke
    uv run $DEPS python notebooks/hierarchical_train.py                  # full
    uv run $DEPS python notebooks/hierarchical_train.py --cutmix 0.15    # mixed-authorship arm
"""

from __future__ import annotations

# Asserts here are invariant checks, and docstrings on dunders would be noise.
# ruff: file-ignore[undocumented-magic-method, undocumented-public-init, assert]
import argparse
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from is_it_slop_preprocessing import tokenize
from loguru import logger
from magika_style_train import (
    DATA_DIR,
    per_dataset_f1,
    pick_device,
    score,
    threshold_at_best_f1,
    threshold_at_precision,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

SEED = 42
ROOT_DIR = Path(__file__).parent.parent.resolve()
OUT_DIR = ROOT_DIR / "model_artifacts" / "hierarchical"

# o200k_base ids fit comfortably below this; the lookup table is sized to cover the whole space.
O200K_CEILING = 201_000


@dataclass(frozen=True)
class Config:
    """Everything that changes a trained artifact."""

    chunk_size: int = 192
    max_chunks: int = 8
    """8 x 192 = 1536 tokens, matching the flat token arm's coverage on purpose."""
    min_fill: float = 0.10
    """Drop a trailing chunk this empty; it carries almost no evidence and adds a noisy vote."""
    vocab_size: int = 32768
    """Distinct token rows kept, ranked by corpus frequency. Rare ids collapse onto one OOV row."""
    embed_dim: int = 128
    group: int = 4
    hidden: int = 256
    dropout: float = 0.1
    spatial_dropout: float = 0.1
    chunk_loss_weight: float = 0.3
    """lambda in L_doc + lambda * L_chunk. Chunk labels are the document label copied down, so they
    are a noisy auxiliary signal rather than the objective."""
    cutmix: float = 0.0
    """Fraction of batches that get chunk-level splicing with proportional soft targets."""
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 4
    limit: int | None = None
    seed: int = SEED

    @property
    def pad_id(self) -> int:
        return self.vocab_size + 1

    @property
    def oov_id(self) -> int:
        return self.vocab_size

    @property
    def n_embeddings(self) -> int:
        return self.vocab_size + 2

    @property
    def groups_per_chunk(self) -> int:
        return self.chunk_size // self.group


def artifact_tag(cfg: Config) -> str:
    parts = [f"c{cfg.chunk_size}", f"k{cfg.max_chunks}", f"v{cfg.vocab_size}"]
    if cfg.cutmix > 0:
        parts.append(f"cutmix{cfg.cutmix:g}")
    if cfg.limit is not None:
        parts.append(str(cfg.limit))
    return "-".join(parts)


# --------------------------------------------------------------------------------------------------
# Vocabulary: frequency-ranked, not modulo-folded
# --------------------------------------------------------------------------------------------------


def build_vocab(texts: Sequence[str], cfg: Config, cache: Path) -> np.ndarray:
    """Lookup table mapping raw o200k ids to compact row indices, ranked by corpus frequency.

    Ids outside the top `vocab_size` share a single OOV row, so collisions land on rare tokens
    instead of being distributed at random across common ones.
    """
    if cache.exists():
        lut = np.load(cache)
        logger.info(f"Reusing vocab {cache.name} (top {cfg.vocab_size:,})")
        return lut

    t0 = time.time()
    counts: Counter[int] = Counter()
    step = 20_000
    for start in range(0, len(texts), step):
        for ids in tokenize(list(texts[start : start + step])):
            counts.update(ids)

    lut = np.full(O200K_CEILING, cfg.oov_id, dtype=np.int32)
    for rank, (token_id, _) in enumerate(counts.most_common(cfg.vocab_size)):
        lut[token_id] = rank

    covered = sum(c for _, c in counts.most_common(cfg.vocab_size)) / max(sum(counts.values()), 1)
    logger.info(
        f"Vocab: {len(counts):,} distinct ids -> top {cfg.vocab_size:,} rows "
        f"covering {covered:.2%} of occurrences ({time.time() - t0:.0f}s)"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lut)
    return lut


# --------------------------------------------------------------------------------------------------
# Encoding: document -> [max_chunks, chunk_size] of compact ids
# --------------------------------------------------------------------------------------------------


def chunk_document(ids: np.ndarray, cfg: Config, out: np.ndarray) -> None:
    """Fill `out` ([max_chunks, chunk_size]) with constant-size chunks, padding the remainder.

    Chunks are contiguous and non-overlapping. A trailing chunk below `min_fill` is dropped unless
    it is the only one, since with no overlap it discards at most `min_fill * chunk_size` tokens.
    """
    cs = cfg.chunk_size
    out.fill(cfg.pad_id)
    if len(ids) == 0:
        return

    n_full, remainder = divmod(len(ids), cs)
    n_chunks = n_full + (1 if remainder >= cfg.min_fill * cs else 0)
    n_chunks = max(1, min(n_chunks, cfg.max_chunks))

    for i in range(n_chunks):
        piece = ids[i * cs : (i + 1) * cs]
        out[i, : len(piece)] = piece


def encode_corpus(texts: Sequence[str], cfg: Config, lut: np.ndarray, cache: Path) -> np.ndarray:
    """Encode to a memmapped `(n, max_chunks, chunk_size)` matrix.

    Written to a temporary path and renamed on success: shape alone cannot distinguish a complete
    cache from one abandoned part-way, and a half-written cache would be reused silently.
    """
    shape = (len(texts), cfg.max_chunks, cfg.chunk_size)
    assert cfg.n_embeddings <= np.iinfo(np.uint16).max, "vocab no longer fits uint16"

    if cache.exists():
        existing = np.memmap(cache, dtype=np.uint16, mode="r").reshape(-1, cfg.max_chunks, cfg.chunk_size)
        if existing.shape == shape:
            logger.info(f"Reusing encoding cache {cache.name} {shape}")
            return existing
        logger.warning(f"Cache {cache.name} has shape {existing.shape}, expected {shape}; re-encoding")

    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(cache.suffix + ".partial")
    out = np.memmap(partial, dtype=np.uint16, mode="w+", shape=shape)

    t0 = time.time()
    step = 20_000
    for start in range(0, len(texts), step):
        batch = [t.strip() for t in texts[start : start + step]]
        for j, ids in enumerate(tokenize(batch)):
            chunk_document(lut[np.asarray(ids, dtype=np.int64)], cfg, out[start + j])

    out.flush()
    del out
    partial.replace(cache)
    logger.info(f"Encoded {len(texts):,} docs -> {cache.name} in {time.time() - t0:.1f}s")
    return np.memmap(cache, dtype=np.uint16, mode="r").reshape(shape)


class DocumentDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert len(features) == len(labels), f"{len(features)} features vs {len(labels)} labels"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.features[idx], dtype=np.int64))
        return x, torch.tensor(float(self.labels[idx]), dtype=torch.float32)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------


class SpatialDropout1d(nn.Module):
    """Drop whole feature channels across all positions, as in Magika's spatial dropout."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return self.drop(x).squeeze(-1).permute(0, 2, 1)


class HierarchicalSlopClassifier(nn.Module):
    """Shared per-chunk trunk, a chunk head, and an attention-pooled document head.

    The trunk weights are shared across chunks, so adding chunks costs compute but no parameters.
    Padding is masked out of both pooling steps: a global max over mostly-pad positions would
    otherwise let padding decide the chunk representation.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.chunk_size % cfg.group == 0, f"chunk_size {cfg.chunk_size} not divisible by {cfg.group}"
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.n_embeddings, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.spatial_drop = SpatialDropout1d(cfg.spatial_dropout)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.group * cfg.embed_dim, cfg.hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.chunk_norm = nn.LayerNorm(cfg.hidden)
        self.chunk_head = nn.Linear(cfg.hidden, 1)

        self.attend = nn.Linear(cfg.hidden, 1)
        # Attention-pooled and max-pooled document vectors are concatenated: the first asks which
        # chunks matter, the second preserves "one chunk was decisive".
        self.doc_head = nn.Sequential(
            nn.LayerNorm(2 * cfg.hidden), nn.Dropout(cfg.dropout), nn.Linear(2 * cfg.hidden, 1)
        )

    def encode_chunks(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ids [B, C, L] into chunk vectors [B, C, H] plus a real-chunk mask [B, C]."""
        cfg = self.cfg
        b, c, length = ids.shape
        token_real = ids != cfg.pad_id

        h = self.embed(ids.reshape(b * c, length))
        h = self.spatial_drop(h)
        h = h.reshape(b * c, cfg.groups_per_chunk, cfg.group * cfg.embed_dim)
        h = self.trunk(h)

        # A group is real if any token in it is; -inf elsewhere so padding cannot win the max.
        group_real = token_real.reshape(b * c, cfg.groups_per_chunk, cfg.group).any(dim=-1)
        h = h.masked_fill(~group_real.unsqueeze(-1), float("-inf"))
        h = h.amax(dim=1)
        # A chunk with no real tokens is all -inf; zero it so it cannot pollute the document pool.
        chunk_real = token_real.any(dim=-1)
        h = torch.where(chunk_real.reshape(b * c, 1), h, torch.zeros_like(h))

        # Slot 0 always counts as real. Chunks fill from index 0, so for any document with content
        # this is already true and the clamp is a no-op; it exists so a document that tokenizes to
        # nothing still has one finite slot. Without it the pools reduce over an empty set and the
        # document logit is NaN, which propagates through the batch loss during training.
        # Must be out-of-place: `chunk_real` was already consumed by the `torch.where` above, so
        # mutating it breaks the autograd graph.
        chunk_mask = chunk_real | (torch.arange(c, device=ids.device) == 0)
        return self.chunk_norm(h).reshape(b, c, cfg.hidden), chunk_mask

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce document logits [B], chunk logits [B, C], and the real-chunk mask [B, C]."""
        h, chunk_real = self.encode_chunks(ids)
        chunk_logits = self.chunk_head(h).squeeze(-1)

        weights = self.attend(h).squeeze(-1).masked_fill(~chunk_real, float("-inf"))
        weights = torch.softmax(weights, dim=1)
        attended = (weights.unsqueeze(-1) * h).sum(dim=1)

        pooled_max = h.masked_fill(~chunk_real.unsqueeze(-1), float("-inf")).amax(dim=1)
        doc_logits = self.doc_head(torch.cat([attended, pooled_max], dim=-1)).squeeze(-1)
        return doc_logits, chunk_logits, chunk_real


# --------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------


def chunk_cutmix(ids: torch.Tensor, y: torch.Tensor, cfg: Config, generator: torch.Generator) -> tuple:
    """Splice whole chunks between two documents; the target becomes the proportion of AI chunks.

    Applied at chunk granularity rather than token granularity because the chunk grid is already the
    unit the document head pools over, so the proportional label is exact rather than approximate.
    Produces the partially-AI documents the corpus does not contain.
    """
    b, c, _ = ids.shape
    partner = torch.randperm(b, generator=generator, device=ids.device)
    real = (ids != cfg.pad_id).any(dim=-1)

    # Take a contiguous run of chunk slots from the partner document.
    cut = torch.randint(1, c, (1,), generator=generator, device=ids.device).item()
    mixed = ids.clone()
    mixed[:, cut:, :] = ids[partner][:, cut:, :]

    real_mixed = (mixed != cfg.pad_id).any(dim=-1)
    own = (real[:, :cut].sum(dim=1)).float()
    other = (real_mixed[:, cut:].sum(dim=1)).float()
    total = (own + other).clamp(min=1.0)
    y_mixed = (y * own + y[partner] * other) / total
    return mixed, y_mixed


def losses(
    doc_logits: torch.Tensor, chunk_logits: torch.Tensor, chunk_real: torch.Tensor, y: torch.Tensor, cfg: Config
) -> torch.Tensor:
    """Document loss plus a masked, down-weighted chunk loss over real chunks only."""
    bce = nn.functional.binary_cross_entropy_with_logits
    doc_loss = bce(doc_logits, y)

    per_chunk = bce(chunk_logits, y.unsqueeze(1).expand_as(chunk_logits), reduction="none")
    chunk_loss = (per_chunk * chunk_real).sum() / chunk_real.sum().clamp(min=1)
    return doc_loss + cfg.chunk_loss_weight * chunk_loss


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        doc_logits, _, _ = model(x.to(device))
        out.append(torch.sigmoid(doc_logits).float().cpu().numpy())
    return np.concatenate(out)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    """Train with early stopping on validation AUC, keeping the best weights seen."""
    from sklearn.metrics import roc_auc_score  # ruff: ignore[import-outside-top-level]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    best_auc, stale = -1.0, 0
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # ruff: ignore[redefined-loop-name]
            if cfg.cutmix > 0 and torch.rand(1, generator=generator).item() < cfg.cutmix:
                x, y = chunk_cutmix(x, y, cfg, generator)  # ruff: ignore[redefined-loop-name]
            opt.zero_grad()
            loss = losses(*model(x), y, cfg)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)

        auc = roc_auc_score(y_val, predict(model, val_loader, device))
        logger.info(f"epoch {epoch:>3}  loss {total / seen:.4f}  val_auc {auc:.4f}  ({time.time() - t0:.0f}s)")

        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info(f"early stop at epoch {epoch}; best val_auc {best_auc:.4f}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


class DocumentOnly(nn.Module):
    """Export wrapper exposing just the document probability, which is what inference consumes."""

    def __init__(self, model: HierarchicalSlopClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        doc_logits, _, _ = self.model(chunks)
        return torch.sigmoid(doc_logits)


def export_hierarchical_onnx(model: nn.Module, cfg: Config, path: Path) -> None:
    """Export to a single self-contained ONNX file taking [batch, max_chunks, chunk_size].

    `external_data=False` matters: the default spills weights to a `.onnx.data` sidecar, and
    `crates/is-it-slop/build.rs` embeds one ONNX file into the binary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    dummy = torch.full((1, cfg.max_chunks, cfg.chunk_size), cfg.pad_id, dtype=torch.long)
    dummy[0, 0, :] = 0  # at least one real chunk, so masking has a live path to trace
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["chunks"],
        output_names=["ai_probability"],
        dynamic_axes={"chunks": {0: "batch"}, "ai_probability": {0: "batch"}},
        opset_version=18,
        external_data=False,
    )
    sidecars = list(path.parent.glob(f"{path.name}.data"))
    assert not sidecars, f"weights spilled to {sidecars}; the artifact is not self-contained"
    logger.info(f"ONNX -> {path} ({path.stat().st_size / 1024:.0f} KB, self-contained)")


# --------------------------------------------------------------------------------------------------


def load_split(name: str, limit: int | None) -> pl.DataFrame:
    df = pl.scan_parquet(DATA_DIR / f"{name}.parquet").select("text", "label", "dataset").collect()
    if limit and limit < len(df):
        df = df.sample(n=limit, seed=SEED, shuffle=True)
    logger.info(
        f"{name:11s} {len(df):>8,} rows  human={int((df['label'] == 0).sum()):,} ai={int((df['label'] == 1).sum()):,}"
    )
    return df


def length_report(texts: Sequence[str], y: np.ndarray, probs: np.ndarray, threshold: float) -> pl.DataFrame:
    """Accuracy by document length, the axis that separated every arm measured so far."""
    n_tokens = np.array([len(t) for t in tokenize([t.strip() for t in texts])])
    hit = (probs >= threshold).astype(np.int8) == y
    edges = (0, 50, 100, 150, 200, 300, 500, 1000, 2000, np.iinfo(np.int64).max)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (n_tokens >= lo) & (n_tokens < hi)
        if mask.sum() < 20:
            continue
        rows.append({
            "tokens": f"{lo}-{'inf' if hi == edges[-1] else hi}",
            "n": int(mask.sum()),
            "accuracy": round(float(hit[mask].mean()), 4),
        })
    return pl.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=Config.max_chunks,
        help="8 matches the flat token arm's 1536-token coverage; 16 covers 98.8%% of documents whole",
    )
    p.add_argument("--vocab-size", type=int, default=Config.vocab_size)
    p.add_argument("--embed-dim", type=int, default=Config.embed_dim, help="halving this halves the artifact")
    p.add_argument("--chunk-loss-weight", type=float, default=Config.chunk_loss_weight)
    p.add_argument(
        "--cutmix",
        type=float,
        default=Config.cutmix,
        help="fraction of batches spliced for mixed-authorship targets; 0 keeps the run comparable to the other arms",
    )
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--limit", type=int, default=None, help="rows per split, for smoke runs")
    p.add_argument("--target-precision", type=float, default=0.99)
    args = p.parse_args()

    cfg = Config(
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        chunk_loss_weight=args.chunk_loss_weight,
        cutmix=args.cutmix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
    )
    torch.manual_seed(cfg.seed)
    device = pick_device()
    tag = artifact_tag(cfg)
    logger.info(f"device={device}  {tag}  coverage={cfg.chunk_size * cfg.max_chunks:,} tokens")

    splits = {name: load_split(name, cfg.limit) for name in ("train", "test", "validation")}
    labels = {name: df["label"].to_numpy() for name, df in splits.items()}

    cache_dir = DATA_DIR / "hierarchical_cache"
    # Vocabulary is derived from the training split only; deriving it from test would leak.
    lut = build_vocab(splits["train"]["text"].to_list(), cfg, cache_dir / f"vocab-v{cfg.vocab_size}.npy")
    features = {
        name: encode_corpus(df["text"].to_list(), cfg, lut, cache_dir / f"{name}-{tag}.npy")
        for name, df in splits.items()
    }

    loaders = {
        name: DataLoader(
            DocumentDataset(features[name], labels[name]),
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            drop_last=(name == "train"),
        )
        for name in splits
    }

    model = HierarchicalSlopClassifier(cfg).to(device)
    params = sum(t.numel() for t in model.parameters())
    embed_params = cfg.n_embeddings * cfg.embed_dim
    logger.info(
        f"parameters: {params:,} ({params * 4 / 1024 / 1024:.2f} MB) of which "
        f"embedding {embed_params:,} ({100 * embed_params / params:.0f}%)"
    )

    t0 = time.time()
    model = train(model, loaders["train"], loaders["validation"], labels["validation"], cfg, device)
    train_seconds = time.time() - t0

    val_probs = predict(model, loaders["validation"], device)
    thresholds = {
        f"precision_{args.target_precision}": threshold_at_precision(
            labels["validation"], val_probs, args.target_precision
        ),
        "best_f1": threshold_at_best_f1(labels["validation"], val_probs),
    }
    logger.info(f"thresholds fitted on validation: {thresholds}")

    test_probs = predict(model, loaders["test"], device)
    results = {
        "config": cfg.__dict__,
        "parameters": params,
        "train_seconds": train_seconds,
        "validation": {k: score(labels["validation"], val_probs, t) for k, t in thresholds.items()},
        "test": {k: score(labels["test"], test_probs, t) for k, t in thresholds.items()},
    }
    for split_name in ("validation", "test"):
        for rule, metrics in results[split_name].items():
            logger.info(f"{split_name:10s} [{rule}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / f"hierarchical-{tag}.onnx"
    export_hierarchical_onnx(DocumentOnly(model), cfg, onnx_path)

    best = thresholds["best_f1"]
    breakdown = per_dataset_f1(splits["test"]["dataset"].cast(pl.Utf8).to_list(), labels["test"], test_probs, best)
    lengths = length_report(splits["test"]["text"].to_list(), labels["test"], test_probs, best)
    with pl.Config(tbl_rows=40, tbl_width_chars=140):
        logger.info(f"per-dataset F1 on test:\n{breakdown}")
        logger.info(f"accuracy by document length:\n{lengths}")

    breakdown.write_csv(OUT_DIR / f"per-dataset-{tag}.csv")
    lengths.write_csv(OUT_DIR / f"per-length-{tag}.csv")
    np.savez_compressed(
        OUT_DIR / f"test-probs-{tag}.npz",
        hierarchical=test_probs,
        label=labels["test"],
        dataset=np.asarray(splits["test"]["dataset"].cast(pl.Utf8).to_list()),
    )
    (OUT_DIR / f"results-{tag}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    torch.save(model.state_dict(), OUT_DIR / f"hierarchical-{tag}.pt")
    logger.info(f"artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
