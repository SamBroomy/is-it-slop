#!/usr/bin/env python
"""EXPERIMENTAL: a Magika-shaped neural classifier, trained on the same data as the production model.

Reference: Fratantonio et al., "Magika: AI-Powered Content-Type Detection", ICSE 2025, Section V-B.

The point of this script is a controlled comparison against the production TF-IDF + stacked-linear
pipeline in `train.py`. Both read the identical released parquets, so the only thing that varies is
how features are formed:

    train.py  : whole document -> overlapping chunks -> enumerated 2-4gram TF-IDF -> linear ensemble
    this file : one fixed window (begin/middle/end) -> learned embedding -> MLP -> global max pool

Magika's trunk is a *learned* n-gram detector: global max pooling over position-wise dense layers
fires on the strongest local pattern anywhere in the window, position-invariantly. So this is not
"classic ML vs deep learning" so much as enumerated features vs learned ones over the same signal.

Two deliberate departures from `train.py`, both so the arms are measured the same way:
  - No chunking. Magika's premise is a single constant-size input, which is what buys it
    constant-time inference. Document-level probability comes straight out of the model.
  - The decision threshold is fitted on `validation.parquet` and only then applied to
    `test.parquet`. `train.py:410` currently fits it on the test set it reports.

Run (torch is not a project dependency; it is injected for this experiment only):

    DEPS="--with torch --with onnx --with onnxscript"
    uv run $DEPS python notebooks/magika_style_train.py --help
    uv run $DEPS python notebooks/magika_style_train.py --limit 20000   # smoke
    uv run $DEPS python notebooks/magika_style_train.py                 # full
    uv run $DEPS python notebooks/magika_style_train.py --encoding tokens
"""

from __future__ import annotations

# Asserts here are invariant checks, and docstrings on dunders would be noise.
# ruff: file-ignore[undocumented-magic-method, undocumented-public-init, assert]
import argparse
import json
import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.export import Dim
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

SEED = 42
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Pinned to the released dataset artifact rather than inherited from notebooks/__init__.py, whose
# RETRAIN_VECTORIZER force-bump points DATA_DIR at a version that does not exist on disk.
DATA_DIR = ROOT_DIR / "data" / "3.0.0"
OUT_DIR = ROOT_DIR / "model_artifacts" / "magika_style"

# Magika encodes each byte as an integer in [0, 255] and reserves 256 as the padding symbol.
BYTE_PAD = 256
BYTE_VOCAB = 257

Encoding = Literal["bytes", "tokens"]


@dataclass(frozen=True)
class Config:
    """Everything that changes a trained artifact. Serialised next to the model for provenance."""

    encoding: Encoding = "bytes"
    window: int = 512
    """Symbols taken from each of the beginning, middle and end. Magika uses 512."""
    embed_dim: int = 128
    group: int = 4
    """Adjacent positions folded into one entity before the trunk. Magika's Reshape uses 4."""
    hidden: int = 256
    dropout: float = 0.1
    spatial_dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 4
    max_vocab: int = 32768
    """Token encoding only. o200k_base has ~200k entries; a full embedding would be ~100MB, so ids
    are folded modulo this bound. Ignored when encoding='bytes'."""
    limit: int | None = None
    seed: int = SEED

    @property
    def seq_len(self) -> int:
        return 3 * self.window

    @property
    def vocab(self) -> int:
        return BYTE_VOCAB if self.encoding == "bytes" else self.max_vocab + 1

    @property
    def pad_id(self) -> int:
        return BYTE_PAD if self.encoding == "bytes" else self.max_vocab


def artifact_tag(cfg: Config) -> str:
    """Name caches and artifacts after the settings that change their contents.

    The window is only included when it differs from the default, so runs made before it was
    configurable keep their existing filenames.
    """
    parts = [cfg.encoding]
    if cfg.window != Config.window:
        parts.append(f"w{cfg.window}")
    if cfg.limit is not None:
        parts.append(str(cfg.limit))
    return "-".join(parts)


# --------------------------------------------------------------------------------------------------
# Encoding: document -> one fixed-size window
# --------------------------------------------------------------------------------------------------


def _place(dst: np.ndarray, symbols: np.ndarray, *, align: Literal["left", "right"]) -> None:
    """Write `symbols` into `dst`, leaving the untouched remainder as padding."""
    n = min(len(symbols), dst.shape[0])
    if n == 0:
        return
    if align == "left":
        dst[:n] = symbols[:n]
    else:
        dst[-n:] = symbols[len(symbols) - n :]


def encode_one(symbols: np.ndarray, cfg: Config, out: np.ndarray) -> None:
    """Fill `out` (length `cfg.seq_len`) with the begin/middle/end windows of `symbols`.

    Mirrors Magika: the beginning block is left-aligned, the end block is right-aligned so it really
    is the tail of the document, and the middle block is centred with no randomness. Short documents
    are padded rather than truncated, which is why the model's input is constant-size.
    """
    w = cfg.window
    out.fill(cfg.pad_id)
    total = len(symbols)

    _place(out[:w], symbols[:w], align="left")

    if total <= w:
        # Whole document already fits the beginning block; middle and end add nothing but padding.
        _place(out[2 * w :], symbols, align="right")
        return

    mid_start = max(0, (total - w) // 2)
    _place(out[w : 2 * w], symbols[mid_start : mid_start + w], align="left")
    _place(out[2 * w :], symbols[-w:], align="right")


def encode_corpus(texts: Sequence[str], cfg: Config, cache: Path) -> np.ndarray:
    """Encode `texts` into a memmapped `(n, seq_len)` matrix, reusing `cache` when it already fits.

    Held on disk because the full training split is 473,848 x 1536; materialising that in RAM
    alongside torch is avoidable waste.

    Written to a temporary path and renamed on success. Shape alone cannot distinguish a complete
    cache from one abandoned part-way, and a half-written cache would be silently reused forever.
    """
    dtype = np.uint16 if cfg.encoding == "bytes" else np.int32
    shape = (len(texts), cfg.seq_len)

    if cache.exists():
        existing = np.memmap(cache, dtype=dtype, mode="r").reshape(-1, cfg.seq_len)
        if existing.shape == shape:
            logger.info(f"Reusing encoding cache {cache.name} {shape}")
            return existing
        logger.warning(f"Cache {cache.name} has shape {existing.shape}, expected {shape}; re-encoding")

    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(cache.suffix + ".partial")
    out = np.memmap(partial, dtype=dtype, mode="w+", shape=shape)

    t0 = time.time()
    if cfg.encoding == "bytes":
        for i, text in enumerate(texts):
            # Magika strips surrounding whitespace before selecting bytes so the window carries
            # content rather than indentation. frombuffer views the bytes as uint8 without a copy.
            raw = text.strip().encode("utf-8", errors="ignore")
            encode_one(np.frombuffer(raw, dtype=np.uint8), cfg, out[i])
    else:
        from is_it_slop_preprocessing import tokenize  # ruff: ignore[import-outside-top-level]

        # Batched so the Rust tokenizer parallelises, but written straight through to the memmap.
        step = 20_000
        for start in range(0, len(texts), step):
            batch = [t.strip() for t in texts[start : start + step]]
            for j, ids in enumerate(tokenize(batch)):
                encode_one(np.asarray(ids, dtype=np.int64) % cfg.max_vocab, cfg, out[start + j])

    out.flush()
    del out
    partial.replace(cache)
    logger.info(f"Encoded {len(texts):,} docs -> {cache.name} in {time.time() - t0:.1f}s")
    return np.memmap(cache, dtype=dtype, mode="r").reshape(shape)


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        assert len(features) == len(labels), f"{len(features)} features vs {len(labels)} labels"
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.features[idx], dtype=np.int64))
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------


class SpatialDropout1d(nn.Module):
    """Drop whole feature channels across all positions, as in Magika's spatial dropout."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T, 1) so Dropout2d zeroes channels, not positions.
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return self.drop(x).squeeze(-1).permute(0, 2, 1)


class MagikaStyleClassifier(nn.Module):
    """Magika's architecture with a 2-way head.

    Embedding -> group adjacent positions -> two dense+gelu layers -> global max pool -> logits.
    LayerNorm stays active at inference (it carries learned parameters); only the dropouts are
    training-only.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.seq_len % cfg.group == 0, f"seq_len {cfg.seq_len} not divisible by group {cfg.group}"
        self.cfg = cfg
        grouped_dim = cfg.group * cfg.embed_dim

        self.embed = nn.Embedding(cfg.vocab, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.spatial_drop = SpatialDropout1d(cfg.spatial_dropout)
        self.trunk = nn.Sequential(
            nn.Linear(grouped_dim, cfg.hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spatial_drop(self.embed(x))
        # Fold groups of `group` positions into single entities: (B, T, E) -> (B, T/g, g*E).
        h = h.reshape(h.shape[0], self.cfg.seq_len // self.cfg.group, -1)
        h = self.trunk(h)
        h = h.amax(dim=1)  # global max pool over position
        return self.head(h)


# --------------------------------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------------------------------


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        logits = model(x.to(device))
        out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
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
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_auc = -1.0
    best_state: dict[str, torch.Tensor] = {}
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # ruff: ignore[redefined-loop-name]
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)

        auc = roc_auc_score(y_val, predict_proba(model, val_loader, device))
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


def threshold_at_precision(y: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Lowest threshold whose precision on the AI class still clears `target`, maximising recall.

    Magika's rule (Section V-D): fix precision, take the best remaining recall. For slop detection
    the class worth protecting is human text, so precision on 'AI' is the constraint that matters --
    it bounds how often a human is wrongly accused.
    """
    precision, recall, thresholds = precision_recall_curve(y, probs)
    # precision_recall_curve returns len(thresholds) + 1 points; drop the trailing sentinel.
    ok = np.where(precision[:-1] >= target)[0]
    if len(ok) == 0:
        logger.warning(f"No threshold reaches precision {target:.2f}; falling back to best F1")
        return threshold_at_best_f1(y, probs)
    return float(thresholds[ok[np.argmax(recall[:-1][ok])]])


def threshold_at_best_f1(y: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
    return float(thresholds[min(int(np.argmax(f1)), len(thresholds) - 1)])


def score(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc": float(roc_auc_score(y, probs)),
    }


def per_dataset_f1(datasets: Sequence[str], y: np.ndarray, probs: np.ndarray, threshold: float) -> pl.DataFrame:
    """F1 broken out by source dataset -- the analogue of Magika's per-content-type table.

    A single headline number over 27 sources of wildly different size hides where a model actually
    fails, which is the reporting habit the paper argues against.
    """
    pred = (probs >= threshold).astype(np.int8)
    df = pl.DataFrame({"dataset": list(datasets), "y": y, "pred": pred})
    rows = []
    for (name,), grp in df.group_by(["dataset"], maintain_order=True):
        gy, gp = grp["y"].to_numpy(), grp["pred"].to_numpy()
        rows.append({
            "dataset": str(name),
            "n": len(gy),
            "human": int((gy == 0).sum()),
            "ai": int((gy == 1).sum()),
            "accuracy": float(accuracy_score(gy, gp)),
            # F1 is undefined for a source that contains only one class; accuracy still is not.
            "f1": float(f1_score(gy, gp, zero_division=0)) if len(set(gy.tolist())) > 1 else float("nan"),
        })
    return pl.DataFrame(rows).sort("n", descending=True)


# --------------------------------------------------------------------------------------------------
# Export & latency
# --------------------------------------------------------------------------------------------------


@contextmanager
def quiet_exporter() -> Iterator[None]:
    """Mute torch's own export chatter, which is noise the caller cannot act on.

    Two sources: the exporter logs a line per unregistered torchvision op (we have no torchvision and
    need none), and torch.export's internals raise a FutureWarning about their own `LeafSpec` usage.
    Scoped to the export call so nothing else in the run is silenced.
    """
    registration = logging.getLogger("torch.onnx._internal.exporter._registration")
    previous = registration.level
    registration.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
            yield
    finally:
        registration.setLevel(previous)


def export_onnx(model: nn.Module, cfg: Config, path: Path) -> None:
    """Export to a single self-contained ONNX file.

    `external_data=False` matters: the default spills weights to a `.onnx.data` sidecar, and
    `crates/is-it-slop/build.rs` embeds one ONNX file into the binary. A sidecar would silently
    produce a weightless model at inference.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    dummy = torch.zeros(1, cfg.seq_len, dtype=torch.long)
    with quiet_exporter():
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["window"],
            output_names=["logits"],
            # dynamic_shapes rather than dynamic_axes: the dynamo exporter warns the older form may
            # violate its constraints. Keyed by the forward parameter name; the window length is
            # fixed by the trained input grid.
            dynamic_shapes={"x": {0: Dim("batch", min=1)}},
            opset_version=18,
            external_data=False,
        )
    sidecars = list(path.parent.glob(f"{path.name}.data"))
    assert not sidecars, f"weights spilled to {sidecars}; the artifact is not self-contained"
    logger.info(f"ONNX -> {path} ({path.stat().st_size / 1024:.0f} KB, self-contained)")


def onnx_latency(path: Path, cfg: Config, n: int = 1130, batch: int = 64) -> dict[str, float]:
    """Single-sample and batched CPU latency, following the paper's protocol (Section VI-B).

    n=1130 matches Magika's timing subset so the numbers are read the same way.

    The batched pass is chunked rather than one giant call: the embedding expands each input by
    `embed_dim` floats per position, so a single 1130-row call allocates ~900MB of intermediate and
    measures memory bandwidth instead of inference. 64 rows keeps it compute-bound.
    """
    import onnxruntime as ort  # ruff: ignore[import-outside-top-level]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(cfg.seed)
    x = rng.integers(0, cfg.vocab, size=(n, cfg.seq_len), dtype=np.int64)

    for _ in range(3):  # warm up, as the paper does
        sess.run(None, {"window": x[:8]})

    t0 = time.time()
    for i in range(n):
        sess.run(None, {"window": x[i : i + 1]})
    single = (time.time() - t0) / n * 1000

    t0 = time.time()
    for start in range(0, n, batch):
        sess.run(None, {"window": x[start : start + batch]})
    batched = (time.time() - t0) / n * 1000

    logger.info(f"latency (1 CPU thread): {single:.2f}ms per sample alone, {batched:.3f}ms amortised at batch={batch}")
    return {"ms_single": single, "ms_batched": batched, "batch": batch}


# --------------------------------------------------------------------------------------------------


def load_split(name: str, limit: int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(DATA_DIR / f"{name}.parquet").select("text", "label", "dataset")
    if limit:
        # Deterministic subsample; shuffled so the cut is not one contiguous source.
        lf = lf.collect().sample(n=min(limit, lf.select(pl.len()).collect()[0, 0]), seed=SEED, shuffle=True).lazy()
    df = lf.collect()
    logger.info(
        f"{name:11s} {len(df):>8,} rows  human={int((df['label'] == 0).sum()):,} ai={int((df['label'] == 1).sum()):,}"
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--encoding",
        choices=["bytes", "tokens"],
        default="bytes",
        help="bytes is faithful to the paper and keeps the model ~1MB; tokens reuses the "
        "production tiktoken pipeline but needs a folded vocab",
    )
    p.add_argument(
        "--window",
        type=int,
        default=Config.window,
        help="symbols per block (3 blocks total). Text is ~4.7 bytes per BPE token, so equal windows "
        "do NOT mean equal text coverage: at 512, the byte arm truncates 48%% of documents and the "
        "token arm 5%%. To compare representations rather than context length, match the coverage "
        "(e.g. --encoding tokens --window 328 against --encoding bytes --window 512)",
    )
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--limit", type=int, default=None, help="rows per split, for smoke runs")
    p.add_argument(
        "--target-precision",
        type=float,
        default=0.99,
        help="precision on the AI class the threshold must hold (Magika's rule)",
    )
    # Default 0: the windows are already encoded on disk, so __getitem__ is a memmap read and a
    # cast. Worker processes would add no throughput, and macOS spawns rather than forks, which
    # pickles the memmap into a real array in every worker.
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    cfg = Config(
        encoding=args.encoding,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
    )
    torch.manual_seed(cfg.seed)
    np.random.default_rng(cfg.seed)

    device = pick_device()
    logger.info(f"device={device}  encoding={cfg.encoding}  vocab={cfg.vocab}  seq_len={cfg.seq_len}")

    splits = {name: load_split(name, cfg.limit) for name in ("train", "test", "validation")}

    tag = artifact_tag(cfg)
    cache_dir = DATA_DIR / "magika_cache"
    features = {
        name: encode_corpus(df["text"].to_list(), cfg, cache_dir / f"{name}-{tag}.npy") for name, df in splits.items()
    }
    labels = {name: df["label"].to_numpy() for name, df in splits.items()}

    loaders = {
        name: DataLoader(
            WindowDataset(features[name], labels[name]),
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=args.workers,
            drop_last=False,
        )
        for name in splits
    }

    model = MagikaStyleClassifier(cfg).to(device)
    params = sum(p_.numel() for p_ in model.parameters())
    logger.info(f"parameters: {params:,} ({params * 4 / 1024 / 1024:.2f} MB as float32)")

    t0 = time.time()
    model = train(model, loaders["train"], loaders["validation"], labels["validation"], cfg, device)
    train_seconds = time.time() - t0

    # Threshold from validation only, then applied unchanged to test.
    val_probs = predict_proba(model, loaders["validation"], device)
    thresholds = {
        f"precision_{args.target_precision}": threshold_at_precision(
            labels["validation"], val_probs, args.target_precision
        ),
        "best_f1": threshold_at_best_f1(labels["validation"], val_probs),
    }
    logger.info(f"thresholds fitted on validation: {thresholds}")

    test_probs = predict_proba(model, loaders["test"], device)
    results = {
        "config": cfg.__dict__,
        "parameters": params,
        "train_seconds": train_seconds,
        "validation": {k: score(labels["validation"], val_probs, t) for k, t in thresholds.items()},
        "test": {k: score(labels["test"], test_probs, t) for k, t in thresholds.items()},
    }

    for split_name, table in results.items():
        if split_name in {"validation", "test"}:
            for rule, metrics in table.items():
                logger.info(f"{split_name:10s} [{rule}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / f"magika-style-{tag}.onnx"
    export_onnx(model, cfg, onnx_path)
    results["latency"] = onnx_latency(onnx_path, cfg)

    breakdown = per_dataset_f1(
        splits["test"]["dataset"].cast(pl.Utf8).to_list(), labels["test"], test_probs, thresholds["best_f1"]
    )
    with pl.Config(tbl_rows=40, tbl_width_chars=120):
        logger.info(f"per-dataset F1 on test (best_f1 threshold):\n{breakdown}")
    breakdown.write_csv(OUT_DIR / f"per-dataset-{tag}.csv")

    (OUT_DIR / f"results-{tag}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    torch.save(model.state_dict(), OUT_DIR / f"magika-style-{tag}.pt")
    logger.info(f"artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
