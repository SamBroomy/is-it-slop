#!/usr/bin/env python
"""Run the production TF-IDF ensemble and the Magika-shaped net side by side on the same test split.

Both models are scored through onnxruntime on one CPU thread, so accuracy and latency are measured
by the same harness rather than compared across published numbers.

Three things this fixes relative to the numbers in `model_metadata.json`:

  - Real AUC. Every `roc_auc_score` call in train.py (lines 428, 561, 650, 689) is handed hard 0/1
    predictions rather than probabilities, so the published "AUC" is balanced accuracy. Here the
    production model's ranking quality is measured for the first time.
  - A threshold fitted on validation and only then applied to test, for both arms. train.py fits on
    the test set it reports, which flatters it.
  - Latency including preprocessing. The production path must tokenize, chunk, and materialise a
    105,190-float dense vector per chunk; the Magika path builds one 1,536-int window per document.
    Excluding that would hide the actual cost difference.

Run:

    DEPS="--with torch --with onnx --with onnxscript"
    uv run $DEPS python notebooks/compare_models.py
    uv run $DEPS python notebooks/compare_models.py --limit 20000   # faster pass
"""

from __future__ import annotations

# Asserts here are invariant checks.
# ruff: file-ignore[assert]
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, tokenize
from loguru import logger
from magika_style_train import (
    DATA_DIR,
    Config,
    encode_corpus,
    per_dataset_f1,
    score,
    threshold_at_best_f1,
    threshold_at_precision,
)
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).parent.parent.resolve()
PROD_DIR = ROOT_DIR / "model_artifacts" / "3.0.0"
MAGIKA_DIR = ROOT_DIR / "model_artifacts" / "magika_style"
OUT_DIR = MAGIKA_DIR

SEED = 42


def cpu_session(path: Path) -> ort.InferenceSession:
    """One intra-op thread, matching the paper's single-CPU constraint (Section V-A)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------------------------------
# Production arm: tokenize -> chunk -> TF-IDF -> ensemble -> aggregate
# --------------------------------------------------------------------------------------------------


@dataclass
class ChunkPlan:
    """Flattened chunks plus the document each one came from."""

    chunks: list[list[int]]
    doc_index: np.ndarray
    n_docs: int


def plan_chunks(texts: list[str], chunker: TokenChunker) -> ChunkPlan:
    chunked = chunker.chunk_batch(tokenize(texts))
    flat: list[list[int]] = []
    owner: list[int] = []
    for doc_idx, chunks in enumerate(chunked):
        for chunk in chunks:
            flat.append(chunk)
            owner.append(doc_idx)
    assert len(flat) == len(owner), f"{len(flat)} chunks vs {len(owner)} owners"
    return ChunkPlan(flat, np.asarray(owner), len(texts))


def production_chunk_probs(
    plan: ChunkPlan, vectorizer: TfidfVectorizer, sess: ort.InferenceSession, batch: int = 256
) -> np.ndarray:
    """AI probability per chunk.

    Densified in batches because the ONNX graph takes a dense [n, 105190] float tensor: one chunk is
    420KB dense, so the whole split at once would be hundreds of gigabytes.
    """
    name = sess.get_inputs()[0].name
    out = np.empty(len(plan.chunks), dtype=np.float64)
    for start in range(0, len(plan.chunks), batch):
        window = plan.chunks[start : start + batch]
        dense = vectorizer.vectorize_from_tokens(window).toarray().astype(np.float32)
        probs = sess.run(None, {name: dense})[1]
        out[start : start + len(window)] = probs[:, 1]
    return out


def aggregate(chunk_probs: np.ndarray, plan: ChunkPlan, threshold: float) -> np.ndarray:
    """Confidence-weighted mean per document, matching `train.py`'s `weighted_mean`.

    Weights are distance from the chunk threshold, so decisive chunks dominate ambivalent ones.
    """
    doc_probs = np.zeros(plan.n_docs)
    order = np.argsort(plan.doc_index, kind="stable")
    bounds = np.searchsorted(plan.doc_index[order], np.arange(plan.n_docs + 1))
    for doc_idx in range(plan.n_docs):
        idx = order[bounds[doc_idx] : bounds[doc_idx + 1]]
        if len(idx) == 0:
            continue
        p = chunk_probs[idx]
        w = np.abs(p - threshold)
        doc_probs[doc_idx] = np.average(p, weights=w) if w.sum() > 1e-10 else p.mean()
    return doc_probs


# --------------------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------------------


def time_production(
    texts: list[str], vectorizer: TfidfVectorizer, sess: ort.InferenceSession, chunker: TokenChunker
) -> dict[str, float]:
    """End-to-end per-document cost, and the preprocessing share of it."""
    t0 = time.time()
    for text in texts:
        plan = plan_chunks([text], chunker)
        dense = vectorizer.vectorize_from_tokens(plan.chunks).toarray().astype(np.float32)
        sess.run(None, {sess.get_inputs()[0].name: dense})
    total = (time.time() - t0) / len(texts) * 1000

    t0 = time.time()
    for text in texts:
        plan = plan_chunks([text], chunker)
        vectorizer.vectorize_from_tokens(plan.chunks)
    prep = (time.time() - t0) / len(texts) * 1000

    return {"ms_total": total, "ms_preprocess": prep, "ms_model": total - prep}


def time_magika(windows: np.ndarray, sess: ort.InferenceSession) -> dict[str, float]:
    name = sess.get_inputs()[0].name
    t0 = time.time()
    for i in range(len(windows)):
        sess.run(None, {name: np.asarray(windows[i : i + 1], dtype=np.int64)})
    total = (time.time() - t0) / len(windows) * 1000
    return {"ms_total": total}


# --------------------------------------------------------------------------------------------------


def load_split(name: str, limit: int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(DATA_DIR / f"{name}.parquet").select("text", "label", "dataset")
    df = lf.collect()
    if limit and limit < len(df):
        df = df.sample(n=limit, seed=SEED, shuffle=True)
    logger.info(f"{name:11s} {len(df):>8,} rows")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="rows per split; default is the full split")
    p.add_argument("--latency-samples", type=int, default=300, help="documents used for the timing pass")
    p.add_argument("--target-precision", type=float, default=0.99)
    p.add_argument(
        "--magika-onnx",
        type=Path,
        default=MAGIKA_DIR / "magika-style-tokens.onnx",
        help="the Magika-shaped artifact to compare against",
    )
    p.add_argument("--encoding", choices=["bytes", "tokens"], default="tokens")
    args = p.parse_args()

    for required in (PROD_DIR / "slop-classifier.onnx", PROD_DIR / "tfidf_vectorizer.rkyv", args.magika_onnx):
        if not required.exists():
            msg = f"missing artifact: {required}"
            raise FileNotFoundError(msg)

    cfg = Config(encoding=args.encoding, limit=args.limit)
    splits = {name: load_split(name, args.limit) for name in ("test", "validation")}
    labels = {name: df["label"].to_numpy() for name, df in splits.items()}

    # ---- production arm -------------------------------------------------------------------------
    chunk_cfg = json.loads((PROD_DIR / "token_chunker_config.json").read_text(encoding="utf-8"))
    chunker = TokenChunker(**chunk_cfg)
    chunk_threshold = float((PROD_DIR / "chunk_classification_threshold.txt").read_text(encoding="utf-8"))
    shipped_doc_threshold = float((PROD_DIR / "classification_threshold.txt").read_text(encoding="utf-8"))
    logger.info(f"production chunker={chunk_cfg} chunk_thr={chunk_threshold:.4f} doc_thr={shipped_doc_threshold:.4f}")

    vectorizer = TfidfVectorizer.load(PROD_DIR / "tfidf_vectorizer.rkyv")
    prod_sess = cpu_session(PROD_DIR / "slop-classifier.onnx")

    prod_probs: dict[str, np.ndarray] = {}
    for name, df in splits.items():
        t0 = time.time()
        plan = plan_chunks(df["text"].to_list(), chunker)
        chunk_probs = production_chunk_probs(plan, vectorizer, prod_sess)
        prod_probs[name] = aggregate(chunk_probs, plan, chunk_threshold)
        logger.info(
            f"production {name}: {len(plan.chunks):,} chunks from {plan.n_docs:,} docs "
            f"({len(plan.chunks) / plan.n_docs:.2f} per doc) in {time.time() - t0:.0f}s"
        )

    # ---- magika arm -----------------------------------------------------------------------------
    magika_sess = cpu_session(args.magika_onnx)
    magika_input = magika_sess.get_inputs()[0].name
    tag = cfg.encoding if args.limit is None else f"{cfg.encoding}-{args.limit}"
    magika_probs: dict[str, np.ndarray] = {}
    windows_cache: dict[str, np.ndarray] = {}
    for name, df in splits.items():
        windows = encode_corpus(df["text"].to_list(), cfg, DATA_DIR / "magika_cache" / f"{name}-{tag}.npy")
        windows_cache[name] = windows
        out = np.empty(len(windows), dtype=np.float64)
        for start in range(0, len(windows), 256):
            block = np.asarray(windows[start : start + 256], dtype=np.int64)
            logits = magika_sess.run(None, {magika_input: block})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out[start : start + len(block)] = e[:, 1] / e.sum(axis=1)
        magika_probs[name] = out

    # ---- thresholds fitted on validation, applied to test ----------------------------------------
    arms = {"production": prod_probs, "magika": magika_probs}
    rules = {
        f"precision_{args.target_precision}": lambda y, p: threshold_at_precision(y, p, args.target_precision),
        "best_f1": threshold_at_best_f1,
    }

    results: dict[str, dict] = {}
    for arm, probs in arms.items():
        results[arm] = {"auc_test": float(roc_auc_score(labels["test"], probs["test"])), "rules": {}}
        for rule, fit in rules.items():
            thr = fit(labels["validation"], probs["validation"])
            results[arm]["rules"][rule] = {"threshold": thr, "test": score(labels["test"], probs["test"], thr)}
    # The shipped threshold, for reference against what the released model actually does.
    results["production"]["rules"]["shipped"] = {
        "threshold": shipped_doc_threshold,
        "test": score(labels["test"], prod_probs["test"], shipped_doc_threshold),
    }

    logger.info("=" * 100)
    logger.info(
        f"{'arm':<12} {'rule':<16} {'thr':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'mcc':>7} {'auc':>7}"
    )
    logger.info("-" * 100)
    for arm, res in results.items():
        for rule, entry in res["rules"].items():
            m = entry["test"]
            logger.info(
                f"{arm:<12} {rule:<16} {entry['threshold']:>7.4f} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['mcc']:>7.4f} {res['auc_test']:>7.4f}"
            )
    logger.info("=" * 100)

    # ---- latency ---------------------------------------------------------------------------------
    sample = splits["test"]["text"].to_list()[: args.latency_samples]
    prod_latency = time_production(sample, vectorizer, prod_sess, chunker)
    magika_latency = time_magika(windows_cache["test"][: args.latency_samples], magika_sess)
    logger.info(
        f"latency per document (1 CPU thread, {len(sample)} docs):\n"
        f"  production {prod_latency['ms_total']:>8.2f}ms  "
        f"(preprocess {prod_latency['ms_preprocess']:.2f}ms + model {prod_latency['ms_model']:.2f}ms)\n"
        f"  magika     {magika_latency['ms_total']:>8.2f}ms  (single fixed window, no chunking)\n"
        f"  speedup    {prod_latency['ms_total'] / magika_latency['ms_total']:>8.1f}x"
    )

    # ---- artifact footprint ----------------------------------------------------------------------
    prod_bytes = sum(f.stat().st_size for f in PROD_DIR.iterdir() if f.suffix in {".onnx", ".rkyv"})
    magika_bytes = args.magika_onnx.stat().st_size
    logger.info(
        f"artifact size: production {prod_bytes / 1024 / 1024:.2f}MB "
        f"(onnx + vectorizer), magika {magika_bytes / 1024 / 1024:.2f}MB"
    )

    results["latency"] = {"production": prod_latency, "magika": magika_latency}
    results["artifact_bytes"] = {"production": prod_bytes, "magika": magika_bytes}

    # ---- per-dataset, both arms ------------------------------------------------------------------
    datasets = splits["test"]["dataset"].cast(pl.Utf8).to_list()
    tables = {
        arm: per_dataset_f1(datasets, labels["test"], probs["test"], results[arm]["rules"]["best_f1"]["threshold"])
        for arm, probs in arms.items()
    }
    side_by_side = (
        tables["production"]
        .select("dataset", "n", "human", "ai", prod_acc="accuracy", prod_f1="f1")
        .join(tables["magika"].select("dataset", magika_acc="accuracy", magika_f1="f1"), on="dataset")
        .with_columns(acc_delta=pl.col("magika_acc") - pl.col("prod_acc"))
        .sort("n", descending=True)
    )
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        logger.info(f"per-dataset, test split (best_f1 thresholds):\n{side_by_side}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    side_by_side.write_csv(OUT_DIR / f"comparison-per-dataset-{tag}.csv")
    (OUT_DIR / f"comparison-{tag}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    # Document-level probabilities, so follow-up analysis (accuracy by length, calibration,
    # disagreement) does not need another pass over the ensemble.
    np.savez_compressed(
        OUT_DIR / f"test-probs-{tag}.npz",
        production=prod_probs["test"],
        magika=magika_probs["test"],
        label=labels["test"],
        dataset=np.asarray(datasets),
    )
    logger.info(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
