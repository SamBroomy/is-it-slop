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
import dataclasses
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
from hierarchical_train import Config as HierConfig
from hierarchical_train import encode_corpus as hier_encode_corpus
from hierarchical_train import encoding_tag as hier_encoding_tag
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, tokenize
from loguru import logger
from magika_style_train import (
    DATA_DIR,
    Config,
    artifact_tag,
    encode_corpus,
    per_dataset_f1,
    score,
    threshold_at_best_f1,
    threshold_at_precision,
)
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).parent.parent.resolve()
PROD_DIR = ROOT_DIR / "model_artifacts" / "3.0.0"
MAGIKA_DIR = ROOT_DIR / "model_artifacts" / "magika_style"
HIER_DIR = ROOT_DIR / "model_artifacts" / "hierarchical"
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
# Hierarchical arm: [batch, chunks, chunk_size], chunk axis dynamic
# --------------------------------------------------------------------------------------------------


def load_hierarchical(tag: str) -> tuple[HierConfig, Path]:
    """Recover a hierarchical run's config from the results it wrote, rather than parsing its name.

    Raises:
        FileNotFoundError: if the run's results JSON or ONNX artifact is missing.

    """
    results = HIER_DIR / tag / "results.json"
    onnx_path = HIER_DIR / tag / "slop-classifier.onnx"
    for required in (results, onnx_path):
        if not required.exists():
            msg = f"missing artifact: {required}"
            raise FileNotFoundError(msg)
    stored = json.loads(results.read_text(encoding="utf-8"))["config"]
    fields = {f.name for f in dataclasses.fields(HierConfig)}
    return HierConfig(**{k: v for k, v in stored.items() if k in fields}), onnx_path


def real_chunk_counts(chunks: np.ndarray, pad_id: int) -> np.ndarray:
    """Count the leading chunks holding real tokens, per document."""
    return (np.asarray(chunks) != pad_id).any(axis=2).sum(axis=1).clip(min=1)


def hierarchical_probs(chunks: np.ndarray, sess: ort.InferenceSession, batch: int = 128) -> np.ndarray:
    """Document AI probability. The graph emits a probability directly, so there is no softmax here."""
    name = sess.get_inputs()[0].name
    out = np.empty(len(chunks), dtype=np.float64)
    for start in range(0, len(chunks), batch):
        block = np.asarray(chunks[start : start + batch], dtype=np.int64)
        out[start : start + len(block)] = sess.run(None, {name: block})[0].ravel()
    return out


def time_hierarchical(chunks: np.ndarray, sess: ort.InferenceSession, pad_id: int) -> dict[str, float]:
    """Per-document cost feeding only each document's real chunks.

    That is the deployment path now the chunk axis is dynamic: a short document sends one chunk
    rather than a padded grid, so cost tracks document length instead of the training budget.
    """
    name = sess.get_inputs()[0].name
    counts = real_chunk_counts(chunks, pad_id)
    t0 = time.time()
    for i, k in enumerate(counts):
        sess.run(None, {name: np.asarray(chunks[i : i + 1, :k, :], dtype=np.int64)})
    total = (time.time() - t0) / len(counts) * 1000
    return {"ms_total": total, "mean_chunks": float(counts.mean())}


# --------------------------------------------------------------------------------------------------


def score_hierarchical(
    tags: list[str], splits: dict[str, pl.DataFrame], limit: int | None
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[ort.InferenceSession, HierConfig, np.ndarray]]]:
    """Score each named hierarchical run, reusing the vocabulary and encoding it trained with.

    `limit` names this comparison's row count so a subsampled pass writes its own caches. The
    trained config carries `limit=None`, so without this a `--limit` run would overwrite the
    full-size caches training built with short ones.

    Raises:
        FileNotFoundError: if a run's vocabulary cache is missing.

    """
    probs_by_arm: dict[str, dict[str, np.ndarray]] = {}
    sessions: dict[str, tuple[ort.InferenceSession, HierConfig, np.ndarray]] = {}
    for tag in tags:
        cfg, onnx_path = load_hierarchical(tag)
        # Rebuilding the vocabulary here could disagree with the one the run trained against.
        # Read from the bundle rather than the dataset cache, so the comparison uses exactly the
        # vocabulary shipped alongside the graph.
        vocab_path = HIER_DIR / tag / "token_vocabulary.safetensors"
        if not vocab_path.exists():
            msg = f"missing vocabulary {vocab_path}; retrain this tag to write a complete bundle"
            raise FileNotFoundError(msg)
        lut = load_file(vocab_path)["token_to_row"].numpy()
        sess = cpu_session(onnx_path)
        cache_cfg = dataclasses.replace(cfg, limit=limit)

        probs: dict[str, np.ndarray] = {}
        for name, df in splits.items():
            cache = DATA_DIR / "hierarchical_cache" / f"{name}-{hier_encoding_tag(cache_cfg)}.npy"
            encoded = hier_encode_corpus(df["text"].to_list(), cfg, lut, cache)
            probs[name] = hierarchical_probs(encoded, sess)
            if name == "test":
                sessions[tag] = (sess, cfg, encoded)
        probs_by_arm[f"hier:{tag}"] = probs
        logger.info(f"hierarchical {tag}: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
    return probs_by_arm, sessions


def report_metrics(results: dict, width: int) -> None:
    """One row per arm and threshold rule, all measured on the same test split."""
    rule = "=" * (width + 74)
    logger.info(rule)
    logger.info(
        f"{'arm':<{width}} {'rule':<16} {'thr':>7} {'acc':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'mcc':>7} {'auc':>7}"
    )
    logger.info("-" * (width + 74))
    for arm, res in results.items():
        for name, entry in res["rules"].items():
            m = entry["test"]
            logger.info(
                f"{arm:<{width}} {name:<16} {entry['threshold']:>7.4f} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
                f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['mcc']:>7.4f} {res['auc_test']:>7.4f}"
            )
    logger.info(rule)


def report_cost(latency: dict, sizes: dict, width: int, n_docs: int) -> None:
    """Per-document latency and artifact size, with speedup relative to production."""
    lines = [f"latency and size per arm (1 CPU thread, {n_docs} docs):"]
    for arm, entry in latency.items():
        detail = ""
        if "ms_preprocess" in entry:
            detail = f"  (preprocess {entry['ms_preprocess']:.2f} + model {entry['ms_model']:.2f})"
        elif "mean_chunks" in entry:
            detail = f"  ({entry['mean_chunks']:.2f} chunks/doc, real chunks only)"
        speedup = latency["production"]["ms_total"] / entry["ms_total"]
        lines.append(
            f"  {arm:<{width}} {entry['ms_total']:>7.2f}ms  {speedup:>5.1f}x  "
            f"{sizes[arm] / 1024 / 1024:>6.2f}MB{detail}"
        )
    logger.info("\n".join(lines))


def dataset_table(datasets: list[str], y: np.ndarray, arms: dict, thresholds: dict) -> pl.DataFrame:
    """Per-source accuracy, one column per arm, each at its own validation-fitted threshold."""
    table = per_dataset_f1(datasets, y, arms["production"]["test"], thresholds["production"]).select(
        "dataset", "n", "human", "ai", production="accuracy"
    )
    for arm, probs in arms.items():
        if arm == "production":
            continue
        column = per_dataset_f1(datasets, y, probs["test"], thresholds[arm]).select("dataset", **{arm: "accuracy"})
        table = table.join(column, on="dataset")
    return table.sort("n", descending=True)


def length_table(texts: list[str], y: np.ndarray, arms: dict, thresholds: dict) -> pl.DataFrame:
    """Accuracy by document length, the axis that separated every arm measured so far."""
    n_tokens = np.array([len(t) for t in tokenize([t.strip() for t in texts])])
    edges = (0, 50, 100, 150, 200, 300, 500, 1000, 2000, np.iinfo(np.int64).max)
    rows = []
    for lo, hi in itertools.pairwise(edges):
        mask = (n_tokens >= lo) & (n_tokens < hi)
        if mask.sum() < 20:
            continue
        row: dict[str, object] = {"tokens": f"{lo}-{'inf' if hi == edges[-1] else hi}", "n": int(mask.sum())}
        for arm, probs in arms.items():
            hit = (probs["test"][mask] >= thresholds[arm]).astype(np.int8) == y[mask]
            row[arm] = round(float(hit.mean()), 4)
        rows.append(row)
    return pl.DataFrame(rows)


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
    p.add_argument(
        "--window", type=int, default=Config.window, help="must match the window the Magika artifact was trained with"
    )
    p.add_argument(
        "--hierarchical",
        action="append",
        default=[],
        metavar="TAG",
        help="tag of a hierarchical run to include, e.g. c192-k8-v32768. Repeatable. Its config is "
        "read from the results JSON it wrote, so nothing needs restating here.",
    )
    args = p.parse_args()

    for required in (PROD_DIR / "slop-classifier.onnx", PROD_DIR / "tfidf_vectorizer.rkyv", args.magika_onnx):
        if not required.exists():
            msg = f"missing artifact: {required}"
            raise FileNotFoundError(msg)

    cfg = Config(encoding=args.encoding, window=args.window, limit=args.limit)
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
    tag = artifact_tag(cfg)
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

    # ---- hierarchical arms ------------------------------------------------------------------------
    hier_probs, hier_sessions = score_hierarchical(args.hierarchical, splits, args.limit)

    # ---- thresholds fitted on validation, applied to test ----------------------------------------
    arms = {"production": prod_probs, "magika": magika_probs, **hier_probs}
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

    width = max(24, *(len(a) for a in arms))
    report_metrics(results, width)

    # ---- latency and artifact footprint ----------------------------------------------------------
    sample = splits["test"]["text"].to_list()[: args.latency_samples]
    latency = {"production": time_production(sample, vectorizer, prod_sess, chunker)}
    latency["magika"] = time_magika(windows_cache["test"][: args.latency_samples], magika_sess)
    sizes = {
        "production": sum(f.stat().st_size for f in PROD_DIR.iterdir() if f.suffix in {".onnx", ".rkyv"}),
        "magika": args.magika_onnx.stat().st_size,
    }
    for hier_tag, (sess, hier_cfg, encoded) in hier_sessions.items():
        latency[f"hier:{hier_tag}"] = time_hierarchical(encoded[: args.latency_samples], sess, hier_cfg.pad_id)
        # Whole bundle, not just the graph: the vocabulary is required at inference, and production
        # is likewise measured as ONNX plus vectorizer.
        sizes[f"hier:{hier_tag}"] = sum(
            f.stat().st_size for f in (HIER_DIR / hier_tag).iterdir() if f.suffix in {".onnx", ".safetensors"}
        )

    report_cost(latency, sizes, width, len(sample))
    results["latency"] = latency
    results["artifact_bytes"] = sizes

    # ---- per-dataset and per-length, every arm ----------------------------------------------------
    datasets = splits["test"]["dataset"].cast(pl.Utf8).to_list()
    thresholds = {arm: results[arm]["rules"]["best_f1"]["threshold"] for arm in arms}
    side_by_side = dataset_table(datasets, labels["test"], arms, thresholds)
    by_length = length_table(splits["test"]["text"].to_list(), labels["test"], arms, thresholds)

    with pl.Config(tbl_rows=40, tbl_width_chars=200):
        logger.info(f"per-dataset accuracy, test split (best_f1 thresholds):\n{side_by_side}")
        logger.info(f"accuracy by document length:\n{by_length}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    side_by_side.write_csv(OUT_DIR / f"comparison-per-dataset-{tag}.csv")
    by_length.write_csv(OUT_DIR / f"comparison-per-length-{tag}.csv")
    (OUT_DIR / f"comparison-{tag}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    # Document-level probabilities, so follow-up analysis (calibration, where arms disagree) does
    # not need another pass over the ensemble.
    np.savez_compressed(
        OUT_DIR / f"test-probs-{tag}.npz",
        label=labels["test"],
        dataset=np.asarray(datasets),
        **{arm.replace(":", "_"): probs["test"] for arm, probs in arms.items()},
    )
    logger.info(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
