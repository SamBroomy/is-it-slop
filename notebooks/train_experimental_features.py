#!/usr/bin/env python
# coding: utf-8
"""EXPERIMENTAL: Multi-expert training with statistical features.

This script explores incorporating statistical features (burstiness, entropy, etc.)
alongside TF-IDF features. It's gated behind the `statistical-features` Rust flag
and is NOT part of the production training pipeline (use train.py instead).

Used for research and experimentation with feature engineering approaches.
"""

# In[ ]:


import json
import logging
import os
import random
import time
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import onnx
import polars as pl
import seaborn as sns
from __init__ import (
    CHUNK_CLASSIFICATION_THRESHOLD_PATH,
    CHUNKER_CONFIG_PATH,
    CLASSIFICATION_THRESHOLD_PATH,
    MODEL_DIR,
    MODEL_ONNX_PATH,
    PLOT_DIR,
    RETRAIN_VECTORIZER,
    RETRAINED_MODEL_VERSION,
    SEED,
    VECTORIZER_BIN_PATH,
    ProbabilisticClassifier,
    df_test,
    df_train,
)
from is_it_slop_preprocessing import (
    TfidfVectorizer,
    TokenChunker,
    VectorizerParams,
    __version__,
    extract_combined_batch,
    fit_tfidf_auto_batch,
    tokenize,
)
from loguru import logger
from onnxruntime.transformers.onnx_model import OnnxModel
from plots import (
    aggregation_comparison,
    analyze_features_by_ngram_length,
    chunk_agreement_analysis,
    chunking_behavior_analysis,
    compare_token_distributions,
    compute_best_thresholds,
    confidence_correctness_analysis,
    dataset_bias_analysis,
    decision_boundary_analysis,
    embedding_visualization,
    plot_calibration_curves,
    plot_prediction_distributions,
    roc_curve_analysis,
    top_ngrams_visualization,
)
from skl2onnx import to_onnx
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Python random
random.seed(SEED)

np.random.default_rng(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["ORT_DETERMINISTIC"] = "1"

mlflow.set_tracking_uri("sqlite:///notebooks/mlflow.db")
mlflow.set_experiment("is-it-slop-training-pipeline")
logging.basicConfig(level=logging.INFO)
logging.getLogger("is_it_slop_preprocessing").setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
print(f"Bindings version: {__version__}")
print(f"Pipeline model version output: {RETRAINED_MODEL_VERSION}")


warnings.filterwarnings("ignore")
# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.figsize"] = (12, 8)

print("Vectorizer exists:", VECTORIZER_BIN_PATH.exists())


# In[ ]:


df_test.select("text").sink_csv("test_texts.csv", include_header=False)
df_train.select("text").sink_csv("train_texts.csv", include_header=False)


# In[ ]:


X_train = df_train.select("text").collect().to_series().to_numpy()
y_train = df_train.select("label").collect().to_series().to_numpy()

X_test = df_test.select("text").collect().to_series().to_numpy()
y_test = df_test.select("label").collect().to_series().to_numpy()

total_samples = len(X_train) + len(X_test)
logger.info(f"Total samples: {total_samples}")
logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Start MLflow run
mlflow.start_run()

# Log dataset info
mlflow.log_param("total_samples", total_samples)
mlflow.log_param("train_samples", len(X_train))
mlflow.log_param("test_samples", len(X_test))
mlflow.log_param("preprocessing_version", __version__)
mlflow.log_param("model_version", str(RETRAINED_MODEL_VERSION))

# ==============================================================================
# Phase 0: Markdown Bias Diagnostics
# ==============================================================================

import re

logger.info("\n" + "=" * 80)
logger.info("MARKDOWN BIAS DIAGNOSTICS")
logger.info("=" * 80)

# Define markdown patterns
MARKDOWN_PATTERNS = {
    "heading": r"^#+\s+",  # # Heading, ## Subheading
    "bold_asterisk": r"\*\*[^*]+\*\*",  # **bold**
    "italic_asterisk": r"(?<!\*)\*(?!\*)[\w\s]+\*(?!\*)",  # *italic*
    "code_block": r"```",  # ```code```
    "inline_code": r"`[^`]+`",  # `code`
    "list_item": r"^\s*[-*+]\s+",  # - item, * item
    "numbered_list": r"^\s*\d+\.\s+",  # 1. item
    "blockquote": r"^>\s+",  # > quote
}


def has_markdown(text: str) -> tuple[bool, dict[str, int]]:
    """Check if text contains markdown patterns."""
    pattern_counts = {}
    has_any = False

    for pattern_name, pattern in MARKDOWN_PATTERNS.items():
        count = len(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
        pattern_counts[pattern_name] = count
        if count > 0:
            has_any = True

    return has_any, pattern_counts


# Analyze training set
logger.info("Analyzing training set...")
train_markdown_stats: dict[str, dict[str, int | dict[str, int]]] = {}

for text, label in zip(X_train, y_train, strict=True):
    label_key = "ai" if label == 1 else "human"
    has_md, counts = has_markdown(text)

    if label_key not in train_markdown_stats:
        train_markdown_stats[label_key] = {"total": 0, "with_markdown": 0, "patterns": {}}

    train_markdown_stats[label_key]["total"] += 1  # type: ignore[operator]
    if has_md:
        train_markdown_stats[label_key]["with_markdown"] += 1  # type: ignore[operator]

    for pattern, count in counts.items():
        if pattern not in train_markdown_stats[label_key]["patterns"]:  # type: ignore[operator]
            train_markdown_stats[label_key]["patterns"][pattern] = 0  # type: ignore[index, assignment]
        train_markdown_stats[label_key]["patterns"][pattern] += count  # type: ignore[index, operator]

# Compute percentages
ai_total = train_markdown_stats["ai"]["total"]  # type: ignore[assignment]
human_total = train_markdown_stats["human"]["total"]  # type: ignore[assignment]
ai_with_md = train_markdown_stats["ai"]["with_markdown"]  # type: ignore[assignment]
human_with_md = train_markdown_stats["human"]["with_markdown"]  # type: ignore[assignment]

ai_pct = (ai_with_md / ai_total) * 100
human_pct = (human_with_md / human_total) * 100

logger.info("\nMarkdown usage in training data:")
logger.info(f"  AI samples with markdown: {ai_with_md}/{ai_total} ({ai_pct:.2f}%)")
logger.info(f"  Human samples with markdown: {human_with_md}/{human_total} ({human_pct:.2f}%)")
logger.info(f"  Bias ratio (AI/Human): {ai_pct / human_pct:.2f}x")

# Log to MLflow
mlflow.log_metric("markdown_bias_ai_pct", ai_pct)
mlflow.log_metric("markdown_bias_human_pct", human_pct)
mlflow.log_metric("markdown_bias_ratio", ai_pct / human_pct if human_pct > 0 else 0)

# Pattern-specific breakdown
logger.info("\nPer-pattern breakdown (count per sample):")
for pattern_name in MARKDOWN_PATTERNS:
    ai_count = train_markdown_stats["ai"]["patterns"].get(pattern_name, 0) / ai_total  # type: ignore[operator]
    human_count = train_markdown_stats["human"]["patterns"].get(pattern_name, 0) / human_total  # type: ignore[operator]
    logger.info(f"  {pattern_name:20s}: AI={ai_count:.3f}, Human={human_count:.3f}")
    mlflow.log_metric(f"markdown_pattern_ai_{pattern_name}", ai_count)
    mlflow.log_metric(f"markdown_pattern_human_{pattern_name}", human_count)

logger.info("=" * 80 + "\n")


# In[ ]:


logger.info("Fitting Rust TF-IDF vectorizer...")
t1 = time.time()
RETRAIN_VECTORIZER = True
if RETRAIN_VECTORIZER or not VECTORIZER_BIN_PATH.exists():
    logger.info("Training new Vectorizer")
    params = VectorizerParams(min_df=100, max_df=0.7)

    # Log vectorizer params
    mlflow.log_param("ngram_range", f"{params.ngram_range}")
    mlflow.log_param("min_df", params.min_df)
    mlflow.log_param("max_df", params.max_df)
    mlflow.log_param("retrain_vectorizer", True)

    # vectorizer = TfidfVectorizer.fit(X_train, params)
    # logger.info(f"Fitted vectorizer in {time.time() - t1:.2f} seconds")
    # t2 = time.time()
    # X_train_tfidf = vectorizer.transform(X_train)
    # logger.info(f"Transformed train data {X_train_tfidf.shape} in {time.time() - t2:.2f} seconds")

    # 1. Fit vocabulary on FULL texts (with automatic batching for large datasets)
    vectorizer, used_batching = fit_tfidf_auto_batch(X_train, params, batch_size=50_000, auto_batch_threshold=100_000)

    logger.info(f"Fitted vectorizer in {time.time() - t1:.2f} seconds")
    mlflow.log_param("used_batched_fit", used_batching)
    if used_batching:
        mlflow.log_param("batch_size", 50_000)
    t2 = time.time()
else:
    logger.info("Loading Pre-trained Vectorizer")

    vectorizer = TfidfVectorizer.load(VECTORIZER_BIN_PATH)
    mlflow.log_param("retrain_vectorizer", False)
    logger.info(f"Loaded vectorizer in {time.time() - t1:.2f} seconds")
    t2 = time.time()

# 2. Tokenize all texts
logger.info("Tokenizing texts...")
train_tokens = tokenize(X_train)
test_tokens = tokenize(X_test)
# 3: Chunk at token level
chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)
logger.info("Chunking tokens...")
train_chunked = chunker.chunk_batch(train_tokens)
test_chunked = chunker.chunk_batch(test_tokens)


# 4: Flatten chunks and replicate labels
def flatten_with_labels(
    chunked_tokens: list[list[list[int]]], labels: np.ndarray
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    flat_chunks = []
    flat_labels = []
    chunk_to_doc_idx = []

    for doc_idx, (chunks, label) in enumerate(zip(chunked_tokens, labels, strict=True)):
        for chunk in chunks:
            flat_chunks.append(chunk)
            flat_labels.append(label)
            chunk_to_doc_idx.append(doc_idx)

    return flat_chunks, np.array(flat_labels), np.array(chunk_to_doc_idx)


train_chunk_tokens, y_train_chunked, train_chunk_to_doc = flatten_with_labels(train_chunked, y_train)
test_chunk_tokens, y_test_chunked, test_chunk_to_doc = flatten_with_labels(test_chunked, y_test)
logger.info(f"Training samples: {len(y_train)} → {len(y_train_chunked)} (after chunking)")

# 5: Vectorize from pre-tokenized chunks
logger.info("Vectorizing chunks...")
X_train_tfidf = vectorizer.vectorize_from_tokens(train_chunk_tokens)
X_test_tfidf = vectorizer.vectorize_from_tokens(test_chunk_tokens)

logger.info(f"Transformed test data {X_test_tfidf.shape} in {time.time() - t2:.2f} seconds")
logger.info(f"Train TF-IDF matrix: {X_train_tfidf.shape}")
sparsity = 100 * (1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape))  # pyright: ignore[reportCallIssue, reportArgumentType]
logger.info(f"TF-IDF Sparsity: {sparsity:.2f}%")

# Log TF-IDF metrics
mlflow.log_metric("n_tfidf_features", X_train_tfidf.shape[1])  # pyright: ignore[reportOptionalSubscript]
mlflow.log_metric("tfidf_sparsity_percent", sparsity)

# ==============================================================================
# 6: Extract Statistical Features (Rust via PyO3)
# ==============================================================================

logger.info("Extracting statistical features...")
t3 = time.time()

# Extract features (Rust handles reverse tokenization internally)
X_train_stat = extract_combined_batch(
    full_texts=X_train.tolist(),  # Convert numpy array to list of strings
    chunk_tokens_batch=train_chunked,
)
X_test_stat = extract_combined_batch(full_texts=X_test.tolist(), chunk_tokens_batch=test_chunked)

logger.info(f"Statistical features shape: {X_train_stat.shape}")
logger.info(f"Extracted statistical features in {time.time() - t3:.2f} seconds")

# Log statistical features info
mlflow.log_metric("n_statistical_features", X_train_stat.shape[1])

# ==============================================================================
# 6b: Extract Word N-gram Features (Discourse Expert)
# ==============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

logger.info("Extracting word n-gram features...")
t_word_start = time.time()

# Word-level TF-IDF (discourse patterns - alphabetic tokens only)
word_vectorizer = SklearnTfidfVectorizer(
    analyzer="word",  # Word-level (phrases)
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=100,  # Same as token vectorizer
    max_df=0.7,  # Same as token vectorizer
    max_features=30000,  # Sufficient for discourse patterns
    sublinear_tf=True,  # log(1 + tf) scaling
    lowercase=True,  # Normalize case
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # Alphabetic only (excludes numerics/punctuation)
)

# Fit on full training texts (document-level, not chunks)
X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

logger.info(f"Word n-gram matrix (doc-level): {X_train_word.shape}")
word_sparsity = 100 * (1 - X_train_word.nnz / np.prod(X_train_word.shape))
logger.info(f"Word n-gram sparsity: {word_sparsity:.2f}%")


# Replicate for chunks (same word features for all chunks from same doc)
def replicate_for_chunks(doc_features, chunk_to_doc):
    """Replicate document-level features for each chunk."""
    from scipy.sparse import vstack

    chunk_features = []
    for chunk_idx in range(len(chunk_to_doc)):
        doc_idx = chunk_to_doc[chunk_idx]
        chunk_features.append(doc_features[doc_idx])
    return vstack(chunk_features)


X_train_word_chunked = replicate_for_chunks(X_train_word, train_chunk_to_doc)
X_test_word_chunked = replicate_for_chunks(X_test_word, test_chunk_to_doc)

logger.info(f"Word n-gram matrix (chunk-level): {X_train_word_chunked.shape}")
logger.info(f"Extracted word n-grams in {time.time() - t_word_start:.2f} seconds")

# Log to MLflow
mlflow.log_metric("n_word_features", X_train_word.shape[1])
mlflow.log_metric("word_sparsity_percent", word_sparsity)

# Save word vectorizer for inference (joblib format)
import joblib

WORD_VECTORIZER_PATH = MODEL_DIR / "word_tfidf_vectorizer.joblib"
joblib.dump(word_vectorizer, WORD_VECTORIZER_PATH)
logger.info(f"Saved word vectorizer to {WORD_VECTORIZER_PATH}")

# ==============================================================================
# 7: Combine TF-IDF + Statistical + Word N-gram Features
# ==============================================================================

logger.info("Combining TF-IDF, statistical, and word n-gram features...")
t5 = time.time()

from scipy.sparse import csr_matrix, hstack

# Convert scaled statistical features to sparse format
X_train_stat_sparse = csr_matrix(X_train_stat)
X_test_stat_sparse = csr_matrix(X_test_stat)

# Concatenate horizontally: [TF-IDF | Statistical | Word N-grams]
X_train_combined = hstack([X_train_tfidf, X_train_stat_sparse, X_train_word_chunked])
X_test_combined = hstack([X_test_tfidf, X_test_stat_sparse, X_test_word_chunked])

logger.info(f"Combined feature matrix shape (TF-IDF + stats + words): {X_train_combined.shape}")
logger.info(f"Combined features in {time.time() - t5:.2f} seconds")

# Log combined metrics
total_features = X_train_combined.shape[1]
mlflow.log_metric("n_total_features", total_features)  # pyright: ignore[reportOptionalSubscript]
combined_sparsity = 100 * (1 - X_train_combined.nnz / np.prod(X_train_combined.shape))  # pyright: ignore[reportCallIssue, reportArgumentType]
mlflow.log_metric("combined_sparsity_percent", combined_sparsity)

logger.info(
    f"Total features: {total_features} = "
    f"{X_train_tfidf.shape[1]} TF-IDF + "
    f"{X_train_stat.shape[1]} statistical + "
    f"{X_train_word_chunked.shape[1]} word n-grams"
)
logger.info(f"Combined sparsity: {combined_sparsity:.2f}%")

# ============================================================
# Define Feature Groups for Three-Expert Architecture
# ============================================================

logger.info("Setting up feature-specific expert architecture (3 experts)...")

# Feature column indices
n_tfidf_features = X_train_tfidf.shape[1]
n_stat_features = X_train_stat.shape[1]
n_word_features = X_train_word_chunked.shape[1]

TFIDF_COLS = list(range(n_tfidf_features))  # Columns 0 to n_tfidf_features-1
STAT_COLS = list(range(n_tfidf_features, n_tfidf_features + n_stat_features))
WORD_COLS = list(range(n_tfidf_features + n_stat_features, n_tfidf_features + n_stat_features + n_word_features))

logger.info(f"TF-IDF features: {len(TFIDF_COLS)} columns (0-{n_tfidf_features - 1})")
logger.info(
    f"Statistical features: {len(STAT_COLS)} columns ({n_tfidf_features}-{n_tfidf_features + n_stat_features - 1})"
)
logger.info(
    f"Word n-gram features: {len(WORD_COLS)} columns ({n_tfidf_features + n_stat_features}-{total_features - 1})"
)

# ONNX-compatible feature selectors
tfidf_selector = ColumnTransformer(
    transformers=[("tfidf", "passthrough", TFIDF_COLS)],
    remainder="drop",
    sparse_threshold=1.0,  # Keep sparse
)

stat_selector = ColumnTransformer(transformers=[("stats", "passthrough", STAT_COLS)], remainder="drop")

word_selector = ColumnTransformer(
    transformers=[("chars", "passthrough", WORD_COLS)],
    remainder="drop",
    sparse_threshold=1.0,  # Keep sparse
)

mlflow.log_param("architecture", "three_expert_ensemble")
mlflow.log_param("tfidf_feature_count", len(TFIDF_COLS))
mlflow.log_param("stat_feature_count", len(STAT_COLS))
mlflow.log_param("word_feature_count", len(WORD_COLS))


# In[ ]:


def create_chunk_level_dataframe(
    df: pl.DataFrame, chunked_tokens: list[list[list[int]]], labels: np.ndarray
) -> pl.DataFrame:

    chunk_data = []

    chunk_idx = 0
    for doc_idx, (row, chunks, label) in enumerate(zip(df.iter_rows(named=True), chunked_tokens, labels, strict=True)):
        for chunk_position, _chunk in enumerate(chunks):
            chunk_row = {
                **row,  # Copy all original columns (text, dataset, label, etc.)
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "chunk_position": chunk_position,
                "num_chunks_in_doc": len(chunks),
                # "chunk_tokens": chunk,
                "chunk_label": label,  # Replicated label
            }
            chunk_data.append(chunk_row)
            chunk_idx += 1

    return pl.DataFrame(chunk_data)


# Usage in your training notebook (after chunking)
logger.info("Creating chunk-level DataFrames...")

df_train_chunks = create_chunk_level_dataframe(
    df_train.collect(),  # Must be collected
    train_chunked,
    y_train,
)

df_test_chunks = create_chunk_level_dataframe(df_test.collect(), test_chunked, y_test)


# In[ ]:


# ============================================================
# Expert 1: TF-IDF Specialist
# ============================================================

logger.info("Building TF-IDF Expert Ensemble...")

# Models for TF-IDF (can include MultinomialNB - features are non-negative!)
nb = MultinomialNB(alpha=0.01)

sgd_tfidf = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.00005,
    class_weight="balanced",
    early_stopping=True,
    max_iter=2000,
    tol=1e-4,
    random_state=SEED,
    learning_rate="optimal",
    n_jobs=-1,
)

svc_tfidf = LinearSVC(C=1.0, loss="squared_hinge", max_iter=2000, class_weight="balanced", random_state=SEED)
svc_tfidf_calibrated = CalibratedClassifierCV(svc_tfidf, cv=3, method="sigmoid")

# TF-IDF Expert: Pipeline with feature selection
tfidf_expert = Pipeline([
    ("select_features", tfidf_selector),
    (
        "ensemble",
        StackingClassifier(
            estimators=[("nb", nb), ("sgd", sgd_tfidf), ("svc", svc_tfidf_calibrated)],
            final_estimator=LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED),
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1,
            verbose=1,
        ),
    ),
])

mlflow.log_param("tfidf_expert_models", ["MultinomialNB", "SGDClassifier", "LinearSVC"])
mlflow.log_param("tfidf_expert_meta", "LogisticRegression")

# ============================================================
# Expert 2: Statistical Features Specialist
# ============================================================

logger.info("Building Statistical Features Expert Ensemble...")

# Models for statistical features (dense, can be negative after scaling)
sgd_stat = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.0001,
    class_weight="balanced",
    early_stopping=True,
    max_iter=1000,
    tol=1e-4,
    random_state=SEED,
    learning_rate="optimal",
    n_jobs=-1,
)

logreg_stat = LogisticRegression(
    penalty="l2",
    C=10.0,  # Less regularization for few features
    solver="saga",
    max_iter=500,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)

rf_stat = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)

# Statistical Expert: Pipeline with feature selection + scaling
stat_expert = Pipeline([
    ("select_features", stat_selector),
    ("scaler", StandardScaler()),  # Z-score scaling in pipeline!
    (
        "ensemble",
        StackingClassifier(
            estimators=[("sgd", sgd_stat), ("logreg", logreg_stat), ("rf", rf_stat)],
            final_estimator=LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED),
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1,
            verbose=1,
        ),
    ),
])

mlflow.log_param("stat_expert_models", ["SGDClassifier", "LogisticRegression", "RandomForestClassifier"])
mlflow.log_param("stat_expert_meta", "LogisticRegression")
mlflow.log_param("stat_expert_scaling", "StandardScaler")

# ============================================================
# Expert 3: Discourse Specialist (Word N-grams)
# ============================================================

logger.info("Building Discourse Expert Ensemble (word n-grams)...")

# Models for word n-grams (similar to TF-IDF - non-negative sparse)
nb_word = MultinomialNB(alpha=0.01)

sgd_word = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.00005,
    class_weight="balanced",
    early_stopping=True,
    max_iter=2000,
    tol=1e-4,
    random_state=SEED,
    learning_rate="optimal",
    n_jobs=-1,
)

svc_word = LinearSVC(C=1.0, loss="squared_hinge", max_iter=2000, class_weight="balanced", random_state=SEED)
svc_word_calibrated = CalibratedClassifierCV(svc_word, cv=3, method="sigmoid")

# Discourse Expert: Pipeline with feature selection (NO scaling - counts are non-negative)
linguistic_expert = Pipeline([
    ("select_features", word_selector),
    (
        "ensemble",
        StackingClassifier(
            estimators=[("nb", nb_word), ("sgd", sgd_word), ("svc", svc_word_calibrated)],
            final_estimator=LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED),
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1,
            verbose=1,
        ),
    ),
])

mlflow.log_param("discourse_expert_models", ["MultinomialNB", "SGDClassifier", "LinearSVC"])
mlflow.log_param("discourse_expert_meta", "LogisticRegression")
mlflow.log_param("discourse_expert_analyzer", "word")
mlflow.log_param("discourse_expert_ngram_range", "(1, 3)")

# ============================================================
# Meta-Ensemble: Combine 3 Experts
# ============================================================

logger.info("Building Meta-Ensemble (StackingClassifier with 3 experts)...")

meta_ensemble = StackingClassifier(
    estimators=[("tfidf_expert", tfidf_expert), ("stat_expert", stat_expert), ("linguistic_expert", linguistic_expert)],
    final_estimator=LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED),
    cv=3,
    stack_method="predict_proba",
    n_jobs=-1,
    verbose=True,
)

mlflow.log_param("meta_ensemble_type", "StackingClassifier")
mlflow.log_param("meta_ensemble_experts", ["tfidf", "statistical", "discourse"])
mlflow.log_param("meta_ensemble_final_estimator", "LogisticRegression")
mlflow.log_param("meta_ensemble_cv", 3)

# Train
logger.info("Training meta-ensemble with feature-specific experts...")
t_train_start = time.time()
meta_ensemble.fit(X_train_combined, y_train_chunked)
training_time = time.time() - t_train_start

logger.info(f"Training completed in {training_time:.2f} seconds")
mlflow.log_metric("training_time_seconds", training_time)

# Extract individual expert ensembles for metrics
tfidf_expert_ensemble = meta_ensemble.named_estimators_["tfidf_expert"].named_steps["ensemble"]
stat_expert_ensemble = meta_ensemble.named_estimators_["stat_expert"].named_steps["ensemble"]
linguistic_expert_ensemble = meta_ensemble.named_estimators_["linguistic_expert"].named_steps["ensemble"]

# Update model dictionary for compatibility with existing plotting code
# NOTE: Only include models that can handle the full combined feature matrix.
# Individual base models within experts (tfidf_nb, stat_sgd, etc.) expect
# feature-specific subsets and cannot be plotted directly with X_combined.
models: dict[str, ProbabilisticClassifier] = {
    # Expert-level models (these are Pipelines that handle feature selection)
    "tfidf_expert": meta_ensemble.named_estimators_["tfidf_expert"],
    "stat_expert": meta_ensemble.named_estimators_["stat_expert"],
    "linguistic_expert": meta_ensemble.named_estimators_["linguistic_expert"],
    # Meta-ensemble (used as "ensemble" for backward compatibility)
    "ensemble": meta_ensemble,
}

# For detailed metrics, we already computed individual model performance above
# using the appropriate feature subsets. No need to re-plot them here.

# For compatibility with ONNX export code
ensemble = meta_ensemble


# In[ ]:


type(X_train_combined.dtype)


# In[ ]:


# ============================================================
# Detailed Metrics Collection Function
# ============================================================


def compute_detailed_metrics(model, X_test, y_test, threshold: float, prefix: str = "") -> dict[str, float]:
    """Compute comprehensive metrics for a model.

    Args:
        model: Fitted classifier with predict_proba
        X_test: Test features
        y_test: True labels
        threshold: Classification threshold
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics

    """
    # Get predictions
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(np.int8)

    # Compute threshold-independent metrics
    auc = roc_auc_score(y_test, probs)

    # Find best F1 threshold
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    # Compute threshold-dependent metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision_score_val = precision_score(y_test, y_pred)
    recall_score_val = recall_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        f"{prefix}auc": auc,
        f"{prefix}best_f1": best_f1,
        f"{prefix}best_f1_threshold": best_f1_threshold,
        f"{prefix}mcc": mcc,
        f"{prefix}accuracy": accuracy,
        f"{prefix}precision": precision_score_val,
        f"{prefix}recall": recall_score_val,
        f"{prefix}f1": f1_score_val,
        f"{prefix}tp": int(tp),
        f"{prefix}fp": int(fp),
        f"{prefix}tn": int(tn),
        f"{prefix}fn": int(fn),
    }


chunk_probs = meta_ensemble.predict_proba(X_test_combined)[:, 1]  # pyright: ignore[reportCallIssue, reportArgumentType]
chunk_probs


# In[ ]:


logger.info("\n" + "=" * 80)
logger.info("COMPREHENSIVE METRICS ANALYSIS")
logger.info("=" * 80)

# ============================================================
# Level 1: Individual Models within TF-IDF Expert
# ============================================================

logger.info("\n--- TF-IDF Expert: Individual Models ---")

# Get the fitted selector from the pipeline
fitted_tfidf_selector = meta_ensemble.named_estimators_["tfidf_expert"].named_steps["select_features"]

for i, (name, _) in enumerate(tfidf_expert_ensemble.estimators):
    model = tfidf_expert_ensemble.estimators_[i]

    # Extract TF-IDF features for testing
    X_test_tfidf = fitted_tfidf_selector.transform(X_test_combined)

    metrics = compute_detailed_metrics(
        model,
        X_test_tfidf,
        y_test_chunked,
        threshold=0.5,  # Use default, will report best F1 threshold
        prefix=f"tfidf_{name}_",
    )

    logger.info(
        f"{name:15s} - AUC: {metrics[f'tfidf_{name}_auc']:.4f}, "
        f"Best F1: {metrics[f'tfidf_{name}_best_f1']:.4f} "
        f"(threshold: {metrics[f'tfidf_{name}_best_f1_threshold']:.4f})"
    )

    # Log to MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"chunked_{metric_name}", value)

# ============================================================
# Level 2: Individual Models within Statistical Expert
# ============================================================

logger.info("\n--- Statistical Expert: Individual Models ---")

# Get the fitted components from the pipeline
fitted_stat_selector = meta_ensemble.named_estimators_["stat_expert"].named_steps["select_features"]
stat_scaler = meta_ensemble.named_estimators_["stat_expert"].named_steps["scaler"]

for i, (name, _) in enumerate(stat_expert_ensemble.estimators):
    model = stat_expert_ensemble.estimators_[i]

    # Extract statistical features (and scale)
    X_test_stat_extracted = fitted_stat_selector.transform(X_test_combined)
    X_test_stat_scaled = stat_scaler.transform(X_test_stat_extracted)

    metrics = compute_detailed_metrics(model, X_test_stat_scaled, y_test_chunked, threshold=0.5, prefix=f"stat_{name}_")

    logger.info(
        f"{name:15s} - AUC: {metrics[f'stat_{name}_auc']:.4f}, "
        f"Best F1: {metrics[f'stat_{name}_best_f1']:.4f} "
        f"(threshold: {metrics[f'stat_{name}_best_f1_threshold']:.4f})"
    )

    # Log to MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"chunked_{metric_name}", value)

# ============================================================
# Level 3: Expert-Level Performance
# ============================================================

logger.info("\n--- Expert-Level Performance ---")

# TF-IDF Expert
tfidf_expert_metrics = compute_detailed_metrics(
    meta_ensemble.named_estimators_["tfidf_expert"],
    X_test_combined,
    y_test_chunked,
    threshold=0.5,
    prefix="tfidf_expert_",
)

logger.info(
    f"TF-IDF Expert  - AUC: {tfidf_expert_metrics['tfidf_expert_auc']:.4f}, "
    f"Best F1: {tfidf_expert_metrics['tfidf_expert_best_f1']:.4f}"
)

for metric_name, value in tfidf_expert_metrics.items():
    mlflow.log_metric(f"chunked_{metric_name}", value)

# Statistical Expert
stat_expert_metrics = compute_detailed_metrics(
    meta_ensemble.named_estimators_["stat_expert"],
    X_test_combined,
    y_test_chunked,
    threshold=0.5,
    prefix="stat_expert_",
)

logger.info(
    f"Statistical Expert - AUC: {stat_expert_metrics['stat_expert_auc']:.4f}, "
    f"Best F1: {stat_expert_metrics['stat_expert_best_f1']:.4f}"
)

for metric_name, value in stat_expert_metrics.items():
    mlflow.log_metric(f"chunked_{metric_name}", value)

# ============================================================
# Level 2b: Individual Models within Discourse Expert
# ============================================================

logger.info("\n--- Discourse Expert: Individual Models ---")

fitted_word_selector = meta_ensemble.named_estimators_["linguistic_expert"].named_steps["select_features"]

for i, (name, _) in enumerate(linguistic_expert_ensemble.estimators):
    model = linguistic_expert_ensemble.estimators_[i]

    # Extract word n-gram features
    X_test_word = fitted_word_selector.transform(X_test_combined)

    metrics = compute_detailed_metrics(model, X_test_word, y_test_chunked, threshold=0.5, prefix=f"word_{name}_")

    logger.info(
        f"{name:15s} - AUC: {metrics[f'word_{name}_auc']:.4f}, "
        f"Best F1: {metrics[f'word_{name}_best_f1']:.4f} "
        f"(threshold: {metrics[f'word_{name}_best_f1_threshold']:.4f})"
    )

    # Log to MLflow
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"chunked_{metric_name}", value)

# Discourse Expert (Expert-level)
discourse_expert_metrics = compute_detailed_metrics(
    meta_ensemble.named_estimators_["linguistic_expert"],
    X_test_combined,
    y_test_chunked,
    threshold=0.5,
    prefix="discourse_expert_",
)

logger.info(
    f"Discourse Expert - AUC: {discourse_expert_metrics['discourse_expert_auc']:.4f}, "
    f"Best F1: {discourse_expert_metrics['discourse_expert_best_f1']:.4f}"
)

for metric_name, value in discourse_expert_metrics.items():
    mlflow.log_metric(f"chunked_{metric_name}", value)

# ============================================================
# Level 4: Meta-Ensemble Performance (Chunk-Level)
# ============================================================

logger.info("\n--- Meta-Ensemble Performance (Chunk-Level) ---")

best_chunked_threshold, best_chunked_threshold_roc = compute_best_thresholds(y_test_chunked, chunk_probs)

logger.info(f"Best F1 threshold at chunk level: {best_chunked_threshold:.4f}")
logger.info(f"Best ROC-AUC threshold at chunk level: {best_chunked_threshold_roc:.4f}")

meta_ensemble_metrics = compute_detailed_metrics(
    meta_ensemble, X_test_combined, y_test_chunked, threshold=best_chunked_threshold, prefix="meta_ensemble_"
)

logger.info(
    f"Meta Ensemble - AUC: {meta_ensemble_metrics['meta_ensemble_auc']:.4f}, "
    f"F1: {meta_ensemble_metrics['meta_ensemble_f1']:.4f}"
)

for metric_name, value in meta_ensemble_metrics.items():
    mlflow.log_metric(f"chunked_{metric_name}", value)

logger.info("=" * 80 + "\n")

mlflow.log_param("best_chunked_threshold", best_chunked_threshold)
mlflow.log_param("best_chunked_threshold_roc", best_chunked_threshold_roc)


# In[ ]:


chunnked_y_pred = (chunk_probs >= best_chunked_threshold).astype(np.int8)


# In[ ]:


# 1 is best, 0 is random, -1 is worst
test_mcc = matthews_corrcoef(y_test_chunked, chunnked_y_pred)
logger.info(f"Validation MCC: {test_mcc:.4f}")

test_auc: float = roc_auc_score(y_test_chunked, chunnked_y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Validation AUC: {test_auc:.4f}")
accuracy: float = accuracy_score(y_test_chunked, chunnked_y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Accuracy:   {accuracy:.4f}")
precision: float = precision_score(y_test_chunked, chunnked_y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Precision:  {precision:.4f}")
recall: float = recall_score(y_test_chunked, chunnked_y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Recall:     {recall:.4f}")
f1: float = f1_score(y_test_chunked, chunnked_y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"F1 Score:   {f1:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test_chunked, chunnked_y_pred).ravel()
logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
logger.info("Confusion Matrix:")
logger.info("              Predicted")
logger.info("                 0      1")
logger.info(f"Actual  0    {tn:5d}  {fp:5d}")
logger.info(f"        1    {fn:5d}  {tp:5d}")
dis = ConfusionMatrixDisplay.from_predictions(y_test_chunked, chunnked_y_pred)
plot_path = PLOT_DIR / "chunked_confusion_matrix.png"
dis.figure_.savefig(plot_path, bbox_inches="tight")
mlflow.log_artifact(str(plot_path))

# Log all metrics to MLflow
mlflow.log_metric("chunked_test_mcc", test_mcc)
mlflow.log_metric("chunked_test_auc", test_auc)
mlflow.log_metric("chunked_accuracy", accuracy)
mlflow.log_metric("chunked_precision", precision)
mlflow.log_metric("chunked_recall", recall)
mlflow.log_metric("chunked_f1_score", f1)
mlflow.log_metric("chunked_true_positives", int(tp))
mlflow.log_metric("chunked_false_positives", int(fp))
mlflow.log_metric("chunked_true_negatives", int(tn))
mlflow.log_metric("chunked_false_negatives", int(fn))


# In[ ]:


def aggregate_chunk_predictions(
    chunk_probs: np.ndarray,
    chunk_to_doc_idx: np.ndarray,
    n_docs: int,
    method: Literal["mean", "max", "weighted_mean"] = "mean",
    threshold: float = 0.5,
) -> np.ndarray:
    doc_probs = np.zeros(n_docs)

    if method == "mean":
        # Simple average of all chunks per document
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                doc_probs[doc_idx] = chunk_probs[mask].mean()

    elif method == "max":
        # Most suspicious chunk (conservative for AI detection)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                doc_probs[doc_idx] = chunk_probs[mask].max()

    elif method == "weighted_mean":
        # Weight by distance from threshold (confidence-weighted)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                chunk_probs_doc = chunk_probs[mask]
                # Higher weight for more confident predictions
                weights = np.abs(chunk_probs_doc - threshold)
                # Handle edge case: if all weights are zero, fall back to mean
                if weights.sum() > 1e-10:
                    doc_probs[doc_idx] = np.average(chunk_probs_doc, weights=weights)
                else:
                    doc_probs[doc_idx] = chunk_probs_doc.mean()

    return doc_probs


y_probs = aggregate_chunk_predictions(
    chunk_probs,
    test_chunk_to_doc,
    n_docs=len(y_test),
    method="weighted_mean",  # Try: "mean", "max", "weighted_mean"
    threshold=best_chunked_threshold,
)


# In[ ]:


from sklearn.calibration import calibration_curve

# Check if probabilities are well-calibrated
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.plot(prob_pred, prob_true, "s-", label="Ensemble")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.savefig(PLOT_DIR / "calibration_check.png")


# In[ ]:


roc_curve_analysis(X_train_combined, y_train_chunked, X_test_combined, y_test_chunked, models)


# In[ ]:


# 2) use precision-recall curve for exact best F1 (thresholds length differs)


best_threshold, best_threshold_roc = compute_best_thresholds(y_test, y_probs)

# Log thresholds
mlflow.log_metric("best_threshold_f1", best_threshold)
mlflow.log_metric("best_threshold_youden", best_threshold_roc)


# In[ ]:


y_pred = (y_probs >= best_threshold).astype(np.int8)


# In[ ]:


# 1 is best, 0 is random, -1 is worst
test_mcc = matthews_corrcoef(y_test, y_pred)
logger.info(f"Validation MCC: {test_mcc:.4f}")

test_auc: float = roc_auc_score(y_test, y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Validation AUC: {test_auc:.4f}")
accuracy: float = accuracy_score(y_test, y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Accuracy:   {accuracy:.4f}")
precision: float = precision_score(y_test, y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Precision:  {precision:.4f}")
recall: float = recall_score(y_test, y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"Recall:     {recall:.4f}")
f1: float = f1_score(y_test, y_pred)  # pyright: ignore[reportAssignmentType]
logger.info(f"F1 Score:   {f1:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
logger.info("Confusion Matrix:")
logger.info("              Predicted")
logger.info("                 0      1")
logger.info(f"Actual  0    {tn:5d}  {fp:5d}")
logger.info(f"        1    {fn:5d}  {tp:5d}")
dis = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plot_path = PLOT_DIR / "confusion_matrix.png"
dis.figure_.savefig(plot_path, bbox_inches="tight")
mlflow.log_artifact(str(plot_path))

# Log all metrics to MLflow
mlflow.log_metric("test_mcc", test_mcc)
mlflow.log_metric("test_auc", test_auc)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("true_positives", int(tp))
mlflow.log_metric("false_positives", int(fp))
mlflow.log_metric("true_negatives", int(tn))
mlflow.log_metric("false_negatives", int(fn))


# In[ ]:


from sklearn.utils import resample

n_bootstraps = 100
f1_scores = []

for _ in range(n_bootstraps):
    # Resample with replacement
    idx = resample(range(len(y_test)), random_state=None)
    y_test_boot = y_test[idx]
    y_pred_boot = y_pred[idx]
    f1_scores.append(f1_score(y_test_boot, y_pred_boot))

f1_scores = np.array(f1_scores)
logger.info(
    f"F1: {f1_scores.mean():.4f} (95% CI: [{np.percentile(f1_scores, 2.5):.4f}, {np.percentile(f1_scores, 97.5):.4f}])"
)
# 2025-12-11 22:03:00.434 | INFO     | __main__:<module>:14 - F1: 0.9436 (95% CI: [0.9421, 0.9452])


# In[ ]:


# Save vectorizer in both formats:
# 1. JSON-wrapped format for Python (with metadata)
# vectorizer.save(VECTORIZER_JSON_PATH)
# logger.info(f"Saved json vectorizer to {VECTORIZER_JSON_PATH}")
# 2. Raw rkyv format for Rust (no JSON wrapper)
vectorizer.save(VECTORIZER_BIN_PATH)
logger.info(f"Saved binary vectorizer to {VECTORIZER_BIN_PATH}")

Path(CLASSIFICATION_THRESHOLD_PATH).write_text(str(best_threshold), encoding="utf-8")
logger.info(f"Saved classification threshold to {CLASSIFICATION_THRESHOLD_PATH}")
Path(CHUNK_CLASSIFICATION_THRESHOLD_PATH).write_text(str(best_chunked_threshold), encoding="utf-8")
logger.info(f"Saved chunk classification threshold to {CHUNK_CLASSIFICATION_THRESHOLD_PATH}")
with Path(CHUNKER_CONFIG_PATH).open("w", encoding="utf-8") as f:
    json.dump(chunker.to_dict(), f)
logger.info(f"Saved chunker config to {CHUNKER_CONFIG_PATH}")

# Convert to ONNX
# Disable ZipMap to output probabilities as a 2D tensor [batch_size, num_classes]
logger.info("Converting to ONNX...")
onx: onnx.ModelProto = to_onnx(
    ensemble,
    X_train_combined[:1].toarray(),  # Sample for shape inference (all 3 feature types)
    options={
        type(ensemble): {"zipmap": False}  # Output probabilities as tensor, not dict
    },
)  # pyright: ignore[reportAssignmentType]
onnx.checker.check_model(onx, full_check=True)


# with MODEL_ONNX_PATH.open("wb") as f:
#     f.write(onx.SerializeToString())#deterministic=True))
# logger.info(f"Saved ONNX model to {MODEL_ONNX_PATH}")

# onnx_model = onnx.load(MODEL_ONNX_PATH)


# To get rid of the following errors we need to prune the graph
# "CleanUnusedInitializersAndNodeArgs] Removing initializer 'classes_ind'. It is not used by any node and should be removed from the model"
onnx_model = OnnxModel(onx)
onnx_model.prune_graph()
onnx_model.save_model_to_file(MODEL_ONNX_PATH)


# Log artifacts to MLflow
mlflow.log_artifact(str(MODEL_ONNX_PATH))
mlflow.log_artifact(str(VECTORIZER_BIN_PATH))
mlflow.log_artifact(str(WORD_VECTORIZER_PATH))
# mlflow.log_artifact(str(VECTORIZER_JSON_PATH))
mlflow.log_artifact(str(CLASSIFICATION_THRESHOLD_PATH))
mlflow.log_artifact(str(CHUNK_CLASSIFICATION_THRESHOLD_PATH))
mlflow.log_artifact(str(CHUNKER_CONFIG_PATH))


# In[ ]:


best_threshold


# In[ ]:


import onnxruntime as rt

sess = rt.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name

test_input = X_train_combined[:2]  # .astype(np.float64)  # .todense()

input_name = sess.get_inputs()[0].name

pred_onx = sess.run(None, {input_name: test_input.toarray()})


# In[ ]:


input_meta = sess.get_inputs()[0]


# In[ ]:


input_meta


# In[ ]:


model_pred = ensemble.predict_proba(test_input)
model_pred


# In[ ]:


pred_onx[1]


# In[ ]:


np.allclose(pred_onx[1], model_pred)  # pyright: ignore[reportArgumentType]


# In[ ]:


plot_prediction_distributions(X_test_combined, y_test_chunked, models)


# In[ ]:


plot_calibration_curves(X_test_combined, y_test_chunked, models)


# In[ ]:


decision_boundary_analysis(
    X_test_combined,
    y_test_chunked,
    ensemble.predict_proba(X_test_combined),  # pyright: ignore[reportArgumentType]
    decision_threshold=best_chunked_threshold,  # type: ignore[reportArgumentType]
)


# In[ ]:


analyze_features_by_ngram_length(vectorizer, models, top_n=40)


# In[ ]:


dataset_bias_analysis(df_test.collect().to_pandas(), y_probs, y_pred, best_threshold)


# In[ ]:


y_test_chunked


# In[ ]:


X_test_combined


# In[ ]:


embedding_visualization(
    X_test_combined, y_test_chunked, df_test_chunks.select("dataset").to_series().to_numpy(), sample_size=20_000
)


# In[ ]:


texts_human = df_test.filter(pl.col("label") == 0).select("text").collect().to_series()
texts_ai = df_test.filter(pl.col("label") == 1).select("text").collect().to_series()
compare_token_distributions(texts_human, texts_ai)


# In[ ]:


# artifact_position_analysis(
#     df_test.select("text").collect().to_series().to_list(),
#     y_test,
#     vectorizer,
#     ensemble,  # type: ignore[reportArgumentType]
#     best_threshold,
# )


# In[ ]:


# per_dataset_accuracy_analysis(X_test_combined, models["svc"], threshold=best_threshold)


# In[ ]:


# =============================================================================
# Additional v5.0 Visualizations (Chunking-specific)
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("Generating v5.0 chunking-specific visualizations...")
logger.info("=" * 80 + "\n")


# In[ ]:


# 1. Top predictive n-grams
logger.info("1/5: Top predictive n-grams...")
# Use fitted final estimator from TF-IDF expert (most interpretable for TF-IDF features)
tfidf_meta_estimator = tfidf_expert_ensemble.final_estimator_
top_ngrams_visualization(vectorizer, tfidf_meta_estimator.coef_.ravel(), top_n=20)  # pyright: ignore[reportAttributeAccessIssue]


# In[ ]:


# 2. Chunk agreement analysis
logger.info("2/5: Chunk agreement analysis...")
chunk_agreement_analysis(chunk_probs, test_chunk_to_doc, y_test, y_pred, best_chunked_threshold, len(y_test))


# In[ ]:


# 3. Aggregation method comparison
logger.info("3/5: Aggregation method comparison...")
aggregation_comparison(chunk_probs, test_chunk_to_doc, y_test, best_chunked_threshold, best_threshold, len(y_test))


# In[ ]:


# 4. Chunking behavior analysis
logger.info("4/5: Chunking behavior analysis...")
chunking_behavior_analysis(test_chunk_to_doc, test_chunked, len(y_test))


# In[ ]:


# 5. Confidence vs correctness analysis
logger.info("5/5: Confidence vs correctness analysis...")
confidence_correctness_analysis(y_probs, y_test, y_pred, best_threshold)


# In[ ]:


logger.info("\n" + "=" * 80)
logger.info("All visualizations complete!")
logger.info("=" * 80 + "\n")


# In[ ]:


# End MLflow run
mlflow.end_run()
logger.info("MLflow run completed")
