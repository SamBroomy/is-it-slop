#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import os
import random
import time
import warnings
from datetime import UTC, datetime
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
    df_validation,
)
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, VectorizerParams, __version__, tokenize
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
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB

# Python random
random.seed(SEED)

np.random.default_rng(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["ORT_DETERMINISTIC"] = "1"
mlflow.set_tracking_uri("sqlite:///notebooks/mlflow.db")
mlflow.set_experiment("is-it-slop-training-pipeline")
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("is_it_slop_preprocessing").setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("skl2onnx").setLevel(logging.INFO)
print(f"Bindings version: {__version__}")
print(f"Pipeline model version output: {RETRAINED_MODEL_VERSION}")


warnings.filterwarnings("ignore")
# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["figure.figsize"] = (12, 8)

print("Vectorizer exists:", VECTORIZER_BIN_PATH.exists())

# Start training timer for metadata
training_start_time = time.time()


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


# In[ ]:


# ==============================================================================
# Load Dataset Metadata (computed during curation)
# ==============================================================================

from __init__ import DATA_DIR

logger.info("Loading dataset metadata...")
dataset_metadata_path = DATA_DIR / "dataset_metadata.json"

if not dataset_metadata_path.exists():
    error = f"Dataset metadata not found at {dataset_metadata_path}"
    logger.error(error)
    logger.error("Please run dataset curation first: just dataset-curation")
    raise FileNotFoundError(error)

with dataset_metadata_path.open("r", encoding="utf-8") as f:
    dataset_metadata = json.load(f)

logger.info(f"Loaded dataset metadata (version: {dataset_metadata['dataset_version']})")
logger.info(f"Dataset created: {dataset_metadata['created_timestamp']}")
logger.info(f"Total samples: {dataset_metadata['sample_counts']['total']}")


# In[ ]:


# Load validation set for evaluation
X_validation = df_validation.select("text").collect().to_series().to_numpy()
y_validation = df_validation.select("label").collect().to_series().to_numpy()

logger.info(f"Validation samples: {len(X_validation)}")
mlflow.log_param("validation_samples", len(X_validation))


# In[ ]:


logger.info("Fitting Rust TF-IDF vectorizer...")
t1 = time.time()
RETRAIN_VECTORIZER = True
if RETRAIN_VECTORIZER or not VECTORIZER_BIN_PATH.exists():
    logger.info("Training new Vectorizer")

    # Scaled for full dataset (473K texts)
    # min_df=0.001 → min 473 docs (0.1% of corpus) - filters rare n-grams
    # Previous v2.1.0: 30% sampling (142K texts), min_df=100 (0.07%)
    params = VectorizerParams(min_df=0.0007, max_df=0.7)

    # Log vectorizer params
    mlflow.log_param("ngram_range", f"{params.ngram_range}")
    mlflow.log_param("min_df", params.min_df)
    mlflow.log_param("max_df", params.max_df)
    mlflow.log_param("retrain_vectorizer", True)

    # Fit vocabulary (batching is automatic based on dataset size)
    vectorizer = TfidfVectorizer.fit(X_train, params)

    logger.info(f"Fitted vectorizer in {time.time() - t1:.2f} seconds")
    t2 = time.time()
else:
    logger.info("Loading Pre-trained Vectorizer")

    vectorizer = TfidfVectorizer.load(VECTORIZER_BIN_PATH)
    params = vectorizer.params
    mlflow.log_param("retrain_vectorizer", False)
    logger.info(f"Loaded vectorizer in {time.time() - t1:.2f} seconds")
    t2 = time.time()

# Tokenize all texts (batching is automatic based on dataset size)
logger.info("Tokenizing texts...")
train_tokens = tokenize(X_train)
test_tokens = tokenize(X_test)

# Chunk at token level
chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)
logger.info("Chunking tokens...")
train_chunked = chunker.chunk_batch(train_tokens)
test_chunked = chunker.chunk_batch(test_tokens)


# Flatten chunks and replicate labels
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

# Vectorize from pre-tokenized chunks
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


# In[ ]:


# ==============================================================================
# Compute Chunking Statistics (model-specific, not dataset property)
# ==============================================================================

logger.info("Computing chunking statistics...")


def compute_statistics(values: list[int] | np.ndarray) -> dict:
    """Compute statistical summary (mean, std, min, max, percentiles)."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p50": int(np.percentile(arr, 50)),
        "p95": int(np.percentile(arr, 95)),
    }


# Chunks per document (model-specific: depends on chunker config)
chunks_per_doc_stats = {
    "train": compute_statistics([len(chunks) for chunks in train_chunked]),
    "test": compute_statistics([len(chunks) for chunks in test_chunked]),
}

logger.info(
    f"Chunks per doc (train): mean={chunks_per_doc_stats['train']['mean']:.1f}, "
    f"std={chunks_per_doc_stats['train']['std']:.1f}"
)


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

    schema = {
        **df.schema,
        "doc_idx": pl.Int64,
        "chunk_idx": pl.Int64,
        "chunk_position": pl.Int64,
        "num_chunks_in_doc": pl.Int64,
        "chunk_label": pl.Int8,
    }
    return pl.DataFrame(chunk_data, schema=schema)


# Usage in your training notebook (after chunking)
logger.info("Creating chunk-level DataFrames...")

df_train_chunks = create_chunk_level_dataframe(
    df_train.collect(),  # Must be collected
    train_chunked,
    y_train,
)

df_test_chunks = create_chunk_level_dataframe(df_test.collect(), test_chunked, y_test)


# In[ ]:


# Train ensemble with manual GroupKFold stacking
logger.info("Training ensemble...")
cv = 5
# Define base models
nb = MultinomialNB(alpha=0.01)

sgd = SGDClassifier(
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

logreg = LogisticRegression(
    penalty="l2", C=1.0, solver="saga", max_iter=200, class_weight="balanced", random_state=SEED, n_jobs=-1
)

# LinearSVC - very fast, needs calibration for probabilities
svc = LinearSVC(C=1.0, loss="squared_hinge", max_iter=2000, class_weight="balanced", random_state=SEED)
svc_calibrated = CalibratedClassifierCV(svc, cv=cv, method="sigmoid")

estimators: list[tuple[str, BaseEstimator]] = [("sgd", sgd), ("logreg", logreg), ("svc", svc_calibrated), ("nb", nb)]

mlflow.log_param("ensemble_estimators", [name for name, _ in estimators])
mlflow.log_param("model_type", "StackingClassifier")
mlflow.log_param("cv_folds", cv)

meta_lr = LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED)
meta_calibrated = CalibratedClassifierCV(meta_lr, cv=cv, method="sigmoid")
ensemble = StackingClassifier(
    estimators=[(name, est) for name, est in estimators],
    final_estimator=meta_lr,
    cv=cv,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
    verbose=True,
)


ensemble.fit(X_train_tfidf, y_train_chunked)  # type: ignore[reportArgumentType]

models: dict[str, ProbabilisticClassifier] = {
    "sgd": ensemble.estimators_[0],
    "logreg": ensemble.estimators_[1],
    "svc": ensemble.estimators_[2],
    "nb": ensemble.estimators_[3],
    "ensemble": ensemble,
}  # pyright: ignore[reportAssignmentType]


# In[ ]:


type(X_train_tfidf.dtype)


# In[ ]:


chunk_probs = ensemble.predict_proba(X_test_tfidf)[:, 1]  # pyright: ignore[reportCallIssue, reportArgumentType]
chunk_probs


# In[ ]:


best_chunked_threshold, best_chunked_threshold_roc = compute_best_thresholds(y_test_chunked, chunk_probs)
logger.info(f"Best F1 threshold at chunk level: {best_chunked_threshold:.4f}")
logger.info(f"Best ROC-AUC threshold at chunk level: {best_chunked_threshold_roc:.4f}")
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
    threshold: float,
    method: Literal["mean", "max", "weighted_mean"] = "weighted_mean",
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
                weights = np.abs(chunk_probs_doc - threshold)
                # Handle edge case: if all weights are zero, fall back to mean
                if weights.sum() > 1e-10:
                    doc_probs[doc_idx] = np.average(chunk_probs_doc, weights=weights)
                else:
                    doc_probs[doc_idx] = chunk_probs_doc.mean()

    return doc_probs


y_probs = aggregate_chunk_predictions(
    chunk_probs, test_chunk_to_doc, n_docs=len(y_test), threshold=best_chunked_threshold, method="weighted_mean"
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


roc_curve_analysis(X_train_tfidf, y_train_chunked, X_test_tfidf, y_test_chunked, models)


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


# ==============================================================================
# Validation Set Evaluation (Holdout Performance)
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("VALIDATION SET EVALUATION")
logger.info("=" * 80)

# Tokenize and chunk validation set
logger.info("Processing validation set...")
validation_tokens = tokenize(X_validation)
validation_chunked = chunker.chunk_batch(validation_tokens)

# Flatten chunks for vectorization
validation_chunk_tokens, y_validation_chunked, validation_chunk_to_doc = flatten_with_labels(
    validation_chunked, y_validation
)

# Vectorize
X_validation_tfidf = vectorizer.vectorize_from_tokens(validation_chunk_tokens)
logger.info(f"Validation samples: {len(y_validation)} → {len(y_validation_chunked)} chunks")

# Predict at chunk level
validation_chunk_probs = ensemble.predict_proba(X_validation_tfidf)[:, 1]  # pyright: ignore[reportCallIssue, reportArgumentType]
validation_chunked_y_pred = (validation_chunk_probs >= best_chunked_threshold).astype(np.int8)

# Chunk-level metrics
val_chunk_mcc = matthews_corrcoef(y_validation_chunked, validation_chunked_y_pred)
val_chunk_auc: float = roc_auc_score(y_validation_chunked, validation_chunked_y_pred)  # pyright: ignore[reportAssignmentType]
val_chunk_accuracy: float = accuracy_score(y_validation_chunked, validation_chunked_y_pred)  # pyright: ignore[reportAssignmentType]
val_chunk_precision: float = precision_score(y_validation_chunked, validation_chunked_y_pred)  # pyright: ignore[reportAssignmentType]
val_chunk_recall: float = recall_score(y_validation_chunked, validation_chunked_y_pred)  # pyright: ignore[reportAssignmentType]
val_chunk_f1: float = f1_score(y_validation_chunked, validation_chunked_y_pred)  # pyright: ignore[reportAssignmentType]
val_chunk_tn, val_chunk_fp, val_chunk_fn, val_chunk_tp = confusion_matrix(
    y_validation_chunked, validation_chunked_y_pred
).ravel()

logger.info("Chunk-level validation metrics:")
logger.info(f"  MCC: {val_chunk_mcc:.4f}")
logger.info(f"  AUC: {val_chunk_auc:.4f}")
logger.info(f"  F1: {val_chunk_f1:.4f}")

# Log to MLflow
mlflow.log_metric("validation_chunked_mcc", val_chunk_mcc)
mlflow.log_metric("validation_chunked_auc", val_chunk_auc)
mlflow.log_metric("validation_chunked_accuracy", val_chunk_accuracy)
mlflow.log_metric("validation_chunked_precision", val_chunk_precision)
mlflow.log_metric("validation_chunked_recall", val_chunk_recall)
mlflow.log_metric("validation_chunked_f1_score", val_chunk_f1)
mlflow.log_metric("validation_chunked_tp", int(val_chunk_tp))
mlflow.log_metric("validation_chunked_fp", int(val_chunk_fp))
mlflow.log_metric("validation_chunked_tn", int(val_chunk_tn))
mlflow.log_metric("validation_chunked_fn", int(val_chunk_fn))

# Aggregate to document level
validation_y_probs = aggregate_chunk_predictions(
    validation_chunk_probs,
    validation_chunk_to_doc,
    n_docs=len(y_validation),
    threshold=best_chunked_threshold,
    method="weighted_mean",
)

validation_y_pred = (validation_y_probs >= best_threshold).astype(np.int8)

# Document-level metrics
val_doc_mcc = matthews_corrcoef(y_validation, validation_y_pred)
val_doc_auc: float = roc_auc_score(y_validation, validation_y_pred)  # pyright: ignore[reportAssignmentType]
val_doc_accuracy: float = accuracy_score(y_validation, validation_y_pred)  # pyright: ignore[reportAssignmentType]
val_doc_precision: float = precision_score(y_validation, validation_y_pred)  # pyright: ignore[reportAssignmentType]
val_doc_recall: float = recall_score(y_validation, validation_y_pred)  # pyright: ignore[reportAssignmentType]
val_doc_f1: float = f1_score(y_validation, validation_y_pred)  # pyright: ignore[reportAssignmentType]
val_doc_tn, val_doc_fp, val_doc_fn, val_doc_tp = confusion_matrix(y_validation, validation_y_pred).ravel()

logger.info("Document-level validation metrics:")
logger.info(f"  MCC: {val_doc_mcc:.4f}")
logger.info(f"  AUC: {val_doc_auc:.4f}")
logger.info(f"  Accuracy: {val_doc_accuracy:.4f}")
logger.info(f"  Precision: {val_doc_precision:.4f}")
logger.info(f"  Recall: {val_doc_recall:.4f}")
logger.info(f"  F1: {val_doc_f1:.4f}")
logger.info(f"  Confusion: TP={val_doc_tp}, FP={val_doc_fp}, TN={val_doc_tn}, FN={val_doc_fn}")

# Log to MLflow
mlflow.log_metric("validation_mcc", val_doc_mcc)
mlflow.log_metric("validation_auc", val_doc_auc)
mlflow.log_metric("validation_accuracy", val_doc_accuracy)
mlflow.log_metric("validation_precision", val_doc_precision)
mlflow.log_metric("validation_recall", val_doc_recall)
mlflow.log_metric("validation_f1_score", val_doc_f1)
mlflow.log_metric("validation_tp", int(val_doc_tp))
mlflow.log_metric("validation_fp", int(val_doc_fp))
mlflow.log_metric("validation_tn", int(val_doc_tn))
mlflow.log_metric("validation_fn", int(val_doc_fn))

# Add validation chunking statistics
chunks_per_doc_stats["validation"] = compute_statistics([len(chunks) for chunks in validation_chunked])

logger.info("=" * 80 + "\n")


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
    X_train_tfidf[:1].toarray(),  # Sample for shape inference
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
# mlflow.log_artifact(str(VECTORIZER_JSON_PATH))
mlflow.log_artifact(str(CLASSIFICATION_THRESHOLD_PATH))
mlflow.log_artifact(str(CHUNK_CLASSIFICATION_THRESHOLD_PATH))
mlflow.log_artifact(str(CHUNKER_CONFIG_PATH))


# In[ ]:


# ==============================================================================
# Collect and Save Model Metadata
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("COLLECTING MODEL METADATA")
logger.info("=" * 80)


def compute_artifact_sizes(model_dir: Path) -> dict:
    """Compute sizes of all artifact files."""
    files = {}
    total_bytes = 0

    for file_path in model_dir.iterdir():
        if file_path.is_file():
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            files[file_path.name] = {"size_bytes": size_bytes, "size_mb": round(size_mb, 2)}
            total_bytes += size_bytes

    return {"files": files, "total_size_mb": round(total_bytes / (1024 * 1024), 2)}


# Collect training-specific metadata
training_duration = time.time() - training_start_time
artifact_sizes = compute_artifact_sizes(MODEL_DIR)

# Merge dataset metadata with model metadata
model_metadata = {
    "metadata_version": "1.0.0",
    "version_info": {
        "model_version": str(RETRAINED_MODEL_VERSION),
        "dataset_version": dataset_metadata["dataset_version"],

        "training_timestamp": datetime.now(UTC).isoformat(),
        "dataset_created": dataset_metadata["created_timestamp"],
    },
    "dataset_info": {
        # Use dataset metadata for composition and base statistics
        **dataset_metadata
    },
    "model_config": {
        "model_type": "StackingClassifier",
        "base_estimators": [name for name, _ in estimators],
        "meta_estimator": "LogisticRegression",
        "vectorizer": {
            "type": "TF-IDF",
            "ngram_range": list(params.ngram_range),
            "min_df": params.min_df,
            "max_df": params.max_df,
            "n_features": X_train_tfidf.shape[1],  # pyright: ignore[reportOptionalSubscript]
            "sparsity_percent": round(sparsity, 2),
        },
        "chunking": chunker.to_dict(),
        "thresholds": {"document_level": round(best_threshold, 6), "chunk_level": round(best_chunked_threshold, 6)},
    },
    "performance_metrics": {
        "test": {
            "document_level": {
                "mcc": round(test_mcc, 4),
                "auc": round(test_auc, 4),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            },
            "chunk_level": {
                "mcc": round(test_mcc, 4),
                "auc": round(test_auc, 4),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
        },
        "validation": {
            "document_level": {
                "mcc": round(val_doc_mcc, 4),
                "auc": round(val_doc_auc, 4),
                "accuracy": round(val_doc_accuracy, 4),
                "precision": round(val_doc_precision, 4),
                "recall": round(val_doc_recall, 4),
                "f1": round(val_doc_f1, 4),
                "confusion_matrix": {
                    "tp": int(val_doc_tp),
                    "fp": int(val_doc_fp),
                    "tn": int(val_doc_tn),
                    "fn": int(val_doc_fn),
                },
            },
            "chunk_level": {
                "mcc": round(val_chunk_mcc, 4),
                "auc": round(val_chunk_auc, 4),
                "accuracy": round(val_chunk_accuracy, 4),
                "precision": round(val_chunk_precision, 4),
                "recall": round(val_chunk_recall, 4),
                "f1": round(val_chunk_f1, 4),
            },
        },
    },
    "chunking_statistics": {
        # Model-specific: depends on chunker configuration
        "chunks_per_document": chunks_per_doc_stats
    },
    "artifact_info": artifact_sizes,
    "training_info": {"seed": SEED, "cv_folds": cv, "training_duration_seconds": round(training_duration, 2)},
}

# Save metadata to JSON
metadata_path = MODEL_DIR / "model_metadata.json"
with metadata_path.open("w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=None)

logger.info(f"Saved model metadata to {metadata_path}")
logger.info(f"Training duration: {training_duration:.2f} seconds")
logger.info(f"Total artifact size: {artifact_sizes['total_size_mb']:.2f} MB")
logger.info(f"Dataset version: {dataset_metadata['dataset_version']}")
logger.info(f"Markdown bias (from dataset): {dataset_metadata['markdown_bias']['ratio']:.2f}x")

# Log metadata file to MLflow
mlflow.log_artifact(str(metadata_path))

logger.info("=" * 80 + "\n")


# In[ ]:


best_threshold


# In[ ]:


import onnxruntime as rt

sess = rt.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name

test_input = X_train_tfidf[:2]  # .astype(np.float64)  # .todense()

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


plot_prediction_distributions(X_test_tfidf, y_test_chunked, models)


# In[ ]:


plot_calibration_curves(X_test_tfidf, y_test_chunked, models)


# In[ ]:


decision_boundary_analysis(
    X_test_tfidf,
    y_test_chunked,
    ensemble.predict_proba(X_test_tfidf),  # pyright: ignore[reportArgumentType]
    decision_threshold=best_chunked_threshold,  # type: ignore[reportArgumentType]
)


# In[ ]:


analyze_features_by_ngram_length(vectorizer, models, top_n=40)


# In[ ]:


dataset_bias_analysis(df_test.collect().to_pandas(), y_probs, y_pred, best_threshold)


# In[ ]:


y_test_chunked


# In[ ]:


X_test_tfidf


# In[ ]:


embedding_visualization(
    X_test_tfidf, y_test_chunked, df_test_chunks.select("dataset").to_series().to_numpy(), sample_size=20_000
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


# per_dataset_accuracy_analysis(X_test_tfidf, models["svc"], threshold=best_threshold)


# In[ ]:


logger.info("\n" + "=" * 80)
logger.info("Generating chunking-specific visualizations...")
logger.info("=" * 80 + "\n")


# In[ ]:


# 1. Top predictive n-grams
logger.info("1/5: Top predictive n-grams...")
# Use fitted logreg from ensemble (most interpretable single model)
top_ngrams_visualization(vectorizer, models["logreg"].coef_.ravel(), top_n=25)  # pyright: ignore[reportAttributeAccessIssue]


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

