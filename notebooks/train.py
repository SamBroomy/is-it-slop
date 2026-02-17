#!/usr/bin/env python
# coding: utf-8

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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

    # 1. Fit vocabulary on FULL texts
    vectorizer = TfidfVectorizer.fit(X_train, params)

    logger.info(f"Fitted vectorizer in {time.time() - t1:.2f} seconds")
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


logger.info("Transforming test data...")
# X_test_tfidf = vectorizer.transform(X_test)

logger.info(f"Transformed test data {X_test_tfidf.shape} in {time.time() - t2:.2f} seconds")
logger.info(f"Train Feature matrix: {X_train_tfidf.shape}")
sparsity = 100 * (1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape))  # pyright: ignore[reportCallIssue, reportArgumentType]
logger.info(f"Sparsity: {sparsity:.2f}%")

# Log feature matrix metrics
mlflow.log_metric("n_features", X_train_tfidf.shape[1])  # pyright: ignore[reportOptionalSubscript]
mlflow.log_metric("sparsity_percent", sparsity)


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


# Train ensemble
logger.info("Training ensemble...")

nb = MultinomialNB(alpha=0.01)

# cn = ComplementNB(alpha=0.01)

sgd = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.00005,
    class_weight="balanced",
    early_stopping=True,
    max_iter=2000,  # 8000
    tol=1e-4,
    random_state=SEED,
    learning_rate="optimal",
    n_jobs=-1,
)

logreg = LogisticRegression(
    penalty="l2", C=1.0, solver="saga", max_iter=200, class_weight="balanced", random_state=SEED, n_jobs=-1
)  # 1000

# LinearSVC - very fast, needs calibration for probabilities
svc = LinearSVC(
    C=1.0,
    loss="squared_hinge",  # Good for sparse data
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)
# Wrap for probability calibration (needed for ensemble voting='soft')
svc_calibrated = CalibratedClassifierCV(svc, cv=5, method="sigmoid")

estimators: list[tuple[str, BaseEstimator, float]] = [
    ("sgd", sgd, 0.15),
    ("logreg", logreg, 0.30),
    ("svc", svc_calibrated, 0.4),
    ("nb", nb, 0.15),
    # ("cnb", cn, 0.05),
]
voting = "soft"

assert abs(sum(weight for _, _, weight in estimators) - 1.0) < 1e-6, "Weights must sum to 1.0"  # noqa: S101

mlflow.log_param("ensemble_estimators", [name for name, _, _ in estimators])
mlflow.log_param("ensemble_weights", [weight for _, _, weight in estimators])
mlflow.log_param("model_type", "StackingClassifier")
mlflow.log_param("voting", voting)

# ensemble = VotingClassifier(
#     estimators=[(name, model) for name, model, _ in estimators],
#     weights=[weight for _, _, weight in estimators],
#     voting=voting,
#     n_jobs=-1,
#     flatten_transform=False,
#     verbose=True,
# )


meta_lr = LogisticRegression(max_iter=200, n_jobs=-1, random_state=SEED)
meta_calibrated = CalibratedClassifierCV(meta_lr, cv=5, method="sigmoid")
ensemble = StackingClassifier(
    estimators=[(name, est) for name, est, _ in estimators],
    final_estimator=meta_lr,
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
    verbose=True,
)

# Retrain
ensemble.fit(X_train_tfidf, y_train_chunked)
# This is just a list but to save to onnx we need it as a numpy array
# ensemble.weights = np.array(ensemble.weights)  # pyright: ignore[reportAttributeAccessIssue]

# Use a Protocol or Union type for classifiers with predict_proba


models: dict[str, ProbabilisticClassifier] = {
    "sgd": ensemble.estimators_[0],
    "logreg": ensemble.estimators_[1],
    "svc": ensemble.estimators_[2],
    "nb": ensemble.estimators_[3],
    # "cnb": ensemble.estimators_[4],
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
    method: Literal["mean", "max", "weighted_mean"] = "mean",
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
        # Weight by distance from 0.5 (confidence-weighted)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                chunk_probs_doc = chunk_probs[mask]
                # Higher weight for more confident predictions
                weights = np.abs(chunk_probs_doc - best_chunked_threshold)
                doc_probs[doc_idx] = np.average(chunk_probs_doc, weights=weights)

    return doc_probs


y_probs = aggregate_chunk_predictions(
    chunk_probs,
    test_chunk_to_doc,
    n_docs=len(y_test),
    method="weighted_mean",  # Try: "mean", "max", "weighted_mean"
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


best_threshold


# In[ ]:


import onnxruntime as rt

sess = rt.InferenceSession(MODEL_ONNX_PATH, providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name

test_input = X_train_tfidf[:2]  # .astype(np.float64)  # .todense()

input_name = sess.get_inputs()[0].name

pred_onx = sess.run(None, {input_name: test_input.toarray()})


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


# =============================================================================
# Additional v5.0 Visualizations (Chunking-specific)
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("Generating v5.0 chunking-specific visualizations...")
logger.info("=" * 80 + "\n")


# In[ ]:


# 1. Top predictive n-grams
logger.info("1/5: Top predictive n-grams...")
# Use fitted logreg from ensemble (most interpretable single model)
top_ngrams_visualization(vectorizer, models["logreg"].coef_.ravel(), top_n=20)  # pyright: ignore[reportAttributeAccessIssue]


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

