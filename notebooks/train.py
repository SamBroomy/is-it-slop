#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import onnx
import polars as pl
import seaborn as sns
from __init__ import (
    CLASSIFICATION_THRESHOLD_PATH,
    MODEL_ONNX_PATH,
    PLOT_DIR,
    RETRAIN_VECTORIZER,
    RETRAINED_MODEL_VERSION,
    SEED,
    VECTORIZER_BIN_PATH,
    VECTORIZER_JSON_PATH,
    ProbabilisticClassifier,
    df_test,
    df_train,
)
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams, __version__
from loguru import logger
from onnxruntime.transformers.onnx_model import OnnxModel
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.ensemble import VotingClassifier
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

from .plots import (
    analyze_features_by_ngram_length,
    artifact_position_analysis,
    compare_token_distributions,
    compute_best_thresholds,
    dataset_bias_analysis,
    decision_boundary_analysis,
    embedding_visualization,
    per_dataset_accuracy_analysis,
    plot_calibration_curves,
    plot_prediction_distributions,
    roc_curve_analysis,
)

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
plt.rcParams["savefig.dpi"] = 1200
plt.rcParams["figure.figsize"] = (12, 8)

print("Vectorizer exists:", VECTORIZER_BIN_PATH.exists())


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

if RETRAIN_VECTORIZER or not VECTORIZER_BIN_PATH.exists():
    logger.info("Training new Vectorizer")
    params = VectorizerParams(ngram_range=(2, 4), min_df=50, max_df=0.8)

    # Log vectorizer params
    mlflow.log_param("ngram_range", f"{params.ngram_range}")
    mlflow.log_param("min_df", params.min_df)
    mlflow.log_param("max_df", params.max_df)
    mlflow.log_param("retrain_vectorizer", True)

    (vectorizer, X_train_tfidf) = TfidfVectorizer.fit_transform(X_train, params)
    logger.info(
        f"Fitted vectorizer and transformed training data {X_train_tfidf.shape} in {time.time() - t1:.2f} seconds"
    )
    t2 = time.time()
else:
    logger.info("Loading Pre-trained Vectorizer")

    vectorizer = TfidfVectorizer.load(VECTORIZER_BIN_PATH)
    mlflow.log_param("retrain_vectorizer", False)
    logger.info(f"Loaded vectorizer in {time.time() - t1:.2f} seconds")
    t2 = time.time()
    X_train_tfidf = vectorizer.transform(X_train)
    logger.info(f"Transformed training data {X_train_tfidf.shape} in {time.time() - t2:.2f} seconds")
    t2 = time.time()
logger.info("Transforming test data...")
X_test_tfidf = vectorizer.transform(X_test)

logger.info(f"Transformed test data {X_test_tfidf.shape} in {time.time() - t2:.2f} seconds")
logger.info(f"Train Feature matrix: {X_train_tfidf.shape}")
sparsity = 100 * (1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape))  # pyright: ignore[reportCallIssue, reportArgumentType]
logger.info(f"Sparsity: {sparsity:.2f}%")

# Log feature matrix metrics
mlflow.log_metric("n_features", X_train_tfidf.shape[1])  # pyright: ignore[reportOptionalSubscript]
mlflow.log_metric("sparsity_percent", sparsity)


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
    max_iter=8000,
    tol=1e-4,
    random_state=SEED,
    learning_rate="optimal",
    n_jobs=-1,
)

logreg = LogisticRegression(
    penalty="l2", C=1.0, solver="saga", max_iter=1000, class_weight="balanced", random_state=SEED, n_jobs=-1
)

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
    ("sgd", sgd, 0.25),
    ("logreg", logreg, 0.30),
    ("svc", svc_calibrated, 0.30),
    ("nb", nb, 0.15),
    # ("cnb", cn, 0.05),
]
voting = "soft"

assert abs(sum(weight for _, _, weight in estimators) - 1.0) < 1e-6, "Weights must sum to 1.0"  # noqa: S101

mlflow.log_param("ensemble_estimators", [name for name, _, _ in estimators])
mlflow.log_param("ensemble_weights", [weight for _, _, weight in estimators])
mlflow.log_param("model_type", "VotingClassifier")
mlflow.log_param("voting", voting)

ensemble = VotingClassifier(
    estimators=[(name, model) for name, model, _ in estimators],
    weights=[weight for _, _, weight in estimators],
    voting=voting,
    n_jobs=-1,
    flatten_transform=False,
    verbose=True,
)


# Retrain
ensemble.fit(X_train_tfidf, y_train)
# This is just a list but to save to onnx we need it as a numpy array
ensemble.weights = np.array(ensemble.weights)  # pyright: ignore[reportAttributeAccessIssue]

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


y_probs = ensemble.predict_proba(X_test_tfidf)[:, 1]
y_probs


# In[ ]:


roc_curve_analysis(X_train_tfidf, y_train, X_test_tfidf, y_test, models)


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


# Save vectorizer in both formats:
# 1. JSON-wrapped format for Python (with metadata)
vectorizer.save(VECTORIZER_JSON_PATH)
logger.info(f"Saved json vectorizer to {VECTORIZER_JSON_PATH}")
# 2. Raw bincode format for Rust (no JSON wrapper)
vectorizer.save(VECTORIZER_BIN_PATH)
logger.info(f"Saved binary vectorizer to {VECTORIZER_BIN_PATH}")

Path(CLASSIFICATION_THRESHOLD_PATH).write_text(str(best_threshold), encoding="utf-8")
logger.info(f"Saved classification threshold to {CLASSIFICATION_THRESHOLD_PATH}")
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
mlflow.log_artifact(str(VECTORIZER_JSON_PATH))
mlflow.log_artifact(str(CLASSIFICATION_THRESHOLD_PATH))


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


assert np.allclose(pred_onx[1], model_pred)  # pyright: ignore[reportArgumentType]  # noqa: S101


# In[ ]:


plot_prediction_distributions(X_test_tfidf, y_test, models)


# In[ ]:


plot_calibration_curves(X_test_tfidf, y_test, models)


# In[ ]:


decision_boundary_analysis(
    X_test_tfidf, y_test, ensemble.predict_proba(X_test_tfidf), decision_threshold=best_threshold
)


# In[ ]:


analyze_features_by_ngram_length(vectorizer, models, top_n=20)


# In[ ]:


dataset_bias_analysis(df_test.collect().to_pandas(), ensemble.predict_proba(X_test_tfidf), best_threshold)


# In[ ]:


embedding_visualization(
    X_test_tfidf, y_test, df_test.select("dataset").collect().to_series().to_numpy(), sample_size=20_000
)


# In[ ]:


texts_human = df_test.filter(pl.col("label") == 0).select("text").collect().to_series()
texts_ai = df_test.filter(pl.col("label") == 1).select("text").collect().to_series()
compare_token_distributions(texts_human, texts_ai)


# In[ ]:


artifact_position_analysis(
    df_test.select("text").collect().to_series().to_list(), y_test, vectorizer,
    ensemble,  # type: ignore[reportArgumentType]
    best_threshold
)


# In[ ]:


per_dataset_accuracy_analysis(X_test_tfidf, models["svc"], threshold=best_threshold)


# In[ ]:


# End MLflow run
mlflow.end_run()
logger.info("MLflow run completed")

