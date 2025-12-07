#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import tiktoken
from __init__ import PLOT_DIR, RETRAIN_VECTORIZER, RETRAINED_MODEL_VERSION, SEED, VECTORIZER_BIN_PATH, df_test, df_train
from is_it_slop_preprocessing import TfidfVectorizer, VectorizerParams, __version__
from loguru import logger
from matplotlib import gridspec
from scipy.sparse import csr_matrix
from scipy.stats import entropy, gaussian_kde
from skl2onnx import to_onnx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, LinearSVC, calibration_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


class ProbabilisticClassifier(Protocol):
    def predict_proba(self, X) -> np.ndarray: ...  # noqa: ANN001
    def predict(self, X) -> np.ndarray: ...  # noqa: ANN001
    def fit(self, X, y) -> "ProbabilisticClassifier": ...  # noqa: ANN001
    @property
    def coef_(self) -> np.ndarray: ...


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


def roc_curve_analysis(
    X_train_tfidf: csr_matrix, y_train: np.ndarray, X_test_tfidf: csr_matrix, y_test: np.ndarray, models: dict
) -> None:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_tfidf, y_train)
    models_with_dummy = models.copy()
    models_with_dummy["dummy"] = dummy

    _fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    ax_roc.grid(linestyle="--")
    ax_det.grid(linestyle="--")

    for name, clf in models_with_dummy.items():
        (color, linestyle) = ("blue", "-") if name == "dummy" else (None, None)
        y_pred_ = clf.predict_proba(X_test_tfidf)[:, 1]
        # y_pred = clf.predict_proba(X_test_tfidf)[:, 1] if name != "dummy" else clf.predict(X_test_tfidf)

        RocCurveDisplay.from_predictions(
            y_test, y_pred_, ax=ax_roc, name=name, curve_kwargs={"color": color, "linestyle": linestyle}
        )
        DetCurveDisplay.from_predictions(y_test, y_pred_, ax=ax_det, name=name, color=color, linestyle=linestyle)
    plt.legend()
    plot_path = PLOT_DIR / "roc_det_curve_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


roc_curve_analysis(X_train_tfidf, y_train, X_test_tfidf, y_test, models)


# In[ ]:


# 2) use precision-recall curve for exact best F1 (thresholds length differs)


def compute_best_thresholds(y_test: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    plt.figure(figsize=(10, 8))

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2, color="#3498db")
    plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2, color="#e74c3c")

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, color="#2ecc71", linestyle="--")

    best_idx = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold (Precision-Recall curve): {best_threshold:.4f} with F1: {f1_scores[best_idx]:.4f}")

    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_test, probs)
    youden = true_positive_rate - false_positive_rate
    best_idx_roc = np.argmax(youden)
    best_threshold_roc = roc_thresholds[best_idx_roc]
    logger.info(
        f"Best threshold (Youden's J statistic): {best_threshold_roc:.4f} with Youden: {youden[best_idx_roc]:.4f}"
    )
    aoc_score = auc(false_positive_rate, true_positive_rate)
    logger.info(f"ROC AUC: {aoc_score:.4f}")
    plt.plot(roc_thresholds, youden, label="Youden's J Statistic", linewidth=2, color="#9b59b6", linestyle=":")

    plt.axvline(best_threshold, color="#2ecc71", linestyle="--", label=f"Best Threshold: {best_threshold:.4f}")
    plt.axvline(
        best_threshold_roc, color="#9b59b6", linestyle="--", label=f"Best Youden Threshold: {best_threshold_roc:.4f}"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 Score vs. Threshold")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plot_path = PLOT_DIR / "precision_recall_f1_thresholds.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))

    return best_threshold, best_threshold_roc


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

# 2025-12-01 12:46:23.856 | INFO     | __main__:<module>:3 - Validation MCC: 0.9267
# 2025-12-01 12:46:23.860 | INFO     | __main__:<module>:6 - Validation AUC: 0.9634
# 2025-12-01 12:46:23.862 | INFO     | __main__:<module>:8 - Accuracy:   0.9634
# 2025-12-01 12:46:23.866 | INFO     | __main__:<module>:10 - Precision:  0.9660
# 2025-12-01 12:46:23.871 | INFO     | __main__:<module>:12 - Recall:     0.9613
# 2025-12-01 12:46:23.878 | INFO     | __main__:<module>:14 - F1 Score:   0.9636
# 2025-12-01 12:46:23.880 | INFO     | __main__:<module>:17 - TP: 22074, FP: 778, TN: 21742, FN: 888
# 2025-12-01 12:46:23.881 | INFO     | __main__:<module>:18 - Confusion Matrix:
# 2025-12-01 12:46:23.881 | INFO     | __main__:<module>:19 -               Predicted
# 2025-12-01 12:46:23.882 | INFO     | __main__:<module>:20 -                  0      1
# 2025-12-01 12:46:23.882 | INFO     | __main__:<module>:21 - Actual  0    21742    778
# 2025-12-01 12:46:23.883 | INFO     | __main__:<module>:22 -         1      888  22074


# 2025-12-01 12:58:52.663 | INFO     | __main__:<module>:3 - Validation MCC: 0.9284
# 2025-12-01 12:58:52.668 | INFO     | __main__:<module>:6 - Validation AUC: 0.9642
# 2025-12-01 12:58:52.670 | INFO     | __main__:<module>:8 - Accuracy:   0.9642
# 2025-12-01 12:58:52.676 | INFO     | __main__:<module>:10 - Precision:  0.9685
# 2025-12-01 12:58:52.682 | INFO     | __main__:<module>:12 - Recall:     0.9603
# 2025-12-01 12:58:52.686 | INFO     | __main__:<module>:14 - F1 Score:   0.9644
# 2025-12-01 12:58:52.688 | INFO     | __main__:<module>:17 - TP: 22051, FP: 717, TN: 21803, FN: 911
# 2025-12-01 12:58:52.689 | INFO     | __main__:<module>:18 - Confusion Matrix:
# 2025-12-01 12:58:52.689 | INFO     | __main__:<module>:19 -               Predicted
# 2025-12-01 12:58:52.689 | INFO     | __main__:<module>:20 -                  0      1
# 2025-12-01 12:58:52.689 | INFO     | __main__:<module>:21 - Actual  0    21803    717
# 2025-12-01 12:58:52.690 | INFO     | __main__:<module>:22 -         1      911  22051


# In[ ]:


import onnx
from __init__ import CLASSIFICATION_THRESHOLD_PATH, MODEL_ONNX_PATH, VECTORIZER_JSON_PATH
from onnxruntime.transformers.onnx_model import OnnxModel

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


def plot_prediction_distributions(X_test_tfidf: csr_matrix, y_test: np.ndarray, models: dict) -> None:
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, :]),  # This spans both columns in the last row
    ]
    # fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    # axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        probs_ = model.predict_proba(X_test_tfidf)[:, 1]
        ax = axes[idx]

        # Separate by true label
        human_probs = probs_[y_test == 0]
        ai_probs = probs_[y_test == 1]

        ax.hist(human_probs, bins=50, alpha=0.5, label="Human (true)", color="blue")
        ax.hist(ai_probs, bins=50, alpha=0.5, label="AI (true)", color="red")
        ax.set_xlabel("Predicted Probability (AI class)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name.upper()} - Prediction Distribution")
        ax.legend()
        ax.set_yscale("log")  # Log scale to see tails
        human_max = human_probs.max()
        ai_min = ai_probs.min()
        print(f"{name:10s} - Human max prob: {human_max:.4f}, AI min prob: {ai_min:.4f}")
        if human_max < ai_min:
            print(f"           -> PERFECT SEPARATION! Gap: {ai_min - human_max:.4f}")
        else:
            print(f"           -> Overlap region: {human_max - ai_min:.4f}")

    plt.tight_layout()
    plot_path = PLOT_DIR / "model_prediction_distributions.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


plot_prediction_distributions(X_test_tfidf, y_test, models)


# In[ ]:


def plot_calibration_curves(X_test_tfidf: csr_matrix, y_test: np.ndarray, models: dict) -> None:

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap("Accent", len(models))

    for idx, (name, model) in enumerate(models.items()):
        prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name.upper(), color=colors(idx))

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves (All Models)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = PLOT_DIR / "calibration_curves.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


plot_calibration_curves(X_test_tfidf, y_test, models)


# In[ ]:


def decision_boundary_analysis(
    X_tfidf: csr_matrix | np.ndarray,
    y: np.ndarray,
    y_pred_proba: np.ndarray,
    sample_size: int = 3000,
    decision_threshold: float = 0.5,
) -> None:
    """Analyze model decision boundary characteristics.

    Reveals:
    - Confidence distribution
    - Calibration quality
    - Uncertainty regions
    """
    # Sample for performance
    if len(y) > sample_size:
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(y), sample_size, replace=False)
        X_sample = X_tfidf[indices]
        y_sample = y[indices]
        proba_sample = y_pred_proba[indices]
    else:
        X_sample = X_tfidf
        y_sample = y
        proba_sample = y_pred_proba

    _fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Confidence distribution by class
    ax = axes[0, 0]

    human_probs = proba_sample[y_sample == 0, 1]
    ai_probs = proba_sample[y_sample == 1, 1]

    ax.hist(human_probs, bins=50, alpha=0.6, color="#3498db", label="Human (true)", density=True)
    ax.hist(ai_probs, bins=50, alpha=0.6, color="#e74c3c", label="AI (true)", density=True)
    ax.axvline(x=decision_threshold, color="black", linestyle="--", linewidth=2, label="Threshold")
    ax.set_xlabel("Predicted Probability (AI class)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confidence distribution (log scale)
    ax = axes[0, 1]

    ax.hist(human_probs, bins=50, alpha=0.6, color="#3498db", label="Human (true)", density=True)
    ax.hist(ai_probs, bins=50, alpha=0.6, color="#e74c3c", label="AI (true)", density=True)
    ax.axvline(x=decision_threshold, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Probability (AI class)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Prediction Distribution (Log Scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Calibration curve
    ax = axes[0, 2]

    prob_true, prob_pred = calibration_curve(y_sample, proba_sample[:, 1], n_bins=10, strategy="uniform")

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, markersize=8, color="#e67e22", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Uncertainty vs correctness
    ax = axes[1, 0]

    # Calculate uncertainty (distance from decision boundary)
    uncertainty = np.abs(proba_sample[:, 1] - decision_threshold)
    correct = (proba_sample[:, 1] > decision_threshold) == y_sample

    correct_uncertainty = uncertainty[correct]
    incorrect_uncertainty = uncertainty[~correct]

    ax.hist(correct_uncertainty, bins=30, alpha=0.6, color="#2ecc71", label="Correct", density=True)
    ax.hist(incorrect_uncertainty, bins=30, alpha=0.6, color="#e74c3c", label="Incorrect", density=True)
    ax.set_xlabel(f"Uncertainty (distance from threshold {decision_threshold:.4f})")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution by Correctness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Precision-Recall by threshold
    ax = axes[1, 1]

    precision, recall, thresholds = precision_recall_curve(y_sample, proba_sample[:, 1])

    ax.plot(thresholds, precision[:-1], label="Precision", linewidth=2, color="#3498db")
    ax.plot(thresholds, recall[:-1], label="Recall", linewidth=2, color="#e74c3c")

    # F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    ax.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, color="#2ecc71", linestyle="--")

    ax.axvline(x=decision_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall, F1 vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 6. Confusion regions in 2D projection
    ax = axes[1, 2]

    # Use first 2 SVD components for visualization
    print("Computing 2D projection for decision boundary...")
    svd_2d = TruncatedSVD(n_components=2, random_state=SEED)
    X_2d = svd_2d.fit_transform(X_sample)

    # Create meshgrid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    _xx, _yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Plot decision regions
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=proba_sample[:, 1],
        cmap="RdYlBu_r",
        s=20,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel("First SVD Component")
    ax.set_ylabel("Second SVD Component")
    ax.set_title("Decision Space (2D Projection)")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted Prob (AI)")
    plt.tight_layout()
    plot_path = PLOT_DIR / "decision_boundary_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


decision_boundary_analysis(
    X_test_tfidf, y_test, ensemble.predict_proba(X_test_tfidf), decision_threshold=best_threshold
)


# In[ ]:


def analyze_features_by_ngram_length(vectorizer: TfidfVectorizer, models, top_n: int = 20) -> None:  # noqa: ANN001

    vocab = vectorizer.vocabulary

    for name, model in models.items():
        print(f"\n\nFeature analysis for {name.upper()}:")

        try:
            coefs = model.coef_[0]
        except AttributeError:
            print(f"Model {name} does not have coef_ attribute, skipping feature analysis.")
            continue
        # Get top features for each class
        top_ai_indices = np.argsort(coefs)[-top_n:][::-1]
        top_human_indices = np.argsort(coefs)[:top_n]

        # Reverse vocabulary lookup
        idx_to_ngram = {idx: ngram for ngram, idx in vocab.items()}

        print(f"Top {top_n} features predicting AI text:")
        for idx in top_ai_indices:
            if idx in idx_to_ngram:
                print(f"  '{idx_to_ngram[idx]}': {coefs[idx]:.4f}")

        print(f"\nTop {top_n} features predicting Human text:")
        for idx in top_human_indices:
            if idx in idx_to_ngram:
                print(f"  '{idx_to_ngram[idx]}': {coefs[idx]:.4f}")


analyze_features_by_ngram_length(vectorizer, models, top_n=20)


# In[ ]:


def dataset_bias_analysis(df_test: pd.DataFrame, y_pred_proba: np.ndarray, decision_threshold: float = 0.5) -> None:
    """Analyze dataset-specific biases and patterns.

    Reveals:
    - Per-dataset prediction distributions
    - Dataset separability (potential artifacts)
    - Source-specific biases
    """
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df_analysis = df_test.copy()
    df_analysis["pred_proba_ai"] = y_pred_proba[:, 1]
    df_analysis["pred_label"] = (y_pred_proba[:, 1] > decision_threshold).astype(int)
    df_analysis["correct"] = df_analysis["pred_label"] == df_analysis["label"]

    # 1. Prediction distribution by dataset
    ax = axes[0, 0]

    datasets = df_analysis["dataset"].unique()
    datasets_sorted = sorted(datasets, key=lambda d: df_analysis[df_analysis["dataset"] == d]["pred_proba_ai"].mean())

    data_violin = [
        df_analysis[df_analysis["dataset"] == d]["pred_proba_ai"].to_numpy() for d in datasets_sorted[:15]
    ]  # Top 15 for readability

    parts = ax.violinplot(data_violin, positions=range(len(data_violin)), showmeans=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor("#9b59b6")
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(datasets_sorted[:15])))
    ax.set_xticklabels([d[:20] for d in datasets_sorted[:15]], rotation=45, ha="right")
    ax.set_ylabel("Predicted Probability (AI class)")
    ax.set_title("Prediction Distribution by Dataset (Top 15)")
    ax.axhline(y=decision_threshold, color="red", linestyle="--", linewidth=1, label="Decision Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Dataset accuracy vs AI proportion
    ax = axes[0, 1]

    dataset_stats = (
        df_analysis.groupby("dataset").agg({"correct": "mean", "label": "mean", "pred_proba_ai": "mean"}).reset_index()
    )

    dataset_stats.columns = ["dataset", "accuracy", "true_ai_ratio", "pred_ai_avg"]

    scatter = ax.scatter(
        dataset_stats["true_ai_ratio"],
        dataset_stats["accuracy"],
        s=dataset_stats["pred_ai_avg"] * 500,
        c=dataset_stats["pred_ai_avg"],
        cmap="RdYlBu_r",
        alpha=0.6,
        edgecolors="black",
        linewidth=1,
    )

    ax.set_xlabel("True AI Ratio in Dataset")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Dataset Accuracy vs AI Content Ratio\n(bubble size = avg predicted AI prob)")
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Avg Predicted AI Prob")

    # Annotate outliers
    for _, row in dataset_stats.iterrows():
        if row["accuracy"] < 0.85 or abs(row["true_ai_ratio"] - row["pred_ai_avg"]) > 0.3:
            ax.annotate(
                row["dataset"][:15],
                xy=(row["true_ai_ratio"], row["accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )

    # 3. Confusion heatmap by dataset
    ax = axes[1, 0]

    # Get top datasets by size
    top_datasets = df_analysis["dataset"].value_counts().head(12).index
    df_top = df_analysis[df_analysis["dataset"].isin(top_datasets)]

    # Create confusion matrix per dataset
    confusion_data = []
    dataset_labels = []

    for dataset in top_datasets:
        df_ds = df_top[df_top["dataset"] == dataset]
        tp = ((df_ds["label"] == 1) & (df_ds["pred_label"] == 1)).sum()
        fp = ((df_ds["label"] == 0) & (df_ds["pred_label"] == 1)).sum()
        tn = ((df_ds["label"] == 0) & (df_ds["pred_label"] == 0)).sum()
        fn = ((df_ds["label"] == 1) & (df_ds["pred_label"] == 0)).sum()

        total = tp + fp + tn + fn
        confusion_data.append([tn / total, fp / total, fn / total, tp / total])
        dataset_labels.append(dataset[:20])

    confusion_matrix = np.array(confusion_data)

    im = ax.imshow(confusion_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["TN", "FP", "FN", "TP"])
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels, fontsize=8)
    ax.set_title("Normalized Confusion Matrix by Dataset")

    plt.colorbar(im, ax=ax, label="Proportion")

    # 4. Error rate by dataset
    ax = axes[1, 1]

    error_rates = df_analysis.groupby("dataset").agg({"correct": lambda x: 1 - x.mean()}).reset_index()
    error_rates.columns = ["dataset", "error_rate"]
    error_rates = error_rates.sort_values("error_rate", ascending=False).head(15)

    colors_err = [
        "#e74c3c" if rate > 0.1 else "#f39c12" if rate > 0.05 else "#2ecc71" for rate in error_rates["error_rate"]
    ]

    y_pos = np.arange(len(error_rates))
    ax.barh(y_pos, error_rates["error_rate"], color=colors_err, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d[:25] for d in error_rates["dataset"]], fontsize=8)
    ax.set_xlabel("Error Rate")
    ax.set_title("Top 15 Datasets by Error Rate")
    ax.invert_yaxis()
    ax.axvline(x=0.05, color="orange", linestyle="--", linewidth=1, label="5% threshold")
    ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1, label="10% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plot_path = PLOT_DIR / "dataset_bias_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


dataset_bias_analysis(df_test.collect().to_pandas(), ensemble.predict_proba(X_test_tfidf), best_threshold)


# In[ ]:


def embedding_visualization(
    X_tfidf: csr_matrix, y: np.ndarray, dataset_labels: np.ndarray, sample_size: int = 10_000
) -> None:
    """Comprehensive visualization combining label-based and dataset-based embeddings.

    Shows 6 subplots:
    1. t-SNE colored by true label (human/AI)
    2. t-SNE colored by dataset source
    3. Class density contours
    4. Dataset centroids in t-SNE space
    5. SVD explained variance
    6. Class centroids with separation metric
    """
    # Sample for performance
    if len(y) > sample_size:
        rng = np.random.default_rng(SEED)
        indices = rng.choice(len(y), sample_size, replace=False)
        X_sample = X_tfidf[indices]
        y_sample = y[indices]
        dataset_sample = dataset_labels[indices]
    else:
        X_sample = X_tfidf
        y_sample = y
        dataset_sample = dataset_labels

    print(f"Sample shape: {X_sample.shape}")
    print(f"Reducing from {X_sample.shape[1]} to 50 dimensions...")

    # SVD reduction: high-dim -> 50 dimensions
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_sample)
    print(f"SVD complete. Shape: {X_svd.shape}")

    # t-SNE: 50 dimensions -> 2 dimensions
    print("Computing t-SNE embedding (50 -> 2 dimensions)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_svd)
    print(f"t-SNE complete. Shape: {X_tsne.shape}")

    # Create figure with 3x2 layout
    _fig, axes = plt.subplots(3, 2, figsize=(18, 24))

    human_mask = y_sample == 0
    ai_mask = y_sample == 1

    # ============================================================
    # 1. t-SNE coloured by label (human/AI)
    # ============================================================
    ax = axes[0, 0]

    ax.scatter(
        X_tsne[human_mask, 0],
        X_tsne[human_mask, 1],
        c="#3498db",
        alpha=0.4,
        s=20,
        label=f"Human (n={human_mask.sum()})",
        edgecolors="none",
    )
    ax.scatter(
        X_tsne[ai_mask, 0],
        X_tsne[ai_mask, 1],
        c="#e74c3c",
        alpha=0.4,
        s=20,
        label=f"AI (n={ai_mask.sum()})",
        edgecolors="none",
    )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("t-SNE Embedding (Colored by True Label)", fontsize=13, fontweight="bold")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # 2. t-SNE colored by dataset
    # ============================================================
    ax = axes[0, 1]

    unique_datasets = np.unique(dataset_sample)
    n_datasets = len(unique_datasets)
    colors_ds = plt.cm.tab20(np.linspace(0, 1, n_datasets))  # type: ignore[AttributeAccessIssue]

    for i, dataset in enumerate(unique_datasets[:20]):  # Limit to 20 for visibility
        mask = dataset_sample == dataset
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1], c=[colors_ds[i]], alpha=0.5, s=15, label=dataset[:15], edgecolors="none"
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("t-SNE Embedding (Colored by Dataset)", fontsize=13, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(1.05, 1),  # loc="upper left",
        fontsize=7,
        ncol=2,
    )
    ax.grid(True, alpha=0.3)

    # ============================================================
    # 3. Density contours by class
    # ============================================================
    ax = axes[1, 0]

    # Calculate KDE bounds
    x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # KDE for human points
    if human_mask.sum() > 10:
        kde_human = gaussian_kde(X_tsne[human_mask].T)
        z_human = kde_human(positions).reshape(xx.shape)
        ax.contour(xx, yy, z_human, colors="#3498db", alpha=0.6, linewidths=2, levels=5)

    # KDE for AI points
    if ai_mask.sum() > 10:
        kde_ai = gaussian_kde(X_tsne[ai_mask].T)
        z_ai = kde_ai(positions).reshape(xx.shape)
        ax.contour(xx, yy, z_ai, colors="#e74c3c", alpha=0.6, linewidths=2, levels=5)

    # Scatter on top
    ax.scatter(
        X_tsne[human_mask, 0], X_tsne[human_mask, 1], c="#3498db", alpha=0.2, s=10, label="Human", edgecolors="none"
    )
    ax.scatter(X_tsne[ai_mask, 0], X_tsne[ai_mask, 1], c="#e74c3c", alpha=0.2, s=10, label="AI", edgecolors="none")

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Class Density Contours", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ============================================================
    # 4. Dataset centroids in t-SNE space
    # ============================================================
    ax = axes[1, 1]

    # Compute centroids per dataset
    centroids = []
    centroid_labels = []
    centroid_colors = []

    for dataset in unique_datasets[:15]:
        mask = dataset_sample == dataset
        if mask.sum() > 10:  # Only if sufficient samples
            centroid = X_tsne[mask].mean(axis=0)
            centroids.append(centroid)
            centroid_labels.append(dataset[:15])

            # Color by majority label
            majority_label = y_sample[mask].mean()
            color = "#e74c3c" if majority_label > 0.5 else "#3498db"
            centroid_colors.append(color)

    centroids = np.array(centroids)

    # Plot all points in background (light)
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c="gray", alpha=0.1, s=5, edgecolors="none")

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c=centroid_colors, s=200, alpha=0.8, edgecolors="black", linewidth=2)

    # Annotate centroids
    for i, label in enumerate(centroid_labels):
        ax.annotate(
            label,
            xy=(centroids[i, 0], centroids[i, 1]),
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title(
        "Dataset Centroids in t-SNE Space\n(Blue=Human-majority, Red=AI-majority)", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # ============================================================
    # 5. SVD explained variance
    # ============================================================
    ax = axes[2, 0]

    cumsum_var = np.cumsum(svd.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, linewidth=2, color="#9b59b6")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="50%")
    ax.axhline(y=0.8, color="orange", linestyle="--", linewidth=1, label="80%")
    ax.set_xlabel("Number of SVD Components", fontsize=11)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
    ax.set_title("SVD Variance Explained", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text showing how many components needed
    comp_50 = np.argmax(cumsum_var >= 0.5) + 1 if any(cumsum_var >= 0.5) else 50
    comp_80 = np.argmax(cumsum_var >= 0.8) + 1 if any(cumsum_var >= 0.8) else 50

    ax.text(
        0.05,
        0.95,
        f"50% variance: {comp_50} components\n"
        f"80% variance: {comp_80} components\n"
        f"Total variance (50 comp): {cumsum_var[-1]:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # ============================================================
    # 6. Class centroids with separation metric
    # ============================================================
    ax = axes[2, 1]

    # Calculate centroids
    centroid_human = X_tsne[human_mask].mean(axis=0)
    centroid_ai = X_tsne[ai_mask].mean(axis=0)

    # Plot all points
    ax.scatter(
        X_tsne[human_mask, 0], X_tsne[human_mask, 1], c="#3498db", alpha=0.3, s=15, label="Human", edgecolors="none"
    )
    ax.scatter(X_tsne[ai_mask, 0], X_tsne[ai_mask, 1], c="#e74c3c", alpha=0.3, s=15, label="AI", edgecolors="none")

    # Plot centroids
    ax.scatter(
        *centroid_human, c="blue", s=500, marker="*", edgecolors="black", linewidth=2, label="Human centroid", zorder=5
    )
    ax.scatter(*centroid_ai, c="red", s=500, marker="*", edgecolors="black", linewidth=2, label="AI centroid", zorder=5)

    # Draw line between centroids
    ax.plot(
        [centroid_human[0], centroid_ai[0]],
        [centroid_human[1], centroid_ai[1]],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Centroid separation",
    )

    # Calculate and display separation distance
    separation = np.linalg.norm(centroid_ai - centroid_human)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title(f"Class Centroids (Separation: {separation:.2f})", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    interpretation = "✓ Large separation = easy classification\n"
    interpretation += "⚠ Small separation = difficult task\n"
    interpretation += "✓ Distinct clusters = clear patterns\n"
    interpretation += "⚠ Mixed points = overlapping features"

    ax.text(
        0.02,
        0.98,
        interpretation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / "embedding_visualization.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


# Usage:
embedding_visualization(
    X_test_tfidf, y_test, df_test.select("dataset").collect().to_series().to_numpy(), sample_size=20_000
)


# In[ ]:


def compare_token_distributions(texts_human: pl.Series, texts_ai: pl.Series) -> None:
    """Compare token frequency distributions."""
    enc = tiktoken.get_encoding("o200k_base")

    tokens_human = [t for text in texts_human for t in enc.encode(text)]
    tokens_ai = [t for text in texts_ai for t in enc.encode(text)]

    freq_human = Counter(tokens_human)
    freq_ai = Counter(tokens_ai)

    # Calculate KL divergence
    vocab = set(freq_human.keys()) | set(freq_ai.keys())
    p = np.array([freq_human.get(t, 0) for t in vocab]) + 1e-10
    q = np.array([freq_ai.get(t, 0) for t in vocab]) + 1e-10
    p /= p.sum()
    q /= q.sum()

    kl_div = entropy(p, q)
    print(f"KL divergence (Human || AI): {kl_div:.4f}")

    # Plot token rank distributions (Zipf's law)
    _fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    rank_human = sorted(freq_human.values(), reverse=True)
    rank_ai = sorted(freq_ai.values(), reverse=True)

    ax[0].loglog(rank_human, label="Human")
    ax[0].loglog(rank_ai, label="AI", alpha=0.7)
    ax[0].set_title("Token Frequency Distributions")
    ax[0].legend()

    # Plot unique token counts
    ax[1].bar(["Human", "AI"], [len(freq_human), len(freq_ai)])
    ax[1].set_title("Vocabulary Size")

    plt.tight_layout()
    plot_path = PLOT_DIR / "token_distribution_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


texts_human = df_test.filter(pl.col("label") == 0).select("text").collect().to_series()
texts_ai = df_test.filter(pl.col("label") == 1).select("text").collect().to_series()
compare_token_distributions(texts_human, texts_ai)


# In[ ]:


def artifact_position_analysis(
    texts: list[str],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    model: dict[str, ClassifierMixin],
    decision_threshold: float = 0.5,
) -> None:
    """Analyze if model relies on positional artifacts (start/end of documents).

    Reveals:
    - Positional feature importance
    - Boundary artifact detection
    - Content vs structural learning
    """
    enc = tiktoken.get_encoding("o200k_base")

    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract positional chunks
    start_chunks = []
    middle_chunks = []
    end_chunks = []
    chunk_labels = []

    print(f"Extracting positional chunks from {len(texts)} texts...")
    for text, label in zip(texts, labels, strict=False):
        tokens = enc.encode(text)

        if len(tokens) < 100:
            continue

        # Take first/last/middle 50 tokens
        chunk_size = min(50, len(tokens) // 5)

        start = enc.decode(tokens[:chunk_size])
        end = enc.decode(tokens[-chunk_size:])
        mid_start = len(tokens) // 2 - chunk_size // 2
        middle = enc.decode(tokens[mid_start : mid_start + chunk_size])

        start_chunks.append(start)
        middle_chunks.append(middle)
        end_chunks.append(end)
        chunk_labels.append(label)

    chunk_labels = np.array(chunk_labels)

    # Vectorize chunks
    print("Vectorizing chunks...")
    X_start = vectorizer.transform(start_chunks)
    X_middle = vectorizer.transform(middle_chunks)
    X_end = vectorizer.transform(end_chunks)

    # Predict on chunks
    prob_start = model.predict_proba(X_start)[:, 1]
    prob_middle = model.predict_proba(X_middle)[:, 1]
    prob_end = model.predict_proba(X_end)[:, 1]

    # 1. Position prediction comparison
    ax = axes[0, 0]

    positions = ["Start", "Middle", "End"]
    human_mask = chunk_labels == 0
    ai_mask = chunk_labels == 1

    data_human = [prob_start[human_mask], prob_middle[human_mask], prob_end[human_mask]]

    data_ai = [prob_start[ai_mask], prob_middle[ai_mask], prob_end[ai_mask]]

    x = np.arange(len(positions))
    width = 0.35

    means_human = [np.mean(d) for d in data_human]
    stds_human = [np.std(d) for d in data_human]
    means_ai = [np.mean(d) for d in data_ai]
    stds_ai = [np.std(d) for d in data_ai]

    ax.bar(
        x - width / 2, means_human, width, yerr=stds_human, label="Human (true)", color="#3498db", alpha=0.7, capsize=5
    )
    ax.bar(x + width / 2, means_ai, width, yerr=stds_ai, label="AI (true)", color="#e74c3c", alpha=0.7, capsize=5)

    ax.set_ylabel("Mean Predicted Probability (AI class)")
    ax.set_title("Model Predictions by Document Position")
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()
    ax.axhline(y=decision_threshold, color="black", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Position correlation scatter
    ax = axes[0, 1]

    ax.scatter(prob_start[human_mask], prob_end[human_mask], alpha=0.4, s=20, color="#3498db", label="Human")
    ax.scatter(prob_start[ai_mask], prob_end[ai_mask], alpha=0.4, s=20, color="#e74c3c", label="AI")

    # Line of best fit (all points)
    all_x = np.concatenate([prob_start[human_mask], prob_start[ai_mask]])
    all_y = np.concatenate([prob_end[human_mask], prob_end[ai_mask]])
    m, b = np.polyfit(all_x, all_y, 1)
    ax.plot([0, 1], [m * 0 + b, m * 1 + b], color="black", linestyle=":", linewidth=2, label="Best fit")
    # Perpendicular separation line through the mean
    x_mean = np.mean(all_x)
    y_mean = np.mean(all_y)
    m_perp = -1 / m if m != 0 else 0  # Perpendicular slope
    # y = m_perp * (x - x_mean) + y_mean
    x_vals = np.array([0, 1])
    y_perp = m_perp * (x_vals - x_mean) + y_mean
    ax.plot(x_vals, y_perp, color="purple", linestyle="--", linewidth=2, label="Separation boundary")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Prediction on START chunk")
    ax.set_ylabel("Prediction on END chunk")
    ax.set_title("Start vs End Chunk Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Calculate correlation
    corr_start_end = np.corrcoef(prob_start, prob_end)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr_start_end:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # 3. Accuracy by position
    ax = axes[1, 0]

    acc_start = ((prob_start > decision_threshold) == chunk_labels).mean()
    acc_middle = ((prob_middle > decision_threshold) == chunk_labels).mean()
    acc_end = ((prob_end > decision_threshold) == chunk_labels).mean()

    accuracies = [acc_start, acc_middle, acc_end]
    colors_acc = ["#e74c3c" if a == max(accuracies) else "#3498db" for a in accuracies]

    bars = ax.bar(positions, accuracies, color=colors_acc, alpha=0.7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy by Position")
    ax.set_ylim([0, 1])

    # Add value labels
    for bar, acc in zip(bars, accuracies, strict=False):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")

    # 4. Position variance analysis
    ax = axes[1, 1]

    # Calculate prediction variance across positions for each sample
    all_probs = np.stack([prob_start, prob_middle, prob_end], axis=1)
    variances = np.var(all_probs, axis=1)

    # Split by correctness
    correct_mask = (prob_middle > decision_threshold) == chunk_labels
    var_correct = variances[correct_mask]
    var_incorrect = variances[~correct_mask]

    ax.hist(var_correct, bins=30, alpha=0.6, color="#2ecc71", label="Correct", density=True)
    ax.hist(var_incorrect, bins=30, alpha=0.6, color="#e74c3c", label="Incorrect", density=True)
    ax.set_xlabel("Prediction Variance Across Positions")
    ax.set_ylabel("Density")
    ax.set_title("Position Variance by Correctness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    mean_var_correct = np.mean(var_correct)
    mean_var_incorrect = np.mean(var_incorrect)

    interpretation = "High variance = position-dependent predictions (artifacts)\n"
    interpretation += f"Mean var (correct): {mean_var_correct:.4f}\n"
    interpretation += f"Mean var (incorrect): {mean_var_incorrect:.4f}"

    ax.text(
        0.98,
        0.97,
        interpretation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / "artifact_position_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))


artifact_position_analysis(
    df_test.select("text").collect().to_series().to_list(), y_test, vectorizer, ensemble, best_threshold
)


# In[ ]:


# Get predictions
def per_dataset_accuracy_analysis(df_test: pl.LazyFrame) -> None:
    svc_probs = models["svc"].predict_proba(X_test_tfidf)[:, 1]
    svc_pred = (svc_probs >= 0.444).astype(int)

    # Add predictions to dataframe
    df_test_full = df_test.with_columns([
        pl.Series("prediction", svc_pred),
        pl.Series("prob_ai", svc_probs),
        pl.Series("correct", (svc_pred == df_test.select("label").collect().to_series().to_numpy()).astype(int)),
    ])

    # Accuracy by dataset source
    accuracy_by_dataset = (
        df_test_full.group_by("dataset")
        .agg([
            pl.len().alias("count"),
            pl.col("correct").mean().alias("accuracy"),
            pl.col("prob_ai").mean().alias("avg_prob_ai"),
        ])
        .sort("accuracy")
        .collect()
    )

    print("\nAccuracy by Dataset Source:")
    print(accuracy_by_dataset)

    # Find the hardest/easiest datasets
    print("\nEasiest datasets (might be artifacts):")
    print(accuracy_by_dataset.tail(5))

    print("\nHardest datasets (more realistic):")
    print(accuracy_by_dataset.head(5))


per_dataset_accuracy_analysis(df_test, y_pred)


# In[ ]:


# End MLflow run
mlflow.end_run()
logger.info("MLflow run completed")

