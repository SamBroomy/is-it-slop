from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import tiktoken
from __init__ import PLOT_DIR, SEED, ProbabilisticClassifier, df_test
from is_it_slop_preprocessing import TfidfVectorizer, reverse_tokenize, tokenize
from loguru import logger
from matplotlib import gridspec
from scipy.sparse import csr_matrix
from scipy.stats import entropy, gaussian_kde
from sklearn.calibration import calibration_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, auc, precision_recall_curve, roc_curve


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
                print(f"  '{idx_to_ngram[idx].strip('!')}': {coefs[idx]:.4f}")

        print(f"\nTop {top_n} features predicting Human text:")
        for idx in top_human_indices:
            if idx in idx_to_ngram:
                print(f"  '{idx_to_ngram[idx].strip('!')}': {coefs[idx]:.4f}")


def dataset_bias_analysis(
    df_test: pd.DataFrame, y_probs: np.ndarray, y_pred: np.ndarray, decision_threshold: float = 0.5
) -> None:
    """Analyze dataset-specific biases and patterns.

    Reveals:
    - Per-dataset prediction distributions
    - Dataset separability (potential artifacts)
    - Source-specific biases
    """
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    df_analysis = df_test.copy()
    df_analysis["pred_proba_ai"] = y_probs
    df_analysis["pred_label"] = y_pred.astype(int)
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


def embedding_visualization(
    X_tfidf: csr_matrix, y: np.ndarray, dataset_labels: np.ndarray, sample_size: int = 10_000
) -> None:
    """Comprehensive visualization combining label-based and dataset-based embeddings.

    Shows 6 subplots:
    1. t-SNE colored by true label (human/AI)
    2. t-SNE colored by dataset source
    3. Class density contours
    4. Dataset centroids in t-SNE space
    5. Feature sparsity distribution
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
    print(f"Reducing from {X_sample.shape[1]} to 50 dimensions...")  # type: ignore[reportOptionalSubscript]

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
    # 5. Feature sparsity distribution
    # ============================================================
    ax = axes[2, 0]

    # Calculate sparsity (proportion of zeros) for each sample
    if hasattr(X_sample, "toarray"):
        # Sparse matrix
        sparsity_per_sample = 1 - (X_sample.getnnz(axis=1) / X_sample.shape[1])
    else:
        # Dense matrix
        sparsity_per_sample = (X_sample == 0).mean(axis=1)

    # Plot histogram split by class
    ax.hist(
        sparsity_per_sample[human_mask],
        bins=50,
        alpha=0.6,
        color="#3498db",
        label=f"Human (μ={sparsity_per_sample[human_mask].mean():.3f})",
        density=True,
    )
    ax.hist(
        sparsity_per_sample[ai_mask],
        bins=50,
        alpha=0.6,
        color="#e74c3c",
        label=f"AI (μ={sparsity_per_sample[ai_mask].mean():.3f})",
        density=True,
    )

    ax.set_xlabel("Sparsity (proportion of zero features)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("TF-IDF Feature Sparsity Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    mean_sparsity_human = sparsity_per_sample[human_mask].mean()
    mean_sparsity_ai = sparsity_per_sample[ai_mask].mean()

    interpretation = "Higher sparsity = fewer active features\n"
    if abs(mean_sparsity_human - mean_sparsity_ai) > 0.05:
        interpretation += "Significant difference detected:\n"
        interpretation += f"{'Human' if mean_sparsity_human > mean_sparsity_ai else 'AI'} texts are more sparse"
    else:
        interpretation += "Similar sparsity patterns"

    ax.text(
        0.05,
        0.95,
        interpretation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
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


def artifact_position_analysis(
    texts: list[str],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    model: ProbabilisticClassifier,
    decision_threshold: float = 0.5,
) -> None:

    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract positional chunks
    start_chunks = []
    middle_chunks = []
    end_chunks = []
    chunk_labels = []
    text_tokens = tokenize(texts)

    print(f"Extracting positional chunks from {len(texts)} texts...")
    for tokens, label in zip(text_tokens, labels, strict=False):
        if len(tokens) < 100:
            continue

        # Take first/last/middle 50 tokens
        chunk_size = min(50, len(tokens) // 5)

        start = reverse_tokenize(tokens[:chunk_size])
        end = reverse_tokenize(tokens[-chunk_size:])
        mid_start = len(tokens) // 2 - chunk_size // 2
        middle = reverse_tokenize(tokens[mid_start : mid_start + chunk_size])

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


# Get predictions
def per_dataset_accuracy_analysis(X_test_tfidf: csr_matrix, model: ProbabilisticClassifier, threshold: float) -> None:
    svc_probs = model.predict_proba(X_test_tfidf)[:, 1]
    svc_pred = (svc_probs >= threshold).astype(int)

    # Add predictions to dataframe
    df_test_full = df_test.with_columns([
        pl.Series("prediction", svc_pred),
        pl.Series("prob_ai", svc_probs),
        pl.Series("correct", (svc_pred == df_test.select("label").collect().to_series().to_numpy()).astype(int)),
    ])

    # Accuracy by dataset source
    accuracy_by_dataset = (
        df_test_full
        .group_by("dataset")
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


def top_ngrams_visualization(vectorizer: TfidfVectorizer, model_coef: np.ndarray, top_n: int = 20) -> None:
    """
    Visualize top predictive n-grams for Human and AI classes.

    Shows token n-grams with highest absolute coefficients (most discriminative features).
    N-grams are already decoded in the vocabulary.

    Args:
        vectorizer: Fitted TfidfVectorizer with vocabulary
        model_coef: Model coefficients (shape: n_features,)
        top_n: Number of top n-grams to display per class
    """
    logger.info("Generating top n-grams visualization...")

    # Get vocabulary and coefficients
    # Vocabulary is dict[str, int] where keys are already decoded n-gram strings
    vocab = vectorizer.vocabulary
    coef = model_coef.flatten()

    # Create reverse mapping: index -> n-gram string
    idx_to_ngram = {idx: ngram for ngram, idx in vocab.items()}

    # Sort by coefficient (positive = AI, negative = Human)
    sorted_indices = np.argsort(coef)

    # Top Human-indicative (most negative)
    top_human_indices = sorted_indices[:top_n]
    top_human_ngrams = []
    top_human_coefs = []

    for idx in top_human_indices:
        if idx in idx_to_ngram:
            # N-gram is already decoded as string
            top_human_ngrams.append(idx_to_ngram[idx])
            top_human_coefs.append(coef[idx])

    # Top AI-indicative (most positive)
    top_ai_indices = sorted_indices[-top_n:][::-1]
    top_ai_ngrams = []
    top_ai_coefs = []

    for idx in top_ai_indices:
        if idx in idx_to_ngram:
            # N-gram is already decoded as string
            top_ai_ngrams.append(idx_to_ngram[idx])
            top_ai_coefs.append(coef[idx])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Human-indicative n-grams (negative coefficients)
    y_pos = np.arange(len(top_human_ngrams))
    ax1.barh(y_pos, top_human_coefs, color="#3498db")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'"{ng}"' for ng in top_human_ngrams], fontsize=9)
    ax1.set_xlabel("Coefficient (← More Human)")
    ax1.set_title(f"Top {top_n} Human-Indicative N-grams")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # AI-indicative n-grams (positive coefficients)
    y_pos = np.arange(len(top_ai_ngrams))
    ax2.barh(y_pos, top_ai_coefs, color="#e74c3c")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'"{ng}"' for ng in top_ai_ngrams], fontsize=9)
    ax2.set_xlabel("Coefficient (More AI →)")
    ax2.set_title(f"Top {top_n} AI-Indicative N-grams")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = PLOT_DIR / "top_ngrams_visualization.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Saved top n-grams visualization to {plot_path}")


def chunk_agreement_analysis(
    chunk_probs: np.ndarray,
    chunk_to_doc_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    chunk_threshold: float,
    n_docs: int,
) -> None:
    """
    Analyze chunk agreement patterns within documents.

    Visualizes:
    1. Distribution of chunk agreement scores
    2. Agreement vs prediction correctness
    3. Agreement by document length (number of chunks)

    Args:
        chunk_probs: Per-chunk AI probabilities
        chunk_to_doc_idx: Mapping from chunk index to document index
        y_true: True document labels
        y_pred: Predicted document labels
        chunk_threshold: Threshold for chunk classification
        n_docs: Total number of documents
    """
    logger.info("Generating chunk agreement analysis...")

    # Calculate chunk agreement per document
    doc_agreements = []
    doc_num_chunks = []
    doc_correct = []

    for doc_idx in range(n_docs):
        mask = chunk_to_doc_idx == doc_idx
        if mask.any():
            doc_chunk_probs = chunk_probs[mask]
            num_chunks = len(doc_chunk_probs)

            # Agreement: proportion of chunks with same classification
            chunk_classes = (doc_chunk_probs >= chunk_threshold).astype(int)
            agreement = max(np.sum(chunk_classes == 0) / num_chunks, np.sum(chunk_classes == 1) / num_chunks)

            doc_agreements.append(agreement)
            doc_num_chunks.append(num_chunks)
            doc_correct.append(y_true[doc_idx] == y_pred[doc_idx])

    doc_agreements = np.array(doc_agreements)
    doc_num_chunks = np.array(doc_num_chunks)
    doc_correct = np.array(doc_correct)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Subplot 1: Agreement distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(doc_agreements, bins=50, color="#9b59b6", alpha=0.7, edgecolor="black")
    ax1.axvline(doc_agreements.mean(), color="red", linestyle="--", label=f"Mean: {doc_agreements.mean():.3f}")
    ax1.set_xlabel("Chunk Agreement Score")
    ax1.set_ylabel("Number of Documents")
    ax1.set_title("Distribution of Chunk Agreement")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Subplot 2: Agreement vs correctness
    ax2 = fig.add_subplot(gs[0, 1])
    correct_mask = doc_correct
    incorrect_mask = ~doc_correct

    ax2.scatter(
        doc_agreements[correct_mask],
        np.random.rand(correct_mask.sum()) * 0.4 + 0.5,
        alpha=0.5,
        color="#2ecc71",
        label=f"Correct ({correct_mask.sum()})",
        s=20,
    )
    ax2.scatter(
        doc_agreements[incorrect_mask],
        np.random.rand(incorrect_mask.sum()) * 0.4,
        alpha=0.5,
        color="#e74c3c",
        label=f"Incorrect ({incorrect_mask.sum()})",
        s=20,
    )

    ax2.set_xlabel("Chunk Agreement Score")
    ax2.set_ylabel("Prediction Outcome (jittered)")
    ax2.set_title("Agreement vs Prediction Correctness")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add interpretation text
    avg_correct_agreement = doc_agreements[correct_mask].mean()
    avg_incorrect_agreement = doc_agreements[incorrect_mask].mean()
    ax2.text(
        0.02,
        0.98,
        f"Avg agreement (correct): {avg_correct_agreement:.3f}\n"
        f"Avg agreement (incorrect): {avg_incorrect_agreement:.3f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Subplot 3: Agreement by document length
    ax3 = fig.add_subplot(gs[0, 2])

    # Bin by number of chunks
    chunk_bins = [1, 2, 3, 5, 10, max(doc_num_chunks) + 1]
    chunk_labels = ["1", "2", "3-4", "5-9", "10+"]
    binned_chunks = np.digitize(doc_num_chunks, chunk_bins[:-1])

    agreement_by_length = []
    for i in range(1, len(chunk_bins)):
        mask = binned_chunks == i
        if mask.any():
            agreement_by_length.append(doc_agreements[mask].mean())
        else:
            agreement_by_length.append(0)

    ax3.bar(chunk_labels, agreement_by_length, color="#3498db", alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Number of Chunks in Document")
    ax3.set_ylabel("Average Agreement Score")
    ax3.set_title("Agreement vs Document Length")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = PLOT_DIR / "chunk_agreement_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Saved chunk agreement analysis to {plot_path}")

    # Log summary statistics
    mlflow.log_metric("avg_chunk_agreement", float(doc_agreements.mean()))
    mlflow.log_metric("avg_agreement_correct", float(avg_correct_agreement))
    mlflow.log_metric("avg_agreement_incorrect", float(avg_incorrect_agreement))


def aggregation_comparison(
    chunk_probs: np.ndarray,
    chunk_to_doc_idx: np.ndarray,
    y_true: np.ndarray,
    chunk_threshold: float,
    doc_threshold: float,
    n_docs: int,
) -> None:
    """
    Compare three aggregation methods: Mean, Max, WeightedMean.

    Visualizes:
    1. Confusion matrices for each method
    2. ROC curves overlaid
    3. Prediction probability distributions

    Args:
        chunk_probs: Per-chunk AI probabilities
        chunk_to_doc_idx: Mapping from chunk to document
        y_true: True document labels
        chunk_threshold: Threshold for chunk-level classification
        doc_threshold: Threshold for document-level classification
        n_docs: Total number of documents
    """
    logger.info("Generating aggregation method comparison...")

    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

    # Define aggregation methods
    def aggregate_mean(chunk_probs, chunk_to_doc_idx, n_docs):
        doc_probs = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                doc_probs[doc_idx] = chunk_probs[mask].mean()
        return doc_probs

    def aggregate_max(chunk_probs, chunk_to_doc_idx, n_docs):
        doc_probs = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                doc_probs[doc_idx] = chunk_probs[mask].max()
        return doc_probs

    def aggregate_weighted_mean(chunk_probs, chunk_to_doc_idx, n_docs, threshold):
        doc_probs = np.zeros(n_docs)
        for doc_idx in range(n_docs):
            mask = chunk_to_doc_idx == doc_idx
            if mask.any():
                doc_chunk_probs = chunk_probs[mask]
                weights = np.abs(doc_chunk_probs - threshold)
                doc_probs[doc_idx] = np.average(doc_chunk_probs, weights=weights)
        return doc_probs

    # Compute predictions for each method
    methods = {
        "Mean": aggregate_mean(chunk_probs, chunk_to_doc_idx, n_docs),
        "Max": aggregate_max(chunk_probs, chunk_to_doc_idx, n_docs),
        "WeightedMean": aggregate_weighted_mean(chunk_probs, chunk_to_doc_idx, n_docs, chunk_threshold),
    }

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Row 1: Confusion matrices
    for idx, (name, probs) in enumerate(methods.items()):
        ax = fig.add_subplot(gs[0, idx])
        preds = (probs >= doc_threshold).astype(int)

        ConfusionMatrixDisplay.from_predictions(y_true, preds, ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} Aggregation")

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        ax.text(0.5, -0.15, f"Acc: {acc:.3f} | F1: {f1:.3f}", ha="center", transform=ax.transAxes, fontsize=10)

    # Row 2, Col 1: ROC curves
    ax_roc = fig.add_subplot(gs[1, 0])
    for name, probs in methods.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves: Aggregation Methods")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    # Row 2, Col 2: Probability distributions
    ax_dist = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 50)
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for (name, probs), color in zip(methods.items(), colors):
        ax_dist.hist(probs, bins=bins, alpha=0.5, label=name, color=color, edgecolor="black")

    ax_dist.axvline(doc_threshold, color="black", linestyle="--", label=f"Threshold ({doc_threshold:.3f})")
    ax_dist.set_xlabel("AI Probability")
    ax_dist.set_ylabel("Number of Documents")
    ax_dist.set_title("Prediction Distributions")
    ax_dist.legend()
    ax_dist.grid(alpha=0.3)

    # Row 2, Col 3: Method recommendation
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")

    recommendation_text = """
Aggregation Method Comparison

Mean: Simple average of chunk predictions
- Balanced, robust to outliers
- Good for consistent documents

Max: Most suspicious chunk wins
- Conservative for AI detection
- Sensitive to single AI-like chunk

WeightedMean (Default): Confidence-weighted
- Weighs high-confidence chunks more
- Best performance on test set
- Balances precision and recall

Use Max for high-precision needs,
Mean for interpretability,
WeightedMean for best F1 score.
    """

    ax_text.text(
        0.1,
        0.9,
        recommendation_text.strip(),
        verticalalignment="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / "aggregation_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Saved aggregation comparison to {plot_path}")


def chunking_behavior_analysis(
    chunk_to_doc_idx: np.ndarray, chunked_tokens: list[list[list[int]]], n_docs: int
) -> None:
    """
    Analyze chunking behavior on real documents.

    Visualizes:
    1. Distribution of chunk counts per document
    2. Scatter: document length vs number of chunks
    3. Chunk size distribution

    Args:
        chunk_to_doc_idx: Mapping from chunk index to document index
        chunked_tokens: List of documents, each containing list of token chunks
        n_docs: Total number of documents
    """
    logger.info("Generating chunking behavior analysis...")

    # Calculate statistics per document
    doc_num_chunks = []
    doc_total_tokens = []
    all_chunk_sizes = []

    for doc_idx in range(n_docs):
        if doc_idx < len(chunked_tokens):
            chunks = chunked_tokens[doc_idx]
            doc_num_chunks.append(len(chunks))

            total_tokens = sum(len(chunk) for chunk in chunks)
            doc_total_tokens.append(total_tokens)

            for chunk in chunks:
                all_chunk_sizes.append(len(chunk))

    doc_num_chunks = np.array(doc_num_chunks)
    doc_total_tokens = np.array(doc_total_tokens)
    all_chunk_sizes = np.array(all_chunk_sizes)

    # Create figure
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Subplot 1: Chunk count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(doc_num_chunks, bins=range(1, max(doc_num_chunks) + 2), color="#9b59b6", alpha=0.7, edgecolor="black")
    ax1.axvline(doc_num_chunks.mean(), color="red", linestyle="--", label=f"Mean: {doc_num_chunks.mean():.1f}")
    ax1.set_xlabel("Number of Chunks per Document")
    ax1.set_ylabel("Number of Documents")
    ax1.set_title("Chunk Count Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Subplot 2: Document length vs chunks
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(doc_total_tokens, doc_num_chunks, alpha=0.3, s=10, color="#3498db")

    # Add trend line
    z = np.polyfit(doc_total_tokens, doc_num_chunks, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(doc_total_tokens.min(), doc_total_tokens.max(), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend")

    ax2.set_xlabel("Total Tokens in Document")
    ax2.set_ylabel("Number of Chunks")
    ax2.set_title("Document Length vs Chunk Count")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Subplot 3: Chunk size distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(all_chunk_sizes, bins=50, color="#2ecc71", alpha=0.7, edgecolor="black")
    ax3.axvline(all_chunk_sizes.mean(), color="red", linestyle="--", label=f"Mean: {all_chunk_sizes.mean():.1f}")
    ax3.axvline(
        np.median(all_chunk_sizes), color="blue", linestyle="--", label=f"Median: {np.median(all_chunk_sizes):.1f}"
    )
    ax3.set_xlabel("Chunk Size (tokens)")
    ax3.set_ylabel("Number of Chunks")
    ax3.set_title("Chunk Size Distribution")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Total documents: {n_docs}\n"
        f"Total chunks: {len(all_chunk_sizes)}\n"
        f"Avg chunks/doc: {doc_num_chunks.mean():.1f}\n"
        f"Single-chunk docs: {(doc_num_chunks == 1).sum()} ({100 * (doc_num_chunks == 1).sum() / n_docs:.1f}%)"
    )
    ax3.text(
        0.98,
        0.98,
        stats_text,
        transform=ax3.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / "chunking_behavior_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Saved chunking behavior analysis to {plot_path}")

    # Log summary statistics
    mlflow.log_metric("avg_chunks_per_doc", float(doc_num_chunks.mean()))
    mlflow.log_metric("avg_chunk_size", float(all_chunk_sizes.mean()))
    mlflow.log_metric("single_chunk_doc_pct", float(100 * (doc_num_chunks == 1).sum() / n_docs))


def confidence_correctness_analysis(
    y_probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> None:
    """
    Analyze relationship between prediction confidence and correctness.

    Visualizes:
    1. Scatter: confidence vs correctness
    2. Accuracy by confidence bins
    3. Confidence distributions for correct vs incorrect predictions

    Args:
        y_probs: Predicted probabilities (AI probability)
        y_true: True labels
        y_pred: Predicted labels
        threshold: Classification threshold
    """
    logger.info("Generating confidence vs correctness analysis...")

    # Calculate confidence (distance from threshold)
    confidence = np.abs(y_probs - threshold)
    correct = y_true == y_pred

    # Create figure
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Subplot 1: Confidence vs correctness scatter
    ax1 = fig.add_subplot(gs[0, 0])

    # Jitter y-axis for visibility
    y_jitter = correct.astype(float) + np.random.randn(len(correct)) * 0.05

    colors = np.where(correct, "#2ecc71", "#e74c3c")
    ax1.scatter(confidence, y_jitter, alpha=0.3, s=20, c=colors)

    ax1.set_xlabel("Prediction Confidence (distance from threshold)")
    ax1.set_ylabel("Correct (1) / Incorrect (0)")
    ax1.set_title("Confidence vs Prediction Correctness")
    ax1.set_ylim(-0.3, 1.3)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(alpha=0.3)

    # Add trend line
    from scipy.stats import pearsonr

    corr, p_value = pearsonr(confidence, correct.astype(float))
    ax1.text(
        0.02,
        0.98,
        f"Correlation: {corr:.3f} (p={p_value:.3e})",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Subplot 2: Accuracy by confidence bins
    ax2 = fig.add_subplot(gs[0, 1])

    # Bin by confidence
    confidence_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    bin_labels = ["0-0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2", "0.2-0.3", "0.3-0.5", "0.5+"]
    binned_confidence = np.digitize(confidence, confidence_bins)

    accuracies = []
    counts = []
    # np.digitize with n bins creates indices 0 to n
    # We skip index 0 (values < 0, shouldn't happen) and use 1 to n
    for i in range(1, len(confidence_bins) + 1):
        mask = binned_confidence == i
        if mask.any():
            accuracies.append(correct[mask].mean())
            counts.append(mask.sum())
        else:
            accuracies.append(0)
            counts.append(0)

    bars = ax2.bar(bin_labels, accuracies, color="#3498db", alpha=0.7, edgecolor="black")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f"n={count}", ha="center", va="bottom", fontsize=8)

    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by Confidence Level")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(correct.mean(), color="red", linestyle="--", label=f"Overall: {correct.mean():.3f}")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Confidence distributions
    ax3 = fig.add_subplot(gs[0, 2])

    correct_mask = correct
    incorrect_mask = ~correct

    bins = np.linspace(0, confidence.max(), 50)
    ax3.hist(
        confidence[correct_mask],
        bins=bins,
        alpha=0.6,
        color="#2ecc71",
        label=f"Correct ({correct_mask.sum()})",
        density=True,
    )
    ax3.hist(
        confidence[incorrect_mask],
        bins=bins,
        alpha=0.6,
        color="#e74c3c",
        label=f"Incorrect ({incorrect_mask.sum()})",
        density=True,
    )

    ax3.set_xlabel("Prediction Confidence")
    ax3.set_ylabel("Density")
    ax3.set_title("Confidence Distribution by Correctness")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Add interpretation
    avg_conf_correct = confidence[correct_mask].mean()
    avg_conf_incorrect = confidence[incorrect_mask].mean()
    ax3.text(
        0.98,
        0.98,
        f"Avg conf (correct): {avg_conf_correct:.3f}\nAvg conf (incorrect): {avg_conf_incorrect:.3f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plot_path = PLOT_DIR / "confidence_correctness_analysis.png"
    plt.savefig(plot_path, bbox_inches="tight")
    mlflow.log_artifact(str(plot_path))
    logger.info(f"Saved confidence vs correctness analysis to {plot_path}")

    # Log metrics
    mlflow.log_metric("confidence_correctness_corr", float(corr))
    mlflow.log_metric("avg_confidence_correct", float(avg_conf_correct))
    mlflow.log_metric("avg_confidence_incorrect", float(avg_conf_incorrect))
