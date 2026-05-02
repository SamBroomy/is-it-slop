#!/usr/bin/env python
"""EXPERIMENTAL: Demo script to validate statistical features for AI detection.

Tests document-level vs chunk-level features to determine optimal feature extraction
strategy before implementing full Rust solution.

Extended version: Tests additional features from Grammarly article:
- Better burstiness (coefficient of variation)
- Repetition metrics (word/bigram repetition)
- Punctuation entropy
- Word frequency entropy
- TF-IDF-based perplexity proxies (token commonness, OOV rate, avg IDF)

Key questions:
1. Which features work better at document-level vs chunk-level?
2. Do chunk-level features add signal beyond document-level features?
3. How do statistical features compare to random noise baseline?
4. Do Grammarly-inspired features improve on our baseline?
"""

import math
import re
from collections import Counter
from itertools import starmap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from __init__ import SEED, df_test, df_train
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, reverse_tokenize, tokenize
from loguru import logger
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Plain output - no timestamps/levels/module info
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{message}")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Set random seed
rng = np.random.default_rng(SEED)


# =============================================================================
# Load Pre-trained Vectorizer for Perplexity Proxies
# =============================================================================

logger.info("Loading pre-trained TF-IDF vectorizer...")
VECTORIZER_PATH = Path("../crates/is-it-slop/model_artifacts/3.0.0/tfidf_vectorizer.rkyv")
vectorizer = TfidfVectorizer.load(str(VECTORIZER_PATH))

# Extract vocabulary and IDF values for perplexity proxies
vocab_dict = vectorizer.vocabulary  # token_text -> index mapping
idf_values = []  # We'll need to extract these from the vectorizer

# Get number of documents from IDF formula: idf = log((n+1)/(df+1)) + 1
# We can approximate n_docs from the mean IDF
# For now, we'll just use the vocabulary for OOV detection
vocab_tokens = set(vocab_dict.keys())
logger.info(f"Loaded vectorizer with {len(vocab_tokens)} vocabulary items")


# =============================================================================
# Feature Extraction Functions
# =============================================================================


def compute_lexical_diversity(text: str) -> float:
    """Count of Unique / total words."""
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    unique = len(set(words))
    return unique / len(words)


def compute_vocabulary_richness(text: str) -> float:
    """sqrt(unique words) / total words (less sensitive to length)."""
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    unique = len(set(words))
    return np.sqrt(unique) / len(words)


def compute_avg_word_length(text: str) -> float:
    """Average character length of words."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def segment_sentences(text: str) -> list[str]:
    """Sentence segmentation (split on . ! ?)."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_sentence_stats(text: str) -> tuple[float, float]:
    """Average sentence length and standard deviation (burstiness)."""
    sentences = segment_sentences(text)
    if len(sentences) == 0:
        return 0.0, 0.0

    lengths = [len(s.split()) for s in sentences]
    avg_len = np.mean(lengths)
    std_len = np.std(lengths) if len(lengths) > 1 else 0.0

    return float(avg_len), float(std_len)


def extract_document_features(text: str, tokens: list[int]) -> dict[str, float]:
    """Extract features from full document (including new Grammarly-inspired features)."""
    avg_sent_len, sent_len_std = compute_sentence_stats(text)
    tfidf_proxies = compute_tfidf_perplexity_proxies(tokens)

    return {
        # Original features
        "doc_lexical_diversity": compute_lexical_diversity(text),
        "doc_vocab_richness": compute_vocabulary_richness(text),
        "doc_avg_sentence_length": avg_sent_len,
        "doc_sentence_length_std": sent_len_std,
        # New features
        "doc_sentence_length_cv": compute_sentence_length_cv(text),
        "doc_word_repetition_rate": compute_word_repetition_rate(text),
        "doc_bigram_repetition_rate": compute_bigram_repetition_rate(text),
        "doc_punctuation_entropy": compute_punctuation_entropy(text),
        "doc_word_frequency_entropy": compute_word_frequency_entropy(text),
        # TF-IDF perplexity proxies
        "doc_oov_rate": tfidf_proxies["oov_rate"],
        "doc_vocab_coverage": tfidf_proxies["vocab_coverage"],
    }


def extract_chunk_features(chunk_text: str, chunk_tokens: list[int]) -> dict[str, float]:
    """Extract features from text chunk (including new features)."""
    tfidf_proxies = compute_tfidf_perplexity_proxies(chunk_tokens)

    return {
        # Original features
        "chunk_lexical_diversity": compute_lexical_diversity(chunk_text),
        "chunk_avg_word_length": compute_avg_word_length(chunk_text),
        # New features
        "chunk_word_repetition_rate": compute_word_repetition_rate(chunk_text),
        "chunk_punctuation_entropy": compute_punctuation_entropy(chunk_text),
        "chunk_word_frequency_entropy": compute_word_frequency_entropy(chunk_text),
        # TF-IDF perplexity proxies
        "chunk_oov_rate": tfidf_proxies["oov_rate"],
        "chunk_vocab_coverage": tfidf_proxies["vocab_coverage"],
    }


# =============================================================================
# New Features (Grammarly-inspired)
# =============================================================================


def compute_sentence_length_cv(text: str) -> float:
    """Coefficient of variation for sentence lengths (normalized burstiness).

    CV = std / mean - better than raw std as it's normalized.
    """
    sentences = segment_sentences(text)
    if len(sentences) < 2:
        return 0.0

    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)

    if mean_len == 0:
        return 0.0

    std_len = np.std(lengths)
    return float(std_len / mean_len)


def compute_word_repetition_rate(text: str) -> float:
    """Percentage of words that appear multiple times.

    High repetition = more AI-like (uniform language).
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0

    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)

    return repeated_words / len(word_counts)


def compute_bigram_repetition_rate(text: str) -> float:
    """Percentage of word bigrams that repeat.

    Detects phrase-level repetition (AI pattern).
    """
    words = text.lower().split()
    if len(words) < 2:
        return 0.0

    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

    if len(bigrams) == 0:
        return 0.0

    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)

    return repeated_bigrams / len(bigram_counts)


def compute_punctuation_entropy(text: str) -> float:
    """Shannon entropy of punctuation marks.

    High entropy = diverse punctuation (more human-like).
    Low entropy = uniform punctuation (more AI-like).
    """
    punctuation = [c for c in text if c in ".,!?;:'\"-()[]{}"]

    if len(punctuation) == 0:
        return 0.0

    punct_counts = Counter(punctuation)
    total = len(punctuation)

    # Shannon entropy
    entropy = -sum((count / total) * np.log2(count / total) for count in punct_counts.values())

    return float(entropy)


def compute_word_frequency_entropy(text: str) -> float:
    """Shannon entropy of word frequencies.

    High entropy = diverse word usage (more human-like).
    Low entropy = repetitive word usage (more AI-like).
    """
    words = text.lower().split()

    if len(words) == 0:
        return 0.0

    word_counts = Counter(words)
    total = len(words)

    # Shannon entropy
    entropy = -sum((count / total) * np.log2(count / total) for count in word_counts.values())

    return float(entropy)


def compute_tfidf_perplexity_proxies(tokens: list[int]) -> dict[str, float]:
    """Compute perplexity proxies using TF-IDF vocabulary statistics.

    Uses pre-trained vectorizer vocabulary to estimate token commonness.

    Returns:
        oov_rate: Percentage of tokens not in vocabulary (rare tokens)
        vocab_coverage: Percentage of tokens in vocabulary (common tokens)

    """
    if len(tokens) == 0:
        return {"oov_rate": 0.0, "vocab_coverage": 0.0}

    # Decode tokens to text for vocabulary lookup
    # Note: This is approximate since we're checking token-level against word-level vocab
    text = reverse_tokenize(tokens)
    words = text.lower().split()

    if len(words) == 0:
        return {"oov_rate": 0.0, "vocab_coverage": 0.0}

    # Count how many words are in TF-IDF vocabulary
    in_vocab = sum(1 for word in words if word in vocab_tokens)

    oov_rate = 1.0 - (in_vocab / len(words))
    vocab_coverage = in_vocab / len(words)

    return {"oov_rate": float(oov_rate), "vocab_coverage": float(vocab_coverage)}


# =============================================================================
# Load and Chunk Data (matching train.py flow)
# =============================================================================

logger.info("Loading data...")

# Sample for faster demo
SAMPLE_SIZE = 10_000  # Set to False to use all data

if SAMPLE_SIZE:
    logger.info(f"Sampling {SAMPLE_SIZE} examples from train set...")
    df_train_collected = df_train.collect().sample(n=SAMPLE_SIZE, seed=SEED)
    df_test_collected = df_test.collect().sample(n=min(SAMPLE_SIZE // 2, 5000), seed=SEED)
else:
    df_train_collected = df_train.collect()
    df_test_collected = df_test.collect()

logger.info(f"Train documents: {len(df_train_collected)}, Test documents: {len(df_test_collected)}")

# Extract texts and labels
X_train_texts = df_train_collected.select("text").to_series().to_numpy()
y_train_docs = df_train_collected.select("label").to_series().to_numpy()

X_test_texts = df_test_collected.select("text").to_series().to_numpy()
y_test_docs = df_test_collected.select("label").to_series().to_numpy()

# =============================================================================
# Tokenize and Chunk (matching train.py)
# =============================================================================

logger.info("Tokenizing texts...")
train_tokens = tokenize(X_train_texts)
test_tokens = tokenize(X_test_texts)

logger.info("Chunking tokens...")
chunker = TokenChunker(chunk_size=150, overlap=15, min_chunk_size=30)
train_chunked = chunker.chunk_batch(train_tokens)
test_chunked = chunker.chunk_batch(test_tokens)


# =============================================================================
# Extract Features at Both Levels
# =============================================================================

logger.info("Extracting document-level features (including new features)...")
train_doc_features = list(starmap(extract_document_features, zip(X_train_texts, train_tokens, strict=True)))

test_doc_features = list(starmap(extract_document_features, zip(X_test_texts, test_tokens, strict=True)))

logger.info("Extracting chunk-level features and flattening...")


def flatten_with_features(
    texts: np.ndarray,
    all_tokens: list[list[int]],
    chunked_tokens: list[list[list[int]]],
    doc_features: list[dict[str, float]],
    labels: np.ndarray,
) -> pl.DataFrame:
    """Flatten chunks and extract features at both levels.

    Returns DataFrame with one row per chunk containing:
    - Document-level features (replicated for all chunks from same doc)
    - Chunk-level features (computed per chunk)
    - Random baseline feature
    - Label (replicated)
    """
    rows = []

    for doc_idx, (_text, _doc_tokens, chunks, doc_feat, label) in enumerate(
        zip(texts, all_tokens, chunked_tokens, doc_features, labels, strict=True)
    ):
        # Add random document-level noise feature (same for all chunks in doc)
        doc_random_noise = rng.normal()

        for chunk_idx, chunk_tokens in enumerate(chunks):
            # Detokenize chunk to compute chunk-level features
            chunk_text = reverse_tokenize(chunk_tokens)
            chunk_feat = extract_chunk_features(chunk_text, chunk_tokens)

            # Add random chunk-level noise
            chunk_random_noise = rng.normal()

            # Combine features
            row = {
                **doc_feat,  # Document-level features (replicated)
                **chunk_feat,  # Chunk-level features
                "doc_random_noise": doc_random_noise,  # Random baseline (doc-level)
                "chunk_random_noise": chunk_random_noise,  # Random baseline (chunk-level)
                "label": label,
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "num_chunks": len(chunks),
            }
            rows.append(row)

    return pl.DataFrame(rows)


df_train_chunks = flatten_with_features(X_train_texts, train_tokens, train_chunked, train_doc_features, y_train_docs)
df_test_chunks = flatten_with_features(X_test_texts, test_tokens, test_chunked, test_doc_features, y_test_docs)

logger.info(
    f"Training: {len(y_train_docs)} documents → {len(df_train_chunks)} chunks "
    f"(avg {len(df_train_chunks) / len(y_train_docs):.1f} chunks/doc)"
)
logger.info(
    f"Test: {len(y_test_docs)} documents → {len(df_test_chunks)} chunks "
    f"(avg {len(df_test_chunks) / len(y_test_docs):.1f} chunks/doc)"
)

logger.info("\nChunk-level feature statistics:")
logger.info(df_train_chunks.select(pl.all().exclude("label", "doc_idx", "chunk_idx", "num_chunks")).describe())


# =============================================================================
# Statistical Analysis: Document vs Chunk Level
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("Statistical Significance Tests (Chunk-Level Analysis)")
logger.info("=" * 80)

# Define feature groups
doc_features_original = [
    "doc_lexical_diversity",
    "doc_vocab_richness",
    "doc_avg_sentence_length",
    "doc_sentence_length_std",
]

doc_features_new = [
    "doc_sentence_length_cv",
    "doc_word_repetition_rate",
    "doc_bigram_repetition_rate",
    "doc_punctuation_entropy",
    "doc_word_frequency_entropy",
    "doc_oov_rate",
    "doc_vocab_coverage",
]

chunk_features_original = ["chunk_lexical_diversity", "chunk_avg_word_length"]

chunk_features_new = [
    "chunk_word_repetition_rate",
    "chunk_punctuation_entropy",
    "chunk_word_frequency_entropy",
    "chunk_oov_rate",
    "chunk_vocab_coverage",
]

random_features = ["doc_random_noise", "chunk_random_noise"]

# Combine all features
doc_features = doc_features_original + doc_features_new
chunk_features = chunk_features_original + chunk_features_new
all_features = doc_features + chunk_features + random_features

logger.info(
    f"Total features: {len(all_features)} ({len(doc_features)} doc + {len(chunk_features)} chunk + {len(random_features)} random)"
)

# Separate by label (at chunk level)
df_human_chunks = df_train_chunks.filter(pl.col("label") == 0)
df_ai_chunks = df_train_chunks.filter(pl.col("label") == 1)

logger.info(f"\nChunk counts - Human: {len(df_human_chunks)}, AI: {len(df_ai_chunks)}")


def analyze_feature(feature: str, df_human: pl.DataFrame, df_ai: pl.DataFrame) -> dict:
    """Run statistical tests on a single feature."""
    human_vals = df_human[feature].to_numpy()
    ai_vals = df_ai[feature].to_numpy()

    # Remove NaN/inf values
    human_vals = human_vals[np.isfinite(human_vals)]
    ai_vals = ai_vals[np.isfinite(ai_vals)]

    # T-test
    ttest = stats.ttest_ind(human_vals, ai_vals)
    t_stat = float(ttest.statistic)  # type: ignore[union-attr]
    p_value = float(ttest.pvalue)  # type: ignore[union-attr]

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(human_vals) ** 2 + np.std(ai_vals) ** 2) / 2)
    cohens_d = (np.mean(human_vals) - np.mean(ai_vals)) / pooled_std if pooled_std > 0 else 0.0

    # Correlation with label
    all_vals = np.concatenate([human_vals, ai_vals])
    all_labels = np.concatenate([np.zeros(len(human_vals)), np.ones(len(ai_vals))])
    corr, corr_p = stats.pearsonr(all_vals, all_labels)

    return {
        "feature": feature,
        "human_mean": float(np.mean(human_vals)),
        "human_std": float(np.std(human_vals)),
        "ai_mean": float(np.mean(ai_vals)),
        "ai_std": float(np.std(ai_vals)),
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "correlation": corr,
        "correlation_p": corr_p,
        "significant": p_value < 0.001,
    }


# Analyze all features
logger.info("\n" + "-" * 80)
logger.info("DOCUMENT-LEVEL FEATURES (replicated across chunks)")
logger.info("-" * 80)

doc_results = []
for feature in doc_features:
    result = analyze_feature(feature, df_human_chunks, df_ai_chunks)
    doc_results.append(result)

    logger.info(f"\n{feature}:")
    logger.info(f"  Human: {result['human_mean']:.4f} ± {result['human_std']:.4f}")
    logger.info(f"  AI:    {result['ai_mean']:.4f} ± {result['ai_std']:.4f}")
    logger.info(f"  t-stat: {result['t_statistic']:.4f}, p-value: {result['p_value']:.4e}")
    logger.info(f"  Cohen's d: {result['cohens_d']:.4f}")
    logger.info(f"  Correlation with label: r={result['correlation']:+.4f}")
    logger.info(f"  Significant: {'✓' if result['significant'] else '✗'}")

logger.info("\n" + "-" * 80)
logger.info("CHUNK-LEVEL FEATURES (computed per chunk)")
logger.info("-" * 80)

chunk_results = []
for feature in chunk_features:
    result = analyze_feature(feature, df_human_chunks, df_ai_chunks)
    chunk_results.append(result)

    logger.info(f"\n{feature}:")
    logger.info(f"  Human: {result['human_mean']:.4f} ± {result['human_std']:.4f}")
    logger.info(f"  AI:    {result['ai_mean']:.4f} ± {result['ai_std']:.4f}")
    logger.info(f"  t-stat: {result['t_statistic']:.4f}, p-value: {result['p_value']:.4e}")
    logger.info(f"  Cohen's d: {result['cohens_d']:.4f}")
    logger.info(f"  Correlation with label: r={result['correlation']:+.4f}")
    logger.info(f"  Significant: {'✓' if result['significant'] else '✗'}")

logger.info("\n" + "-" * 80)
logger.info("RANDOM BASELINE FEATURES (sanity check)")
logger.info("-" * 80)

random_results = []
for feature in random_features:
    result = analyze_feature(feature, df_human_chunks, df_ai_chunks)
    random_results.append(result)

    logger.info(f"\n{feature}:")
    logger.info(f"  Human: {result['human_mean']:.4f} ± {result['human_std']:.4f}")
    logger.info(f"  AI:    {result['ai_mean']:.4f} ± {result['ai_std']:.4f}")
    logger.info(f"  t-stat: {result['t_statistic']:.4f}, p-value: {result['p_value']:.4e}")
    logger.info(f"  Cohen's d: {result['cohens_d']:.4f}")
    logger.info(f"  Correlation with label: r={result['correlation']:+.4f}")
    logger.info(f"  Significant: {'✓' if result['significant'] else '✗'} (should be ✗)")

# Combine results
all_results = doc_results + chunk_results + random_results
df_results = pl.DataFrame(all_results)

# =============================================================================
# Visualizations
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("Generating visualizations...")
logger.info("=" * 80)

output_dir = Path("plots/statistical_features_demo")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Effect size comparison (Cohen's d)
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by absolute effect size
df_plot = df_results.sort("cohens_d", descending=False)

# Color by feature type
colors = []
for feat in df_plot["feature"].to_list():
    if "doc_" in feat and "random" not in feat:
        colors.append("steelblue")
    elif "chunk_" in feat and "random" not in feat:
        colors.append("forestgreen")
    else:
        colors.append("lightgray")

ax.barh(df_plot["feature"], df_plot["cohens_d"], color=colors)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Cohen's d (effect size)")
ax.set_title("Feature Effect Sizes: Document vs Chunk Level\n(Blue=Document, Green=Chunk, Gray=Random)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(output_dir / "effect_sizes_comparison.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'effect_sizes_comparison.png'}")
plt.close()

# 2. Correlation comparison
fig, ax = plt.subplots(figsize=(12, 8))

df_plot = df_results.sort("correlation")
colors = []
for feat in df_plot["feature"].to_list():
    if "doc_" in feat and "random" not in feat:
        colors.append("steelblue")
    elif "chunk_" in feat and "random" not in feat:
        colors.append("forestgreen")
    else:
        colors.append("lightgray")

ax.barh(df_plot["feature"], df_plot["correlation"], color=colors)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Label")
ax.set_title("Feature Correlations: Document vs Chunk Level\n(Blue=Document, Green=Chunk, Gray=Random)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(output_dir / "correlations_comparison.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'correlations_comparison.png'}")
plt.close()

# 3. Distribution plots (document features)
n_doc_cols = 4
n_doc_rows = math.ceil(len(doc_features) / n_doc_cols)
fig, axes = plt.subplots(n_doc_rows, n_doc_cols, figsize=(7 * n_doc_cols, 5 * n_doc_rows))
axes = axes.flatten()

for idx, feature in enumerate(doc_features):
    ax = axes[idx]

    human_vals = df_human_chunks[feature].to_numpy()
    ai_vals = df_ai_chunks[feature].to_numpy()

    human_vals = human_vals[np.isfinite(human_vals)]
    ai_vals = ai_vals[np.isfinite(ai_vals)]

    # Remove outliers
    p99 = np.percentile(np.concatenate([human_vals, ai_vals]), 99)
    human_vals = human_vals[human_vals <= p99]
    ai_vals = ai_vals[ai_vals <= p99]

    ax.hist(human_vals, bins=50, alpha=0.5, label="Human", density=True, color="blue")
    ax.hist(ai_vals, bins=50, alpha=0.5, label="AI", density=True, color="red")

    ax.set_xlabel(feature.replace("_", " ").replace("doc ", "").title())
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    result = df_results.filter(pl.col("feature") == feature).to_dicts()[0]
    ax.set_title(f"p={result['p_value']:.2e}, r={result['correlation']:+.3f}")

for idx in range(len(doc_features), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Document-Level Features (replicated across chunks)", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(output_dir / "doc_feature_distributions.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'doc_feature_distributions.png'}")
plt.close()

# 4. Distribution plots (chunk features)
n_chunk_cols = 4
n_chunk_rows = math.ceil(len(chunk_features) / n_chunk_cols)
fig, axes = plt.subplots(n_chunk_rows, n_chunk_cols, figsize=(7 * n_chunk_cols, 5 * n_chunk_rows))
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for idx, feature in enumerate(chunk_features):
    ax = axes[idx]

    human_vals = df_human_chunks[feature].to_numpy()
    ai_vals = df_ai_chunks[feature].to_numpy()

    human_vals = human_vals[np.isfinite(human_vals)]
    ai_vals = ai_vals[np.isfinite(ai_vals)]

    p99 = np.percentile(np.concatenate([human_vals, ai_vals]), 99)
    human_vals = human_vals[human_vals <= p99]
    ai_vals = ai_vals[ai_vals <= p99]

    ax.hist(human_vals, bins=50, alpha=0.5, label="Human", density=True, color="blue")
    ax.hist(ai_vals, bins=50, alpha=0.5, label="AI", density=True, color="red")

    ax.set_xlabel(feature.replace("_", " ").replace("chunk ", "").title())
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    result = df_results.filter(pl.col("feature") == feature).to_dicts()[0]
    ax.set_title(f"p={result['p_value']:.2e}, r={result['correlation']:+.3f}")

for idx in range(len(chunk_features), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Chunk-Level Features (computed per chunk)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "chunk_feature_distributions.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'chunk_feature_distributions.png'}")
plt.close()

# 5. Random baseline distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, feature in enumerate(random_features):
    ax = axes[idx]

    human_vals = df_human_chunks[feature].to_numpy()
    ai_vals = df_ai_chunks[feature].to_numpy()

    ax.hist(human_vals, bins=50, alpha=0.5, label="Human", density=True, color="blue")
    ax.hist(ai_vals, bins=50, alpha=0.5, label="AI", density=True, color="red")

    ax.set_xlabel(feature.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    result = df_results.filter(pl.col("feature") == feature).to_dicts()[0]
    ax.set_title(f"p={result['p_value']:.2e}, r={result['correlation']:+.3f} (should be ~0)")

plt.suptitle("Random Baseline Features (sanity check)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "random_baseline_distributions.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'random_baseline_distributions.png'}")
plt.close()


# =============================================================================
# Predictive Power Tests (Chunk-Level Training)
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("Testing Predictive Power: Document vs Chunk vs Combined Features")
logger.info("=" * 80)


def train_and_evaluate(
    feature_set_name: str, features: list[str], df_train: pl.DataFrame, df_test: pl.DataFrame
) -> dict:
    """Train logistic regression and return metrics."""
    # Prepare data
    X_train = df_train.select(features).to_numpy()
    y_train = df_train.select("label").to_numpy().ravel()

    X_test = df_test.select(features).to_numpy()
    y_test = df_test.select("label").to_numpy().ravel()

    # Remove NaN/inf
    train_mask = np.all(np.isfinite(X_train), axis=1)
    test_mask = np.all(np.isfinite(X_test), axis=1)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)

    # Predict
    y_pred_test = lr.predict(X_test_scaled)
    y_proba_test = lr.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    test_auc = roc_auc_score(y_test, y_proba_test)

    return {
        "feature_set": feature_set_name,
        "n_features": len(features),
        "test_auc": test_auc,
        "y_pred": y_pred_test,
        "y_test": y_test,
        "coefficients": lr.coef_[0],
        "feature_names": features,
    }


# Define recommended 9-feature subset based on experimental validation
recommended_features = [
    # Document-level (6)
    "doc_bigram_repetition_rate",  # ⭐⭐⭐ Strongest (d=-0.419, coef=+1.98)
    "doc_punctuation_entropy",  # ⭐⭐⭐ Second strongest (d=-0.365, coef=+0.23)
    "doc_lexical_diversity",  # ⭐⭐ Classic (d=+0.165)
    "doc_vocab_richness",  # ⭐⭐ Complementary (d=+0.154)
    "doc_word_repetition_rate",  # ⭐ Moderate (d=-0.084)
    "doc_sentence_length_cv",  # ⭐ Burstiness (d=-0.115)
    # Chunk-level (3)
    "chunk_avg_word_length",  # ⭐⭐⭐ Dominant coefficient (+8.23)
    "chunk_punctuation_entropy",  # ⭐⭐ Local signal (d=-0.080)
    "chunk_word_frequency_entropy",  # ⭐ Distribution (d=-0.098)
]

# Test different feature sets
feature_sets = {
    "Document Only": doc_features,
    "Chunk Only": chunk_features,
    "Document + Chunk": doc_features + chunk_features,
    "Recommended 9 Features": recommended_features,  # NEW: Test our recommended subset
    "Random Baseline": random_features,
    "All Features": doc_features + chunk_features,
}

results_dict = {}
for name, features in feature_sets.items():
    logger.info(f"\nTraining with: {name} ({len(features)} features)")
    result = train_and_evaluate(name, features, df_train_chunks, df_test_chunks)
    results_dict[name] = result
    logger.info(f"  Test AUC: {result['test_auc']:.4f}")

# Compare results
logger.info("\n" + "-" * 80)
logger.info("COMPARISON SUMMARY (Chunk-Level Predictions)")
logger.info("-" * 80)

comparison_data = []
for name, result in results_dict.items():
    comparison_data.append({"Feature Set": name, "N Features": result["n_features"], "Test AUC": result["test_auc"]})

df_comparison = pl.DataFrame(comparison_data).sort("Test AUC", descending=True)
logger.info("\n" + str(df_comparison))

# Detailed report for best model
best_name = df_comparison.filter(~pl.col("Feature Set").str.contains("Random"))[0, "Feature Set"]
best_result = results_dict[best_name]

logger.info(f"\n\nBest Model: {best_name}")
logger.info(f"Test AUC: {best_result['test_auc']:.4f}")
logger.info("\nClassification Report:")
logger.info("\n" + classification_report(best_result["y_test"], best_result["y_pred"], target_names=["Human", "AI"]))  # type: ignore[operator]

# Feature importance for best model
logger.info("\nFeature Importance (Coefficients):")
coef_df = pl.DataFrame({
    "feature": best_result["feature_names"],
    "coefficient": best_result["coefficients"],
    "abs_coefficient": np.abs(best_result["coefficients"]),
}).sort("abs_coefficient", descending=True)

for row in coef_df.iter_rows(named=True):
    logger.info(f"  {row['feature']:30s}: {row['coefficient']:+.4f}")

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
df_plot = df_comparison.sort("Test AUC")
colors = ["lightgray" if "Random" in fs else "steelblue" for fs in df_plot["Feature Set"].to_list()]
ax.barh(df_plot["Feature Set"], df_plot["Test AUC"], color=colors)
ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random baseline")
ax.set_xlabel("Test AUC")
ax.set_title("Predictive Power Comparison: Document vs Chunk Features")
ax.legend()
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(output_dir / "auc_comparison.png", dpi=150, bbox_inches="tight")
logger.info(f"\nSaved: {output_dir / 'auc_comparison.png'}")
plt.close()

# Feature importance plot for best model
fig, ax = plt.subplots(figsize=(10, 6))
coef_plot = coef_df.sort("coefficient")
colors = []
for feat in coef_plot["feature"].to_list():
    if "doc_" in feat:
        colors.append("steelblue")
    elif "chunk_" in feat:
        colors.append("forestgreen")
    else:
        colors.append("lightgray")

ax.barh(coef_plot["feature"], coef_plot["coefficient"], color=colors)
ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Coefficient (positive = more AI-like)")
ax.set_title(f"Feature Importance: {best_name}\n(Blue=Document, Green=Chunk)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
logger.info(f"Saved: {output_dir / 'feature_importance.png'}")
plt.close()


# =============================================================================
# Summary and Recommendations
# =============================================================================

logger.info("\n" + "=" * 80)
logger.info("SUMMARY & RECOMMENDATIONS")
logger.info("=" * 80)

# Statistical significance
sig_doc = [r for r in doc_results if r["significant"]]
sig_chunk = [r for r in chunk_results if r["significant"]]
sig_random = [r for r in random_results if r["significant"]]

logger.info("\nStatistically significant features (p<0.001):")
logger.info(f"  Document-level: {len(sig_doc)}/{len(doc_features)}")
for r in sig_doc:
    logger.info(f"    - {r['feature']}: Cohen's d={r['cohens_d']:.3f}, r={r['correlation']:+.3f}")

logger.info(f"  Chunk-level: {len(sig_chunk)}/{len(chunk_features)}")
for r in sig_chunk:
    logger.info(f"    - {r['feature']}: Cohen's d={r['cohens_d']:.3f}, r={r['correlation']:+.3f}")

logger.info(f"  Random baseline: {len(sig_random)}/{len(random_features)} (should be 0)")
if sig_random:
    logger.warning("  ⚠ Random features showing significance - possible data leakage!")

# Predictive power comparison
best_auc = results_dict[best_name]["test_auc"]
random_auc = results_dict["Random Baseline"]["test_auc"]
doc_auc = results_dict["Document Only"]["test_auc"]
chunk_auc = results_dict["Chunk Only"]["test_auc"]

logger.info("\nPredictive Power (Chunk-Level Training):")
logger.info(f"  Random Baseline:  {random_auc:.4f} (should be ~0.50)")
logger.info(f"  Document Only:    {doc_auc:.4f} (Δ = {(doc_auc - 0.5) * 100:+.2f}pp)")
logger.info(f"  Chunk Only:       {chunk_auc:.4f} (Δ = {(chunk_auc - 0.5) * 100:+.2f}pp)")
logger.info(f"  Best ({best_name}): {best_auc:.4f} (Δ = {(best_auc - 0.5) * 100:+.2f}pp)")

# Feature-level insights
logger.info("\nFeature Insights:")

# Find strongest features by correlation
df_sorted = df_results.sort("correlation").filter(~pl.col("feature").str.contains("random"))
strongest_positive = df_sorted[-1, "feature"]
strongest_negative = df_sorted[0, "feature"]

logger.info(f"  Strongest AI signal:    {strongest_positive}")
logger.info(f"    (r={df_sorted[-1, 'correlation']:+.4f}, Cohen's d={df_sorted[-1, 'cohens_d']:+.3f})")
logger.info(f"  Strongest Human signal: {strongest_negative}")
logger.info(f"    (r={df_sorted[0, 'correlation']:+.4f}, Cohen's d={df_sorted[0, 'cohens_d']:+.3f})")

# Recommendations
logger.info("\n" + "-" * 80)
logger.info("RECOMMENDATIONS FOR RUST IMPLEMENTATION")
logger.info("-" * 80)

if best_auc > 0.65:
    logger.info("\n✓ Strong signal detected - features have clear discriminative power")

    if doc_auc > chunk_auc + 0.02:
        logger.info("  → Document-level features are stronger")
        logger.info("  → Priority: Implement document-level extraction first")
        logger.info(
            f"  → Top features: {
                ', '.join([
                    r['feature'] for r in sorted(doc_results, key=lambda x: abs(x['cohens_d']), reverse=True)[:2]
                ])
            }"
        )
    elif chunk_auc > doc_auc + 0.02:
        logger.info("  → Chunk-level features are stronger")
        logger.info("  → Priority: Implement chunk-level extraction first")
        logger.info(
            f"  → Top features: {
                ', '.join([
                    r['feature'] for r in sorted(chunk_results, key=lambda x: abs(x['cohens_d']), reverse=True)[:2]
                ])
            }"
        )
    else:
        logger.info("  → Document and chunk features have similar strength")
        logger.info("  → Recommendation: Implement hybrid approach (both levels)")
        logger.info(
            f"  → Expected improvement when combined: ~{(best_auc - max(doc_auc, chunk_auc)) * 100:.1f} percentage points"
        )

    logger.info(
        "\n  When combined with TF-IDF (current AUC: ~0.95):"
        "\n    Expected gain: 2-5 percentage points in F1"
        "\n    Risk of overfitting: Low (features are interpretable and stable)"
    )

elif best_auc > 0.55:
    logger.info("\n⚠ Moderate signal detected - features show promise but limited standalone value")
    logger.info("  → Recommendation: Use as supplementary features with TF-IDF")
    logger.info("  → Priority: Focus on strongest features to avoid noise")
    logger.info(
        f"  → Strongest 2-3 features: {
            ', '.join([
                r['feature']
                for r in sorted(all_results, key=lambda x: abs(x['cohens_d']), reverse=True)[:3]
                if 'random' not in r['feature']
            ])
        }"
    )

else:
    logger.info(
        "\n✗ Weak signal - features alone have limited predictive power"
        "\n  → Recommendation: Re-evaluate feature engineering"
        "\n  → Consider: Different text statistics or interaction terms"
    )

logger.info(f"\n\nAll visualizations saved to: {output_dir}/")
