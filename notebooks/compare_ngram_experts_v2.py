#!/usr/bin/env python
"""EXPERIMENTAL: Compare Character N-grams vs Word N-grams for Linguistic Expert.

Trains simple models on each feature type independently to determine which
provides better signal for AI text detection.
"""

import re
import time

import numpy as np
from __init__ import SEED, df_test, df_train
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{message}")

# Set random seed
np.random.seed(SEED)

logger.info("Loading data...")
df_train_ = df_train.select("text", "label").collect().sample(fraction=0.5)
df_test_ = df_test.select("text", "label").collect().sample(fraction=0.5)
X_train = df_train_.select("text").to_series().to_numpy()
y_train = df_train_.select("label").to_series().to_numpy()
X_test = df_test_.select("text").to_series().to_numpy()
y_test = df_test_.select("label").to_series().to_numpy()

logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# ==============================================================================
# Investigate '0' Token Issue First
# ==============================================================================

logger.info("=" * 80)
logger.info("INVESTIGATING '0' TOKEN ISSUE")
logger.info("=" * 80)

# Find texts containing '0' tokens
zero_pattern = r"\b0\b"  # Word boundary around '0'

ai_texts_with_zero = [text for text, label in zip(X_train, y_train, strict=False) if label == 1 and re.search(zero_pattern, text)]
human_texts_with_zero = [text for text, label in zip(X_train, y_train, strict=False) if label == 0 and re.search(zero_pattern, text)]

logger.info(
    f"\nAI texts containing '0': {len(ai_texts_with_zero)}/{(y_train == 1).sum()} ({100 * len(ai_texts_with_zero) / (y_train == 1).sum():.2f}%)"
)
logger.info(
    f"Human texts containing '0': {len(human_texts_with_zero)}/{(y_train == 0).sum()} ({100 * len(human_texts_with_zero) / (y_train == 0).sum():.2f}%)"
)

# Show examples
logger.info("\nExample AI texts with '0' (showing context):")
for i, text in enumerate(ai_texts_with_zero[:3]):
    match = re.search(r".{0,80}\b0\b.{0,80}", text)
    snippet = match.group(0) if match else text[:160]
    logger.info(f"  [{i + 1}] ...{snippet.strip()}...")

logger.info("\nExample Human texts with '0' (showing context):")
for i, text in enumerate(human_texts_with_zero[:3]):
    match = re.search(r".{0,80}\b0\b.{0,80}", text)
    snippet = match.group(0) if match else text[:160]
    logger.info(f"  [{i + 1}] ...{snippet.strip()}...")

logger.info("\n" + "=" * 80 + "\n")

# ==============================================================================
# Experiment 1: Character N-grams (Strict - No Markdown Artifacts)
# ==============================================================================

logger.info("=" * 80)
logger.info("EXPERIMENT 1: CHARACTER N-GRAMS (STRICT - NO MARKDOWN)")
logger.info("=" * 80)

logger.info("Extracting character n-gram features (excluding markdown symbols)...")
t_char_start = time.time()


# Custom analyzer that strips markdown symbols before extracting char n-grams
def clean_char_analyzer(text):
    """Extract character n-grams after removing markdown symbols."""
    # Remove markdown symbols
    # Keep: letters, digits, spaces, basic punctuation (.,!?;:'"")
    # Remove: #, *, `, _, [, ], (, ), -, +, ~, =, <, >, {, }, |, \
    cleaned = re.sub(r"[#*`_\[\]()\-+=~<>{}|\\]", "", text)

    # Also remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Extract 4-5 character n-grams from cleaned text
    # We have to do this manually since we're using a custom preprocessor
    ngrams = []
    for n in range(4, 6):  # 4-5 grams
        ngrams.extend(cleaned[i : i + n].lower() for i in range(len(cleaned) - n + 1))
    return ngrams


char_vectorizer = TfidfVectorizer(
    analyzer=clean_char_analyzer,  # Custom analyzer without markdown
    min_df=500,  # Much stricter (5x more aggressive)
    max_df=0.5,  # Stricter (was 0.7)
    max_features=15000,  # Reduced to focus on most important patterns
    sublinear_tf=True,
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char = char_vectorizer.transform(X_test)

char_time = time.time() - t_char_start
char_sparsity = 100 * (1 - X_train_char.nnz / np.prod(X_train_char.shape))

logger.info(f"Character features: {X_train_char.shape}")
logger.info(f"Sparsity: {char_sparsity:.2f}%")
logger.info(f"Extraction time: {char_time:.2f}s")

# Train simple ensemble
logger.info("\nTraining character n-gram models...")
t_train_start = time.time()

char_nb = MultinomialNB(alpha=0.01)
char_sgd = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.00005,
    class_weight="balanced",
    early_stopping=True,
    max_iter=2000,
    random_state=SEED,
    n_jobs=-1,
)
char_svc = LinearSVC(C=1.0, loss="squared_hinge", max_iter=2000, class_weight="balanced", random_state=SEED)
char_svc_calibrated = CalibratedClassifierCV(char_svc, cv=3, method="sigmoid")

char_ensemble = StackingClassifier(
    estimators=[("nb", char_nb), ("sgd", char_sgd), ("svc", char_svc_calibrated)],
    final_estimator=LogisticRegression(max_iter=200, random_state=SEED),
    cv=3,
    stack_method="predict_proba",
    n_jobs=-1,
)

char_ensemble.fit(X_train_char, y_train)
char_train_time = time.time() - t_train_start

logger.info(f"Training time: {char_train_time:.2f}s")

# Evaluate
char_probs = char_ensemble.predict_proba(X_test_char)[:, 1]
char_auc = roc_auc_score(y_test, char_probs)

# Find best F1 threshold
precision, recall, thresholds = precision_recall_curve(y_test, char_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
char_best_f1 = f1_scores[best_f1_idx]
char_best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

char_pred = (char_probs >= char_best_threshold).astype(np.int8)
char_mcc = matthews_corrcoef(y_test, char_pred)
char_accuracy = accuracy_score(y_test, char_pred)
char_precision = precision_score(y_test, char_pred)
char_recall = recall_score(y_test, char_pred)
char_f1 = f1_score(y_test, char_pred)

logger.info("\nCharacter N-gram Results:")
logger.info(f"  AUC:       {char_auc:.4f}")
logger.info(f"  Best F1:   {char_best_f1:.4f} (threshold: {char_best_threshold:.4f})")
logger.info(f"  MCC:       {char_mcc:.4f}")
logger.info(f"  Accuracy:  {char_accuracy:.4f}")
logger.info(f"  Precision: {char_precision:.4f}")
logger.info(f"  Recall:    {char_recall:.4f}")
logger.info(f"  F1:        {char_f1:.4f}")

# Top features (use SGD model coefficients - most interpretable)
logger.info("\nTop 30 AI-indicative character n-grams:")
char_feature_names = char_vectorizer.get_feature_names_out()

# Get coefficients from SGD base model (index 1 in estimators)
char_sgd_model = char_ensemble.estimators_[1]  # SGD is second estimator
char_coef = char_sgd_model.coef_.ravel()

logger.info(f"DEBUG: Total features: {len(char_feature_names)}, Coef shape: {char_coef.shape}")
logger.info(f"DEBUG: Coef range: [{char_coef.min():.4f}, {char_coef.max():.4f}]")

char_top_ai_idx = np.argsort(char_coef)[-30:][::-1]
logger.info(f"DEBUG: Top AI indices count: {len(char_top_ai_idx)}")

for i, idx in enumerate(char_top_ai_idx):
    logger.info(f"  [{i + 1:2d}] '{char_feature_names[idx]}': {char_coef[idx]:+.4f}")

logger.info("\nTop 30 Human-indicative character n-grams:")
char_top_human_idx = np.argsort(char_coef)[:30]
logger.info(f"DEBUG: Top Human indices count: {len(char_top_human_idx)}")

for i, idx in enumerate(char_top_human_idx):
    logger.info(f"  [{i + 1:2d}] '{char_feature_names[idx]}': {char_coef[idx]:+.4f}")

# ==============================================================================
# Experiment 2: Word N-grams (Alphabetic Only - No Numerics)
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 2: WORD N-GRAMS (ALPHABETIC ONLY)")
logger.info("=" * 80)

logger.info("Extracting word n-gram features (alphabetic tokens only)...")
t_word_start = time.time()

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=100,
    max_df=0.7,
    max_features=30000,
    sublinear_tf=True,
    lowercase=True,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # Only alphabetic, 2+ chars (excludes '0', '1', etc.)
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_test_word = word_vectorizer.transform(X_test)

word_time = time.time() - t_word_start
word_sparsity = 100 * (1 - X_train_word.nnz / np.prod(X_train_word.shape))

logger.info(f"Word features: {X_train_word.shape}")
logger.info(f"Sparsity: {word_sparsity:.2f}%")
logger.info(f"Extraction time: {word_time:.2f}s")

# Train simple ensemble
logger.info("\nTraining word n-gram models...")
t_train_start = time.time()

word_nb = MultinomialNB(alpha=0.01)
word_sgd = SGDClassifier(
    loss="modified_huber",
    penalty="l2",
    alpha=0.00005,
    class_weight="balanced",
    early_stopping=True,
    max_iter=2000,
    random_state=SEED,
    n_jobs=-1,
)
word_svc = LinearSVC(C=1.0, loss="squared_hinge", max_iter=2000, class_weight="balanced", random_state=SEED)
word_svc_calibrated = CalibratedClassifierCV(word_svc, cv=3, method="sigmoid")

word_ensemble = StackingClassifier(
    estimators=[("nb", word_nb), ("sgd", word_sgd), ("svc", word_svc_calibrated)],
    final_estimator=LogisticRegression(max_iter=200, random_state=SEED),
    cv=3,
    stack_method="predict_proba",
    n_jobs=-1,
)

word_ensemble.fit(X_train_word, y_train)
word_train_time = time.time() - t_train_start

logger.info(f"Training time: {word_train_time:.2f}s")

# Evaluate
word_probs = word_ensemble.predict_proba(X_test_word)[:, 1]
word_auc = roc_auc_score(y_test, word_probs)

# Find best F1 threshold
precision, recall, thresholds = precision_recall_curve(y_test, word_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
word_best_f1 = f1_scores[best_f1_idx]
word_best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

word_pred = (word_probs >= word_best_threshold).astype(np.int8)
word_mcc = matthews_corrcoef(y_test, word_pred)
word_accuracy = accuracy_score(y_test, word_pred)
word_precision = precision_score(y_test, word_pred)
word_recall = recall_score(y_test, word_pred)
word_f1 = f1_score(y_test, word_pred)

logger.info("\nWord N-gram Results:")
logger.info(f"  AUC:       {word_auc:.4f}")
logger.info(f"  Best F1:   {word_best_f1:.4f} (threshold: {word_best_threshold:.4f})")
logger.info(f"  MCC:       {word_mcc:.4f}")
logger.info(f"  Accuracy:  {word_accuracy:.4f}")
logger.info(f"  Precision: {word_precision:.4f}")
logger.info(f"  Recall:    {word_recall:.4f}")
logger.info(f"  F1:        {word_f1:.4f}")

# Top features (use SGD model coefficients - most interpretable)
logger.info("\nTop 30 AI-indicative word n-grams:")
word_feature_names = word_vectorizer.get_feature_names_out()

# Get coefficients from SGD base model (index 1 in estimators)
word_sgd_model = word_ensemble.estimators_[1]  # SGD is second estimator
word_coef = word_sgd_model.coef_.ravel()

logger.info(f"DEBUG: Total features: {len(word_feature_names)}, Coef shape: {word_coef.shape}")
logger.info(f"DEBUG: Coef range: [{word_coef.min():.4f}, {word_coef.max():.4f}]")

word_top_ai_idx = np.argsort(word_coef)[-30:][::-1]
logger.info(f"DEBUG: Top AI indices count: {len(word_top_ai_idx)}")

for i, idx in enumerate(word_top_ai_idx):
    logger.info(f"  [{i + 1:2d}] '{word_feature_names[idx]}': {word_coef[idx]:+.4f}")

logger.info("\nTop 30 Human-indicative word n-grams:")
word_top_human_idx = np.argsort(word_coef)[:30]
logger.info(f"DEBUG: Top Human indices count: {len(word_top_human_idx)}")

for i, idx in enumerate(word_top_human_idx):
    logger.info(f"  [{i + 1:2d}] '{word_feature_names[idx]}': {word_coef[idx]:+.4f}")

# ==============================================================================
# Comparison Summary
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("COMPARISON SUMMARY")
logger.info("=" * 80)

logger.info("\nFeature Statistics:")
logger.info(f"  Character n-grams: {X_train_char.shape[1]:,} features, {char_sparsity:.2f}% sparse")
logger.info(f"  Word n-grams:      {X_train_word.shape[1]:,} features, {word_sparsity:.2f}% sparse")

logger.info("\nExtraction Time:")
logger.info(f"  Character n-grams: {char_time:.2f}s")
logger.info(f"  Word n-grams:      {word_time:.2f}s ({word_time / char_time:.2f}x)")

logger.info("\nTraining Time:")
logger.info(f"  Character n-grams: {char_train_time:.2f}s")
logger.info(f"  Word n-grams:      {word_train_time:.2f}s ({word_train_time / char_train_time:.2f}x)")

logger.info("\nPerformance (AUC):")
logger.info(f"  Character n-grams: {char_auc:.4f}")
logger.info(f"  Word n-grams:      {word_auc:.4f} ({'+' if word_auc > char_auc else ''}{word_auc - char_auc:+.4f})")

logger.info("\nPerformance (Best F1):")
logger.info(f"  Character n-grams: {char_best_f1:.4f}")
logger.info(
    f"  Word n-grams:      {word_best_f1:.4f} ({'+' if word_best_f1 > char_best_f1 else ''}{word_best_f1 - char_best_f1:+.4f})"
)

logger.info("\nPerformance (MCC):")
logger.info(f"  Character n-grams: {char_mcc:.4f}")
logger.info(f"  Word n-grams:      {word_mcc:.4f} ({'+' if word_mcc > char_mcc else ''}{word_mcc - char_mcc:+.4f})")

logger.info("\nPerformance (F1 @ optimal threshold):")
logger.info(f"  Character n-grams: {char_f1:.4f}")
logger.info(f"  Word n-grams:      {word_f1:.4f} ({'+' if word_f1 > char_f1 else ''}{word_f1 - char_f1:+.4f})")

# Determine winner
logger.info("\n" + "=" * 80)
if word_auc > char_auc and word_best_f1 > char_best_f1:
    logger.info("WINNER: WORD N-GRAMS (better AUC and F1)")
    logger.info("Recommendation: Use word n-grams for linguistic expert")
elif char_auc > word_auc and char_best_f1 > word_best_f1:
    logger.info("WINNER: CHARACTER N-GRAMS (better AUC and F1)")
    logger.info("Recommendation: Use character n-grams for linguistic expert")
else:
    logger.info("MIXED RESULTS: One method wins on AUC, the other on F1")
    logger.info("Recommendation: Compare top features and interpretability")

logger.info("=" * 80)

# ==============================================================================
# Markdown Pattern Analysis
# ==============================================================================

logger.info("\n" + "=" * 80)
logger.info("MARKDOWN PATTERN ANALYSIS")
logger.info("=" * 80)

MARKDOWN_PATTERNS = {
    "heading": r"^#+\s+",
    "bold_asterisk": r"\*\*[^*]+\*\*",
    "list_item": r"^\s*[-*+]\s+",
    "inline_code": r"`[^`]+`",
}


def has_markdown_patterns(text: str) -> bool:
    """Check if text contains any markdown patterns."""
    return any(re.search(pattern, text, re.MULTILINE) for pattern in MARKDOWN_PATTERNS.values())


# Analyze prediction errors on markdown-containing texts
logger.info("\nAnalyzing predictions on markdown-containing texts...")

markdown_mask = np.array([has_markdown_patterns(text) for text in X_test])
non_markdown_mask = ~markdown_mask

logger.info(
    f"Test samples with markdown: {markdown_mask.sum()}/{len(X_test)} ({100 * markdown_mask.sum() / len(X_test):.2f}%)"
)

# Character n-grams on markdown texts
if markdown_mask.sum() > 0:
    char_markdown_auc = roc_auc_score(y_test[markdown_mask], char_probs[markdown_mask])
    char_non_markdown_auc = roc_auc_score(y_test[non_markdown_mask], char_probs[non_markdown_mask])

    word_markdown_auc = roc_auc_score(y_test[markdown_mask], word_probs[markdown_mask])
    word_non_markdown_auc = roc_auc_score(y_test[non_markdown_mask], word_probs[non_markdown_mask])

    logger.info("\nCharacter N-grams:")
    logger.info(f"  AUC on markdown texts:     {char_markdown_auc:.4f}")
    logger.info(f"  AUC on non-markdown texts: {char_non_markdown_auc:.4f}")
    logger.info(f"  Gap:                       {char_markdown_auc - char_non_markdown_auc:+.4f}")

    logger.info("\nWord N-grams:")
    logger.info(f"  AUC on markdown texts:     {word_markdown_auc:.4f}")
    logger.info(f"  AUC on non-markdown texts: {word_non_markdown_auc:.4f}")
    logger.info(f"  Gap:                       {word_markdown_auc - word_non_markdown_auc:+.4f}")

    if abs(word_markdown_auc - word_non_markdown_auc) < abs(char_markdown_auc - char_non_markdown_auc):
        logger.info("\n✓ Word n-grams show more consistent performance across markdown/non-markdown texts")
    else:
        logger.info("\n✓ Character n-grams show more consistent performance across markdown/non-markdown texts")

logger.info("=" * 80)
