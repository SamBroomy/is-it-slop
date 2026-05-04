#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import random
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
import seaborn as sns
from __init__ import RETRAINED_MODEL_VERSION, SEED, VECTORIZER_BIN_PATH, df_train
from is_it_slop_preprocessing import __version__
from loguru import logger

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


from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# In[ ]:


from new_dale_chall_readability import cloze_score, reading_level
from nltk.classify.textcat import TextCat

tc = TextCat()


def process_words(text: str) -> dict[str, int | float | str]:
        # Language detection with fallback
    # try:
    #     detected_lang = tc.guess_language(text)
    #     lang = lgn.langname(detected_lang, typ="short")
    #     print(f"Detected language: {detected_lang} -> {lang}")
    # except Exception:
    #     lang = 'english'

    # # Tokenization with fallback
    # try:
    #     sentences = sent_tokenize(text, language=lang)
    #     words = word_tokenize(text.lower(), language=lang)
    # except LookupError:
        # Fallback to English if language not supported
    sentences = sent_tokenize(text, language="english")
    words = word_tokenize(text.lower(), language="english")

    words = [w for w in words if w.isalpha()]  # Only alphabetic tokens

    # Handle empty text
    if not words:
        print("No words found in text.")
        return {
            "total_words": 0,
            "unique_words": 0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "num_sentences": len(sentences),
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "sentence_length_std": 0.0,
            "dale_chall_score": 0.0,
            "dale_chall_reading_level": "N/A",
            "num_paragraphs": 1,
        }

    # Compute features
    total_words = len(words)
    unique_words = len(set(words))
    lexical_diversity = unique_words / total_words
    avg_word_length = np.mean([len(w) for w in words])

    num_sentences = len(sentences)
    # try:
    #     sentence_lengths = [len(word_tokenize(s, language=lang)) for s in sentences]
    # except LookupError:
    sentence_lengths = [len(word_tokenize(s, language="english")) for s in sentences]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0
    sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0.0
    sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0.0

    # Dale-Chall readability (expensive, only compute if needed)
    try:
        dale_chall_score = cloze_score(text)
        dale_chall_reading_level = reading_level(text)
    except Exception:
        dale_chall_score = 0.0
        dale_chall_reading_level = "N/A"

    # Paragraphs
    num_paragraphs = max(1, len([p for p in text.split("\n\n") if p.strip()]))

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": float(avg_word_length),
        "num_sentences": num_sentences,
        "avg_sentence_length": float(avg_sentence_length),
        "sentence_length_variance": float(sentence_length_variance),
        "sentence_length_std": float(sentence_length_std),
        "dale_chall_score": float(dale_chall_score),
        "dale_chall_reading_level": dale_chall_reading_level,
        "num_paragraphs": num_paragraphs,
    }


process_words(
    "This is a sample text. It contains several sentences. The purpose is to test the text processing function."
)
df_exp = df_train.collect().sample(200_000).lazy().with_columns(
    pl.col("text")
    .map_elements(
        process_words,
        return_dtype=pl.Struct([
            pl.Field("total_words", pl.Int64),
            pl.Field("unique_words", pl.Int64),
            pl.Field("lexical_diversity", pl.Float64),
            pl.Field("avg_word_length", pl.Float64),
            pl.Field("num_sentences", pl.Int64),
            pl.Field("avg_sentence_length", pl.Float64),
            pl.Field("sentence_length_variance", pl.Float64),
            pl.Field("sentence_length_std", pl.Float64),
            pl.Field("dale_chall_score", pl.Float64),
            pl.Field("dale_chall_reading_level", pl.Utf8),
            pl.Field("num_paragraphs", pl.Int64),
        ]),
    )
    .alias("text_features")
).unnest("text_features").filter(pl.col("total_words") != 0).collect()


# In[ ]:


df_exp


# In[ ]:


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_feature_discrimination(
    features_df: pl.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    logger.info("Analyzing feature discrimination...")

    # Separate by class
    human_df = features_df.filter(pl.col("label") == 0)
    ai_df = features_df.filter(pl.col("label") == 1)

    # Features to analyze (exclude IDs and labels)
    feature_cols = [
        "lexical_diversity",
        "avg_word_length",
        "avg_sentence_length",
        "sentence_length_variance",
        "sentence_length_std",
        "dale_chall_score",
        "total_words",
        "unique_words",
        "num_sentences",
        "num_paragraphs",
    ]

    results = {}

    print("\n" + "=" * 80)
    print("FEATURE DISCRIMINATION ANALYSIS")
    print("=" * 80 + "\n")

    for feature in feature_cols:
        human_values = human_df[feature].to_numpy()
        ai_values = ai_df[feature].to_numpy()

        # Remove any NaN/inf values
        human_values = human_values[np.isfinite(human_values)]
        ai_values = ai_values[np.isfinite(ai_values)]

        if len(human_values) == 0 or len(ai_values) == 0:
            continue

        # Statistical tests
        t_stat, p_value = stats.ttest_ind(human_values, ai_values)
        effect_size = cohen_d(human_values, ai_values)

        # Store results
        results[feature] = {
            "human_mean": float(np.mean(human_values)),
            "human_std": float(np.std(human_values)),
            "ai_mean": float(np.mean(ai_values)),
            "ai_std": float(np.std(ai_values)),
            "p_value": float(p_value),
            "cohens_d": float(effect_size),
            "t_statistic": float(t_stat),
        }

        # Print results
        print(f"{feature.upper().replace('_', ' ')}")
        print(f"  Human: {results[feature]['human_mean']:.4f} ± {results[feature]['human_std']:.4f}")
        print(f"  AI:    {results[feature]['ai_mean']:.4f} ± {results[feature]['ai_std']:.4f}")
        print(f"  Difference: {results[feature]['human_mean'] - results[feature]['ai_mean']:.4f}")
        print(f"  p-value: {results[feature]['p_value']:.6f}", end="")

        # Significance stars
        if p_value < 0.001:
            print(" ***")
        elif p_value < 0.01:
            print(" **")
        elif p_value < 0.05:
            print(" *")
        else:
            print()

        print(f"  Cohen's d: {results[feature]['cohens_d']:.4f}", end="")

        # Effect size interpretation
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            interpretation = "(negligible)"
        elif abs_d < 0.5:
            interpretation = "(small)"
        elif abs_d < 0.8:
            interpretation = "(medium)"
        else:
            interpretation = "(large)"
        print(f" {interpretation}")

        # Discriminative power
        if p_value < 0.01 and abs_d > 0.3:
            print("  ✅ DISCRIMINATIVE FEATURE")
        else:
            print("  ❌ Not discriminative")

        print()

    print("=" * 80)
    print("\nLEGEND:")
    print("  *** p < 0.001 (highly significant)")
    print("  **  p < 0.01  (significant)")
    print("  *   p < 0.05  (marginally significant)")
    print("\nCohen's d interpretation:")
    print("  < 0.2: negligible effect")
    print("  0.2-0.5: small effect")
    print("  0.5-0.8: medium effect")
    print("  > 0.8: large effect")
    print("\n✅ = Discriminative (p < 0.01 AND |d| > 0.3)")
    print("=" * 80 + "\n")


# In[ ]:


analyze_feature_discrimination(df_exp)
