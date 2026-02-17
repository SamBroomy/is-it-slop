#!/usr/bin/env python
"""
Post-training visualization script.

Generates additional analysis plots using trained model artifacts:
1. Top predictive n-grams
2. Chunk agreement analysis
3. Aggregation method comparison
4. Chunking behavior analysis
5. Confidence vs correctness analysis

Run after training to regenerate visualizations without retraining.
"""

import json
import time

import numpy as np
from __init__ import (
    CHUNK_CLASSIFICATION_THRESHOLD_PATH,
    CHUNKER_CONFIG_PATH,
    CLASSIFICATION_THRESHOLD_PATH,
    MODEL_ONNX_PATH,
    VECTORIZER_BIN_PATH,
    df_test,
)
from is_it_slop_preprocessing import TfidfVectorizer, TokenChunker, tokenize
from loguru import logger
from plots import (
    aggregation_comparison,
    chunk_agreement_analysis,
    chunking_behavior_analysis,
    confidence_correctness_analysis,
    top_ngrams_visualization,
)
from sklearn.metrics import accuracy_score

# Import ONNX runtime
import onnxruntime as ort


def load_artifacts():
    """Load trained model artifacts."""
    logger.info("Loading model artifacts...")

    # Load vectorizer
    vectorizer = TfidfVectorizer.load(VECTORIZER_BIN_PATH)
    logger.info(f"Loaded vectorizer with {len(vectorizer.vocabulary)} features")

    # Load chunker config
    with open(CHUNKER_CONFIG_PATH) as f:
        chunker_config = json.load(f)
    chunker = TokenChunker(
        chunk_size=chunker_config["chunk_size"],
        overlap=chunker_config["overlap"],
        min_chunk_size=chunker_config["min_chunk_size"],
    )
    logger.info(f"Loaded chunker: {chunker_config}")

    # Load ONNX model
    onnx_session = ort.InferenceSession(str(MODEL_ONNX_PATH))
    logger.info("Loaded ONNX model")

    # Load thresholds
    with open(CLASSIFICATION_THRESHOLD_PATH) as f:
        doc_threshold = float(f.read().strip())
    with open(CHUNK_CLASSIFICATION_THRESHOLD_PATH) as f:
        chunk_threshold = float(f.read().strip())

    logger.info(f"Document threshold: {doc_threshold:.4f}")
    logger.info(f"Chunk threshold: {chunk_threshold:.4f}")

    return vectorizer, chunker, onnx_session, doc_threshold, chunk_threshold


def prepare_test_data(vectorizer: TfidfVectorizer, chunker: TokenChunker):
    """Prepare test data with chunking."""
    logger.info("Preparing test data...")

    # Collect test data
    df_test_collected = df_test.collect()
    test_texts = df_test_collected.select("text").to_series().to_list()
    y_test = df_test_collected.select("label").to_series().to_numpy()

    logger.info(f"Test set size: {len(test_texts)} documents")

    # Tokenize
    logger.info("Tokenizing...")
    t0 = time.time()
    test_tokens = tokenize(test_texts)
    logger.info(f"Tokenized {len(test_tokens)} texts in {time.time() - t0:.2f}s")

    # Chunk
    logger.info("Chunking...")
    t0 = time.time()
    test_chunked = chunker.chunk_batch(test_tokens)
    logger.info(f"Chunked in {time.time() - t0:.2f}s")

    # Flatten chunks and track document mapping
    all_chunks = []
    chunk_to_doc = []
    for doc_idx, doc_chunks in enumerate(test_chunked):
        for chunk in doc_chunks:
            all_chunks.append(chunk)
            chunk_to_doc.append(doc_idx)

    chunk_to_doc = np.array(chunk_to_doc)
    logger.info(f"Total chunks: {len(all_chunks)}")

    # Vectorize chunks
    logger.info("Vectorizing chunks...")
    t0 = time.time()
    X_test_chunks = vectorizer.vectorize_from_tokens_batch(all_chunks)
    logger.info(f"Vectorized {X_test_chunks.shape[0]} chunks in {time.time() - t0:.2f}s")

    return test_texts, test_tokens, test_chunked, y_test, X_test_chunks, chunk_to_doc


def run_inference(onnx_session, X_test_chunks):
    """Run ONNX inference on test chunks."""
    logger.info("Running ONNX inference...")
    t0 = time.time()

    # Prepare input
    input_name = onnx_session.get_inputs()[0].name
    X_dense = X_test_chunks.toarray().astype(np.float32)

    # Run inference
    outputs = onnx_session.run(None, {input_name: X_dense})

    # Extract probabilities (second column = AI probability)
    chunk_probs = outputs[1][:, 1]

    logger.info(f"Inference completed in {time.time() - t0:.2f}s")
    return chunk_probs


def aggregate_predictions(chunk_probs: np.ndarray, chunk_to_doc: np.ndarray, n_docs: int, chunk_threshold: float):
    """Aggregate chunk predictions using weighted mean."""
    doc_probs = np.zeros(n_docs)

    for doc_idx in range(n_docs):
        mask = chunk_to_doc == doc_idx
        if mask.any():
            doc_chunk_probs = chunk_probs[mask]
            # Weighted mean by distance from threshold
            weights = np.abs(doc_chunk_probs - chunk_threshold)
            doc_probs[doc_idx] = np.average(doc_chunk_probs, weights=weights)

    return doc_probs


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Generating extra visualizations from trained model")
    logger.info("=" * 80)

    # Load artifacts
    vectorizer, chunker, onnx_session, doc_threshold, chunk_threshold = load_artifacts()

    # Prepare test data
    test_texts, test_tokens, test_chunked, y_test, X_test_chunks, chunk_to_doc = prepare_test_data(vectorizer, chunker)

    # Run inference
    chunk_probs = run_inference(onnx_session, X_test_chunks)

    # Aggregate predictions
    logger.info("Aggregating predictions...")
    doc_probs = aggregate_predictions(chunk_probs, chunk_to_doc, len(y_test), chunk_threshold)
    doc_preds = (doc_probs >= doc_threshold).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, doc_preds)
    logger.info(f"Document-level accuracy: {accuracy:.4f}")

    # Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating visualizations...")
    logger.info("=" * 80 + "\n")

    # Extract model coefficients from ONNX
    # For LogisticRegression, coefficients are stored in the model
    from onnx import numpy_helper

    onnx_model = onnx_session._model_proto
    # Find the LinearClassifier node and extract coefficients
    # This is a simplified approach - actual extraction may vary
    coef = None
    for initializer in onnx_model.graph.initializer:
        if "coefficients" in initializer.name.lower():
            coef = numpy_helper.to_array(initializer)
            break

    # If coefficients not found, create dummy for visualization
    if coef is None:
        logger.warning("Could not extract coefficients from ONNX model")
        logger.warning("Skipping top n-grams visualization")
    else:
        # 1. Top n-grams visualization
        logger.info("1/5: Top predictive n-grams...")
        top_ngrams_visualization(vectorizer, coef, top_n=20)

    # 2. Chunk agreement analysis
    logger.info("2/5: Chunk agreement analysis...")
    chunk_agreement_analysis(chunk_probs, chunk_to_doc, y_test, doc_preds, chunk_threshold, len(y_test))

    # 3. Aggregation comparison
    logger.info("3/5: Aggregation method comparison...")
    aggregation_comparison(chunk_probs, chunk_to_doc, y_test, chunk_threshold, doc_threshold, len(y_test))

    # 4. Chunking behavior analysis
    logger.info("4/5: Chunking behavior analysis...")
    chunking_behavior_analysis(chunk_to_doc, test_chunked, len(y_test))

    # 5. Confidence vs correctness analysis
    logger.info("5/5: Confidence vs correctness analysis...")
    confidence_correctness_analysis(doc_probs, y_test, doc_preds, doc_threshold)

    logger.info("\n" + "=" * 80)
    logger.info("All visualizations generated successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
