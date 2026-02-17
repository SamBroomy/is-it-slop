//! Model artifact loading and global state management.
//!
//! This module embeds trained model artifacts at compile time using `include_bytes!`
//! macros. Artifacts are loaded lazily on first access via [`LazyLock`].
//!
//! # Model Artifacts
//!
//! The following artifacts are embedded:
//! - **ONNX model** (`slop-classifier.onnx`): sklearn `LogisticRegression` exported to ONNX
//! - **TF-IDF vectorizer** (`tfidf_vectorizer.rkyv`): Vocabulary and IDF weights
//! - **Token chunker config** (`token_chunker_config.json`): Chunking parameters
//! - **Classification thresholds**: Document-level and chunk-level thresholds
//!
//! All artifacts are downloaded from GitHub releases during build if not present locally.
//!
//! # Global Statics
//!
//! - [`MODEL`]: ONNX Runtime session (thread-safe via `Mutex`)
//! - [`PRE_PROCESSOR`]: TF-IDF vectorizer (immutable after initialization)
//! - [`TOKEN_CHUNKER`]: Chunking configuration (immutable)
//! - [`CLASSIFICATION_THRESHOLD`]: Document-level threshold
//! - [`CHUNK_CLASSIFICATION_THRESHOLD`]: Per-chunk threshold
//!
//! # Example
//!
//! ```rust
//! use is_it_slop::model::{MODEL, PRE_PROCESSOR, TOKEN_CHUNKER};
//!
//! // Artifacts are loaded on first access (lazy initialization)
//! let vectorizer = &*PRE_PROCESSOR; // Dereference LazyLock
//! let chunker = &*TOKEN_CHUNKER;
//! ```

use std::sync::{LazyLock, Mutex};

use is_it_slop_preprocessing::pre_processor::{TfidfVectorizer, TokenChunker};
use ort::session::{Session, builder::GraphOptimizationLevel};

include!(concat!(env!("OUT_DIR"), "/threshold.rs"));

/// Current model version
///
/// This is set during build time based on the model artifacts used.
/// The model version is used to ensure that the underlying model and tokenizer are compatible.
pub const MODEL_VERSION: &str = env!("MODEL_VERSION");

/// Raw ONNX model bytes embedded at compile time.
///
/// This static contains the binary content of the ONNX model file, included
/// using `include_bytes!` during compilation. The bytes are used to initialize
/// the [`MODEL`] session.
pub static MODEL_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("CLASSIFIER_MODEL_FILENAME")
));

/// Global ONNX Runtime session for inference.
///
/// Lazily initialized on first access. The session is wrapped in a `Mutex` to
/// enable thread-safe access, as ONNX Runtime sessions are not `Sync`.
///
/// The model is configured with:
/// - Graph optimization level 3 (maximum optimizations)
/// - 4 intra-op threads for parallel computation
pub static MODEL: LazyLock<Mutex<Session>> = LazyLock::new(|| {
    let session = Session::builder()
        .expect("Unable to create ONNX Runtime session builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Unable to set optimization level")
        .with_intra_threads(4)
        .expect("Unable to set intra threads")
        .commit_from_memory(MODEL_BYTES)
        .expect("Unable to load model from static bytes");

    Mutex::new(session)
});

/// Raw TF-IDF vectorizer bytes embedded at compile time.
///
/// Contains the serialized vectorizer in rkyv format, including vocabulary
/// and IDF weights. Used to initialize [`PRE_PROCESSOR`].
pub static TOKENIZER_BYTES: &[u8] = include_bytes!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("TOKENIZER_FILENAME"),
));

/// Global TF-IDF vectorizer for text preprocessing.
///
/// Lazily initialized on first access. Loads the vocabulary and IDF weights
/// from the embedded [`TOKENIZER_BYTES`]. The vectorizer is immutable after
/// initialization and safe to share across threads.
pub static PRE_PROCESSOR: LazyLock<TfidfVectorizer> = LazyLock::new(|| {
    TfidfVectorizer::from_bytes(TOKENIZER_BYTES)
        .expect("Unable to load tokenizer from static bytes")
});

/// Raw token chunker configuration embedded at compile time.
///
/// JSON string containing the chunking parameters (chunk size, overlap,
/// minimum chunk size). Used to initialize [`TOKEN_CHUNKER`].
pub static TOKEN_CHUNKER_SETTINGS: &str = include_str!(concat!(
    env!("MODEL_ARTIFACTS_DIR"),
    "/",
    env!("TOKEN_CHUNKER_CONFIG_FILENAME")
));

/// Global token chunker for splitting long documents.
///
/// Lazily initialized on first access. Loads the chunking configuration
/// from the embedded [`TOKEN_CHUNKER_SETTINGS`] JSON. The chunker is
/// immutable after initialization and safe to share across threads.
pub static TOKEN_CHUNKER: LazyLock<TokenChunker> = LazyLock::new(|| {
    TokenChunker::from_json_str(TOKEN_CHUNKER_SETTINGS).expect("Valid chunker config")
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_format() {
        // Verify MODEL_VERSION is in semver-like format (X.Y.Z)
        assert!(!MODEL_VERSION.is_empty());

        let parts: Vec<&str> = MODEL_VERSION.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Model version should have at least major.minor"
        );

        // Each part should be parseable as a number
        for part in parts {
            assert!(
                part.parse::<u32>().is_ok(),
                "Version part '{part}' should be a number"
            );
        }
    }

    #[test]
    fn test_model_lazy_initialization() {
        // Accessing MODEL should initialize it (only once)
        // Model bytes are embedded at compile time, so this is fast
        {
            let _guard1 = MODEL.lock().unwrap();
            // Drop guard before acquiring again
        }
        {
            let _guard2 = MODEL.lock().unwrap();
            // Should not panic - LazyLock ensures single initialization
        }
    }

    #[test]
    fn test_pre_processor_lazy_initialization() {
        // Accessing PRE_PROCESSOR should initialize it
        let num_features = PRE_PROCESSOR.num_features();
        assert!(
            num_features > 0,
            "Vectorizer should have non-zero vocabulary"
        );

        // Second access should return same instance
        let num_features2 = PRE_PROCESSOR.num_features();
        assert_eq!(num_features, num_features2);
    }

    #[test]
    fn test_token_chunker_lazy_initialization() {
        // Accessing TOKEN_CHUNKER should initialize it
        let chunk_size = TOKEN_CHUNKER.chunk_size;
        assert!(chunk_size > 0, "Chunker should have non-zero chunk size");

        // Verify chunker has reasonable defaults
        assert!(TOKEN_CHUNKER.overlap < TOKEN_CHUNKER.chunk_size);
        assert!(TOKEN_CHUNKER.min_chunk_size <= TOKEN_CHUNKER.chunk_size);
    }

    #[test]
    fn test_model_artifacts_non_empty() {
        // Verify embedded model bytes are non-empty
        assert!(!MODEL_BYTES.is_empty(), "Model bytes should be non-empty");
        assert!(
            MODEL_BYTES.len() > 1000,
            "Model should be at least 1KB (sanity check)"
        );

        // Verify tokenizer bytes are non-empty
        assert!(
            !TOKENIZER_BYTES.is_empty(),
            "Tokenizer bytes should be non-empty"
        );

        // Verify chunker config is non-empty
        assert!(
            !TOKEN_CHUNKER_SETTINGS.is_empty(),
            "Chunker config should be non-empty"
        );
    }

    #[test]
    fn test_classification_thresholds_valid() {
        // Verify thresholds are in valid probability range [0, 1]
        const {
            assert!(
                CLASSIFICATION_THRESHOLD >= 0.0 && CLASSIFICATION_THRESHOLD <= 1.0,
                "Classification threshold should be in [0, 1]"
            );
        }
        const {
            assert!(
                CHUNK_CLASSIFICATION_THRESHOLD >= 0.0 && CHUNK_CLASSIFICATION_THRESHOLD <= 1.0,
                "Chunk classification threshold should be in [0, 1]"
            );
        }
        const {
            // Thresholds should be reasonable (not at extremes)
            assert!(
                CLASSIFICATION_THRESHOLD > 0.1 && CLASSIFICATION_THRESHOLD < 0.9,
                "Classification threshold should be reasonable"
            );
        }
        const {
            assert!(
                CHUNK_CLASSIFICATION_THRESHOLD > 0.1 && CHUNK_CLASSIFICATION_THRESHOLD < 0.9,
                "Chunk threshold should be reasonable"
            );
        }
    }

    #[test]
    fn test_pre_processor_can_transform() {
        // Verify preprocessor can actually transform text
        let tokens = vec![vec![1, 2, 3, 4, 5]];
        let x = PRE_PROCESSOR.vectorize_from_tokens(&tokens);

        assert_eq!(x.rows(), 1);
        assert_eq!(x.cols(), PRE_PROCESSOR.num_features());
    }

    #[test]
    fn test_token_chunker_can_chunk() {
        // Verify chunker can actually chunk tokens
        let tokens: Vec<u32> = (0..200).collect();
        let chunks = TOKEN_CHUNKER.chunk(&tokens);

        assert!(!chunks.is_empty());
        assert!(
            chunks.len() > 1,
            "200 tokens should produce multiple chunks"
        );

        // Verify chunks are within expected size
        for chunk in &chunks {
            assert!(
                chunk.len() >= TOKEN_CHUNKER.min_chunk_size || chunks.len() == 1,
                "Chunk size should respect min_chunk_size"
            );
        }
    }

    #[test]
    fn test_chunker_config_json_valid() {
        // Verify TOKEN_CHUNKER_SETTINGS is valid JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(TOKEN_CHUNKER_SETTINGS);
        assert!(parsed.is_ok(), "Chunker config should be valid JSON");

        let config = parsed.unwrap();
        assert!(config.get("chunk_size").is_some());
        assert!(config.get("overlap").is_some());
        assert!(config.get("min_chunk_size").is_some());
    }

    #[test]
    fn test_model_onnx_inputs_outputs() {
        // Verify model has expected inputs/outputs
        // Model is already embedded in binary, so access is fast
        let model = MODEL.lock().unwrap();

        let inputs = model.inputs();
        assert!(!inputs.is_empty(), "Model should have at least one input");

        let outputs = model.outputs();
        assert!(
            outputs.len() >= 2,
            "Model should have at least 2 outputs (labels + probabilities)"
        );
    }

    #[test]
    #[allow(clippy::borrow_as_ptr)]
    #[allow(clippy::ref_as_ptr)]
    fn test_lazy_locks_are_static() {
        // Verify LazyLocks maintain same instance across accesses
        // Get pointer to PRE_PROCESSOR

        let ptr1: *const TfidfVectorizer = &*PRE_PROCESSOR;

        // Access again
        let ptr2: *const TfidfVectorizer = &*PRE_PROCESSOR;

        // Should be same address (singleton)
        assert_eq!(ptr1, ptr2, "LazyLock should maintain single instance");
    }

    #[test]
    fn test_model_bytes_not_empty() {
        // Verify model bytes are embedded correctly
        // Note: include_bytes! doesn't guarantee alignment, which is fine -
        // ONNX Runtime handles unaligned input, and rkyv uses AlignedVec for vectorizer
        assert!(!MODEL_BYTES.is_empty(), "Model bytes should not be empty");
        assert!(
            !TOKENIZER_BYTES.is_empty(),
            "Tokenizer bytes should not be empty"
        );

        // Sanity check: ONNX models typically start with specific magic bytes
        // But we just check they're present, not their alignment
        assert!(MODEL_BYTES.len() > 100, "Model should be reasonably sized");
    }
}
